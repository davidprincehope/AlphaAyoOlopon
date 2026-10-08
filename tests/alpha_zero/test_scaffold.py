"""Contracts that can silently corrupt AlphaZero training if violated."""

import json
import pickle
import subprocess
import sys

import numpy as np
import pytest
import pyspiel

from Algorithms.alpha_zero import game as adapter
from Algorithms.alpha_zero.config import Settings, load_settings
from Algorithms.alpha_zero.runtime import ROOT


def test_training_observation_is_player_relative_and_preserves_horizon():
    game = pyspiel.load_game(adapter.GAME_NAME)
    state = game.new_initial_state()
    assert game.num_distinct_actions() == 6
    assert game.observation_tensor_shape() == [15]
    assert np.asarray(state.observation_tensor()).shape == (15,)
    original = np.asarray(state.observation_tensor()).copy()
    assert original[14] == 1
    assert state.move_number() == 0
    state._current_player = 1
    alternate = np.asarray(state.observation_tensor()).copy()
    assert alternate[14] == original[14]
    # While play is active, the side to move determines the perspective.
    np.testing.assert_array_equal(state.observation_tensor(0), state.observation_tensor(1))
    state.apply_action(state.legal_actions()[0])
    assert state.observation_tensor()[-1] == pytest.approx(0.999)


def test_asymmetric_canonical_encoding_rotates_without_reversing_or_mutating():
    board = [4, 1, 9, 2, 6, 7, 2, 8, 1, 0, 5, 3]
    captured = [10, 6]
    expected = (
        [4, 1, 9, 2, 6, 7, 2, 8, 1, 0, 5, 3, 10, 6],
        [2, 8, 1, 0, 5, 3, 4, 1, 9, 2, 6, 7, 6, 10],
    )
    for player in (0, 1):
        encoded = adapter.canonical_observation(board, captured, player, 250, 1000)
        np.testing.assert_allclose(
            encoded,
            np.asarray([value / 48 for value in expected[player]] + [0.75],
                       dtype=np.float32),
        )
        assert encoded.shape == (15,)
        for action in range(6):
            assert encoded[action] == pytest.approx(board[6 * player + action] / 48)
        assert encoded[12] == pytest.approx(captured[player] / 48)
        assert encoded[13] == pytest.approx(captured[1 - player] / 48)
    assert board == [4, 1, 9, 2, 6, 7, 2, 8, 1, 0, 5, 3]
    assert captured == [10, 6]

    game = pyspiel.load_game(adapter.GAME_NAME)
    state = game.new_initial_state()
    state.board = board.copy()
    state.captured = captured.copy()
    for player in (0, 1):
        state._current_player = player
        before = (state.board.copy(), state.captured.copy(), state._current_player,
                  state.move_number(), state._positions_since_capture.copy())
        observation = np.asarray(state.observation_tensor(1 - player)).copy()
        np.testing.assert_allclose(observation[:14], np.asarray(expected[player]) / 48)
        assert observation[14] == 1
        assert before == (state.board, state.captured, state._current_player,
                          state.move_number(), state._positions_since_capture)
        for action in range(6):
            assert state._action_to_house(player, action) == 6 * player + action
            assert observation[action] == pytest.approx(
                state.board[state._action_to_house(player, action)] / 48)


def test_horizon_counts_remaining_seeds_and_returns_max_utility():
    game = pyspiel.load_game(adapter.GAME_NAME, {'max_moves': 1, 'cutoff': 'collect'})
    state = game.new_initial_state()
    state.apply_action(0)
    assert state.is_terminal() and state.truncated
    assert state.current_player() == pyspiel.PlayerId.TERMINAL
    assert state.legal_actions() == []
    assert sum(state.returns()) == 0
    assert state.observation_tensor(0)[-1] == 0
    assert sum(state.board) == 0
    assert sum(state.captured) == 48
    sign = np.sign(state.captured[0] - state.captured[1])
    assert sign in (-1, 1)
    assert state.returns() == [sign, -sign]


def test_original_rules_and_seed_conservation_match():
    rng = np.random.default_rng(7)
    for _ in range(12):
        original = pyspiel.load_game('ayo_olopon').new_initial_state()
        training = pyspiel.load_game(adapter.GAME_NAME).new_initial_state()
        for _ in range(1000):
            assert original.board == training.board
            assert original.captured == training.captured
            assert original.current_player() == training.current_player()
            assert original.legal_actions() == training.legal_actions()
            assert original.returns() == training.returns()
            assert sum(training.board) + sum(training.captured) == 48
            if original.is_terminal():
                break
            action = int(rng.choice(original.legal_actions()))
            original.apply_action(action)
            training.apply_action(action)
        assert training.is_terminal()
        assert not training.truncated


def test_clone_and_serialization_preserve_history_without_aliasing():
    game = pyspiel.load_game(adapter.GAME_NAME)
    state = game.new_initial_state()
    state.apply_action(0)
    saved_history = set(state._positions_since_capture)
    for copy in (state.clone(), pickle.loads(pickle.dumps(state)),
                 pyspiel.deserialize_game_and_state(pyspiel.serialize_game_and_state(game, state))[1]):
        assert copy.ply == state.ply
        assert copy._positions_since_capture == saved_history
        action = copy.legal_actions()[0]
        expected = state.child(action)
        copy.apply_action(action)
        assert copy.board == expected.board
        assert copy.returns() == expected.returns()
        assert state.ply == 1
        assert state._positions_since_capture == saved_history


@pytest.mark.parametrize('kwargs', [
    {'actors': 0}, {'max_simulations': 1}, {'eval_levels': 1},
    {'temperature': 0}, {'learning_rate': float('nan')},
    {'replay_buffer_size': 8, 'train_batch_size': 8, 'replay_buffer_reuse': 4},
    {'policy_epsilon': 1.1}, {'nn_model': 'resnet'}, {'max_moves': True},
    {'cutoff': 'unknown'}, {'replay_buffer_reuse': False},
    {'nn_width': 128}, {'nn_depth': 2}, {'nn_api_version': 'nnx'},
    {'decouple_weight_decay': True},
])
def test_invalid_settings_fail_before_starting_workers(kwargs):
    with pytest.raises(ValueError):
        Settings(**kwargs)


def test_json_rejects_typo_and_loads_presets(tmp_path):
    config = tmp_path / 'config.json'
    config.write_text(json.dumps({'actor': 2}))
    with pytest.raises(ValueError, match='Unknown settings'):
        load_settings(config)
    for name in ('smoke', 'starter', 'sanity'):
        settings = load_settings(ROOT / f'experiments/alpha_zero/configs/{name}.json')
        assert settings.nn_model == 'ayo_mlp'
        assert (settings.nn_width, settings.nn_depth) == (256, 3)


def test_dry_run_selects_repository_sources_and_creates_no_output(tmp_path):
    output = tmp_path / 'unused'
    result = subprocess.run(
        [sys.executable, '-m', 'experiments.alpha_zero.train', '--dry-run',
         '--output', str(output)], cwd=ROOT, text=True, capture_output=True, timeout=60,
    )
    assert result.returncode == 0, result.stderr
    manifest = json.loads(result.stdout)
    assert manifest['observation_shape'] == [15]
    assert manifest['observation_version'] == adapter.OBSERVATION_VERSION
    assert manifest['architecture_version'] == 'ayo_mlp_185863_v1'
    assert manifest['value_perspective'] == 'player_to_move'
    assert not output.exists()



def test_game_pickling_restores_configuration_for_spawn():
    game = pyspiel.load_game(adapter.GAME_NAME, {'max_moves': 37, 'cutoff': 'collect'})
    restored = pickle.loads(pickle.dumps(game))
    assert restored.max_game_length() == 37
    state = restored.new_initial_state()
    assert state.max_moves == 37
    assert state.cutoff == 'collect'
    assert state.board == [4] * 12


def test_failed_worker_is_reported_instead_of_waiting_forever():
    code = '''
from Algorithms.alpha_zero.runtime import use_repository_open_spiel, guard_worker_failures
use_repository_open_spiel()
from open_spiel.python.utils import spawn
guard_worker_failures()
if __name__ == '__main__':
    # int does not accept OpenSpiel's queue keyword: intentional worker failure.
    worker = spawn.Process(int)
    worker.join(15)
    try:
        worker.queue.get_nowait()
    except RuntimeError as exc:
        assert 'worker exited' in str(exc)
    else:
        raise AssertionError('Dead worker was not detected')
'''
    result = subprocess.run([sys.executable, '-c', code], cwd=ROOT, text=True,
                            capture_output=True, timeout=30)
    assert result.returncode == 0, result.stderr
