"""Stage 6: complete trajectories become aligned, randomized training data."""

from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np

from Algorithms.alpha_zero.runtime import use_repository_open_spiel

use_repository_open_spiel()
import pyspiel  # noqa: E402
from Algorithms.alpha_zero import game as ayo_game  # noqa: E402
from open_spiel.python.algorithms.alpha_zero import alpha_zero, replay_buffer, utils  # noqa: E402


def record(marker, player, game_id):
    # Distinct, finite, normalized canonical states with a visible record ID.
    x = np.array([marker / 48] + [4 / 48] * 11 +
                 [player / 48, (1 - player) / 48, 0.5], dtype=np.float32)
    pi = np.array([0.7, 0, 0.2, 0, 0.1, 0], dtype=np.float32)
    return utils.TrainInput(
        observation=jnp.asarray(x),
        legals_mask=jnp.asarray([1, 0, 1, 0, 1, 0], dtype=bool),
        policy=jnp.asarray(pi),
        value=jnp.asarray(1.0 if player == game_id % 2 else -1.0,
                          dtype=jnp.float32),
    )


def assert_record_invariants(sample):
    assert sample.observation.shape == (15,)
    assert sample.policy.shape == (6,)
    assert sample.value.shape == ()
    assert sample.observation.dtype == jnp.float32
    assert sample.policy.dtype == jnp.float32
    assert sample.value.dtype == jnp.float32
    assert sample.legals_mask.dtype == jnp.bool
    assert np.all(np.isfinite(sample.observation))
    assert np.all(np.isfinite(sample.policy))
    assert np.isfinite(sample.value)
    assert np.all(np.asarray(sample.policy) >= 0)
    assert np.all(np.asarray(sample.policy)[~np.asarray(sample.legals_mask)] == 0)
    np.testing.assert_allclose(sample.policy.sum(), 1, atol=1e-6)
    assert -1 <= sample.value <= 1
    assert 0 <= sample.observation[14] <= 1


def test_capacity_eviction_preserves_aligned_records():
    buffer = replay_buffer.Buffer(3, seed=9)
    for i in range(6):
        buffer.append(record(i, i % 2, i // 2))
        assert len(buffer) <= 3
    assert len(buffer) == 3 and buffer.total_seen == 6
    retained = {int(round(float(row[0]) * 48))
                for row in np.asarray(buffer.data.observation)}
    assert retained == {3, 4, 5}
    for i in range(3):
        sample = utils.TrainInput(
            observation=buffer.data.observation[i],
            legals_mask=buffer.data.legals_mask[i],
            policy=buffer.data.policy[i],
            value=buffer.data.value[i],
        )
        assert_record_invariants(sample)
        marker = int(round(float(sample.observation[0]) * 48))
        expected = record(marker, marker % 2, marker // 2)
        np.testing.assert_array_equal(sample.observation, expected.observation)
        np.testing.assert_array_equal(sample.policy, expected.policy)
        np.testing.assert_array_equal(sample.value, expected.value)


def test_cross_game_sampling_and_mixed_player_batch():
    buffer = replay_buffer.Buffer(12, seed=7)
    for game_id in range(5):
        for player in (0, 1):
            buffer.append(record(2 * game_id + player, player, game_id))
    mixed_games = False
    mixed_players = False
    saw_duplicate = False
    for _ in range(12):
        sampled = buffer.sample(4)
        assert sampled.observation.shape == (4, 15)
        assert sampled.policy.shape == (4, 6)
        assert sampled.value.shape == (4,)
        assert sampled.observation.dtype == sampled.policy.dtype == sampled.value.dtype == jnp.float32
        ids = [int(round(float(x) * 48)) for x in sampled.observation[:, 0]]
        mixed_games |= len({i // 2 for i in ids}) > 1
        mixed_players |= len({i % 2 for i in ids}) > 1
        saw_duplicate |= len(set(ids)) < len(ids)
        for i in range(4):
            example = utils.TrainInput(
                observation=sampled.observation[i],
                legals_mask=sampled.legals_mask[i],
                policy=sampled.policy[i], value=sampled.value[i],
            )
            assert_record_invariants(example)
            expected = record(ids[i], ids[i] % 2, ids[i] // 2)
            np.testing.assert_array_equal(example.value, expected.value)
    assert mixed_games and mixed_players
    assert saw_duplicate  # JAX randint samples with replacement.


def test_end_to_end_game_to_replay_to_minibatch():
    game = pyspiel.load_game(ayo_game.GAME_NAME, {"max_moves": 2})

    class Bot:
        def mcts_search(self, state):
            legal = state.legal_actions()
            children = [SimpleNamespace(action=action, explore_count=count)
                        for action, count in zip(legal, [7, 2, 1] + [0] * 3)]
            return SimpleNamespace(
                children=children, total_reward=0.0, explore_count=10,
                best_child=lambda: children[0],
            )

    logger = SimpleNamespace(print=lambda *args: None, opt_print=lambda *args: None)
    trajectory = alpha_zero._play_game(logger, 0, game, (Bot(), Bot()), 1.0, 0)
    assert len(trajectory.states) == 2
    assert [int(s.current_player) for s in trajectory.states] == [0, 1]
    buffer = replay_buffer.Buffer(4, seed=1)
    for state, example in zip(trajectory.states, alpha_zero.replay_examples(trajectory)):
        assert example.observation.shape == (15,)
        assert state.action == list(state.legals_mask).index(1)
        assert state.policy[state.action] < 1  # Visits, not chosen-action one-hot.
        np.testing.assert_array_equal(example.observation, state.observation)
        np.testing.assert_allclose(example.policy, state.policy, atol=1e-7)
        np.testing.assert_allclose(example.value, trajectory.returns[state.current_player])
        assert_record_invariants(example)
        buffer.append(example)
    sampled = buffer.sample(4)
    assert sampled.observation.shape == (4, 15)
    assert sampled.policy.shape == (4, 6)
    assert sampled.value.shape == (4,)
    original = [np.asarray(s.observation) for s in trajectory.states]
    for i in range(4):
        matches = [j for j, obs in enumerate(original)
                   if np.array_equal(np.asarray(sampled.observation[i]), obs)]
        assert len(matches) == 1
        j = matches[0]
        np.testing.assert_allclose(sampled.policy[i], trajectory.states[j].policy, atol=1e-7)
        np.testing.assert_allclose(sampled.value[i], trajectory.returns[trajectory.states[j].current_player])
