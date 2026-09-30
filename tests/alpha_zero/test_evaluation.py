"""Round evaluation reuses agents and cannot mutate the learning state."""
import json
import random
from types import SimpleNamespace

import numpy as np
import pytest

from Algorithms.alpha_zero.runtime import use_repository_open_spiel
use_repository_open_spiel()
import pyspiel
from Algorithms.alpha_zero import game as adapter
from Algorithms.alpha_zero.config import Settings
from Algorithms.alpha_zero.evaluation import evaluate_model, evaluate_round, make_neural_agent
from Algorithms.alpha_zero.inference import InferenceModel, export_round, load_parameters
from experiments.alpha_zero.offline_evaluate import parse_steps, available_steps


class UniformModel:
    def inference(self, observation, mask):
        policy = np.asarray(mask, dtype=float)
        return 0.0, policy / policy.sum()


@pytest.mark.parametrize("opponent", [
    {"name": "RAND"}, {"name": "GREEDY_HSTAR"},
    {"name": "GREEDY", "params": {"heuristic": "H_C"}},
    {"name": "MINIMAX", "params": {"maximum_depth": 1}},
    {"name": "MCTS", "params": {"simulations": 3, "rollouts_per_leaf": 1}},
])
def test_existing_agents_balanced_and_rng_isolated(opponent):
    settings = Settings(max_moves=4, max_simulations=4, evaluation_games=4)
    game = pyspiel.load_game(settings.game_string)
    python_rng, numpy_rng = random.getstate(), np.random.get_state()
    result = evaluate_model(game, UniformModel(), settings, 101, opponent=opponent)
    assert random.getstate() == python_rng
    for a, b in zip(np.random.get_state(), numpy_rng):
        np.testing.assert_array_equal(a, b)
    assert result["training_step"] == 101
    assert result["games"] == result["wins"] + result["draws"] + result["losses"] == 4
    assert result["win_rate"] == result["wins"] / 4
    assert result["score_rate"] == (result["wins"] + 0.5 * result["draws"]) / 4
    assert result["score_rate"] == (result["player0_score_rate"] + result["player1_score_rate"]) / 2
    assert result["summary"]["policies"]["ALPHAZERO"]["seats"] == {"0": 2, "1": 2}
    assert result["mcts_simulations"] == 4


def test_neural_selection_is_max_visit_then_smallest_action(monkeypatch):
    from open_spiel.python.algorithms import mcts
    observed = {}
    class FakeBot:
        def __init__(self, *args, **kwargs):
            observed.update(kwargs)
        def mcts_search(self, state):
            return SimpleNamespace(children=[SimpleNamespace(action=a, explore_count=n)
                                              for a, n in [(5, 9), (2, 9), (0, 1)]])
    monkeypatch.setattr(mcts, "MCTSBot", FakeBot)
    game = pyspiel.load_game(Settings(max_moves=4).game_string)
    bot = make_neural_agent(game, UniformModel(), 4, 1.5, 0)
    assert bot.step(game.new_initial_state()) == 2
    assert observed["dirichlet_noise"] is None
    assert observed["solve"] is False


def test_inference_exports_and_legacy_load_without_optimizer(tmp_path, monkeypatch):
    import jax
    import optax
    from dataclasses import asdict
    from open_spiel.python.algorithms.alpha_zero import model_linen
    settings = Settings(max_moves=4, max_simulations=4, evaluation_games=2)
    model = model_linen.Model.build_model("ayo_mlp", [15], 6, 256, 3, 0.0001, 0.001, str(tmp_path))
    before = model._state
    model.save_checkpoint(1)
    manifest = {"settings": asdict(settings), "architecture_version": "ayo_mlp_185863_v1",
                "observation_version": adapter.OBSERVATION_VERSION,
                "value_perspective": "player_to_move", "game": settings.game_string}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    export = export_round(tmp_path, 1, model._state.params, manifest)
    def forbidden(*args, **kwargs):
        raise AssertionError("Offline evaluation must not create an optimizer or training model")
    monkeypatch.setattr(optax, "adam", forbidden)
    monkeypatch.setattr(model_linen.Model, "build_model", forbidden)
    from open_spiel.python.algorithms.alpha_zero import replay_buffer
    monkeypatch.setattr(replay_buffer.Buffer, "__init__", forbidden)
    game = pyspiel.load_game(settings.game_string)
    state = game.new_initial_state()
    expected = model.inference(state.observation_tensor(), state.legal_actions_mask())
    for path in (export, tmp_path / "checkpoint-1"):
        loaded = InferenceModel(load_parameters(path))
        actual = loaded.inference(state.observation_tensor(), state.legal_actions_mask())
        for x, y in zip(expected, actual):
            np.testing.assert_allclose(x, y, atol=1e-6)
        evaluate_model(game, loaded, settings, 1)
    config = SimpleNamespace(path=str(tmp_path), session_id="test", **asdict(settings))
    report = evaluate_round(config, game, model, 2)
    assert report["status"] == "completed"
    assert model._state is before
    assert int(model._state.step) == 0
    assert available_steps(tmp_path) == [1, 2]


def test_evaluation_failure_does_not_fail_training_hook(tmp_path):
    config = SimpleNamespace(path=str(tmp_path), session_id="test")
    with pytest.warns(UserWarning, match="training continues"):
        result = evaluate_round(config, None, None, 101)
    assert result["status"] == "error"
    assert result["training_step"] == 101
    assert json.loads((tmp_path / "evaluation.jsonl").read_text())["status"] == "error"


def test_range_selection_and_invalid_game_counts():
    assert parse_steps(["10:30:10", "2,5", "10"]) == [2, 5, 10, 20, 30]
    for bad in ("-1", "4:1", "1:4:0"):
        with pytest.raises(ValueError):
            parse_steps([bad])
    with pytest.raises(ValueError, match="even"):
        Settings(evaluation_games=3)


def test_generalized_runner_preserves_original_benchmark():
    from experiments.agent_benchmark.random_vs_greedy_hstar import play_game
    from experiments.agent_benchmark.agents import make_agent
    game = pyspiel.load_game("ayo_olopon")
    factories = {name: (lambda game, seed, name=name: make_agent(game, seed, {"name": name}))
                 for name in ("RAND", "GREEDY_HSTAR")}
    old = play_game(0, ("RAND", "GREEDY_HSTAR"), 17, 4)
    generalized = play_game(0, ("RAND", "GREEDY_HSTAR"), 17, 4,
                            game=game, agent_factories=factories)
    old.pop("elapsed_seconds")
    generalized.pop("elapsed_seconds")
    assert old == generalized
