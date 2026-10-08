"""Verify MCTS routing, parallel seed invariance, auditable replay, and failure artifacts."""
from concurrent.futures import Future
from dataclasses import asdict, replace
import json

import numpy as np
import pytest

from Algorithms.alpha_zero.config import Settings
from experiments.checkpoint_strength import neural_vs_mcts as comparison
from experiments.checkpoint_strength.opening_dataset import replay


class Model:
    def inference(self, observation, mask):
        policy = np.arange(1, 7, dtype=float) * mask
        return 0.0, policy / policy.sum()


@pytest.fixture
def source(tmp_path, monkeypatch):
    run = tmp_path / "training"
    run.mkdir()
    settings = Settings()
    manifest = {"settings": asdict(settings), "game": settings.game_string,
                "architecture_version": "ayo_mlp_185863_v1",
                "observation_version": comparison.offline.adapter.OBSERVATION_VERSION,
                "value_perspective": "player_to_move"}
    (run / "manifest.json").write_text(json.dumps(manifest))
    path = run / "model-600"
    path.write_text("frozen C600 parameters")
    item = {"learner_round": 600, "requested_rounds": [600], "checkpoint_source": str(path),
            "checkpoint_sha256": comparison.offline.source_hash(path), "model": Model()}
    monkeypatch.setattr(comparison.offline, "prepare_checkpoints", lambda *args: ([item], []))
    game = comparison.offline.pyspiel.load_game(settings.game_string)
    full = comparison.load_dataset(comparison.NeuralVsMCTSConfig().openings, game)
    subset = replace(full, openings=tuple(o for o in full.openings if len(o.actions) == 10)[:2])
    monkeypatch.setattr(comparison, "load_dataset", lambda *args: subset)
    return run, game, item, subset


@pytest.mark.parametrize("mcts_simulations", [None, 8])
def test_parallel_pairs_preserve_agents_seeds_and_legal_outcomes(source, tmp_path, monkeypatch, mcts_simulations):
    run, game, item, dataset = source
    calls = []
    original = comparison.make_bot
    neural_calls = []
    original_neural = comparison.make_neural_agent

    def traced(game, simulations, **kwargs):
        calls.append((simulations, kwargs))
        return original(game, simulations, **kwargs)

    monkeypatch.setattr(comparison, "make_bot", traced)

    def traced_neural(game, model, simulations, uct_c, seed):
        neural_calls.append((simulations, uct_c, seed))
        return original_neural(game, model, simulations, uct_c, seed)

    monkeypatch.setattr(comparison, "make_neural_agent", traced_neural)
    config = comparison.NeuralVsMCTSConfig(games=4, simulations=4, seed=23, workers=1,
                                          mcts_simulations=mcts_simulations)
    first = comparison.run_neural_vs_mcts(run, config, tmp_path / "serial")

    class Pool:
        def __init__(self, **kwargs):
            assert kwargs["mp_context"].get_start_method() == "spawn"
            assert kwargs["initargs"][2] == item["checkpoint_sha256"]
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def submit(self, function, index, opening, dataset_hash):
            future = Future()
            future.set_result(comparison.play_pair(game, item["model"], config, 1.5,
                                                    index, opening, dataset_hash))
            return future

    monkeypatch.setattr(comparison, "ProcessPoolExecutor", Pool)
    second = comparison.run_neural_vs_mcts(run, replace(config, workers=2), tmp_path / "parallel")
    assert first["policies"] == second["policies"]
    rows = []
    for name in ("serial", "parallel"):
        records = [json.loads(line) for line in (tmp_path / name / "results.games.jsonl").read_text().splitlines()]
        rows.append([{k: v for k, v in r.items() if k != "elapsed_seconds"} for r in records])
        manifest = json.loads((tmp_path / name / "results.manifest.json").read_text())
        assert manifest["status"] == "completed" and manifest["completed_games"] == 4
        assert not manifest["evaluation_policy"]["MCTS"]["neural_value"]
        assert manifest["evaluation_policy"]["C600"]["simulations"] == 4
        assert manifest["evaluation_policy"]["MCTS"]["simulations"] == (4 if mcts_simulations is None else 8)
        assert manifest["games_sha256"] == comparison.offline.source_hash(tmp_path / name / "results.games.jsonl")
    assert rows[0] == rows[1]
    assert [r["seed"] for r in rows[0]] == [23, 23, 24, 24]
    assert [r["opening_id"] for r in rows[0]] == [o.opening_id for o in dataset.openings for _ in (0, 1)]
    assert [entry[0] for entry in calls] == [4 if mcts_simulations is None else 8] * 8
    assert neural_calls == [(4, 1.5, seed) for seed in (23, 23, 24, 24)] * 2
    assert all(opts["uct_c"] == 1.5 and opts["rollouts_per_leaf"] == 1
               for sims, opts in calls)
    for record in rows[0]:
        state = replay(game, record["opening_actions"] + record["continuation_actions"])
        assert state.is_terminal() and list(state.returns()) == record["returns_by_player"]
        assert record["game_length"] <= 1000
    assert first["policies"]["C600"]["wins"] == first["policies"]["MCTS"]["losses"]
    assert first["policies"]["C600"]["score_rate"] + first["policies"]["MCTS"]["score_rate"] == 1
    assert all(first["by_seat"]["C600"][seat]["games"] == 2 for seat in ("P0", "P1"))
    with pytest.raises(FileExistsError):
        comparison.run_neural_vs_mcts(run, config, tmp_path / "serial")
    with pytest.raises(ValueError, match="outside"):
        comparison.run_neural_vs_mcts(run, config, run / "results")


def test_changed_checkpoint_fails_with_partial_artifacts(source, tmp_path):
    run, _, item, _ = source
    output = tmp_path / "comparison"
    def mutate(manifest):
        if manifest["completed_games"] == 0:
            from pathlib import Path
            Path(item["checkpoint_source"]).write_text("changed parameters")
    with pytest.raises(RuntimeError, match="checkpoint changed"):
        comparison.run_neural_vs_mcts(run, comparison.NeuralVsMCTSConfig(games=2, simulations=2, workers=1),
                                      output, on_progress=mutate)
    manifest = json.loads((output / "results.manifest.json").read_text())
    assert manifest["status"] == "failed" and manifest["completed_games"] == 2
    assert not (output / "results.json").exists()


@pytest.mark.parametrize("changes", [{"checkpoint": 0}, {"games": 3}, {"simulations": 1},
                                    {"simulations": None}, {"mcts_simulations": 1},
                                    {"mcts_simulations": True}, {"mcts_simulations": 128.0},
                                    {"workers": 0}, {"workers": 29}, {"seed": -1}, {"seed": 2**32 - 1}])
def test_invalid_configuration(changes):
    with pytest.raises(ValueError):
        comparison.NeuralVsMCTSConfig(**changes)


def test_saved_equal_and_unequal_budget_configurations():
    from pathlib import Path
    root = Path("experiments/checkpoint_strength/configs")
    for filename, expected in (("c600_vs_mcts_64.json", (64, 64)),
                               ("c600_vs_mcts_128.json", (128, 128)),
                               ("c600_64_vs_mcts_128.json", (64, 128))):
        config = comparison.NeuralVsMCTSConfig(**json.loads((root / filename).read_text()))
        assert (config.simulations, config.mcts_budget) == expected
