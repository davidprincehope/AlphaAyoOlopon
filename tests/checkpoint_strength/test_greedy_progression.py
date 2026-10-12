"""Check compatibility with the original evaluator and parallel seed invariance."""
from concurrent.futures import Future
from dataclasses import asdict, replace
import json

import numpy as np
import pytest

from experiments.checkpoint_strength import greedy_progression as progression
from Algorithms.alpha_zero.config import Settings
from Algorithms.alpha_zero.evaluation import evaluate_model


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
                "observation_version": progression.offline.adapter.OBSERVATION_VERSION,
                "value_perspective": "player_to_move"}
    (run / "manifest.json").write_text(json.dumps(manifest))
    path = run / "model-650"
    path.write_text("frozen C650 parameters")
    item = {"learner_round": 650, "requested_rounds": [650], "checkpoint_source": str(path),
            "checkpoint_sha256": progression.offline.source_hash(path), "model": Model()}
    monkeypatch.setattr(progression.offline, "prepare_checkpoints", lambda *args: ([item], []))
    game = progression.offline.pyspiel.load_game(settings.game_string)
    full = progression.load_dataset(progression.GreedyCheckpointConfig(650, 4).openings, game)
    dataset = replace(full, openings=tuple(o for o in full.openings if len(o.actions) == 10)[:2])
    monkeypatch.setattr(progression, "load_dataset", lambda *args: dataset)
    return run, game, item, dataset, settings


def without_time(records):
    return [{key: value for key, value in row.items() if key != "elapsed_seconds"} for row in records]


def test_parallel_preserves_original_evaluator_protocol(source, tmp_path, monkeypatch):
    run, game, item, dataset, settings = source
    config = progression.GreedyCheckpointConfig(650, 4, games=4, seed=23, workers=1)
    calls = []
    stock_greedy = progression.make_agent

    def traced_greedy(game, seed, agent_config):
        calls.append((seed, agent_config))
        return stock_greedy(game, seed, agent_config)

    monkeypatch.setattr(progression, "make_agent", traced_greedy)
    serial = progression.run_checkpoint(run, config, tmp_path / "serial")

    class Pool:
        def __init__(self, **kwargs):
            assert kwargs["mp_context"].get_start_method() == "spawn"
            assert kwargs["initargs"][2] == item["checkpoint_sha256"]
            assert kwargs["max_workers"] == 2
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def submit(self, function, index, opening, dataset_hash):
            future = Future()
            future.set_result(progression.play_pair(game, item["model"], config, settings.uct_c,
                                                     index, opening, dataset_hash))
            return future

    monkeypatch.setattr(progression, "ProcessPoolExecutor", Pool)
    monkeypatch.setattr(progression, "as_completed", lambda futures: reversed(futures))
    parallel = progression.run_checkpoint(run, replace(config, workers=2), tmp_path / "parallel")
    original = evaluate_model(game, item["model"], settings, 650, opponent=progression.OPPONENT,
                              games=4, simulations=4, seed=23, openings=dataset)
    assert without_time(serial["matches"]) == without_time(parallel["matches"]) == without_time(original["matches"])
    for key in ("summary", "by_seat", "wins", "draws", "losses", "score_rate", "termination_reasons"):
        assert serial[key] == parallel[key] == original[key]
    assert calls == [(seed, progression.OPPONENT) for seed in (23, 23, 24, 24)] * 2
    assert [row["seed"] for row in serial["matches"]] == [23, 23, 24, 24]
    for row in serial["matches"]:
        state = progression.replay(game, row["opening_actions"] + row["continuation_actions"])
        assert state.is_terminal() and list(state.returns()) == row["returns_by_player"]
    manifest = json.loads((tmp_path / "parallel" / "results.manifest.json").read_text())
    assert manifest["status"] == "completed" and manifest["completed_games"] == 4
    assert manifest["evaluation_policy"]["opponent"] == progression.OPPONENT
    assert manifest["games_sha256"] == progression.offline.source_hash(tmp_path / "parallel" / "results.games.jsonl")
    with pytest.raises(FileExistsError):
        progression.run_checkpoint(run, config, tmp_path / "serial")
    with pytest.raises(ValueError, match="outside"):
        progression.run_checkpoint(run, config, run / "results")


def test_checkpoint_mutation_fails_and_preserves_partial_games(source, tmp_path):
    run, _, item, _, _ = source
    from pathlib import Path
    def mutate(manifest):
        if manifest["completed_games"] == 0:
            Path(item["checkpoint_source"]).write_text("changed parameters")
    with pytest.raises(RuntimeError, match="checkpoint changed"):
        progression.run_checkpoint(run, progression.GreedyCheckpointConfig(650, 2, games=2, workers=1),
                                    tmp_path / "failed", on_progress=mutate)
    manifest = json.loads((tmp_path / "failed" / "results.manifest.json").read_text())
    assert manifest["status"] == "failed" and manifest["completed_games"] == 2
    assert not (tmp_path / "failed" / "results.json").exists()


@pytest.mark.parametrize("changes", [{"checkpoint": 0}, {"games": 3}, {"simulations": 1},
                                    {"workers": 0}, {"workers": 29}, {"seed": -1}, {"seed": 2**32 - 1}])
def test_invalid_configuration(changes):
    with pytest.raises(ValueError):
        progression.GreedyCheckpointConfig(**{"checkpoint": 650, "simulations": 64, **changes})
