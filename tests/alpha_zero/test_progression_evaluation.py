"""Offline progression configuration, provenance, fallback, and partial results."""
from dataclasses import asdict, replace
import json
from pathlib import Path
import sys

import numpy as np
import pytest

from Algorithms.alpha_zero.config import Settings
from Algorithms.alpha_zero.inference import export_round
from experiments.alpha_zero import offline_evaluate as offline

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def frozen_run(tmp_path):
    import jax
    import jax.numpy as jnp
    from open_spiel.python.algorithms.alpha_zero.model_linen import AlphaZeroModel
    settings = Settings(max_moves=4, max_simulations=4, evaluation_games=4)
    manifest = {"settings": asdict(settings), "architecture_version": "ayo_mlp_185863_v1",
                "observation_version": offline.adapter.OBSERVATION_VERSION,
                "value_perspective": "player_to_move", "game": settings.game_string}
    network = AlphaZeroModel("ayo_mlp", (15, 1, 1), 6, 256, 3)
    params = network.init(jax.random.PRNGKey(7), jnp.zeros((15, 1, 1)), False)["params"]
    run = tmp_path / "source"
    run.mkdir()
    (run / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    for step in (0, 1, 2, 3, 50):
        export_round(run, step, params, manifest)
    return run, manifest


def records(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_phase1_config_and_validation():
    config = offline.load_config(ROOT / "experiments/alpha_zero/configs/progression_phase1.json")
    assert config.checkpoints == [1, *range(50, 601, 50)]
    assert len(config.checkpoints) * config.games == 2600
    assert config.games == 200 and config.simulations == 64
    assert config.opponent == {"name": "GREEDY_HSTAR", "params": {}}
    assert config.fallback_first
    for change in ({"games": 3}, {"games": 0}, {"simulations": 1}, {"seed": -1},
                   {"checkpoints": [0]}, {"checkpoints": [-1]}, {"heartbeat_seconds": 0},
                   {"opponent": {"name": "RAND", "params": []}}):
        with pytest.raises(ValueError):
            replace(config, **change)


def test_earliest_valid_trained_fallback_and_deduplication(frozen_run):
    run, manifest = frozen_run
    (run / "inference-checkpoints/step-000001.npz").unlink()
    (run / "inference-checkpoints/step-000002.npz").write_bytes(b"corrupt")
    assert offline.available_steps(run) == [2, 3, 50]  # Never C0.
    config = offline.EvaluationConfig(checkpoints=[1, 3, 50], games=2, fallback_first=True)
    game = offline.pyspiel.load_game(manifest["game"])
    events = []
    selected, rejected = offline.prepare_checkpoints(run, config, game, manifest,
                                                      lambda event, **data: events.append((event, data)))
    assert [item["learner_round"] for item in selected] == [3, 50]
    assert selected[0]["requested_rounds"] == [1, 3]
    assert [item["learner_round"] for item in rejected] == [1, 2]
    assert events[0][0] == "checkpoint_substitution"
    with pytest.raises(FileNotFoundError):
        offline.prepare_checkpoints(run, replace(config, checkpoints=[100]), game, manifest, print)
    with pytest.raises(FileNotFoundError):
        offline.prepare_checkpoints(run, replace(config, fallback_first=False), game, manifest, print)


def test_progression_is_frozen_balanced_reproducible_and_streamed(frozen_run, tmp_path, monkeypatch):
    import optax
    from open_spiel.python.algorithms.alpha_zero import model_linen, replay_buffer
    run, _ = frozen_run
    before = {str(p.relative_to(run)): offline.source_hash(p) for p in run.rglob("*") if p.is_file()}
    def forbidden(*args, **kwargs):
        raise AssertionError("Offline evaluation cannot initialize training state")
    monkeypatch.setattr(optax, "adam", forbidden)
    monkeypatch.setattr(model_linen.Model, "build_model", forbidden)
    monkeypatch.setattr(replay_buffer.Buffer, "__init__", forbidden)
    config = offline.EvaluationConfig(checkpoints=[1, 50], games=4, simulations=4,
                                      opponent={"name": "GREEDY_HSTAR"}, seed=21)
    completed = []
    output = tmp_path / "curve.jsonl"
    result = offline.run_evaluation(run, config, output, on_checkpoint=completed.append)
    assert result["status"] == "completed" and result["completed_games"] == 8
    assert result["completed_checkpoints"] == len(completed) == 2
    reports = records(output)
    assert [r["learner_round"] for r in reports] == [1, 50]
    for report in reports:
        assert report["games"] == 4
        assert report["score_rate"] == (report["wins"] + .5 * report["draws"]) / 4
        for stats in report["by_seat"].values():
            assert stats["games"] == stats["wins"] + stats["draws"] + stats["losses"] == 2
            assert stats["win_rate"] == stats["wins"] / 2
            assert stats["score_rate"] == (stats["wins"] + .5 * stats["draws"]) / 2
            assert stats["average_game_length"] == 4
    matches = records(output.with_suffix(".games.jsonl"))
    assert [r["seed"] for r in matches] == [21, 21, 22, 22] * 2
    assert [r["player_0_policy"] for r in matches] == ["ALPHAZERO", "OPPONENT"] * 4
    events = records(output.with_suffix(".events.jsonl"))
    assert sum(r["event"] == "game_completed" for r in events) == 8
    assert events[-1]["event"] == "experiment_completed"
    saved = json.loads(output.with_suffix(".manifest.json").read_text())
    assert saved["evaluation_policy"]["dirichlet_noise"] is None
    assert saved["evaluation_policy"]["temperature_sampling"] is False
    assert saved["code_sha256"]["Algorithms/H_star_ayo.py"]
    assert before == {str(p.relative_to(run)): offline.source_hash(p) for p in run.rglob("*") if p.is_file()}
    again = tmp_path / "again.jsonl"
    offline.run_evaluation(run, config, again)
    without_timing = lambda rows: [{k: v for k, v in r.items() if k != "elapsed_seconds"} for r in rows]
    assert without_timing(matches) == without_timing(records(again.with_suffix(".games.jsonl")))
    with pytest.raises(FileExistsError):
        offline.run_evaluation(run, config, output)
    with pytest.raises(ValueError, match="outside"):
        offline.run_evaluation(run, config, run / "results.jsonl")


def test_failed_game_preserves_partial_records(frozen_run, tmp_path, monkeypatch):
    run, _ = frozen_run
    original = offline.evaluate_model
    def fail_during_matches(*args, on_game, **kwargs):
        def callback(record):
            on_game(record)
            raise RuntimeError("injected match failure")
        return original(*args, on_game=callback, **kwargs)
    monkeypatch.setattr(offline, "evaluate_model", fail_during_matches)
    output = tmp_path / "failed.jsonl"
    with pytest.raises(RuntimeError, match="injected"):
        offline.run_evaluation(run, offline.EvaluationConfig(checkpoints=[1], games=4), output)
    assert records(output) == []
    assert len(records(output.with_suffix(".games.jsonl"))) == 1
    saved = json.loads(output.with_suffix(".manifest.json").read_text())
    assert saved["status"] == "failed" and saved["completed_games"] == 1
    assert records(output.with_suffix(".events.jsonl"))[-1]["event"] == "experiment_failed"


def test_heartbeat_reports_a_running_match(frozen_run, tmp_path, monkeypatch):
    import time
    run, _ = frozen_run
    original = offline.evaluate_model
    def delayed_match(*args, **kwargs):
        time.sleep(1.2)
        return original(*args, **kwargs)
    monkeypatch.setattr(offline, "evaluate_model", delayed_match)
    output = tmp_path / "heartbeat.jsonl"
    offline.run_evaluation(run, offline.EvaluationConfig(checkpoints=[1], games=2,
                          heartbeat_seconds=1), output)
    events = records(output.with_suffix(".events.jsonl"))
    assert any(r["event"] == "heartbeat" and r["phase"] == "matches" and
               r["learner_round"] == 1 and r["completed_games"] == 0 for r in events)
    assert events[-1]["event"] == "experiment_completed"


def test_cli_overrides_phase1_without_training(monkeypatch, tmp_path):
    received = []
    monkeypatch.setattr(offline, "run_evaluation", lambda *args: received.append(args))
    monkeypatch.setattr(sys, "argv", ["offline_evaluate", "--run", str(tmp_path), "--config",
        str(ROOT / "experiments/alpha_zero/configs/progression_phase1.json"),
        "--checkpoints", "1,25:75:25", "--games", "4", "--simulations", "8",
        "--opponent", "RAND", "--seed", "0", "--no-fallback-first"])
    offline.main()
    config = received[0][1]
    assert config.checkpoints == [1, 25, 50, 75]
    assert config.games == 4 and config.simulations == 8 and config.seed == 0
    assert config.opponent == {"name": "RAND", "params": {}}
    assert config.fallback_first is False
