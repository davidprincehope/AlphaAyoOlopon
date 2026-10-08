"""A single worker finishes 64 before 128, preserving partial results on failure."""
from dataclasses import asdict, replace
import json
from types import SimpleNamespace

import pytest
from experiments.checkpoint_strength import modal_compare as remote
from experiments.alpha_zero import offline_evaluate as offline
from Algorithms.alpha_zero.config import Settings


@pytest.mark.parametrize("fail_second", [False, True])
def test_sequential_budgets_and_partial_persistence(tmp_path, monkeypatch, fail_second):
    import jax
    from experiments.checkpoint_strength import opening_dataset
    calls, commits = [], []
    monkeypatch.setattr(remote.base, "CHECKPOINT_ROOT", tmp_path / "source")
    monkeypatch.setattr(remote.base, "RESULTS_ROOT", tmp_path / "results")
    monkeypatch.setattr(remote.base, "results_volume", SimpleNamespace(commit=lambda: commits.append(True)))
    monkeypatch.setattr(jax, "devices", lambda: [SimpleNamespace(platform="gpu")])
    monkeypatch.setattr(opening_dataset, "load_dataset", lambda *args: SimpleNamespace(content_sha256="fixed-hash"))
    run = tmp_path / "source/run"
    run.mkdir(parents=True)
    (run / "manifest.json").write_text(json.dumps({"settings": asdict(Settings())}))
    def evaluate(run, config, output, *, on_checkpoint):
        calls.append(config.simulations)
        if config.simulations == 128 and fail_second:
            raise RuntimeError("second budget failed")
        report = {"learner_round": 1, "games": 400, "win_rate": config.simulations / 256,
                  "score_rate": 0.75, "by_seat": {}, "first_player_advantage": {},
                  "opening_dataset": {"content_sha256": "fixed-hash"}}
        output.parent.mkdir(parents=True)
        output.write_text(json.dumps(report) + "\n")
        on_checkpoint(report)
        return {"status": "completed"}
    monkeypatch.setattr(offline, "run_evaluation", evaluate)
    low = offline.EvaluationConfig(checkpoints=[1], games=400, simulations=64, openings="fixed.json")
    args = ("run", "new-evaluation", asdict(low), asdict(replace(low, simulations=128)))
    if fail_second:
        with pytest.raises(RuntimeError, match="second budget failed"):
            remote.compare_remote.local(*args)
    else:
        assert remote.compare_remote.local(*args)["status"] == "completed"
    root = tmp_path / "results/new-evaluation"
    status = json.loads((root / "comparison.manifest.json").read_text())
    assert calls == [64, 128]
    assert status["completed_budgets"] == ([64] if fail_second else [64, 128])
    assert status["completed_games"] == (400 if fail_second else 800)
    assert status["status"] == ("failed" if fail_second else "completed")
    assert (root / "64/results.jsonl").exists()
    assert len(commits) >= 4
    if not fail_second:
        comparison = json.loads((root / "budget_comparison.json").read_text())
        assert comparison[0]["win_rate_change_percentage_points"] == 25
    with pytest.raises(FileExistsError):
        remote.compare_remote.local(*args)


def test_comparison_rejects_different_openings_or_checkpoint_sets():
    with pytest.raises(ValueError, match="same unique checkpoints"):
        remote.comparison_rows([{"learner_round": 1}], [{"learner_round": 2}])
    with pytest.raises(ValueError, match="identical opening datasets"):
        remote.comparison_rows([{"learner_round": 1, "opening_dataset": {"hash": "a"}}],
                               [{"learner_round": 1, "opening_dataset": {"hash": "b"}}])


def test_recovery_reuses_complete_checkpoints_then_runs_all_at_128(tmp_path, monkeypatch):
    import jax
    from experiments.checkpoint_strength import opening_dataset
    monkeypatch.setattr(remote.base, "CHECKPOINT_ROOT", tmp_path / "source")
    monkeypatch.setattr(remote.base, "RESULTS_ROOT", tmp_path / "results")
    monkeypatch.setattr(remote.base, "results_volume", SimpleNamespace(commit=lambda: None))
    monkeypatch.setattr(jax, "devices", lambda: [SimpleNamespace(platform="gpu")])
    monkeypatch.setattr(opening_dataset, "load_dataset", lambda *args: SimpleNamespace(content_sha256="fixed-hash"))
    run = tmp_path / "source/run"
    (run / "inference-checkpoints").mkdir(parents=True)
    (run / "manifest.json").write_text(json.dumps({"settings": asdict(Settings())}))
    checkpoint = run / "inference-checkpoints/step-000001.npz"
    checkpoint.write_bytes(b"unchanged model")
    low = offline.EvaluationConfig(checkpoints=[1, 600], games=400, simulations=64, openings="fixed.json")
    previous = tmp_path / "results/previous"
    (previous / "64").mkdir(parents=True)
    (previous / "comparison.manifest.json").write_text(json.dumps({
        "configuration_64": asdict(low), "opening_content_sha256": "fixed-hash",
        "completed_budgets": [], "current_simulations": 64}))
    (previous / "64/results.manifest.json").write_text(json.dumps({
        "run_manifest_sha256": offline.source_hash(run / "manifest.json"), "code_sha256": {}}))
    def report(step):
        return {"learner_round": step, "games": 400, "matches": [{}] * 400, "status": "completed",
                "checkpoint_source": str(checkpoint), "checkpoint_sha256": offline.source_hash(checkpoint),
                "win_rate": 0.5, "score_rate": 0.75, "by_seat": {}, "first_player_advantage": {},
                "opening_dataset": {"content_sha256": "fixed-hash"}}
    original = json.dumps(report(1)) + "\n"
    (previous / "64/results.jsonl").write_text(original)
    calls = []
    def evaluate(run, config, output, *, on_checkpoint):
        calls.append((config.simulations, config.checkpoints))
        output.parent.mkdir(parents=True)
        rows = [report(step) for step in config.checkpoints]
        output.write_text("".join(json.dumps(r) + "\n" for r in rows))
        for row in rows:
            on_checkpoint(row)
    monkeypatch.setattr(offline, "run_evaluation", evaluate)
    result = remote.compare_remote.local("run", "recovery", asdict(low),
                                         asdict(replace(low, simulations=128)), "previous")
    assert calls == [(64, [600]), (128, [1, 600])]
    assert result["status"] == "completed" and result["completed_games"] == 1600
    assert result["reused_checkpoints"] == [1]
    assert (previous / "64/results.jsonl").read_text() == original
    merged = [json.loads(line) for line in (tmp_path / "results/recovery/64/results.jsonl").read_text().splitlines()]
    assert [row["learner_round"] for row in merged] == [1, 600]
    checkpoint.write_bytes(b"changed model")
    with pytest.raises(ValueError, match="Incomplete or incompatible"):
        remote.reusable_reports(previous, run, low, "fixed-hash")
