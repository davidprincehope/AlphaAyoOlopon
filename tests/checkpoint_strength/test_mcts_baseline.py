from dataclasses import replace
import json
from types import SimpleNamespace

import pytest
from Algorithms.alpha_zero.runtime import use_repository_open_spiel
use_repository_open_spiel()
import pyspiel
from Algorithms.alpha_zero import game as adapter
from experiments.checkpoint_strength.opening_dataset import load_dataset
from experiments.checkpoint_strength.mcts_baseline import run_baseline


def test_plain_mcts_is_reproducible_and_pairs_identical_openings(tmp_path):
    game = pyspiel.load_game("ayo_olopon_alpha_zero(max_moves=1000,cutoff=collect)")
    full = load_dataset("experiments/checkpoint_strength/dataset/ayo_mixed_v2_200.json", game)
    dataset = replace(full, openings=tuple(o for o in full.openings if len(o.actions) == 10)[:2])
    for name in ("first", "second"):
        result = run_baseline(game, dataset, tmp_path / name, budgets=(3, 4))
        assert result["status"] == "completed" and result["completed_games"] == 8
        assert not result["neural_policy"] and not result["neural_value"]
    for budget in (3, 4):
        rows = []
        for name in ("first", "second"):
            records = [json.loads(line) for line in (tmp_path / name / str(budget) / "games.jsonl").read_text().splitlines()]
            rows.append([{k: v for k, v in r.items() if k != "elapsed_seconds"} for r in records])
        assert rows[0] == rows[1]
        assert [r["opening_id"] for r in rows[0]] == [o.opening_id for o in dataset.openings for _ in (0, 1)]
        assert [r["seed"] for r in rows[0]] == [20261003, 20261003, 20261004, 20261004]
        assert rows[0][0]["player_0_policy"] == rows[0][1]["player_1_policy"] == "MCTS"
        report = json.loads((tmp_path / "first" / str(budget) / "results.json").read_text())
        assert report["games"] == 4 and report["first_player_advantage"]["P0"]["games"] == 4
    with pytest.raises(FileExistsError):
        run_baseline(game, dataset, tmp_path / "first", budgets=(3, 4))


@pytest.mark.parametrize("dependency_success", [False, True])
def test_baseline_waits_for_successful_neural_comparison(tmp_path, monkeypatch, dependency_success):
    from experiments.checkpoint_strength import modal_mcts_baseline as remote
    from experiments.checkpoint_strength import opening_dataset, mcts_baseline
    order = []
    monkeypatch.setattr(remote.base, "RESULTS_ROOT", tmp_path)
    monkeypatch.setattr(remote.base, "results_volume", SimpleNamespace(commit=lambda: None, reload=lambda: None))
    def wait(**kwargs):
        order.append("wait")
        if not dependency_success:
            raise RuntimeError("upstream failed")
        return {"status": "completed"}
    monkeypatch.setattr(remote.modal, "FunctionCall", SimpleNamespace(from_id=lambda call: SimpleNamespace(get=wait)))
    previous = tmp_path / "neural"
    (previous / "128").mkdir(parents=True)
    config = {"openings": "fixed", "seed": 20261003, "games": 400,
              "opponent": {"name": "GREEDY_HSTAR", "params": {}}}
    (previous / "comparison.manifest.json").write_text(json.dumps({"status": "completed", "completed_budgets": [64, 128],
        "configuration_64": config, "opening_content_sha256": "fixed-hash"}))
    (previous / "128/results.jsonl").write_text(json.dumps({"opening_dataset": {"content_sha256": "fixed-hash"}}) + "\n")
    (previous / "128/results.manifest.json").write_text(json.dumps({"game": "ayo_olopon_alpha_zero", "uct_c": 1.5}))
    monkeypatch.setattr(opening_dataset, "load_dataset", lambda *args: SimpleNamespace(content_sha256="fixed-hash", openings=[None] * 200))
    def baseline(*args, **kwargs):
        order.append("baseline")
        assert kwargs["seed"] == 20261003 and kwargs["uct_c"] == 1.5
        return {"completed_games": 800}
    monkeypatch.setattr(mcts_baseline, "run_baseline", baseline)
    if dependency_success:
        assert remote.baseline_remote.local("baseline", "neural", "fc-test")["status"] == "completed"
        assert order == ["wait", "baseline"]
    else:
        with pytest.raises(RuntimeError, match="upstream failed"):
            remote.baseline_remote.local("baseline", "neural", "fc-test")
        assert order == ["wait"]
        assert json.loads((tmp_path / "baseline/queue.manifest.json").read_text())["status"] == "failed"
