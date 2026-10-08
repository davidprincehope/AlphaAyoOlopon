"""A progression must compare all requested checkpoints under identical protocols."""
from dataclasses import asdict
import copy
import json

import pytest

from experiments.checkpoint_strength.mcts_progression import CHECKPOINTS, configurations, result_row, validate_curve
from experiments.alpha_zero.modal_common import ROOT


def test_protocol_matrix_has_all_52_cases():
    matrix = configurations()
    assert len(matrix) == 4
    assert sum(len(rows) for rows in matrix.values()) == 52
    assert list(CHECKPOINTS) == [1, *range(50, 601, 50)]
    for rows in matrix.values():
        assert tuple(rows) == CHECKPOINTS
        original = asdict(rows[600])
        for step, config in rows.items():
            expected = {**original, "checkpoint": step}
            assert asdict(config) == expected
            assert config.games == 400
    assert [(rows[1].neural_mode, rows[1].simulations, rows[1].mcts_budget)
            for rows in matrix.values()] == [("puct", 64, 64), ("puct", 128, 128),
                                            ("puct", 64, 128), ("policy", 0, 64)]


def evidence():
    folder = ROOT / "docs/results/alpha_zero_600/checkpoint-c600-vs-mcts-64-20261005"
    result = json.loads((folder / "results.json").read_text())
    manifest = json.loads((folder / "results.manifest.json").read_text())
    result["config"] = manifest["config"] = asdict(configurations()["puct64-mcts64"][600])
    plan = {"cases": {"puct64-mcts64": {"600": result["config"]}},
            "checkpoint_sha256": {"600": manifest["checkpoints"][0]["checkpoint_sha256"]},
            "training_manifest_sha256": manifest["run_manifest_sha256"],
            "opening_dataset_sha256": result["opening_dataset"]["content_sha256"]}
    return result, manifest, plan


def test_real_completed_report_produces_progression_row():
    result, manifest, plan = evidence()
    row = result_row("puct64-mcts64", 600, result, manifest, plan)
    assert row["learner_round"] == 600 and row["games"] == 400
    assert row["score_rate"] == 0.6975


@pytest.mark.parametrize("change", ["budget", "games", "checkpoint_hash", "dataset", "score"])
def test_mismatched_report_cannot_enter_curve(change):
    result, manifest, plan = evidence()
    result, manifest = copy.deepcopy(result), copy.deepcopy(manifest)
    if change == "budget":
        result["config"]["simulations"] = 128
    elif change == "games":
        result["games"] = 200
    elif change == "checkpoint_hash":
        manifest["checkpoints"][0]["checkpoint_sha256"] = "wrong"
    elif change == "dataset":
        result["opening_dataset"]["content_sha256"] = "wrong"
    else:
        result["policies"]["C600"]["score_rate"] = 0.5
    with pytest.raises(ValueError):
        result_row("puct64-mcts64", 600, result, manifest, plan)


def test_partial_or_duplicate_checkpoints_cannot_be_called_complete():
    rows = [{"learner_round": step, "games": 400} for step in CHECKPOINTS]
    validate_curve(rows)
    with pytest.raises(ValueError):
        validate_curve(rows[:-1])
    with pytest.raises(ValueError):
        validate_curve(rows[:-1] + [rows[0]])
