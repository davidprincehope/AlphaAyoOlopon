"""Guard the approved evaluation scope against excluded or duplicate cases."""
from collections import Counter
from copy import deepcopy
import json

import pytest
from experiments.checkpoint_strength.le4capture_suite import (
    ROOT, CHECKPOINTS, DATASET_HASH, EXCLUDED_CURVES, SOURCE_RUN, SEED, case_specs, validate_plan,
)
from experiments.checkpoint_strength.opening_dataset import LEGACY_OPENINGS


def plan():
    return {"schema_version": 2, "cases": case_specs(), "opening_dataset_sha256": DATASET_HASH,
            "source_run": SOURCE_RUN, "seed": SEED, "total_games": 17600, "total_cases": 44,
            "excluded_curves": list(EXCLUDED_CURVES),
            "checkpoint_sha256": {str(step): "0" * 64 for step in CHECKPOINTS}}


def test_selected_suite_contains_every_allowed_case_once():
    cases = validate_plan(plan())["cases"]
    assert len({case["id"] for case in cases}) == 44
    assert sum(case["config"]["games"] for case in cases) == 17600
    assert Counter(case["kind"] for case in cases) == {
        "greedy_eoh": 13, "neural_vs_mcts": 27, "head_to_head": 2, "mcts_vs_greedy_eoh": 2}
    assert [case["config"]["checkpoint"] for case in cases if case["id"].startswith("puct64-mcts128/")] == [600]
    assert len([case for case in cases if case["id"].startswith("policy-mcts64/")]) == 13
    assert all(case["config"]["simulations"] == 64 for case in cases if case["kind"] == "greedy_eoh")
    assert all(case["config"]["simulations"] in (0, 64) for case in cases if case["kind"] == "neural_vs_mcts")


def test_c600_endpoints_precede_progression_and_are_not_duplicated():
    cases = case_specs()
    assert [case["id"] for case in cases[:4]] == [
        "greedy64/c000600", "puct64-mcts64/c000600", "puct64-mcts128/c000600", "policy-mcts64/c000600"]
    assert [case["config"]["checkpoints"] for case in cases[4:6]] == [[50, 350], [350, 600]]
    assert [case["config"]["simulations"] for case in cases[6:8]] == [64, 128]
    assert not any(case["config"].get("checkpoint") == 600 for case in cases[8:])


@pytest.mark.parametrize("mutation", [
    lambda p: p["cases"].pop(),
    lambda p: p["cases"].append(deepcopy(p["cases"][0])),
    lambda p: p["cases"][0]["config"].update(simulations=128),
    lambda p: p["cases"][1]["config"].update(simulations=128, mcts_simulations=128),
    lambda p: p.update(opening_dataset_sha256="0" * 64),
    lambda p: p.update(seed=1),
    lambda p: p["checkpoint_sha256"].pop("600"),
])
def test_changed_or_excluded_protocol_is_rejected(mutation):
    value = plan()
    mutation(value)
    with pytest.raises(ValueError):
        validate_plan(value)


def test_historical_plan_validates_without_rewriting_its_paths():
    value = plan()
    value["opening_dataset"] = LEGACY_OPENINGS
    for case in value["cases"]:
        case["config"]["openings"] = LEGACY_OPENINGS
    before = deepcopy(value)
    assert validate_plan(value) is value
    assert value == before


def test_published_results_load_with_the_renamed_book():
    from experiments.checkpoint_strength.collect_le4capture_suite import load_published_suite
    _, rows, verified = load_published_suite(ROOT / "docs/results/checkpoint_progression")
    assert len(rows) == 56
    assert sum(row["games"] for row in rows) == verified["games_replayed"] == 22400


@pytest.mark.parametrize("filename", ["results.json", "experiment.plan.json", "verification.json"])
def test_published_plotting_rejects_modified_evidence(tmp_path, filename):
    from experiments.checkpoint_strength.collect_le4capture_suite import load_published_suite
    source = ROOT / "docs/results/checkpoint_progression"
    for name in ("results.json", "experiment.plan.json", "verification.json", "source_provenance.json"):
        (tmp_path / name).write_bytes((source / name).read_bytes())
    path = tmp_path / filename
    data = json.loads(path.read_text(encoding="utf-8"))
    if filename == "results.json":
        data[0]["policies"]["ALPHAZERO"]["wins"] += 1
    elif filename == "experiment.plan.json":
        data["seed"] += 1
    else:
        data["status"] = "failed"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="Published suite artifact changed"):
        load_published_suite(tmp_path)
