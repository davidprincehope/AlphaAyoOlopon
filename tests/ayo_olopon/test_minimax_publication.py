"""Protect the published minimax evidence and the corrected default command."""

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import pytest

from scripts import check_repository

ROOT = Path(__file__).resolve().parents[2]
PREFIX = "docs/results/classical_search/corrected_minimax_depths_1_to_5_vs_random/"


@pytest.fixture(scope="module")
def publication():
    report = json.loads((ROOT / PREFIX / "results.json").read_text(encoding="utf-8"))
    names = [PREFIX + "results.json", PREFIX + "paired_scores.csv", *report["contract"]["code_sha256"]]
    return {name: (ROOT / name).read_bytes() for name in names}


def test_verified_publication_matches_all_recorded_source_hashes_and_pair_scores(publication):
    assert check_repository.validate_minimax_publication(publication) == []


@pytest.mark.parametrize("corruption", ["outcomes", "source", "csv", "superseded", "interval"])
def test_corrupted_or_superseded_minimax_evidence_is_rejected(publication, corruption):
    content = deepcopy(publication)
    report = json.loads(content[PREFIX + "results.json"])
    if corruption == "outcomes":
        report["results"][0]["wins"] += 1
    elif corruption == "source":
        content["Algorithms/H_star_ayo.py"] += b"\n# changed evaluator\n"
    elif corruption == "csv":
        content[PREFIX + "paired_scores.csv"] += b"bad row\n"
    elif corruption == "superseded":
        content["experiments/minimax_depth_vs_random/results/depth_1_summary.json"] = b"{}"
    else:
        report["results"][0]["score_interval_95"]["resamples"] = 1
    content[PREFIX + "results.json"] = json.dumps(report).encode()
    assert check_repository.validate_minimax_publication(content)


def test_default_cli_censors_action_limits_instead_of_awarding_a_result(tmp_path):
    output = tmp_path / "censored"
    subprocess.run([sys.executable, "-B", "experiments/minimax_depth_vs_random/run_experiment.py",
                    "--games", "2", "--depths", "1", "--max-actions", "1", "--workers", "1",
                    "--output-dir", str(output)], cwd=ROOT, check=True, capture_output=True, timeout=60)
    result, = json.loads((output / "experiment_summary.json").read_text(encoding="utf-8"))["results"]
    assert result["unresolved_games"] == 2
    assert result["wins"] == result["draws"] == result["losses"] == 0
    assert result["score_rate"] is None
