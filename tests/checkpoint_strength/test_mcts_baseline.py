from dataclasses import replace
import json

import pytest
from Algorithms.alpha_zero.runtime import use_repository_open_spiel
use_repository_open_spiel()
import pyspiel
from Algorithms.alpha_zero import game as adapter
from experiments.checkpoint_strength.opening_dataset import load_dataset
from experiments.checkpoint_strength.mcts_baseline import run_baseline


def test_plain_mcts_is_reproducible_and_pairs_identical_openings(tmp_path):
    game = pyspiel.load_game("ayo_olopon_alpha_zero(max_moves=1000,cutoff=collect)")
    full = load_dataset("experiments/checkpoint_strength/dataset/dataset_le4capture_v3_200.json", game)
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
