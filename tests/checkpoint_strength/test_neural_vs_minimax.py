"""Reject protocol and outcome corruption before accepting a minimax pair."""
from copy import deepcopy

import numpy as np
import pytest

from experiments.checkpoint_strength import neural_vs_minimax as comparison


class Model:
    def inference(self, observation, mask):
        mask = np.asarray(mask, dtype=float)
        return 0.0, mask / mask.sum()


@pytest.fixture(scope="module")
def completed_pair():
    config = comparison.offline.EvaluationConfig(
        checkpoints=[600], games=2, simulations=2, seed=20261003,
        opponent={"name": "MINIMAX", "params": {"maximum_depth": 1}})
    game = comparison.offline.pyspiel.load_game("ayo_olopon_alpha_zero(max_moves=1000,cutoff=collect)")
    dataset = comparison.load_dataset(
        "experiments/checkpoint_strength/dataset/dataset_le4capture_v3_200.json", game)
    previous = comparison.WORKER
    try:
        comparison.WORKER = game, Model(), config, 1.5
        records = comparison.play_opening(0, dataset.openings[0], dataset.content_sha256)
    finally:
        comparison.WORKER = previous
    return game, records, dataset.openings[0], config, dataset.content_sha256


def test_pair_replays_and_has_swapped_seats(completed_pair):
    game, records, opening, config, digest = completed_pair
    comparison.verify_pair(game, records, 0, opening, config, digest)
    interval = comparison.paired_score_interval(records, "C600", config.seed)
    assert interval["opening_pairs"] == 1


@pytest.mark.parametrize("field", ["seed", "game_id", "opening_pair_id", "final_captured",
                                  "returns_by_player", "policy_outcome", "continuation_actions"])
def test_corrupted_saved_pair_is_rejected(completed_pair, field):
    game, original, opening, config, digest = completed_pair
    records = deepcopy(original)
    if field in ("seed", "game_id", "opening_pair_id"):
        records[0][field] += 1
    elif field == "policy_outcome":
        records[0][field]["C600"] = "invalid"
    elif field == "continuation_actions":
        records[0][field][0] = 99
    else:
        records[0][field][0] += 1
    with pytest.raises(ValueError):
        comparison.verify_pair(game, records, 0, opening, config, digest)
