"""First-player outcomes combine both policy assignments and count draws."""
import pytest
from experiments.checkpoint_strength.metrics import first_player_advantage


def test_first_player_measure_is_independent_of_agent_identity():
    records = [{"returns_by_player": [1, -1], "opening_plies": 2,
                "player_0_policy": "ALPHAZERO", "player_1_policy": "OPPONENT"},
               {"returns_by_player": [1, -1], "opening_plies": 2,
                "player_0_policy": "OPPONENT", "player_1_policy": "ALPHAZERO"}]
    stats = first_player_advantage(records)
    assert stats["P0"]["wins"] == 2 and stats["P0"]["score_rate"] == 1
    assert stats["P1"]["losses"] == 2 and stats["P1"]["score_rate"] == 0
    assert stats["p0_score_excess_over_half"] == 0.5
    assert stats["by_opening_depth"]["2"]["P0"] == stats["P0"]


def test_draws_and_depth_groups_are_accounted_for():
    records = [{"returns_by_player": r, "opening_plies": d}
               for r, d in [([1, -1], 4), ([-1, 1], 6), ([0, 0], 6)]]
    stats = first_player_advantage(records)
    assert stats["P0"]["wins"] == stats["P0"]["draws"] == stats["P0"]["losses"] == 1
    assert stats["P0"]["score_rate"] == stats["P1"]["score_rate"] == 0.5
    assert stats["p0_score_excess_over_half"] == 0
    assert stats["by_opening_depth"]["6"]["P0"]["score_rate"] == 0.25
    with pytest.raises(ValueError):
        first_player_advantage([])
    with pytest.raises(ValueError):
        first_player_advantage([{"returns_by_player": [1, -1], "opening_plies": 3}])
