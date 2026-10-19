"""Checks against censored outcomes, corrupt replay evidence and mixed protocols."""

from copy import deepcopy
import json

import pytest

from experiments.minimax_depth_vs_random import rerun_experiment as experiment


@pytest.mark.parametrize("game_horizon", [1, 1000])
def test_internal_and_external_limits_are_unresolved(game_horizon):
    game = experiment.pyspiel.load_game("ayo_olopon", {"max_game_length": game_horizon})
    bot = experiment.ayo_minimax.make_bot(game, maximum_depth=1)
    record = experiment.play_game(game, bot, 1, 0, 42, 1)
    assert record["engine_terminal"] == (game_horizon == 1)
    assert record["truncated"]
    assert record["minimax_outcome"] == "unresolved"
    assert record["returns_by_player"] is None
    summary = experiment.summarize([record])
    assert summary["wins"] == summary["draws"] == summary["losses"] == 0
    assert summary["score_rate"] is None
    assert summary["score_rate_bounds"] == [0.0, 1.0]
    experiment.verify_record(game, record, 1, 0, 42, 1)


@pytest.fixture(scope="module")
def completed_game():
    game = experiment.pyspiel.load_game("ayo_olopon")
    bot = experiment.ayo_minimax.make_bot(game, maximum_depth=1)
    return game, experiment.play_game(game, bot, 1, 1, 42, 1000)


@pytest.mark.parametrize("field", ["seed", "minimax_outcome", "final_captured", "actions"])
def test_corrupt_saved_game_is_rejected(completed_game, field):
    game, original = completed_game
    record = deepcopy(original)
    if field == "seed":
        record[field] += 1
    elif field == "minimax_outcome":
        record[field] = "unresolved"
    else:
        record[field][0] += 99
    with pytest.raises(ValueError):
        experiment.verify_record(game, record, 1, 1, 42, 1000)


def test_incorrect_random_seed_actions_are_rejected(completed_game):
    game, original = completed_game
    record = deepcopy(original)
    record["seed"] = 43
    with pytest.raises(ValueError, match="Random action"):
        experiment.verify_record(game, record, 1, 1, 43, 1000)


def test_resume_replays_evidence_and_rejects_changed_protocol(tmp_path):
    report = experiment.run(2, [1, 2], 42, 1000, tmp_path, workers=1)
    assert all(row["verification"]["games_replayed"] == 2 for row in report["results"])
    resumed = experiment.run(2, [1, 2], 42, 1000, tmp_path, workers=1, resume=True)
    assert report == resumed
    with pytest.raises(ValueError, match="unchanged"):
        experiment.run(2, [1, 2], 43, 1000, tmp_path, workers=1, resume=True)
    with pytest.raises(FileExistsError):
        experiment.run(2, [1, 2], 42, 1000, tmp_path, workers=1)
    log_path = tmp_path / "depth_1.games.jsonl"
    rows = [json.loads(line) for line in log_path.read_text().splitlines()]
    rows[0]["returns_by_player"][0] += 1
    log_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    with pytest.raises(ValueError, match="Replay"):
        experiment.run(2, [1, 2], 42, 1000, tmp_path, workers=1, resume=True)


def test_bootstrap_preserves_seat_pair_dependence():
    # Every pair scores exactly half, although individual game scores vary.
    records = [{"game_id": i, "minimax_outcome": result, "truncated": False}
               for i, result in enumerate(("win", "loss", "loss", "win"))]
    scores = experiment.pair_scores(list(reversed(records)))
    interval = experiment.bootstrap_interval(scores, 42)
    assert interval["lower"] == interval["upper"] == 0.5
    records[0]["truncated"] = True
    assert experiment.pair_scores(records) is None
