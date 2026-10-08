"""Check two-model routing, pair-level uncertainty, and immutable artifacts."""
from dataclasses import asdict
import json

import numpy as np
import pytest

from Algorithms.alpha_zero.config import Settings
from experiments.checkpoint_strength import head_to_head as h2h
from experiments.checkpoint_strength.opening_dataset import replay


class Model:
    def __init__(self, step):
        self.step = step

    def inference(self, observation, mask):
        weights = np.arange(1, 7, dtype=float)
        if self.step == 350:
            weights = weights[::-1]
        policy = weights * mask
        return 0.0, policy / policy.sum()


@pytest.fixture
def source(tmp_path, monkeypatch):
    run = tmp_path / "training"
    run.mkdir()
    settings = Settings()
    manifest = {"settings": asdict(settings), "game": settings.game_string,
                "architecture_version": "ayo_mlp_185863_v1",
                "observation_version": h2h.adapter.OBSERVATION_VERSION,
                "value_perspective": "player_to_move"}
    (run / "manifest.json").write_text(json.dumps(manifest))
    items = []
    for step in (50, 350):
        path = run / f"model-{step}"
        path.write_text(f"frozen parameters for {step}")
        items.append({"learner_round": step, "requested_rounds": [step],
                      "checkpoint_source": str(path), "checkpoint_sha256": h2h.offline.source_hash(path),
                      "model": Model(step)})
    monkeypatch.setattr(h2h.offline, "prepare_checkpoints", lambda *args: (items, []))
    return run


def test_two_models_play_both_seats_and_logs_replay(source, tmp_path, monkeypatch):
    calls = []
    make = h2h.make_neural_agent

    def traced(game, model, simulations, uct_c, seed):
        calls.append((model.step, simulations, seed))
        return make(game, model, simulations, uct_c, seed)

    monkeypatch.setattr(h2h, "make_neural_agent", traced)
    config = h2h.HeadToHeadConfig([350, 50], games=4, simulations=4, seed=23)
    output = tmp_path / "comparison"
    before = h2h.offline.source_hash(source)
    report = h2h.run_head_to_head(source, config, output)
    manifest = json.loads((output / "results.manifest.json").read_text())
    records = [json.loads(line) for line in (output / "results.games.jsonl").read_text().splitlines()]
    assert manifest["status"] == "completed" and manifest["completed_games"] == 4
    assert manifest["opening_dataset"]["selection"] == "prefix_smoke_test"
    assert [item["learner_round"] for item in manifest["checkpoints"]] == [350, 50]
    assert calls == [(step, 4, seed) for seed in (23, 23, 24, 24) for step in (350, 50)]
    assert report["policies"]["C350"]["wins"] == report["policies"]["C50"]["losses"]
    assert report["policies"]["C350"]["score_rate"] + report["policies"]["C50"]["score_rate"] == 1
    for name in ("C50", "C350"):
        assert all(report["by_seat"][name][seat]["games"] == 2 for seat in ("P0", "P1"))
    game = h2h.offline.pyspiel.load_game(Settings().game_string)
    for index in range(2):
        a, b = records[2 * index:2 * index + 2]
        assert a["opening_id"] == b["opening_id"] and a["seed"] == b["seed"]
        assert a["player_0_policy"] == b["player_1_policy"] == "C350"
        for record in (a, b):
            state = replay(game, record["opening_actions"] + record["continuation_actions"])
            assert state.is_terminal() and list(state.returns()) == record["returns_by_player"]
            assert record["game_length"] <= 1000
    assert h2h.offline.source_hash(source) == before
    with pytest.raises(FileExistsError):
        h2h.run_head_to_head(source, config, output)
    with pytest.raises(ValueError, match="outside"):
        h2h.run_head_to_head(source, config, source / "results")


def pair_records(outcomes):
    return [{"opening_pair_id": index // 2, "opening_id": str(index // 2), "seed": index // 2,
             "opening_state_sha256": str(index // 2),
             "player_0_policy": "C50" if index % 2 == 0 else "C350",
             "player_1_policy": "C350" if index % 2 == 0 else "C50",
             "policy_returns": {"C350": value, "C50": -value}}
            for index, value in enumerate(outcomes)]


def test_interval_resamples_pairs_instead_of_individual_games():
    split = h2h.paired_score_interval(pair_records([1, -1, 1, -1]), "C350", 17)
    clustered = h2h.paired_score_interval(pair_records([1, 1, -1, -1]), "C350", 17)
    assert split["score_rate"] == clustered["score_rate"] == 0.5
    assert split["lower"] == split["upper"] == 0.5
    assert clustered["lower"] == 0 and clustered["upper"] == 1
    with pytest.raises(ValueError, match="complete"):
        h2h.paired_score_interval(pair_records([1, -1, 1]), "C350", 17)


def test_changed_source_fails_and_retains_partial_artifacts(source, tmp_path):
    output = tmp_path / "comparison"

    def mutate(manifest):
        if manifest["completed_games"] == 0:
            (source / "model-50").write_text("changed parameters")

    with pytest.raises(RuntimeError, match="checkpoint changed"):
        h2h.run_head_to_head(source, h2h.HeadToHeadConfig([50, 350], games=2, simulations=2),
                             output, on_progress=mutate)
    manifest = json.loads((output / "results.manifest.json").read_text())
    assert manifest["status"] == "failed" and manifest["completed_games"] == 2
    assert not (output / "results.json").exists()


@pytest.mark.parametrize("changes", [{"checkpoints": [50, 50]}, {"checkpoints": [0, 350]},
                                    {"games": 3}, {"simulations": 1}, {"seed": -1}])
def test_invalid_experiment_configuration(changes):
    with pytest.raises(ValueError):
        h2h.HeadToHeadConfig(**{"checkpoints": [50, 350], **changes})
