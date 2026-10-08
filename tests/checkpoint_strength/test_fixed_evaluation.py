"""The new opt-in protocol pairs openings without changing historical defaults."""
from collections import Counter
from dataclasses import asdict, replace
import json

import numpy as np
import pytest
from Algorithms.alpha_zero.runtime import use_repository_open_spiel
use_repository_open_spiel()
import pyspiel
from Algorithms.alpha_zero import game as adapter  # noqa: F401
from Algorithms.alpha_zero.config import Settings
from Algorithms.alpha_zero.evaluation import evaluate_model
from experiments.alpha_zero import offline_evaluate as offline
from experiments.agent_benchmark.agents import make_agent
from experiments.agent_benchmark.random_vs_greedy_hstar import run_matches
from experiments.checkpoint_strength.opening_dataset import DEFAULT_OPENINGS, ROOT, load_dataset, replay


class UniformModel:
    def inference(self, observation, mask):
        policy = np.asarray(mask, dtype=float)
        return 0.0, policy / policy.sum()


def test_all_100_openings_are_paired_once_and_replay_to_logged_results():
    game = pyspiel.load_game("ayo_olopon")
    dataset = load_dataset(DEFAULT_OPENINGS, game)
    factories = {name: lambda game, seed, name=name: make_agent(game, seed, {"name": name})
                 for name in ("GREEDY_HSTAR", "RAND")}
    summary, records = run_matches(game, factories, 200, 17, openings=dataset)
    assert summary["games"] == 200
    assert summary["policies"]["GREEDY_HSTAR"]["seats"] == {"0": 100, "1": 100}
    assert len({r["opening_position_sha256"] for r in records}) == 100
    assert set(Counter(r["opening_id"] for r in records).values()) == {2}
    for index in range(100):
        a, b = records[2 * index:2 * index + 2]
        assert a["opening_id"] == b["opening_id"] == dataset.openings[index].opening_id
        assert a["opening_actions"] == b["opening_actions"]
        assert a["player_0_policy"] == b["player_1_policy"] == "GREEDY_HSTAR"
        assert a["seed"] == b["seed"] == 17 + index
        for record in (a, b):
            assert record["game_length"] == 6 + record["continuation_length"] <= 1000
            final = replay(game, record["opening_actions"] + record["continuation_actions"])
            assert final.is_terminal() and final.captured == record["final_captured"]
    with pytest.raises(ValueError, match="repeats are forbidden"):
        run_matches(game, factories, 202, openings=dataset)


def test_models_share_identical_openings_and_fixed_seed_order():
    settings = Settings(max_simulations=4)
    game = pyspiel.load_game(settings.game_string)
    dataset = load_dataset(DEFAULT_OPENINGS, game)
    reports = [evaluate_model(game, UniformModel(), settings, checkpoint, games=4,
                              opponent={"name": "GREEDY_HSTAR"}, openings=dataset) for checkpoint in (1, 600)]
    strip_timing = lambda rows: [{k: v for k, v in r.items() if k != "elapsed_seconds"} for r in rows]
    assert strip_timing(reports[0]["matches"]) == strip_timing(reports[1]["matches"])


def test_new_config_is_opt_in_and_invalid_dataset_fails_before_artifacts(tmp_path):
    config = offline.load_config(ROOT / "experiments/checkpoint_strength/configs/progression_fixed_v1.json")
    historical = offline.load_config(ROOT / "experiments/alpha_zero/configs/progression_phase1.json")
    assert config.openings == DEFAULT_OPENINGS and historical.openings is None
    assert config.games == 200 and len(config.checkpoints) == 13 and config.simulations == 64
    run = tmp_path / "source"
    run.mkdir()
    settings = Settings()
    manifest = {"settings": asdict(settings), "game": settings.game_string,
                "observation_version": adapter.OBSERVATION_VERSION,
                "architecture_version": "ayo_mlp_185863_v1", "value_perspective": "player_to_move"}
    (run / "manifest.json").write_text(json.dumps(manifest))
    output = tmp_path / "results.jsonl"
    with pytest.raises(FileNotFoundError):
        offline.run_evaluation(run, replace(config, openings=str(tmp_path / "missing.json")), output)
    assert not output.exists()
