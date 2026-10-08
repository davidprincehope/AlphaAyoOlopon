"""Protocols and result checks for four 13-checkpoint MCTS progression studies."""
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path

from experiments.alpha_zero.modal_common import ROOT
from experiments.checkpoint_strength.neural_vs_mcts import NeuralVsMCTSConfig

CHECKPOINTS = (1, *range(50, 601, 50))
CASES = (
    ("puct64-mcts64", "c600_vs_mcts_64.json"),
    ("puct128-mcts128", "c600_vs_mcts_128.json"),
    ("puct64-mcts128", "c600_64_vs_mcts_128.json"),
    ("policy-mcts64", "c600_policy_vs_mcts_64.json"),
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def configurations():
    return {
        name: {step: replace(NeuralVsMCTSConfig(**json.loads(
            (ROOT / "experiments/checkpoint_strength/configs" / filename).read_text(encoding="utf-8"))),
            checkpoint=step) for step in CHECKPOINTS}
        for name, filename in CASES
    }


def make_plan(run, remote_run, evaluation_name):
    run = Path(run)
    configs = configurations()
    return {
        "experiment": "neural_mcts_learning_progression", "evaluation_name": evaluation_name,
        "remote_run": remote_run, "checkpoints": list(CHECKPOINTS),
        "source_run": run.relative_to(ROOT).as_posix(),
        "training_manifest_sha256": sha(run / "manifest.json"),
        "checkpoint_sha256": {str(step): sha(run / "inference-checkpoints" / f"step-{step:06d}.npz")
                              for step in CHECKPOINTS},
        "opening_dataset_sha256": json.loads((ROOT / next(iter(configs.values()))[1].openings)
                                               .read_text(encoding="utf-8"))["content_sha256"],
        "games_per_checkpoint": 400, "games_per_condition": 5200, "total_games": 20800,
        "opening_pairs": 200, "seed": 20261003,
        "cases": {name: {str(step): asdict(config) for step, config in rows.items()}
                  for name, rows in configs.items()},
        "score_definition": "(wins + 0.5 * draws) / games",
        "confidence_intervals": "95% percentile bootstrap of 200 whole opening pairs, 10000 resamples",
        "source_mount": "read_only", "max_gpu_containers": 2,
    }


def result_row(case, step, result, manifest, plan):
    expected = plan["cases"][case][str(step)]
    policy = f"C{step}"
    if (result["status"] != "completed" or manifest["status"] != "completed"
            or result["config"] != expected or manifest["config"] != expected
            or result["games"] != 400 or manifest["completed_games"] != 400):
        raise ValueError(f"Incomplete or mismatched progression case: {case}/C{step}")
    if (manifest["checkpoints"][0]["checkpoint_sha256"] != plan["checkpoint_sha256"][str(step)]
            or manifest["run_manifest_sha256"] != plan["training_manifest_sha256"]
            or result["opening_dataset"]["content_sha256"] != plan["opening_dataset_sha256"]):
        raise ValueError(f"Progression source mismatch: {case}/C{step}")
    stats, interval = result["policies"][policy], result["score_intervals"][policy]
    if (stats["wins"] + stats["draws"] + stats["losses"] != 400
            or stats["score_rate"] != (stats["wins"] + 0.5 * stats["draws"]) / 400
            or interval["opening_pairs"] != 200 or interval["confidence_level"] != 0.95):
        raise ValueError(f"Progression statistics mismatch: {case}/C{step}")
    return {
        "condition": case, "learner_round": step, "games": 400,
        **{key: stats[key] for key in ("wins", "draws", "losses", "score_rate", "win_rate")},
        "score_interval": interval, "by_seat": result["by_seat"][policy],
        "by_opening_depth": {depth: policies[policy] for depth, policies in result["by_opening_depth"].items()},
        "configuration": expected, "checkpoint_sha256": plan["checkpoint_sha256"][str(step)],
        "artifacts": f"{case}/c{step:06d}",
    }


def validate_curve(rows):
    if [row["learner_round"] for row in rows] != list(CHECKPOINTS):
        raise ValueError("A completed progression requires exactly the 13 requested checkpoints in order")
    if sum(row["games"] for row in rows) != 5200:
        raise ValueError("A completed progression requires 5200 games")
