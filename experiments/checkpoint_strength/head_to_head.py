"""Direct matches between two frozen neural checkpoints on paired openings."""
import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from importlib.metadata import version
import json
from pathlib import Path
import random
import threading
import time

import numpy as np

from experiments.alpha_zero import offline_evaluate as offline
from Algorithms.alpha_zero import game as adapter
from Algorithms.alpha_zero.config import Settings
from Algorithms.alpha_zero.evaluation import make_neural_agent
from experiments.agent_benchmark.random_vs_greedy_hstar import run_matches, summarize
from experiments.checkpoint_strength.metrics import first_player_advantage
from experiments.checkpoint_strength.opening_dataset import ROOT, digest, load_dataset


@dataclass(frozen=True)
class HeadToHeadConfig:
    checkpoints: list[int]
    games: int = 400
    simulations: int = 64
    seed: int = 20261003
    openings: str = "experiments/checkpoint_strength/dataset/ayo_mixed_v2_200.json"

    def __post_init__(self):
        if (not isinstance(self.checkpoints, list) or len(self.checkpoints) != 2
                or any(type(s) is not int or s < 1 for s in self.checkpoints)
                or len(set(self.checkpoints)) != 2):
            raise ValueError("Select exactly two distinct positive checkpoint rounds")
        offline.EvaluationConfig(checkpoints=self.checkpoints, games=self.games,
                                 simulations=self.simulations, seed=self.seed,
                                 openings=self.openings)


def paired_score_interval(records, policy, seed, resamples=10000):
    """Percentile bootstrap of opening-pair mean scores, counting draws as 0.5."""
    pairs = {}
    for record in records:
        pairs.setdefault(record["opening_pair_id"], []).append(record)
    scores = []
    for pair in pairs.values():
        if (len(pair) != 2 or pair[0]["opening_id"] != pair[1]["opening_id"]
                or pair[0]["seed"] != pair[1]["seed"]
                or pair[0]["opening_state_sha256"] != pair[1]["opening_state_sha256"]
                or pair[0]["player_0_policy"] != pair[1]["player_1_policy"]
                or pair[0]["player_1_policy"] != pair[1]["player_0_policy"]):
            raise ValueError("Confidence intervals require complete swapped-seat opening pairs")
        scores.append(np.mean([(r["policy_returns"][policy] + 1) / 2 for r in pair]))
    if not scores:
        raise ValueError("No opening pairs")
    scores = np.asarray(scores)
    rng = np.random.default_rng(seed)
    sampled = scores[rng.integers(len(scores), size=(resamples, len(scores)))].mean(axis=1)
    lower, upper = np.quantile(sampled, [0.025, 0.975])
    return {"method": "opening-pair percentile bootstrap", "confidence_level": 0.95,
            "opening_pairs": len(scores), "resamples": resamples, "seed": seed,
            "score_rate": float(scores.mean()), "lower": float(lower), "upper": float(upper),
            "scope": "Sampled continuation positions; excludes training-seed and search-seed uncertainty"}


def run_head_to_head(run, config, output, *, on_progress=None):
    run, output = Path(run).resolve(), Path(output).resolve()
    if output.is_relative_to(run):
        raise ValueError("Results must be outside the source training run")
    if output.exists():
        raise FileExistsError(output)
    training = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    if (training.get("architecture_version") != "ayo_mlp_185863_v1"
            or training.get("observation_version") != adapter.OBSERVATION_VERSION
            or training.get("value_perspective") != "player_to_move"):
        raise ValueError("Incompatible checkpoint architecture/observation/value contract")
    settings = Settings(**training["settings"])
    if training.get("game") != settings.game_string:
        raise ValueError("Saved game and settings disagree")
    game = offline.pyspiel.load_game(settings.game_string)
    dataset = load_dataset(config.openings, game)
    if config.games > 2 * len(dataset.openings):
        raise ValueError("Requested games exceed twice the opening count; repeats are forbidden")
    started = time.perf_counter()
    print(json.dumps({"event": "head_to_head_preflight", "config": asdict(config)}), flush=True)
    models, _ = offline.prepare_checkpoints(
        run, offline.EvaluationConfig(checkpoints=config.checkpoints), game, training,
        lambda event, **values: print(json.dumps({"event": event, **values}), flush=True))
    by_round = {item["learner_round"]: item for item in models}
    names = [f"C{step}" for step in config.checkpoints]
    factories = {
        name: (lambda game, seed, step=step: make_neural_agent(
            game, by_round[step]["model"], config.simulations, settings.uct_c, seed))
        for name, step in zip(names, config.checkpoints)}
    selected = dataset.openings[:config.games // 2]
    opening_provenance = {
        "benchmark_id": dataset.benchmark_id, "path": dataset.path,
        "file_sha256": dataset.file_sha256, "content_sha256": dataset.content_sha256,
        "opening_count": len(selected), "available_openings": len(dataset.openings),
        "games_per_opening": 2,
        "selection": "entire_dataset" if len(selected) == len(dataset.openings) else "prefix_smoke_test",
        "selected_openings_sha256": digest([{"opening_id": o.opening_id, "actions": o.actions,
                                             "state_sha256": o.state_sha256} for o in selected])}
    sources = [
        "experiments/checkpoint_strength/head_to_head.py",
        "experiments/checkpoint_strength/modal_head_to_head.py",
        "experiments/checkpoint_strength/opening_dataset.py",
        "experiments/checkpoint_strength/metrics.py",
        "experiments/alpha_zero/offline_evaluate.py",
        "experiments/agent_benchmark/random_vs_greedy_hstar.py",
        "Algorithms/alpha_zero/evaluation.py", "Algorithms/alpha_zero/inference.py",
        "Algorithms/alpha_zero/config.py", "Algorithms/alpha_zero/game.py",
        "Algorithms/alpha_zero/runtime.py", "Model/ayo_olopon/ayo_olopon.py",
        "open_spiel/open_spiel/python/algorithms/mcts.py",
        "open_spiel/open_spiel/python/algorithms/alpha_zero/evaluator.py",
        "open_spiel/open_spiel/python/algorithms/alpha_zero/model_linen.py"]
    manifest = {
        "experiment": "neural_checkpoint_head_to_head", "schema_version": 1,
        "status": "running", "run": str(run), "config": asdict(config),
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "game": settings.game_string, "uct_c": settings.uct_c,
        "run_manifest_sha256": offline.source_hash(run / "manifest.json"),
        "model_contract": {key: training[key] for key in (
            "architecture_version", "observation_version", "value_perspective")},
        "checkpoints": [{key: value for key, value in by_round[step].items() if key != "model"}
                        for step in config.checkpoints],
        "opening_dataset": opening_provenance,
        "seed_schedule": "seed + opening index; same seed for each swapped-seat pair",
        "evaluation_policy": {"simulations_per_agent": config.simulations,
                              "dirichlet_noise": None, "temperature_sampling": False,
                              "action_selection": "maximum visit count, lowest action ID breaks ties"},
        "score_definition": "win=1, draw=0.5, loss=0",
        "code_sha256": {name: offline.source_hash(ROOT / name) for name in sources},
        "packages": {name: version(name) for name in ("open_spiel", "numpy", "jax", "flax")},
        "completed_games": 0, "total_games": config.games,
        "artifacts": {"manifest": "results.manifest.json", "games": "results.games.jsonl",
                      "summary": "results.json"}}
    output.mkdir(parents=True, exist_ok=False)

    def save_manifest():
        temporary = output / "results.manifest.json.tmp"
        temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        temporary.replace(output / "results.manifest.json")

    def commit():
        save_manifest()
        if on_progress is not None:
            on_progress(dict(manifest))

    commit()
    stop = threading.Event()

    def heartbeat():
        while not stop.wait(30):
            print(json.dumps({"event": "head_to_head_heartbeat",
                              "completed_games": manifest["completed_games"],
                              "total_games": config.games}), flush=True)

    worker = threading.Thread(target=heartbeat, daemon=True)
    worker.start()
    python_rng, numpy_rng = random.getstate(), np.random.get_state()
    try:
        random.seed(config.seed)
        np.random.seed(config.seed)
        with (output / "results.games.jsonl").open("x", encoding="utf-8") as stream:
            def record_game(record):
                stream.write(json.dumps(record) + "\n")
                stream.flush()
                manifest["completed_games"] += 1
                done = manifest["completed_games"]
                if done == 1 or done % 10 == 0:
                    print(json.dumps({"event": "head_to_head_game_completed", "completed_games": done,
                                      "total_games": config.games, "opening_id": record["opening_id"]}), flush=True)
                if done % 20 == 0:
                    commit()
            summary, records = run_matches(game, factories, config.games, config.seed,
                                           openings=dataset, on_game=record_game)
        for item in models:
            if offline.source_hash(item["checkpoint_source"]) != item["checkpoint_sha256"]:
                raise RuntimeError("Source checkpoint changed during evaluation")
        if offline.source_hash(run / "manifest.json") != manifest["run_manifest_sha256"]:
            raise RuntimeError("Training manifest changed during evaluation")
        report = {
            "status": "completed", "config": asdict(config), "opening_dataset": opening_provenance,
            "policies": summary["policies"], "games": summary["games"],
            "termination_reasons": summary["termination_reasons"],
            "by_seat": {name: {f"P{seat}": summarize([
                r for r in records if r[f"player_{seat}_policy"] == name])["policies"][name]
                for seat in (0, 1)} for name in names},
            "by_opening_depth": {str(depth): summarize([
                r for r in records if r["opening_plies"] == depth])["policies"]
                for depth in sorted({r["opening_plies"] for r in records})},
            "first_player_advantage": first_player_advantage(records),
            "score_intervals": {name: paired_score_interval(records, name, config.seed) for name in names}}
        (output / "results.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        manifest.update(status="completed", finished_at_utc=datetime.now(timezone.utc).isoformat(),
                        elapsed_seconds=time.perf_counter() - started,
                        games_sha256=offline.source_hash(output / "results.games.jsonl"),
                        summary_sha256=offline.source_hash(output / "results.json"))
        commit()
        print(json.dumps({"event": "head_to_head_completed", "policies": report["policies"],
                          "score_intervals": report["score_intervals"]}), flush=True)
        return report
    except BaseException as exc:
        manifest.update(status="failed", error=repr(exc))
        commit()
        raise
    finally:
        stop.set()
        worker.join(timeout=1)
        random.setstate(python_rng)
        np.random.set_state(numpy_rng)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = HeadToHeadConfig(**json.loads(args.config.read_text(encoding="utf-8")))
    run_head_to_head(args.run, config, args.output)


if __name__ == "__main__":
    main()
