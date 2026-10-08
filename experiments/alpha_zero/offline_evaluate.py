"""Evaluate frozen learner rounds through the shared Ayo agent/match runner."""
import argparse
from contextlib import ExitStack
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import threading
import time

from Algorithms.alpha_zero.runtime import use_repository_open_spiel
use_repository_open_spiel()
import pyspiel
from Algorithms.alpha_zero import game as adapter
from Algorithms.alpha_zero.config import Settings
from Algorithms.alpha_zero.evaluation import evaluate_model
from Algorithms.alpha_zero.inference import InferenceModel, checkpoint_path, load_parameters


@dataclass(frozen=True)
class EvaluationConfig:
    checkpoints: list[int] | None = None  # None selects all retained trained rounds.
    games: int = 100
    simulations: int | None = None
    opponent: dict = field(default_factory=lambda: {"name": "RAND", "params": {}})
    seed: int = 0
    fallback_first: bool = False
    heartbeat_seconds: int = 30
    openings: str | None = None  # Opt-in; historical configurations retain standard starts.

    def __post_init__(self):
        if type(self.games) is not int or self.games < 2 or self.games % 2:
            raise ValueError("games must be positive and even")
        if self.openings is not None and (not isinstance(self.openings, str) or not self.openings.strip()):
            raise ValueError("openings must be a saved dataset path")
        if self.simulations is not None and (type(self.simulations) is not int or self.simulations < 2):
            raise ValueError("simulations must be >= 2")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be nonnegative")
        if type(self.fallback_first) is not bool:
            raise ValueError("fallback_first must be a boolean")
        if type(self.heartbeat_seconds) is not int or not 1 <= self.heartbeat_seconds <= 60:
            raise ValueError("heartbeat_seconds must be between 1 and 60")
        if self.checkpoints is not None and (not isinstance(self.checkpoints, list) or
                not self.checkpoints or any(type(s) is not int or s < 1 for s in self.checkpoints)):
            raise ValueError("checkpoints must contain positive trained learner rounds (C0 is untrained)")
        if (not isinstance(self.opponent, dict) or not isinstance(self.opponent.get("name"), str)
                or not isinstance(self.opponent.get("params", {}), dict)):
            raise ValueError("opponent requires a name and a params object")


def load_config(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Evaluation configuration must be a JSON object")
    return EvaluationConfig(**data)


def available_steps(run):
    run = Path(run)
    steps = {int(p.stem.removeprefix("step-"))
             for p in (run / "inference-checkpoints").glob("step-*.npz")
             if p.stem.removeprefix("step-").isdigit()}
    steps.update(int(p.name.removeprefix("checkpoint-"))
                 for p in run.glob("checkpoint-*")
                 if p.is_dir() and p.name.removeprefix("checkpoint-").isdigit())
    steps.update(int(p.name.removeprefix("step-"))
                 for p in (run / "training-checkpoints").glob("step-*")
                 if p.name.removeprefix("step-").isdigit() and (p / "COMPLETE").exists())
    return sorted(s for s in steps if s > 0)


def parse_steps(tokens):
    steps = set()
    for token in tokens:
        for part in token.split(","):
            if ":" in part:
                values = [int(x) for x in part.split(":")]
                if len(values) not in (2, 3):
                    raise ValueError("Use START:END[:STRIDE], with an inclusive end")
                start, end = values[:2]
                stride = values[2] if len(values) == 3 else 1
                if start < 1 or end < start or stride < 1:
                    raise ValueError("Invalid trained checkpoint range")
                steps.update(range(start, end + 1, stride))
            else:
                step = int(part)
                if step < 1:
                    raise ValueError("Select positive trained learner rounds; C0 is untrained and -1 is mutable")
                steps.add(step)
    if not steps:
        raise ValueError("No checkpoints selected")
    return sorted(steps)


def source_hash(path):
    """Hash a parameter export or the files in a historical model checkpoint."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    digest = hashlib.sha256()
    files = [path] if path.is_file() else sorted(p for p in path.rglob("*") if p.is_file())
    for file in files:
        if path.is_dir():
            digest.update(file.relative_to(path).as_posix().encode())
        with file.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
    return digest.hexdigest()


def prepare_checkpoints(run, config, game, manifest, log):
    """Preflight all frozen models; C1 alone may use the earliest valid trained round."""
    import numpy as np
    requested = sorted(set(config.checkpoints)) if config.checkpoints is not None else available_steps(run)
    if not requested:
        raise ValueError("No retained trained checkpoints found")
    state = game.new_initial_state()
    loaded = {}
    def load(step):
        if step not in loaded:
            path = checkpoint_path(run, step)
            if path.is_file():
                with np.load(path, allow_pickle=False) as data:
                    metadata = json.loads(str(data["__metadata__"]))
                contract = ("architecture_version", "observation_version", "value_perspective", "game")
                if metadata["training_step"] != step or any(
                        metadata["manifest"].get(key) != manifest.get(key) for key in contract):
                    raise ValueError(f"Checkpoint C{step} metadata does not match the run")
            model = InferenceModel(load_parameters(path))
            value, policy = model.inference(state.observation_tensor(), state.legal_actions_mask())
            value, policy = np.asarray(value), np.asarray(policy)
            if (value.size != 1 or policy.shape != (game.num_distinct_actions(),)
                    or not np.isfinite(value).all() or not np.isfinite(policy).all()
                    or not np.isclose(policy.sum(), 1) or (policy < 0).any()):
                raise ValueError(f"Invalid inference output for C{step}")
            loaded[step] = (path, model, source_hash(path))
        return loaded[step]
    selected, errors = [], []
    for step in requested:
        actual = step
        try:
            path, model, digest = load(step)
        except Exception as exc:
            if step != 1 or not config.fallback_first:
                raise
            errors.append({"learner_round": step, "error": repr(exc)})
            for actual in available_steps(run):
                if actual == 1:
                    continue
                try:
                    path, model, digest = load(actual)
                    break
                except Exception as candidate_error:
                    errors.append({"learner_round": actual, "error": repr(candidate_error)})
            else:
                raise ValueError("C1 is unavailable and no valid trained fallback exists") from exc
            log("checkpoint_substitution", requested_round=1, learner_round=actual, rejected=errors)
        item = next((item for item in selected if item["learner_round"] == actual), None)
        if item is not None:
            item["requested_rounds"].append(step)
        else:
            selected.append({"learner_round": actual, "requested_rounds": [step],
                             "checkpoint_source": str(path), "checkpoint_sha256": digest, "model": model})
        log("checkpoint_validated", requested_round=step, learner_round=actual, checkpoint_source=str(path))
    return selected, errors


def run_evaluation(run, config, output=None, *, on_checkpoint=None):
    """Run finite offline matches. The only writes are new evaluation artifacts."""
    run = Path(run).resolve()
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    if (manifest["observation_version"] != adapter.OBSERVATION_VERSION or
            manifest.get("architecture_version") != "ayo_mlp_185863_v1" or
            manifest.get("value_perspective") != "player_to_move"):
        raise ValueError("Incompatible checkpoint architecture/observation/value contract")
    settings = Settings(**manifest["settings"])
    if manifest.get("game") != settings.game_string:
        raise ValueError("Saved game and settings disagree")
    config = replace(config, simulations=config.simulations or settings.max_simulations)
    game = pyspiel.load_game(settings.game_string)
    openings, opening_provenance = None, None
    if config.openings is not None:
        from experiments.checkpoint_strength.opening_dataset import digest, load_dataset
        openings = load_dataset(config.openings, game)  # Once for all checkpoints.
        if config.games > 2 * len(openings.openings):
            raise ValueError("Requested games exceed twice the opening count; repeats are forbidden")
        selected_openings = openings.openings[:config.games // 2]
        opening_provenance = {
            "protocol": "fixed_openings_v1", "benchmark_id": openings.benchmark_id,
            "path": openings.path, "file_sha256": openings.file_sha256,
            "content_sha256": openings.content_sha256, "opening_count": len(selected_openings),
            "available_openings": len(openings.openings), "games_per_opening": 2,
            "selection": "entire_dataset" if len(selected_openings) == len(openings.openings) else "prefix_smoke_test",
            "selected_openings_sha256": digest([{"opening_id": o.opening_id, "actions": o.actions,
                                                 "state_sha256": o.state_sha256} for o in selected_openings])}
    from experiments.agent_benchmark.agents import make_agent
    make_agent(game, config.seed, config.opponent)
    output = Path(output).resolve() if output is not None else None
    paths = ({"results": output, "games": output.with_suffix(".games.jsonl"),
              "events": output.with_suffix(".events.jsonl"),
              "manifest": output.with_suffix(".manifest.json")} if output else {})
    if paths and (len(set(paths.values())) != 4 or any(p.is_relative_to(run) for p in paths.values())):
        raise ValueError("Use a results .jsonl path outside the source training run")
    for path in paths.values():
        if path.exists():
            raise FileExistsError(path)
    started = time.perf_counter()
    progress = {"phase": "preflight", "learner_round": None, "completed_games": 0}
    lock, stop = threading.RLock(), threading.Event()
    experiment = {"experiment": "alphazero_learning_progression", "schema_version": 1,
                  "status": "running", "run": str(run), "config": asdict(config),
                  "game": settings.game_string, "uct_c": settings.uct_c,
                  "model_contract": {key: manifest[key] for key in (
                      "architecture_version", "observation_version", "value_perspective")},
                  "run_manifest_sha256": source_hash(run / "manifest.json"),
                  "starting_position": "standard_initial_state", "seed_schedule": "seed + game_id // 2; paired seats; same seeds across checkpoints",
                  "evaluation_policy": {"dirichlet_noise": None, "temperature_sampling": False,
                                        "action_selection": "maximum visit count, lowest action ID breaks ties"},
                  "confidence_intervals": None,
                  "confidence_interval_note": "The shared match runner has no confidence interval estimator. Raw paired-seed games are retained for later uncertainty analysis.",
                  "packages": {name: version(name) for name in ("open_spiel", "numpy", "jax", "flax", "orbax-checkpoint")},
                  "artifacts": {key: str(path) for key, path in paths.items()},
                  "completed_checkpoints": 0, "completed_games": 0}
    if openings is not None:
        experiment.update(starting_position="fixed_openings_v1", opening_dataset=opening_provenance,
                          seed_schedule="seed + opening index; same opening/order/paired seats/seeds across checkpoints",
                          confidence_interval_note="No intervals computed. The two seat games form an opening pair; use opening pairs for any later uncertainty analysis.")
    root = Path(__file__).resolve().parents[2]
    sources = ["Algorithms/H_star_ayo.py", "Algorithms/eoh_ayo.py", "Algorithms/alpha_zero/game.py",
               "Algorithms/alpha_zero/inference.py", "Algorithms/alpha_zero/evaluation.py",
               "Algorithms/alpha_zero/runtime.py", "Algorithms/alpha_zero/config.py",
               "experiments/agent_benchmark/agents.py", "experiments/agent_benchmark/random_vs_greedy_hstar.py",
               "experiments/alpha_zero/offline_evaluate.py", "Model/ayo_olopon/ayo_olopon.py",
               "open_spiel/open_spiel/python/algorithms/mcts.py",
               "open_spiel/open_spiel/python/algorithms/alpha_zero/evaluator.py",
               "open_spiel/open_spiel/python/algorithms/alpha_zero/model_linen.py"]
    if openings is not None:
        sources.append("experiments/checkpoint_strength/opening_dataset.py")
        sources.append("experiments/checkpoint_strength/metrics.py")
    experiment["code_sha256"] = {name: source_hash(root / name) for name in sources}
    with ExitStack() as stack:
        streams = {}
        if output:
            output.parent.mkdir(parents=True, exist_ok=True)
            for name, path in paths.items():
                if name == "manifest":
                    path.open("x", encoding="utf-8").close()
                else:
                    streams[name] = stack.enter_context(path.open("x", encoding="utf-8"))
        def save_manifest():
            if output:
                temporary = paths["manifest"].with_suffix(".json.tmp")
                temporary.write_text(json.dumps(experiment, indent=2) + "\n", encoding="utf-8")
                temporary.replace(paths["manifest"])
        def log(event, **values):
            with lock:
                row = {"event": event, "time_utc": datetime.now(timezone.utc).isoformat(),
                       "elapsed_seconds": round(time.perf_counter() - started, 3), **values}
                line = json.dumps(row)
                print(line, flush=True)
                if "events" in streams:
                    streams["events"].write(line + "\n")
                    streams["events"].flush()
        def heartbeat():
            while not stop.wait(config.heartbeat_seconds):
                with lock:
                    log("heartbeat", **progress)
        thread = threading.Thread(target=heartbeat, daemon=True)
        thread.start()
        try:
            log("experiment_started", run=str(run), **asdict(config), artifacts=experiment["artifacts"])
            save_manifest()
            selected, errors = prepare_checkpoints(run, config, game, manifest, log)
            experiment["checkpoints"] = [{k: v for k, v in item.items() if k != "model"} for item in selected]
            experiment["substitution_rejections"] = errors
            experiment["total_games"] = len(selected) * config.games
            save_manifest()
            for index, item in enumerate(selected, 1):
                step = item["learner_round"]
                with lock:
                    progress.update(phase="matches", learner_round=step, completed_games=0,
                                    checkpoint_index=index, total_checkpoints=len(selected), games=config.games)
                log("checkpoint_started", **progress, simulations=config.simulations, opponent=config.opponent)
                counts = {"win": 0, "draw": 0, "loss": 0}
                checkpoint_started = time.perf_counter()
                def record_game(record):
                    row = {"learner_round": step, "requested_rounds": item["requested_rounds"], **record}
                    if "games" in streams:
                        streams["games"].write(json.dumps(row) + "\n")
                        streams["games"].flush()
                    outcome = record["policy_outcome"]["ALPHAZERO"]
                    counts[outcome] += 1
                    with lock:
                        progress["completed_games"] += 1
                        done = progress["completed_games"]
                        experiment["completed_games"] += 1
                    remaining = experiment["total_games"] - experiment["completed_games"]
                    log("game_completed", learner_round=step, game_id=record["game_id"], seed=record["seed"],
                        seat="P0" if record["player_0_policy"] == "ALPHAZERO" else "P1",
                        outcome=outcome, game_length=record["game_length"], game_seconds=record["elapsed_seconds"],
                        **({"opening_id": record["opening_id"], "opening_plies": record["opening_plies"]}
                           if openings is not None else {}),
                        checkpoint_games=f"{done}/{config.games}", total_games=f"{experiment['completed_games']}/{experiment['total_games']}",
                        wins=counts["win"], draws=counts["draw"], losses=counts["loss"],
                        score_rate=(counts["win"] + 0.5 * counts["draw"]) / done,
                        estimated_remaining_seconds=round((time.perf_counter() - started) / experiment["completed_games"] * remaining, 1))
                report = evaluate_model(game, item["model"], settings, step, opponent=config.opponent,
                                        games=config.games, simulations=config.simulations, seed=config.seed,
                                        on_game=record_game, **({"openings": openings} if openings is not None else {}))
                if openings is not None:
                    report["opening_dataset"] = opening_provenance
                    if all(record["opening_plies"] % 2 == 0 for record in report["matches"]):
                        from experiments.checkpoint_strength.metrics import first_player_advantage
                        report["first_player_advantage"] = first_player_advantage(report["matches"])
                if source_hash(item["checkpoint_source"]) != item["checkpoint_sha256"]:
                    raise RuntimeError(f"Source checkpoint C{step} changed during evaluation")
                report.update({k: v for k, v in item.items() if k != "model"})
                report["checkpoint_elapsed_seconds"] = time.perf_counter() - checkpoint_started
                if "results" in streams:
                    streams["results"].write(json.dumps(report) + "\n")
                    streams["results"].flush()
                experiment["completed_checkpoints"] += 1
                save_manifest()
                log("checkpoint_completed", **{k: v for k, v in report.items() if k not in ("matches", "summary")})
                if on_checkpoint is not None:
                    on_checkpoint(report)
            experiment["status"] = "completed"
            experiment["elapsed_seconds"] = time.perf_counter() - started
            save_manifest()
            log("experiment_completed", completed_checkpoints=len(selected), total_games=experiment["completed_games"],
                artifacts=experiment["artifacts"])
            return experiment
        except BaseException as exc:
            experiment.update(status="failed", error=repr(exc), elapsed_seconds=time.perf_counter() - started)
            save_manifest()
            log("experiment_failed", error=repr(exc), **progress)
            raise
        finally:
            stop.set()
            thread.join(timeout=2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--config", type=Path, help="Offline evaluation JSON; CLI options override it")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--checkpoint", type=int, help="Single retained learner round")
    selection.add_argument("--checkpoints", nargs="+", help="Rounds or inclusive START:END[:STRIDE] ranges")
    selection.add_argument("--all-checkpoints", action="store_true")
    parser.add_argument("--games", type=int)
    parser.add_argument("--simulations", type=int, help="Neural budget; default: saved self-play budget")
    parser.add_argument("--opponent")
    parser.add_argument("--opponent-params", help="JSON arguments to the existing agent")
    parser.add_argument("--opponent-config", type=Path, help="JSON agent specification, including optional factory")
    parser.add_argument("--opponent-simulations", type=int, help="Compatibility alias for MCTS params.simulations")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--fallback-first", action=argparse.BooleanOptionalAction, default=None,
                        help="If C1 is missing/invalid, use the earliest valid trained round")
    parser.add_argument("--heartbeat-seconds", type=int)
    parser.add_argument("--output", type=Path, help="New results JSONL; also creates games/events/manifest siblings")
    args = parser.parse_args()
    try:
        if not (args.config or args.all_checkpoints or args.checkpoint is not None or args.checkpoints):
            raise ValueError("Select checkpoints or supply --config")
        config = load_config(args.config) if args.config else EvaluationConfig()
        overrides = {name: getattr(args, name) for name in ("games", "simulations", "seed", "fallback_first", "heartbeat_seconds")
                     if getattr(args, name) is not None}
        if args.all_checkpoints:
            overrides["checkpoints"] = None
        elif args.checkpoint is not None or args.checkpoints:
            overrides["checkpoints"] = parse_steps([str(args.checkpoint)] if args.checkpoint is not None else args.checkpoints)
        opponent = json.loads(json.dumps(config.opponent))
        if args.opponent:
            opponent = {"name": args.opponent, "params": {}}
        if args.opponent_params:
            opponent["params"] = json.loads(args.opponent_params)
        if args.opponent_config:
            opponent = json.loads(args.opponent_config.read_text(encoding="utf-8"))
        if args.opponent_simulations is not None:
            opponent.setdefault("params", {})["simulations"] = args.opponent_simulations
        run_evaluation(args.run, replace(config, opponent=opponent, **overrides), args.output)
    except (ValueError, OSError, KeyError, TypeError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
