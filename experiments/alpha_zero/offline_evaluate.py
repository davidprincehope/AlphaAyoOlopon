"""Evaluate retained learner rounds through the shared Ayo agent/match runner."""
import argparse
import json
from pathlib import Path

from Algorithms.alpha_zero.runtime import use_repository_open_spiel
use_repository_open_spiel()
import pyspiel
from Algorithms.alpha_zero import game as adapter
from Algorithms.alpha_zero.config import Settings
from Algorithms.alpha_zero.evaluation import evaluate_model
from Algorithms.alpha_zero.inference import InferenceModel, checkpoint_path, load_parameters


def available_steps(run):
    run = Path(run)
    steps = {int(p.stem.removeprefix("step-"))
             for p in (run / "inference-checkpoints").glob("step-*.npz")}
    steps.update(int(p.name.removeprefix("checkpoint-"))
                 for p in run.glob("checkpoint-*")
                 if p.is_dir() and p.name.removeprefix("checkpoint-").isdigit())
    steps.update(int(p.name.removeprefix("step-"))
                 for p in (run / "training-checkpoints").glob("step-*")
                 if (p / "COMPLETE").exists())
    return sorted(steps)


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
                if start < 0 or end < start or stride < 1:
                    raise ValueError("Invalid checkpoint range")
                steps.update(range(start, end + 1, stride))
            else:
                step = int(part)
                if step < 0:
                    raise ValueError("Select retained learner rounds, not the mutable rolling checkpoint -1")
                steps.add(step)
    if not steps:
        raise ValueError("No checkpoints selected")
    return sorted(steps)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--checkpoint", type=int, help="Single retained learner round")
    selection.add_argument("--checkpoints", nargs="+", help="Rounds or inclusive START:END[:STRIDE] ranges")
    selection.add_argument("--all-checkpoints", action="store_true")
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--simulations", type=int, help="Neural budget; default: saved self-play budget")
    parser.add_argument("--opponent", default="RAND")
    parser.add_argument("--opponent-params", default="{}", help="JSON arguments to the existing agent")
    parser.add_argument("--opponent-config", type=Path, help="JSON agent specification, including optional factory")
    parser.add_argument("--opponent-simulations", type=int, help="Compatibility alias for MCTS params.simulations")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, help="New JSONL report; one result per checkpoint")
    args = parser.parse_args()
    try:
        manifest = json.loads((args.run / "manifest.json").read_text(encoding="utf-8"))
        if (manifest["observation_version"] != adapter.OBSERVATION_VERSION or
                manifest.get("architecture_version") != "ayo_mlp_185863_v1" or
                manifest.get("value_perspective") != "player_to_move"):
            raise ValueError("Incompatible checkpoint architecture/observation/value contract")
        settings = Settings(**manifest["settings"])
        steps = available_steps(args.run) if args.all_checkpoints else parse_steps(
            [str(args.checkpoint)] if args.checkpoint is not None else args.checkpoints)
        if not steps:
            raise ValueError("No retained checkpoints found")
        paths = [checkpoint_path(args.run, step) for step in steps]
        opponent = (json.loads(args.opponent_config.read_text()) if args.opponent_config else
                    {"name": args.opponent, "params": json.loads(args.opponent_params)})
        if args.opponent_simulations is not None:
            opponent.setdefault("params", {})["simulations"] = args.opponent_simulations
        if args.games < 2 or args.games % 2 or args.seed < 0:
            raise ValueError("games must be positive and even; seed must be nonnegative")
        budget = args.simulations if args.simulations is not None else settings.max_simulations
        if budget < 2:
            raise ValueError("simulations must be >= 2")
        game = pyspiel.load_game(settings.game_string)
        from experiments.agent_benchmark.agents import make_agent
        make_agent(game, args.seed, opponent)  # Validate baseline before loading any model.
    except (ValueError, OSError, KeyError, TypeError) as exc:
        parser.error(str(exc))
    stream = None
    try:
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            stream = args.output.open("x", encoding="utf-8")
        for step, path in zip(steps, paths):
            model = InferenceModel(load_parameters(path))
            report = evaluate_model(game, model, settings, step, opponent=opponent,
                                    games=args.games, simulations=budget, seed=args.seed)
            report["checkpoint_source"] = str(path)
            report["run"] = str(args.run.resolve())
            if stream:
                stream.write(json.dumps(report) + "\n")
                stream.flush()
            print(json.dumps({k: v for k, v in report.items() if k not in ("matches", "summary")}))
    finally:
        if stream:
            stream.close()


if __name__ == "__main__":
    main()
