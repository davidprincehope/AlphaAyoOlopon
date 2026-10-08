"""Run the vendored OpenSpiel learner: python -m experiments.alpha_zero.train."""

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import json
import multiprocessing
import shutil
from pathlib import Path

from Algorithms.alpha_zero.config import Settings, load_settings
from Algorithms.alpha_zero import checkpointing
from Algorithms.alpha_zero.runtime import ROOT, guard_worker_failures, provenance, use_repository_open_spiel

# Top-level registration is intentional: Windows/spawn workers need to register
# the custom game BEFORE multiprocessing unpickles the game argument.
use_repository_open_spiel()
from Algorithms.alpha_zero import game as ayo_game  # noqa: E402
import pyspiel  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resume", type=Path, help="Run directory containing complete training snapshots")
    parser.add_argument("--resume-step", type=int, help="Select a saved learner step; older snapshots require a new --output")
    parser.add_argument("--max-steps", type=int, help="Total target learner step, not additional rounds; 0 means unlimited")
    parser.add_argument("--evaluation-games", type=int, help="Per-round games; 0 skips matches but keeps inference exports")
    parser.add_argument("--actors", type=int, help="Self-play processes for this session, including on resume")
    parser.add_argument("--dry-run", action="store_true", help="Validate settings/game without importing JAX or training")
    args = parser.parse_args()
    if args.resume_step is not None and not args.resume:
        parser.error("--resume-step requires --resume")
    snapshot = None
    completed_step = 0
    if args.resume:
        try:
            snapshot = checkpointing.find_snapshot(args.resume, args.resume_step)
            metadata, saved = checkpointing.validate_snapshot(snapshot)
            settings = Settings(**metadata["manifest"]["settings"])
            completed_step = saved["completed_step"]
            if args.config:
                requested = asdict(load_settings(args.config))
                expected = asdict(settings)
                requested.pop("max_steps")
                expected.pop("max_steps")
                if requested != expected:
                    parser.error("Resume configuration differs from snapshot; only --max-steps may change")
        except (ValueError, OSError, KeyError) as exc:
            parser.error(str(exc))
    else:
        settings = load_settings(args.config or ROOT / "experiments/alpha_zero/configs/smoke.json")
    if args.max_steps is not None:
        settings = replace(settings, max_steps=args.max_steps)
    if args.evaluation_games is not None:
        try:
            settings = replace(settings, evaluation_games=args.evaluation_games)
        except ValueError as exc:
            parser.error(str(exc))
    if args.actors is not None:
        try:
            settings = replace(settings, actors=args.actors)
        except ValueError as exc:
            parser.error(str(exc))
    if snapshot and settings.max_steps != 0 and settings.max_steps <= completed_step:
        parser.error(f"Target already reached: snapshot step {completed_step}; provide --max-steps greater than this")
    game = pyspiel.load_game(settings.game_string)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = (args.output or args.resume or ROOT / "runs/alpha_zero" / timestamp).resolve()
    same_run = bool(args.resume and output == args.resume.resolve())
    if same_run and snapshot != checkpointing.find_snapshot(args.resume):
        parser.error("Resuming an older snapshot requires a new --output directory")
    manifest = {
        "settings": asdict(settings), "game": settings.game_string,
        "observation_version": ayo_game.OBSERVATION_VERSION,
        "architecture_version": "ayo_mlp_185863_v1",
        "observation_shape": game.observation_tensor_shape(),
        "num_actions": game.num_distinct_actions(), "value_perspective": "player_to_move",
        "output": str(output), **provenance(),
    }
    if snapshot:
        if any(manifest[key] != metadata["manifest"][key] for key in checkpointing.CONTRACT):
            parser.error("Snapshot architecture, observation encoding, or game rules are incompatible")
        manifest["resume_source"] = str(snapshot)
    if args.dry_run:
        print(json.dumps(manifest, indent=2))
        return
    try:
        from open_spiel.python.algorithms.alpha_zero import alpha_zero
        from open_spiel.python.utils import spawn
    except ImportError as exc:
        raise SystemExit("Training dependencies missing. Install requirements-alpha-zero.txt. " + str(exc)) from exc
    guard_worker_failures()
    output.mkdir(parents=True, exist_ok=same_run)
    session = output / "sessions" / timestamp
    session.mkdir(parents=True, exist_ok=False)
    if same_run:
        checkpointing.prepare_history(output, session, completed_step)
    elif snapshot:
        evaluations = args.resume / "evaluation.jsonl"
        if evaluations.exists():
            shutil.copy2(evaluations, output / "evaluation.jsonl")
        history = args.resume / "learner.jsonl"
        if history.exists():
            shutil.copy2(history, output / "learner.jsonl")
            checkpointing.prepare_history(output, session, completed_step)
    if not same_run:
        checkpointing.write_json(output / "manifest.json", manifest)
    checkpointing.write_json(session / "session.json", {
        "session_id": timestamp, "source_snapshot": str(snapshot) if snapshot else None,
        "start_step": completed_step + 1, "target_step": settings.max_steps,
        "settings": asdict(settings), "worker_policy": "restart_fresh_games"})
    config = alpha_zero.Config(**settings.upstream_kwargs(output)).replace(
        resume_snapshot=str(snapshot) if snapshot else "", session_id=timestamp,
        log_path=str(session))
    print(f"{'Resuming' if snapshot else 'Starting fresh training'} in {output}; next step {completed_step + 1}", flush=True)
    with spawn.main_handler():
        alpha_zero.alpha_zero(config)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
