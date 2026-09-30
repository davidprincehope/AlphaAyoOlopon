"""Run the vendored OpenSpiel learner: python -m experiments.alpha_zero.train."""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import multiprocessing
from pathlib import Path

from Algorithms.alpha_zero.config import load_settings
from Algorithms.alpha_zero.runtime import ROOT, guard_worker_failures, provenance, use_repository_open_spiel

# Top-level registration is intentional: Windows/spawn workers need to register
# the custom game BEFORE multiprocessing unpickles the game argument.
use_repository_open_spiel()
from Algorithms.alpha_zero import game as ayo_game  # noqa: E402
import pyspiel  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "experiments/alpha_zero/configs/smoke.json")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dry-run", action="store_true", help="Validate settings/game without importing JAX or training")
    args = parser.parse_args()
    settings = load_settings(args.config)
    game = pyspiel.load_game(settings.game_string)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = (args.output or ROOT / "runs/alpha_zero" / timestamp).resolve()
    manifest = {
        "settings": asdict(settings), "game": settings.game_string,
        "observation_version": ayo_game.OBSERVATION_VERSION,
        "architecture_version": "ayo_mlp_185863_v1",
        "observation_shape": game.observation_tensor_shape(),
        "num_actions": game.num_distinct_actions(), "value_perspective": "player_to_move",
        "output": str(output), **provenance(),
    }
    if args.dry_run:
        print(json.dumps(manifest, indent=2))
        return
    try:
        from open_spiel.python.algorithms.alpha_zero import alpha_zero
        from open_spiel.python.utils import spawn
    except ImportError as exc:
        raise SystemExit("Training dependencies missing. Install requirements-alpha-zero.txt. " + str(exc)) from exc
    guard_worker_failures()
    config = alpha_zero.Config(**settings.upstream_kwargs(output))
    # Never overwrite a previous run; upstream always initializes new weights.
    output.mkdir(parents=True, exist_ok=False)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Starting fresh training in {output}", flush=True)
    with spawn.main_handler():
        alpha_zero.alpha_zero(config)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
