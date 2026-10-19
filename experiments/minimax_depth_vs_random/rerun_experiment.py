"""Corrected, paired minimax rerun; preserve the historical runner and results."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import multiprocessing
import os
from pathlib import Path
import random
import sys
import time

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from Algorithms.alpha_zero.runtime import use_repository_open_spiel
import open_spiel  # Resolve the outer checkout before repairing its package path.
use_repository_open_spiel()
import numpy as np
import pyspiel
from Algorithms import ayo_minimax
from Model.ayo_olopon import ayo_olopon  # noqa: F401 - registers the game

SOURCES = (
    "Algorithms/H_star_ayo.py", "Algorithms/ayo_minimax.py",
    "Algorithms/alpha_zero/runtime.py", "Model/ayo_olopon/ayo_olopon.py",
    "open_spiel/open_spiel/python/algorithms/minimax.py",
    "experiments/minimax_depth_vs_random/rerun_experiment.py",
)
WORKER = None


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def finish_record(state, seat):
    """Censor internal/external limits, including engine-adjudicated terminals."""
    truncated = bool(state.truncated) or not state.is_terminal()
    if truncated:
        return {"truncated": True, "termination_reason": "action_limit",
                "returns_by_player": None, "minimax_outcome": "unresolved"}
    returns = list(state.returns())
    value = returns[seat]
    return {"truncated": False,
            "termination_reason": "terminal_draw" if returns == [0.0, 0.0] else "terminal_win",
            "returns_by_player": returns,
            "minimax_outcome": "win" if value > 0 else "loss" if value < 0 else "draw"}


def play_game(game, bot, depth, game_id, seed, max_actions):
    state, seat, rng = game.new_initial_state(), game_id % 2, random.Random(seed)
    started, search_seconds, actions = time.perf_counter(), 0.0, []
    while not state.is_terminal() and len(actions) < max_actions:
        if state.current_player() == seat:
            decision_started = time.perf_counter()
            action = int(bot.step(state))
            search_seconds += time.perf_counter() - decision_started
        else:
            action = int(rng.choice(list(state.legal_actions())))
        if action not in state.legal_actions():
            raise ValueError(f"Illegal action {action}")
        state.apply_action(action)
        actions.append(action)
    return {
        "depth": depth, "game_id": game_id, "pair_id": game_id // 2,
        "seed": seed, "minimax_seat": seat,
        **finish_record(state, seat), "game_length": len(actions), "actions": actions,
        "final_captured": list(state.captured), "final_board": list(state.board),
        "engine_terminal": state.is_terminal(), "engine_returns_by_player": list(state.returns()),
        "minimax_decision_seconds": search_seconds, "elapsed_seconds": time.perf_counter() - started,
    }


def initialize(max_actions):
    global WORKER
    WORKER = pyspiel.load_game("ayo_olopon", {"max_game_length": max_actions})


def play_pair(depth, pair_id, seed, max_actions):
    bot = ayo_minimax.make_bot(WORKER, maximum_depth=depth)
    return [play_game(WORKER, bot, depth, 2 * pair_id + seat, seed + pair_id, max_actions)
            for seat in (0, 1)]


def verify_record(game, record, depth, game_id, seed, max_actions):
    seat = game_id % 2
    expected = {"depth": depth, "game_id": game_id, "pair_id": game_id // 2,
                "seed": seed, "minimax_seat": seat}
    if any(record.get(k) != v for k, v in expected.items()):
        raise ValueError("Saved depth, game ID, seed or seat protocol mismatch")
    if not 0 < len(record["actions"]) <= max_actions:
        raise ValueError("Invalid action trace length")
    state, rng = game.new_initial_state(), random.Random(seed)
    for action in record["actions"]:
        if state.is_terminal() or action not in state.legal_actions():
            raise ValueError("Illegal move or move after termination")
        if state.current_player() != seat and action != rng.choice(list(state.legal_actions())):
            raise ValueError("Random action does not match the recorded seed")
        state.apply_action(action)
    expected.update(finish_record(state, seat), game_length=len(record["actions"]),
                    final_captured=list(state.captured), final_board=list(state.board),
                    engine_terminal=state.is_terminal(), engine_returns_by_player=list(state.returns()))
    if any(record.get(k) != v for k, v in expected.items()):
        raise ValueError("Replay does not match the recorded terminal result")
    if not state.is_terminal() and len(record["actions"]) != max_actions:
        raise ValueError("Unfinished trace did not reach the action limit")


def summarize(records):
    counts, games = Counter(r["minimax_outcome"] for r in records), len(records)
    wins, draws, losses, unresolved = (counts[k] for k in ("win", "draw", "loss", "unresolved"))
    completed, points = games - unresolved, wins + 0.5 * draws
    return {
        "games": games, "completed_games": completed, "wins": wins, "draws": draws,
        "losses": losses, "unresolved_games": unresolved,
        "win_rate": wins / games if games and not unresolved else None,
        "score_rate": points / games if games and not unresolved else None,
        "completed_only_score_rate": points / completed if completed else None,
        "score_rate_bounds": [points / games, (points + unresolved) / games] if games else None,
        "seat_counts": {str(s): sum(r["minimax_seat"] == s for r in records) for s in (0, 1)},
        "termination_reasons": dict(Counter(r["termination_reason"] for r in records)),
        "outcomes_by_termination": {reason: dict(Counter(r["minimax_outcome"] for r in records
            if r["termination_reason"] == reason)) for reason in sorted({r["termination_reason"] for r in records})},
        "average_game_length": sum(r["game_length"] for r in records) / games if games else None,
        "maximum_game_length": max((r["game_length"] for r in records), default=None),
        "total_minimax_decision_seconds": sum(r["minimax_decision_seconds"] for r in records),
        "total_elapsed_seconds": sum(r["elapsed_seconds"] for r in records),
    }


def pair_scores(records):
    ordered = sorted(records, key=lambda r: r["game_id"])
    if len(ordered) % 2 or [r["game_id"] for r in ordered] != list(range(len(ordered))):
        raise ValueError("Statistics require complete, unique, swapped-seat pairs")
    if any(r["truncated"] for r in ordered):
        return None
    return np.array([{"win": 1.0, "draw": 0.5, "loss": 0.0}[r["minimax_outcome"]]
                     for r in ordered]).reshape(-1, 2).mean(axis=1)


def bootstrap_interval(values, seed, resamples=10000):
    rng, samples = np.random.default_rng(seed), []
    for start in range(0, resamples, 100):
        indices = rng.integers(0, len(values), size=(min(100, resamples - start), len(values)))
        samples.extend(values[indices].mean(axis=1))
    return {"method": "percentile bootstrap of whole seed/seat pairs", "confidence": 0.95,
            "resamples": resamples, "pairs": len(values), "seed": seed,
            "lower": float(np.quantile(samples, 0.025)), "upper": float(np.quantile(samples, 0.975))}


def build_contract(games, depths, seed, max_actions):
    return {
        "schema_version": 2, "games_per_depth": games, "depths": list(depths),
        "master_seed": seed, "max_actions": max_actions,
        "game": f"ayo_olopon(max_game_length={max_actions})", "starting_position": "standard_initial_state",
        "implementation": "Python/OpenSpiel alpha-beta", "depth_unit": "complete player actions (plies)",
        "evaluator": "Algorithms/H_star_ayo.py:evaluate_state; clip(raw H*, -10, 10) / 11",
        "nonterminal_value_bounds": [-10 / 11, 10 / 11], "terminal_utilities": [-1, 0, 1],
        "tie_break": "first action in OpenSpiel legal-action order",
        "seed_protocol": "master_seed + pair_id; same seed for both seats and every depth; Python random.Random.choice",
        "limit_protocol": "search-visible horizon; internal/external limits unresolved and excluded from W/D/L; full-sample score bounds reported",
        "code_sha256": {name: sha256(ROOT / name) for name in SOURCES},
        "runtime": {"python": sys.version, "numpy": np.__version__,
                    "open_spiel": importlib.metadata.version("open_spiel"), "pyspiel_sha256": sha256(pyspiel.__file__)},
    }


def run(games, depths, seed, max_actions, output_dir, workers=3, resume=False):
    if games < 2 or games % 2 or not depths or any(d < 1 for d in depths) or len(set(depths)) != len(depths):
        raise ValueError("Use positive unique depths and a positive even game count")
    if max_actions < 1 or workers < 1:
        raise ValueError("Action horizon and worker count must be positive")
    output_dir = Path(output_dir).resolve()
    contract = build_contract(games, depths, seed, max_actions)
    manifest_path = output_dir / "manifest.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    if manifest_path.exists():
        if not resume:
            raise FileExistsError("Evidence exists; use --resume or a fresh output directory")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["contract"] != contract:
            raise ValueError("Resume requires unchanged code, runtime and protocol")
    else:
        if any(output_dir.iterdir()):
            raise FileExistsError("Output directory contains unrelated files")
        manifest = {"experiment": "corrected_ayo_minimax_depth_vs_random", "contract": contract,
                    "started_at_utc": datetime.now(timezone.utc).isoformat()}
        for name in SOURCES:
            path = output_dir / "source_snapshot" / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes((ROOT / name).read_bytes())
    manifest.update(status="running", workers=workers)
    write_json(manifest_path, manifest)
    game, all_records = pyspiel.load_game("ayo_olopon", {"max_game_length": max_actions}), {}
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"),
                             initializer=initialize, initargs=(max_actions,)) as pool:
        for depth in depths:
            log_path, records = output_dir / f"depth_{depth}.games.jsonl", {}
            if log_path.exists():
                for line in log_path.read_text(encoding="utf-8").splitlines():
                    record = json.loads(line)
                    game_id = record["game_id"]
                    if game_id in records or not 0 <= game_id < games:
                        raise ValueError("Duplicate or out-of-range saved game ID")
                    verify_record(game, record, depth, game_id, seed + game_id // 2, max_actions)
                    records[game_id] = record
            for pair_id in range(games // 2):
                if (2 * pair_id in records) != (2 * pair_id + 1 in records):
                    raise ValueError("Resume log contains an incomplete seat pair")
            print(f"Starting depth {depth}: {len(records)}/{games} games verified", flush=True)
            pending = [i for i in range(games // 2) if 2 * i not in records]
            futures = {pool.submit(play_pair, depth, i, seed, max_actions): i for i in pending}
            with log_path.open("a", encoding="utf-8", newline="\n") as stream:
                for future in as_completed(futures):
                    pair_id, pair = futures[future], future.result()
                    for seat, record in enumerate(pair):
                        verify_record(game, record, depth, 2 * pair_id + seat, seed + pair_id, max_actions)
                    stream.write("".join(json.dumps(r, sort_keys=True) + "\n" for r in pair))
                    stream.flush()
                    records.update({r["game_id"]: r for r in pair})
                    if len(records) % 100 == 0 or len(records) == games:
                        progress = summarize(list(records.values()))
                        write_json(output_dir / f"depth_{depth}.progress.json", progress)
                        print(json.dumps({"depth": depth, **{k: progress[k] for k in
                            ("games", "wins", "draws", "losses", "unresolved_games", "score_rate")}}), flush=True)
            ordered = [records[i] for i in range(games)]
            scores = pair_scores(ordered)
            result = {"depth": depth, **summarize(ordered),
                      "by_seat": {str(s): summarize([r for r in ordered if r["minimax_seat"] == s]) for s in (0, 1)},
                      "verification": {"games_replayed": games, "random_seed_actions_checked": True,
                                       "complete_swapped_seat_pairs": True},
                      "log_file": log_path.name, "log_sha256": sha256(log_path),
                      "score_interval_95": bootstrap_interval(scores, seed) if scores is not None else None}
            write_json(output_dir / f"depth_{depth}.summary.json", result)
            all_records[depth] = ordered
    comparisons = []
    references = list(zip(depths, depths[1:])) + [(depths[0], d) for d in depths[2:]]
    for lower, higher in references:
        lo, hi = pair_scores(all_records[lower]), pair_scores(all_records[higher])
        differences = hi - lo if lo is not None and hi is not None else None
        comparisons.append({"reference_depth": lower, "comparison_depth": higher,
            "score_rate_difference": float(differences.mean()) if differences is not None else None,
            "paired_interval_95": bootstrap_interval(differences, seed) if differences is not None else None})
    if build_contract(games, depths, seed, max_actions) != contract:
        raise ValueError("Experiment sources or runtime changed during the run")
    report = {"experiment": manifest["experiment"], "status": "completed", "contract": contract,
              "results": [json.loads((output_dir / f"depth_{d}.summary.json").read_text(encoding="utf-8")) for d in depths],
              "depth_comparisons": comparisons,
              "inference_scope": "standard start against uniform random; depth is not a runtime budget; intervals are individual, not multiplicity-adjusted"}
    write_json(output_dir / "experiment_summary.json", report)
    manifest.update(status="completed", finished_at_utc=datetime.now(timezone.utc).isoformat(),
                    completed_games=sum(len(rows) for rows in all_records.values()),
                    results_sha256=sha256(output_dir / "experiment_summary.json"))
    write_json(manifest_path, manifest)
    print(f"Complete. Summary: {output_dir / 'experiment_summary.json'}", flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=10000)
    parser.add_argument("--depths", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--seed", type=int, default=20261019)
    parser.add_argument("--max-actions", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runs/minimax_depth_vs_random/corrected-20261019")
    args = parser.parse_args()
    run(args.games, args.depths, args.seed, args.max_actions, args.output_dir, args.workers, args.resume)


if __name__ == "__main__":
    main()
