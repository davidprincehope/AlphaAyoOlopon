"""Investigate opening depths, then freeze a benchmark from the saved candidates.

Run with python -m experiments.checkpoint_strength.generate_openings.
Both stages refuse to overwrite existing artifacts.
"""
import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import random

import numpy as np
import pyspiel
from Algorithms import H_star_ayo
from Model.ayo_olopon import ayo_olopon  # noqa: F401
from experiments.checkpoint_strength.opening_dataset import digest, position_digest, rules_contract, snapshot


DEPTHS = (2, 4, 6, 8, 10)
DEFAULT_SEED = 20261004


def generate_candidates(game, count=5000, seed=DEFAULT_SEED, depths=DEPTHS):
    if type(count) is not int or count < 1 or type(seed) is not int or seed < 0:
        raise ValueError("count must be positive and seed nonnegative")
    if not depths or len(set(depths)) != len(depths) or any(type(d) is not int or d < 1 for d in depths):
        raise ValueError("depths must be distinct positive integers")
    records = []
    for depth in depths:
        rng = random.Random(seed + depth)
        for index in range(count):
            state, actions = game.new_initial_state(), []
            while len(actions) < depth and not state.is_terminal():
                action = int(rng.choice(state.legal_actions()))
                state.apply_action(action)
                actions.append(action)
            saved = snapshot(state)
            records.append({"candidate_id": f"d{depth:02d}-{index:06d}", "target_plies": depth,
                            "actions": actions, "plies": len(actions), "state": saved,
                            "state_sha256": digest(saved), "position_sha256": position_digest(saved),
                            "hstar_p0": float(H_star_ayo.evaluate_state(state, 0))})
    return records


def distribution(values):
    if not values:
        return None
    return {"min": float(min(values)), "p10": float(np.percentile(values, 10)),
            "median": float(np.median(values)), "p90": float(np.percentile(values, 90)),
            "max": float(max(values)), "mean": float(np.mean(values))}


def characterize(records):
    result = {}
    for depth in sorted({r["target_plies"] for r in records}):
        trials = [r for r in records if r["target_plies"] == depth]
        live = [r for r in trials if r["plies"] == depth and not r["state"]["is_terminal"]]
        unique_positions = len({r["position_sha256"] for r in live})
        states = [r["state"] for r in live]
        differences = [s["captured"][0] - s["captured"][1] for s in states]
        result[str(depth)] = {
            "trials": len(trials), "eligible_nonterminal": len(live),
            "terminal_trials": sum(r["state"]["is_terminal"] for r in trials),
            "terminal_rate": sum(r["state"]["is_terminal"] for r in trials) / len(trials),
            "terminal_by_ply": dict(sorted(Counter(str(r["plies"]) for r in trials if r["state"]["is_terminal"]).items())),
            "unique_visible_positions": unique_positions,
            "unique_full_states": len({r["state_sha256"] for r in live}),
            "unique_action_sequences": len({tuple(r["actions"]) for r in live}),
            "duplicate_position_rate": 1 - unique_positions / len(live) if live else None,
            "legal_actions_histogram": dict(sorted(Counter(str(len(s["legal_actions"])) for s in states).items())),
            "forced_move_rate": sum(len(s["legal_actions"]) == 1 for s in states) / len(states) if states else None,
            "captured_total": distribution([sum(s["captured"]) for s in states]),
            "capture_difference_p0": distribution(differences),
            "absolute_capture_difference": distribution([abs(d) for d in differences]),
            "capture_leader_counts": dict(Counter("P0" if d > 0 else "P1" if d < 0 else "tied" for d in differences)),
            "remaining_seeds": distribution([sum(s["board"]) for s in states]),
            "hstar_p0": distribution([r["hstar_p0"] for r in live]),
        }
    return result


def save_investigation(directory, game, count=5000, seed=DEFAULT_SEED):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    candidate_path, report_path = directory / "candidates.jsonl.gz", directory / "depth_analysis.json"
    if candidate_path.exists() or report_path.exists():
        raise FileExistsError("Investigation artifacts already exist; choose a new directory")
    records = generate_candidates(game, count, seed)
    raw = "".join(json.dumps(r, sort_keys=True, separators=(",", ":")) + "\n" for r in records).encode()
    with candidate_path.open("xb") as handle:
        handle.write(gzip.compress(raw, mtime=0))
    report = {"schema_version": 1, "generation_seed": seed, "trials_per_depth": count,
              "depths": list(DEPTHS), "rules_contract": rules_contract(game),
              "sampling": "independent paths from initial state; uniform legal actions; random.Random(seed + depth)",
              "ply_definition": "one apply_action decision; relay sowing is internal to an action",
              "candidates_file": candidate_path.name,
              "candidates_sha256": hashlib.sha256(candidate_path.read_bytes()).hexdigest(),
              "candidates_content_sha256": hashlib.sha256(raw).hexdigest(),
              "statistics_population": "all trials; position distributions use all eligible nonterminal trials before deduplication",
              "by_depth": characterize(records)}
    with report_path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(report, indent=2) + "\n")
    return report


def freeze_benchmark(directory, output, depth, count=100, benchmark_id="ayo_fixed_v1"):
    directory, output = Path(directory), Path(output)
    if output.exists():
        raise FileExistsError(output)
    if type(count) is not int or count < 1:
        raise ValueError("count must be positive")
    report = json.loads((directory / "depth_analysis.json").read_text(encoding="utf-8"))
    raw = (directory / report["candidates_file"]).read_bytes()
    if hashlib.sha256(raw).hexdigest() != report["candidates_sha256"]:
        raise ValueError("Saved candidates hash mismatch")
    records = [json.loads(line) for line in gzip.decompress(raw).decode().splitlines()]
    selected, seen = [], set()
    for record in records:
        if (record["target_plies"] != depth or record["plies"] != depth or
                record["state"]["is_terminal"] or record["position_sha256"] in seen):
            continue
        seen.add(record["position_sha256"])
        selected.append({"opening_id": f"{benchmark_id}-{len(selected):03d}", **record})
        if len(selected) == count:
            break
    if len(selected) != count:
        raise ValueError(f"Depth {depth} has fewer than {count} distinct nonterminal candidates")
    payload = {"schema_version": 1, "benchmark_id": benchmark_id, "opening_count": count,
               "rules_contract": report["rules_contract"], "generation_seed": report["generation_seed"],
               "depth": depth, "selection": "first distinct nonterminal visible positions in saved candidate order; no strength or capture filtering",
               "investigation_sha256": digest(report),
               "candidates_sha256": report["candidates_sha256"],
               "openings": selected}
    payload["content_sha256"] = digest(payload)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(payload, indent=2) + "\n")
    return payload


def freeze_mixed_benchmark(directory, output, depth_counts, benchmark_id="ayo_mixed_v2"):
    """Freeze globally distinct openings at multiple depths from saved trials."""
    directory, output = Path(directory), Path(output)
    if output.exists():
        raise FileExistsError(output)
    if (not isinstance(depth_counts, dict) or not depth_counts or
            any(type(depth) is not int or depth not in DEPTHS or
                type(count) is not int or count < 1 for depth, count in depth_counts.items())):
        raise ValueError("depth_counts requires supported depths and positive integer counts")
    report = json.loads((directory / "depth_analysis.json").read_text(encoding="utf-8"))
    compressed = (directory / report["candidates_file"]).read_bytes()
    if hashlib.sha256(compressed).hexdigest() != report["candidates_sha256"]:
        raise ValueError("Saved candidates hash mismatch")
    raw = gzip.decompress(compressed)
    if hashlib.sha256(raw).hexdigest() != report["candidates_content_sha256"]:
        raise ValueError("Saved candidate content hash mismatch")
    records = [json.loads(line) for line in raw.splitlines()]
    buckets, seen = {}, set()
    for depth, count in sorted(depth_counts.items()):
        bucket = []
        for record in records:
            if (record["target_plies"] != depth or record["plies"] != depth or
                    record["state"]["is_terminal"] or record["position_sha256"] in seen):
                continue
            seen.add(record["position_sha256"])
            bucket.append(record)
            if len(bucket) == count:
                break
        if len(bucket) != count:
            raise ValueError(f"Depth {depth} has fewer than {count} globally distinct nonterminal candidates")
        buckets[depth] = bucket
    # Interleave depths so small smoke-test prefixes also contain varied depths.
    selected = []
    for index in range(max(depth_counts.values())):
        for depth, bucket in sorted(buckets.items()):
            if index < len(bucket):
                selected.append({"opening_id": f"{benchmark_id}-{len(selected):03d}", **bucket[index]})
    payload = {"schema_version": 1, "benchmark_id": benchmark_id,
               "opening_count": len(selected), "depths": sorted(depth_counts),
               "depth_counts": {str(d): n for d, n in sorted(depth_counts.items())},
               "rules_contract": report["rules_contract"], "generation_seed": report["generation_seed"],
               "selection": "first globally distinct nonterminal visible positions per depth in saved candidate order; depth buckets interleaved; no strength or capture filtering",
               "investigation_sha256": digest(report), "candidates_sha256": report["candidates_sha256"],
               "statistics": characterize(selected), "openings": selected}
    payload["content_sha256"] = digest(payload)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(payload, indent=2) + "\n")
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    investigate = commands.add_parser("investigate")
    investigate.add_argument("--directory", required=True, type=Path)
    investigate.add_argument("--count", type=int, default=5000, help="Trials at each of 2, 4, 6, 8, 10 plies")
    investigate.add_argument("--seed", type=int, default=DEFAULT_SEED)
    investigate.add_argument("--max-actions", type=int, default=1000)
    freeze = commands.add_parser("freeze")
    freeze.add_argument("--directory", required=True, type=Path)
    freeze.add_argument("--output", required=True, type=Path)
    freeze.add_argument("--depth", required=True, type=int, choices=DEPTHS)
    freeze.add_argument("--count", type=int, default=100)
    freeze.add_argument("--benchmark-id", default="ayo_fixed_v1")
    mixed = commands.add_parser("freeze-mixed")
    mixed.add_argument("--directory", required=True, type=Path)
    mixed.add_argument("--output", required=True, type=Path)
    mixed.add_argument("--allocation", required=True, nargs="+", help="DEPTH:COUNT pairs, such as 2:20 4:45 6:45 8:45 10:45")
    mixed.add_argument("--benchmark-id", default="ayo_mixed_v2")
    args = parser.parse_args()
    if args.command == "investigate":
        game = pyspiel.load_game("ayo_olopon", {"max_game_length": args.max_actions})
        result = save_investigation(args.directory, game, args.count, args.seed)
        print(json.dumps(result["by_depth"], indent=2))
    elif args.command == "freeze":
        result = freeze_benchmark(args.directory, args.output, args.depth, args.count, args.benchmark_id)
        print(json.dumps({k: v for k, v in result.items() if k != "openings"}, indent=2))
    else:
        pairs = [tuple(int(x) for x in pair.split(":")) for pair in args.allocation]
        if any(len(pair) != 2 for pair in pairs) or len({pair[0] for pair in pairs}) != len(pairs):
            parser.error("Use distinct DEPTH:COUNT pairs")
        result = freeze_mixed_benchmark(args.directory, args.output, dict(pairs), args.benchmark_id)
        print(json.dumps({k: v for k, v in result.items() if k not in ("openings", "statistics")}, indent=2))


if __name__ == "__main__":
    main()
