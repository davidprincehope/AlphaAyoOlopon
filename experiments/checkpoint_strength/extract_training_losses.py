"""Extract canonical learner-round losses and cross-check matching session logs."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re


def extract(run, output):
    root = Path(__file__).resolve().parents[2]
    def source_name(path):
        resolved = path.resolve()
        return (resolved.relative_to(root).as_posix() if resolved.is_relative_to(root)
                else f"{run.name}/{resolved.relative_to(run.resolve()).as_posix()}")
    history = run / "learner.jsonl"
    records = [json.loads(line) for line in history.read_text().splitlines() if line.strip()]
    assert [r["step"] for r in records] == list(range(1, 601)), "Expected exactly rounds 1 through 600 in order"
    manifest = json.loads((run / "manifest.json").read_text())
    logs, sources = {}, {source_name(history): hashlib.sha256(history.read_bytes()).hexdigest()}
    for session in sorted({r["session_id"] for r in records}):
        path = run / "sessions" / session / "log-learner.txt"
        sources[source_name(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        step = None
        session_rows = {}
        for line in path.read_text().splitlines():
            match = re.search(r"Step: (\d+)", line)
            if match:
                step = int(match[1])
                session_rows[step] = {}
            match = re.search(r"Collected\s+(\d+) states from\s+(\d+) games", line)
            if match and step is not None:
                session_rows[step].update(states=int(match[1]), games=int(match[2]))
            match = re.search(r"Losses\(total: ([\d.]+), policy: ([\d.]+), value: ([\d.]+), l2: ([\d.]+)\)", line)
            if match and step is not None:
                session_rows[step]["loss"] = dict(zip(("sum", "policy", "value", "l2reg"), map(float, match.groups())))
        logs[session] = session_rows
    rows, previous = [], {"total_states": 0, "total_trajectories": 0, "gradient_updates": 0}
    for line_number, r in enumerate(records, 1):
        loss, log = r["loss"], logs[r["session_id"]][r["step"]]
        assert all(math.isfinite(loss[k]) and loss[k] >= 0 for k in ("policy", "value", "l2reg", "sum"))
        assert math.isclose(loss["sum"], sum(loss[k] for k in ("policy", "value", "l2reg")), abs_tol=2e-6)
        assert all(abs(loss[k] - log["loss"][k]) <= 0.000501 for k in loss), f"Console loss mismatch at {r['step']}"
        states = r["total_states"] - previous["total_states"]
        games = r["total_trajectories"] - previous["total_trajectories"]
        updates = r["gradient_updates"] - previous["gradient_updates"]
        assert states == log["states"] and games == log["games"] == r["game_length"]["num"]
        expected_updates = min(r["total_states"], manifest["settings"]["replay_buffer_size"]) // manifest["settings"]["train_batch_size"]
        assert updates == expected_updates, f"Gradient update count mismatch at {r['step']}"
        rows.append({"learner_round": r["step"], "policy_loss": loss["policy"], "value_loss": loss["value"],
                     "l2_loss": loss["l2reg"], "total_loss": loss["sum"], "games_this_round": games,
                     "positions_this_round": states, "cumulative_games": r["total_trajectories"],
                     "cumulative_positions": r["total_states"], "updates_this_round": updates,
                     "cumulative_updates": r["gradient_updates"], "session_id": r["session_id"],
                     "time_utc": r["time_str"], "source_line": line_number})
        previous = r
    audit = {"rounds": len(rows), "missing_rounds": [], "duplicate_rounds": [],
             "console_loss_matches": len(rows), "console_precision": 3,
             "console_absolute_tolerance": 0.000501, "finite_nonnegative_losses": True,
             "total_loss_components_match": True, "collection_counts_match_console": True,
             "gradient_update_counts_match_settings": True, "sessions": sorted(logs),
             "source_sha256": sources,
             "loss_definition": "Mean minibatch training loss over all optimizer updates in each learner round, sampled from replay. Not per-game or per-position losses.",
             "policy_definition": "Mean softmax cross-entropy against self-play MCTS policy targets.",
             "value_definition": "Mean squared error against final game-outcome targets in the player-to-move perspective.",
             "verification_limit": "Verifies recorded losses against rounded console logs and code definitions; does not recompute historical training minibatches.",
             "first": rows[0], "last": rows[-1]}
    output.mkdir(parents=True, exist_ok=True)
    (output / "loss_data.json").write_text(json.dumps({"rows": rows, "audit": audit}, indent=2) + "\n")
    print(json.dumps({k: v for k, v in audit.items() if k != "source_sha256"}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    extract(args.run, args.output)
