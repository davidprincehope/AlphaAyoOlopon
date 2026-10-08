"""Collect, replay, and verify the four 13-checkpoint progression experiments."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time

import modal
import numpy as np

from experiments.alpha_zero.modal_common import ROOT, validate_name
from experiments.checkpoint_strength.collect_mcts_suite import read, download_case
from experiments.checkpoint_strength.head_to_head import paired_score_interval
from experiments.checkpoint_strength.mcts_progression import result_row, validate_curve, sha
from experiments.alpha_zero import offline_evaluate as offline


def verify_case(folder, result, manifest, plan, case, step):
    policy = f"C{step}"
    row = result_row(case, step, result, manifest, plan)
    records = [json.loads(line) for line in (folder / "results.games.jsonl").read_text(encoding="utf-8").splitlines()]
    counts = Counter(record["policy_outcome"][policy] for record in records)
    for outcome, field in (("win", "wins"), ("draw", "draws"), ("loss", "losses")):
        if counts[outcome] != row[field]:
            raise ValueError("Game outcomes disagree with report")
    signature = []
    for i in range(200):
        a, b = records[2 * i:2 * i + 2]
        if (a["opening_pair_id"] != i or b["opening_pair_id"] != i
                or a["player_0_policy"] != b["player_1_policy"] or a["player_1_policy"] != b["player_0_policy"]
                or any(a[key] != b[key] for key in ("opening_id", "opening_state_sha256", "seed", "opening_actions"))):
            raise ValueError("Opening pair mismatch")
        signature.append([a["opening_id"], a["opening_state_sha256"], a["seed"]])
    interval = paired_score_interval(records, policy, plan["seed"])
    for field in ("score_rate", "lower", "upper"):
        if not np.isclose(interval[field], row["score_interval"][field], atol=1e-12, rtol=0):
            raise ValueError("Recomputed opening-pair interval mismatch")
    game = offline.pyspiel.load_game(manifest["game"])
    actions = 0
    for record in records:
        state = game.new_initial_state()
        for action in record["opening_actions"] + record["continuation_actions"]:
            if state.is_terminal() or action not in state.legal_actions():
                raise ValueError("Illegal replayed action")
            state.apply_action(action)
            actions += 1
        if (not state.is_terminal() or state.returns() != record["returns_by_player"]
                or state.captured != record["final_captured"] or state.move_number() != record["game_length"]
                or state.truncated != record["truncated"]):
            raise ValueError("Replayed outcome disagrees with recorded result")
    verification = {"status": "passed", "games_replayed": 400, "actions_replayed": actions,
                    "outcomes_recomputed": True, "confidence_interval_recomputed": True,
                    "source_hashes_match": True, "opening_signature": signature,
                    "files": {name: sha(folder / name) for name in ("results.json", "results.games.jsonl")}}
    (folder / "verification.json").write_text(json.dumps(verification, indent=2) + "\n", encoding="utf-8")
    return row, verification


def collect(local, launch, plan, volume):
    progress, rows = {}, {case: [] for case in plan["cases"]}
    signature, verified_games = None, 0
    for key in launch["function_calls"]:
        case, component = key.split("/")
        step = int(component[1:])
        content = read(volume, f"/{launch['evaluation_name']}/{key}/results.manifest.json")
        manifest = json.loads(content) if content else None
        progress[key] = {"status": manifest["status"], "completed_games": manifest["completed_games"]} if manifest else {
            "status": "waiting", "completed_games": 0}
        if manifest and manifest["status"] == "failed":
            raise RuntimeError(f"{key}: {manifest.get('error')}")
        if not manifest or manifest["status"] != "completed":
            continue
        folder = local / key
        if not (folder / "results.manifest.json").exists():
            result = download_case(volume, f"/{launch['evaluation_name']}/{key}", folder,
                                   manifest, step, plan["cases"][case][str(step)])
        else:
            result = json.loads((folder / "results.json").read_text(encoding="utf-8"))
        if (folder / "verification.json").exists():
            verified = json.loads((folder / "verification.json").read_text(encoding="utf-8"))
            if any(sha(folder / name) != digest for name, digest in verified["files"].items()):
                raise ValueError(f"Previously verified files changed: {key}")
            row = result_row(case, step, result, manifest, plan)
        else:
            row, verified = verify_case(folder, result, manifest, plan, case, step)
            print(json.dumps({"event": "checkpoint_case_verified", "case": case, "checkpoint": step,
                              "games": 400, "score_rate": row["score_rate"]}), flush=True)
        signature = signature or verified["opening_signature"]
        if signature != verified["opening_signature"]:
            raise ValueError("Progression cases use different opening schedules")
        verified_games += 400
        rows[case].append(row)
    content = read(volume, f"/{launch['evaluation_name']}/progression.manifest.json")
    remote = json.loads(content) if content else None
    if remote and remote["status"] == "failed":
        raise RuntimeError(f"Progression finalization failed: {remote.get('error')}")
    compact = {}
    for case in rows:
        rows[case].sort(key=lambda row: row["learner_round"])
        case_progress = {key: value for key, value in progress.items() if key.startswith(case + "/")}
        compact[case] = {
            "completed_checkpoints": len(rows[case]),
            "completed_games": sum(value["completed_games"] for value in case_progress.values()),
            "active": {key.split("/")[1]: value["completed_games"] for key, value in case_progress.items()
                       if value["status"] == "running"},
        }
    complete = all(len(curve) == 13 for curve in rows.values()) and remote and remote["status"] == "completed"
    snapshot = {"status": "completed" if complete else "running", "completed_games": sum(
        value["completed_games"] for value in progress.values()), "verified_games": verified_games,
        "total_games": 20800, "conditions": compact, "updated_at_utc": datetime.now(timezone.utc).isoformat()}
    (local / "status.json").write_text(json.dumps({**snapshot, "checkpoint_cases": progress}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(snapshot), flush=True)
    if complete:
        for curve in rows.values():
            validate_curve(curve)
        for name in ("progression.manifest.json", "progression.results.json"):
            content = read(volume, f"/{launch['evaluation_name']}/{name}")
            if content is None:
                raise RuntimeError(f"Missing completed progression artifact: {name}")
            (local / name).write_bytes(content)
        expected = json.loads((local / "progression.results.json").read_text(encoding="utf-8"))
        if expected != rows:
            raise ValueError("Verified local progression disagrees with remote collector")
        (local / "verification.json").write_text(json.dumps({
            "status": "passed", "games_replayed": 20800, "checkpoint_cases": 52,
            "same_opening_pairs_and_seeds": True, "all_scores_and_intervals_recomputed": True,
            "source_checkpoints_read_only": True,
            "files": {str(path.relative_to(local)): sha(path) for path in local.glob("*/c*/results*.json*")},
        }, indent=2) + "\n", encoding="utf-8")
    return bool(complete)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-name", required=True)
    parser.add_argument("--watch", action="store_true")
    args = parser.parse_args()
    local = ROOT / "runs/checkpoint_strength" / validate_name(args.evaluation_name)
    launch = json.loads((local / "launch.json").read_text(encoding="utf-8"))
    plan = json.loads((local / "experiment.plan.json").read_text(encoding="utf-8"))
    if os.environ.get("MODAL_PROFILE") != launch["profile"]:
        raise ValueError("Select the launch profile explicitly before collecting")
    volume = modal.Volume.from_name(launch["results_volume"])
    while True:
        if collect(local, launch, plan, volume) or not args.watch:
            return
        time.sleep(30)


if __name__ == "__main__":
    main()
