"""Collect and locally replay each case in the sequential le4capture suite."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time

import modal
from experiments.checkpoint_strength.le4capture_suite import (
    ROOT, DATASET_HASH, case_files, case_policies, save_json, sha, validate_case, validate_plan,
)


def collect(local, launch, plan, volume):
    remote = "/" + launch["evaluation_name"]
    try:
        manifest_bytes = b"".join(volume.read_file(remote + "/suite.manifest.json"))
    except (FileNotFoundError, modal.exception.NotFoundError):
        return {"status": "initializing", "verified_games": 0, "total_games": plan["total_games"]}
    manifest = json.loads(manifest_bytes)
    rows = []
    complete_ids = set(manifest["completed_cases"])
    for case in plan["cases"]:
        if case["id"] not in complete_ids:
            continue
        folder = local / case["id"]
        verification = folder / "local-verification.json"
        if verification.exists():
            checked = json.loads(verification.read_text())
            if any(sha(folder / name) != expected for name, expected in checked["files"].items()):
                raise ValueError("Previously verified artifact changed: " + case["id"])
            row = json.loads((folder / "verified-summary.json").read_text())
        else:
            folder.mkdir(parents=True, exist_ok=True)
            checked_bytes = b"".join(volume.read_file(f"{remote}/{case['id']}/verification.json"))
            checked_remote = json.loads(checked_bytes)
            for name in case_files(case):
                content = b"".join(volume.read_file(f"{remote}/{case['id']}/{name}"))
                destination = folder / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(content)
                if sha(destination) != checked_remote["files"][name]:
                    raise ValueError("Downloaded artifact differs: " + name)
            (folder / "verification.json").write_bytes(checked_bytes)
            row, checked = validate_case(folder, case, plan, source_root=local / "source-snapshot")
            save_json(verification, checked)
            save_json(folder / "verified-summary.json", row)
            print(json.dumps({"event": "case_locally_verified", "case_id": case["id"], "policies": row["policies"]}), flush=True)
        rows.append(row)
    active = manifest.get("active_case")
    active_progress = 0
    if active and active not in complete_ids:
        case = next(case for case in plan["cases"] if case["id"] == active)
        try:
            status = json.loads(b"".join(volume.read_file(f"{remote}/{active}/{case_files(case)[0]}")))
            active_progress = status["completed_games"]
        except (FileNotFoundError, modal.exception.NotFoundError):
            pass
    snapshot = {"status": manifest["status"], "verified_cases": len(rows), "verified_games": len(rows)*400,
                "total_cases": plan["total_cases"], "total_games": plan["total_games"], "active_case": active, "active_completed_games": active_progress,
                "updated_at_utc": datetime.now(timezone.utc).isoformat()}
    save_json(local / "status.json", snapshot)
    save_json(local / "verified-progress.json", rows)
    if manifest["status"] == "failed":
        raise RuntimeError(f"{active}: {manifest.get('error')}")
    if manifest["status"] == "completed":
        if len(rows) != plan["total_cases"] or manifest["completed_games"] != plan["total_games"]:
            raise ValueError("Completed suite is missing selected cases")
        (local / "suite.manifest.json").write_bytes(manifest_bytes)
        suite_bytes = b"".join(volume.read_file(remote + "/suite.results.json"))
        if json.loads(suite_bytes) != rows:
            raise ValueError("Locally recomputed suite differs from the remote results")
        (local / "suite.results.json").write_bytes(suite_bytes)
        save_json(local / "verification.json", {"status": "passed", "games_replayed": plan["total_games"], "cases_verified": plan["total_cases"],
                  "opening_dataset_sha256": plan["opening_dataset_sha256"], "files": {
                      "experiment.plan.json": sha(local / "experiment.plan.json"), "suite.results.json": sha(local / "suite.results.json"),
                      "suite.manifest.json": sha(local / "suite.manifest.json")}, "verified_at_utc": datetime.now(timezone.utc).isoformat()})
    return snapshot


def load_verified_suite(local):
    local = Path(local)
    plan = validate_plan(json.loads((local / "experiment.plan.json").read_text()))
    verified = json.loads((local / "verification.json").read_text())
    if verified["status"] != "passed" or verified["games_replayed"] != plan["total_games"] or verified["cases_verified"] != plan["total_cases"]:
        raise ValueError("Export requires the complete locally verified suite")
    for name, expected in verified["files"].items():
        if sha(local / name) != expected:
            raise ValueError("Verified suite artifact changed: " + name)
    rows = json.loads((local / "suite.results.json").read_text())
    if [row["case_id"] for row in rows] != [case["id"] for case in plan["cases"]]:
        raise ValueError("Published cases differ from the selected plan")
    for case in plan["cases"]:
        folder = local / case["id"]
        checked = json.loads((folder / "local-verification.json").read_text())
        if checked["status"] != "passed" or any(sha(folder / name) != digest for name, digest in checked["files"].items()):
            raise ValueError("Verified case artifact changed: " + case["id"])
    return plan, rows, verified


def load_published_suite(directory):
    """Check compact exports for plotting; this does not replay the original games."""
    directory = Path(directory)
    provenance = json.loads((directory / "source_provenance.json").read_text(encoding="utf-8"))
    for name in ("experiment.plan.json", "verification.json"):
        if sha(directory / name) != provenance["sources"][name]["source_sha256"]:
            raise ValueError("Published suite artifact changed: " + name)
    if sha(directory / "results.json") != provenance["exported_results_sha256"]:
        raise ValueError("Published suite artifact changed: results.json")
    plan = validate_plan(json.loads((directory / "experiment.plan.json").read_text(encoding="utf-8")))
    verified = json.loads((directory / "verification.json").read_text(encoding="utf-8"))
    if (verified["status"] != "passed" or verified["games_replayed"] != plan["total_games"]
            or verified["cases_verified"] != plan["total_cases"] or verified["opening_dataset_sha256"] != DATASET_HASH
            or verified["files"]["experiment.plan.json"] != sha(directory / "experiment.plan.json")):
        raise ValueError("Published suite requires the original completed verification")
    rows = json.loads((directory / "results.json").read_text(encoding="utf-8"))
    if [row["case_id"] for row in rows] != [case["id"] for case in plan["cases"]]:
        raise ValueError("Published cases differ from the selected plan")
    for row, case in zip(rows, plan["cases"]):
        if (row["games"] != 400 or row["kind"] != case["kind"] or row["config"] != case["config"]
                or set(row["policies"]) != set(case_policies(case))):
            raise ValueError("Published case differs from the selected plan: " + case["id"])
        for name, stats in row["policies"].items():
            interval = row["score_intervals"][name]
            if (stats["games"] != 400 or stats["wins"] + stats["draws"] + stats["losses"] != 400
                    or stats["win_rate"] != stats["wins"] / 400
                    or stats["score_rate"] != (stats["wins"] + .5 * stats["draws"]) / 400
                    or interval["score_rate"] != stats["score_rate"]
                    or not 0 <= interval["lower"] <= stats["score_rate"] <= interval["upper"] <= 1):
                raise ValueError("Published outcomes or interval differ: " + case["id"])
    return plan, rows, verified


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-name", default="le4capture-mlp-suite-20261011")
    parser.add_argument("--watch", action="store_true")
    args = parser.parse_args()
    local = ROOT / "runs/checkpoint_strength" / args.evaluation_name
    launch = json.loads((local / "launch.json").read_text())
    if os.environ.get("MODAL_PROFILE") != launch["profile"]:
        raise ValueError("Select the recorded Modal workspace")
    plan = validate_plan(json.loads((local / "experiment.plan.json").read_text()))
    volume = modal.Volume.from_name(launch["results_volume"])
    deadline = time.monotonic() + 24*60*60
    previous = None
    while True:
        snapshot = collect(local, launch, plan, volume)
        key = snapshot["status"], snapshot["verified_games"], snapshot.get("active_case"), snapshot.get("active_completed_games")
        if key != previous:
            print(json.dumps(snapshot), flush=True)
            previous = key
        if snapshot["status"] == "completed" or not args.watch:
            return
        if time.monotonic() >= deadline:
            raise TimeoutError("Suite collection exceeded 24 hours")
        time.sleep(30)


if __name__ == "__main__":
    main()
