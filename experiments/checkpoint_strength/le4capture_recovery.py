"""Recover orchestration state without changing the frozen game evaluators."""
from datetime import datetime, timezone
import json
from pathlib import Path
from uuid import uuid4

from experiments.checkpoint_strength.le4capture_suite import case_files, sha


def prepare_case_output(root, case):
    """Reuse complete outputs; retain interrupted attempts before a fresh case."""
    root = Path(root).resolve()
    output = (root / case["id"]).resolve()
    if not output.is_relative_to(root) or output == root:
        raise ValueError("Case output must stay within the evaluation directory")
    if not output.exists():
        return False
    files = case_files(case)
    manifest = json.loads((output/files[0]).read_text()) if (output/files[0]).exists() else {}
    if (manifest.get("status") == "completed" and manifest.get("completed_games") == 400
            and all((output/name).is_file() for name in files)):
        return True
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
    archive = root / "_interrupted_cases" / case["id"] / (stamp+"-"+uuid4().hex[:8])
    if not archive.resolve().is_relative_to(root):
        raise ValueError("Case archive must stay within the evaluation directory")
    archive.parent.mkdir(parents=True, exist_ok=True)
    output.rename(archive)
    return False


def load_saved_suite(root, plan, *, explicit_resume=False, recover_case=None):
    """Validate the completed prefix and its previously verified artifact bytes."""
    root = Path(root)
    if json.loads((root/"experiment.plan.json").read_text()) != plan:
        raise ValueError("Recovery plan differs from the frozen experiment")
    status = json.loads((root/"suite.manifest.json").read_text())
    if status["status"] == "failed" and not explicit_resume:
        raise ValueError("A failed suite requires explicit recovery")
    rows = json.loads((root/"suite.results.json").read_text())
    count = len(status["completed_cases"])
    if (count > len(plan["cases"]) or len(rows) > len(plan["cases"])
            or status["completed_cases"] != [case["id"] for case in plan["cases"][:count]]
            or [row["case_id"] for row in rows] != [case["id"] for case in plan["cases"][:len(rows)]]
            or status["completed_games"] != 400*count):
        raise ValueError("Saved completion state is not the selected case prefix")
    for case in plan["cases"][:max(count, len(rows))]:
        folder = root/case["id"]
        checked = json.loads((folder/"verification.json").read_text())
        if (checked["status"] != "passed" or checked["case_id"] != case["id"]
                or checked["games_replayed"] != 400 or checked["opening_pairs"] != 200
                or checked["opening_dataset_sha256"] != plan["opening_dataset_sha256"]
                or set(checked["files"]) != set(case_files(case))
                or any(sha(folder/name) != digest for name, digest in checked["files"].items())):
            raise ValueError("Previously verified case differs: " + case["id"])
    if len(rows) < count:
        if recover_case is None:
            raise ValueError("Recovery of an interrupted summary write requires terminal replay")
        for case in plan["cases"][len(rows):count]:
            row = recover_case(root/case["id"], case, plan)
            if row["case_id"] != case["id"]:
                raise ValueError("Recovered summary has the wrong case")
            rows.append(row)
    status.update(completed_cases=[row["case_id"] for row in rows], completed_games=400*len(rows))
    return status, rows


def pending_call_id(status, case):
    return status.get("active_call_id") if status.get("active_case") == case["id"] else None
