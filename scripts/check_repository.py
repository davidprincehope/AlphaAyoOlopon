"""Check the proposed repository contents without printing credential values."""
import argparse
import csv
import hashlib
import io
import json
import re
import subprocess
from pathlib import Path, PurePosixPath
from urllib.parse import unquote
import zipfile

ROOT = Path(__file__).resolve().parents[1]
MAX_BYTES = 100 * 1024 * 1024
RULES = {
    "private-key": rb"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----",
    "aws-access-key": rb"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b",
    "github-token": rb"\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{50,})\b",
    "openai-token": rb"\bsk-(?:proj-|svcacct-)?[A-Za-z0-9_-]{32,}\b",
    "slack-token": rb"\bxox[baprs]-[A-Za-z0-9-]{20,}\b",
    "modal-token-secret": rb"\bas-[A-Za-z0-9]{30,}\b",
}
PATTERNS = {name: re.compile(pattern) for name, pattern in RULES.items()}


def git(*args, data=None):
    return subprocess.check_output(["git", *args], input=data, cwd=ROOT)


def bad_path(name):
    path = PurePosixPath(name)
    parts = set(path.parts)
    if (parts & {"__pycache__", ".venv", "venv", ".pytest_cache", ".aws", ".codex", ".git"}
            or name.startswith(("runs/", "build/", "outputs/", "experiments/EOH/vendor/"))
            or path.suffix.lower() in {".pyc", ".pyo", ".npz", ".ocdbt", ".log"}):
        return "generated-or-local-file"
    if ((path.name == ".env" or path.name.startswith(".env.")) and path.name != ".env.example"
            or path.name in {".modal.toml", "credentials"}
            or path.suffix.lower() in {".pem", ".key"}):
        return "credential-file"
    return None


def secrets(data):
    return [name for name, pattern in PATTERNS.items() if pattern.search(data)]


def read_blobs(blobs):
    process = subprocess.Popen(["git", "cat-file", "--batch"], cwd=ROOT,
                               stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    try:
        for blob in blobs:
            process.stdin.write((blob + "\n").encode())
            process.stdin.flush()
            header = process.stdout.readline().decode().split()
            if len(header) != 3 or header[:2] != [blob, "blob"]:
                raise RuntimeError("Unexpected Git object stream")
            size = int(header[2])
            data = process.stdout.read(size)
            if len(data) != size or process.stdout.read(1) != b"\n":
                raise RuntimeError("Incomplete Git object stream")
            yield data
    finally:
        process.stdin.close()
        process.wait()


def current(staged):
    entries = {}
    submodules = []
    for row in git("ls-files", "--stage", "-z").split(b"\0"):
        if not row:
            continue
        metadata, raw_path = row.split(b"\t", 1)
        mode, blob, stage = metadata.decode().split()
        path = raw_path.decode("utf-8")
        if stage != "0":
            raise ValueError(f"Unresolved Git index entry: {path}")
        if mode == "160000":
            submodules.append(path)
        else:
            entries[path] = blob
    if not staged:
        for path in git("ls-files", "--others", "--exclude-standard", "-z").split(b"\0"):
            if path:
                entries[path.decode("utf-8")] = None
        entries = {name: blob for name, blob in entries.items() if (ROOT / name).is_file()}
    problems, content = [], {}
    total = 0
    payloads = read_blobs(entries.values()) if staged else ((ROOT / name).read_bytes() for name in entries)
    for name, data in zip(entries, payloads):
        content[name] = data
        total += len(data)
        reason = bad_path(name)
        if reason:
            problems.append({"path": name, "rule": reason})
        if len(data) > MAX_BYTES:
            problems.append({"path": name, "rule": "file-over-100-MiB"})
        for rule in secrets(data):
            problems.append({"path": name, "rule": rule})
        if name.endswith(".zip"):
            with zipfile.ZipFile(io.BytesIO(data)) as archive:
                if archive.testzip() is not None:
                    problems.append({"path": name, "rule": "invalid-archive"})
                for member in archive.namelist():
                    reason = bad_path(member)
                    if reason:
                        problems.append({"path": name + ":" + member, "rule": reason})
                    for rule in secrets(archive.read(member)):
                        problems.append({"path": name + ":" + member, "rule": rule})

    # README links must be present in the proposed tree, or in its submodule.
    for name in (name for name in content if name.endswith(".md")):
        source = content.get(name, b"").decode("utf-8")
        for target in re.findall(r"\]\(([^)]+)\)", source):
            if target.startswith(("https://", "http://", "#", "mailto:")):
                continue
            relative = unquote(target.split("#", 1)[0])
            absolute = (ROOT / PurePosixPath(name).parent / relative).resolve()
            if not absolute.is_relative_to(ROOT):
                problems.append({"path": name, "rule": "nonportable-link", "target": target})
                continue
            key = absolute.relative_to(ROOT).as_posix()
            available = key in content or any(item.startswith(key.rstrip("/") + "/") for item in content)
            if any(key.startswith(module + "/") for module in submodules):
                available = absolute.exists()
            if not available:
                problems.append({"path": name, "rule": "missing-published-link", "target": target})

    # Compare hashes to the bytes that would actually be committed.
    for name, data in content.items():
        if name.endswith("figure_provenance.json"):
            provenance = json.loads(data)
            checks = provenance.get("figure_sha256", {})
            sources = provenance.get("source_sha256", {})
            if isinstance(sources, str):
                sources = {provenance["source"]: sources}
            for relative, expected in sources.items():
                target = str(PurePosixPath(provenance.get("experiment", "")) / relative
                             if "/" not in relative else PurePosixPath(relative))
                # Full local run evidence is intentionally excluded from Git.
                if target.startswith("runs/") or relative in {"experiment.plan.json", "suite.results.json", "verification.json"} and provenance.get("source_run"):
                    continue
                actual = content.get(target)
                if actual is None or hashlib.sha256(actual).hexdigest() != expected:
                    problems.append({"path": target, "rule": "published-source-hash"})
            if provenance.get("chart_data_sha256"):
                target = str(PurePosixPath(name).parent / "chart_data.csv")
                actual = content.get(target)
                if actual is None or hashlib.sha256(actual).hexdigest() != provenance["chart_data_sha256"]:
                    problems.append({"path": target, "rule": "published-data-hash"})
        elif name == "docs/results/mlp600_training/source_provenance.json":
            checks = {path: row["export_sha256"] for path, row in json.loads(data)["sources"].items()}
        else:
            continue
        for relative, expected in checks.items():
            target = str(PurePosixPath(name).parent / relative)
            actual = content.get(target)
            if actual is None or hashlib.sha256(actual).hexdigest() != expected:
                problems.append({"path": target, "rule": "published-artifact-hash"})

    audit_name = "docs/results/classical_search/source_audit/source_audit.json"
    if audit_name in content:
        classical_folders = {
            "greedy_hstar": "classical_greedy_eoh_hstar_vs_random",
            "minimax_depth_1": "classical_minimax_depth_1_python_vs_random",
            "minimax_depth_2": "classical_minimax_depth_2_python_vs_random",
            "minimax_depth_3": "classical_minimax_depth_3_python_vs_random",
            "minimax_depth_4": "classical_minimax_depth_4_cpp_vs_random",
            "minimax_depth_5": "classical_minimax_depth_5_cpp_vs_random",
            "mcts_1000": "classical_uct_mcts_1000_simulations_cpp_vs_random",
        }
        audit = json.loads(content[audit_name])["large_benchmarks"]
        for case_id, folder in classical_folders.items():
            result_name = "docs/results/classical_search/" + folder + "/results.json"
            if result_name not in content:
                problems.append({"path": result_name, "rule": "missing-classical-experiment"})
                continue
            result = json.loads(content[result_name])
            provenance = result.pop("source_provenance")
            record = audit[case_id]
            if (result != json.loads(content[record["summary"]])
                    or provenance["audit_case_id"] != case_id or provenance["saved_recount"] != record):
                problems.append({"path": result_name, "rule": "classical-results-differ-from-source"})
            for name, expected in provenance["sources"].items():
                if name not in content or hashlib.sha256(content[name]).hexdigest() != expected:
                    problems.append({"path": result_name, "rule": "classical-source-hash"})
            stats = result["summary"]["policies"]["GREEDY_HSTAR"] if case_id == "greedy_hstar" else result.get("summary", result)
            if (stats["games"] != 10000 or stats["wins"] + stats["draws"] + stats["losses"] != 10000
                    or stats["score_rate"] != (stats["wins"] + 0.5 * stats["draws"]) / 10000
                    or any(stats[key] != record[key] for key in ("wins", "draws", "losses", "win_rate", "score_rate"))):
                problems.append({"path": result_name, "rule": "classical-outcome-mismatch"})

    prefix = "docs/results/checkpoint_progression/"
    if prefix + "results.json" in content:
        cases = json.loads(content[prefix + "results.json"])
        if len(cases) != 56 or sum(row["games"] for row in cases) != 22400:
            problems.append({"path": prefix + "results.json", "rule": "incomplete-benchmark"})
        scores = {}
        for row in cases:
            for name, stats in row["policies"].items():
                if (stats["wins"] + stats["draws"] + stats["losses"] != stats["games"]
                        or stats["score_rate"] != (stats["wins"] + 0.5 * stats["draws"]) / stats["games"]):
                    problems.append({"path": prefix + "results.json", "rule": "outcome-score-mismatch"})
                if row["case_id"].split("/", 1)[0] in ("greedy64", "puct64-mcts64", "policy-mcts64"):
                    scores[(row["case_id"], name)] = stats["score_rate"]
        provenance = json.loads(content[prefix + "source_provenance.json"])
        if hashlib.sha256(content[prefix + "results.json"]).hexdigest() != provenance["exported_results_sha256"]:
            problems.append({"path": prefix + "results.json", "rule": "published-artifact-hash"})
        csv_name = "visualizations/figures/checkpoint_progression/chart_data.csv"
        rows = list(csv.DictReader(io.StringIO(content[csv_name].decode("utf-8"))))
        if len(rows) != len(scores) or {(r["case_id"], r["policy"]) for r in rows} != set(scores):
            problems.append({"path": csv_name, "rule": "incomplete-plotted-cases"})
        for row in rows:
            if abs(float(row["score_percent"]) / 100 - scores[(row["case_id"], row["policy"])]) > 1e-12:
                problems.append({"path": csv_name, "rule": "plotted-score-mismatch"})
        c600_folders = {
            "greedy64/c000600": "c600_puct64_vs_greedy_eoh_hstar",
            "puct64-mcts64/c000600": "c600_puct64_vs_mcts64",
            "puct64-mcts128/c000600": "c600_puct64_vs_mcts128",
            "policy-mcts64/c000600": "c600_policy_only_vs_mcts64",
            "head-to-head/350-600": "c600_puct64_vs_c350_puct64",
        }
        endpoints = {row["case_id"]: row for row in cases if row["case_id"] in c600_folders}
        for case_id, folder in c600_folders.items():
            result_name = "docs/results/c600_comparisons/" + folder + "/results.json"
            if result_name not in content:
                problems.append({"path": result_name, "rule": "missing-c600-experiment"})
                continue
            exported = json.loads(content[result_name])
            source = exported.pop("source_provenance")
            if exported != endpoints[case_id]:
                problems.append({"path": result_name, "rule": "c600-results-differ-from-source"})
            for name, expected in source["sources"].items():
                if name not in content or hashlib.sha256(content[name]).hexdigest() != expected:
                    problems.append({"path": result_name, "rule": "c600-source-hash"})
            if hashlib.sha256(content[source["dataset"]]).hexdigest() != source["dataset_file_sha256"]:
                problems.append({"path": result_name, "rule": "c600-dataset-mismatch"})
        capture_name = "docs/results/c600_comparisons/c600_puct64_vs_capture_only_greedy/results.json"
        if capture_name not in content:
            problems.append({"path": capture_name, "rule": "missing-c600-experiment"})
        else:
            capture = json.loads(content[capture_name])
            plan = json.loads(content[prefix + "experiment.plan.json"])
            if (capture["checkpoint"] != 600 or capture["games"] != 400
                    or capture["verification"]["status"] != "passed"
                    or capture["checkpoint_sha256"] != plan["checkpoint_sha256"]["600"]
                    or capture["opening_dataset"]["content_sha256"] != plan["opening_dataset_sha256"]):
                problems.append({"path": capture_name, "rule": "c600-model-or-dataset-mismatch"})
    minimax_prefix = "docs/results/c600_comparisons/c600_puct64_vs_minimax_depth6/"
    if minimax_prefix + "results.json" in content:
        result = json.loads(content[minimax_prefix + "results.json"])
        manifest = json.loads(content[minimax_prefix + "evaluation_manifest.json"])
        reference = json.loads(content["docs/results/checkpoint_progression/experiment.plan.json"])
        stats = result["policies"]["C600"]
        interval = result["score_intervals"]["C600"]
        if (result["status"] != "completed" or result["games"] != 400
                or result["opening_pairs"] != 200 or result["checkpoint"] != 600
                or result["neural_simulations"] != 64
                or result["opponent"] != {"name": "MINIMAX", "params": {"maximum_depth": 6}}
                or result["checkpoint_sha256"] != reference["checkpoint_sha256"]["600"]
                or result["opening_dataset"]["content_sha256"] != reference["opening_dataset_sha256"]
                or result["verification"]["games_replayed"] != 400
                or not result["verification"]["complete_swapped_seat_pairs"]
                or stats["wins"] + stats["draws"] + stats["losses"] != 400
                or stats["seats"] != {"0": 200, "1": 200}
                or stats["score_rate"] != (stats["wins"] + 0.5 * stats["draws"]) / 400
                or stats["win_rate"] != stats["wins"] / 400
                or interval["opening_pairs"] != 200 or interval["resamples"] != 10000
                or abs(interval["score_rate"] - stats["score_rate"]) > 1e-12
                or manifest["status"] != "completed" or manifest["completed_games"] != 400
                or manifest["contract"]["checkpoint_sha256"] != result["checkpoint_sha256"]
                or manifest["results_sha256"] != hashlib.sha256(content[minimax_prefix + "results.json"]).hexdigest()):
            problems.append({"path": minimax_prefix + "results.json", "rule": "c600-minimax-result-mismatch"})
    print(json.dumps({"scope": "staged" if staged else "worktree", "files": len(entries),
                      "total_MiB": round(total / 1024 ** 2, 2), "problems": problems}, indent=2))
    return not problems


def history():
    objects = {}
    for line in git("rev-list", "--objects", "--all").decode("utf-8").splitlines():
        blob, _, name = line.partition(" ")
        objects[blob] = name
    check = git("cat-file", "--batch-check=%(objectname) %(objecttype) %(objectsize)",
                data=("\n".join(objects) + "\n").encode())
    blobs = [(blob, int(size)) for blob, kind, size in (line.split() for line in check.decode().splitlines())
             if kind == "blob"]
    findings, large = [], []
    process = subprocess.Popen(["git", "cat-file", "--batch"], cwd=ROOT,
                               stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    try:
        for blob, size in blobs:
            process.stdin.write((blob + "\n").encode())
            process.stdin.flush()
            header = process.stdout.readline().decode().split()
            if header != [blob, "blob", str(size)]:
                raise RuntimeError("Unexpected Git object stream")
            data = process.stdout.read(size)
            if len(data) != size or process.stdout.read(1) != b"\n":
                raise RuntimeError("Incomplete Git object stream")
            for rule in secrets(data):
                findings.append({"blob": blob, "path": objects[blob], "rule": rule})
            if size > 50 * 1024 * 1024:
                large.append({"blob": blob, "path": objects[blob], "MiB": round(size / 1024 ** 2, 2)})
    finally:
        process.stdin.close()
        process.wait()
    print(json.dumps({"scope": "reachable-history", "blobs_checked": len(blobs),
                      "credential_findings": findings, "blobs_over_50_MiB": large}, indent=2))
    return not findings and all(row["MiB"] <= 100 for row in large)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--staged", action="store_true")
    mode.add_argument("--worktree", action="store_true")
    mode.add_argument("--history", action="store_true")
    args = parser.parse_args()
    raise SystemExit(0 if (history() if args.history else current(args.staged)) else 1)
