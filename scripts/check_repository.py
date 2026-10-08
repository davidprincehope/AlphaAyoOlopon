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
    for name in ("README.md", "docs/results/alpha_zero_600/README.md"):
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
                if target.startswith("runs/"):
                    continue
                actual = content.get(target)
                if actual is None or hashlib.sha256(actual).hexdigest() != expected:
                    problems.append({"path": target, "rule": "published-source-hash"})
            if provenance.get("chart_data_sha256"):
                target = str(PurePosixPath(name).parent / "chart_data.csv")
                actual = content.get(target)
                if actual is None or hashlib.sha256(actual).hexdigest() != provenance["chart_data_sha256"]:
                    problems.append({"path": target, "rule": "published-data-hash"})
        elif name == "docs/results/alpha_zero_600/source_provenance.json":
            checks = {path: row["export_sha256"] for path, row in json.loads(data)["sources"].items()}
        else:
            continue
        for relative, expected in checks.items():
            target = str(PurePosixPath(name).parent / relative)
            actual = content.get(target)
            if actual is None or hashlib.sha256(actual).hexdigest() != expected:
                problems.append({"path": target, "rule": "published-artifact-hash"})

    prefix = "docs/results/alpha_zero_600/"
    if prefix + "progression.results.json" in content:
        curves = json.loads(content[prefix + "progression.results.json"])
        checkpoints = [1, *range(50, 601, 50)]
        scores = {}
        for case, rows in curves.items():
            if ([row["learner_round"] for row in rows] != checkpoints
                    or any(row["games"] != 400 for row in rows)):
                problems.append({"path": prefix + "progression.results.json", "rule": "incomplete-progression"})
            for row in rows:
                if row["score_rate"] != (row["wins"] + 0.5 * row["draws"]) / row["games"]:
                    problems.append({"path": prefix + "progression.results.json", "rule": "outcome-score-mismatch"})
                scores[(case, row["learner_round"])] = row["score_rate"]
        csv_name = "visualizations/figures/mcts_learning_progression_13_checkpoints/chart_data.csv"
        rows = list(csv.DictReader(io.StringIO(content[csv_name].decode("utf-8"))))
        if len(rows) != 52 or len(scores) != 52:
            problems.append({"path": csv_name, "rule": "expected-52-observations"})
        for row in rows:
            if abs(float(row["score_percent"]) / 100 - scores[(row["condition"], int(row["learner_round"]))]) > 1e-12:
                problems.append({"path": csv_name, "rule": "plotted-score-mismatch"})
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
