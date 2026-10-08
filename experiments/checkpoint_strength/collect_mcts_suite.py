"""Monitor the four repeat MCTS comparisons and fetch their verified artifacts."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time

import modal
from experiments.alpha_zero.modal_common import ROOT, validate_name


def read(volume, remote):
    try:
        return b"".join(volume.read_file(remote))
    except (FileNotFoundError, modal.exception.NotFoundError):
        return None


def download_case(volume, remote, local, manifest, checkpoint, configuration):
    local.mkdir(parents=True, exist_ok=True)
    for name, hash_key in (("results.json", "summary_sha256"), ("results.games.jsonl", "games_sha256")):
        content = read(volume, remote + "/" + name)
        if content is None or hashlib.sha256(content).hexdigest() != manifest[hash_key]:
            raise RuntimeError(f"Artifact hash mismatch: {remote}/{name}")
        temporary = local / (name + ".tmp")
        temporary.write_bytes(content)
        temporary.replace(local / name)
    result = json.loads((local / "results.json").read_text())
    assert result["status"] == "completed" and result["games"] == 400
    assert result["config"] == configuration
    assert manifest["checkpoints"][0]["learner_round"] == checkpoint
    records = [json.loads(line) for line in (local / "results.games.jsonl").read_text().splitlines()]
    assert [row["game_id"] for row in records] == list(range(400))
    (local / "results.manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return result


def write_summary(local, launch):
    labels = {"puct64-mcts64": "64 vs 64", "puct128-mcts128": "128 vs 128",
              "puct64-mcts128": "64 vs 128", "policy-mcts64": "Policy only vs 64"}
    old_names = {"puct64-mcts64": "checkpoint-c600-vs-mcts-64-20261005",
                 "puct128-mcts128": "checkpoint-c600-vs-mcts-128-20261005",
                 "puct64-mcts128": "checkpoint-c600-64-vs-mcts-128-20261005",
                 "policy-mcts64": "checkpoint-c600-policy-vs-mcts-64-20261005"}
    checkpoint = launch["checkpoint"]
    lines = [f"# C{checkpoint} versus vanilla MCTS", "",
             "Four comparisons, 400 games each: the same 200 openings, swapped seats, and paired seeds.", "",
             "| Neural / MCTS | W / D / L | Score | 95% interval | Earlier C600 score | Change (pp) |",
             "| --- | --- | ---: | --- | ---: | ---: |"]
    for name in launch["cases"]:
        result = json.loads((local / name / "results.json").read_text())
        stats = result["policies"][f"C{checkpoint}"]
        interval = result["score_intervals"][f"C{checkpoint}"]
        old = json.loads((ROOT / "runs/checkpoint_strength" / old_names[name] / "results.json").read_text())["policies"]["C600"]
        lines.append(f"| {labels[name]} | {stats['wins']} / {stats['draws']} / {stats['losses']} | "
                     f"{100 * stats['score_rate']:.3f}% | {100 * interval['lower']:.3f}-{100 * interval['upper']:.3f}% | "
                     f"{100 * old['score_rate']:.3f}% | {100 * (stats['score_rate'] - old['score_rate']):+.3f} |")
    lines += ["", "Score counts draws as half a win. Intervals bootstrap complete opening pairs.",
              "Policy only selects the highest-probability legal move, without search or value-based action selection.", ""]
    (local / "RESULTS.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation-name", required=True)
    parser.add_argument("--watch", action="store_true")
    args = parser.parse_args()
    name = validate_name(args.evaluation_name)
    if os.environ.get("MODAL_PROFILE") != "davidprincehope":
        raise ValueError("Select the fresh davidprincehope profile explicitly")
    local = ROOT / "runs/checkpoint_strength" / name
    launch = json.loads((local / "launch.json").read_text())
    volume = modal.Volume.from_name(launch["results_volume"])
    deadline = time.monotonic() + 2 * 60 * 60
    previous = None
    while True:
        progress = {}
        for case, configuration in launch["cases"].items():
            validate_name(case)
            remote = f"/{name}/{case}"
            content = read(volume, remote + "/results.manifest.json")
            manifest = json.loads(content) if content else None
            progress[case] = {"status": manifest["status"], "completed_games": manifest["completed_games"],
                              "total_games": manifest["total_games"]} if manifest else {"status": "waiting", "completed_games": 0, "total_games": 400}
            if manifest and manifest["status"] == "failed":
                raise RuntimeError(f"{case} failed: {manifest.get('error')}")
            if manifest and manifest["status"] == "completed" and not (local / case / "results.manifest.json").exists():
                download_case(volume, remote, local / case, manifest, launch["checkpoint"], configuration)
        root_content = read(volume, f"/{name}/suite.manifest.json")
        suite = json.loads(root_content) if root_content else None
        snapshot = {"status": suite["status"] if suite else "waiting", "checkpoint": launch["checkpoint"],
                    "updated_at_utc": datetime.now(timezone.utc).isoformat(), "cases": progress}
        temporary = local / "status.json.tmp"
        temporary.write_text(json.dumps(snapshot, indent=2) + "\n")
        temporary.replace(local / "status.json")
        if progress != previous:
            print(json.dumps(snapshot), flush=True)
            previous = progress
        if suite and suite["status"] == "failed":
            raise RuntimeError(suite.get("error", "Suite finalization failed"))
        if suite and suite["status"] == "completed":
            for filename in ("suite.manifest.json", "suite.results.json"):
                content = read(volume, f"/{name}/{filename}")
                if content is None:
                    raise RuntimeError("Missing completed suite artifact")
                (local / filename).write_bytes(content)
            assert all(item["status"] == "completed" and item["completed_games"] == 400 for item in progress.values())
            write_summary(local, launch)
            hashes = {str(path.relative_to(local)): hashlib.sha256(path.read_bytes()).hexdigest()
                      for path in local.rglob("results*.json*")}
            (local / "download.manifest.json").write_text(json.dumps(
                {"status": "completed", "completed_games": 1600, "files": hashes}, indent=2) + "\n")
            print(json.dumps({"event": "mcts_suite_downloaded", "checkpoint": launch["checkpoint"], "games": 1600}), flush=True)
            return
        if not args.watch:
            return
        if time.monotonic() >= deadline:
            raise TimeoutError("MCTS suite monitoring exceeded two hours")
        time.sleep(30)


if __name__ == "__main__":
    main()
