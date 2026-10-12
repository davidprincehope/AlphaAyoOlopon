"""Export compact summaries only after every selected case is locally verified."""
import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.checkpoint_strength.collect_le4capture_suite import load_verified_suite
from experiments.checkpoint_strength.le4capture_suite import save_json, sha


def export(run, output):
    run, output = Path(run), Path(output)
    plan, rows, verification = load_verified_suite(run)
    output.mkdir(parents=True, exist_ok=True)
    save_json(output / "experiment.plan.json", plan)
    save_json(output / "results.json", rows)
    save_json(output / "verification.json", verification)
    manifest = json.loads((run / "suite.manifest.json").read_text())
    sources = {name: {"source": (run/name).relative_to(ROOT).as_posix(), "source_sha256": sha(run/name)}
               for name in ("experiment.plan.json", "suite.results.json", "verification.json", "suite.manifest.json")}
    table = []
    for row in rows:
        for name, stats in row["policies"].items():
            interval = row["score_intervals"][name]
            table.append({"case": row["case_id"], "agent": name, "games": 400, "wins": stats["wins"], "draws": stats["draws"],
                          "losses": stats["losses"], "win_rate": stats["win_rate"], "score_rate": stats["score_rate"],
                          "ci_lower": interval["lower"], "ci_upper": interval["upper"]})
    with (output/"results.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    save_json(output/"source_provenance.json", {"opening_dataset_sha256": plan["opening_dataset_sha256"],
              "model": plan["architecture"], "sources": sources, "dataset_file_sha256": plan["opening_dataset_file_sha256"],
              "checkpoint_sha256": plan["checkpoint_sha256"], "code_sha256": plan["code_sha256"],
              "orchestration_revisions": manifest.get("orchestration_revisions", []),
              "recoveries": manifest.get("recoveries", []),
              "exporter_sha256": sha(Path(__file__)), "exported_results_sha256": sha(output/"results.json")})
    print(json.dumps({"cases_exported": len(rows), "games_verified": verification["games_replayed"], "output": str(output)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True,
                        help="Directory containing a complete locally verified le4capture suite")
    parser.add_argument("--output", type=Path, default=ROOT/"docs/results/checkpoint_progression")
    args = parser.parse_args()
    export(args.run, args.output)
