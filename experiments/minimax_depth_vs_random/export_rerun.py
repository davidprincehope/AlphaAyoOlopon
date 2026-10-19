"""Verify a completed rerun and publish its summary and compact paired scores."""

import argparse
import csv
import io
import json
from pathlib import Path

from experiments.minimax_depth_vs_random import rerun_experiment as experiment


def percent(value):
    return "unresolved" if value is None else f"{100 * value:.2f}%"


def interval_text(interval, signed=False):
    if interval is None:
        return "withheld"
    spec = "+.2f" if signed else ".2f"
    return f"{100 * interval['lower']:{spec}} to {100 * interval['upper']:{spec}}"


def export(run_dir, output):
    run_dir, output = Path(run_dir).resolve(), Path(output).resolve()
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    source = run_dir / "experiment_summary.json"
    if manifest["status"] != "completed" or experiment.sha256(source) != manifest["results_sha256"]:
        raise ValueError("A complete, hash-verified run is required")
    report = json.loads(source.read_text(encoding="utf-8"))
    contract = manifest["contract"]
    if report["contract"] != contract:
        raise ValueError("Manifest/result contract mismatch")
    for name, expected in contract["code_sha256"].items():
        if (experiment.sha256(experiment.ROOT / name) != expected
                or experiment.sha256(run_dir / "source_snapshot" / name) != expected):
            raise ValueError(f"Source mismatch: {name}")
    game = experiment.pyspiel.load_game("ayo_olopon", {"max_game_length": contract["max_actions"]})
    games, records_by_depth = contract["games_per_depth"], {}
    for result in report["results"]:
        depth, log_path = result["depth"], run_dir / result["log_file"]
        if experiment.sha256(log_path) != result["log_sha256"]:
            raise ValueError("Raw log hash mismatch")
        records = sorted((json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()),
                         key=lambda row: row["game_id"])
        if len(records) != games:
            raise ValueError("Incomplete raw log")
        for index, record in enumerate(records):
            experiment.verify_record(game, record, depth, index,
                                     contract["master_seed"] + index // 2, contract["max_actions"])
        if any(result.get(key) != value for key, value in experiment.summarize(records).items()):
            raise ValueError("Summary does not match raw game outcomes")
        scores = experiment.pair_scores(records)
        expected_interval = experiment.bootstrap_interval(scores, contract["master_seed"]) if scores is not None else None
        if expected_interval != result["score_interval_95"]:
            raise ValueError("Confidence interval does not match raw game outcomes")
        records_by_depth[depth] = records
        print(f"Replayed and recounted depth {depth}: {len(records)} games", flush=True)
    for comparison in report["depth_comparisons"]:
        lo = experiment.pair_scores(records_by_depth[comparison["reference_depth"]])
        hi = experiment.pair_scores(records_by_depth[comparison["comparison_depth"]])
        differences = hi - lo if lo is not None and hi is not None else None
        expected = experiment.bootstrap_interval(differences, contract["master_seed"]) if differences is not None else None
        if (expected != comparison["paired_interval_95"] or comparison["score_rate_difference"]
                != (float(differences.mean()) if differences is not None else None)):
            raise ValueError("Paired depth comparison mismatch")
    output.mkdir(parents=True, exist_ok=True)
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    depths = contract["depths"]
    writer.writerow(["pair_id", "seed"] + [f"depth_{d}_seat_{s}" for d in depths for s in (0, 1)])
    for pair_id in range(games // 2):
        values = []
        for depth in depths:
            for seat in (0, 1):
                record = records_by_depth[depth][2 * pair_id + seat]
                values.append({"win": 1, "draw": 0.5, "loss": 0, "unresolved": ""}[record["minimax_outcome"]])
        writer.writerow([pair_id, contract["master_seed"] + pair_id, *values])
    matrix = output / "paired_scores.csv"
    matrix.write_text(stream.getvalue(), encoding="utf-8", newline="")
    report["publication_verification"] = {
        "source_run": run_dir.relative_to(experiment.ROOT).as_posix(),
        "started_at_utc": manifest["started_at_utc"],
        "finished_at_utc": manifest["finished_at_utc"],
        "source_results_sha256": experiment.sha256(source),
        "source_manifest_sha256": experiment.sha256(run_dir / "manifest.json"),
        "exporter_sha256": experiment.sha256(__file__),
        "games_replayed_and_recounted": games * len(depths),
        "intervals_recomputed": True, "paired_scores_sha256": experiment.sha256(matrix),
    }
    experiment.write_json(output / "results.json", report)
    lines = ["# Corrected minimax depth comparison versus random", "",
             f"Completed rerun: **{games:,} games per depth**, with {games // 2:,} in each seat. "
             "Every condition uses the same Python game, OpenSpiel alpha-beta search, standard start and seed schedule.", "",
             "| Depth | Wins | Draws | Losses | Win rate | Score rate | 95% score interval | Unresolved |",
             "| ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |"]
    for r in report["results"]:
        score_interval = interval_text(r["score_interval_95"]) + "%" if r["score_interval_95"] else "withheld"
        lines.append(f"| {r['depth']} | {r['wins']:,} | {r['draws']:,} | {r['losses']:,} | "
                     f"{percent(r['win_rate'])} | {percent(r['score_rate'])} | "
                     f"{score_interval} | {r['unresolved_games']:,} |")
    lines += ["", "Score credits draws as half a win. Confidence intervals use 10,000 bootstrap resamples of whole seed/seat pairs.", "",
              "| Depth | Player 0 score | Player 1 score | Mean actions | Maximum actions |",
              "| ---: | ---: | ---: | ---: | ---: |"]
    for r in report["results"]:
        lines.append(f"| {r['depth']} | {percent(r['by_seat']['0']['score_rate'])} | "
                     f"{percent(r['by_seat']['1']['score_rate'])} | {r['average_game_length']:.2f} | {r['maximum_game_length']} |")
    lines += ["", "| Depth comparison | Score difference (percentage points) | Paired 95% interval |",
              "| --- | ---: | --- |"]
    for c in report["depth_comparisons"]:
        difference = "withheld" if c["score_rate_difference"] is None else f"{100 * c['score_rate_difference']:+.2f}"
        lines.append(f"| {c['comparison_depth']} minus {c['reference_depth']} | {difference} | {interval_text(c['paired_interval_95'], True)} |")
    lines += ["", "Intervals are individual and are not adjusted for multiple comparisons. "
              "The comparison preserves matching seeds across depths and both seats within each pair.", "",
              "The heuristic is clipped H* divided by 11, strictly inside terminal utilities ±1. "
              "The 1,000-action horizon is visible to search. Internal and external limit cases are unresolved; "
              "they receive no W/D/L credit. The JSON also records full-sample score bounds if limits occur. "
              "Rule-based repetition resolution counts as a completed game.", "",
              "These results measure performance against uniform random play from the standard start. "
              "They do not establish general playing strength, an equal-runtime comparison, or a new ranking against historical MCTS. "
              "The rerun changes both evaluator scaling and the limit protocol; differences from historical results cannot be attributed to scaling alone.", "",
              "[Results and source hashes](results.json) | [Paired game scores](paired_scores.csv) | "
              "[Protocol and rerun command](../../../../experiments/minimax_depth_vs_random/README.md)", "",
              "The export replays and recounts every game, verifies raw log/source hashes, and recomputes all intervals. "
              "The CSV retains both seat scores for every seed at every depth (1 = win, 0.5 = draw, 0 = loss; blank = unresolved). "
              "Full move traces and exact source snapshots remain in the local run named in `publication_verification`.", ""]
    (output / "README.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Published verified results: {output}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    export(args.run, args.output)
