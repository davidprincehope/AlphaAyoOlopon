"""Render DATASET <=4 CAPTURE benchmarks from verified runs or published summaries."""
import argparse
import csv
from decimal import Decimal, ROUND_HALF_UP
import json
import os
from pathlib import Path
import tempfile

from experiments.checkpoint_strength.collect_le4capture_suite import load_published_suite, load_verified_suite
from experiments.checkpoint_strength.le4capture_suite import ROOT, CHECKPOINTS, EXCLUDED_CURVES, sha
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "alpha-ayo-matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np

from visualizations.paper_style import BLUE, DPI, FIGSIZE, PIXELS, STYLE, finish_axes, style_sha256

COMPARISON_TITLES = {
    "puct64-mcts64": "Neural PUCT 64 versus MCTS 64",
    "policy-mcts64": "Learned policy versus MCTS 64",
    "greedy64": "Neural PUCT 64 versus Greedy EOH",
}

FIGURE_NAMES = {
    "puct64-mcts64": "puct64_vs_mcts64",
    "policy-mcts64": "policy_only_vs_mcts64",
    "greedy64": "puct64_vs_greedy_eoh_hstar",
}


def percent_label(rate):
    return str((Decimal(str(rate))*100).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))+"%"


def progression(rows, condition, output):
    rows = sorted(rows, key=lambda row: row["config"]["checkpoint"])
    if [row["config"]["checkpoint"] for row in rows] != list(CHECKPOINTS):
        raise ValueError("A progression chart requires all 13 selected checkpoints")
    scores, lower, upper = [], [], []
    for row in rows:
        name = "ALPHAZERO" if condition == "greedy64" else f"C{row['config']['checkpoint']}"
        interval = row["score_intervals"][name]
        scores.append(100 * row["policies"][name]["score_rate"])
        lower.append(100 * interval["lower"])
        upper.append(100 * interval["upper"])
    scores, lower, upper = map(np.asarray, (scores, lower, upper))
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=FIGSIZE)
        ax.errorbar(CHECKPOINTS, scores, yerr=[scores-lower, upper-scores], fmt="none", ecolor=BLUE, alpha=.55, capsize=3, elinewidth=1.2)
        ax.plot(CHECKPOINTS, scores, "o-", color=BLUE, linewidth=1.5, markersize=4.5)
        ax.axhline(50, color="#999999", linewidth=1, linestyle=(0, (4, 4)))
        final_name = "ALPHAZERO" if condition == "greedy64" else "C600"
        ax.annotate(percent_label(rows[-1]["policies"][final_name]["score_rate"]), (600, scores[-1]), xytext=(9, 0), textcoords="offset points", color=BLUE, fontsize=10, va="center")
        ax.set(xlim=(-15, 675), ylim=(0, 100), xlabel="Learner round", ylabel="Score rate")
        ax.set_xticks([1, 100, 200, 300, 400, 500, 600])
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.yaxis.set_major_formatter(PercentFormatter(100, decimals=0))
        finish_axes(fig, ax)
        path = output / (FIGURE_NAMES[condition] + ".png")
        fig.savefig(path, dpi=DPI, facecolor="white")
        plt.close(fig)
        return path


def render(run, output, *, published=False):
    run, output = Path(run), Path(output)
    plan, rows, verified = (load_published_suite if published else load_verified_suite)(run)
    output.mkdir(parents=True, exist_ok=True)
    plotted_rows = [row for row in rows if row["case_id"].split("/", 1)[0] in COMPARISON_TITLES]
    files = []
    for condition in COMPARISON_TITLES:
        selected = [row for row in rows if row["case_id"].startswith(condition+"/")]
        files.append(progression(selected, condition, output))
    table = []
    for row in plotted_rows:
        for policy, stats in row["policies"].items():
            interval = row["score_intervals"][policy]
            table.append({"case_id": row["case_id"], "policy": policy, "games": 400, "wins": stats["wins"], "draws": stats["draws"],
                          "losses": stats["losses"], "win_percent": 100*stats["win_rate"], "score_percent": 100*stats["score_rate"],
                          "ci_lower_percent": 100*interval["lower"], "ci_upper_percent": 100*interval["upper"]})
    with (output / "chart_data.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    source_files = ("experiment.plan.json", "results.json", "verification.json", "source_provenance.json") if published else (
        "experiment.plan.json", "suite.results.json", "verification.json")
    try:
        source_directory = run.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        source_directory = run.resolve().as_posix()
    provenance = {"opening_dataset_sha256": plan["opening_dataset_sha256"], "source_run": plan["source_run"], "games_verified": verified["games_replayed"],
                  "case_count": plan["total_cases"], "plotted_case_count": len(plotted_rows),
                  "plotted_games": sum(row["games"] for row in plotted_rows), "figure_count": len(COMPARISON_TITLES),
                  "excluded_curves": EXCLUDED_CURVES,
                  "source_kind": "published_verified_summaries" if published else "locally_verified_run",
                  "source_sha256": {source_directory + "/" + name: sha(run/name) for name in source_files},
                  "chart_data_sha256": sha(output / "chart_data.csv"),
                  "script": Path(__file__).relative_to(ROOT).as_posix(), "script_sha256": sha(Path(__file__)), "style_sha256": style_sha256(), "figure_sha256": {path.name: sha(path) for path in files}}
    (output / "figure_provenance.json").write_text(json.dumps(provenance, indent=2)+"\n", encoding="utf-8")
    (output / "README.md").write_text(
        "# DATASET ≤4 CAPTURE benchmark figures\n\n"
        "Three checkpoint progression figures: neural PUCT 64 versus MCTS 64, policy-only versus MCTS 64, "
        "and neural PUCT 64 versus Greedy EOH H*. Each includes C1 and C50–C600 in 50-round increments. "
        "The 39 plotted cases contain 15,600 games. PUCT 64 versus MCTS 128 is retained as a C600-only comparison "
        "in its [experiment results](../../../docs/results/c600_comparisons/c600_puct64_vs_mcts128/README.md). The historical published suite retains 56 verified cases and 22,400 games.\n\n"
        "All use the same 200 frozen openings with starting captured-seed gap at most four, both seats and paired seeds. "
        "The entire neural 128/MCTS 128 and neural 128/Greedy EOH curves are excluded. "
        "Each point contains 400 games, including the C600 endpoint of each curve.\n\n"
        "The dashed horizontal line marks a 50% score; error bars show pointwise 95% intervals. Lines connect observations without smoothing. "
        "Scores credit draws as half a win. Intervals are pointwise 95% bootstrap intervals from 10,000 resamples of whole opening pairs. "
        "They describe sampled-position variation, excluding training-seed and search-seed uncertainty. Equal simulation budgets do not imply equal runtime.\n\n"
        "Each PNG contains one plot, with serif typography and a light grid, exported at 300 dpi.  `chart_data.csv` contains both players' statistics for the 39 plotted cases. "
        "The provenance file records source and output hashes.\n\n"
        "The default command regenerates charts from hash-checked published summaries. It preserves the original study's values and intervals; "
        "replaying the original matches requires separately retained raw histories. Use `--run <local-suite>` for a new, fully verified evaluation.\n\n"
        "```powershell\n.venv/Scripts/python.exe -B -m visualizations.plot_checkpoint_progression\n```\n", encoding="utf-8")
    with (output/"README.md").open("a", encoding="utf-8") as guide:
        guide.write("\n[Plotted values](chart_data.csv), "
                    "[source provenance](figure_provenance.json), and "
                    "[verified summaries](../../../docs/results/checkpoint_progression/README.md).\n\n"
                    "| Figure | PNG |\n| --- | --- |\n")
        names = [(FIGURE_NAMES[condition], COMPARISON_TITLES[condition]) for condition in COMPARISON_TITLES]
        for stem, title in names:
            guide.write(f"| {title} | [PNG]({stem}.png) |\n")
    from PIL import Image
    for path in files:
        with Image.open(path) as picture:
            if picture.size != PIXELS:
                raise ValueError("Unexpected PNG dimensions")
            picture.verify()
    print(json.dumps({"figures": len(COMPARISON_TITLES), "plotted_cases": len(plotted_rows), "games_verified": verified["games_replayed"], "output": str(output)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--run", type=Path, help="Complete locally verified suite, including raw histories")
    source.add_argument("--published", type=Path, help="Compact published summaries (the default)")
    parser.add_argument("--output", type=Path, default=ROOT/"visualizations/figures/checkpoint_progression")
    args = parser.parse_args()
    render(args.run or args.published or ROOT/"docs/results/checkpoint_progression",
           args.output, published=args.run is None)
