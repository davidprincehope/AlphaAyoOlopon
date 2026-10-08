"""Four progression curves from the complete 13-checkpoint MCTS experiments."""
import argparse
import csv
import json
from pathlib import Path
import zipfile

from visualizations.c600_mcts_conditions import ROOT, STYLE, BLUE, require, sha256
from experiments.checkpoint_strength.mcts_progression import CHECKPOINTS, validate_curve
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np

LABELS = {
    "puct64-mcts64": "AlphaZero PUCT: 64 simulations/move  |  Plain MCTS UCT: 64 simulations/move",
    "puct128-mcts128": "AlphaZero PUCT: 128 simulations/move  |  Plain MCTS UCT: 128 simulations/move",
    "puct64-mcts128": "AlphaZero PUCT: 64 simulations/move  |  Plain MCTS UCT: 128 simulations/move",
    "policy-mcts64": "AlphaZero policy alone (no search)  |  Plain MCTS UCT: 64 simulations/move",
}


def render(run, output):
    run, output = run.resolve(), output.resolve()
    manifest = json.loads((run / "progression.manifest.json").read_text(encoding="utf-8"))
    verification = json.loads((run / "verification.json").read_text(encoding="utf-8"))
    require(manifest["status"] == "completed" and manifest["completed_games"] == 20800
            and verification["status"] == "passed" and verification["games_replayed"] == 20800,
            "Only the completed, verified 20,800-game progression can be plotted")
    for name, expected in verification["files"].items():
        require(sha256(run / name) == expected, f"Verified source artifact changed: {name}")
    curves = json.loads((run / "progression.results.json").read_text(encoding="utf-8"))
    require(set(curves) == set(LABELS), "Expected all four progression conditions")
    for rows in curves.values():
        validate_curve(rows)
    output.mkdir(parents=True, exist_ok=True)
    files, table = [], []
    for condition, subtitle in LABELS.items():
        rows = curves[condition]
        scores = np.array([row["score_rate"] * 100 for row in rows])
        lower = np.array([row["score_interval"]["lower"] * 100 for row in rows])
        upper = np.array([row["score_interval"]["upper"] * 100 for row in rows])
        with plt.rc_context(STYLE):
            fig = plt.figure(figsize=(11, 6.5))
            fig.text(0.09, 0.95, "AlphaZero learning progression", fontsize=18, weight="bold", va="top")
            fig.text(0.09, 0.885, subtitle, fontsize=11, color="#444444", va="top")
            fig.text(0.09, 0.835,
                     "13 checkpoints  |  400 games per checkpoint  |  5,200 games total",
                     fontsize=10, color="#666666", va="top")
            ax = fig.add_axes((0.09, 0.25, 0.84, 0.50))
            ax.errorbar(CHECKPOINTS, scores, yerr=[scores - lower, upper - scores], fmt="none",
                        ecolor=BLUE, alpha=0.5, capsize=4, elinewidth=1.2)
            ax.plot(CHECKPOINTS, scores, "o-", color=BLUE, markersize=5, linewidth=1.6,
                    label="Checkpoint score")
            ax.axhline(50, linestyle=(0, (4, 4)), color="#888888", linewidth=1,
                       label="Equal score (50%)")
            ax.text(610, scores[-1], f"{scores[-1]:.2f}%", fontsize=11, color=BLUE, va="center",
                    bbox={"facecolor": "white", "edgecolor": "none", "pad": 2})
            ax.set(xlim=(-8, 668), ylim=(0, 100), xlabel="Completed learner rounds",
                   ylabel="AlphaZero score against plain MCTS")
            ax.set_xticks(CHECKPOINTS)
            ax.set_yticks([0, 25, 50, 75, 100])
            ax.yaxis.set_major_formatter(PercentFormatter(100, decimals=0))
            ax.set_axisbelow(True)
            ax.grid(axis="y", color="#E6E6E6", linewidth=0.7)
            ax.tick_params(length=3, color="#555555", axis="both")
            ax.legend(loc="lower right", frameon=False, fontsize=10)
            fig.text(0.09, 0.12,
                     "Score counts draws as half a win. The same 200 openings are played in both seats at every checkpoint.",
                     fontsize=9.5, color="#555555")
            fig.text(0.09, 0.075,
                     "Error bars: pointwise 95% opening-pair bootstrap intervals; exclude training-seed and search-seed uncertainty.",
                     fontsize=9.5, color="#555555")
            fig.text(0.09, 0.03, "Lines connect the 13 evaluated checkpoints without smoothing. MCTS uses uniform random rollouts.",
                     fontsize=9.5, color="#555555")
            for extension in ("png", "svg"):
                path = output / f"{condition}.{extension}"
                fig.savefig(path, dpi=300)
                files.append(path)
            plt.close(fig)
        for row in rows:
            table.append({"condition": condition, "learner_round": row["learner_round"], "games": row["games"],
                          "wins": row["wins"], "draws": row["draws"], "losses": row["losses"],
                          "score_percent": 100 * row["score_rate"],
                          "ci_lower_percent": 100 * row["score_interval"]["lower"],
                          "ci_upper_percent": 100 * row["score_interval"]["upper"]})
    with (output / "chart_data.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    sources = [run / name for name in ("progression.results.json", "progression.manifest.json", "verification.json", "experiment.plan.json")]
    provenance = {"experiment": run.relative_to(ROOT).as_posix(), "games": 20800,
                  "conditions": 4, "checkpoints_per_condition": 13, "games_per_checkpoint": 400,
                  "design": "Four separate unsmoothed learner-round/score curves, common 0–100% scale, pointwise paired intervals",
                  "source_sha256": {path.name: sha256(path) for path in sources},
                  "script_sha256": sha256(Path(__file__)),
                  "figure_sha256": {path.name: sha256(path) for path in files}}
    (output / "figure_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    (output / "README.md").write_text(
        "# AlphaZero progression against plain MCTS\n\n"
        "Four separate 13-checkpoint progression studies: PUCT 64/MCTS 64, PUCT 128/MCTS 128, "
        "PUCT 64/MCTS 128, and direct policy/MCTS 64. Checkpoints are C1 and C50–C600 at 50-round intervals.\n\n"
        "Each point contains 400 games: the same 200 mixed-depth openings, each in both seats, "
        "with identical paired seeds. Each curve has 5,200 games; all four contain 20,800 games. "
        "All games were replayed locally and all scores and intervals recomputed before plotting.\n\n"
        "Scores count draws as half a win. Error bars are saved pointwise 95% opening-pair percentile "
        "bootstrap intervals from 10,000 resamples; they exclude training-seed and search-seed uncertainty. "
        "Equal simulation counts do not imply equal runtime or compute.\n\n"
        "PNG exports are 300 dpi. SVG exports retain editable text. `chart_data.csv` contains the 52 plotted observations.\n\n"
        "```powershell\n.venv/Scripts/python.exe -B -m visualizations.mcts_learning_progression\n```\n",
        encoding="utf-8")
    with zipfile.ZipFile(output / "mcts_learning_progression_charts.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for path in files + [output / name for name in ("chart_data.csv", "figure_provenance.json", "README.md")]:
            archive.write(path, arcname=path.name)
    from PIL import Image
    from xml.etree import ElementTree as ET
    for path in files:
        if path.suffix == ".png":
            with Image.open(path) as picture:
                require(picture.size == (3300, 1950), "Unexpected PNG dimensions")
                picture.verify()
        else:
            svg = ET.parse(path).getroot()
            require(len(list(svg.iter("{http://www.w3.org/2000/svg}text"))) >= 20,
                    "SVG labels are not editable text")
    with zipfile.ZipFile(output / "mcts_learning_progression_charts.zip") as archive:
        require(archive.testzip() is None and len(archive.namelist()) == 11,
                "Chart archive failed verification")
    print(json.dumps({"figures": 4, "checkpoints_each": 13, "verified_games": 20800, "output": str(output)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=ROOT / "runs/checkpoint_strength/checkpoint-mcts-progression-13x400-20261006-v1")
    parser.add_argument("--output", type=Path, default=ROOT / "visualizations/figures/mcts_learning_progression_13_checkpoints")
    args = parser.parse_args()
    render(args.run, args.output)
