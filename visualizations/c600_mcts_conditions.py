"""Plot the four completed C600/MCTS conditions as individual PNG/SVG figures."""

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import os
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "visualizations/.matplotlib"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np

CONDITIONS = (
    ("c600_64_mcts_64", "checkpoint-c600-vs-mcts-64-20261005", "puct", 64, 64),
    ("c600_128_mcts_128", "checkpoint-c600-vs-mcts-128-20261005", "puct", 128, 128),
    ("c600_64_mcts_128", "checkpoint-c600-64-vs-mcts-128-20261005", "puct", 64, 128),
    ("c600_policy_mcts_64", "checkpoint-c600-policy-vs-mcts-64-20261005", "policy", 0, 64),
)
BLUE, GREY, ORANGE = "#0072B2", "#8A8A8A", "#D55E00"
STYLE = {
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 11, "axes.labelsize": 11, "xtick.labelsize": 10,
    "ytick.labelsize": 10, "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#555555", "axes.linewidth": 0.8,
    "svg.fonttype": "none", "savefig.facecolor": "white",
}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_condition(source_root, condition):
    slug, directory, mode, neural_budget, mcts_budget = condition
    folder = source_root / directory
    report_path, games_path = folder / "results.json", folder / "results.games.jsonl"
    manifest_path = folder / "results.manifest.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = [json.loads(line) for line in games_path.read_text(encoding="utf-8").splitlines()
               if line.strip()]
    config = report["config"]
    require(report["status"] == manifest["status"] == "completed", f"Incomplete condition: {slug}")
    require(config == manifest["config"], f"Configuration mismatch: {slug}")
    require(config["checkpoint"] == 600 and config.get("neural_mode", "puct") == mode
            and config["simulations"] == neural_budget
            and (config.get("mcts_simulations") or neural_budget) == mcts_budget,
            f"Unexpected decision rule or budget: {slug}")
    require(len(records) == report["games"] == manifest["completed_games"] == 400,
            f"Expected 400 games: {slug}")
    require(sha256(games_path) == manifest["games_sha256"]
            and sha256(report_path) == manifest["summary_sha256"], f"Source hash mismatch: {slug}")
    require(len({r["game_id"] for r in records}) == 400, f"Duplicate games: {slug}")
    require(not any(r["truncated"] for r in records), f"Unexpected truncated game: {slug}")
    pairs = defaultdict(list)
    for record in records:
        pairs[record["opening_pair_id"]].append(record)
    require(len(pairs) == 200, f"Expected 200 opening pairs: {slug}")
    for pair in pairs.values():
        require(len(pair) == 2, f"Incomplete opening pair: {slug}")
        a, b = pair
        require(all(a[key] == b[key] for key in (
            "opening_id", "seed", "opening_state_sha256", "opening_dataset_sha256"))
            and a["player_0_policy"] == b["player_1_policy"]
            and a["player_1_policy"] == b["player_0_policy"], f"Seat/pair mismatch: {slug}")
    for policy in ("C600", "MCTS"):
        counts = Counter(r["policy_outcome"][policy] for r in records)
        stats, interval = report["policies"][policy], report["score_intervals"][policy]
        for outcome, field in (("win", "wins"), ("draw", "draws"), ("loss", "losses")):
            require(counts[outcome] == stats[field], f"Outcome mismatch: {slug}/{policy}/{field}")
        score = (counts["win"] + 0.5 * counts["draw"]) / 400
        require(np.isclose(score, stats["score_rate"]), f"Score mismatch: {slug}/{policy}")
        require(interval["method"] == "opening-pair percentile bootstrap"
                and interval["confidence_level"] == 0.95 and interval["opening_pairs"] == 200,
                f"Unexpected uncertainty method: {slug}")
        # Independently reproduce the saved interval, preserving raw pair order.
        pair_scores = np.array([np.mean([(r["policy_returns"][policy] + 1) / 2 for r in pair])
                                for pair in pairs.values()])
        rng = np.random.default_rng(interval["seed"])
        means = pair_scores[rng.integers(200, size=(interval["resamples"], 200))].mean(axis=1)
        lower, upper = np.quantile(means, [0.025, 0.975])
        require(np.allclose([score, lower, upper],
                            [interval["score_rate"], interval["lower"], interval["upper"]],
                            atol=1e-12, rtol=0), f"Bootstrap mismatch: {slug}/{policy}")
    signature = [(r["game_id"], r["opening_id"], r["opening_state_sha256"], r["seed"],
                  r["player_0_policy"], r["player_1_policy"])
                 for r in sorted(records, key=lambda r: r["game_id"])]
    evidence = {
        "condition": slug, "source_directory": folder.relative_to(ROOT).as_posix(),
        "neural_mode": mode, "c600_simulations": neural_budget, "mcts_simulations": mcts_budget,
        "games": 400, "opening_pairs": 200, "policies": report["policies"],
        "score_intervals": report["score_intervals"],
        "checkpoint_sha256": manifest["checkpoints"][0]["checkpoint_sha256"],
        "opening_dataset_sha256": report["opening_dataset"]["content_sha256"],
        "source_sha256": {p.name: sha256(p) for p in (report_path, games_path, manifest_path)},
        "verification": {"outcomes_recomputed": True, "scores_recomputed": True,
                         "source_hashes_match": True, "opening_pairs_valid": True,
                         "bootstrap_intervals_reproduced": True, "all_games_natural": True},
    }
    return evidence, signature


def render_condition(data, output):
    mode = data["neural_mode"]
    c600_label = ("C600 policy alone (no search)" if mode == "policy"
                  else f"C600 PUCT: {data['c600_simulations']} simulations/move")
    subtitle = f"{c600_label}  |  MCTS UCT: {data['mcts_simulations']} simulations/move"
    stats = data["policies"]["C600"]
    with plt.rc_context(STYLE):
        fig = plt.figure(figsize=(10, 5.8))
        fig.text(0.08, 0.94, "C600 versus vanilla MCTS", fontsize=18, weight="bold", va="top")
        fig.text(0.08, 0.875, subtitle, fontsize=11, color="#444444", va="top")
        fig.text(0.08, 0.825, "400 games  |  200 mixed-depth openings, each played in both seats",
                 fontsize=10, color="#666666", va="top")
        ax = fig.add_axes((0.08, 0.25, 0.43, 0.45))
        ax.set_title("Game outcomes", loc="left", fontsize=12, pad=17, weight="bold")
        values = [stats["wins"], stats["draws"], stats["losses"]]
        percentages = [100 * value / 400 for value in values]
        bars = ax.bar([0, 1, 2], percentages, width=0.58, color=[BLUE, GREY, ORANGE], zorder=3)
        ax.set(ylim=(0, 100), xlim=(-0.6, 2.6), ylabel="Share of games")
        ax.set_xticks([0, 1, 2], ["C600 wins", "Draws", "MCTS wins"])
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.yaxis.set_major_formatter(PercentFormatter(100, decimals=0))
        ax.set_axisbelow(True)
        ax.grid(axis="y", color="#E6E6E6", linewidth=0.7)
        ax.tick_params(axis="x", length=0, pad=10)
        ax.tick_params(axis="y", length=3, color="#555555")
        for bar, count, percent in zip(bars, values, percentages):
            ax.text(bar.get_x() + bar.get_width() / 2, percent + 3,
                    f"{percent:.2f}%\n{count} games", ha="center", va="bottom", fontsize=10)

        score_ax = fig.add_axes((0.665, 0.30, 0.29, 0.40))
        score_ax.set_title("Score with 95% interval", loc="left", fontsize=12,
                           pad=17, weight="bold")
        score_ax.axvline(50, linestyle=(0, (4, 4)), color="#999999", linewidth=1)
        for policy, y, color in (("C600", 1, BLUE), ("MCTS", 0, ORANGE)):
            score = 100 * data["policies"][policy]["score_rate"]
            interval = data["score_intervals"][policy]
            low, high = 100 * interval["lower"], 100 * interval["upper"]
            score_ax.errorbar(score, y, xerr=[[score - low], [high - score]], fmt="o",
                              color=color, markersize=7, elinewidth=2, capsize=5, capthick=1.5)
            score_ax.text(score, y + 0.19, f"{score:.2f}%", ha="center", va="bottom",
                          fontsize=12, color=color, weight="bold",
                          bbox={"facecolor": "white", "edgecolor": "none", "pad": 1})
        score_ax.set(xlim=(0, 100), ylim=(-0.5, 1.6), xlabel="Score (draw = half a win)")
        score_ax.set_xticks([0, 25, 50, 75, 100])
        score_ax.xaxis.set_major_formatter(PercentFormatter(100, decimals=0))
        score_ax.set_yticks([1, 0], ["C600", "MCTS"])
        score_ax.spines["left"].set_visible(False)
        score_ax.tick_params(axis="y", length=0, pad=8)
        score_ax.tick_params(axis="x", length=3, color="#555555")
        interval = data["score_intervals"]["C600"]
        fig.text(0.665, 0.18,
                 f"C600 95% interval: {100 * interval['lower']:.2f}%–{100 * interval['upper']:.2f}%",
                 fontsize=10, color="#555555")
        fig.text(0.08, 0.105,
                 "Score = (wins + 0.5 × draws) / games. Dashed line marks an equal score of 50%.",
                 fontsize=9.5, color="#555555")
        fig.text(0.08, 0.06,
                 "95% intervals resample whole opening pairs; they exclude training-seed and search-seed uncertainty.",
                 fontsize=9.5, color="#555555")
        for extension in ("png", "svg"):
            fig.savefig(output / f"{data['condition']}.{extension}", dpi=300)
        plt.close(fig)


def render(source_root, output):
    evidence, reference_signature = [], None
    for condition in CONDITIONS:
        data, signature = load_condition(source_root, condition)
        if reference_signature is None:
            reference_signature = signature
        require(signature == reference_signature, "Conditions do not use identical opening/seat/seed schedules")
        if evidence:
            require(data["checkpoint_sha256"] == evidence[0]["checkpoint_sha256"]
                    and data["opening_dataset_sha256"] == evidence[0]["opening_dataset_sha256"],
                    "Conditions do not share the same checkpoint and opening dataset")
        evidence.append(data)
    output.mkdir(parents=True, exist_ok=True)
    for data in evidence:
        render_condition(data, output)
    with (output / "chart_data.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = ["condition", "c600_simulations", "mcts_simulations", "agent", "games", "wins",
                  "draws", "losses", "score_percent", "ci_lower_percent", "ci_upper_percent"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for data in evidence:
            for policy in ("C600", "MCTS"):
                stats, interval = data["policies"][policy], data["score_intervals"][policy]
                writer.writerow({"condition": data["condition"], "c600_simulations": data["c600_simulations"],
                                 "mcts_simulations": data["mcts_simulations"], "agent": policy,
                                 **{key: stats[key] for key in ("games", "wins", "draws", "losses")},
                                 "score_percent": 100 * stats["score_rate"],
                                 "ci_lower_percent": 100 * interval["lower"],
                                 "ci_upper_percent": 100 * interval["upper"]})
    figure_files = [output / f"{data['condition']}.{ext}" for data in evidence for ext in ("png", "svg")]
    provenance = {
        "design": "Four individual figures with common 0–100% axes: outcome shares and agent scores with 95% intervals.",
        "conditions": evidence, "same_checkpoint_openings_seats_and_seeds": True,
        "script_sha256": sha256(Path(__file__)), "matplotlib": matplotlib.__version__,
        "numpy": np.__version__, "figure_sha256": {p.name: sha256(p) for p in figure_files},
    }
    (output / "figure_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    readme = ["# Individual C600 versus MCTS charts", "",
              "Each figure shows outcome shares and both agents' scores, using common 0–100% scales.",
              "Score counts a draw as half a win; outcome percentages are the observed shares of all 400 games.",
              "Intervals are the saved 95% percentile bootstrap intervals over 200 whole opening pairs (10,000 resamples).",
              "They exclude training-seed and search-seed uncertainty. Equal simulation counts do not mean equal runtime or compute.",
              "", "PNG: 300 dpi. SVG: vector graphics with editable text.", "",
              "| Condition (C600 / MCTS) | C600 score | C600 95% interval | Files |",
              "| --- | ---: | ---: | --- |"]
    for data in evidence:
        label = ("Policy alone / 64" if data["neural_mode"] == "policy"
                 else f"{data['c600_simulations']} / {data['mcts_simulations']}")
        score, ci, slug = data["policies"]["C600"]["score_rate"], data["score_intervals"]["C600"], data["condition"]
        readme.append(f"| {label} | {100 * score:.2f}% | {100 * ci['lower']:.2f}%–{100 * ci['upper']:.2f}% | "
                      f"[PNG]({slug}.png), [SVG]({slug}.svg) |")
    readme += ["", "All 1,600 game outcomes, paired seat assignments, scores, source hashes, and saved confidence intervals",
               "were checked before plotting. No new games or training were run.", "",
               "Reproduce from the repository root:", "",
               "```powershell", ".venv/Scripts/python.exe -B -m visualizations.c600_mcts_conditions", "```", "",
               "Source result locations and SHA-256 hashes are recorded in `figure_provenance.json`."]
    (output / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")
    with zipfile.ZipFile(output / "c600_mcts_individual_charts.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for path in figure_files + [output / name for name in ("chart_data.csv", "figure_provenance.json", "README.md")]:
            archive.write(path, arcname=path.name)
    print(json.dumps({"figures": 4, "verified_games": 1600, "output": str(output),
                      "c600_scores": [data["policies"]["C600"]["score_rate"] for data in evidence]}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=ROOT / "runs/checkpoint_strength")
    parser.add_argument("--output", type=Path, default=ROOT / "visualizations/figures/c600_vs_mcts_conditions")
    args = parser.parse_args()
    render(args.source_root.resolve(), args.output.resolve())
