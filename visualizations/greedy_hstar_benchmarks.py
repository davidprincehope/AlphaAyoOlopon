"""Plot the completed plain-MCTS and mixed-opening progression experiments."""

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import zipfile

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np

from visualizations.c600_mcts_conditions import BLUE, GREY, ORANGE, ROOT, STYLE, require, sha256

PLAIN = ROOT / "runs/checkpoint_strength/checkpoint-strength-plain-mcts-20261005/results-final.jsonl"
MIXED = ROOT / "runs/checkpoint_strength/checkpoint-strength-mixed-v2-recovered-20261006"
CHECKPOINTS = [1, *range(50, 601, 50)]
SEED, RESAMPLES = 20261003, 10000


def read_lines(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def source_evidence(paths):
    return {path.relative_to(ROOT).as_posix(): sha256(path) for path in paths}


def load_plain(path):
    rows = sorted(read_lines(path), key=lambda row: row["simulations"])
    require([row["simulations"] for row in rows] == [64, 128], "Expected both plain-MCTS budgets")
    for row in rows:
        require(row["status"] == "completed" and row["games"] == 400
                and row["agent"] == "MCTS" and row["opponent"] == "GREEDY_HSTAR",
                "Unexpected plain-MCTS experiment")
        require(row["wins"] + row["draws"] + row["losses"] == 400, "Outcome total mismatch")
        require(np.isclose(row["score_rate"], (row["wins"] + 0.5 * row["draws"]) / 400),
                "Plain-MCTS score mismatch")
        for field in ("games", "wins", "draws", "losses", "score_rate"):
            require(row[field] == row["summary"]["policies"]["MCTS"][field], "Summary mismatch")
        for field in ("games", "wins", "draws", "losses"):
            require(sum(seat[field] for seat in row["by_seat"].values()) == row[field], "Seat totals mismatch")
        opponent = row["summary"]["policies"]["GREEDY_HSTAR"]
        require(opponent["wins"] == row["losses"] and opponent["losses"] == row["wins"]
                and opponent["draws"] == row["draws"]
                and np.isclose(opponent["score_rate"] + row["score_rate"], 1), "Opponent mismatch")
        require(row["opening_dataset"]["opening_count"] == 200 and row["seats"] == {"0": 200, "1": 200},
                "Expected 200 paired openings")
    require(rows[0]["opening_dataset"] == rows[1]["opening_dataset"] and rows[0]["seed"] == rows[1]["seed"],
            "Plain-MCTS budgets do not share their openings and seed")
    return rows


def bootstrap(scores, indices):
    lower, upper = np.quantile(scores[indices].mean(axis=1), [0.025, 0.975])
    return float(lower), float(upper)


def load_progression(folder):
    manifest_path = folder / "comparison.manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    require(manifest["status"] == "completed" and manifest["completed_games"] == 10400
            and manifest["completed_checkpoints"] == 26 and manifest["completed_budgets"] == [64, 128],
            "Expected the completed 10,400-game progression experiment")
    sources = {
        64: folder.parent / "checkpoint-strength-mixed-v2-recovery-20261005/results-64-final.jsonl",
        128: folder / "128/results.jsonl",
    }
    comparison_path = folder / "budget_comparison.json"
    comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
    require([row["learner_round"] for row in comparison] == CHECKPOINTS, "Comparison checkpoint mismatch")
    rng = np.random.default_rng(SEED)
    indices = rng.integers(200, size=(RESAMPLES, 200))
    curves, pair_scores, reference_signature, hashes = {}, {}, None, {}
    for budget, path in sources.items():
        rows = read_lines(path)
        require([row["learner_round"] for row in rows] == CHECKPOINTS, "Missing or duplicate checkpoints")
        config = manifest[f"configuration_{budget}"]
        require(config["simulations"] == budget and config["games"] == 400
                and config["opponent"]["name"] == "GREEDY_HSTAR", "Unexpected progression config")
        curves[budget], pair_scores[budget] = [], []
        for row in rows:
            step = row["learner_round"]
            require(row["status"] == "completed" and row["games"] == 400
                    and row["mcts_simulations"] == budget, "Incomplete checkpoint report")
            model = ROOT / f"runs/alpha_zero/modal-a100-600-20261003/inference-checkpoints/step-{step:06d}.npz"
            hashes.setdefault(step, sha256(model))
            require(row["checkpoint_sha256"] == hashes[step], "Checkpoint hash mismatch")
            records = sorted(row["matches"], key=lambda record: record["game_id"])
            require([record["game_id"] for record in records] == list(range(400)), "Game ID mismatch")
            counts = Counter(record["policy_outcome"]["ALPHAZERO"] for record in records)
            for outcome, field in (("win", "wins"), ("draw", "draws"), ("loss", "losses")):
                require(counts[outcome] == row[field], "Checkpoint outcomes mismatch")
            require(not any(record["truncated"] for record in records), "Unexpected adjudicated game")
            signature, scores = [], []
            for i in range(200):
                a, b = records[2 * i:2 * i + 2]
                require(a["opening_pair_id"] == b["opening_pair_id"] == i
                        and a["player_0_policy"] == b["player_1_policy"] == "ALPHAZERO"
                        and a["player_1_policy"] == b["player_0_policy"] == "OPPONENT"
                        and all(a[key] == b[key] for key in (
                            "opening_id", "seed", "opening_state_sha256", "opening_actions")),
                        "Incomplete or mismatched opening pair")
                require(a["opening_dataset_sha256"] == b["opening_dataset_sha256"]
                        == manifest["opening_content_sha256"], "Opening dataset mismatch")
                signature.append((a["opening_id"], a["seed"], a["opening_state_sha256"]))
                for record in (a, b):
                    expected = {"win": 1, "draw": 0, "loss": -1}[record["policy_outcome"]["ALPHAZERO"]]
                    require(record["policy_returns"]["ALPHAZERO"] == expected, "Outcome/return mismatch")
                scores.append(np.mean([(record["policy_returns"]["ALPHAZERO"] + 1) / 2 for record in (a, b)]))
            if reference_signature is None:
                reference_signature = signature
            require(signature == reference_signature, "Checkpoint/budget opening schedules differ")
            scores = np.asarray(scores)
            require(np.isclose(scores.mean(), row["score_rate"]), "Recomputed score mismatch")
            ci = bootstrap(scores, indices)
            curves[budget].append({"learner_round": step, "simulations": budget, "games": 400,
                                   "wins": row["wins"], "draws": row["draws"], "losses": row["losses"],
                                   "score_percent": 100 * row["score_rate"],
                                   "ci_lower_percent": 100 * ci[0], "ci_upper_percent": 100 * ci[1]})
            pair_scores[budget].append(scores)
    differences = []
    for i, step in enumerate(CHECKPOINTS):
        delta = pair_scores[128][i] - pair_scores[64][i]
        lower, upper = bootstrap(delta, indices)
        for budget in (64, 128):
            require(np.isclose(curves[budget][i]["score_percent"] / 100,
                               comparison[i][f"score_rate_{budget}"]), "Saved budget comparison mismatch")
        require(np.isclose(100 * delta.mean(), comparison[i]["score_rate_change_percentage_points"]),
                "Saved budget change mismatch")
        differences.append({"learner_round": step, "score_change_pp": float(100 * delta.mean()),
                            "ci_lower_pp": 100 * lower, "ci_upper_pp": 100 * upper})
    evidence = {
        "source_sha256": source_evidence([manifest_path, comparison_path, *sources.values()]),
        "checkpoint_sha256": hashes, "completed_games_verified": 10400,
        "same_opening_seat_seed_schedule": True, "outcomes_and_scores_recomputed": True,
        "saved_budget_comparison_matches": True,
        "uncertainty": {"computed_for_these_figures": True, "method": "opening-pair percentile bootstrap",
                        "confidence_level": 0.95, "resamples": RESAMPLES, "seed": SEED,
                        "opening_pairs": 200, "interval_type": "pointwise",
                        "scope": "Sampled positions; excludes training-seed and search-seed uncertainty",
                        "difference_method": "Resample differences for the same opening pairs across budgets"},
    }
    return curves, differences, evidence


def save_figure(fig, output, stem):
    for extension in ("png", "svg"):
        fig.savefig(output / f"{stem}.{extension}", dpi=300)
    plt.close(fig)


def render_plain(rows, output):
    with plt.rc_context(STYLE):
        fig = plt.figure(figsize=(10.5, 5.8))
        fig.text(0.08, 0.94, "Plain MCTS versus greedy H*", fontsize=18, weight="bold", va="top")
        fig.text(0.08, 0.875, "UCT with random rollouts  |  64 versus 128 simulations per move",
                 fontsize=11, color="#444444", va="top")
        fig.text(0.08, 0.825, "400 games per budget  |  The same 200 mixed-depth openings, each played in both seats",
                 fontsize=10, color="#666666", va="top")
        outcomes = fig.add_axes((0.13, 0.31, 0.46, 0.38))
        outcomes.set_title("Game outcomes", loc="left", fontsize=12, pad=17, weight="bold")
        for row, y in zip(rows, (1, 0)):
            left = 0
            for field, color in (("wins", BLUE), ("draws", GREY), ("losses", ORANGE)):
                share = 100 * row[field] / 400
                outcomes.barh(y, share, left=left, height=0.49, color=color, zorder=3)
                outcomes.text(left + share / 2, y, f"{share:.2f}%\n{row[field]} games",
                              ha="center", va="center", color="white", fontsize=10)
                left += share
        outcomes.set(xlim=(0, 100), ylim=(-0.6, 1.7), xlabel="Share of games")
        outcomes.set_yticks([1, 0], ["64 sims", "128 sims"])
        outcomes.set_xticks([0, 25, 50, 75, 100])
        outcomes.xaxis.set_major_formatter(PercentFormatter(100, decimals=0))
        outcomes.spines["left"].set_visible(False)
        outcomes.tick_params(axis="y", length=0, pad=8)
        handles = [plt.Rectangle((0, 0), 1, 1, color=color) for color in (BLUE, GREY, ORANGE)]
        outcomes.legend(handles, ["MCTS wins", "Draws", "Greedy H* wins"], frameon=False,
                        loc="upper left", bbox_to_anchor=(0, -0.25), ncol=3,
                        fontsize=9.5, handlelength=1.2, columnspacing=1.1, borderaxespad=0)
        score_ax = fig.add_axes((0.765, 0.31, 0.19, 0.38))
        score_ax.set_title("MCTS score", loc="left", fontsize=12, pad=17, weight="bold")
        score_ax.axvline(50, linestyle=(0, (4, 4)), color="#999999", linewidth=1)
        for row, y in zip(rows, (1, 0)):
            score = 100 * row["score_rate"]
            score_ax.plot(score, y, "o", color=BLUE, markersize=8)
            score_ax.text(score, y + 0.21, f"{score:.2f}%", ha="center", va="bottom",
                          fontsize=12, color=BLUE, weight="bold",
                          bbox={"facecolor": "white", "edgecolor": "none", "pad": 1})
        score_ax.set(xlim=(0, 100), ylim=(-0.6, 1.7), xlabel="Score")
        score_ax.set_xticks([0, 50, 100])
        score_ax.xaxis.set_major_formatter(PercentFormatter(100, decimals=0))
        score_ax.set_yticks([1, 0], ["64 sims", "128 sims"])
        score_ax.spines["left"].set_visible(False)
        score_ax.tick_params(axis="y", length=0, pad=8)
        fig.text(0.08, 0.105, "Score = (wins + 0.5 × draws) / games. Dashed line marks an equal score of 50%.",
                 fontsize=9.5, color="#555555")
        fig.text(0.08, 0.06, "Observed results; the supplied summary contains no opening-pair records or confidence intervals.",
                 fontsize=9.5, color="#555555")
        save_figure(fig, output, "plain_mcts_vs_greedy_hstar")


def render_progression(curves, differences, output):
    with plt.rc_context(STYLE):
        fig = plt.figure(figsize=(11, 8))
        fig.text(0.09, 0.95, "Checkpoint strength against greedy H*", fontsize=18, weight="bold", va="top")
        fig.text(0.09, 0.90, "64 versus 128 neural PUCT simulations per move", fontsize=11, color="#444444", va="top")
        fig.text(0.09, 0.855, "13 checkpoints  |  400 games per checkpoint and budget  |  10,400 games total",
                 fontsize=10, color="#666666", va="top")
        ax = fig.add_axes((0.09, 0.445, 0.82, 0.33))
        for budget, color, marker in ((64, BLUE, "o"), (128, ORANGE, "s")):
            rows = curves[budget]
            scores = np.array([row["score_percent"] for row in rows])
            lower = np.array([row["ci_lower_percent"] for row in rows])
            upper = np.array([row["ci_upper_percent"] for row in rows])
            ax.errorbar(CHECKPOINTS, scores, yerr=[scores - lower, upper - scores], fmt="none",
                        ecolor=color, alpha=0.40, capsize=3, elinewidth=1.1)
            ax.plot(CHECKPOINTS, scores, marker=marker, color=color, linewidth=1.6,
                    markersize=4.5, label=f"{budget} simulations")
            ax.text(607, scores[-1], f"{scores[-1]:.2f}%", color=color, fontsize=10, va="center")
        ax.axhline(50, linestyle=(0, (4, 4)), color="#999999", linewidth=1)
        ax.set(xlim=(-8, 654), ylim=(45, 85), ylabel="Score against greedy H*")
        ax.set_yticks([50, 60, 70, 80])
        ax.yaxis.set_major_formatter(PercentFormatter(100, decimals=0))
        ax.set_xticks([1, 100, 200, 300, 400, 500, 600])
        ax.tick_params(labelbottom=False, length=3, color="#555555")
        ax.set_axisbelow(True)
        ax.grid(axis="y", color="#E6E6E6", linewidth=0.7)
        ax.legend(frameon=False, loc="upper right", ncol=2, fontsize=10)

        delta_ax = fig.add_axes((0.09, 0.22, 0.82, 0.15), sharex=ax)
        delta_ax.set_title("Score change from using 128 rather than 64 simulations", loc="left",
                           fontsize=11, weight="bold", pad=12)
        changes = np.array([row["score_change_pp"] for row in differences])
        lower = np.array([row["ci_lower_pp"] for row in differences])
        upper = np.array([row["ci_upper_pp"] for row in differences])
        delta_ax.errorbar(CHECKPOINTS, changes, yerr=[changes - lower, upper - changes],
                          fmt="o", color="#333333", ecolor="#777777", capsize=3,
                          markersize=4, elinewidth=1.1)
        delta_ax.axhline(0, linestyle=(0, (4, 4)), color="#999999", linewidth=1)
        delta_ax.set(ylim=(-4, 9), xlabel="Learner round", ylabel="Score change\n(percentage points)")
        delta_ax.set_yticks([-4, 0, 4, 8])
        delta_ax.set_axisbelow(True)
        delta_ax.grid(axis="y", color="#E6E6E6", linewidth=0.7)
        delta_ax.tick_params(length=3, color="#555555")
        fig.text(0.09, 0.105, "Scores count draws as half a win. Every point uses the same 200 openings, each played in both seats.",
                 fontsize=9.5, color="#555555")
        fig.text(0.09, 0.065, "Pointwise 95% intervals bootstrap whole opening pairs; budget differences use matched pairs.",
                 fontsize=9.5, color="#555555")
        fig.text(0.09, 0.025, "Intervals exclude training-seed and search-seed uncertainty. Lines connect tested checkpoints without smoothing.",
                 fontsize=9.5, color="#555555")
        save_figure(fig, output, "mixed_opening_progression_64_128")


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def render(plain_path, mixed_folder, output):
    plain = load_plain(plain_path)
    curves, differences, evidence = load_progression(mixed_folder)
    require(plain[0]["opening_dataset"]["content_sha256"] ==
            json.loads((mixed_folder / "comparison.manifest.json").read_text(encoding="utf-8"))["opening_content_sha256"],
            "The two experiments use different opening datasets")
    output.mkdir(parents=True, exist_ok=True)
    render_plain(plain, output)
    render_progression(curves, differences, output)
    write_csv(output / "plain_mcts_chart_data.csv", [
        {"simulations": row["simulations"], "games": row["games"], "wins": row["wins"],
         "draws": row["draws"], "losses": row["losses"], "score_percent": 100 * row["score_rate"]}
        for row in plain])
    write_csv(output / "progression_chart_data.csv", curves[64] + curves[128])
    write_csv(output / "budget_difference_chart_data.csv", differences)
    figures = [output / f"{stem}.{extension}" for stem in (
        "plain_mcts_vs_greedy_hstar", "mixed_opening_progression_64_128") for extension in ("png", "svg")]
    provenance = {
        "plain_mcts": {"source_sha256": source_evidence([plain_path]), "completed_games": 800,
                       "summary_outcomes_scores_and_seat_totals_checked": True,
                       "confidence_intervals": None,
                       "uncertainty_note": "No opening-pair records in the supplied summary; descriptive results only"},
        "progression": evidence, "script_sha256": sha256(Path(__file__)),
        "matplotlib": plt.matplotlib.__version__, "numpy": np.__version__,
        "figure_sha256": {path.name: sha256(path) for path in figures},
    }
    (output / "figure_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    (output / "README.md").write_text(
        "# Greedy H* benchmark figures\n\n"
        "- `plain_mcts_vs_greedy_hstar`: outcomes and MCTS scores at 64 and 128 simulations (800 games). "
        "Observed results only: the supplied summary has no game-pair records or uncertainty intervals.\n"
        "- `mixed_opening_progression_64_128`: the completed 13-checkpoint curves at both budgets (10,400 games), "
        "plus the matched-opening score change at each checkpoint.\n\n"
        "The progression figure computes new pointwise 95% percentile bootstrap intervals, "
        "using 10,000 resamples of 200 whole opening pairs and seed 20261003. "
        "The lower panel resamples per-opening score differences across budgets. "
        "Intervals describe sampled-position variation and exclude training-seed and search-seed uncertainty.\n\n"
        "Each opening is played in both seats; score counts a draw as half a win. "
        "These are continuation-strength experiments on fixed random prefixes. "
        "Changing the evaluation budget does not change the trained checkpoints.\n\n"
        "The recovery manifest certifies completion. The completed consolidated 64-budget reports are "
        "read from the sibling `checkpoint-strength-mixed-v2-recovery-20261005/results-64-final.jsonl`, "
        "and the 128-budget reports from `128/results.jsonl`. All 10,400 embedded game outcomes, "
        "opening pairs, checkpoint hashes and saved budget comparisons were checked before plotting.\n\n"
        "PNG exports are 300 dpi; SVG exports retain editable text. CSVs contain plotted values, "
        "including the newly calculated confidence intervals. Source and output hashes are in `figure_provenance.json`.\n\n"
        "Reproduce from the repository root:\n\n```powershell\n"
        ".venv/Scripts/python.exe -B -m visualizations.greedy_hstar_benchmarks\n```\n",
        encoding="utf-8")
    supporting = [output / name for name in ("plain_mcts_chart_data.csv", "progression_chart_data.csv",
                                            "budget_difference_chart_data.csv", "figure_provenance.json", "README.md")]
    with zipfile.ZipFile(output / "greedy_hstar_benchmark_charts.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for path in figures + supporting:
            archive.write(path, arcname=path.name)
    print(json.dumps({"figures": 2, "plain_mcts_games": 800, "progression_games_verified": 10400,
                      "plain_mcts_scores": [row["score_rate"] for row in plain], "output": str(output)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plain-results", type=Path, default=PLAIN)
    parser.add_argument("--progression-folder", type=Path, default=MIXED)
    parser.add_argument("--output", type=Path, default=ROOT / "visualizations/figures/greedy_hstar_benchmarks")
    args = parser.parse_args()
    render(args.plain_results.resolve(), args.progression_folder.resolve(), args.output.resolve())
