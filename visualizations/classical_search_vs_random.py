"""Figure 4: recorded classical-agent scores and cap shares against random play."""

import argparse
from collections import Counter, defaultdict
import csv
from decimal import Decimal, ROUND_HALF_UP
import hashlib
import json
import math
import os
from pathlib import Path
from xml.etree import ElementTree

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "visualizations/.matplotlib"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from PIL import Image

BLUE, ORANGE = "#0072B2", "#D55E00"
STYLE = {
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 11, "axes.labelsize": 11, "xtick.labelsize": 10,
    "ytick.labelsize": 10.5, "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#555555", "axes.linewidth": 0.8,
    "svg.fonttype": "none", "savefig.facecolor": "white",
}
AUDIT = ROOT / "docs/results/classical_search/source_audit.json"
NOTES = ROOT / "docs/results/classical_search/README.md"
CONDITIONS = [
    ("greedy_hstar", "greedy", "Python", None),
    *[(f"minimax_depth_{depth}", "minimax", "Python" if depth <= 3 else "C++", depth)
      for depth in range(1, 6)],
    ("mcts_1000", "mcts", "C++", None),
]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def percentage_label(value):
    rounded = Decimal(str(value)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    return f"{rounded:.2f}%"


def recount_log(name, kind, audit, metadata):
    path = ROOT / audit["log"]
    raw = path.read_bytes()
    raw_hash = hashlib.sha256(raw).hexdigest()
    require(raw_hash == audit["log_sha256"], f"Audited raw log changed: {name}")
    stored_hash_mode = None
    if metadata.get("log_sha256"):
        lf_hash = hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest()
        require(metadata["log_sha256"] in (raw_hash, lf_hash), f"Stored log hash mismatch: {name}")
        stored_hash_mode = "exact bytes" if metadata["log_sha256"] == raw_hash else "LF-normalized bytes"
    outcomes, seats, terms = Counter(), Counter(), Counter()
    by_term = defaultdict(Counter)
    for index, line in enumerate(raw.splitlines()):
        record = json.loads(line)
        require(record["game_id"] == index, f"Unordered game IDs: {name}")
        if kind == "greedy":
            seat = 0 if record["player_0_policy"] == "GREEDY_HSTAR" else 1
            outcome = record["policy_outcome"]["GREEDY_HSTAR"]
        else:
            seat, outcome = record[kind + "_seat"], record[kind + "_outcome"]
        term = record["termination_reason"]
        require(record["truncated"] == (term == "action_limit"), f"Cap flag mismatch: {name}")
        if term == "action_limit":
            require(record["game_length"] == 300, f"Unexpected action cap: {name}")
            require(outcome == "win" and record["final_captured"][seat]
                    > record["final_captured"][1 - seat], f"Cap adjudication mismatch: {name}")
        outcomes[outcome] += 1
        seats[str(seat)] += 1
        terms[term] += 1
        by_term[term][outcome] += 1
    require(dict(seats) == audit["seats"], f"Seat balance mismatch: {name}")
    require(dict(terms) == audit["termination_reasons"], f"Termination recount mismatch: {name}")
    require({term: dict(counts) for term, counts in by_term.items()}
            == audit["outcomes_by_termination"], f"Termination/outcome recount mismatch: {name}")
    for singular, plural in (("win", "wins"), ("draw", "draws"), ("loss", "losses")):
        require(outcomes[singular] == audit[plural], f"Outcome recount mismatch: {name}")
    return {"log": audit["log"], "log_sha256": raw_hash,
            "games_recounted": sum(outcomes.values()), "stored_hash_match_mode": stored_hash_mode}


def load_results(verify_logs=False):
    evidence = json.loads(AUDIT.read_text(encoding="utf-8"))["large_benchmarks"]
    rows, sources, recounts = [], [AUDIT, NOTES], {}
    for name, kind, implementation, depth in CONDITIONS:
        audit = evidence[name]
        if kind == "minimax":
            filename = f"depth_{depth}_summary.json" if depth <= 3 else f"depth_{depth}_games_10000_cpp_summary.json"
            expected_source = f"experiments/minimax_depth_vs_random/results/{filename}"
        elif kind == "greedy":
            expected_source = "experiments/agent_benchmark/results/random_vs_greedy_hstar_10000.json"
        else:
            expected_source = "experiments/mcts_vs_random/results/mcts_uct_1000sims_games_10000_cpp_summary.json"
        require(audit["summary"] == expected_source, f"Expected the individual source summary: {name}")
        source = ROOT / expected_source
        metadata = json.loads(source.read_text(encoding="utf-8"))
        sources.append(source)
        require(metadata["status"] == "completed" and metadata["max_actions"] == 300,
                f"Expected a completed 300-action-cap benchmark: {name}")
        aggregate = (metadata["summary"]["policies"]["GREEDY_HSTAR"]
                     if kind == "greedy" else metadata.get("summary", metadata))
        games = aggregate["games"]
        require(games == metadata["games"] == audit["games"] == 10_000,
                f"Expected 10,000 games: {name}")
        wins, draws, losses = [aggregate[field] for field in ("wins", "draws", "losses")]
        require(all(type(value) is int and value >= 0 for value in (wins, draws, losses))
                and wins + draws + losses == games, f"Invalid W/D/L counts: {name}")
        score = (wins + 0.5 * draws) / games
        require(math.isclose(score, aggregate["score_rate"], abs_tol=1e-12)
                and math.isclose(wins / games, aggregate["win_rate"], abs_tol=1e-12),
                f"Stored outcome rates do not match the counts: {name}")
        require(audit["stored_summary_matches_recount"] is True
                and audit["seats"] == {"0": 5000, "1": 5000}, f"Invalid source audit: {name}")
        for field in ("wins", "draws", "losses", "score_rate", "win_rate"):
            require(aggregate[field] == audit[field], f"Summary differs from source audit: {name}: {field}")
        terms = audit["termination_reasons"]
        require(sum(terms.values()) == games, f"Incomplete termination accounting: {name}")
        if kind == "greedy":
            require(metadata["starting_position"] == "standard_initial_state"
                    and metadata["summary"]["termination_reasons"] == terms,
                    "Greedy start or termination counts differ from the audit")
            cap_games = aggregate["action_limits"]
            budget, label = "1-ply successor evaluation", "Greedy H*\n1-ply · Python"
        elif kind == "minimax":
            require(metadata["depth"] == depth, f"Wrong minimax depth: {name}")
            if implementation == "Python":
                require(metadata["starting_position"] == "standard_initial_state"
                        and aggregate["termination_reasons"] == terms,
                        f"Minimax start or termination counts differ: {name}")
                cap_games = aggregate["termination_reasons"].get("action_limit", 0)
            else:
                require("C++ AlphaBetaSearch" in metadata["engine"], f"Unexpected native engine: {name}")
                cap_games = aggregate["truncated_games"]
            budget, label = f"Depth {depth}", f"Minimax + H* · depth {depth}\nHistorical · {implementation}"
        else:
            require(metadata["simulations_per_move"] == 1000 and metadata["rollouts_per_leaf"] == 1
                    and "C++ MCTSBot (UCT)" in metadata["engine"], "Unexpected MCTS decision rule")
            cap_games = aggregate["truncated_games"]
            budget, label = "1,000 UCT simulations; one random rollout per leaf", "UCT MCTS\n1,000 simulations · C++"
        require(cap_games == terms.get("action_limit", 0), f"Cap count differs from audit: {name}")
        audited_outcomes = Counter()
        for term, counts in audit["outcomes_by_termination"].items():
            require(sum(counts.values()) == terms[term], f"Termination outcome count mismatch: {name}")
            audited_outcomes.update(counts)
        require(audited_outcomes == Counter(win=wins, draw=draws, loss=losses),
                f"Audited outcome totals differ: {name}")
        capped_outcomes = audit["outcomes_by_termination"].get("action_limit", {})
        require(not cap_games or (kind == "minimax" and capped_outcomes == {"win": cap_games}),
                f"Expected all capped minimax games to be credited as wins: {name}")
        if verify_logs:
            recounts[name] = recount_log(name, kind, audit, metadata)
        rows.append({"condition": name, "agent": "Minimax + H*" if kind == "minimax" else "Greedy H*" if kind == "greedy" else "UCT MCTS",
                     "implementation": implementation, "historical_minimax": kind == "minimax",
                     "budget_per_move": budget, "games": games, "wins": wins, "draws": draws, "losses": losses,
                     "score_percent": (100 * wins + 50 * draws) / games,
                     "cap_adjudicated_games": cap_games, "cap_percent": 100 * cap_games / games,
                     "cap_adjudicated_wins": capped_outcomes.get("win", 0),
                     "source_summary": expected_source, "plot_label": label})
    return rows, sources, recounts


def render(output, verify_logs=False):
    rows, sources, recounts = load_results(verify_logs)
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    figure_name = "classical_search_vs_random"
    with plt.rc_context(STYLE):
        fig = plt.figure(figsize=(12.4, 8.2))
        fig.text(0.06, 0.955, "Classical search against random play", fontsize=20,
                 weight="bold", va="top")
        fig.text(0.06, 0.9,
                 "Standard initial board  |  10,000 games per condition  |  5,000 games in each seat",
                 fontsize=11, color="#555555", va="top")
        score_ax = fig.add_axes((0.255, 0.31, 0.325, 0.49))
        cap_ax = fig.add_axes((0.66, 0.31, 0.30, 0.49), sharey=score_ax)
        score_ax.set_title("a  Reported score rate", loc="left", fontsize=13,
                           weight="bold", pad=15)
        cap_ax.set_title("b  Cap-adjudicated games", loc="left", fontsize=13,
                         weight="bold", pad=15)
        for ax, measure, color in ((score_ax, "score_percent", BLUE), (cap_ax, "cap_percent", ORANGE)):
            ax.axhspan(0.5, 5.5, facecolor="#F2F2F2", zorder=0)
            ax.barh(range(len(rows)), [row[measure] for row in rows], height=0.55,
                    color=color, zorder=3)
            ax.set(xlim=(0, 100), ylim=(6.65, -0.65))
            ax.set_xticks([0, 25, 50, 75, 100])
            ax.xaxis.set_major_formatter(PercentFormatter(100, decimals=0))
            ax.set_axisbelow(True)
            ax.grid(axis="x", color="#E0E0E0", linewidth=0.7)
            ax.tick_params(axis="x", length=3.5, color="#555555")
            ax.tick_params(axis="y", length=0)
            for index, row in enumerate(rows):
                value = row[measure]
                inside = value >= 20
                ax.text(value - 2 if inside else value + 2, index, percentage_label(value),
                        ha="right" if inside else "left", va="center", fontsize=10.5,
                        color="white" if inside else color, zorder=4)
        score_ax.set_yticks(range(len(rows)), [row["plot_label"] for row in rows])
        score_ax.tick_params(axis="y", pad=12)
        cap_ax.tick_params(axis="y", labelleft=False)
        score_ax.set_xlabel("Score = (wins + 0.5 × draws) / games", labelpad=10)
        cap_ax.set_xlabel("Games ending at the 300-action cap", labelpad=10)
        fig.text(0.06, 0.215,
                 "Historical minimax sweep (shaded rows): all capped games were credited as wins; reported scores include them.",
                 fontsize=10.5, color="#333333", weight="bold")
        fig.text(0.06, 0.165,
                 "The cap was adjudicated by captured-seed lead. Score rate is not a natural-game win rate.",
                 fontsize=10, color="#555555")
        fig.text(0.06, 0.115,
                 "Audited minimax source uses terminal ±1 and H* leaves up to ±10; a corrected rerun is needed.",
                 fontsize=10, color="#555555")
        fig.text(0.06, 0.065,
                 "Implementations and search budgets differ; this is not an equal-runtime comparison. MCTS uses random rollouts.",
                 fontsize=10, color="#555555")
        fig.text(0.06, 0.025,
                 "Sources: individual benchmark summaries and docs/results/classical_search/README.md",
                 fontsize=9, color="#666666")
        figure_paths = [output / f"{figure_name}.{extension}" for extension in ("png", "svg")]
        for path in figure_paths:
            fig.savefig(path, dpi=300)
        plt.close(fig)

    table_path = output / "chart_data.csv"
    fields = [key for key in rows[0] if key != "plot_label"]
    with table_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    with Image.open(output / f"{figure_name}.png") as picture:
        require(picture.size == (3720, 2460), "Unexpected PNG dimensions")
        picture.verify()
    svg = ElementTree.parse(output / f"{figure_name}.svg").getroot()
    require(len(list(svg.iter("{http://www.w3.org/2000/svg}text"))) >= 40,
            "SVG labels must retain editable text")
    provenance = {
        "conditions": 7, "games_per_condition": 10000, "total_recorded_games": 70000,
        "source_sha256": {path.relative_to(ROOT).as_posix(): sha256(path) for path in sources},
        "script_sha256": sha256(Path(__file__)),
        "score_definition": "(wins + 0.5 * draws) / games; includes cap-adjudicated outcomes",
        "cap_definition": "action_limit games / all 10,000 games in the condition; 300 player actions",
        "cap_adjudication": "Captured-seed differential; every capped minimax game was credited as a win",
        "historical_minimax_depths": [1, 2, 3, 4, 5],
        "python_minimax_depths": [1, 2, 3], "cpp_minimax_depths": [4, 5],
        "limitations": ["Historical minimax terminal/leaf scale mismatch; corrected rerun needed",
                        "Native historical executable build provenance was not reconstructed",
                        "Different implementations and budgets; not an equal-runtime comparison",
                        "Scores are not natural-game win rates; no uncertainty estimates are plotted"],
        "design": "Two aligned horizontal-bar panels on common 0–100% scales; historical rows shaded",
        "verification": {"individual_summaries_match_audit": True, "wdl_and_rates_recomputed": True,
                         "cap_counts_match_audit": True, "all_capped_minimax_outcomes_are_wins": True,
                         "audited_seat_balance": {"0": 5000, "1": 5000},
                         "raw_logs_recounted": verify_logs,
                         "raw_games_recounted": sum(row["games_recounted"] for row in recounts.values()),
                         "png_dimensions": [3720, 2460], "svg_editable_text": True},
        "raw_log_verification": recounts,
        "chart_data_sha256": sha256(table_path),
        "figure_sha256": {path.name: sha256(path) for path in figure_paths},
        "matplotlib": matplotlib.__version__,
    }
    (output / "figure_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    source_links = "\n".join(f"- [{row['condition']}](../../../{row['source_summary']})" for row in rows)
    (output / "README.md").write_text(
        "# Classical search against random play\n\n"
        "Figure 4 shows reported score rates alongside cap-adjudicated game shares for Greedy H*, "
        "minimax + H* at depths 1–5, and UCT MCTS with 1,000 simulations and one random rollout per leaf. "
        "Each condition has 10,000 games from the standard initial board, 5,000 in each seat.\n\n"
        "Minimax rows are historical and shaded. Depths 1–3 use Python; depths 4–5 use C++. "
        "Greedy H* uses Python; the UCT MCTS baseline uses C++. All seven records use a 300-action cap, "
        "adjudicated by captured-seed lead. Every capped minimax game was credited as a win. "
        "Scores include capped outcomes and count draws as half a win; they are not natural-game win rates.\n\n"
        "The audited minimax source combines terminal utilities of ±1 with H* leaf estimates up to ±10. "
        "A corrected rerun is needed before drawing a clean conclusion about depth. Historical native "
        "executable build provenance was not reconstructed. Implementations and budgets differ, "
        "so this is not an equal-runtime comparison. No uncertainty estimates are plotted.\n\n"
        "[PNG](classical_search_vs_random.png) is 300 dpi; [SVG](classical_search_vs_random.svg) "
        "retains editable text. [chart_data.csv](chart_data.csv) contains the seven plotted rows; "
        "[figure_provenance.json](figure_provenance.json) records source and export hashes.\n\n"
        "## Sources\n\n"
        "The plot uses the individual summaries listed in "
        "[the source audit](../../../docs/results/classical_search/README.md), "
        "cross-checking [its saved recount](../../../docs/results/classical_search/source_audit.json). "
        "The combined minimax experiment summary is not used.\n\n"
        f"{source_links}\n\n"
        "## Regenerate\n\n"
        "The summaries and saved audit are included in Git:\n\n"
        "```powershell\n.venv/Scripts/python.exe -B -m visualizations.classical_search_vs_random\n```\n\n"
        "To additionally recount the original 70,000 local game records and verify their hashes:\n\n"
        "```powershell\n.venv/Scripts/python.exe -B -m visualizations.classical_search_vs_random --verify-logs\n```\n",
        encoding="utf-8")
    print(json.dumps({"conditions": len(rows), "games": sum(row["games"] for row in rows),
                      "raw_games_recounted": provenance["verification"]["raw_games_recounted"],
                      "output": str(output)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "visualizations/figures/classical_search_vs_random")
    parser.add_argument("--verify-logs", action="store_true", help="Recount the original local JSONL logs as well")
    args = parser.parse_args()
    render(args.output, args.verify_logs)
