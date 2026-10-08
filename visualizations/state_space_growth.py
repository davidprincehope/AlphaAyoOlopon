"""Figure 2: new and cumulative Ayo position keys by discovery depth."""

import argparse
import csv
import hashlib
from itertools import accumulate
import json
import os
from pathlib import Path
from xml.etree import ElementTree

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "visualizations/.matplotlib"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import LogFormatterMathtext, NullLocator
from PIL import Image

BLUE, ORANGE = "#0072B2", "#D55E00"
STYLE = {
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 11, "axes.labelsize": 11, "axes.titlesize": 13,
    "xtick.labelsize": 10, "ytick.labelsize": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#555555", "axes.linewidth": 0.8,
    "svg.fonttype": "none", "savefig.facecolor": "white",
}
POSITION_KEY = "12 pit counts + 2 captured scores + player to move / terminal marker"
EXCLUDED_FIELDS = ["repetition history", "elapsed move count"]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_counts(source):
    data = json.loads(source.read_text(encoding="utf-8"))
    require(data["complete"] is True, "Only a completed enumeration can be plotted")
    require(type(data["max_depth"]) is int and data["max_depth"] >= 0,
            "Maximum depth must be a nonnegative integer")
    rows = data["states_by_depth"]
    require([row["depth"] for row in rows] == list(range(data["max_depth"] + 1)),
            "Discovery depths must be contiguous and ordered from zero")
    require(all(type(row["depth"]) is int and type(row["states"]) is int
                and row["states"] > 0 for row in rows),
            "Every depth must have a positive integer discovery count")
    require(rows[0]["states"] == 1, "Depth zero must contain the initial position")
    counts = [row["states"] for row in rows]
    require(sum(counts) == data["total_unique_states"],
            "Per-depth counts do not sum to the reported unique-position total")
    require(max(counts) == data["peak_frontier"],
            "Peak discovery count does not match the reported peak frontier")
    require(0 <= data["terminal_states"] <= data["total_unique_states"],
            "Terminal-position count is outside the unique-position total")
    require(data["legal_transitions_generated"] == data["duplicate_attempts"]
            + data["total_unique_states"] - 1,
            "Transitions do not match duplicate attempts plus newly discovered keys")
    return data, [row["depth"] for row in rows], counts, list(accumulate(counts))


def render(source, output):
    source, output = source.resolve(), output.resolve()
    data, depths, counts, cumulative = load_counts(source)
    output.mkdir(parents=True, exist_ok=True)
    peak_index = counts.index(max(counts))
    total = data["total_unique_states"]
    max_depth = depths[-1]
    cumulative_millions = [value / 1_000_000 for value in cumulative]
    figure_name = "state_space_growth"

    with plt.rc_context(STYLE):
        fig = plt.figure(figsize=(12.4, 7))
        fig.text(0.08, 0.95, "Growth of the discovered position space",
                 fontsize=19, weight="bold", va="top")
        fig.text(0.08, 0.885,
                 f"Ayo Olopon  |  {total:,} distinct position keys  |  "
                 f"Discovery depths 0–{max_depth}  |  Complete enumeration",
                 fontsize=10.5, color="#555555", va="top")
        new_ax = fig.add_axes((0.08, 0.30, 0.40, 0.47))
        cumulative_ax = fig.add_axes((0.59, 0.30, 0.38, 0.47))
        new_ax.set_title("a  Newly discovered positions", loc="left", pad=16,
                         weight="bold")
        cumulative_ax.set_title("b  Cumulative positions", loc="left", pad=16,
                                weight="bold")

        new_ax.plot(depths, counts, color=BLUE, linewidth=1.8,
                    marker="o", markersize=2.5, markeredgewidth=0)
        new_ax.set_yscale("log")
        new_ax.set_ylim(0.65, 100_000_000)
        new_ax.set_yticks([1, 100, 10_000, 1_000_000, 100_000_000])
        new_ax.yaxis.set_major_formatter(LogFormatterMathtext())
        new_ax.yaxis.set_minor_locator(NullLocator())
        new_ax.set_ylabel("New position keys (log scale)")
        new_ax.plot(depths[peak_index], counts[peak_index], "o", color=BLUE,
                    markersize=6, markeredgecolor="white", markeredgewidth=0.7)
        new_ax.annotate(f"Peak: {counts[peak_index]:,}\nat depth {depths[peak_index]}",
                        xy=(depths[peak_index], counts[peak_index]),
                        xytext=(39, 20_000_000), color=BLUE, fontsize=10,
                        va="center", arrowprops={"arrowstyle": "-", "color": BLUE,
                                                 "linewidth": 0.8},
                        bbox={"facecolor": "white", "edgecolor": "none", "pad": 2})
        new_ax.annotate(f"{counts[-1]:,} new position\nat depth {max_depth}",
                        xy=(max_depth, counts[-1]), xytext=(65, 6),
                        color=BLUE, fontsize=10, va="center",
                        arrowprops={"arrowstyle": "-", "color": BLUE,
                                    "linewidth": 0.8},
                        bbox={"facecolor": "white", "edgecolor": "none", "pad": 2})

        cumulative_ax.plot(depths, cumulative_millions, color=ORANGE, linewidth=2)
        cumulative_ax.set_ylim(0, 315)
        cumulative_ax.set_yticks([0, 50, 100, 150, 200, 250, 300])
        cumulative_ax.set_ylabel("Cumulative position keys (millions)")
        cumulative_ax.plot(max_depth, cumulative_millions[-1], "o", color=ORANGE,
                           markersize=6, markeredgecolor="white", markeredgewidth=0.7)
        cumulative_ax.annotate(f"{total:,} total keys",
                               xy=(max_depth, cumulative_millions[-1]),
                               xytext=(48, 300), color=ORANGE, fontsize=10,
                               va="center", arrowprops={"arrowstyle": "-", "color": ORANGE,
                                                        "linewidth": 0.8},
                               bbox={"facecolor": "white", "edgecolor": "none", "pad": 2})

        for ax in (new_ax, cumulative_ax):
            ax.set_xlim(-2, max_depth + 3)
            ax.set_xticks([0, 20, 40, 60, 80, 100])
            ax.set_xlabel("Discovery depth (player actions)", labelpad=9)
            ax.set_axisbelow(True)
            ax.grid(axis="y", color="#E6E6E6", linewidth=0.7)
            ax.tick_params(length=3.5, color="#555555")

        fig.text(0.08, 0.17, f"Position key = {POSITION_KEY}.",
                 fontsize=10, color="#444444")
        fig.text(0.08, 0.125,
                 "Repetition history and elapsed move count are excluded; "
                 "these are not counts of every history-dependent game state.",
                 fontsize=9.5, color="#555555")
        fig.text(0.08, 0.08,
                 "Discovery depth is the first BFS level. "
                 "One player action includes all relay laps; lines connect recorded counts without smoothing.",
                 fontsize=9.5, color="#555555")
        fig.text(0.08, 0.035,
                 f"Source: {source.relative_to(ROOT).as_posix()}",
                 fontsize=9, color="#666666")
        figure_paths = [output / f"{figure_name}.{extension}" for extension in ("png", "svg")]
        for path in figure_paths:
            fig.savefig(path, dpi=300)
        plt.close(fig)

    table_path = output / "chart_data.csv"
    with table_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["discovery_depth", "new_position_keys", "cumulative_position_keys"])
        writer.writerows(zip(depths, counts, cumulative))

    with Image.open(output / f"{figure_name}.png") as picture:
        require(picture.size == (3720, 2100), "Unexpected PNG dimensions")
        picture.verify()
    svg = ElementTree.parse(output / f"{figure_name}.svg").getroot()
    require(len(list(svg.iter("{http://www.w3.org/2000/svg}text"))) >= 30,
            "SVG labels must retain editable text")

    provenance = {
        "source": source.relative_to(ROOT).as_posix(),
        "source_sha256": sha256(source),
        "enumerator_sha256": sha256(ROOT / "experiments/enumerate_state_space.cc"),
        "script_sha256": sha256(Path(__file__)),
        "position_key": POSITION_KEY,
        "excluded_key_fields": EXCLUDED_FIELDS,
        "depth_definition": "First-discovery breadth-first-search level; one edge is one complete player action",
        "complete": data["complete"],
        "discovery_depths": len(depths), "max_depth": max_depth,
        "total_unique_position_keys": total, "terminal_position_keys": data["terminal_states"],
        "peak_new_position_keys": {"depth": depths[peak_index], "count": counts[peak_index]},
        "panels": [
            {"measure": "new_position_keys", "x_scale": "linear", "y_scale": "log10"},
            {"measure": "cumulative_position_keys", "x_scale": "linear", "y_scale": "linear",
             "display_units": "millions"},
        ],
        "design": "Two panels; all recorded depths; no smoothing, fitting, or uncertainty estimates",
        "verification": {
            "ordered_contiguous_depths": True, "positive_integer_counts": True,
            "per_depth_sum_matches_total": True, "peak_matches_frontier": True,
            "transition_accounting_matches": True, "png_dimensions": [3720, 2100],
            "svg_editable_text": True,
        },
        "chart_data_sha256": sha256(table_path),
        "figure_sha256": {path.name: sha256(path) for path in figure_paths},
        "matplotlib": matplotlib.__version__,
    }
    (output / "figure_provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    (output / "README.md").write_text(
        "# Growth of the discovered position space\n\n"
        "Figure 2 plots newly discovered position keys by discovery depth on a log scale "
        "and their cumulative count on a linear scale in a second panel. "
        f"The completed enumeration has {total:,} keys over depths 0–{max_depth}. "
        f"The discovery count peaks at depth {depths[peak_index]} with {counts[peak_index]:,} new keys.\n\n"
        f"Position key: {POSITION_KEY}. Repetition history and elapsed move count are excluded. "
        "Discovery depth is the first BFS level; a complete player action includes all relay laps. "
        "These counts describe the enumerator's position keys, not every history-dependent state "
        "or a solution of the game. No smoothing or uncertainty estimates are applied.\n\n"
        "Source: [enumeration results](../../../experiments/ayo_state_space_results.json). "
        "The source is included in Git, so the figure can be regenerated from a fresh clone.\n\n"
        "[PNG](state_space_growth.png) is 300 dpi; [SVG](state_space_growth.svg) retains editable text. "
        f"[chart_data.csv](chart_data.csv) contains all {len(depths)} depth counts and their cumulative sums. "
        "[figure_provenance.json](figure_provenance.json) records source, script, and export hashes.\n\n"
        "```powershell\n.venv/Scripts/python.exe -B -m visualizations.state_space_growth\n```\n",
        encoding="utf-8")
    print(json.dumps({"depths": len(depths), "total_position_keys": total,
                      "peak_depth": depths[peak_index], "output": str(output)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "experiments/ayo_state_space_results.json")
    parser.add_argument("--output", type=Path, default=ROOT / "visualizations/figures/state_space_growth")
    args = parser.parse_args()
    render(args.source, args.output)
