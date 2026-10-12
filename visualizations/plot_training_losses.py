"""Separate academic figures for the three recorded MLP600 training losses."""

import argparse
from contextlib import redirect_stdout
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "alpha-ayo-matplotlib"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiments.checkpoint_strength.extract_training_losses import extract
from visualizations.paper_style import BLUE, FIGSIZE, STYLE, finish_axes, style_sha256

LOSS_DPI = 600
LOSS_SCALES = {
    "total_loss": ((0, 1.9), [0, 0.5, 1, 1.5]),
    "value_loss": ((0.10, 0.70), [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]),
    "policy_loss": ((0.65, 1.10), [0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00, 1.05, 1.10]),
}


def plot_loss(ax, rows, field, label):
    ax.plot([row["learner_round"] for row in rows], [row[field] for row in rows],
            color=BLUE, linewidth=1.0)
    limits, ticks = LOSS_SCALES[field]
    ax.set(xlabel="Learner round", ylabel=label, xlim=(1, 600), ylim=limits, yticks=ticks)
    ax.set_title(label.title() + " vs. Learner Round", fontsize=12, fontweight="normal", pad=10)
    ax.set_xticks([1, 100, 200, 300, 400, 500, 600])


def render(run: Path, output: Path):
    run, output = run.resolve(), output.resolve()
    with redirect_stdout(io.StringIO()):
        extract(run, output)
    data = json.loads((output / "loss_data.json").read_text(encoding="utf-8"))
    rows = data["rows"]
    snapshot = json.loads((run / "training-checkpoints/step-000600/metadata.json").read_text(encoding="utf-8"))
    settings = snapshot["manifest"]["settings"]
    if snapshot["completed_step"] != 600 or settings["decouple_weight_decay"]:
        raise ValueError("This figure expects the completed 600-round run with L2 in its training objective.")

    curves = [("total_loss", "Total training loss"),
              ("value_loss", "Value loss"),
              ("policy_loss", "Policy loss")]
    figure_names = []
    for field, label in curves:
        with plt.rc_context(STYLE):
            fig, ax = plt.subplots(figsize=FIGSIZE)
            plot_loss(ax, rows, field, label)
            finish_axes(fig, ax)
            fig.subplots_adjust(left=0.14, top=0.90)
            name = "total_training_loss.png" if field == "total_loss" else field + ".png"
            fig.savefig(output / name, dpi=LOSS_DPI)
            figure_names.append(name)
            plt.close(fig)

    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
        for ax, (field, label) in zip(axes, curves):
            plot_loss(ax, rows, field, label)
            ax.set_axisbelow(True)
            ax.grid(axis="y", color="#DDDDDD", linewidth=0.5)
            ax.tick_params(direction="out", length=3, width=0.8)
        fig.subplots_adjust(left=0.065, right=0.985, bottom=0.17, top=0.90, wspace=0.38)
        name = "training_losses_combined.png"
        fig.savefig(output / name, dpi=LOSS_DPI)
        figure_names.append(name)
        plt.close(fig)

    provenance = {
        "run": run.relative_to(ROOT).as_posix(), "rounds": len(rows),
        "design": "Three separate figures and a requested three-panel comparison; vertical axes label each loss; unsmoothed recorded losses; linear axes.",
        "curves": ["policy_loss", "value_loss", "total_loss"],
        "export_dpi": LOSS_DPI, "y_axis_limits": {key: list(value[0]) for key, value in LOSS_SCALES.items()},
        "scale_note": "Value and policy axes start above zero; every recorded value remains visible.",
        "total_definition": "policy_loss + value_loss + l2_loss",
        "loss_definition": data["audit"]["loss_definition"],
        "verification": {key: data["audit"][key] for key in (
            "console_loss_matches", "finite_nonnegative_losses", "total_loss_components_match",
            "gradient_update_counts_match_settings", "missing_rounds", "duplicate_rounds")},
        "source_sha256": data["audit"]["source_sha256"],
        "script": Path(__file__).relative_to(ROOT).as_posix(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "style_sha256": style_sha256(),
        "figure_sha256": {name: hashlib.sha256((output / name).read_bytes()).hexdigest()
                          for name in figure_names},
        "matplotlib": matplotlib.__version__,
    }
    (output / "figure_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    (output / "README.md").write_text(
        "# MLP600 training losses\n\n"
        "Three separate figures show mean minibatch losses in each of the 600 learner rounds. "
        "Values are recorded optimizer statistics from evolving replay data; they measure fitting, "
        "not held-out generalization or playing strength. No smoothing is applied.\n\n"
        "| Figure | Experimental measure |\n| --- | --- |\n"
        "| [Total training loss](total_training_loss.png) | Policy + value + weight-only L2 regularization |\n"
        "| [Value loss](value_loss.png) | Mean squared error against final game outcomes |\n"
        "| [Policy loss](policy_loss.png) | Cross-entropy against self-play MCTS targets |\n\n"
        "The standalone plots and the [merged three-panel comparison](training_losses_combined.png) "
        "are exported at 600 dpi. Vertical axes identify total training loss, value loss and policy loss. "
        "Value loss uses a 0.10 to 0.70 range and policy loss uses 0.65 to 1.10, with finer ticks. "
        "These linear axes start above zero to resolve variation; all 600 observations remain visible. "
        "The merged image keeps the same separate scales; compare trends using the tick values. "
        "[Recorded values and audit](loss_data.json) and [provenance](figure_provenance.json) "
        "retain the numerical evidence and source hashes.\n\n"
        "```powershell\n.venv/Scripts/python.exe -B -m visualizations.plot_training_losses\n```\n",
        encoding="utf-8")
    print(json.dumps({"rounds": len(rows), "console_loss_matches": data["audit"]["console_loss_matches"],
                      "figures": figure_names}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=ROOT / "runs/alpha_zero/modal-a100-600-20261003")
    parser.add_argument("--output", type=Path, default=ROOT / "visualizations/figures/mlp600_training_losses")
    args = parser.parse_args()
    render(args.run, args.output)
