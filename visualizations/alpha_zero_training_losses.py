"""The agreed three-curve figure for the 600-round AlphaZero training run."""

import argparse
from contextlib import redirect_stdout
import hashlib
import io
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "visualizations/.matplotlib"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiments.checkpoint_strength.extract_training_losses import extract


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

    style = {
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 11, "axes.labelsize": 12, "axes.titlesize": 15,
        "xtick.labelsize": 10, "ytick.labelsize": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#444444", "axes.linewidth": 0.8,
        "svg.fonttype": "none", "savefig.facecolor": "white",
    }
    with plt.rc_context(style):
        fig, ax = plt.subplots(figsize=(9, 5.4))
        fig.subplots_adjust(left=0.09, right=0.98, bottom=0.20, top=0.83)
        rounds = [row["learner_round"] for row in rows]
        curves = [("policy_loss", "Policy loss", "#0072B2", 1.4),
                  ("value_loss", "Value loss", "#D55E00", 1.4),
                  ("total_loss", "Total loss", "#222222", 1.6)]
        for field, label, color, width in curves:
            ax.plot(rounds, [row[field] for row in rows], label=label, color=color,
                    linewidth=width, solid_capstyle="round")
        ax.set(xlabel="Training round", ylabel="Loss", xlim=(1, 600), ylim=(0, 2))
        ax.set_xticks([1, 100, 200, 300, 400, 500, 600])
        ax.set_yticks([0, 0.5, 1, 1.5, 2])
        ax.set_axisbelow(True)
        ax.grid(axis="y", color="#E7E7E7", linewidth=0.7)
        ax.tick_params(length=3.5, color="#444444")
        ax.legend(loc="upper right", frameon=False, fontsize=10.5, handlelength=2.7)
        fig.text(0.09, 0.93, "Neural network training loss", fontsize=15, weight="semibold", va="top")
        fig.text(0.09, 0.875,
                 f"Ayo Olopon  |  600 training rounds  |  {settings['max_simulations']} simulations per move",
                 fontsize=10.5, color="#555555", va="top")
        fig.text(0.09, 0.055,
                 "Mean minibatch loss per round. Total = policy + value + L2 regularization.",
                 fontsize=10, color="#555555", va="bottom")
        for extension in ("png", "svg"):
            fig.savefig(output / f"training_losses.{extension}", dpi=300)
        plt.close(fig)

    provenance = {
        "run": run.relative_to(ROOT).as_posix(), "rounds": len(rows),
        "design": "One plot, three recorded curves; no smoothing; linear axes.",
        "curves": ["policy_loss", "value_loss", "total_loss"],
        "total_definition": "policy_loss + value_loss + l2_loss",
        "loss_definition": data["audit"]["loss_definition"],
        "verification": {key: data["audit"][key] for key in (
            "console_loss_matches", "finite_nonnegative_losses", "total_loss_components_match",
            "gradient_update_counts_match_settings", "missing_rounds", "duplicate_rounds")},
        "source_sha256": data["audit"]["source_sha256"],
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "figure_sha256": {name: hashlib.sha256((output / name).read_bytes()).hexdigest()
                          for name in ("training_losses.png", "training_losses.svg")},
        "matplotlib": matplotlib.__version__,
    }
    (output / "figure_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"rounds": len(rows), "console_loss_matches": data["audit"]["console_loss_matches"],
                      "png": str(output / "training_losses.png"), "svg": str(output / "training_losses.svg")}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=ROOT / "runs/alpha_zero/modal-a100-600-20261003")
    parser.add_argument("--output", type=Path, default=ROOT / "visualizations/figures/alpha_zero_training_losses_600")
    args = parser.parse_args()
    render(args.run, args.output)
