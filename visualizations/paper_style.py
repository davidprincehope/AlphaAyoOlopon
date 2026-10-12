"""Shared style for single-axis experimental figures; captions live in reports."""

import hashlib
from pathlib import Path

FIGSIZE = (7, 4.3)
DPI = 300
PIXELS = (2100, 1290)
BLUE = "#0072B2"
STYLE = {
    "font.family": "serif",
    "font.serif": ["STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 11,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "axes.edgecolor": "#333333",
    "savefig.facecolor": "white",
}


def finish_axes(fig, ax):
    fig.subplots_adjust(left=0.14, right=0.97, bottom=0.17, top=0.96)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.5)
    ax.tick_params(direction="out", length=3, width=0.8)


def style_sha256():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
