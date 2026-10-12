# Experiment figures

Each figure answers a reported experimental question. Use the shared academic style in `paper_style.py`: serif fonts, clear titles and axes, restrained colours and light gridlines. Keep one plot per standalone image. The requested combined training-loss image is a separate three-panel export. Do not add decorative charts or unsupported trend fits.

| Plotter | Output folder | Evidence | Resolution |
| --- | --- | --- | --- |
| `plot_training_losses.py` | [mlp600_training_losses](figures/mlp600_training_losses/README.md) | Total, value and policy losses; separate and combined versions | 600 dpi |
| `plot_checkpoint_progression.py` | [checkpoint_progression](figures/checkpoint_progression/README.md) | PUCT 64 vs MCTS 64, policy-only vs MCTS 64, PUCT 64 vs Greedy EoH H* | 300 dpi |
| `plot_state_space_growth.py` | [state_space_growth](figures/state_space_growth/README.md) | New and cumulative position keys by discovery depth | 300 dpi |

From the repository root, with dependencies installed:

```sh
python -B -m visualizations.plot_training_losses
python -B -m visualizations.plot_checkpoint_progression
python -B -m visualizations.plot_state_space_growth
```

Training-loss regeneration requires the original local learner logs. The other plotters use included, verified summaries. All scripts accept `--output`; training accepts `--run`, checkpoint progression accepts `--published` or `--run`, and state-space growth accepts `--source`.

Exports include PNGs, plotted data and `figure_provenance.json`. Recorded values are not smoothed. Checkpoint intervals resample whole opening pairs; each point contains 400 games. Losses describe replay fitting, while checkpoint evaluations measure playing strength. [Experiment results](../docs/results/README.md).
