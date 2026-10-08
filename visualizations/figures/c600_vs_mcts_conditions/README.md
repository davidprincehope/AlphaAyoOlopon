# Individual C600 versus MCTS charts

Each figure shows outcome shares and both agents' scores, using common 0–100% scales.
Score counts a draw as half a win; outcome percentages are the observed shares of all 400 games.
Intervals are the saved 95% percentile bootstrap intervals over 200 whole opening pairs (10,000 resamples).
They exclude training-seed and search-seed uncertainty. Equal simulation counts do not mean equal runtime or compute.

PNG: 300 dpi. SVG: vector graphics with editable text.

| Condition (C600 / MCTS) | C600 score | C600 95% interval | Files |
| --- | ---: | ---: | --- |
| 64 / 64 | 69.75% | 67.25%–72.38% | [PNG](c600_64_mcts_64.png), [SVG](c600_64_mcts_64.svg) |
| 128 / 128 | 68.50% | 66.00%–71.00% | [PNG](c600_128_mcts_128.png), [SVG](c600_128_mcts_128.svg) |
| 64 / 128 | 66.50% | 64.12%–69.00% | [PNG](c600_64_mcts_128.png), [SVG](c600_64_mcts_128.svg) |
| Policy alone / 64 | 48.50% | 44.88%–52.25% | [PNG](c600_policy_mcts_64.png), [SVG](c600_policy_mcts_64.svg) |

All 1,600 game outcomes, paired seat assignments, scores, source hashes, and saved confidence intervals
were checked before plotting. No new games or training were run.

Reproduce from the repository root:

```powershell
.venv/Scripts/python.exe -B -m visualizations.c600_mcts_conditions
```

Source result locations and SHA-256 hashes are recorded in `figure_provenance.json`.
