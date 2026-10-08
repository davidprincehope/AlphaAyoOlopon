# Greedy H* benchmark figures

- `plain_mcts_vs_greedy_hstar`: outcomes and MCTS scores at 64 and 128 simulations (800 games). Observed results only: the supplied summary has no game-pair records or uncertainty intervals.
- `mixed_opening_progression_64_128`: the completed 13-checkpoint curves at both budgets (10,400 games), plus the matched-opening score change at each checkpoint.

The progression figure computes new pointwise 95% percentile bootstrap intervals, using 10,000 resamples of 200 whole opening pairs and seed 20261003. The lower panel resamples per-opening score differences across budgets. Intervals describe sampled-position variation and exclude training-seed and search-seed uncertainty.

Each opening is played in both seats; score counts a draw as half a win. These are continuation-strength experiments on fixed random prefixes. Changing the evaluation budget does not change the trained checkpoints.

The recovery manifest certifies completion. The completed consolidated 64-budget reports are read from the sibling `checkpoint-strength-mixed-v2-recovery-20261005/results-64-final.jsonl`, and the 128-budget reports from `128/results.jsonl`. All 10,400 embedded game outcomes, opening pairs, checkpoint hashes and saved budget comparisons were checked before plotting.

PNG exports are 300 dpi; SVG exports retain editable text. CSVs contain plotted values, including the newly calculated confidence intervals. Source and output hashes are in `figure_provenance.json`.

Reproduce from the repository root:

```powershell
.venv/Scripts/python.exe -B -m visualizations.greedy_hstar_benchmarks
```
