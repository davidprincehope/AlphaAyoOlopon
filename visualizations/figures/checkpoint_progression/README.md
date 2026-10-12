# DATASET ≤4 CAPTURE benchmark figures

Three checkpoint progression figures: neural PUCT 64 versus MCTS 64, policy-only versus MCTS 64, and neural PUCT 64 versus Greedy EOH H*. Each includes C1 and C50–C600 in 50-round increments. The 39 plotted cases contain 15,600 games. PUCT 64 versus MCTS 128 is retained as a C600-only comparison in its [experiment results](../../../docs/results/c600_comparisons/c600_puct64_vs_mcts128/README.md). The historical published suite retains 56 verified cases and 22,400 games.

All use the same 200 frozen openings with starting captured-seed gap at most four, both seats and paired seeds. The entire neural 128/MCTS 128 and neural 128/Greedy EOH curves are excluded. Each point contains 400 games, including the C600 endpoint of each curve.

The dashed horizontal line marks a 50% score; error bars show pointwise 95% intervals. Lines connect observations without smoothing. Scores credit draws as half a win. Intervals are pointwise 95% bootstrap intervals from 10,000 resamples of whole opening pairs. They describe sampled-position variation, excluding training-seed and search-seed uncertainty. Equal simulation budgets do not imply equal runtime.

Each PNG contains one plot, with serif typography and a light grid, exported at 300 dpi.  `chart_data.csv` contains both players' statistics for the 39 plotted cases. The provenance file records source and output hashes.

The default command regenerates charts from hash-checked published summaries. It preserves the original study's values and intervals; replaying the original matches requires separately retained raw histories. Use `--run <local-suite>` for a new, fully verified evaluation.

```powershell
.venv/Scripts/python.exe -B -m visualizations.plot_checkpoint_progression
```

[Plotted values](chart_data.csv), [source provenance](figure_provenance.json), and [verified summaries](../../../docs/results/checkpoint_progression/README.md).

| Figure | PNG |
| --- | --- |
| Neural PUCT 64 versus MCTS 64 | [PNG](puct64_vs_mcts64.png) |
| Learned policy versus MCTS 64 | [PNG](policy_only_vs_mcts64.png) |
| Neural PUCT 64 versus Greedy EOH | [PNG](puct64_vs_greedy_eoh_hstar.png) |
