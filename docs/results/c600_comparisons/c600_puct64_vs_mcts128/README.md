# C600 PUCT 64 versus MCTS 128

Completed, replay-verified experiment: 400 games from the 200 frozen DATASET ≤4 CAPTURE openings, each in both seats.

| C600 wins | Draws | C600 losses | Win rate | Score rate | 95% score interval |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 232 | 99 | 69 | 58.00% | 70.38% | 67.75% to 73.00% |

[results.json](results.json) contains both players' statistics, the experiment configuration, paired confidence intervals, seat/depth/capture-margin breakdowns and source hashes.

Score credits draws as half a win. Confidence intervals resample whole opening pairs, 10,000 times.

This is the standalone C600 comparison retained after removing the PUCT 64/MCTS 128 progression. It is reported as an experiment, without a chart.

The source result is preserved in the [historical checkpoint suite](../../checkpoint_progression/README.md). Dataset and checkpoint hashes match the retained training export; no rerun was required. Replaying original matches requires separately retained raw histories.

[All experiment folders](../../README.md).
