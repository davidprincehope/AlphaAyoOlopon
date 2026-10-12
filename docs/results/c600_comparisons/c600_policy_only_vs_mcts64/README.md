# C600 policy only versus MCTS 64

Completed, replay-verified experiment: 400 games from the 200 frozen DATASET ≤4 CAPTURE openings, each in both seats.

| C600 wins | Draws | C600 losses | Win rate | Score rate | 95% score interval |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 138 | 117 | 145 | 34.50% | 49.13% | 45.38% to 52.88% |

[results.json](results.json) contains both players' statistics, the experiment configuration, paired confidence intervals, seat/depth/capture-margin breakdowns and source hashes.

Score credits draws as half a win. Confidence intervals resample whole opening pairs, 10,000 times.

C600 chooses its highest-probability legal action with no search. The policy-only checkpoint progression is also retained.

The source result is preserved in the [historical checkpoint suite](../../checkpoint_progression/README.md). Dataset and checkpoint hashes match the retained training export; no rerun was required. Replaying original matches requires separately retained raw histories.

[All experiment folders](../../README.md).
