# Corrected minimax depth comparison versus random

Completed rerun: **10,000 games per depth**, with 5,000 in each seat. Every condition uses the same Python game, OpenSpiel alpha-beta search, standard start and seed schedule.

| Depth | Wins | Draws | Losses | Win rate | Score rate | 95% score interval | Unresolved |
| ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| 1 | 7,571 | 886 | 1,543 | 75.71% | 80.14% | 79.46 to 80.81% | 0 |
| 2 | 9,376 | 308 | 316 | 93.76% | 95.30% | 94.92 to 95.66% | 0 |
| 3 | 9,889 | 90 | 21 | 98.89% | 99.34% | 99.21 to 99.47% | 0 |
| 4 | 9,841 | 146 | 13 | 98.41% | 99.14% | 99.00 to 99.28% | 0 |
| 5 | 9,838 | 112 | 50 | 98.38% | 98.94% | 98.77 to 99.11% | 0 |

Score credits draws as half a win. Confidence intervals use 10,000 bootstrap resamples of whole seed/seat pairs.

Depth 3 had the highest observed score. The improvement from depth 1 to depth 3 was **19.20 percentage points**, with a paired 95% interval of **18.54 to 19.87 points**. Point estimates declined at depths 4 and 5. These results show strong gains through depth 3 and a nonmonotonic relationship across the five tested depths.

| Depth | Player 0 score | Player 1 score | Mean actions | Maximum actions |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 63.60% | 96.68% | 32.32 | 230 |
| 2 | 90.72% | 99.88% | 17.70 | 206 |
| 3 | 98.71% | 99.97% | 17.45 | 185 |
| 4 | 98.32% | 99.96% | 17.32 | 159 |
| 5 | 98.04% | 99.84% | 14.02 | 146 |

| Depth comparison | Score difference (percentage points) | Paired 95% interval |
| --- | ---: | --- |
| 2 minus 1 | +15.16 | +14.38 to +15.93 |
| 3 minus 2 | +4.04 | +3.65 to +4.44 |
| 4 minus 3 | -0.20 | -0.39 to -0.01 |
| 5 minus 4 | -0.20 | -0.42 to +0.01 |
| 3 minus 1 | +19.20 | +18.54 to +19.87 |
| 4 minus 1 | +19.00 | +18.32 to +19.68 |
| 5 minus 1 | +18.80 | +18.09 to +19.50 |

Intervals are individual and are not adjusted for multiple comparisons. The comparison preserves matching seeds across depths and both seats within each pair.

The heuristic is clipped H* divided by 11, strictly inside terminal utilities ±1. The 1,000-action horizon is visible to search. Internal and external limit cases are unresolved; they receive no W/D/L credit. The JSON also records full-sample score bounds if limits occur. Rule-based repetition resolution counts as a completed game.

These results measure performance against uniform random play from the standard start. They do not establish general playing strength, an equal-runtime comparison, or a new ranking against historical MCTS. The rerun changes both evaluator scaling and the limit protocol; differences from historical results cannot be attributed to scaling alone.

[Results and source hashes](results.json) | [Paired game scores](paired_scores.csv) | [Protocol and rerun command](../../../../experiments/minimax_depth_vs_random/README.md)

The export replays and recounts every game, verifies raw log/source hashes, and recomputes all intervals. The CSV retains both seat scores for every seed at every depth (1 = win, 0.5 = draw, 0 = loss; blank = unresolved). Full move traces and exact source snapshots remain in the local run named in `publication_verification`.
