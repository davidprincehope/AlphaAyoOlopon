# C600 PUCT 64 versus minimax depth 6

C600 played **400 games** against deterministic depth-6 alpha-beta minimax: the same 200 DATASET ≤4 CAPTURE openings used in the other C600 comparisons, each played from both seats.

| C600 wins | Draws | Losses | Win rate | Score rate | 95% score interval |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 207 | 107 | 86 | 51.75% | 65.13% | 62.50% to 67.88% |

Score credits draws as half a win. Intervals use 10,000 bootstrap resamples of complete opening pairs. The result supports stronger play by C600 against this specific opponent on this opening book; it describes one training run and excludes training-seed and new search-seed uncertainty.

## Evaluation setup

- C600: original 185,863-parameter MLP, 64 PUCT simulations per move, exploration constant 1.5, no root noise or temperature sampling; maximum visits with lowest-action-ID ties.
- Opponent: Python/OpenSpiel alpha-beta minimax, depth 6 measured in complete player actions. Relay laps belong to the same action. Search ties select the first legal action.
- Leaf formula: original H* clipped to [-10, 10], divided directly by 11. Nonterminal scores lie inside terminal outcomes of -1/0/+1. These results use the normalized formula; historical minimax reports used the original scale.
- Rules: `ayo_olopon_alpha_zero(max_moves=1000,cutoff=collect)`, with repetition memory restored by opening-prefix replay.
- Pairing: 200 openings, both seats, seed `20261003 + opening_index` shared by the paired games. Eight local CPU workers; scheduling does not change pair IDs or seeds.
- Verification: all 400 complete legal histories replayed to matching terminal returns, captured scores and move counts. No game reached the 1,000-action horizon.

## Evidence and reproduction

[results.json](results.json) contains outcomes, seat statistics and paired intervals. [evaluation_manifest.json](evaluation_manifest.json) records model, opening-book and evaluated-source hashes, plus the raw match-history hash. Raw histories and resumable pairs are retained locally under ignored `runs/checkpoint_strength/c600_puct64_vs_minimax_depth6`.

```sh
python -B -m experiments.checkpoint_strength.neural_vs_minimax --run runs/alpha_zero/modal-a100-600-20261003 --config experiments/checkpoint_strength/configs/c600_puct64_vs_minimax_depth6.json --output runs/checkpoint_strength/c600-minimax-depth6-rerun --workers 8
```

The runner verifies saved pairs before resume and refuses a changed model, opening book, configuration or source snapshot. Use the same original model to reproduce this comparison.

[All C600 comparisons](../README.md) | [Reproduction guide](../../../../experiments/checkpoint_strength/README.md).
