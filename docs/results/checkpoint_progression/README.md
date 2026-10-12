# Original MLP benchmark: DATASET ≤4 CAPTURE

Completed and locally verified: **56 selected cases, 22,400 games**, the original 185,863-parameter MLP, and 200 frozen DATASET ≤4 CAPTURE openings in both seats.

The opening book limits captured-seed gaps to four and retains random-prefix histories. It defines continuation-strength measurements; the filter does not prove equal position strength.

**DATASET ≤4 CAPTURE** is the public name of the [frozen opening book](../../../experiments/checkpoint_strength/dataset/dataset_le4capture_v3_200.json). Its positions, IDs and bytes are unchanged. Historical paths and identifiers inside the evidence files preserve the original study's hashes.

[All case summaries](results.json), [CSV table](results.csv), [experiment plan](experiment.plan.json), [replay verification](verification.json), [source provenance](source_provenance.json) record the completed study. The aggregate JSON retains every case result and breakdown. Checkpoint exports remain with the MLP training run in ignored `runs/`. Re-auditing the original matches requires separately retained raw game histories.

The three MCTS progression curves and Greedy EOH 64 progression each include C1 and C50â€“C600 in 50-round increments. The two checkpoint matchups and two plain-MCTS/EOH budgets add 800 games each. C600 endpoints are counted once. Neural 128/MCTS 128 and neural 128/EOH are excluded.

## Checkpoint score rates

| Checkpoint | PUCT 64 / MCTS 64 | PUCT 64 / MCTS 128 | Policy alone / MCTS 64 | PUCT 64 / Greedy EOH |
| --- | ---: | ---: | ---: | ---: |
| C1 | 44.00% | 34.38% | 14.00% | 58.88% |
| C50 | 73.38% | 68.75% | 27.13% | 77.88% |
| C100 | 73.13% | 71.38% | 40.00% | 75.38% |
| C150 | 74.00% | 69.75% | 40.38% | 77.50% |
| C200 | 74.88% | 71.63% | 42.25% | 77.25% |
| C250 | 73.25% | 70.25% | 43.63% | 76.50% |
| C300 | 72.38% | 70.13% | 46.25% | 75.25% |
| C350 | 71.25% | 69.50% | 45.38% | 75.63% |
| C400 | 73.00% | 70.50% | 48.50% | 75.38% |
| C450 | 74.13% | 71.38% | 43.38% | 76.75% |
| C500 | 70.63% | 67.75% | 44.75% | 76.13% |
| C550 | 72.75% | 68.63% | 44.63% | 77.13% |
| C600 | 72.63% | 70.38% | 49.13% | 75.63% |

Each cell contains 400 games. Per-case results retain wins, draws, losses, paired confidence intervals, seat/depth/capture-margin breakdowns and termination counts.

[Reviewed figures and plotted values](../../../visualizations/figures/checkpoint_progression/README.md). [MLP training evidence](../mlp600_training/README.md).

The current figures retain three progression curves; PUCT 64/MCTS 128 is
reported as a C600-only experiment in the [C600 result index](../README.md).
The original 56-case records and table above are historical evidence and
remain unchanged. Future evaluation plans omit the other twelve PUCT 64/MCTS
128 progression points.

```powershell
.venv/Scripts/python.exe -B scripts/export_le4capture_results.py --run runs/checkpoint_strength/le4capture-rerun --output runs/publication/le4capture-rerun
.venv/Scripts/python.exe -B -m visualizations.plot_checkpoint_progression --published docs/results/checkpoint_progression --output visualizations/figures/checkpoint_progression
```
