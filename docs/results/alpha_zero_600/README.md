# Completed 600-round AlphaZero studies

This directory contains compact summaries from the retained 600-round training run. Models, replay buffers, optimizer state, raw per-game logs and cloud launch metadata remain in the local `runs/` directory and are excluded from Git.

The four plain-MCTS progression conditions are complete: 13 checkpoints per condition, 400 games per checkpoint, 5,200 games per curve, and 20,800 games overall. All games were legally replayed locally; scores and opening-pair intervals were recomputed.

## Checkpoint progression against plain MCTS

Score counts draws as half a win. The same 200 mixed-depth openings are used in both seats at every checkpoint.

| Checkpoint | PUCT 64 / MCTS 64 | PUCT 128 / MCTS 128 | PUCT 64 / MCTS 128 | Policy alone / MCTS 64 |
| --- | ---: | ---: | ---: | ---: |
| C1 | 46.750% | 42.875% | 37.750% | 18.500% |
| C50 | 72.375% | 71.500% | 67.375% | 30.250% |
| C100 | 70.000% | 70.000% | 67.625% | 39.625% |
| C150 | 70.375% | 68.000% | 66.250% | 42.875% |
| C200 | 72.250% | 72.500% | 68.500% | 46.500% |
| C250 | 72.250% | 70.500% | 67.125% | 46.250% |
| C300 | 70.750% | 69.375% | 67.375% | 45.625% |
| C350 | 70.500% | 68.875% | 65.625% | 48.000% |
| C400 | 73.000% | 69.750% | 66.750% | 51.500% |
| C450 | 71.625% | 70.250% | 68.375% | 47.500% |
| C500 | 70.375% | 69.375% | 66.000% | 50.875% |
| C550 | 71.000% | 68.750% | 67.000% | 47.500% |
| C600 | 69.750% | 68.500% | 66.500% | 48.500% |

Pointwise 95% intervals resample whole opening pairs. They exclude training-seed and search-seed uncertainty. Equal simulation budgets do not imply equal compute.

## Published evidence

- [Training settings](training_manifest.json) and [final training counts](training_summary.json).
- [Four complete MCTS progression curves](progression.results.json), [completion manifest](progression.manifest.json), [experiment plan](experiment.plan.json), and [local verification record](verification.json).
- The `checkpoint-*` folders contain the two checkpoint head-to-head summaries and the four original C600-versus-MCTS summaries. The 64/64 manifest also serves as a small real-report test fixture.
- [Source and export hashes](source_provenance.json), recording the original local artifact and the published copy. Local absolute checkout prefixes are normalized to relative paths.

[Individual progression charts and plotted data](../../../visualizations/figures/mcts_learning_progression_13_checkpoints/README.md) are available as PNG, editable SVG, CSV and a ZIP.

## Regenerate the published summaries

With the original, verified local runs available:

```sh
python -B scripts/export_public_results.py
```

This exports summaries only. It does not train models or run matches. The source hashes refer to the retained raw evidence, which is not distributed by a Git clone.
