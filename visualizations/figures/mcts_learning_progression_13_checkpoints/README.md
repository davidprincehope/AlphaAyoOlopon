# AlphaZero progression against plain MCTS

Four separate 13-checkpoint progression studies: PUCT 64/MCTS 64, PUCT 128/MCTS 128, PUCT 64/MCTS 128, and direct policy/MCTS 64. Checkpoints are C1 and C50–C600 at 50-round intervals.

Each point contains 400 games: the same 200 mixed-depth openings, each in both seats, with identical paired seeds. Each curve has 5,200 games; all four contain 20,800 games. All games were replayed locally and all scores and intervals recomputed before plotting.

Scores count draws as half a win. Error bars are saved pointwise 95% opening-pair percentile bootstrap intervals from 10,000 resamples; they exclude training-seed and search-seed uncertainty. Equal simulation counts do not imply equal runtime or compute.

PNG exports are 300 dpi. SVG exports retain editable text. `chart_data.csv` contains the 52 plotted observations.

```powershell
.venv/Scripts/python.exe -B -m visualizations.mcts_learning_progression
```
