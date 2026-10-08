# Experiment figures

Figure designs are chosen with the user, one experiment at a time. The earlier
batch of generic PNG/PDF figures and the inline score explainer were removed
on 5 October 2026. New agreed figures will be exported as PNG and SVG.

For each experiment, agree on the scientific question first, then select the
measurements, axes, comparisons, and treatment of uncertainty. Review one
figure before applying its design to related results.

## State-space growth figure

Figure 2 shows newly discovered Ayo position keys on a log scale and cumulative
keys on a linear scale in a second panel. All 104 discovery depths are plotted
without smoothing. The chart labels the position key and the excluded history
and move-count fields; discovery depth counts complete player actions.

```powershell
.venv/Scripts/python.exe -B -m visualizations.state_space_growth
```

[PNG/SVG exports, plotted data, and provenance](figures/state_space_growth/README.md)
are generated from [the enumeration summary](../experiments/ayo_state_space_results.json).
The plotter verifies the completed-run flag, contiguous depths, per-depth sum,
peak frontier, and transition accounting. The source is included in Git, so
this figure can be regenerated from a fresh clone.

## Classical search against random play

Figure 4 places reported score rates and cap-adjudicated game shares in aligned
panels for Greedy H*, minimax depths 1–5, and UCT MCTS at 1,000 simulations.
Each condition has 10,000 standard-start games, 5,000 in each seat. Minimax rows
are marked historical, with Python depths 1–3 and C++ depths 4–5 distinguished.
Every capped minimax game was credited as a win. The figure states the cap rule,
the audited minimax scale mismatch, and the differing implementations and budgets.

```powershell
.venv/Scripts/python.exe -B -m visualizations.classical_search_vs_random
```

[PNG/SVG exports, plotted data, and provenance](figures/classical_search_vs_random/README.md)
use the seven individual benchmark summaries listed in
[the source audit](../docs/results/classical_search/README.md),
cross-checked against its saved recount. The summaries and audit are included
in Git. Add `--verify-logs` to recount the original 70,000 local game records
and check their hashes, seat balance, outcomes, and cap adjudication.

## Agreed training loss figure

The first agreed figure shows the 600-round A100 training run on one plot:
policy loss, value loss, and total loss against training round. It uses the
recorded losses without smoothing. Total includes L2 regularization. Each
point is the mean minibatch training loss across optimizer updates within that
round.

```powershell
.venv/Scripts/python.exe -B -m visualizations.alpha_zero_training_losses
```

The dedicated script exports `training_losses.png` (300 dpi) and
`training_losses.svg` (editable text) into
`figures/alpha_zero_training_losses_600/`, alongside the source loss table and
figure provenance. It checks all 600 rounds against their learner session logs.

## Individual C600 versus MCTS figures

The four selected conditions are C600/MCTS simulation budgets 64/64, 128/128,
64/128, and direct C600 policy without search versus MCTS at 64. Each has its
own figure, showing observed outcome shares and both agents' scores with the
saved 95% opening-pair bootstrap intervals. All figures share 0–100% scales;
score counts a draw as half a win. Each condition has 400 games on the same
200 openings with seats swapped.

```powershell
.venv/Scripts/python.exe -B -m visualizations.c600_mcts_conditions
```

Exports are in [figures/c600_vs_mcts_conditions/](figures/c600_vs_mcts_conditions/README.md):
four PNGs at 300 dpi, four SVGs with editable text, a data CSV, provenance,
and a ZIP containing the individual figures and their supporting files.
The script checks all 1,600 game outcomes, source hashes, paired seats and
seed schedules, and independently reproduces the saved score intervals.

## MCTS learning-progression figures

The progression experiment evaluates C1 and C50–C600 at 50-round intervals
under all four conditions: neural/MCTS budgets 64/64, 128/128, 64/128, and
policy alone versus MCTS 64. Each condition contains 13 checkpoints × 400
games = 5,200 games. These four curves directly measure change with training;
the earlier C600 figures remain single-checkpoint diagnostics.

```powershell
.venv/Scripts/python.exe -B -m visualizations.mcts_learning_progression
```

The plotter requires the complete, verified 20,800-game experiment. It exports
four separate PNG/SVG figures with learner round on the x-axis, score against
plain MCTS on the y-axis, and opening-pair 95% intervals. See
[the experiment protocol](../experiments/checkpoint_strength/MCTS_PROGRESSION.md).

## Greedy H* benchmark figures

Two further figures cover the completed plain-MCTS comparison at 64 and 128
simulations (800 games) and the 13-checkpoint mixed-opening progression at
both neural search budgets (10,400 games). The plain-MCTS chart shows observed
outcomes and scores. The progression chart shows both score curves and the
matched-opening score difference, with newly computed pointwise 95% intervals
resampling whole opening pairs.

```powershell
.venv/Scripts/python.exe -B -m visualizations.greedy_hstar_benchmarks
```

[PNG/SVG figures and supporting data](figures/greedy_hstar_benchmarks/README.md)
include the source provenance, plotted CSVs, and a ZIP. All progression
outcomes, opening pairs, checkpoint hashes, and saved budget comparisons are
checked before plotting. The completed 64-budget reports are taken from the
consolidated sibling recovery results; interrupted partial games are excluded.

## Experiments available for design

| Experiment | Available evidence | Question to discuss |
| --- | --- | --- |
| Corrected handcrafted heuristic | Candidate development, weight selection, and held-out verification | Selection process, opponent-specific strengths, or verification of the selected evaluator? |
| Transfer-feature study | Transfer-feature development and final test summaries | Whether the added feature improves held-out performance? |
| EoH evolution and selection | Four generations, anchor panels, and finalist scores | Evolutionary progress, diversity of candidates, or why H* was selected? |
| Handcrafted versus EoH | Standard starts, development openings, and 400 fresh openings | Relative evaluator performance at each search depth, or sensitivity to the opening suite? |
| Opening-depth investigation | Diversity, duplicates, terminal paths, forced moves, and remaining seeds at depths 2-10 | Which opening depths provide diverse, playable benchmark positions? |
| AlphaZero training | L4 30-round run and A100 600-round run | Optimization behavior, self-play throughput, or training progression? |
| AlphaZero baseline evaluation | Completed L4 evaluations and varied-opening checkpoint reports | How playing strength changes with training round? |
| Mixed opening progression and plain-MCTS follow-up | Completed reports: 10,400 neural and 800 plain-MCTS games | Agreed figures show search budgets, checkpoint progression, and paired budget differences. |
| C50 versus C350 | 400 paired-opening games, seats, depths, and paired score intervals | Overall strength improvement or the opening positions responsible for it? |
| C350 versus C600 | 400 paired-opening games, seats, depths, and paired score intervals | Size and consistency of the observed difference between later checkpoints? |

## Source handling

Raw experiment results, game traces, training records, and opening datasets
remain the evidence for new figures. Development, selection, verification,
pilot, and historical runs have different roles. Their roles must be settled
for the selected experiment before choosing comparisons.

Minimax's cap-adjudicated outcomes need explicit treatment. Its Python and C++
timings are not a controlled implementation comparison. Checkpoint uncertainty
uses whole opening pairs. Training loss measures optimization, and needs its
own interpretation alongside playing-strength evaluations.

The dedicated scripts above reproduce the reviewed figures. The superseded
generic loader and batch plotters, together with their tests, were archived
locally during repository cleanup. See the
[maintenance guide](../docs/REPOSITORY_MAINTENANCE.md) for the archive inventory.
New rendering work starts after an experiment's figure has been selected together.


## Availability in a fresh clone

The reviewed PNG/SVG exports, plotted tables and figure provenance are included
in Git. See [the published experiment summaries](../docs/results/alpha_zero_600/README.md)
for the complete 13-checkpoint table. Full raw logs and models remain local;
commands above that reverify those records require the corresponding `runs/`
artifacts. No new cloud job is needed to view the published figures.
