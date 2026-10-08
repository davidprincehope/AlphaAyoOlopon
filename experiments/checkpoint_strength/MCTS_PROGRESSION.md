# AlphaZero checkpoint progression against plain MCTS

These four studies measure how playing strength changes across C1 and C50–C600
at 50-round intervals. Each condition evaluates all 13 checkpoints. Each
checkpoint plays 400 games on the same frozen 200 mixed-depth openings, with
both seat assignments and the same paired seed schedule.

| Condition | AlphaZero action rule | Plain MCTS budget | Games |
| --- | --- | ---: | ---: |
| `puct64-mcts64` | Neural PUCT, 64 simulations | 64 | 5,200 |
| `puct128-mcts128` | Neural PUCT, 128 simulations | 128 | 5,200 |
| `puct64-mcts128` | Neural PUCT, 64 simulations | 128 | 5,200 |
| `policy-mcts64` | Highest-probability legal policy move, no search | 64 | 5,200 |

Total: 52 checkpoint-condition cases and 20,800 games. The opponent is the
existing UCT implementation, with uniform legal priors, one random rollout
per leaf, and `solve=False`. Neural search uses the frozen learned policy and
value with no evaluation noise or temperature sampling. Policy-only selection
ignores the value head. Rules, horizon, and exploration constant come from the
saved training manifest. No additional training occurs.

The source volume is mounted read-only. All four conditions use identical
checkpoint hashes, opening IDs, prefix histories, seats and paired seeds.
Scores count draws as half a win. Saved 95% percentile bootstrap intervals
resample complete opening pairs 10,000 times and describe sampled-position
variation, excluding training-seed and search-seed uncertainty. Equal search
budgets do not equate runtime or compute.

Launch using a new evaluation name after local and copied-source preflight:

```powershell
$env:MODAL_PROFILE = 'davidprincehope'
$env:ALPHAZERO_EVAL_GPU = 'A100-40GB'
$env:ALPHAZERO_EVAL_CPU = '10'
$env:ALPHAZERO_EVAL_MEMORY_MIB = '32768'
.venv/Scripts/python.exe -B -m modal run --detach experiments/checkpoint_strength/modal_neural_vs_mcts.py --background --progression --run-name a100-checkpoints-c600-20261006 --evaluation-name checkpoint-mcts-progression-13x400-20261006-v1
```

At most two GPU containers execute checkpoint cases; each case is finite and
the finalizer waits for all 52. Each case commits progress every 20 games and
saves its own results, game log and manifest. The finalizer writes a progression
curve per condition and a combined result, marking completion only after all
13 checkpoints in all four conditions finish.

Collect and verify the launched experiments:

```powershell
.venv/Scripts/python.exe -B -m experiments.checkpoint_strength.collect_mcts_progression --evaluation-name checkpoint-mcts-progression-13x400-20261006-v1 --watch
```

The collector checks artifact hashes, reconstructs each legal game, checks
terminal returns, captures and length, recomputes scores and paired intervals,
and confirms identical opening schedules across all cases. Downloaded data is
saved under `runs/checkpoint_strength/<evaluation-name>/` with a live
`status.json`. A complete verified run contains `progression.results.json`,
`progression.manifest.json` and `verification.json`, plus 52 case directories.

Generate the four individual progression charts after verified completion:

```powershell
.venv/Scripts/python.exe -B -m visualizations.mcts_learning_progression
```

For automatic chart generation while the collector runs, start the local
completion helper in a separate process:

```powershell
.venv/Scripts/python.exe -B -m experiments.checkpoint_strength.finish_mcts_progression --evaluation-name checkpoint-mcts-progression-13x400-20261006-v1
```

It waits for the completed local verification, renders all four charts, checks
PNG dimensions, editable SVG text and ZIP integrity, and records its result
in `chart_status.json` beside the experiment's live `status.json`.

The plotter requires all 20,800 games to be complete and verified. It exports
four PNG/SVG curves with the same score scale and 95% intervals, a 52-row data
CSV, provenance and a ZIP. The earlier single-checkpoint C600 charts remain
endpoint diagnostics; they are not learning-progression measurements.


## Completed study

The 6 October 2026 study completed all 52 cases and all 20,800 games. Local
replay verification passed, and all four individual progression charts were
exported and visually checked. The [published summaries and checkpoint table](../../docs/results/alpha_zero_600/README.md)
and [PNG/SVG charts with plotted data](../../visualizations/figures/mcts_learning_progression_13_checkpoints/README.md)
are included in Git. Raw per-game logs, model exports and launch records remain
in the ignored local run directory.
