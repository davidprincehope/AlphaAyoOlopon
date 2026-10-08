# Ayo checkpoint-strength opening dataset

For the four 13-checkpoint studies against plain MCTS, see
[MCTS_PROGRESSION.md](MCTS_PROGRESSION.md). Each condition uses 400 games per
checkpoint, for 5,200 games per curve and 20,800 games overall, followed by
four separate learning-progression plots.

This directory contains the opening dataset, its generation/validation tools,
and a separate opt-in configuration for the Modal checkpoint-strength rerun.

## Saved data

- `dataset/investigation_v1/candidates.jsonl.gz`: all **25,000 candidate trials**,
  including duplicate positions and early terminal paths.
- `dataset/investigation_v1/depth_analysis.json`: generation settings, source
  rules hash, file/content hashes, and statistics for all five depths.
- `dataset/ayo_fixed_v1.json`: **100 distinct nonterminal six-ply openings**,
  selected from the saved candidates as a proposed v1 benchmark.

Generation seed: `20261004`. Each of 5,000 trials per depth starts from the
normal 6-pit, 4-seed initial board. A private `random.Random(seed + depth)`
chooses uniformly from legal actions at each decision. A ply is one actual
`apply_action`; relay sowing is internal to that decision. The horizon is 1000
actions, with the existing collect cutoff.

## Depth investigation

| Target plies | Distinct live positions observed | Terminal trials | Duplicate live positions | Forced moves among live trials | Median seeds on board |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2 | 32 | 0.00% | 99.36% | 0.00% | 36 |
| 4 | 623 | 0.36% | 87.49% | 1.22% | 32 |
| 6 | 3,873 | 2.26% | 20.75% | 1.29% | 24 |
| 8 | 4,577 | 6.12% | 2.49% | 2.51% | 16 |
| 10 | 4,368 | 11.94% | 0.79% | 2.79% | 12 |

These are sampled counts, not exhaustive state-space counts. Distributions
use all eligible nonterminal trials before deduplication. The analysis also
contains legal-action histograms, capture totals, signed and absolute capture
differences, capture leaders, remaining seeds, and descriptive H* quantiles.
H* is not a solved value or a selection criterion.

Six plies is the proposed v1 choice: it offers substantially more variation
than four while retaining more playable material and fewer terminal/forced
trials than eight or ten. Two cannot supply 100 distinct positions. Four is
also viable for a shallower dataset; six is a design choice, not a proven optimum.

The proposed set uses the **first 100 distinct nonterminal visible positions**
in saved six-ply candidate order. No heuristic, win-rate, capture-balance, or
branching filter is applied. Duplicate identity is board, captures, and side
to move. The resulting set also contains 100 unique full states. It has 24
median remaining seeds and two forced-move positions. P0 leads captures in 32,
P1 in 45, and 23 are tied.

## Canonical representation

Each record saves its legal action sequence from the initial state, candidate
ID, target/actual plies, board, captures, side to move, legal actions,
terminal/truncation flags, repetition memory, and state/position hashes.
Replaying the actions restores elapsed plies and repetition history exactly;
assigning board fields alone would lose that state.

The candidate file has compressed and decompressed SHA-256 hashes. The frozen
set has a canonical JSON content hash and candidate/report provenance; the
report hash is independent of line endings. The loader checks every selected
opening's legality, snapshot, hashes, uniqueness, and nonterminal status,
as well as the engine source hash (normalized newlines), board, and horizon.

For a future 200-game checkpoint comparison, play each of the 100 openings once
per seat, keeping opening order identical across checkpoints. The two games
from each opening form a pair; they are not independent observations. These
random prefixes measure continuation strength, not opening choice by agents.

## Reproduce or create another set

Run from the repository root with new output paths; existing artifacts are
never overwritten. Inspect the depth report before choosing the freeze depth.

```powershell
.venv/Scripts/python.exe -m experiments.checkpoint_strength.generate_openings investigate --directory runs/checkpoint_strength/investigation-v2 --count 5000 --seed 20261004
.venv/Scripts/python.exe -m experiments.checkpoint_strength.generate_openings freeze --directory runs/checkpoint_strength/investigation-v2 --output runs/checkpoint_strength/ayo-openings-v2.json --depth 6 --count 100 --benchmark-id ayo_fixed_v2
```

Freeze reads saved candidates and does not resample. With identical seed,
engine, and options, action sequences and snapshots reproduce exactly.

Read and reconstruct the dataset without running any matches:

```python
import pyspiel
from Model.ayo_olopon import ayo_olopon
from experiments.checkpoint_strength.opening_dataset import DEFAULT_OPENINGS, load_dataset

game = pyspiel.load_game("ayo_olopon")
dataset = load_dataset(DEFAULT_OPENINGS, game)
state = dataset.openings[0].new_state(game)
```

```powershell
.venv/Scripts/python.exe -B -m pytest tests/checkpoint_strength/test_opening_dataset.py -q
```

## Modal checkpoint rerun

`configs/progression_fixed_v1.json` selects C1 and C50 through C600 at 50-round
intervals: 13 checkpoints, each playing all 100 saved openings once per seat
(200 games per checkpoint; 2,600 total). The opponent is GREEDY_HSTAR, with 64
neural simulations per move and seed `20261003`. Rules and the search constant
come from the saved training manifest. The default evaluation GPU remains L4.

The evaluator loads the dataset once and reuses the exact opening order and
paired seeds across checkpoints. Per-game records include opening IDs, hashes,
prefix actions, and continuation actions. `game_length` includes the opening;
`continuation_length` counts decisions by the evaluated agents. The six opening
plies consume the existing 1000-action horizon. Reports and the experiment
manifest retain the dataset hash and selected opening count.

The `openings` setting is opt-in. Configurations without it use standard starts;
training settings are unchanged. Strength comparisons use the frozen opening
datasets described here.
Smaller even game counts select a fixed prefix, labeled `prefix_smoke_test`.
More games than twice the saved opening count fail instead of cycling positions.

```powershell
$env:PYTHONUTF8 = '1'
$env:ALPHAZERO_EVAL_GPU = 'L4'
.venv/Scripts/python.exe -B -m modal run --detach experiments/alpha_zero/modal_evaluate.py --background --run-name a100-fresh-64-200-20261003 --evaluation-name checkpoint-strength-fixed-v1-20261004 --config experiments/checkpoint_strength/configs/progression_fixed_v1.json
```

The source Volume `alpha-ayo-alphazero-runs` is mounted read-only. New artifacts
go to `alpha-ayo-alphazero-evaluations`, under the chosen evaluation name. Use a
new evaluation name for subsequent reruns; existing files are never overwritten.
Download after completion:

```powershell
.venv/Scripts/python.exe -B -m modal volume get alpha-ayo-alphazero-evaluations checkpoint-strength-fixed-v1-20261004 runs/checkpoint_strength
```

The four evaluation artifacts are `results.jsonl`, `results.games.jsonl`,
`results.events.jsonl`, and `results.manifest.json`. A complete run has
`status=completed`, 13 completed checkpoints, and 2,600 completed games. Scores
from this opening benchmark should be reported separately from prior
standard-start scores.

## Mixed-depth 200-position comparison

`dataset/ayo_mixed_v2_200.json` freezes 200 globally distinct nonterminal
positions from the same saved candidate investigation. Depth allocations are
2:32, 4:42, 6:42, 8:42, 10:42. Only 32 distinct live two-ply positions were
observed, so the remaining positions are divided evenly among the other depths.
Selection takes the first eligible unique candidates and interleaves depths;
no strength or capture-balance filtering is applied.

Content SHA-256: `7b0e43e3621027927f14fdeb7d6016d171af27067136f32f85f3631e08182014`.

The two `configs/progression_mixed_v2_*.json` configurations differ only in
64 versus 128 simulations. Each checkpoint plays every opening in both seats:
400 games per checkpoint, 5,200 games per budget, 10,400 games overall. The
single GPU launcher finishes all 13 checkpoints at 64 before starting 128.
Opening order, paired seeds, opponent, and checkpoint selection are identical.

Each report includes the learner's wins/draws/losses and score in each seat,
plus `first_player_advantage`: P0/P1 outcomes combining both agent assignments,
P0 score minus 0.5, and the same statistics by opening depth. All saved prefixes
have even length, so P0 moves first from every opening. This measures first-player
advantage in these continuation positions; it does not establish the advantage
from the standard initial board. Score counts a draw as half a win.

```powershell
.venv/Scripts/python.exe -B -m experiments.checkpoint_strength.generate_openings freeze-mixed --directory experiments/checkpoint_strength/dataset/investigation_v1 --output runs/checkpoint_strength/another-mixed-set.json --allocation 2:32 4:42 6:42 8:42 10:42 --benchmark-id ayo_mixed_v2_200
$env:PYTHONUTF8 = '1'
$env:ALPHAZERO_EVAL_GPU = 'L4'
.venv/Scripts/python.exe -B -m modal run --detach experiments/checkpoint_strength/modal_compare.py --background --run-name a100-fresh-64-200-20261003 --evaluation-name checkpoint-strength-mixed-v2-200-64-128-20261004
```

Artifacts are saved on `alpha-ayo-alphazero-evaluations` under that evaluation
name. `64/` and `128/` each contain results, per-game records, events, and a
manifest. `comparison.manifest.json` records the active budget and committed
progress after each checkpoint. `budget_comparison.json` contains per-checkpoint
win-rate and score changes in percentage points, together with both seat reports.
The source checkpoints are mounted read-only; existing results are never reused.

```powershell
.venv/Scripts/python.exe -B -m modal volume get alpha-ayo-alphazero-evaluations checkpoint-strength-mixed-v2-200-64-128-20261004 runs/checkpoint_strength
```

For an interrupted 64-stage run, launch with a new evaluation name and
`--resume-from checkpoint-strength-mixed-v2-200-64-128-20261004`. Recovery
checks the configuration, dataset, evaluation source hashes, training manifest,
and checkpoint hashes before reusing complete checkpoint reports. It reruns
unfinished checkpoints in full under `64-recovery/`; partial games remain in
the original directory and do not contribute to the consolidated result.
The new `64/` results combine reused and recovered checkpoints before 128 begins.

## Plain MCTS follow-up

`modal_mcts_baseline.py` queues a CPU-only baseline after the neural comparison
function returns successfully. It also checks the saved comparison is complete.
Failure of that dependency prevents the baseline from starting.

The agent is the existing `Algorithms/ayo_mcts.py`: UCT, uniform legal-action
priors, one uniformly random rollout to terminal per evaluated leaf,
`solve=False`, and seeded OpenSpiel `MCTSBot.step` action selection. It loads
no neural policy or value model. The exploration constant, finite-horizon game,
dataset, seed schedule, and GREEDY_HSTAR opponent are taken from the completed
neural experiment. This is a vanilla-MCTS comparison, changing both tree
selection (PUCT to UCT) and leaf evaluation; it is not a policy-only ablation.
Equal simulation counts do not imply equal runtime or evaluation cost.

There are no checkpoint repetitions: 400 games at 64 and 400 at 128, using the
same 200 openings in both seats. Results and first-player measures are saved
under `benchmark/`; per-game logs include prefixes and continuation actions.
The Modal volume is committed every 20 games and at each completed budget.

## C50 versus C350 directly

`configs/head_to_head_c50_c350_64.json` plays the two frozen neural checkpoints
against each other at 64 PUCT simulations per agent per move. It uses the same
200 mixed-depth openings and seed `20261003`: each position is played once with
C50 in P0 and once with C350 in P0, for 400 games. Both models use the saved
training rules, horizon, and exploration constant, with no noise or temperature
sampling. This measures continuation strength on the saved positions.

```powershell
$env:PYTHONUTF8 = '1'
$env:ALPHAZERO_EVAL_GPU = 'L4'
.venv/Scripts/python.exe -B -m modal run --detach experiments/checkpoint_strength/modal_head_to_head.py --background --evaluation-name checkpoint-head-to-head-c50-c350-64-20261005
```

For C350 versus C600 with the identical 400-game protocol, add
`--config experiments/checkpoint_strength/configs/head_to_head_c350_c600_64.json`
and use evaluation name `checkpoint-head-to-head-c350-c600-64-20261005`.

Checkpoints are mounted read-only. A new evaluation directory is required on
every run. `results.games.jsonl` records both model assignments, opening hashes,
prefixes, seeds, continuation actions, and outcomes. `results.json` reports each
checkpoint's wins/draws/losses and score, per-seat and per-depth results, and
first-player advantage. The 95% percentile bootstrap intervals resample whole
opening pairs (10,000 resamples); they describe variation across the sampled
positions and do not cover training-seed or search-seed uncertainty.

`results.manifest.json` stores both checkpoint hashes, training manifest and
evaluation source hashes, dataset provenance, progress, and final artifact
hashes. The Modal volume is committed every 20 games and at completion.

```powershell
.venv/Scripts/python.exe -B -m modal volume get alpha-ayo-alphazero-evaluations checkpoint-head-to-head-c50-c350-64-20261005 runs/checkpoint_strength
```

```powershell
$env:PYTHONUTF8 = '1'
.venv/Scripts/python.exe -B -m modal run --detach experiments/checkpoint_strength/modal_mcts_baseline.py --evaluation-name checkpoint-strength-plain-mcts-20261005 --after-evaluation checkpoint-strength-mixed-v2-recovery-20261005 --after-call fc-01M459V86J4QDWBZR212C4WJV3
```

## C600 versus plain MCTS directly

`configs/c600_vs_mcts_64.json` plays frozen C600 against the existing vanilla
UCT agent at **64 simulations per move for both agents**. It uses all 200
mixed-depth positions, both seat assignments (400 games), seed `20261003`,
and the training rules, 1000-action horizon, and exploration constant (1.5).
MCTS uses one uniform random rollout per evaluated leaf, no neural network,
and `solve=False`. C600 uses its frozen policy/value with PUCT, no noise, and
maximum-visit action selection. Equal simulations do not imply equal time or
compute; this changes tree selection as well as leaf evaluation.

Eight spawned processes execute complete opening pairs on one shared GPU with
CUDA MPS. Global opening indices determine seeds and IDs, independently of
worker scheduling. Final records are sorted by game ID. Only the parent writes
and commits results, every 20 games and on completion/failure. Source checkpoints
are mounted read-only. A new evaluation directory is required for each run.

```powershell
$env:PYTHONUTF8 = '1'
$env:ALPHAZERO_EVAL_GPU = 'A100-40GB'
$env:ALPHAZERO_EVAL_CPU = '10'
$env:ALPHAZERO_EVAL_MEMORY_MIB = '32768'
.venv/Scripts/python.exe -B -m modal run --detach experiments/checkpoint_strength/modal_neural_vs_mcts.py --background --evaluation-name checkpoint-c600-vs-mcts-64-20261005
```

For the identical paired experiment at 128 simulations per move for both
agents, use `--config experiments/checkpoint_strength/configs/c600_vs_mcts_128.json`
and a new evaluation name, such as `checkpoint-c600-vs-mcts-128-20261005`.
Checkpoint, opening order, paired seeds, search constants, and resources are
identical; only the simulation budget changes.

For **C600 at 64 versus MCTS at 128**, use
`--config experiments/checkpoint_strength/configs/c600_64_vs_mcts_128.json`
and evaluation name `checkpoint-c600-64-vs-mcts-128-20261005`.
`simulations` controls C600; optional `mcts_simulations` controls the opponent.
If omitted or null, MCTS uses `simulations`, preserving the earlier equal-budget
configurations. The manifest records each agent's effective budget separately.

For **direct C600 policy predictions versus MCTS at 64**, use
`--config experiments/checkpoint_strength/configs/c600_policy_vs_mcts_64.json`
and evaluation name `checkpoint-c600-policy-vs-mcts-64-20261005`.
`neural_mode=policy` requires `simulations=0` and an explicit MCTS budget.
C600 makes one prediction of the current state per decision, selecting the
highest-probability legal move, with the lowest action ID breaking ties.
It performs no tree search, successor evaluation, noise, or sampling, and the
value head does not select actions. The same frozen policy/value network is
loaded; its value output is computed but ignored. Existing configurations
default to `neural_mode=puct` and preserve their previous search behavior.

Artifacts on `alpha-ayo-alphazero-evaluations/<evaluation-name>/` are
`results.games.jsonl`, `results.json`, and `results.manifest.json`. They include
checkpoint/source/artifact hashes, hardware, both policies, per-seat/per-depth
scores, first-player advantage, and 95% opening-pair bootstrap score intervals.
The intervals cover sampled-position variation, not training/search-seed
uncertainty. The Modal function has a one-hour timeout and exits after committing.
