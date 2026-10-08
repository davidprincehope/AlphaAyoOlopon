# Ayo Olopon AlphaZero scaffold

This connects the existing Python Ayo rules to the checked-out OpenSpiel
**JAX/Flax AlphaZero** trainer: neural policy/value MCTS, self-play actors,
replay buffer, learner, checkpoint broadcasts, and per-round baseline evaluation.
The original `ayo_olopon` rules are preserved. The local OpenSpiel trainer includes
the canonical Ayo model and complete learner snapshot/resume support.

## Run it

For a single-node A100 with 28 self-play actors on Modal, see
[the Modal launcher and timing guide](MODAL.md).

Run these commands from the repository root, with Python 3.12+ and the local
`open_spiel/` checkout present. The verified environment is the existing
`.venv` with CPython 3.14.3 on Windows CPU.

```powershell
.venv/Scripts/python.exe -m pip install -r requirements-alpha-zero.txt
.venv/Scripts/python.exe -m experiments.alpha_zero.train --dry-run
.venv/Scripts/python.exe -m pytest tests/alpha_zero -q
.venv/Scripts/python.exe -m experiments.alpha_zero.train --config experiments/alpha_zero/configs/smoke.json --output runs/alpha_zero/smoke
.venv/Scripts/python.exe -m experiments.alpha_zero.evaluate --run runs/alpha_zero/smoke --checkpoint 1 --games 20 --simulations 64
```

On Linux/macOS use your virtual environment's `python` in place of
`.venv/Scripts/python.exe`. `--dry-run` validates the game/configuration and
prints provenance without loading JAX or starting workers. The smoke preset
runs one learner round; it tests the pipeline and does not produce a strong agent.

For the complete Stage 7 integration probe, use the dedicated preset and
diagnostic. These are **SANITY TEST VALUES — NOT EXPERIMENTAL HYPERPARAMETERS**.
They cap games at four moves to keep the run small; those games end by horizon
collection, not by natural exhaustion. Give each command a fresh output path.

```powershell
.venv/Scripts/python.exe -m experiments.alpha_zero.train --config experiments/alpha_zero/configs/sanity.json --output runs/alpha_zero/sanity_train
.venv/Scripts/python.exe -m experiments.alpha_zero.sanity_check --output runs/alpha_zero/sanity_diagnostic
.venv/Scripts/python.exe -m experiments.alpha_zero.evaluate --run runs/alpha_zero/sanity_train --checkpoint 2 --games 2 --simulations 4 --opponent random
```

The diagnostic writes `sanity_report.json` with state, search, replay, loss,
gradient, checkpoint, and resumed-training evidence. The full trainer separately
checks that actor processes load new checkpoints. The two-game evaluation is an
execution check and says nothing about playing strength.

After reviewing the decisions below, start a larger experiment:

```powershell
.venv/Scripts/python.exe -m experiments.alpha_zero.train --config experiments/alpha_zero/configs/starter.json --output runs/alpha_zero/starter
.venv/Scripts/python.exe -m experiments.alpha_zero.evaluate --run runs/alpha_zero/starter --checkpoint 100 --opponent mcts --opponent-simulations 64 --games 100 --simulations 64
```

Each fresh output directory must be new. Omitting `--output` generates a timestamped
run directory. Runs contain `manifest.json` (resolved settings, package versions,
source revision, encoding), OpenSpiel's `config.json`, `learner.jsonl`, process
logs under `sessions/<session-id>/`, and `checkpoint-N` directories. `checkpoint--1` is the rolling checkpoint;
the offline curve command selects immutable numbered rounds. Checkpoints are saved on every learner round;
`checkpoint_freq` selects which rounds get a retained numbered checkpoint.

The launcher selects `open_spiel/open_spiel/python` explicitly because the outer
checkout directory otherwise shadows the installed Python package. The installed
OpenSpiel wheel supplies `pyspiel`. Custom game registration and explicit game
serialization run in Windows/spawn workers as well as the parent. Worker failures
are raised during queue polling instead of leaving the learner waiting forever.

## Training decisions and remaining checks

1. **Rule variant — confirmed.** The adapter inherits the current six-house,
   four-seed Python model: relay sowing; intermediate fours captured by the row
   owner; final-seed fours by the mover; feeding constraints; and remaining-seed
   collection by row on repetition, relay cycles, and no legal moves. Rewards
   are win/draw/loss `+1/0/-1`, not seed margin. This is the variant selected for
   training. Other board sizes and the C++ model are outside this scaffold.
2. **Artificial horizon — confirmed.** `max_moves=1000`
   individual player moves. At the limit, the game awards remaining seeds by
   row and determines the result from the final seed counts. The base game also
   enforces its configurable `max_game_length`; this is an experiment rule, not
   a claim about traditional Ayo. Natural terminal results take precedence.
3. **Observation — confirmed.** Keep the compact 15-feature input, which
   omits the set of positions seen since the last capture. MCTS clones preserve
   that set and apply the original rule exactly, but identical observations can
   have different history-dependent futures. We accept this approximate value
   input for the planned run. A history encoder would require a new observation
   version and fresh training; a history count alone would not fully solve it.
4. **Hardware and time budget — deferred to the training machine.** The current
   PC is not the intended training host. The starter settings are CPU starting
   points, not a runtime commitment. Profile states/second and full evaluation
   rounds on the actual host before choosing a substantial run duration. Native
   Windows NVIDIA GPU is not supported by JAX; use a supported Linux setup or consult the
   WSL2 guidance in the [JAX installation guide](https://docs.jax.dev/en/latest/installation.html).
   This upstream trainer is single-device, not distributed multi-GPU training.
   Multiple workers can compete for device memory. No device environment variables
   are silently changed by this scaffold.
5. **Evaluation milestone — confirmed.** Built-in training evaluation uses
   the shared Ayo experiment match runner after every completed learner round.
   RAND for 100 games after each round, balanced 50 per seat, with the same neural
   MCTS simulation budget as self-play. Score rate of at least 95% against RAND
   for three consecutive checkpoints is the signal to add GREEDY_HSTAR to
   retrospective evaluation. This is a reporting milestone only: it does not
   stop training, gate model updates, or automatically launch GREEDY_HSTAR matches.
   Existing Greedy, Minimax, and MCTS agents remain configurable for evaluation.
6. **Training continuation — confirmed.**
   The upstream trainer does not expose one seed covering network initialization,
   actor sampling, MCTS, replay, and multiprocessing scheduling. This scaffold
   records configuration/versions but does not claim deterministic training.
   `evaluate --seed` controls evaluation randomness. Loading a checkpoint for
   inference is supported. Complete training snapshots restore model parameters,
   Adam state, replay contents and sampling RNG, learner step, cumulative games,
   and learner evaluation windows. Workers restart with fresh games. In-flight
   games, unconsumed queues, and worker RNG states are discarded, so resume
   continues training state without reproducing the uninterrupted execution.

## Resume training

The launcher writes complete snapshots to `training-checkpoints/step-NNNNNN/`
every `resume_checkpoint_freq` rounds (default 10), and always at the final round.
These include a model checkpoint, replay arrays and ring bookkeeping, learner
counters, evaluation windows, and versioned metadata with file hashes. A snapshot
is published only after all files are written; interrupted `.pending-*` directories
are ignored. Older complete snapshots are retained. Model-only `checkpoint-N`
directories from earlier versions cannot resume the learner.

```powershell
# Continue a run that stopped at step 100, through total step 200.
.venv/Scripts/python.exe -m experiments.alpha_zero.train --resume runs/alpha_zero/starter --max-steps 200

# Branch from an older complete snapshot into a new run directory.
.venv/Scripts/python.exe -m experiments.alpha_zero.train --resume runs/alpha_zero/starter --resume-step 100 --output runs/alpha_zero/branch --max-steps 200

# Restart fresh self-play workers with three actors at the next learner round.
.venv/Scripts/python.exe -m experiments.alpha_zero.train --resume runs/alpha_zero/starter --actors 3 --max-steps 200
```

`--max-steps` is the total target learner step, not the additional number of
rounds. `0` means unlimited. The saved configuration is restored; the target
step, per-round evaluation game count, and self-play actor count may change via
CLI overrides. `--actors` changes only the number of fresh worker processes;
the model, replay, optimizer, and completed learner step still come from the
snapshot. The effective count is recorded in the new session and subsequent
snapshots. The run-level `manifest.json` retains the initial session settings.
`--evaluation-games 0` skips future per-round matches while continuing to export
each trained model for offline evaluation. An explicitly supplied `--config` must
otherwise match the saved settings. A target
already reached is rejected before workers start. Architecture, game, observation,
and snapshot format compatibility are checked. `--dry-run` also validates resume
selection without starting workers or modifying run files.

The learner restores before workers start. Workers wait for their assigned model
before generating a game. Resume skips checkpoint 0 and starts at the saved
completed step plus one. Ctrl+C requests a graceful stop after the current round,
with a final complete snapshot; an abrupt kill loses work since the last complete
snapshot.

To stop RAND matches in an ongoing run, press Ctrl+C once and wait for the current
round (including its evaluation) to finish and publish a complete snapshot. Pull
the updated code, then resume with
`--resume runs/alpha_zero/starter --max-steps 200 --evaluation-games 0`.
This does not change self-play, replay, optimizer updates, or the total target.

`learner.jsonl` remains the canonical training history. Each invocation records a
session ID, source snapshot, target step, and settings in `sessions/`. During
same-run recovery, history beyond the restored snapshot is removed from the
canonical curve but preserved in that session's `previous-learner.jsonl`; newer
model checkpoints and the rolling checkpoint are also archived there. Resuming
an older snapshot requires a new output directory. The original manifest is kept;
session records and each snapshot record the effective target for that invocation.

## Hyperparameters

Edit `configs/starter.json`; omitted keys use `Algorithms/alpha_zero/config.py`.
These are **untuned starting values**, not an Ayo optimum.

| Setting | Starter | Meaning / decision |
| --- | ---: | --- |
| `nn_model`, `nn_api_version` | `ayo_mlp`, `linen` | Canonical 15-input policy-value model. NNX is not supported for this architecture. |
| `nn_width`, `nn_depth` | 256, 3 | Fixed three-layer shared trunk; heads are independently 128 and 64 units wide (185,863 parameters total). |
| `max_simulations` | 64 | MCTS simulations per move, including the root visit; must be at least 2. |
| `uct_c` | 1.5 | PUCT exploration constant for neural search, despite the upstream name. |
| `policy_alpha`, `policy_epsilon` | 1.0, 0.25 | Symmetric Dirichlet concentration and root-noise mixing weight during self-play. |
| `temperature`, `temperature_drop` | 1.0, 20 | Sample from powered visit counts for the first 20 individual moves, then choose the best child. Temperature must remain positive because upstream still computes policy targets with it. |
| `learning_rate` | 0.001 | Constant Adam learning rate. Optax defaults: betas 0.9/0.999, epsilon 1e-8; no clipping, optimizer weight decay, or schedule. |
| `weight_decay`, `decouple_weight_decay` | 0.0001, false | The canonical model uses Adam plus explicit `weight_decay × sum(weight²)` over dense kernels only; AdamW is rejected. |
| `train_batch_size` | 128 | Positions per gradient update. |
| `replay_buffer_size` | 16384 | Retained positions, not games; the ring overwrites the oldest position when full. |
| `replay_buffer_reuse` | 4 | Collect at least `buffer_size // reuse` new positions per learner round, then take `len(buffer) // batch_size` updates. This is not a boolean despite the upstream annotation. |
| `max_steps` | 100 | Learner rounds, not games or individual gradient updates. Zero runs until interrupted. |
| `actors`, `evaluators` | 2, 0 | Self-play workers and optional legacy asynchronous rollout-MCTS diagnostic workers. Per-round evaluation runs independently of `evaluators`. |
| `evaluation_games`, `evaluation_seed` | 100, 0 | Per-round matches; games must be even. Zero skips matches while retaining per-round model exports. Smoke/sanity use 2 games. |
| `evaluation_opponent` | `{"name":"RAND","params":{}}` | Existing Ayo baseline and its constructor parameters. |
| `checkpoint_freq` | 10 | Retain numbered checkpoints every 10 rounds. |
| `resume_checkpoint_freq` | 10 | Save complete learner snapshots every 10 rounds, and at the final/graceful-stop round. |
| `eval_levels`, `evaluation_window` | 3, 50 | Rollout-MCTS budgets scale as `max_simulations * 10^(level/2)`; average recent evaluation results. `eval_levels >= 2` is required even with zero evaluators due to upstream diagnostics. |
| `max_moves`, `cutoff` | 1000, `collect` | Enforced training horizon and seed-count adjudication, described above. `cutoff` currently supports `collect` only. |

## Verification and upstream metric caveat

Verified on Windows CPU: 20 scaffold tests passed; complete self-play and gradient
updates; checkpoints 0, 1, and 2; concurrent actor/evaluator workers; checkpoint
reload; and balanced-seat evaluation against random play and rollout MCTS.
The intermediate-capture test uses a seed-conserving fixture that leaves a legal
next move, so it checks sowing capture without triggering the feeding end rule.
These checks establish that the pipeline runs, not that the trained policy is strong.

The checked-out upstream evaluator averages the entire allocated results buffer,
including unused slots before the evaluation window fills. Its early reported
averages can therefore be misleading. Use the standalone evaluation command's
counts and score rate for comparisons; it averages only completed games.

## Observation and value contract

Registered training game: `ayo_olopon_alpha_zero`.

| Indices | Contents |
| --- | --- |
| 0..5 | Current player's pits in local action order, divided by 48 |
| 6..11 | Opponent's pits in sowing order, divided by 48 |
| 12..13 | Current player's and opponent's captured seeds, divided by 48 |
| 14 | Remaining moves divided by `max_moves` |

Action IDs `0..5` select the current player's local pit. The policy head has six
outputs and OpenSpiel masks illegal actions. Each replay position uses the
final return of the player recorded as to move at that position. The raw value
prediction has that same player-to-move perspective; the evaluator converts it
to a player-indexed vector only for OpenSpiel MCTS. Search Q values belong to
the player who chose the edge. Terminal observations use the explicitly
requested player as the perspective, e.g. `state.observation_tensor(0)`.
Run manifests record `value_perspective=player_to_move`; older checkpoints are
rejected. The Python replay buffer exists only in memory during a run. Training
samples uniformly **with replacement** from completed games. Its minibatch
contains float32 `[B,15]` observations, float32 `[B,6]` policies, float32
`[B]` values, and bool `[B,6]` legal masks. Policy targets use normalized
`visit_count ** (1 / temperature)` even after action choice becomes greedy.

## Files and extension points

- `Algorithms/alpha_zero/game.py`: adapter, observer, horizon, worker serialization.
- `Algorithms/alpha_zero/config.py`: typed settings and cross-field validation.
- `Algorithms/alpha_zero/runtime.py`: local source selection and worker failure reporting.
- `Algorithms/alpha_zero/bot.py`: load a checkpoint as an OpenSpiel PUCT bot.
- `experiments/alpha_zero/train.py`: upstream trainer launcher and run manifest.
- `experiments/alpha_zero/evaluate.py`: balanced-seat checkpoint evaluation.
- `tests/alpha_zero/test_scaffold.py`: rules parity, seed conservation, observations,
  horizon returns, clone/serialization isolation, configuration, dry run, worker failure.

The CPU requirements pin direct dependencies validated here; manifests record their
resolved versions. OpenSpiel source revision at implementation:
`48401890ee9857e611678302371378175a8e4c6b`. Do not assume another checkout's
AlphaZero Config, value convention, or checkpoint format is interchangeable.

References: [upstream Python AlphaZero](https://github.com/google-deepmind/open_spiel/blob/master/open_spiel/python/algorithms/alpha_zero/README.md),
[OpenSpiel installation](https://openspiel.readthedocs.io/en/latest/install.html).

## Per-round and retrospective evaluation

For the completed 600-round run's checkpoint comparisons, use the
[varied-opening evaluation guide](../checkpoint_strength/README.md) and
[plain-MCTS progression protocol](../checkpoint_strength/MCTS_PROGRESSION.md).

With a positive `evaluation_games`, every completed learner round evaluates its resulting parameters through
`experiments.agent_benchmark.random_vs_greedy_hstar.run_matches`. The same runner
is used by the offline command. The default is 100 matches against RAND, with
50 games in each player seat. Both seats in a pair share a seed. Neural MCTS uses
the saved self-play simulation budget, no root noise, and maximum-visit action
selection (lowest action ID breaks ties).

Change the baseline in the training JSON, for example:

```json
"evaluation_games": 100,
"evaluation_seed": 0,
"evaluation_opponent": {
  "name": "MINIMAX",
  "params": {"maximum_depth": 4}
}
```

Supported agents reuse the existing implementations: RAND, GREEDY_HSTAR,
GREEDY/HEURISTIC (params `heuristic`, such as `H_CTM`, and optional `weights`),
MINIMAX (`maximum_depth`), and MCTS (`simulations`, `rollouts_per_leaf`, `uct_c`,
optional `seed`). A compatible additional agent can specify
`{"name":"CUSTOM","factory":"package.module:make_agent","params":{...}}`;
its factory receives `game`, `seed`, and the parameters and returns an object
with `step(state)` that selects a legal action without changing the supplied state.

`evaluation.jsonl` records one report per completed round: checkpoint/learner step,
opponent specification, seed, search budget, wins/draws/losses, win rate,
score rate `(wins + 0.5*draws)/games`, scores for both seats, average length,
termination counts, and the existing detailed match records/statistics.
Evaluation uses a separate inference view and private RNGs. It adds no replay
positions, performs no optimizer updates, and applies no acceptance or stopping
threshold. Failures produce error reports and training continues. Matches execute
synchronously after training each round, so they add wall-clock time; asynchronous
actors can continue producing self-play while measurement runs.

Each round also writes an atomic immutable parameter-only export to
`inference-checkpoints/step-NNNNNN.npz`, independently of the full training snapshot
interval. These exports contain no optimizer or replay state. The offline loader
also supports historical Orbax model checkpoints and complete training snapshots
by partially restoring only `params`. It never creates a training model, replay
buffer, or optimizer.

```powershell
# One saved learner round, default self-play search budget.
.venv/Scripts/python.exe -m experiments.alpha_zero.offline_evaluate --run runs/alpha_zero/starter --checkpoint 100 --games 100 --opponent RAND --output runs/alpha_zero/rand-round100.jsonl

# All retained rounds from 10 through 100 in increments of 10, versus MCTS.
.venv/Scripts/python.exe -m experiments.alpha_zero.offline_evaluate --run runs/alpha_zero/starter --checkpoints 10:100:10 --games 100 --opponent-config experiments/agent_benchmark/configs/mcts.json --output runs/alpha_zero/mcts-curve.jsonl

# A stronger retrospective search budget against Greedy H*.
.venv/Scripts/python.exe -m experiments.alpha_zero.offline_evaluate --run runs/alpha_zero/starter --all-checkpoints --games 200 --simulations 400 --opponent GREEDY_HSTAR --output runs/alpha_zero/greedy-curve.jsonl
```

`--checkpoints` accepts individual rounds, comma-separated lists, and inclusive
`START:END[:STRIDE]` ranges. Explicitly requested missing checkpoints cause an
error, except C1 when `--fallback-first` is enabled. `--all-checkpoints` selects
available retained trained rounds (excluding C0). Historical runs only
have rounds they actually retained. Mutable rolling checkpoint `-1` is excluded
from curves because it does not reliably identify a learner round. Reports are
flushed after each checkpoint into a new JSONL file; per-game/events JSONL files
and an experiment manifest are also saved. Existing output files are
not overwritten. Use `--opponent-config` for parameterized agents on PowerShell
to avoid native-command JSON quoting issues. `experiments.alpha_zero.evaluate`
is a compatibility entry point for this same command.

Offline evaluation can read an ongoing run's already published immutable exports.
On resume after round 100, the next normal evaluation is round 101. Recovery trims
canonical evaluation history to the restored round and archives later exports
and records with the abandoned session, alongside learner history. No evaluation
of round 100 is automatically repeated just because it was restored.
