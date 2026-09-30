# Ayo Olopon AlphaZero scaffold

This connects the existing Python Ayo rules to the checked-out OpenSpiel
**JAX/Flax AlphaZero** trainer: neural policy/value MCTS, self-play actors,
replay buffer, learner, checkpoint broadcasts, and rollout-MCTS evaluators.
The original `ayo_olopon` game and vendored OpenSpiel files are not changed.

## Run it

Run these commands from the repository root, with Python 3.12+ and the local
`open_spiel/` checkout present. The verified environment is the existing
`.venv` with CPython 3.14.3 on Windows CPU.

```powershell
.venv/Scripts/python.exe -m pip install -r requirements-alpha-zero.txt
.venv/Scripts/python.exe -m experiments.alpha_zero.train --dry-run
.venv/Scripts/python.exe -m pytest tests/alpha_zero -q
.venv/Scripts/python.exe -m experiments.alpha_zero.train --config experiments/alpha_zero/configs/smoke.json --output runs/alpha_zero/smoke
.venv/Scripts/python.exe -m experiments.alpha_zero.evaluate --run runs/alpha_zero/smoke --checkpoint 1 --games 20 --simulations 100
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
.venv/Scripts/python.exe -m experiments.alpha_zero.evaluate --run runs/alpha_zero/starter --checkpoint 100 --opponent mcts --opponent-simulations 100 --games 100 --simulations 100
```

Each output directory must be new. Omitting `--output` generates a timestamped
run directory. Runs contain `manifest.json` (resolved settings, package versions,
source revision, encoding), OpenSpiel's `config.json`, `learner.jsonl`, process
logs, and `checkpoint-N` directories. `checkpoint--1` is the rolling checkpoint;
use `--checkpoint -1` to load it. Checkpoints are saved on every learner round;
`checkpoint_freq` selects which rounds get a retained numbered checkpoint.

The launcher selects `open_spiel/open_spiel/python` explicitly because the outer
checkout directory otherwise shadows the installed Python package. The installed
OpenSpiel wheel supplies `pyspiel`. Custom game registration and explicit game
serialization run in Windows/spawn workers as well as the parent. Worker failures
are raised during queue polling instead of leaving the learner waiting forever.

## Decisions to make before a substantial training run

1. **Confirm the rule variant.** The adapter inherits the current six-house,
   four-seed Python model: relay sowing; intermediate fours captured by the row
   owner; final-seed fours by the mover; feeding constraints; and remaining-seed
   collection by row on repetition, relay cycles, and no legal moves. Rewards
   are win/draw/loss `+1/0/-1`, not seed margin. Confirm these are the rules you
   want to learn. Other board sizes and the C++ model are outside this scaffold.
2. **Choose the artificial horizon and its result.** Default: `max_moves=1000`
   individual player moves. At the limit, the game awards remaining seeds by
   row and determines the result from the final seed counts. The base game also
   enforces its configurable `max_game_length`; this is an experiment rule, not
   a claim about traditional Ayo. Natural terminal results take precedence.
3. **Decide whether the network needs repetition history.** The compact input
   omits the set of positions seen since the last capture. MCTS clones preserve
   that set and apply the original rule exactly, but identical observations can
   have different history-dependent futures. This is an approximate value input,
   not a fully Markov encoding. A history encoder would require a new observation
   version and fresh training; a history count alone would not fully solve it.
4. **Choose hardware and a time budget.** Defaults are small CPU starting points.
   Profile states/second before increasing workers or simulations. Native Windows
   NVIDIA GPU is not supported by JAX; use a supported Linux setup or consult the
   WSL2 guidance in the [JAX installation guide](https://docs.jax.dev/en/latest/installation.html).
   This upstream trainer is single-device, not distributed multi-GPU training.
   Multiple workers can compete for device memory. No device environment variables
   are silently changed by this scaffold.
5. **Define success and evaluation opponents.** Built-in training evaluation uses
   rollout MCTS+Solver, not the handcrafted/EOH/minimax agents or a best-checkpoint
   promotion tournament. The standalone evaluator supports random or rollout MCTS
   without Solver, alternates seats, and reports score `(wins + draws/2)/games`.
   Choose opponent budgets, held-out openings, number of games, and a success
   threshold. A few smoke-test wins are not evidence of strength.
6. **Decide whether exact reproducibility or training resumption is required.**
   The upstream trainer does not expose one seed covering network initialization,
   actor sampling, MCTS, replay, and multiprocessing scheduling. This scaffold
   records configuration/versions but does not claim deterministic training.
   `evaluate --seed` controls evaluation randomness. Loading a checkpoint for
   inference is supported; full resume of replay, RNG, workers, and learner step
   is not implemented. Every training invocation starts fresh.

### Existing rule-test discrepancy to resolve

`tests/ayo_olopon/test_ayo_olopon.py::test_intermediate_four_is_captured_by_row_owner_and_sowing_continues`
currently fails independently of this adapter. Its synthetic position has only
8 board seeds with no captured seeds recorded, despite the normal 48-seed total.
After the intermediate capture, the current model's feeding constraint leaves
no legal move, so it collects the remaining seeds and ends the game. The test
expects those seeds to remain on the board. Decide whether the fixture/expectation
or the feeding rule should change before using the rules as research ground truth.
The scaffold preserves the current model's behavior.

## Hyperparameters

Edit `configs/starter.json`; omitted keys use `Algorithms/alpha_zero/config.py`.
These are **untuned starting values**, not an Ayo optimum.

| Setting | Starter | Meaning / decision |
| --- | ---: | --- |
| `nn_model`, `nn_api_version` | `ayo_mlp`, `linen` | Canonical 15-input policy-value model. NNX is not supported for this architecture. |
| `nn_width`, `nn_depth` | 256, 3 | Fixed three-layer shared trunk; heads are independently 128 and 64 units wide (185,863 parameters total). |
| `max_simulations` | 100 | MCTS simulations per move, including the root visit; must be at least 2. |
| `uct_c` | 1.5 | PUCT exploration constant for neural search, despite the upstream name. |
| `policy_alpha`, `policy_epsilon` | 1.0, 0.25 | Symmetric Dirichlet concentration and root-noise mixing weight during self-play. |
| `temperature`, `temperature_drop` | 1.0, 20 | Sample from powered visit counts for the first 20 individual moves, then choose the best child. Temperature must remain positive because upstream still computes policy targets with it. |
| `learning_rate` | 0.001 | Constant Adam learning rate. Optax defaults: betas 0.9/0.999, epsilon 1e-8; no clipping, optimizer weight decay, or schedule. |
| `weight_decay`, `decouple_weight_decay` | 0.0001, false | The canonical model uses Adam plus explicit `weight_decay × sum(weight²)` over dense kernels only; AdamW is rejected. |
| `train_batch_size` | 128 | Positions per gradient update. |
| `replay_buffer_size` | 16384 | Retained positions, not games; the ring overwrites the oldest position when full. |
| `replay_buffer_reuse` | 4 | Collect at least `buffer_size // reuse` new positions per learner round, then take `len(buffer) // batch_size` updates. This is not a boolean despite the upstream annotation. |
| `max_steps` | 100 | Learner rounds, not games or individual gradient updates. Zero runs until interrupted. |
| `actors`, `evaluators` | 2, 1 | Self-play and evaluation worker counts. Smoke uses 1 and 0. |
| `checkpoint_freq` | 10 | Retain numbered checkpoints every 10 rounds. |
| `eval_levels`, `evaluation_window` | 3, 50 | Rollout-MCTS budgets scale as `max_simulations * 10^(level/2)`; average recent evaluation results. `eval_levels >= 2` is required even with zero evaluators due to upstream diagnostics. |
| `max_moves`, `cutoff` | 1000, `collect` | Enforced training horizon and seed-count adjudication, described above. `cutoff` currently supports `collect` only. |

## Verification and upstream metric caveat

Verified on Windows CPU: 20 scaffold tests passed; complete self-play and gradient
updates; checkpoints 0, 1, and 2; concurrent actor/evaluator workers; checkpoint
reload; and balanced-seat evaluation against random play and rollout MCTS.
The existing rule-test failure above was reproduced without importing the adapter.
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
