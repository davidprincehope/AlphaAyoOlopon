# Train the Ayo MLP

The trainer connects the Python Ayo rules to the repository's OpenSpiel
JAX/Flax learner: neural PUCT self-play, replay, learning and checkpoint
broadcasts. The completed study uses the **185,863-parameter MLP**, not a
residual network. [Training evidence](../../docs/results/mlp600_training/README.md)
and [checkpoint results](../../docs/results/checkpoint_progression/README.md)
describe the recorded run.

## Local setup and smoke run

Use Python 3.12+, initialize the submodule and install dependencies:

```sh
git submodule update --init --recursive
python -m pip install -r requirements-dev.txt
python -m experiments.alpha_zero.train --dry-run
python -m experiments.alpha_zero.train --config experiments/alpha_zero/configs/smoke.json --output runs/alpha_zero/smoke
```

The smoke configuration checks the complete pipeline for one round.
For the recorded model and self-play settings, use
[modal_a100.json](configs/modal_a100.json). Its original target is 200 rounds;
`--max-steps 600` sets the total training target to C600.

```sh
python -m experiments.alpha_zero.train --config experiments/alpha_zero/configs/modal_a100.json --max-steps 600 --output runs/alpha_zero/mlp600
```

This configuration uses 28 actors and is intended for an A100. On a local
machine, `--actors` can reduce the process count. The launcher selects the
checked-out OpenSpiel Python trainer; the installed package supplies `pyspiel`.

## A100 on Modal

Install `modal==1.5.5` and authenticate with `python -m modal setup`, then use
a fresh run name:

```sh
python -m modal run --detach experiments/alpha_zero/modal_train.py --background --config experiments/alpha_zero/configs/modal_a100.json --run-name mlp600 --max-steps 600
```

The defaults request A100-40GB, 30 physical CPU cores, 64 GiB RAM, 28 self-play
actors and CUDA MPS. Actors and the learner share one container; workers use
compiled inference with one library thread each. Per-round evaluation is
disabled in the recorded A100 configuration. `--detach` keeps the app alive;
`--background` submits the function and lets the launcher exit.

Resource overrides are `ALPHAZERO_MODAL_GPU`, `ALPHAZERO_MODAL_CPU`,
`ALPHAZERO_MODAL_MEMORY_MIB` and `ALPHAZERO_MODAL_MPS`. GPU options are
A100-40GB and A100-80GB. Artifacts persist in
`alpha-ayo-alphazero-runs/<run-name>`; inspect the printed dashboard and logs.

## Checkpoints and resume

Training snapshots preserve parameters, Adam state, learner counters and replay.
Immutable inference exports contain parameters for evaluation or gameplay.
Snapshots are published atomically and include hashes; incomplete snapshots
are excluded from resume. In-flight actor trajectories are discarded on restart.

```sh
python -m experiments.alpha_zero.train --resume runs/alpha_zero/mlp600 --max-steps 600
python -m modal run --detach experiments/alpha_zero/modal_train.py --background --run-name mlp600 --resume --max-steps 600
python -m modal volume get alpha-ayo-alphazero-runs mlp600 runs/alpha_zero/downloaded-mlp600
```

The target counts total rounds, rather than additional rounds. Selecting an
older snapshot with `--resume-step` requires a new `--output` directory.
Resume retains saved settings; `--actors` and `--evaluation-games` can change
the execution allocation or evaluation workload for the new session.

The model input contains 15 values: both pit rows, both captured scores and
remaining horizon, arranged relative to the player to move. Pits and scores
are normalized by 48. The shared trunk has three 256-unit ReLU layers;
the policy/value heads have 128/64 units. The value convention is
`player_to_move`, the architecture is `ayo_mlp_185863_v1`, and the game is
`ayo_olopon_alpha_zero(max_moves=1000,cutoff=collect)`.

The objective fits root-visit policy targets and final game returns, plus
weight-only L2 regularization. The recorded settings use 64 search simulations,
PUCT constant 1.5, Adam rate 0.001, L2 coefficient 0.0001, batch 128,
replay capacity 16,384 and reuse setting 4.

## Evaluate a frozen checkpoint

```sh
python -m experiments.alpha_zero.offline_evaluate --run runs/alpha_zero/mlp600 --checkpoint 600 --games 100 --simulations 64 --opponent RAND --output runs/evaluation/c600-random.jsonl
```

The evaluator loads parameters without creating an optimizer. Result paths
must be new and outside the training run. It writes results, game histories,
events and a manifest. Opponents include RAND, GREEDY_CAPTURE, GREEDY_HSTAR,
the handcrafted GREEDY evaluator, MINIMAX and MCTS.
The [benchmark guide](../checkpoint_strength/README.md) covers the frozen
DATASET ≤4 CAPTURE opening book, policy-only play and paired confidence intervals.

Core implementations are [game.py](../../Algorithms/alpha_zero/game.py),
[config.py](../../Algorithms/alpha_zero/config.py),
[checkpointing.py](../../Algorithms/alpha_zero/checkpointing.py),
[inference.py](../../Algorithms/alpha_zero/inference.py) and
[runtime.py](../../Algorithms/alpha_zero/runtime.py).
