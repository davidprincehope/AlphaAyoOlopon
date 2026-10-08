# Single-node A100 on Modal

For **offline checkpoint evaluation** of the completed 600-round run, use
[the varied-opening guide](../checkpoint_strength/README.md) and `modal_evaluate.py`. That finite job mounts
training artifacts read-only, writes a separate evaluation volume, and exits
after committing results. The progression experiment does not use the training launcher.

This runs the existing JAX AlphaZero **self-play and learning loop** in one Linux
container: one A100, 28 local actor processes, and the learner. MCTS, neural
predictions, trajectory queues, replay, and checkpoint broadcasts remain local.
There is no Modal RPC per leaf, move, or game.

The initial resource request is **A100-40GB, 30 physical CPU cores, 64 GiB host
RAM**. GPU size and RAM are starting assumptions, pending the original Lightning
allocation. Modal CPU units are physical cores, not vCPUs. If Lightning's 30 CPUs
were 30 vCPUs, 30 Modal cores is a larger CPU allocation; start with 15 physical
cores for a nominal 30-vCPU comparison and measure the actual loop time.
CPU microarchitecture and shared GPU contention also affect results.

References: [Modal resources](https://modal.com/docs/guide/resources),
[A100 variants](https://modal.com/docs/guide/gpu).

## Verified probe: 3 October 2026

Run `a100-compiled-20261003` completed on an actual NVIDIA A100-SXM4-40GB,
with a request of 30 physical CPU cores, 64 GiB RAM, and 28 local actors.
It used 64 simulations/move, a 16384-position replay buffer, batch size 128,
full game horizons, compiled actor inference, MPS, and no per-round matches.

| Round | New positions | Gradient updates this round | Collection | Updates + model checkpoint | Round |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 4163 | 32 | 39.13 s | 12.89 s | 52.11 s |
| 2 | 4104 | 64 | 17.44 s | 0.73 s | 18.28 s |
| 3 | 4112 | 96 | 17.92 s | 0.88 s | 18.87 s |
| 4 | 4103 | 128 | 17.99 s | 0.98 s | 19.05 s |
| 5 | 4111 | 128 | 19.72 s | 0.92 s | 20.72 s |

Worker wall time was **150.01 seconds** including startup, final snapshot, and
actor shutdown. The run collected 20593 positions from 510 games and completed
448 gradient updates. The complete step-5 learner snapshot and all five
parameter-only exports are on the Volume. These results establish loop speed
for this workload; the original Lightning compute details and hyperparameters
remain unconfirmed. They do not establish playing strength or performance of
additional evaluation matches.

The initial eager-inference probes were stopped without completing a learner
round. They are not quantitative speedup baselines. MPS and compiled inference
were not independently ablated in the successful probe.

## First three-round measurement

From the repository root in PowerShell:

```powershell
$env:PYTHONUTF8 = '1'
.venv/Scripts/python.exe -m pip install modal==1.5.5
.venv/Scripts/python.exe -m modal setup
.venv/Scripts/python.exe -m modal run experiments/alpha_zero/modal_train.py --run-name a100-baseline --max-steps 3
```

Omit `modal setup` if already authenticated. Every fresh run needs a new name.
Defaults: 28 actors, 64 simulations/move, replay capacity 16384, replay reuse 4,
batch size 128, canonical network, and **zero per-round evaluation matches**.
Each round collects at least 4096 positions; a full replay buffer yields 128
gradient updates. Early rounds have fewer updates as the buffer fills. The
three-round probe uses real full-horizon self-play, not shortened sanity games.
Use at least five rounds to compare steady-state learner work at full replay.

The launcher uses `configs/starter.json` with explicit runtime overrides. Supply
the **original Lightning JSON** to match its algorithm settings:

```powershell
.venv/Scripts/python.exe -m modal run experiments/alpha_zero/modal_train.py --config experiments/alpha_zero/configs/starter.json --run-name a100-with-eval --max-steps 5 --actors 28 --evaluation-games 100
```

Keep evaluation count, search budget, replay capacity/reuse, batch size, horizon,
and network identical across comparisons. Sequential per-round matches add work
to the loop. The observed Lightning speedup cannot be attributed solely to the
GPU without a matched workload and CPU allocation.

## Change the compute request

Set these **before** invoking Modal; they configure the function definition:

```powershell
$env:ALPHAZERO_MODAL_GPU = 'A100-80GB'
$env:ALPHAZERO_MODAL_CPU = '30'
$env:ALPHAZERO_MODAL_MEMORY_MIB = '65536'
$env:ALPHAZERO_MODAL_MPS = '1'
```

Training defaults to `A100-40GB` and accepts `A100-40GB` or `A100-80GB`.
Offline evaluation uses L4 by default; see the
[varied-opening guide](../checkpoint_strength/README.md).
CPU and host RAM requests must fit Modal's current scheduling limits.
Local requirements are installed in a Python 3.12 image, with the matching
`jax[cuda12]` extra. The worker fails if JAX cannot see a GPU. The edited local
OpenSpiel sources are shipped with the app; upstream cloning would lose the
custom model, game registration, replay, and resume changes.

The worker selects `spawn` before loading the trainer, disables JAX's default
75% GPU reservation per process, and sets library thread counts to one per
process. All actors still run GPU inference. CUDA MPS starts before JAX so that
the 28 actors and learner can share GPU execution with less context switching.
Set `ALPHAZERO_MODAL_MPS=0` for a comparison without MPS. Missing MPS tools cause
an explicit error when enabled. The controller is stopped at exit.
There is no central batched inference server.

Actor predictions reuse the repository's compiled `InferenceModel` instead of
the trainer's eager forward pass. Parameters are refreshed from the actor model
on every call, so checkpoint broadcasts still change its predictions. The value
and policy are copied to the host together; MCTS does not launch a GPU operation
for each policy scalar. This preserves the network, legal mask, value convention,
and learning objective. Tests compare the two inference paths before and after
a parameter update and exercise training/resume through spawned workers.

Actor checkpoint broadcasts use immutable `checkpoint-live-<id>` directories
between the configured retained checkpoints. The upstream reusable
`checkpoint--1` directory could disappear while Orbax replaced it and an actor
tried to restore it. Each update now gets a separate completed checkpoint before
its ID is queued; older versions remain available for delayed actors throughout
the run. This retains more model checkpoints on the Volume without changing
learning, broadcast frequency, or the full replay snapshot interval.

Reference: [JAX GPU memory allocation](https://docs.jax.dev/en/latest/gpu_memory_allocation.html).
MPS reference: [NVIDIA Multi-Process Service](https://docs.nvidia.com/deploy/mps/latest/index.html).

## Read the measurements and continue

Console output and `learner.jsonl` include measured phase timings:

- `collection_seconds`: time waiting for and ingesting complete trajectories.
- `learning_and_model_checkpoint_seconds`: learner updates and model checkpoint.
- `export_and_evaluation_seconds`: parameter export and optional round matches.
- `round_seconds`: sum of these phases; excludes full replay snapshots.
- `process_elapsed_seconds`: elapsed worker time at each recorded round. Differences
  between consecutive rounds include intervening replay snapshots and broadcasts.

The first elapsed measurement includes learner/actor setup and compilation after
GPU preflight. `modal-summary-<session>.json` also records worker wall time,
including GPU preflight, final snapshot, and actor shutdown. Image build,
scheduling, and container cold start are outside that worker measurement.
Use later round cadence to test the **under-600-second learning loop** target.
The upstream `states_per_s_actor` metric adds a synthetic minute to the first
round; these timings avoid that artifact. Queue backlog can make an individual
collection phase short, so compare several consecutive rounds.

Artifacts are persisted in the Modal Volume `alpha-ayo-alphazero-runs` under the
run name. Modal periodically persists Volume writes; the launcher explicitly
commits at exit. Completed replay snapshots support continuation, including
after interruption; in-flight trajectories are discarded on restart.

```powershell
# Submit a fresh 200-round run and return after Modal accepts it.
.venv/Scripts/python.exe -m modal run --detach experiments/alpha_zero/modal_train.py --background --config experiments/alpha_zero/configs/modal_a100.json --run-name a100-200 --max-steps 200

# Continue to total learner step 200, restoring optimizer and replay.
.venv/Scripts/python.exe -m modal run --detach experiments/alpha_zero/modal_train.py --background --run-name a100-200 --resume --max-steps 200

# Download artifacts to a new local directory.
.venv/Scripts/python.exe -m modal volume get alpha-ayo-alphazero-runs a100-baseline runs/alpha_zero/modal-a100-baseline
```

Use **both `--detach` and `--background`** for unattended training. `--detach`
keeps the app alive, while `--background` submits the function with `spawn()`
and lets the local launcher exit normally without waiting for the result.
The launcher prints the function-call ID and Modal dashboard URL. Monitor its
`train_remote` function logs in that dashboard; the local PC can then be closed.
For terminal streaming, use `modal app logs <app-id> --follow`.

Automatic retries are disabled. After timeout or interruption, explicitly resume
the last complete snapshot. Calls have a 24-hour timeout; run finite sessions
and continue from snapshots as needed. Full snapshots use the saved interval
(starter: every 10 rounds) and are always written at normal completion.
