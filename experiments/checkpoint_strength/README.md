# Reproduce the checkpoint benchmarks

The primary study evaluates the original MLP on **DATASET ≤4 CAPTURE**:
200 globally distinct unfinished positions after 2/4/6/8/10 random actions,
with counts **32/42/42/42/42** and captured-seed gap at most four.
The [frozen opening book](dataset/dataset_le4capture_v3_200.json) is included;
its content SHA-256 is
`681ac741957726f17d3e29241330f76d4d0da80f7dbfd2e05605736ac68d0a9b`.

The public name uses **≤4** because the filter includes differences of exactly
four. Frozen opening IDs and historical report metadata retain their original
internal names to preserve verification hashes. The loader accepts those
historical paths and resolves them to the renamed book.

Replay each prefix to restore repetition memory and elapsed moves. Both seats
give **400 games per case**, sharing seed `20261003 + opening_index`.
The training game uses 1,000 actions and collection at the horizon. Neural
PUCT uses constant 1.5, no root noise or sampling, and maximum visits with
lowest-action-ID ties. Policy-only play chooses the highest-probability legal
move. Plain MCTS uses UCT, uniform priors and one random rollout per leaf.
Greedy EOH uses the unchanged H* successor evaluator.

The [verified suite](../../docs/results/checkpoint_progression/README.md)
historically contains 56 cases and 22,400 games: four curves at C1 and C50–C600 in increments
of 50, two checkpoint head-to-head matchups, and MCTS 64/128 versus Greedy EOH.
The [capture-only greedy comparison](../../docs/results/c600_comparisons/c600_puct64_vs_capture_only_greedy/README.md)
adds 400 games with C600 PUCT 64. Score credits draws as half a win; confidence
intervals resample whole opening pairs, rather than individual games.

New plans contain **44 cases and 17,600 games**: three checkpoint curves
(PUCT 64/MCTS 64, policy-only/MCTS 64, and PUCT 64/Greedy EOH), the standalone
C600 PUCT 64/MCTS 128 comparison, two checkpoint matchups and two MCTS/EOH
baselines. PUCT 64/MCTS 128 is no longer scheduled across the other checkpoints.
Historical schema-1 exports remain unchanged for provenance; new plans use
schema 2. [All C600 comparisons](../../docs/results/README.md) have tabular
results, including the separately completed capture-only greedy experiment.

## Run one comparison locally

Use a retained compatible training directory with `manifest.json` and its
immutable `inference-checkpoints/step-XXXXXX.npz` exports. Results must use a
fresh directory outside the source run.

```sh
python -m experiments.checkpoint_strength.neural_vs_mcts --run runs/alpha_zero/modal-a100-600-20261003 --config experiments/checkpoint_strength/configs/c600_vs_mcts_64.json --output runs/checkpoint_strength/c600-mcts-rerun
python -m experiments.checkpoint_strength.neural_vs_mcts --run runs/alpha_zero/modal-a100-600-20261003 --config experiments/checkpoint_strength/configs/c600_64_vs_mcts_128.json --output runs/checkpoint_strength/c600-mcts128-rerun
python -m experiments.checkpoint_strength.greedy_progression --run runs/alpha_zero/modal-a100-600-20261003 --config experiments/checkpoint_strength/configs/c600_vs_greedy_eoh_64.json --output runs/checkpoint_strength/c600-eoh-rerun
python -m experiments.checkpoint_strength.head_to_head --run runs/alpha_zero/modal-a100-600-20261003 --config experiments/checkpoint_strength/configs/head_to_head_c350_c600_64.json --output runs/checkpoint_strength/c350-c600-rerun
python -m experiments.alpha_zero.offline_evaluate --run runs/alpha_zero/modal-a100-600-20261003 --config experiments/checkpoint_strength/configs/c600_vs_capture_greedy_64.json --output runs/checkpoint_strength/c600-capture-rerun/results.jsonl
```

Configs also cover neural 64/MCTS 128, policy-only/MCTS 64 and C50/C350.
Parallel evaluators assign seeds by opening index, independent of scheduling,
and save ordered game histories, result summaries and manifests.

## C600 versus minimax depth 6

[Completed results](../../docs/results/c600_comparisons/c600_puct64_vs_minimax_depth6/README.md).

This comparison uses the same 200 paired openings, C600 at 64 PUCT simulations per move and deterministic alpha-beta minimax at depth 6. A search depth counts complete player actions, including relay laps. The H* leaf formula is clipped to [-10, 10] and divided by 11; terminal outcomes remain -1/0/+1, so proven wins and losses dominate heuristic estimates.

```sh
python -B -m experiments.checkpoint_strength.neural_vs_minimax --run runs/alpha_zero/modal-a100-600-20261003 --config experiments/checkpoint_strength/configs/c600_puct64_vs_minimax_depth6.json --output runs/checkpoint_strength/c600_puct64_vs_minimax_depth6 --workers 8
```

The runner saves complete opening pairs atomically and replays every game before accepting it. Running the same command resumes saved pairs only if configuration, model, source code and opening hashes still match. Use a fresh output path for a new evaluation. Historical minimax benchmarks used the original H* scale and retain their original results.

## Run the complete suite on Modal

Generate a fresh plan from the retained original MLP exports and the current
evaluation source. The published experiment plan records the original run's
hashes; it is evidence, rather than a plan to reuse after source changes.

```sh
python -m experiments.checkpoint_strength.le4capture_suite --source runs/alpha_zero/modal-a100-600-20261003 --evaluation-name le4capture-rerun --output runs/checkpoint_strength/le4capture-rerun/experiment.plan.json
```

The source volume `alpha-ayo-alphazero-runs` must contain the original run as
`a100-fresh-64-200-20261003`, including all 13 selected exports.
Use your authenticated Modal profile and set resources before launch:

```powershell
$env:ALPHAZERO_EVAL_GPU = 'A100-40GB'
$env:ALPHAZERO_EVAL_CPU = '10'
$env:ALPHAZERO_EVAL_MEMORY_MIB = '32768'
python -m modal run --detach experiments/checkpoint_strength/modal_le4capture_suite.py --background --plan-file runs/checkpoint_strength/le4capture-rerun/experiment.plan.json
python -m experiments.checkpoint_strength.collect_le4capture_suite --evaluation-name le4capture-rerun --watch
```

One 400-game case runs at a time and is verified before the next. GPU cases
use eight workers with CUDA MPS; plain-MCTS/EOH cases use CPU containers.
The collector replays full legal histories and checks source hashes, frozen
seeds, seats, prefixes, terminal captures, outcomes and interval accounting.
Complete locally verified outputs can be exported with
`python -B scripts/export_le4capture_results.py --run <local-suite> --output <new-report-directory>`.

After an interrupted controller has stopped, `--resume` with the same local
plan reuses verified cases and retains partial cases separately. Local source
snapshots preserve the exact evaluated code. Model exports remain with the
MLP training run; new evaluation logs are written under ignored `runs/`.
Retain completed match histories in external storage for later replay audits.
Compact public results and figures remain versioned.

## Opening provenance

The frozen book was selected from a reproducible random-prefix pool without
using model strength or match outcomes. To regenerate the pool and selection
in a new local directory:

```sh
python -m experiments.checkpoint_strength.generate_openings investigate --directory runs/checkpoint_strength/opening-pool --count 5000 --seed 20261004
python -m experiments.checkpoint_strength.generate_openings freeze-mixed --directory runs/checkpoint_strength/opening-pool --output runs/checkpoint_strength/dataset_le4capture_v3_200.json --allocation 2:32 4:42 6:42 8:42 10:42 --benchmark-id ayo_small_gap_v3_200 --max-capture-difference 4
```

The `--benchmark-id` is the immutable historical identifier; retaining it
reproduces the published content hash despite the new public filename.

There are 65 tied-capture positions and 135 with a four-seed gap. The filter
limits starting capture margins but does not establish balanced position
strength. Intervals describe sampled continuation positions and exclude
training-seed and search-seed uncertainty; equal simulations do not imply
equal compute.
