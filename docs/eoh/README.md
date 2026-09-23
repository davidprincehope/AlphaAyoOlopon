# Ayo Olopon EoH experiment

## Status

The full development evolution and the preregistered shortlist selection have run. `H*` is frozen as `g2-m2-05` in [`experiments/EOH/H_star.json`](../../experiments/EOH/H_star.json) and its importable implementation is [`Algorithms/H_star_ayo.py`](../../Algorithms/H_star_ayo.py). The 30-position `EoH_selection` subset was opened only after the shortlist and tie-break were frozen. It is reused selection data, not an independent test set; the independent held-out test remains unopened.

## Reproduction and audit trail

Pinned upstream source is `experiments/EOH/vendor/EoH`, commit `472545785c936dcfc863d2bc0d6109cf23c7ce62` (EoH 0.2). Available operators verified in that source: `i1`, `E1`, `E2`, `M1`, `M2`, `M3`. Official parent-selection and population-management functions were used. Exact artifacts, commands, outcomes, deviations, and checkpoint logs are in [`implementation_log.md`](implementation_log.md), [`methodology_mapping.md`](methodology_mapping.md), and [`results.md`](results.md).

The frozen evaluation configuration used 10 initial sources and three generations of 25 proposals (five per operator), 5 elite population, 5 fixed generated anchors plus seeded random legal play, and 70 development openings × both seats × six opponents = 840 matches per candidate. All 85 candidates were validated and scored, for 71,400 development matches. Three unique top candidates were then scored on 30 reused selection openings × both seats × six opponents = 1,080 matches. Total: 72,480 matches. The detailed frozen parameters are in [`experiment_manifest.yaml`](experiment_manifest.yaml).

No API key or API request was used. GPT-6 Luna generation used fresh chat agents (stateless with respect to this conversation; shared workspace/system instructions still apply). Consequently the number of generation interactions and elapsed wall time are recorded, but a dollar cost cannot be inferred from API pricing or Codex usage. Any GPT-5.6 Luna public API price is not a valid estimate of the actual chat run.

## Results and logs

- Candidate source, lineage, SHA-256, validation, weighted score, fitness, and per-opponent aggregate: `experiments/EOH/candidate_catalog.jsonl` (85 records).
- Every action/outcome/seed/opening/termination/timing for development: `experiments/EOH/matches.jsonl` (71,400 records).
- Selection matches: `experiments/EOH/selection_matches.jsonl` (1,080 records).
- Generation prompts/results and parent schedules: `generation_{1,2,3}_{E1,E2,M1,M2,M3}.jsonl`, `generation_*_parent_plan.json`, and validated files.
- Frozen anchor opponents: `experiments/EOH/anchor_panel.json`.
- Machine-readable aggregate: `experiments/EOH/final_results_summary.json`.
- One-ply move trace: `experiments/EOH/H_star_worked_example.json`.

Candidate JSONL records contain `candidate_id`, `generation`, `operator`, `parents`, `model`, `context_mode`, `prompt_id`, `thought`, `source`, `source_sha256`, `fitness`, `selected`, `score_rate`, and `fitness_details`. Match JSONL records contain `match_id`, `candidate_id`, `opponent_id`, `opening_id`, `opening_hash`, `candidate_seat`, `seed`, `actions`, `outcome`, `score`, `termination_reason`, `truncated`, `game_length`, `elapsed_seconds`, and `split`.

## Policy and candidate validation

Every candidate uses the same greedy interface: for each legal action, copy by `state.child(action)`, evaluate the successor from the original acting player's perspective, assign exact terminal values ±11/0, and take the maximum; ties keep the first legal action. Candidate nonterminal values must be finite and in [-10,10]. The restricted arithmetic AST validator is source-level hardening, not a general-purpose OS sandbox. Candidate inputs are immutable snapshots, and validation checks legal successors across both player perspectives.

The selected `H*` evaluator is in `Algorithms/H_star_ayo.py`. On the stored worked example, actor 1's legal actions `[2, 3, 5]` score `4.78`, `6.06`, and `5.42`; greedy selection chooses action 3. The detailed resulting boards and captured counts are in `H_star_worked_example.json`.

## Tests

Run from repository root:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.ayo_olopon.test_eoh_interface tests.ayo_olopon.test_eoh_upstream_contract tests.ayo_olopon.test_ayo_olopon tests.ayo_olopon.test_heuristic_corrections
.\.venv\Scripts\python.exe experiments\EOH\pilot.py
```

The test suite and fixture pilot passed after the final candidate adapter changes; final verification details are recorded in `implementation_log.md`.
