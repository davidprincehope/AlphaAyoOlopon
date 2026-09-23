# Run log — 400-position comparison

## Checkpoint 1: protocol and implementation

- **Scope:** frozen EoH `H*` against frozen handcrafted `H_CTM`; depth 1, 2, and 3; swap the policies' seats for every opening/depth.
- **User decision:** generate 400 fresh distinct positions (rather than reuse the old 100 plus 300 new).
- **Code:** `run_opening_comparison.py`; search reuses the tested depth wrapper from `experiments/handcrafted_vs_eoh/compare_initial.py`. No heuristic code or prior experiment files were changed.
- **Opening generation:** seed 20260924; target plies sampled uniformly from 4–32; random legal moves through the OpenSpiel Ayo game. Frozen in `opening_set_400.json` before gameplay; file hash `64242aed816452f59e3e84d4ae84278d98100bf06f7fc4fed586d965e8df3f38`.
- **Planned game count:** 2,400 (400 openings × three depths × two seat assignments).

## Checkpoint 2: tests and pilot

- `python -m unittest experiments.handcrafted_vs_eoh_400.test_opening_comparison` — passed, 3 tests: deterministic unique generation, snapshot restoration/playability, and legal depth 1–3 actions.
- `--prepare-only` — passed; froze 400 new positions, 400 distinct hashes.
- Four-game depth-1 smoke run — gameplay passed. The first summary attempt exposed a pilot-only empty-depth division-by-zero; changed empty condition score rates to `null`, then smoke run passed. Pilot outputs are isolated in `pilot/`.
- Two-game depth-3 smoke run — passed in about 6.35 seconds wall time on this host; both seat assignments finished naturally. Evaluator-choice timers summed to about 5.21 seconds across both games. This small pilot is only a rough runtime indicator.
- No held-out test set was opened. No Minimax agent participates; the depth-limited max/min backup is the explicitly defined policy wrapper.

## Checkpoint 3: full run

- **Status:** completed successfully, 2,400/2,400 games.
- **Command:** `python experiments/handcrafted_vs_eoh_400/run_opening_comparison.py --run-only`.
- **Results:** 800 games per depth, each opening used once with each policy in seat 0/seat 1; 400 distinct opening hashes shared across depths. Score rates: greedy depth 1 H_CTM 0.395625 / EoH 0.604375; depth 2 H_CTM 0.506250 / EoH 0.493750; depth 3 H_CTM 0.451250 / EoH 0.548750. See `results_400.json` and `results.md` for W/D/L and seat breakdown.
- **Termination:** 1,897 terminal wins, 503 terminal draws, zero repetition terminations, zero action-limit truncations, zero errors. Longest match: 167 actions, below the 300-action limit.
- **Runtime:** match log was created 08:22:07 and closed 08:34:18 local time on 2026-09-23 (~12m11s). Summed policy evaluator time was ~726.8 seconds across both policies/all conditions; machine-specific.
- **Integrity audit:** all 2,400 match IDs unique; each depth had 800 games; each opening/depth pair had both seat assignments; action counts matched log lengths; all outcomes completed; opening set had exactly 400 unique state hashes. Opening JSON SHA-256: `64242aed816452f59e3e84d4ae84278d98100bf06f7fc4fed586d965e8df3f38`.
- **Outputs:** `opening_set_400.json`, `matches_400.jsonl` (20,883,677 bytes), `results_400.json`.
- **Failures:** first direct invocation of `compare_initial.py` failed on import path; corrected before it was used. First 400-run pilot summary failed on empty-depth division by zero; corrected and the isolated smoke rerun passed. The full run then completed without errors.
