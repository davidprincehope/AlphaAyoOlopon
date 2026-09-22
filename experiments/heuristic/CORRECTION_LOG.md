# Correction log for the completed greedy heuristic study

## Inspection baseline

The original implementation is in `Algorithms/ayo_heuristic.py` and the
experiment runner is `experiments/heuristic/run_experiment.py`. The frozen
opening snapshots are in `openings/opening_positions.json`; the original raw
results are in `raw_results/`. The configuration directory contained no
experiment configuration beyond the runner and manifest.

The original study used terminal values `+5/0/-5`, six weight vectors including
`[8,4,1]`, classified every unfinished 300-action game as a draw by assigning
returns from the final score, and regenerated openings whenever the runner was
invoked. The original raw files are preserved unchanged.

## Corrections

1. **Terminal dominance:** change the selected heuristic's terminal value to
   `+10/0/-10` and validate it against the active positional-weight maximum.
   The selected positional maximum is `4+2+0.5=6.5`, so `W=10` strictly
   dominates it. Affected work: heuristic implementation and terminal tests;
   all greedy results using heuristic evaluation must be rerun.
2. **Equivalent weights:** identify positive scalar multiples before running
   weight experiments. `[8,4,1]` is equivalent to `[4,2,0.5]` and is retained
   only as an equivalence check. Replace it in the corrected six-vector set
   with `[8,3,1]`.
3. **Outcome classification:** record terminal wins, natural terminal draws,
   repetition termination, action-limit truncations, and errors separately.
   Truncations retain the original experimental score convention of `0.5`,
   but are no longer called natural draws.
4. **Frozen openings:** make frozen-opening loading the default. Regeneration
   requires an explicit flag and seed and writes a separate file by default.
   The loader validates state hashes, split counts, and cross-split uniqueness.

## Rerun scope

Corrected candidate and weight comparisons are written under
`raw_results_corrected/`; the original `raw_results/` directory is not
overwritten. The corrected development comparison uses the frozen 70-position
development split. The selected corrected vector is then evaluated once on the
frozen 30-position verification split. Feature definitions, normalization,
opening split, player assignments, deterministic tie-breaking, and the
300-action limit remain unchanged.
