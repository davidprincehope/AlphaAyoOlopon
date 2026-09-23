# EoH H* versus handcrafted on 400 openings

## Frozen protocol

This experiment compares frozen EoH `H*` (`g2-m2-05`) with the frozen handcrafted `H_CTM` evaluator (capture/tactical/mobility weights 4/2/0.5). No tuning or candidate selection occurs here.

The user selected **400 newly generated, distinct reachable positions**. `opening_set_400.json` was frozen before gameplay using seed `20260924`; each position is sampled after a uniformly chosen 4–32 random legal plies from the actual Ayo initial state. Every position is nonterminal, has legal actions, and has a unique SHA-256 state hash. The same 400 positions are used at each depth.

There are three policy conditions: depth 1 (greedy successor), depth 2 (one opponent reply), and depth 3 (root move, opponent reply, root's next move). Search depth counts individual actions; the acting policy maximizes its frozen evaluator and the opponent minimizes it; heuristic evaluation occurs at the cutoff. At every opening and depth, both seat assignments are played. Total: **400 × 3 × 2 = 2,400 games** (800 per depth). Games are deterministic once the openings are frozen. Action ties preserve legal-action order. The 300-action cap, if reached, is scored by captured-seed differential (equal differential is a draw).

Snapshot restoration follows the existing frozen-opening protocol: board, captured scores, and player to move are reconstructed, and the repetition-since-capture set begins with the restored current position. Historical repetition state from the random opening-generation path is not serialized. This is a known snapshot-method limitation applied equally across the policies.

## Commands

```powershell
.\.venv\Scripts\python.exe -m unittest experiments.handcrafted_vs_eoh_400.test_opening_comparison
.\.venv\Scripts\python.exe experiments\handcrafted_vs_eoh_400\run_opening_comparison.py --prepare-only
.\.venv\Scripts\python.exe experiments\handcrafted_vs_eoh_400\run_opening_comparison.py --run-only
```

The opening file is kept separate from prior suites. Use `--regenerate-openings` only to intentionally replace it. A small depth-specific smoke run can be isolated with `--run-only --pilot-depth 3 --pilot-matches 2 --output-dir experiments\handcrafted_vs_eoh_400\pilot_depth3`.

## Artifacts

- `opening_set_400.json`: frozen source positions, source labels, seeds, and hashes.
- `matches_400.jsonl`: per-match opening/seat, all played actions and root action values, outcomes, termination, decisions, and evaluator timings. Written incrementally.
- `results_400.json`: configuration, opening checksum, per-depth/policy WDL, score rate, seat split, runtime, and termination counts.
- `pilot/` and `pilot_depth3/`: isolated smoke-run outputs; not full results.
- `run_log.md`: execution checkpoints, tests, failures, and completion status.
