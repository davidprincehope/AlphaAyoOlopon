# Handcrafted versus EoH Ayo heuristics

This experiment is isolated from the EoH evolution run and the existing handcrafted-selection experiment. It compares the already-frozen handcrafted `H_CTM` (`C/T/M` weights 4/2/0.5) against EoH `H*` (`g2-m2-05`). It does not tune either heuristic.

## Conditions

Two policy depths are compared:

1. **Greedy / one ply:** score each legal successor with that policy's frozen heuristic and choose the maximum.
2. **One-step lookahead / two plies:** for each own action, evaluate every legal opponent reply, back up the minimum score for the root player, then choose the own action with the maximum backed-up score. This is a one-opponent-response lookahead, distinct from the prior one-ply greedy experiment. It is implemented locally and does not invoke `Algorithms/ayo_minimax.py` or affect EoH selection.

For each of 70 frozen development openings, each condition plays twice with seats reversed: 140 games per depth, 280 total. Action ties preserve OpenSpiel legal-action order. The action limit is 300; if reached, the captured-seed difference assigns the result (tie = draw), matching the EoH fitness convention. Only the `development` split is loaded for gameplay; the 30 reused selection/verification openings are excluded, and the independent test set is not accessed.

## Run and artifacts

From repository root:

```powershell
.\.venv\Scripts\python.exe -m unittest experiments.handcrafted_vs_eoh.test_compare
.\.venv\Scripts\python.exe experiments\handcrafted_vs_eoh\compare.py
```

`matches.jsonl` stores each opening/seat matchup, policies, action trace, root/reply heuristic scores, outcome, termination, plies, decisions, and evaluation time. `results.json` stores configuration, seat-balanced W/D/L and score rate by policy and depth, and termination counts. Use `--pilot-matches N` for a quick smoke run; it writes to the same output names, so save/copy prior full results before rerunning a pilot.

## Initial-position follow-up

`compare_initial.py` runs a separate follow-up directly from the standard initial board `[4] * 12`, captured scores `[0,0]`, player 0 to move. It runs both seat assignments at depths 1, 2, and 3 (six games total). Depth counts individual actions; the root player's turns maximize its evaluator and the opponent turns minimize it. Thus depth 3 covers the root move, an opponent response, and the root's next move before static evaluation. It writes `initial_matches.jsonl` and `initial_results.json`; it does not overwrite the development-opening results.

Run it with:

```powershell
.\.venv\Scripts\python.exe -m unittest experiments.handcrafted_vs_eoh.test_compare_initial
.\.venv\Scripts\python.exe experiments\handcrafted_vs_eoh\compare_initial.py
```

This protocol keeps the heuristic definitions fixed but applies a finite opening panel, so results describe this panel and are not a claim of population-wide strength. The one-step-lookahead condition changes how each heuristic is used; report it separately from the original greedy EoH fitness.
