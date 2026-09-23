# Standard initial-position results

This separate follow-up ran the same frozen handcrafted `H_CTM` and EoH `H*` from the standard initial Ayo position: four seeds per pit, captured totals `[0,0]`, player 0 to move. At each depth, the two policies swapped seats, for two games per depth and six games total.

Depth counts individual actions. Each policy maximizes its own frozen heuristic on its turns and assumes the opponent minimizes that same evaluator on opponent turns; values are evaluated at the depth cutoff. Therefore depth 1 is greedy, depth 2 includes one opponent reply, and depth 3 includes the opponent reply plus the root policy's next move. All tie breaks use first legal-action order.

| Search depth | Handcrafted W-D-L | Handcrafted score | EoH W-D-L | EoH score | Terminations |
|---:|---:|---:|---:|---:|---|
| 1 ply | 1-0-1 | 0.50 | 1-0-1 | 0.50 | 2 wins |
| 2 plies | 2-0-0 | 1.00 | 0-0-2 | 0.00 | 2 wins |
| 3 plies | 0-2-0 | 0.50 | 0-2-0 | 0.50 | 2 draws |

There were no action-limit truncations or repetition terminations. The full root action-value traces and seat-specific outcomes are in `initial_matches.jsonl`; aggregate results and the standard-start state hash are in `initial_results.json`.

This is an initial-position demonstration with only two games per depth. Since the policies are deterministic and the starting position is fixed, repeating either seat assignment would replay the same game. Treat these as illustrative paired outcomes, not a statistical strength estimate. The 70-opening seat-balanced run remains the broader comparison.
