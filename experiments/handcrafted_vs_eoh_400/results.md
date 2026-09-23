# Results — 400 fresh openings, seat-balanced, depths 1–3

This run used 400 newly generated unique reachable nonterminal positions (seed `20260924`, uniformly sampled target length 4–32 legal plies). The identical frozen positions were used at all depths. Each opening/depth condition played both seat assignments: 800 games per depth and 2,400 total. Only the two frozen evaluators were compared: handcrafted `H_CTM` (capture/tactical/mobility 4/2/0.5) and EoH `H*` (`g2-m2-05`).

| Depth | H_CTM W-D-L | H_CTM score | EoH H* W-D-L | EoH score |
|---:|---:|---:|---:|---:|
| 1 ply | 191-251-358 | 0.395625 | 358-251-191 | 0.604375 |
| 2 plies | 330-150-320 | 0.506250 | 320-150-330 | 0.493750 |
| 3 plies | 310-102-388 | 0.451250 | 388-102-310 | 0.548750 |

Score is `(wins + 0.5 × draws) / games`. Depth counts individual game actions: depth 1 is greedy evaluation after the root move; depth 2 includes the opponent reply; depth 3 adds the root policy's next move. Each policy maximizes its frozen evaluator on its turns and assumes the opponent minimizes it on theirs. Ties retain legal-action order. No Minimax agent or heuristic update was used.

All 2,400 games ended naturally: 1,897 terminal wins and 503 terminal draws; there were no repetitions, errors, or action-limit truncations. The longest game was 167 actions. Runtime was about 12m11s; summed policy decision-evaluation time was about 12m07s (host dependent).

This 400-opening comparison is a new sample, generated after both heuristics were frozen. It is not reused EoH selection data. The opening profile contained 151 positions sampled at 4–12 plies and 249 at 13–32 plies. Full seat-specific counts, timings, opening hashes, and per-action root value traces are in `results_400.json` and `matches_400.jsonl`.

This is one finite sample; no confidence interval or population-level claim is made here. Snapshot restoration preserves board, captured scores, and player to move, while resetting repetition history to the restored current state, consistent with the earlier frozen-opening protocol.
