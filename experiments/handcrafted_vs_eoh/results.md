# Results: handcrafted `H_CTM` versus EoH `H*`

## Experimental setup

The frozen handcrafted evaluator was `H_CTM` with weights capture=4, immediate tactical potential=2, and mobility=0.5; its terminal values are ±10. EoH was frozen candidate `g2-m2-05` (`H_star_ayo.py`). No parameters were changed.

We ran both policies on the same 70 frozen development openings, twice per opening with the players' seats swapped (140 games per condition). In the two-ply condition, each candidate move is followed by a search over every legal opponent reply; the reply with the minimum evaluator value for the root player is backed up, then the root chooses the maximum. This is a one-opponent-reply heuristic lookahead. It is not the separate Minimax agent. The action cap was 300; capped games would be scored by captured-seed difference, but no game reached the cap. The 30 selection/verification positions and independent test set were not used.

## Results

| Policy condition | Policy | Wins | Draws | Losses | Score rate | Policy decision time* |
|---|---|---:|---:|---:|---:|---:|
| Greedy successor (1 ply) | Handcrafted | 30 | 46 | 64 | 0.3786 | 3.31 s |
| Greedy successor (1 ply) | EoH H* | 64 | 46 | 30 | 0.6214 | 1.07 s |
| One opponent reply (2 ply) | Handcrafted | 55 | 32 | 53 | 0.5071 | 35.42 s |
| One opponent reply (2 ply) | EoH H* | 53 | 32 | 55 | 0.4929 | 12.37 s |

Score rate is `(wins + 0.5 × draws) / games`. Each policy's 140-game total includes 70 games as seat 0 and 70 as seat 1. At one ply, EoH scored 87 points to handcrafted's 53. With one opponent reply, the handcrafted policy scored 71 points to EoH's 69. The lookahead condition changed both score and game trajectories; these are separate policy conditions, not a direct generalization beyond this opening panel.

Termination was natural in all conditions: at one ply there were 94 terminal wins and 46 terminal draws; at two plies, 108 terminal wins and 32 terminal draws. There were no repetitions, errors, or action-limit truncations. Seat-by-seat scores, full actions and per-decision candidate/reply heuristic traces are in `results.json` and `matches.jsonl`.

`*` Policy time sums time spent choosing actions, not full game execution time; timings are host-dependent. The experiment is deterministic and uses no random choices, so there is no game RNG seed.

## Reproduction

From repository root:

```powershell
.\.venv\Scripts\python.exe -m unittest experiments.handcrafted_vs_eoh.test_compare
.\.venv\Scripts\python.exe experiments\handcrafted_vs_eoh\compare.py
```

The opening suite checksum is `971dba4079f260d733418fed6f008130b56778fd947584f8292669b2c92b2735`; output files and configuration are self-described in `results.json`.
