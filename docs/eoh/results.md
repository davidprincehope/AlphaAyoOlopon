# Completed EoH run results

## Frozen setup

Upstream EoH 0.2 is pinned at `472545785c936dcfc863d2bc0d6109cf23c7ce62`. The source implements `i1`, `E1`, `E2`, `M1`, `M2`, and `M3`; official parent selection and population management functions were called. GPT-6 Luna was used through fresh chat agents, with prompts carrying only the required candidate/operator context. No API endpoint/key was used.

The run scored 10 initial candidates and 25 candidates in each of three generations (five per operator per generation). Fitness is lower-is-better `1-S`, where `S` is fixed weighted WDL score versus five frozen anchors (0.18 each) and seeded random legal play (0.10). Candidate evaluation used 70 development openings, both seats, and all six opponents: 840 games per candidate, 71,400 total. Each game used seed `20260922 + opening_id*1000 + candidate_seat*100 + opponent_index`, action cap 300. No game was truncated. The frozen selection protocol ranked the top three unique five-decimal development fitness values, then compared them on all 30 reused selection openings, both seats, and six opponents: 1,080 games. Independent test data were not opened.

## Outcome

`H*` is `g2-m2-05`, source SHA-256 `1be272704c60811ba53ffe28e47c90ea21d50e04e8e3371ab0e01fd291bd2880`, generation 2 M2 mutation of `g1-m1-04`.

| Measure | Development | Reused selection subset |
|---|---:|---:|
| Games | 840 | 360 for H* (1,080 across shortlist) |
| H* W / D / L | 397 / 228 / 215 | 147 / 110 / 103 |
| Unweighted score rate | 0.608333 | 0.561111 |
| Weighted fixed-panel score `S` | 0.602143 | 0.547333 |
| Fitness `1-S` | 0.397857 | Not used for another evolutionary update |
| Truncations | 0 | 0 |
| Median plies | 21 | 28 |

The top-three candidates had identical selection weighted score `S=0.5473333333`. The frozen tie-break chose the lower-AST-complexity g2 candidate; both g2 candidates had 269 AST nodes and the same fixture median call time, so lexical source SHA-256 selected `g2-m2-05` over `g2-m2-01`. `g3-m1-01` had more AST nodes. The full decision inputs are in `selection_tiebreak.json`.

Development best fitness improved from 0.404929 (generation 0) to 0.401714, 0.397857, and 0.397500 (generation 3). Generation 3 did not enter the selection shortlist because the upstream-style population manager collapses duplicate rounded objectives and takes the top three unique objective values only after all generations, as frozen in the selection protocol.

H* development score by opponent: random 0.685714; anchors 1–5: 0.614286, 0.507143, 0.589286, 0.639286, 0.614286. Selection score by opponent: random 0.733333; anchors 1–5: 0.541667, 0.500000, 0.500000, 0.550000, 0.541667.

## Worked move selection

On development opening 0, player 1 has legal actions `[2, 3, 5]`. H* evaluates each actual successor from player 1's perspective, despite `current_player` becoming player 0 in each successor:

| Action | Score | Captured seeds after move | Chosen |
|---:|---:|---:|:---:|
| 2 | 4.78 | `[16, 24]` | |
| 3 | 6.06 | `[16, 24]` | Yes |
| 5 | 5.42 | `[16, 24]` | |

The deterministic greedy policy selects action 3. Exact successor board arrays are in `experiments/EOH/H_star_worked_example.json`.

## Calls, cost, runtime, and deviations

The recorded plan totals 23 proposal-batch chat interactions and approximately 103 raw source proposals; 85 candidates were validated and scored. Attempts included invalid DSL outputs and corrected retries. The rejection archive preserves some but not every discarded/corrected raw output; candidate output provenance for accepted programs is complete, while raw failed-output provenance is incomplete. This is disclosed rather than reconstructing missing prompts. Since the Codex chat plan does not expose per-agent billing or token counts, actual USD cost is unknown. The GPT-5.6 Luna API price proxy in earlier planning notes is not a valid price for GPT-6 Luna chat and is not reported as actual spend.

The captured wall window from first accepted candidate-catalog timestamp through final H* artifact creation was about 47m47s; the first prompt may predate that file timestamp. Summed match-level timers were about 19m34s, with remaining wall time mostly generation/agent wait and orchestration. Match timings are host-specific.

Published-method deviations: (1) chat-agent batching replaced API calls; (2) fixed per-operator schedule replaced asynchronous weighted operator dispatch; (3) evaluation is Ayo gameplay WDL against a designed frozen anchor/random panel; (4) candidate code is restricted to the local arithmetic DSL; (5) selection uses a reused selection split, explicitly not a test result. EoH thought/code evolution and pinned official parent/elite mechanisms were retained. Minimax was not used.
