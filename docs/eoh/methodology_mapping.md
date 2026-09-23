# Methodology mapping and deviations

| EoH requirement | Execution evidence | Status / deviation |
|---|---|---|
| Thought + executable heuristic candidates | 85 candidate records in `experiments/EOH/candidate_catalog.jsonl`; source, thought, lineage, hash, and score logged | GPT-6 Luna chat agents used instead of API `InterfaceLLM`; proposals batched five per operator call, not one REST request per candidate. |
| Official EoH operators | Pinned source audit and generation artifacts: `i1`, `E1`, `E2`, `M1`, `M2`, `M3` | Fixed schedule, five candidates per operator per generation, instead of upstream's asynchronous weighted sampler. |
| Official selection/update | Pinned `Evolution.parent_selection` and `population_management` used to produce generation plans and elites | The Ayo experiment adapter applies fixed seeds and writes explicit plans. Fitness rounded to five decimals and duplicate objectives collapsed as upstream. |
| Candidate execution/feasibility | Restricted AST candidate DSL plus immutable `StateView`, source limits, score bounds, fixture successor validation | No general OS memory/network sandbox. Accepted programs passed the DSL; this is narrower than arbitrary EoH Python programs. |
| Heuristic policy | `Algorithms/eoh_ayo.py`; one-ply greedy over actual legal successors; terminal ±11/0, actor perspective retained | EoH's generated artifact is a state evaluator; all candidates share the same selection wrapper. No Minimax involved. |
| Ayo fitness | 70 development openings × two seats × six opponents; weighted score from five fixed anchors and seeded random | Adapted fixed-instance EoH objective to gameplay wins/draws/losses. Per candidate 840 games; official objective passed as `1-S` (lower is better). |
| Anchor panel | `experiments/EOH/anchor_panel.json` | Five anchors were frozen before candidate fitness, selecting source-distinct candidates with distinct development-state behavior vectors from the allowed first 12 development positions. This is a designed panel adaptation. |
| Selection | `selection_protocol_frozen.json`, `selection_shortlist.json`, `selection_matches.jsonl`, `selection_tiebreak.json` | Top three unique rounded development objectives scored on reused 30-position selection subset. Not an independent holdout. Tie-break was preregistered: fewer AST nodes, lower fixture median runtime, then lexical source hash. |
| Independent test | No test data opened | Correctly remains unopened; no out-of-sample claim is made. |
| Cost/runtime | 23 counted proposal batch interactions; 85 scored candidates; gameplay timers plus wall-clock window recorded in `results.md` | Chat plan token/billing usage not exposed. Dollar cost unknown; no API pricing proxy asserted as actual spend. |

The upstream repository is pinned at `472545785c936dcfc863d2bc0d6109cf23c7ce62`; package version 0.2. Full checkpoint notes and any historical notes superseded by the actual run are in `implementation_log.md`.
