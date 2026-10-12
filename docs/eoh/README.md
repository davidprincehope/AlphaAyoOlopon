# Frozen EoH evaluator H*

Evolution of Heuristics evaluated **85 candidates**: ten initial candidates
and 25 in each of three generations. Each candidate played 70 development
openings in both seats against five frozen anchors and random play:
840 games per candidate, **71,400 development games** in total.

Fitness is `1 - weighted_score`, with weight 0.18 for each anchor and 0.10
for random. Generation used the pinned EoH 0.2 parent-selection and population
management logic, with fresh chat agents proposing candidate programs.
The recorded upstream revision is `472545785c936dcfc863d2bc0d6109cf23c7ce62`.
The working greedy evaluator has no runtime dependency on that package.

The top three distinct rounded development objectives were compared on
30 reused selection openings, both seats and all six opponents: **1,080
shortlist games**. This was reused selection data, not independent test data.
The frozen tie-break used AST complexity, fixture runtime and lexical source
hash to select **g2-m2-05** as H*.

| H* measure | Development | Reused selection |
| --- | ---: | ---: |
| Games | 840 | 360 |
| Wins / draws / losses | 397 / 228 / 215 | 147 / 110 / 103 |
| Unweighted score | 60.83% | 56.11% |
| Weighted score | 60.21% | 54.73% |

[H* implementation](../../Algorithms/H_star_ayo.py) and
[specification](../../experiments/EOH/H_star.json) define the frozen evaluator.
The shared [greedy interface](../../Algorithms/eoh_ayo.py) scores complete
successors from the original actor's perspective. Terminal scores are
±11/0; nonterminal candidate values are finite and bounded by ±10. Ties
choose the first legal action.

For new minimax evaluations, the implementation divides the original clipped
H* formula by 11, keeping heuristic estimates inside terminal utilities of
+/-1. This positive normalization preserves greedy move rankings. The saved
selection evidence and historical match results retain the original formula
and source hashes.

The public evidence includes the
[candidate catalog](../../experiments/EOH/candidate_catalog.jsonl),
[anchor panel](../../experiments/EOH/anchor_panel.json),
[selection protocol](../../experiments/EOH/selection_protocol_frozen.json),
[tie-break](../../experiments/EOH/selection_tiebreak.json),
[aggregate results](../../experiments/EOH/final_results_summary.json),
[worked move](../../experiments/EOH/H_star_worked_example.json) and
[experiment manifest](experiment_manifest.yaml).

A later [400-opening study](../../experiments/handcrafted_vs_eoh_400/README.md)
compared the already-frozen H* and handcrafted evaluator on fresh positions.
The [MLP benchmark](../results/checkpoint_progression/README.md) uses H* as
an evaluation opponent. Large match traces and generation working notes are
private/local artifacts; the selected evaluator and candidate provenance
remain public.
