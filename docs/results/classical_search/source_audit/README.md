# Classical search source audit

The [saved recount](source_audit.json) covers the retained Greedy EoH H* and
UCT MCTS benchmarks against random play: 10,000 standard-start games per
condition, with 5,000 in each seat. It records W/D/L counts, termination
categories and the SHA-256 of the locally recounted game logs.

| Condition | Report | Action-limit outcomes |
| --- | --- | ---: |
| Greedy EoH H* versus random (Python) | [Report](../classical_greedy_eoh_hstar_vs_random/README.md) | 0 |
| UCT MCTS, 1,000 simulations, versus random (C++) | [Report](../classical_uct_mcts_1000_simulations_cpp_vs_random/README.md) | 0 |

Superseded minimax results have been removed. The
[verified minimax study](../corrected_minimax_depths_1_to_5_vs_random/README.md)
provides the current depth comparison, with aligned utilities, one Python
implementation at every depth, paired seeds and a search-visible 1,000-action
horizon. All 50,000 games in that study finished without action-limit outcomes.
Its result file and paired-score CSV carry separate provenance and verification.

The retained greedy and MCTS measurements have not changed. Re-auditing them
requires the original raw logs. The Greedy H* stored hash matches LF-normalized
log bytes; the audit also records the exact locally recounted byte hash.
The historical native MCTS summary contains no stored log hash, so its audit
hash identifies the recounted local file without establishing historical hash
verification. Its executable-build provenance was not reconstructed. These
benchmarks do not provide an equal-runtime comparison with the minimax study.
