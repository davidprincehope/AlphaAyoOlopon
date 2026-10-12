# Classical search source audit

The [saved recount](source_audit.json) preserves seven classical-agent
benchmarks against random play.
Each condition contains 10,000 standard-start games, with 5,000 in each seat.
It records W/D/L counts, termination categories, outcomes by termination, and
the SHA-256 of each retained local game log.

| Condition | Separate experiment report | Cap-adjudicated wins |
| --- | --- | ---: |
| Greedy EoH H* versus random (Python) | [Report](../classical_greedy_eoh_hstar_vs_random/README.md) | 0 |
| Minimax depth 1 versus random (Python) | [Report](../classical_minimax_depth_1_python_vs_random/README.md) | 437 |
| Minimax depth 2 versus random (Python) | [Report](../classical_minimax_depth_2_python_vs_random/README.md) | 3,983 |
| Minimax depth 3 versus random (Python) | [Report](../classical_minimax_depth_3_python_vs_random/README.md) | 5,104 |
| Minimax depth 4 versus random (C++) | [Report](../classical_minimax_depth_4_cpp_vs_random/README.md) | 6,502 |
| Minimax depth 5 versus random (C++) | [Report](../classical_minimax_depth_5_cpp_vs_random/README.md) | 6,962 |
| UCT MCTS, 1,000 simulations, versus random (C++) | [Report](../classical_uct_mcts_1000_simulations_cpp_vs_random/README.md) | 0 |

The cap is 300 player actions, adjudicated by captured-seed lead. All capped
minimax games were credited as wins. Reported score rates include these games
and count draws as half a win. The combined minimax experiment summary covers
only depth 3; the individual summaries above cover the other depths.

The separate minimax adapters ([Python](../../../../Algorithms/ayo_minimax.py),
[C++](../../../../Algorithms/ayo_minimax_cpp.cc)) pass H* leaf estimates as large
as ±10 to OpenSpiel search, whose terminal branch returns ±1/0 utilities
([Python search](../../../../open_spiel/open_spiel/python/algorithms/minimax.py),
[C++ search](../../../../open_spiel/open_spiel/algorithms/minimax.cc)). This scale
mismatch requires a corrected rerun before drawing a clean conclusion about
depth. It does not establish the cause of the historical cap rates. Historical
native executable build provenance was not reconstructed, and the different
implementations and budgets do not establish an equal-runtime comparison.

The recount was retained from the removed article's evidence file; no
benchmark results were changed. Re-auditing the 70,000 games requires the
original raw logs. The Greedy H* summary's
stored hash matches LF-normalized log bytes; the audit also preserves the
exact local byte hash. Python minimax stored hashes match the raw bytes.
Native summaries contain no stored log hashes, so those audit hashes identify
the locally recounted files without establishing historical hash verification.
