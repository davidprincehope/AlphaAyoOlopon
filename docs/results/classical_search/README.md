# Classical search source audit

The [saved recount](source_audit.json) preserves the seven classical-agent
benchmarks used by [Figure 4](../../../visualizations/figures/classical_search_vs_random/README.md).
Each condition contains 10,000 standard-start games, with 5,000 in each seat.
It records W/D/L counts, termination categories, outcomes by termination, and
the SHA-256 of each retained local game log.

| Condition | Individual summary | Cap-adjudicated wins |
| --- | --- | ---: |
| Greedy H* | [Summary](../../../experiments/agent_benchmark/results/random_vs_greedy_hstar_10000.json) | 0 |
| Minimax depth 1 (Python) | [Summary](../../../experiments/minimax_depth_vs_random/results/depth_1_summary.json) | 437 |
| Minimax depth 2 (Python) | [Summary](../../../experiments/minimax_depth_vs_random/results/depth_2_summary.json) | 3,983 |
| Minimax depth 3 (Python) | [Summary](../../../experiments/minimax_depth_vs_random/results/depth_3_summary.json) | 5,104 |
| Minimax depth 4 (C++) | [Summary](../../../experiments/minimax_depth_vs_random/results/depth_4_games_10000_cpp_summary.json) | 6,502 |
| Minimax depth 5 (C++) | [Summary](../../../experiments/minimax_depth_vs_random/results/depth_5_games_10000_cpp_summary.json) | 6,962 |
| UCT MCTS, 1,000 simulations | [Summary](../../../experiments/mcts_vs_random/results/mcts_uct_1000sims_games_10000_cpp_summary.json) | 0 |

The cap is 300 player actions, adjudicated by captured-seed lead. All capped
minimax games were credited as wins. Reported score rates include these games
and count draws as half a win. The combined minimax experiment summary covers
only depth 3; the figure uses the individual summaries above.

The separate minimax adapters ([Python](../../../Algorithms/ayo_minimax.py),
[C++](../../../Algorithms/ayo_minimax_cpp.cc)) pass H* leaf estimates as large
as ±10 to OpenSpiel search, whose terminal branch returns ±1/0 utilities
([Python search](../../../open_spiel/open_spiel/python/algorithms/minimax.py),
[C++ search](../../../open_spiel/open_spiel/algorithms/minimax.cc)). This scale
mismatch requires a corrected rerun before drawing a clean conclusion about
depth. It does not establish the cause of the historical cap rates. Historical
native executable build provenance was not reconstructed, and the different
implementations and budgets do not establish an equal-runtime comparison.

The recount was retained from the removed article's evidence file; no
benchmark results were changed. Figure regeneration checks the individual
summaries against this evidence. With the original local logs available,
`python -B -m visualizations.classical_search_vs_random --verify-logs` also
recounts all 70,000 games and checks their hashes. The Greedy H* summary's
stored hash matches LF-normalized log bytes; the audit also preserves the
exact local byte hash. Python minimax stored hashes match the raw bytes.
Native summaries contain no stored log hashes, so those audit hashes identify
the locally recounted files without establishing historical hash verification.
