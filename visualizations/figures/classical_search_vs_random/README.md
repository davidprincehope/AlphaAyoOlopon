# Classical search against random play

Figure 4 shows reported score rates alongside cap-adjudicated game shares for Greedy H*, minimax + H* at depths 1–5, and UCT MCTS with 1,000 simulations and one random rollout per leaf. Each condition has 10,000 games from the standard initial board, 5,000 in each seat.

Minimax rows are historical and shaded. Depths 1–3 use Python; depths 4–5 use C++. Greedy H* uses Python; the UCT MCTS baseline uses C++. All seven records use a 300-action cap, adjudicated by captured-seed lead. Every capped minimax game was credited as a win. Scores include capped outcomes and count draws as half a win; they are not natural-game win rates.

The audited minimax source combines terminal utilities of ±1 with H* leaf estimates up to ±10. A corrected rerun is needed before drawing a clean conclusion about depth. Historical native executable build provenance was not reconstructed. Implementations and budgets differ, so this is not an equal-runtime comparison. No uncertainty estimates are plotted.

[PNG](classical_search_vs_random.png) is 300 dpi; [SVG](classical_search_vs_random.svg) retains editable text. [chart_data.csv](chart_data.csv) contains the seven plotted rows; [figure_provenance.json](figure_provenance.json) records source and export hashes.

## Sources

The plot uses the individual summaries listed in [the source audit](../../../docs/results/classical_search/README.md), cross-checking [its saved recount](../../../docs/results/classical_search/source_audit.json). The combined minimax experiment summary is not used.

- [greedy_hstar](../../../experiments/agent_benchmark/results/random_vs_greedy_hstar_10000.json)
- [minimax_depth_1](../../../experiments/minimax_depth_vs_random/results/depth_1_summary.json)
- [minimax_depth_2](../../../experiments/minimax_depth_vs_random/results/depth_2_summary.json)
- [minimax_depth_3](../../../experiments/minimax_depth_vs_random/results/depth_3_summary.json)
- [minimax_depth_4](../../../experiments/minimax_depth_vs_random/results/depth_4_games_10000_cpp_summary.json)
- [minimax_depth_5](../../../experiments/minimax_depth_vs_random/results/depth_5_games_10000_cpp_summary.json)
- [mcts_1000](../../../experiments/mcts_vs_random/results/mcts_uct_1000sims_games_10000_cpp_summary.json)

## Regenerate

The summaries and saved audit are included in Git:

```powershell
.venv/Scripts/python.exe -B -m visualizations.classical_search_vs_random
```

To additionally recount the original 70,000 local game records and verify their hashes:

```powershell
.venv/Scripts/python.exe -B -m visualizations.classical_search_vs_random --verify-logs
```
