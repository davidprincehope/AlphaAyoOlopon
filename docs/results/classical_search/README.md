# Classical search experiments

These studies evaluate classical opponents against random play from the standard initial position.

## Corrected minimax rerun

The [corrected depth comparison](corrected_minimax_depths_1_to_5_vs_random/README.md) uses Python/OpenSpiel at every depth, a bounded H* evaluator aligned with terminal utilities, common paired seeds, and a search-visible 1,000-action horizon. All **50,000 games** finished without action-limit outcomes.

| Depth | Games | Score rate |
| ---: | ---: | ---: |
| 1 | 10,000 | 80.14% |
| 2 | 10,000 | 95.30% |
| 3 | 10,000 | 99.34% |
| 4 | 10,000 | 99.14% |
| 5 | 10,000 | 98.94% |

The rerun supports substantial improvement from depth 1 to depth 3, but scores do not increase monotonically beyond depth 3. The report includes paired confidence intervals, results by seat, replay verification, source hashes and published paired scores.

## Other classical benchmarks

| Classical experiment | Implementation | Report | Result file |
| --- | --- | --- | --- |
| Greedy EoH H* versus random | Python | [Report](classical_greedy_eoh_hstar_vs_random/README.md) | [JSON](classical_greedy_eoh_hstar_vs_random/results.json) |
| UCT MCTS, 1,000 simulations, versus random | C++ | [Report](classical_uct_mcts_1000_simulations_cpp_vs_random/README.md) | [JSON](classical_uct_mcts_1000_simulations_cpp_vs_random/results.json) |

[Shared source audit](source_audit/README.md) | [All result groups](../README.md).
