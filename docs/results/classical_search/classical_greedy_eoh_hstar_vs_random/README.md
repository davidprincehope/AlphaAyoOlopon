# Greedy EoH H* versus random (Python)

Recorded benchmark: **10,000 games** from the standard initial position, with the tested agent in each seat for 5,000 games. The opponent uses random legal actions. The move limit is 300 player actions.

| Wins | Draws | Losses | Win rate | Score rate | Move-limit wins |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 7,566 | 881 | 1,553 | 75.66% | 80.07% | 0 |

[results.json](results.json) preserves the original experiment settings and complete summary, together with the saved recount, termination breakdown and source hashes. Score credits draws as half a win.

The agent chooses the highest-valued successor using the frozen EoH H* evaluator, with the first legal action breaking ties. All games completed naturally; there were no move-limit outcomes.

[Shared source audit and methodology](../source_audit/README.md) describe the original recount and its limits. Re-auditing requires the original raw game histories.

[All experiment folders](../../README.md).
