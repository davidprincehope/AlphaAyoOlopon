# UCT MCTS, 1,000 simulations, versus random (C++)

Recorded benchmark: **10,000 games** from the standard initial position, with the tested agent in each seat for 5,000 games. The opponent uses random legal actions. The move limit is 300 player actions.

| Wins | Draws | Losses | Win rate | Score rate | Move-limit wins |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 9,920 | 72 | 8 | 99.20% | 99.56% | 0 |

[results.json](results.json) preserves the original experiment settings and complete summary, together with the saved recount, termination breakdown and source hashes. Score credits draws as half a win.

The native OpenSpiel MCTSBot uses UCT with uniform priors, 1,000 simulations per move, exploration constant sqrt(2), and one random rollout per leaf. There were no move-limit outcomes.

The saved recount identifies the local game-log bytes. The native summary contains no stored log hash, and historical executable-build provenance was not reconstructed.

[Shared source audit and methodology](../source_audit/README.md) describe the original recount and its limits. Re-auditing requires the original raw game histories.

[All experiment folders](../../README.md).
