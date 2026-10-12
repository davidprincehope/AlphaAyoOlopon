# Minimax depth 4 versus random (C++)

Recorded benchmark: **10,000 games** from the standard initial position, with the tested agent in each seat for 5,000 games. The opponent uses random legal actions. The move limit is 300 player actions.

| Wins | Draws | Losses | Win rate | Score rate | Move-limit wins |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 9,291 | 664 | 45 | 92.91% | 96.23% | 6,502 |

[results.json](results.json) preserves the original experiment settings and complete summary, together with the saved recount, termination breakdown and source hashes. Score credits draws as half a win.

Historical results include **6,502 move-limit wins**, adjudicated by captured-seed lead. H* leaf estimates as large as ±10 were combined with terminal utilities of ±1/0. A corrected rerun is needed before drawing conclusions about depth; these records preserve the original measurements.

The saved recount identifies the local game-log bytes. The native summary contains no stored log hash, and historical executable-build provenance was not reconstructed.

[Shared source audit and methodology](../source_audit/README.md) describe the original recount and its limits. Re-auditing requires the original raw game histories.

[All experiment folders](../../README.md).
