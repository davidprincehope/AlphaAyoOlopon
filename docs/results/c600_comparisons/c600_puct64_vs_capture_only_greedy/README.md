# C600 versus capture-only greedy

Completed on 12 October 2026 using the original MLP C600 checkpoint and
64 neural PUCT simulations per move.

| Player | Wins | Draws | Losses | Win rate | Score rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| C600, PUCT 64 | 281 | 100 | 19 | 70.25% | 82.75% |
| Capture-only greedy | 19 | 100 | 281 | 4.75% | 17.25% |

Score rate counts a draw as half a win. C600's 95% opening-pair bootstrap
score interval is **80.50%–85.00%**, with 10,000 resamples.
This interval describes sampled-opening variation and excludes training-seed
and search-seed uncertainty.

The [capture-only opponent](../../../../Algorithms/ayo_capture_greedy.py) resolves
each complete legal action, including all relay laps, and chooses the largest
increase in the mover's captured score. Lowest action ID breaks ties. It uses
the captured score after the action, including any terminal collection, and
has no additional terminal bonus, opponent-score term, tactical feature,
mobility feature, learned evaluator, or deeper search.

The match uses all 200 [DATASET ≤4 CAPTURE openings](../../../../experiments/checkpoint_strength/dataset/dataset_le4capture_v3_200.json),
each replayed in both seats for **400 games**. Prefixes restore repetition
history and elapsed moves. Seeds are `20261003 + opening_index`, shared across
the seat assignments. The training game has a 1,000-action collection horizon.
C600 selects maximum root visits with lowest-action-ID ties, without root
noise or temperature sampling. The original model export SHA-256 is
`543d4842c6b41cdbd3eb8e8221c8ca85c4e113a8096f610f1300fa0d4758cea0`.

Replay verification checked all **400 terminal games**,
**9123 continuation actions**, and
**4630 greedy decisions**, together with
opening identities, seeds, seats, terminal captures, outcomes and source hashes.
The captured-seed maximizing rule was recomputed at every greedy turn.
Action-limit outcomes: **0**. The 21 targeted baseline and
existing evaluator tests passed.

[Verified summary](results.json) retains both players' statistics, per-seat
results, intervals and verification counts. [Evaluation manifest](evaluation_manifest.json)
records the model, game, settings, source hashes and package versions.
Raw game histories are excluded from the public tree; re-auditing the original
matches requires a separately retained copy. The commands below generate a new
set of game histories using the saved checkpoint and opening book.

Reproduce with the retained checkpoint and a fresh result path:

```powershell
.venv/Scripts/python.exe -B -m experiments.alpha_zero.offline_evaluate --run runs/alpha_zero/modal-a100-600-20261003 --config experiments/checkpoint_strength/configs/c600_vs_capture_greedy_64.json --output runs/checkpoint_strength/c600-vs-capture-greedy-rerun/results.jsonl
```
