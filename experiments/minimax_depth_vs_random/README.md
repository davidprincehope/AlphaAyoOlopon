# Minimax depth rerun

Use `run_experiment.py` for the corrected experiment. It delegates to the
source-preserved `rerun_experiment.py` implementation used for the published
50,000-game study. Superseded minimax results and the legacy runners that
adjudicated unfinished games from captured-seed lead have been removed.

[Verified results](../../docs/results/classical_search/corrected_minimax_depths_1_to_5_vs_random/README.md)

From the repository root:

```powershell
.venv/Scripts/python.exe -B -m experiments.minimax_depth_vs_random.run_experiment --games 10000 --depths 1 2 3 4 5 --seed 20261019 --max-actions 1000 --workers 3 --output-dir runs/minimax_depth_vs_random/corrected-20261019
```

Add `--resume` to continue that exact run. Use a fresh output directory for a
different protocol or source revision. Existing evidence is never overwritten
without an explicit resume, which verifies the source/runtime contract and
replays every saved game before accepting it.

Each depth uses the Python game and Python OpenSpiel alpha-beta implementation.
One search depth is one complete player action, including all relay sowing.
The H* leaf evaluator is `clip(raw H*, -10, 10) / 11`, bounded by ±10/11;
OpenSpiel handles terminal positions first with utilities −1, 0 and +1.
This preserves H* ordering among nonterminal positions while keeping proven
wins above all heuristic estimates and proven losses below them. It does not
turn H* into a calibrated probability of winning.

There are 5,000 seed pairs at each depth, each played with minimax in both
seats from the standard start. Pair `i` uses `20261019 + i` at every depth
and in both seats. Random play uses Python `random.Random(seed).choice`
over legal actions. Identical seeds do not imply identical move sequences
after search choices cause the game trajectories to diverge. Minimax ties
select the first action in legal-action order.

The game enforces a search-visible 1,000-action horizon. Any internal or external
action limit is unresolved, including a terminal position adjudicated by the
engine at that horizon. Unresolved games receive no W/D/L credit. The report
gives full-sample score bounds by assigning unresolved games zero or one point;
its ordinary score rate and confidence interval are withheld if any game is
unresolved. A completed-only score is descriptive and may be selection-biased.
Rule-based endings, including repetition resolution, are completed games.

Each raw record saves its action trace, final board, captures, outcome and seed.
The parent process replays every game and checks the seeded random actions.
The manifest retains source hashes, exact source snapshots, package versions
and a hash of the loaded `pyspiel` extension. Per-depth summaries hash their
raw logs. Decision timings are observed under concurrent workers and should
not be interpreted as isolated runtime benchmarks.

Score is `(wins + 0.5 * draws) / games`. Individual 95% percentile bootstrap
intervals use 10,000 resamples of whole seed/seat pairs. Differences between
depths use matched pair scores, preserving their correlation across depths.
The runner reports adjacent-depth and depth-1 comparisons. These intervals
are not adjusted for multiple comparisons. The experiment measures strength
against random from the standard start, not general strength or equal-runtime
performance against other search agents.

Publish a finished run with a second full replay, recount and interval check:

```powershell
.venv/Scripts/python.exe -B -m experiments.minimax_depth_vs_random.export_rerun --run runs/minimax_depth_vs_random/corrected-20261019 --output docs/results/classical_search/corrected_minimax_depths_1_to_5_vs_random
```

The export includes a report, JSON summary and one CSV row per seed pair with
both seat scores at every depth. Full move traces remain in the local run.
