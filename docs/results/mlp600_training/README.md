# MLP training through 600 rounds

The original A100 run used the **185,863-parameter MLP**, 28 self-play actors,
64 PUCT simulations per move, and the 1,000-action collection horizon.
It reached **600 learner rounds**, **63,563 trajectories**, **2,473,058 collected
positions** and **76,608 gradient updates** across resumed sessions.

[Training manifest](training_manifest.json) records the model, game, settings
and package versions. [Training summary](training_summary.json) contains final
counters and losses. [Source provenance](source_provenance.json) identifies
the original local records and the hashes of these published exports.

The [training guide](../../../experiments/alpha_zero/README.md) explains launch,
checkpoints and resume. The current checkpoint measurements are the
[DATASET ≤4 CAPTURE suite](../checkpoint_progression/README.md) and the
[capture-only greedy comparison](../c600_comparisons/c600_puct64_vs_capture_only_greedy/README.md).
The [C600 result index](../README.md) links all seven completed C600
experiments, including PUCT 64 against MCTS 128 and the C350 matchup.

Model exports, replay, Adam state, learner logs and cloud records are local
artifacts under ignored `runs/`. The public tree retains compact evidence for
the reported training and experiments.
