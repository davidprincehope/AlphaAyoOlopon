# Local AlphaZero experiment data

The retained Modal training run ends at C600 and is verified inside this checkout:

- [Final C600 replay buffer](../../runs/alpha_zero/modal-a100-600-20261003/training-checkpoints/step-000600/replay.npz)
- [All C1-C600 network exports](../../runs/alpha_zero/modal-a100-600-20261003/inference-checkpoints/)
- [Saved replay and optimizer snapshots](../../runs/alpha_zero/modal-a100-600-20261003/training-checkpoints/)
- [Training losses and timings](../../runs/alpha_zero/modal-a100-600-20261003/learner.jsonl)
- [Download provenance and SHA-256 verification](../../runs/alpha_zero/modal-a100-600-20261003-download.json)

The final buffer contains 16,384 observations, legal
action masks, MCTS policy targets, and outcome targets. Earlier buffers are
available in 61 saved training snapshots. The analyses remain under the associated
`modal-c600-*` folders. The balanced comparisons using only retained models are in
`runs/checkpoint_strength/balanced-progression-through-c600-20261006/`.

These large generated artifacts follow the repository's existing `.gitignore`
rule and are not distributed by a Git clone. The links above refer to this local
checkout. No new experiment was run as part of fetching the data.


Compact public summaries, completion/verification records and source hashes are
available in [docs/results/alpha_zero_600](../../docs/results/alpha_zero_600/README.md).
The reviewed figure exports and plotted tables are versioned. Their raw-log
verification scripts require the retained local runs described above.
