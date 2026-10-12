# OpenSpiel attribution

AlphaAyoOlopon uses [OpenSpiel](https://github.com/google-deepmind/open_spiel),
including its JAX/Flax AlphaZero implementation and search algorithms, through
the [Ayo fork](https://github.com/davidprincehope/open_spiel/tree/alpha-ayo-mlp-public)
checked out under `open_spiel/`.

OpenSpiel is licensed under the **Apache License, Version 2.0**. The full license
is retained in [open_spiel/LICENSE](open_spiel/LICENSE), and the original
copyright and license notices remain in its source files. This attribution
does not assign a license to AlphaAyoOlopon's original code.

The Ayo fork modifies four upstream files:

- `open_spiel/python/algorithms/alpha_zero/alpha_zero.py`: canonical Ayo replay,
  resumable learner snapshots, checkpoint broadcasts, and evaluation integration.
- `open_spiel/python/algorithms/alpha_zero/evaluator.py`: conversion from
  player-to-move values to player-indexed returns for MCTS.
- `open_spiel/python/algorithms/alpha_zero/model_linen.py`: the canonical Ayo
  policy/value MLP and its mean-squared value loss.
- `open_spiel/python/utils/data_logger.py`: append-mode logging for resumed runs.

Each modified file carries a change notice alongside its retained upstream
license header. The Modal image includes the OpenSpiel license and this
attribution with the copied source, plus an upstream `NOTICE` file if one is
present. Other distributions containing OpenSpiel source or binaries must also
include its license, retained applicable notices, and source modification notices.
