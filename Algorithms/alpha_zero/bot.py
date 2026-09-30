"""Load a trained checkpoint and construct an evaluation-only PUCT bot."""

import json
from pathlib import Path

import numpy as np
import pyspiel

from Algorithms.alpha_zero.runtime import use_repository_open_spiel
from Algorithms.alpha_zero.config import Settings
from Algorithms.alpha_zero.game import OBSERVATION_VERSION


def load_bot(run_path, checkpoint, simulations=100, seed=0):
    """Return (game, bot) using the architecture and rules saved with the run.

    checkpoint is the integer suffix (e.g. 10 or -1), not a checkpoint path.
    Evaluation has no Dirichlet noise and chooses the best search action.
    """
    if simulations < 2:
        raise ValueError('simulations must be >= 2')
    use_repository_open_spiel()
    from open_spiel.python.algorithms import mcts
    from open_spiel.python.algorithms.alpha_zero import evaluator, utils

    run_path = Path(run_path).resolve()
    manifest = json.loads((run_path / 'manifest.json').read_text(encoding='utf-8'))
    if manifest['observation_version'] != OBSERVATION_VERSION:
        raise ValueError('Checkpoint observation version does not match this adapter')
    if manifest.get('architecture_version') != 'ayo_mlp_185863_v1':
        raise ValueError('Checkpoint architecture version does not match this model')
    if manifest.get('value_perspective') != 'player_to_move':
        raise ValueError('Checkpoint value perspective does not match this evaluator')
    settings = Settings(**manifest['settings'])
    checkpoint = int(checkpoint)
    if not (run_path / f'checkpoint-{checkpoint}').is_dir():
        raise FileNotFoundError(run_path / f'checkpoint-{checkpoint}')
    game = pyspiel.load_game(settings.game_string)
    model = utils.api_selector(settings.nn_api_version).Model.build_model(
        settings.nn_model, game.observation_tensor_shape(), game.num_distinct_actions(),
        settings.nn_width, settings.nn_depth, settings.weight_decay,
        settings.learning_rate, str(run_path),
        decouple_weight_decay=settings.decouple_weight_decay,
    )
    model.load_checkpoint(checkpoint)
    neural_evaluator = evaluator.AlphaZeroEvaluator(game, model)
    bot = mcts.MCTSBot(
        game, settings.uct_c, simulations, neural_evaluator, solve=False,
        random_state=np.random.RandomState(seed), dirichlet_noise=None,
        child_selection_fn=mcts.SearchNode.puct_value,
    )
    return game, bot
