"""Load a trained checkpoint and construct an evaluation-only PUCT bot."""

import json
from pathlib import Path

import pyspiel

from Algorithms.alpha_zero.runtime import use_repository_open_spiel
from Algorithms.alpha_zero.config import Settings
from Algorithms.alpha_zero.game import OBSERVATION_VERSION


def load_checkpoint_model(run_path, checkpoint):
    """Return (game, inference model, settings) without creating training state."""
    use_repository_open_spiel()
    from Algorithms.alpha_zero.inference import InferenceModel, load_parameters, checkpoint_path

    run_path = Path(run_path).resolve()
    manifest = json.loads((run_path / 'manifest.json').read_text(encoding='utf-8'))
    if manifest['observation_version'] != OBSERVATION_VERSION:
        raise ValueError('Checkpoint observation version does not match this adapter')
    if manifest.get('architecture_version') != 'ayo_mlp_185863_v1':
        raise ValueError('Checkpoint architecture version does not match this model')
    if manifest.get('value_perspective') != 'player_to_move':
        raise ValueError('Checkpoint value perspective does not match this evaluator')
    settings = Settings(**manifest['settings'])
    if manifest.get('game') != settings.game_string:
        raise ValueError('Saved game and settings disagree')
    checkpoint = int(checkpoint)
    source = checkpoint_path(run_path, checkpoint)
    if source.is_file():
        import numpy as np
        with np.load(source, allow_pickle=False) as data:
            metadata = json.loads(str(data['__metadata__']))
        contract = ('architecture_version', 'observation_version', 'value_perspective', 'game')
        if metadata['training_step'] != checkpoint or any(
                metadata['manifest'].get(key) != manifest.get(key) for key in contract):
            raise ValueError('Checkpoint metadata does not match the run')
    game = pyspiel.load_game(settings.game_string)
    model = InferenceModel(load_parameters(source))
    return game, model, settings


def load_bot(run_path, checkpoint, simulations=None, seed=0):
    """Return (game, bot) using the architecture and rules saved with the run.

    checkpoint is the integer suffix (e.g. 10 or -1), not a checkpoint path.
    Evaluation has no Dirichlet noise and chooses the best search action.
    """
    if simulations is not None and simulations < 2:
        raise ValueError('simulations must be >= 2')
    from Algorithms.alpha_zero.evaluation import make_neural_agent
    game, model, settings = load_checkpoint_model(run_path, checkpoint)
    bot = make_neural_agent(game, model,
                            settings.max_simulations if simulations is None else simulations,
                            settings.uct_c, seed)
    return game, bot
