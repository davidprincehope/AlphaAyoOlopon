"""Frozen model loading validates contracts and stays independent of training."""
from dataclasses import asdict
import json

import numpy as np
import pytest

from Algorithms.alpha_zero.runtime import use_repository_open_spiel
use_repository_open_spiel()
from Algorithms.alpha_zero.bot import load_bot, load_checkpoint_model
from Algorithms.alpha_zero.game import OBSERVATION_VERSION
from Algorithms.alpha_zero.config import Settings
from Algorithms.alpha_zero.inference import export_round
from play.engine import AlphaZeroEngine


@pytest.fixture
def frozen_run(tmp_path):
    import jax
    import jax.numpy as jnp
    from open_spiel.python.algorithms.alpha_zero.model_linen import AlphaZeroModel
    settings = Settings(max_moves=4, max_simulations=4)
    manifest = {'settings': asdict(settings), 'architecture_version': 'ayo_mlp_185863_v1',
                'observation_version': OBSERVATION_VERSION,
                'value_perspective': 'player_to_move', 'game': settings.game_string}
    network = AlphaZeroModel('ayo_mlp', (15, 1, 1), 6, 256, 3)
    params = network.init(jax.random.PRNGKey(7), jnp.zeros((15, 1, 1)), False)['params']
    (tmp_path / 'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
    path = export_round(tmp_path, 1, params, manifest)
    return tmp_path, manifest, path


def test_frozen_engine_and_existing_bot_do_not_create_training_state(frozen_run, monkeypatch):
    import optax
    from open_spiel.python.algorithms.alpha_zero import model_linen, replay_buffer
    run, _, source = frozen_run
    before = source.read_bytes()

    def forbidden(*args, **kwargs):
        raise AssertionError('Playing must not create training state')

    monkeypatch.setattr(optax, 'adam', forbidden)
    monkeypatch.setattr(model_linen.Model, 'build_model', forbidden)
    monkeypatch.setattr(replay_buffer.Buffer, '__init__', forbidden)
    engine = AlphaZeroEngine(run, checkpoint=1, simulations=4)
    game, bot = load_bot(run, 1, simulations=4, seed=9)
    state = game.new_initial_state()
    assert engine.choose_action(state, 9) == bot.step(state)
    assert state.move_number() == 0
    assert source.read_bytes() == before
    assert engine.info['checkpoint'] == 1


@pytest.mark.parametrize('field,value', [('observation_version', 'wrong'),
                                      ('value_perspective', 'player_zero'),
                                      ('architecture_version', 'wrong'),
                                      ('game', 'ayo_olopon')])
def test_rejects_incompatible_manifest(frozen_run, field, value):
    run, manifest, _ = frozen_run
    manifest[field] = value
    (run / 'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
    with pytest.raises(ValueError):
        load_checkpoint_model(run, 1)


def test_rejects_mislabeled_checkpoint(frozen_run):
    run, _, path = frozen_run
    with np.load(path, allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    metadata = json.loads(str(arrays['__metadata__']))
    metadata['training_step'] = 2
    arrays['__metadata__'] = np.asarray(json.dumps(metadata))
    np.savez_compressed(path, **arrays)
    with pytest.raises(ValueError, match='metadata'):
        load_checkpoint_model(run, 1)


@pytest.mark.parametrize('checkpoint,simulations', [(0, 64), (-1, 64), (True, 64), (1, 1), (1, 4097), (1, True)])
def test_invalid_play_configuration_is_rejected_before_loading(checkpoint, simulations, tmp_path):
    with pytest.raises(ValueError):
        AlphaZeroEngine(tmp_path, checkpoint, simulations)
