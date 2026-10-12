"""Share a frozen inference model while giving each decision its own search RNG."""
import threading

import numpy as np

from Algorithms.alpha_zero.bot import load_checkpoint_model
from Algorithms.alpha_zero.evaluation import make_neural_agent


class AlphaZeroEngine:
    def __init__(self, run, checkpoint=600, simulations=64):
        if type(checkpoint) is not int or checkpoint < 1:
            raise ValueError('Choose a positive frozen checkpoint round')
        if type(simulations) is not int or not 2 <= simulations <= 4096:
            raise ValueError('Search simulations must be between 2 and 4096')
        self.game, self.model, self.settings = load_checkpoint_model(run, checkpoint)
        self.checkpoint = checkpoint
        self.simulations = simulations
        self._lock = threading.Lock()
        # Compile and validate the first prediction before accepting games.
        state = self.game.new_initial_state()
        value, policy = self.model.inference(state.observation_tensor(), state.legal_actions_mask())
        if (np.asarray(value).size != 1 or not np.isfinite(value).all()
                or np.asarray(policy).shape != (6,) or not np.isfinite(policy).all()
                or (np.asarray(policy) < 0).any() or not np.isclose(np.sum(policy), 1)):
            raise ValueError('Checkpoint produced an invalid policy/value prediction')

    @property
    def info(self):
        return {'name': 'AlphaZero', 'checkpoint': self.checkpoint,
                'simulations': self.simulations, 'max_moves': self.game.max_game_length()}

    def choose_action(self, state, seed):
        # Host search and compiled inference are serialized across browser sessions.
        with self._lock:
            bot = make_neural_agent(self.game, self.model, self.simulations,
                                    self.settings.uct_c, seed)
            return int(bot.step(state))
