"""Tests for the vanilla Ayo MCTS adapter."""

import numpy as np
from absl.testing import absltest
import pyspiel

from Model.ayo_olopon import ayo_olopon  # pylint: disable=unused-import
from Algorithms import ayo_mcts


class AyoMctsTest(absltest.TestCase):

  def test_mcts_bot_can_choose_an_initial_action(self):
    game = pyspiel.load_game("ayo_olopon")
    bot = ayo_mcts.make_bot(game, simulations=20, seed=7)
    state = game.new_initial_state()

    action = bot.step(state)

    self.assertIn(action, state.legal_actions())

  def test_mcts_bots_can_play_a_complete_game(self):
    game = pyspiel.load_game("ayo_olopon")
    result = ayo_mcts.play_game(game, simulations=5, seed=11)

    self.assertTrue(result["state"].is_terminal())
    self.assertLen(result["returns"], 2)
    self.assertAlmostEqual(sum(result["returns"]), 0.0)
    self.assertGreater(len(result["actions"]), 0)
    self.assertTrue(np.all(np.isfinite(result["returns"])))


if __name__ == "__main__":
  absltest.main()
