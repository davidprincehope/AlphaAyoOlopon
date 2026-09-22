"""Tests for the depth-limited Ayo minimax adapter."""

import numpy as np
from absl.testing import absltest
import pyspiel

from Model.ayo_olopon import ayo_olopon  # pylint: disable=unused-import
from Algorithms import ayo_minimax


class AyoMinimaxTest(absltest.TestCase):

  def test_minimax_bot_can_choose_an_initial_action(self):
    game = pyspiel.load_game("ayo_olopon")
    bot = ayo_minimax.make_bot(game, maximum_depth=2)
    state = game.new_initial_state()

    action = bot.step(state)

    self.assertIn(action, state.legal_actions())

  def test_minimax_bots_can_play_a_complete_game(self):
    game = pyspiel.load_game("ayo_olopon")
    result = ayo_minimax.play_game(game, maximum_depth=2)

    self.assertTrue(result["state"].is_terminal())
    self.assertLen(result["returns"], 2)
    self.assertAlmostEqual(sum(result["returns"]), 0.0)
    self.assertGreater(len(result["actions"]), 0)
    self.assertTrue(np.all(np.isfinite(result["returns"])))


if __name__ == "__main__":
  absltest.main()
