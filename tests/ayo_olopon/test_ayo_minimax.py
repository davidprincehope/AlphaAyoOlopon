"""Tests for the depth-limited Ayo minimax adapter."""

import numpy as np
import copy
from absl.testing import absltest
import pyspiel

from Model.ayo_olopon import ayo_olopon  # pylint: disable=unused-import
from Algorithms import H_star_ayo
from Algorithms import ayo_minimax


class AyoMinimaxTest(absltest.TestCase):

  def test_minimax_defaults_to_h_star_evaluator(self):
    game = pyspiel.load_game("ayo_olopon")
    bot = ayo_minimax.make_bot(game)

    self.assertIs(bot.value_function, H_star_ayo.evaluate_state)

  def test_minimax_bot_can_choose_an_initial_action(self):
    game = pyspiel.load_game("ayo_olopon")
    bot = ayo_minimax.make_bot(game, maximum_depth=2)
    state = game.new_initial_state()

    action = bot.step(state)

    self.assertIn(action, state.legal_actions())

  def test_proven_win_beats_a_high_nonterminal_h_star_estimate(self):
    class Position:
      board = [4] * 12
      captured = [24, 0]
      branch = None

      def clone(self):
        return copy.copy(self)

      def is_terminal(self):
        return self.branch == 0

      def player_return(self, player):
        return 1.0 if player == 0 else -1.0

      def current_player(self):
        return 0 if self.branch is None else 1

      def legal_actions(self):
        return [0, 1]

      def apply_action(self, action):
        self.branch = action

    bot = ayo_minimax.make_bot(pyspiel.load_game("ayo_olopon"), maximum_depth=1)
    value, action = bot.search(Position())
    self.assertEqual(action, 0)
    self.assertEqual(value, 1.0)

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
