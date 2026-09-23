"""Checks the initial-state depth 1/2/3 comparison runner."""

import unittest

import pyspiel

from experiments.handcrafted_vs_eoh import compare as shared
from experiments.handcrafted_vs_eoh import compare_initial
from Model.ayo_olopon import ayo_olopon  # noqa: F401 - registers game


class InitialPositionSearchTest(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    cls.game = pyspiel.load_game("ayo_olopon", {"enable_cycle_reporting": True})
    cls.state = cls.game.new_initial_state()

  def test_standard_starting_state(self):
    self.assertEqual(list(self.state.board), [4] * 12)
    self.assertEqual(list(self.state.captured), [0, 0])
    self.assertEqual(self.state.current_player(), 0)

  def test_all_policies_and_depths_choose_legal_actions(self):
    for heuristic in shared.PLAYERS:
      for depth in compare_initial.SEARCH_DEPTHS:
        action, trace = compare_initial.choose_action(self.state, heuristic, depth)
        self.assertIn(action, self.state.legal_actions())
        self.assertEqual(trace["depth_plies"], depth)
        self.assertEqual(len(trace["root_action_values"]), len(self.state.legal_actions()))
        self.assertGreater(trace["leaf_or_terminal_nodes"], 0)

  def test_invalid_depth_is_rejected(self):
    with self.assertRaises(ValueError):
      compare_initial.choose_action(self.state, "eoh", 4)


if __name__ == "__main__":
  unittest.main()
