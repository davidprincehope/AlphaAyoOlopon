"""Unit tests for the isolated handcrafted-versus-EoH experiment."""

import unittest

import pyspiel

from experiments.handcrafted_vs_eoh import compare
from experiments.heuristic import run_experiment as frozen_protocol
from Model.ayo_olopon import ayo_olopon  # noqa: F401 - registers game


class ComparePolicyTest(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    cls.opening = next(
        item for item in frozen_protocol.load_frozen_openings()
        if item["split"] == "development"
    )
    game = pyspiel.load_game("ayo_olopon", {"enable_cycle_reporting": True})
    cls.state = frozen_protocol.state_from_snapshot(game, cls.opening)

  def test_both_heuristics_choose_legal_greedy_actions(self):
    for heuristic in compare.PLAYERS:
      action, trace = compare.choose_action(self.state, heuristic, depth=1)
      self.assertIn(action, self.state.legal_actions())
      self.assertEqual(trace["depth"], 1)
      self.assertEqual(len(trace["actions"]), len(self.state.legal_actions()))

  def test_two_ply_score_is_worst_reply_for_each_root_action(self):
    for heuristic in compare.PLAYERS:
      action, trace = compare.choose_action(self.state, heuristic, depth=2)
      self.assertIn(action, self.state.legal_actions())
      for row in trace["actions"]:
        if row["reply_values"]:
          self.assertEqual(row["score"], min(reply["score"] for reply in row["reply_values"]))

  def test_depth_must_be_supported(self):
    with self.assertRaises(ValueError):
      compare.choose_action(self.state, "eoh", depth=3)


if __name__ == "__main__":
  unittest.main()
