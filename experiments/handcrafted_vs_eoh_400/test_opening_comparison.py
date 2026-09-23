"""Tests the fresh-opening generator and the depth-policy adapter."""

import unittest

import pyspiel

from experiments.handcrafted_vs_eoh_400 import run_opening_comparison as experiment
from experiments.heuristic import run_experiment as frozen_protocol
from Model.ayo_olopon import ayo_olopon  # noqa: F401 - registers game


class OpeningComparisonTest(unittest.TestCase):
  def test_generation_is_reproducible_and_distinct(self):
    first = experiment.generate_positions(12, seed=20261011)
    second = experiment.generate_positions(12, seed=20261011)
    self.assertEqual(first, second)
    self.assertEqual(len({item["state_hash"] for item in first}), 12)
    self.assertTrue(all(4 <= item["plies_from_initial"] <= 32 for item in first))
    self.assertTrue(all(not item["is_terminal"] for item in first))

  def test_generated_snapshots_restore_as_legal_states(self):
    opening = experiment.generate_positions(1, seed=20261012)[0]
    game = pyspiel.load_game("ayo_olopon", {"enable_cycle_reporting": True})
    state = frozen_protocol.state_from_snapshot(game, opening)
    self.assertFalse(state.is_terminal())
    self.assertTrue(state.legal_actions())

  def test_search_handles_depths_one_through_three(self):
    game = pyspiel.load_game("ayo_olopon", {"enable_cycle_reporting": True})
    state = game.new_initial_state()
    for depth in (1, 2, 3):
      action, trace = experiment.compare_initial.choose_action(state, "eoh", depth)
      self.assertIn(action, state.legal_actions())
      self.assertEqual(trace["depth_plies"], depth)


if __name__ == "__main__":
  unittest.main()
