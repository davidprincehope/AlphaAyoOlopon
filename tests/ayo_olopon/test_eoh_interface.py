"""Contract tests for the local EoH one-ply interface and validator."""
import sys
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments/EOH"))
import pyspiel
from Model.ayo_olopon import ayo_olopon  # registers game
from Algorithms.eoh_ayo import fixture_evaluator, greedy_action, state_view
from candidate_validator import validate


class EohInterfaceTest(unittest.TestCase):
  def test_legal_action_clone_immutability_and_perspective(self):
    game = pyspiel.load_game("ayo_olopon")
    state = game.new_initial_state()
    before = (tuple(state.board), tuple(state.captured), state.current_player())
    action, trace = greedy_action(state, fixture_evaluator)
    self.assertIn(action, state.legal_actions())
    self.assertEqual(before, (tuple(state.board), tuple(state.captured), state.current_player()))
    self.assertEqual([row["action"] for row in trace], state.legal_actions())
    # Actor remains player 0 even though every nonterminal child has player 1 to move.
    self.assertTrue(all(row["successor"].current_player == 1 for row in trace if not row["terminal"]))
    self.assertTrue(all(row["score"] == fixture_evaluator(row["successor"], 0)
                        for row in trace if not row["terminal"]))

  def test_ties_retain_first_legal_action(self):
    state = pyspiel.load_game("ayo_olopon").new_initial_state()
    action, _ = greedy_action(state, lambda view, player: 0.0)
    self.assertEqual(action, min(state.legal_actions()))

  def test_terminal_win_outranks_every_nonterminal_score(self):
    state = pyspiel.load_game("ayo_olopon").new_initial_state()
    state.board = [1, 3, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
    state.captured = [24, 19]
    action, rows = greedy_action(state, lambda view, player: -10.0)
    self.assertEqual(action, 0)
    self.assertTrue(rows[0]["terminal"])
    self.assertEqual(rows[0]["score"], 11.0)

  def test_policy_accepts_seat_swapped_turn(self):
    state = pyspiel.load_game("ayo_olopon").new_initial_state()
    state._current_player = 1
    action, rows = greedy_action(state, fixture_evaluator)
    self.assertIn(action, state.legal_actions())
    self.assertTrue(all(row["successor"].current_player == 0 for row in rows
                        if not row["terminal"]))

  def test_validator_known_good_and_rejects_import(self):
    game = pyspiel.load_game("ayo_olopon")
    view = state_view(game.new_initial_state())
    good = "def evaluate_state(state_view, player_id):\n    return 0.0\n"
    bad = "import os\ndef evaluate_state(state_view, player_id):\n    return 0.0\n"
    slow = "def evaluate_state(state_view, player_id):\n    while True:\n        pass\n"
    self.assertTrue(validate(good, [(view, 0), (view, 1)])["ok"])
    self.assertEqual(validate(bad, [(view, 0)])["reason"], "disallowed_syntax")
    self.assertEqual(validate(slow, [(view, 0)])["reason"], "disallowed_syntax")


if __name__ == "__main__":
  unittest.main()
