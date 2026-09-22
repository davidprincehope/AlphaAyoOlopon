"""Focused tests for the Ayo Olopon heuristic features."""

from absl.testing import absltest
from unittest import mock
import pyspiel

from Model.ayo_olopon import ayo_olopon  # noqa: F401
from Algorithms import ayo_heuristic


def _state_with_position(board, current_player=0, captured=None, terminal=False):
  game = pyspiel.load_game("ayo_olopon", {"enable_cycle_reporting": True})
  state = game.new_initial_state()
  state.board = list(board)
  state.captured = list(captured or [0, 0])
  state.total_seeds = state.num_houses * game.num_seeds_per_house
  state._current_player = current_player
  state._game_over = terminal
  state._returns = ([1.0, -1.0] if captured and captured[0] > captured[1]
                    else [-1.0, 1.0] if captured and captured[1] > captured[0]
                    else [0.0, 0.0])
  state._positions_since_capture = {state._position_key()}
  return state


class AyoHeuristicTest(absltest.TestCase):

  def test_capture_advantage_positive_negative_and_perspective(self):
    state = _state_with_position([2] * 12, captured=[12, 4])
    self.assertEqual(ayo_heuristic.capture_advantage(state, 0), 8)
    self.assertEqual(ayo_heuristic.capture_advantage(state, 1), -8)

  def test_no_immediate_capture(self):
    state = _state_with_position([4] * 12)
    self.assertEqual(ayo_heuristic.immediate_tactical_potential(state, 0), 0)

  def test_one_immediate_capture(self):
    state = _state_with_position([1, 3, 0, 0, 0, 5, 2, 0, 0, 0, 0, 0])
    self.assertEqual(ayo_heuristic.immediate_tactical_potential(state, 0), 4)

  def test_multiple_captures_selects_maximum(self):
    state = _state_with_position([1, 3, 0, 0, 0, 5, 1, 3, 0, 0, 0, 5])
    self.assertEqual(ayo_heuristic.immediate_tactical_potential(state, 0), 4)

  def test_relay_sowing_capture_is_seen(self):
    state = _state_with_position([3, 0, 0, 1, 0, 1, 0, 2, 0, 0, 3, 0])
    self.assertEqual(ayo_heuristic.immediate_tactical_potential(state, 0), 4)

  def test_intermediate_four_belongs_to_row_owner(self):
    state = _state_with_position([0, 0, 0, 0, 0, 5, 3, 0, 0, 0, 0, 0])
    self.assertEqual(ayo_heuristic.immediate_tactical_potential(state, 0), 0)
    self.assertEqual(ayo_heuristic.immediate_tactical_potential(state, 1), 0)

  def test_both_players_as_player_to_move(self):
    board = [1, 3, 0, 0, 0, 5, 2, 0, 0, 0, 0, 0]
    self.assertEqual(ayo_heuristic.mobility(_state_with_position(board, 0), 0), 3)
    self.assertEqual(ayo_heuristic.mobility(_state_with_position(board, 1), 1), 1)

  def test_mobility_and_seed_control(self):
    state = _state_with_position([0, 0, 1, 2, 0, 0, 3, 0, 0, 0, 0, 0])
    self.assertEqual(ayo_heuristic.mobility(state, 0), 2)
    self.assertEqual(ayo_heuristic.seed_control(state, 0), 0)
    self.assertEqual(ayo_heuristic.seed_control(state, 1), 0)
    state = _state_with_position([4, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0])
    self.assertEqual(ayo_heuristic.seed_control(state, 0), 3)

  def test_automatic_four_seed_capture_is_reflected(self):
    state = _state_with_position([1, 3, 0, 0, 0, 5, 2, 0, 0, 0, 0, 0])
    child = state.child(0)
    self.assertEqual(child.captured, [4, 0])
    self.assertEqual(ayo_heuristic.immediate_tactical_potential(state, 0), 4)

  def test_terminal_win_loss_and_draw_override_positional_score(self):
    win = _state_with_position([0] * 12, captured=[25, 23], terminal=True)
    loss = _state_with_position([0] * 12, captured=[23, 25], terminal=True)
    draw = _state_with_position([0] * 12, captured=[24, 24], terminal=True)
    self.assertEqual(ayo_heuristic.evaluate_state(win, 0), 10.0)
    self.assertEqual(ayo_heuristic.evaluate_state(loss, 0), -10.0)
    self.assertEqual(ayo_heuristic.evaluate_state(draw, 0), 0.0)

  def test_perspective_antisymmetry_for_nonterminal_features(self):
    state = _state_with_position([2, 0, 1, 0, 3, 0, 1, 2, 0, 1, 0, 0], captured=[8, 4])
    features_0 = ayo_heuristic.extract_features(state, 0)
    features_1 = ayo_heuristic.extract_features(state, 1)
    self.assertEqual({k: -v for k, v in features_0.items()}, features_1)
    self.assertAlmostEqual(
        ayo_heuristic.evaluate_state(state, 0),
        -ayo_heuristic.evaluate_state(state, 1),
    )

  def test_original_state_is_unchanged(self):
    state = _state_with_position([2, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0])
    before = (list(state.board), list(state.captured), state.current_player(), str(state))
    ayo_heuristic.extract_features(state, 0)
    ayo_heuristic.evaluate_state(state, 1)
    after = (list(state.board), list(state.captured), state.current_player(), str(state))
    self.assertEqual(after, before)

  def test_candidate_heuristics_and_invalid_inputs(self):
    state = _state_with_position([4] * 12)
    self.assertEqual(ayo_heuristic.H_C(state, 0), 0.0)
    self.assertEqual(ayo_heuristic.H_CTMS(state, 0), 1.0)
    with self.assertRaises(ValueError):
      ayo_heuristic.candidate_heuristic("H_ALL")
    with self.assertRaises(ValueError):
      ayo_heuristic.evaluate_state(state, 0, terminal_value=4)

  def test_terminal_value_dominates_selected_positional_maximum(self):
    self.assertGreater(10.0, 4.0 + 2.0 + 0.5)
    self.assertLess(-10.0, -(4.0 + 2.0 + 0.5))
    with self.assertRaises(ValueError):
      ayo_heuristic.evaluate_state(
          _state_with_position([4] * 12), 0,
          weights={"capture": 4, "tactical": 2, "mobility": 0.5},
          terminal_value=6.5,
      )

  def test_terminal_evaluation_skips_feature_calculation_and_uses_perspective(self):
    win = _state_with_position([0] * 12, captured=[25, 23], terminal=True)
    with mock.patch.object(ayo_heuristic, "extract_features", side_effect=AssertionError):
      self.assertEqual(ayo_heuristic.evaluate_state(win, 0), 10.0)
      self.assertEqual(ayo_heuristic.evaluate_state(win, 1), -10.0)
    draw = _state_with_position([0] * 12, captured=[24, 24], terminal=True)
    self.assertEqual(ayo_heuristic.evaluate_state(draw, 0), 0.0)

  def test_positive_scalar_equivalence_detection(self):
    self.assertTrue(ayo_heuristic.equivalent_weight_vectors([4, 2, 0.5], [8, 4, 1]))
    self.assertFalse(ayo_heuristic.equivalent_weight_vectors([4, 2, 0.5], [8, 3, 1]))
    self.assertEqual(
        ayo_heuristic.normalize_weight_vector([4, 2, 0.5]),
        ayo_heuristic.normalize_weight_vector([8, 4, 1]),
    )


if __name__ == "__main__":
  absltest.main()
