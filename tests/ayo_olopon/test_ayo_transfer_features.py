"""Focused tests for transfer-feature definitions."""

from absl.testing import absltest
import pyspiel

from Model.ayo_olopon import ayo_olopon  # noqa: F401
from Algorithms import ayo_transfer_features as transfer


def _state(board, current_player=0, captured=None, terminal=False):
  game = pyspiel.load_game("ayo_olopon", {"enable_cycle_reporting": True})
  state = game.new_initial_state()
  state.board = list(board)
  state.captured = list(captured or [0, 0])
  state._current_player = current_player
  state._game_over = terminal
  state._returns = ([1.0, -1.0] if captured and captured[0] > captured[1]
                    else [-1.0, 1.0] if captured and captured[1] > captured[0]
                    else [0.0, 0.0])
  state._positions_since_capture = {state._position_key()}
  return state


class TransferFeatureTest(absltest.TestCase):

  def test_proximity_symmetry_and_nonterminal_cases(self):
    self.assertEqual(transfer.winning_proximity(_state([4] * 12), 0), 0.0)
    early = transfer.winning_proximity(_state([4] * 12, captured=[2, 1]), 0)
    late = transfer.winning_proximity(_state([4] * 12, captured=[24, 23]), 0)
    self.assertGreater(early, 0)
    self.assertGreater(late, early)
    self.assertEqual(early, -transfer.winning_proximity(_state([4] * 12, captured=[2, 1]), 1))

  def test_exposure_no_capture_one_all_and_perspective(self):
    no_capture = _state([4] * 12)
    self.assertEqual(transfer.capture_option_exposure(no_capture, 0), 0.0)
    one = _state([1, 3, 0, 0, 0, 5, 2, 0, 0, 0, 0, 0])
    self.assertGreater(transfer.capture_option_exposure(one, 0), 0)
    self.assertEqual(
        transfer.capture_option_exposure(one, 1),
        -transfer.capture_option_exposure(one, 0),
    )
    all_capture = _state([0, 0, 0, 0, 0, 1, 3, 0, 0, 0, 0, 0])
    self.assertEqual(transfer.capture_option_exposure(all_capture, 0), 1.0)

  def test_exposure_relay_and_terminal_collection_are_child_based(self):
    relay = _state([3, 0, 0, 1, 0, 1, 0, 2, 0, 0, 3, 0])
    self.assertGreater(transfer.capture_option_exposure(relay, 0), 0)
    no_actions = _state([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    self.assertEqual(transfer.capture_option_exposure(no_actions, 0), 0.0)

  def test_odu_theoretical_examples_and_symmetry(self):
    none = _state([11] + [0] * 11)
    self.assertEqual(transfer.odu_potential(none, 0), 0)
    twelve = _state([12] + [0] * 11)
    self.assertEqual(transfer.odu_potential(twelve, 0), 1)
    self.assertEqual(transfer.odu_potential(twelve, 1), -1)
    twenty_four = _state([24] + [0] * 11)
    self.assertEqual(transfer.odu_potential(twenty_four, 0), 2)
    large = _state([12, 12, 12, 12] + [0] * 8)
    self.assertEqual(transfer.odu_potential(large, 0), 4)
    self.assertEqual(transfer.extract_transfer_features(large, 0)["odu"], 1.0)

  def test_nonmutation_terminal_override_and_evaluator(self):
    state = _state([2, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0], captured=[3, 1])
    before = (list(state.board), list(state.captured), state.current_player())
    value = transfer.evaluate_transfer_state(state, 0, transfer.TransferWeights(proximity=1, exposure=1, odu=1))
    self.assertTrue(value == value)
    self.assertEqual(before, (list(state.board), list(state.captured), state.current_player()))
    win = _state([0] * 12, captured=[25, 23], terminal=True)
    self.assertEqual(transfer.evaluate_transfer_state(win, 0), 10.0)
    self.assertEqual(transfer.evaluate_transfer_state(win, 1), -10.0)
    self.assertEqual(transfer.evaluate_transfer_state(_state([0] * 12, captured=[24, 24], terminal=True), 0), 0.0)


if __name__ == "__main__":
  absltest.main()
