"""Initial tests for the Ayo Olopon OpenSpiel game."""

import pyspiel

from Model.ayo_olopon import ayo_olopon  # noqa: F401


def _state_with_position(board, current_player=0, captured=None):
    """Create a state from a compact board position for rule tests."""
    game = pyspiel.load_game("ayo_olopon")
    state = game.new_initial_state()
    state.board = list(board)
    state.captured = list(captured or [0, 0])
    state.total_seeds = sum(state.board) + sum(state.captured)
    state._current_player = current_player
    state._positions_since_capture = {state._position_key()}
    return state


def test_ayo_game_is_registered():
    assert "ayo_olopon" in pyspiel.registered_names()


def test_initial_board():
    game = pyspiel.load_game("ayo_olopon")
    state = game.new_initial_state()

    assert state.current_player() == 0
    assert state.board == [4] * 12
    assert state.captured == [0, 0]
    assert state.legal_actions() == [0, 1, 2, 3, 4, 5]


def test_sowing_skips_source_house_and_switches_player():
    state = _state_with_position([0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0])

    state.apply_action(2)

    assert state.board == [0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0]
    assert state.current_player() == 1


def test_capture_collects_consecutive_two_and_three_seed_houses():
    state = _state_with_position(
        [0, 0, 8, 0, 0, 1, 1, 1, 1, 1, 2, 3]
    )

    state.apply_action(2)

    assert state.captured == [15, 3]
    assert state.is_terminal()


def test_grand_slam_sows_but_does_not_capture():
    state = _state_with_position(
        [1, 1, 1, 1, 1, 0, 0, 0, 8, 0, 0, 1],
        current_player=1,
    )

    state.apply_action(2)

    assert state.captured == [0, 0]
    assert state.board == [2, 2, 2, 2, 2, 0, 0, 0, 0, 1, 1, 2]


def test_empty_opponent_row_requires_a_feeding_move():
    state = _state_with_position(
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    )

    assert state.legal_actions() == []
