"""Initial tests for the Ayo Olopon OpenSpiel game."""

import pyspiel
import numpy as np

from Model.ayo_olopon import ayo_olopon  # noqa: F401


def _state_with_position(board, current_player=0, captured=None):
    """Create a state from a compact board position for rule tests."""
    game = pyspiel.load_game("ayo_olopon", {"enable_cycle_reporting": True})
    state = game.new_initial_state()
    state.board = list(board)
    state.captured = list(captured or [0, 0])
    state.total_seeds = state.num_houses * game.num_seeds_per_house
    state._current_player = current_player
    state._positions_since_capture = {state._position_key()}
    return state


def test_ayo_game_is_registered():
    assert "ayo_olopon" in pyspiel.registered_names()
    print("Game of the Intellectuals is registered in OpenSpiel.") 


def test_initial_board():
    game = pyspiel.load_game("ayo_olopon")
    state = game.new_initial_state()

    assert state.current_player() == 0
    assert state.board == [4] * 12
    assert state.captured == [0, 0]
    assert state.legal_actions() == [0, 1, 2, 3, 4, 5]
    assert state.is_terminal() is False
    assert state.rewards() == [0, 0]
    assert state.returns() == [0, 0]

def test_player_0_first_action():
    game = pyspiel.load_game("ayo_olopon")
    state = game.new_initial_state()

    state.apply_action(0)

    assert state.board == [2, 7, 1, 6, 1, 6, 6, 6, 0, 1, 6, 6]
    assert state.captured == [0, 0]
    assert state.current_player() == 1

    assert state.legal_actions() == [0, 1, 3, 4, 5]
    assert state.captured == [0, 0]
    assert state.is_terminal() is False
    assert state.rewards() == [0, 0]
    assert state.returns() == [0, 0]    
    observation = np.asarray(state.observation_tensor())
    assert observation.shape == (14,)
    assert np.allclose(observation[:12], np.asarray(state.board) / 48)
    assert np.allclose(observation[12:], [0, 0])


def test_player_1_first_action():
    game = pyspiel.load_game("ayo_olopon")
    state = game.new_initial_state()

    state._current_player = 1
    state.apply_action(0)
    assert state.board == [6, 6, 0, 1, 6, 6, 2, 7, 1, 6, 1, 6]
    assert state.captured == [0, 0]
    assert state.current_player() == 0
    assert state.legal_actions() == [0, 1, 3, 4, 5]
    assert state.captured == [0, 0]
    assert state.is_terminal() is False
    assert state.rewards() == [0, 0]
    assert state.returns() == [0, 0]
    observation = np.asarray(state.observation_tensor())
    assert observation.shape == (14,)
    assert np.allclose(observation[:12], np.asarray(state.board) / 48)
    assert np.allclose(observation[12:], [0, 0])


def test_last_seed_in_empty_pit_ends_move_without_capture():
    state = _state_with_position([0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0])

    state.apply_action(1)

    assert state.board == [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0]
    assert state.captured == [0, 0]
    assert state.current_player() == 1


def test_landing_on_four_captures_and_ends_move():
    state = _state_with_position(
        [1, 3, 0, 0, 0, 5, 2, 0, 0, 0, 0, 0]
    )

    state.apply_action(0)

    assert state.board == [0, 0, 0, 0, 0, 5, 2, 0, 0, 0, 0, 0]
    assert state.captured == [4, 0]
    assert state.current_player() == 1


def test_intermediate_four_is_captured_by_row_owner_and_sowing_continues():
    state = _state_with_position(
        [0, 0, 0, 0, 0, 5, 3, 0, 0, 0, 0, 0]
    )

    state.apply_action(5)

    # The first seed lands in opponent pit 6, changing 3 to 4 while four
    # seeds remain in hand. Player 1 owns that row and receives the capture;
    # the remaining seeds continue to pits 7 through 10.
    assert state.board == [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0]
    assert state.captured == [0, 4]
    assert state.current_player() == 1


def test_non_terminal_landing_pit_is_picked_up_for_relay_sowing():
    state = _state_with_position(
        [2, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0]
    )

    state.apply_action(0)

    assert state.captured == [0, 0]
    assert state.board == [0, 2, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0]
    assert state.current_player() == 1


def test_relay_continues_across_multiple_laps_until_landing_on_four():
    state = _state_with_position(
        [3, 0, 0, 1, 0, 1, 0, 2, 0, 0, 3, 0]
    )

    state.apply_action(0)

    # The last seed lands in pit 3 (2 seeds), then pit 5 (2), then pit 7
    # (3), and finally pit 10 (4), which is captured.
    assert state.board == [0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0]
    assert state.captured == [4, 0]
    assert state.current_player() == 1


def test_relay_cycle_resolves_by_collecting_remaining_seeds():
    state = _state_with_position(
        [2, 1, 0, 2, 1, 0, 1, 0, 1, 3, 1, 0],
        current_player=1,
        captured=[16, 20],
    )

    state.apply_action(3)

    report = state.last_relay_report
    assert report["reason"] == "repeated_relay_state"
    assert report["player"] == 1
    assert report["action"] == 3
    assert report["source_house"] == 9
    assert report["cycle_length"] == 60
    assert len(report["relay_trace"]) == 60
    assert report["resolution"] == "collect_remaining_seeds_by_row"
    assert report["board_after_resolution"] == [0] * 12
    assert report["captured_after_resolution"] == [22, 26]
    assert report["winner"] == 1
    assert state.board == [0] * 12
    assert state.captured == [22, 26]
    assert state.is_terminal()


def test_empty_opponent_row_requires_a_feeding_move():
    state = _state_with_position(
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    )

    assert state.legal_actions() == []


if __name__ == "__main__":
    
    test_ayo_game_is_registered()
    test_initial_board()
    test_player_0_first_action()
    test_player_1_first_action()
    test_last_seed_in_empty_pit_ends_move_without_capture()
    test_landing_on_four_captures_and_ends_move()
    test_non_terminal_landing_pit_is_picked_up_for_relay_sowing()
    test_relay_continues_across_multiple_laps_until_landing_on_four()
    test_empty_opponent_row_requires_a_feeding_move()
