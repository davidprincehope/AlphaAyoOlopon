"""Capture-only greed resolves full moves without an evaluator or lookahead."""
from types import SimpleNamespace

import pyspiel
import pytest

from Algorithms.ayo_capture_greedy import capture_greedy_action
from experiments.agent_benchmark.agents import make_agent
from Model.ayo_olopon import ayo_olopon  # noqa: F401


def test_own_capture_gain_ignores_opponent_score_and_terminal_utility():
    children = {
        4: SimpleNamespace(captured=[12, 40]),
        2: SimpleNamespace(captured=[8, 0]),
        1: SimpleNamespace(captured=[12, 8]),
    }
    state = SimpleNamespace(current_player=lambda: 0, captured=[4, 0],
                            legal_actions=lambda: [4, 2, 1], child=children.__getitem__)
    assert capture_greedy_action(state) == 1
    state.current_player = lambda: 1
    assert capture_greedy_action(state) == 4
    state.legal_actions = lambda: []
    with pytest.raises(ValueError, match="legal actions"):
        capture_greedy_action(state)


def test_full_relay_action_is_scored_without_mutating_parent():
    game = pyspiel.load_game("ayo_olopon")
    state = game.new_initial_state()
    before = state.serialize()
    player = state.current_player()
    gains = {action: state.child(action).captured[player] - state.captured[player]
             for action in state.legal_actions()}
    expected = min(action for action, gain in gains.items() if gain == max(gains.values()))
    for seed in (0, 23):
        agent = make_agent(game, seed, {"name": "GREEDY_CAPTURE", "params": {}})
        assert agent.step(state) == expected
    assert state.serialize() == before
    with pytest.raises(ValueError, match="no parameters"):
        make_agent(game, 0, {"name": "GREEDY_CAPTURE", "params": {"depth": 2}})
