"""Player-to-move targets and player-indexed MCTS boundary/backup."""

import numpy as np
import pytest

from Algorithms.alpha_zero.runtime import use_repository_open_spiel

use_repository_open_spiel()
import pyspiel  # noqa: E402
from open_spiel.python.algorithms import mcts  # noqa: E402
from open_spiel.python.algorithms.alpha_zero import alpha_zero, evaluator  # noqa: E402
from Algorithms.alpha_zero import game as ayo_game  # noqa: E402,F401


@pytest.mark.parametrize("returns,expected", [
    ([1, -1], [1, -1, 1, -1]),
    ([-1, 1], [-1, 1, -1, 1]),
    ([0, 0], [0, 0, 0, 0]),
])
def test_self_play_targets_use_recorded_player(returns, expected):
    players = [0, 1, 0, 1]
    targets = [alpha_zero.player_to_move_value_target(returns, p) for p in players]
    assert targets == expected


class ToyState:
    def __init__(self, tree, path=()):
        self.tree = tree
        self.path = path

    def clone(self):
        return ToyState(self.tree, self.path)

    def current_player(self):
        return self.tree[self.path][0]

    def legal_actions(self, player=None):
        return list(self.tree[self.path][1])

    def legal_actions_mask(self):
        return [int(a in self.legal_actions()) for a in range(6)]

    def observation_tensor(self):
        marker = sum((action + 1) * 6**i for i, action in enumerate(self.path))
        return [float(marker)] + [0.0] * 14

    def apply_action(self, action):
        self.path += (action,)

    def is_terminal(self):
        return False

    def is_chance_node(self):
        return False


class MockNetwork:
    def __init__(self, values):
        self.values = values

    def inference(self, observation, mask):
        marker = int(observation[0])
        legal = np.asarray(mask, dtype=bool)
        policy = legal.astype(float) / legal.sum() if legal.any() else np.zeros(6)
        return self.values[marker], policy


def make_search(tree, values, simulations):
    game = pyspiel.load_game(ayo_game.GAME_NAME)
    neural = evaluator.AlphaZeroEvaluator(game, MockNetwork(values))
    bot = mcts.MCTSBot(
        game, 1.5, simulations, neural, solve=False,
        random_state=np.random.RandomState(7),
        child_selection_fn=mcts.SearchNode.puct_value,
    )
    return neural, bot.mcts_search(ToyState(tree))


def test_player_one_leaf_value_stays_positive_until_mcts_boundary():
    tree = {(): (0, [0]), (0,): (1, [])}
    neural, _ = make_search(tree, {0: 0.0, 1: 0.6}, 2)
    leaf = ToyState(tree, (0,))
    assert neural._inference(leaf)[0] == 0.6
    np.testing.assert_allclose(neural.evaluate(leaf), [-0.6, 0.6])


def test_two_action_q_prefers_player_zero_favorable_branch():
    tree = {(): (0, [0, 1]), (0,): (1, []), (1,): (1, [])}
    _, root = make_search(tree, {0: 0.0, 1: 0.8, 2: -0.3}, 3)
    children = {node.action: node for node in root.children}
    assert children[0].explore_count == children[1].explore_count == 1
    assert children[0].player == children[1].player == 0
    assert children[0].total_reward == pytest.approx(-0.8)
    assert children[1].total_reward == pytest.approx(0.3)
    assert children[1].puct_value(root.explore_count, 0) > children[0].puct_value(root.explore_count, 0)


def test_multilevel_backup_tracks_decision_player():
    tree = {(): (0, [0]), (0,): (1, [0]), (0, 0): (0, [])}
    # All three leaf evaluations describe the same outcome: +0.5 for P0.
    _, root = make_search(tree, {0: 0.5, 1: -0.5, 7: 0.5}, 3)
    child = root.children[0]
    grandchild = child.children[0]
    assert (root.player, child.player, grandchild.player) == (0, 0, 1)
    assert root.total_reward / root.explore_count == pytest.approx(0.5)
    assert child.total_reward / child.explore_count == pytest.approx(0.5)
    assert grandchild.total_reward / grandchild.explore_count == pytest.approx(-0.5)


def test_ayo_nonterminal_moves_always_switch_player():
    game = pyspiel.load_game(ayo_game.GAME_NAME)
    state = game.new_initial_state()
    for _ in range(30):
        if state.is_terminal():
            break
        player = state.current_player()
        state.apply_action(state.legal_actions()[0])
        if not state.is_terminal():
            assert state.current_player() == 1 - player
