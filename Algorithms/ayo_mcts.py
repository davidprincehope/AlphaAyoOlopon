"""Vanilla Monte-Carlo Tree Search for the Ayo Olopon game.

This module reuses OpenSpiel's general-purpose UCT implementation. Ayo is a
sequential, perfect-information, zero-sum game with terminal rewards, so it
is compatible with OpenSpiel's standard Python MCTS bot.
"""

import math

import numpy as np

from open_spiel.python.algorithms import mcts


UCT_C = math.sqrt(2.0)


def make_bot(
    game,
    simulations=1000,
    rollouts_per_leaf=1,
    uct_c=UCT_C,
    seed=None,
):
  """Creates a vanilla UCT bot for an already-loaded Ayo game.

  Args:
    game: A loaded ``ayo_olopon`` ``pyspiel.Game``.
    simulations: Number of tree simulations per move.
    rollouts_per_leaf: Number of random playouts for each evaluated leaf.
    uct_c: UCT exploration constant. ``sqrt(2)`` is the usual default.
    seed: Optional seed for reproducible tree search and rollouts.

  Returns:
    An OpenSpiel ``mcts.MCTSBot``.
  """
  if game.get_type().short_name != "ayo_olopon":
    raise ValueError(
        "make_bot expects the ayo_olopon game, got "
        f"{game.get_type().short_name!r}"
    )
  if simulations < 1:
    raise ValueError("simulations must be at least 1")
  if rollouts_per_leaf < 1:
    raise ValueError("rollouts_per_leaf must be at least 1")

  rng = np.random.RandomState(seed)
  evaluator = mcts.RandomRolloutEvaluator(
      n_rollouts=rollouts_per_leaf, random_state=rng
  )
  return mcts.MCTSBot(
      game=game,
      uct_c=uct_c,
      max_simulations=simulations,
      evaluator=evaluator,
      solve=False,
      random_state=rng,
  )


def play_game(game, simulations=1000, rollouts_per_leaf=1, seed=None):
  """Plays one Ayo game with identical vanilla MCTS bots on both sides."""
  bots = [
      make_bot(game, simulations, rollouts_per_leaf, seed=seed),
      make_bot(
          game,
          simulations,
          rollouts_per_leaf,
          seed=None if seed is None else seed + 1,
      ),
  ]
  state = game.new_initial_state()
  actions = []
  while not state.is_terminal():
    player = state.current_player()
    action = bots[player].step(state)
    actions.append(action)
    state.apply_action(action)
  return {"returns": state.returns(), "actions": actions, "state": state}
