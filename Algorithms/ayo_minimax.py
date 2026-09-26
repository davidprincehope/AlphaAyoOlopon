"""Depth-limited alpha-beta minimax for the Ayo Olopon game."""

from open_spiel.python.algorithms import minimax
from Algorithms import H_star_ayo



evaluate_state = H_star_ayo.evaluate_state


class AyoMinimaxBot:
  """A small depth-limited minimax bot for Ayo."""

  def __init__(self, game, maximum_depth=4, value_function=evaluate_state):
    if game.get_type().short_name != "ayo_olopon":
      raise ValueError(
          "AyoMinimaxBot expects the ayo_olopon game, got "
          f"{game.get_type().short_name!r}"
      )
    if maximum_depth < 1:
      raise ValueError("maximum_depth must be at least 1")
    self.game = game
    self.maximum_depth = maximum_depth
    self.value_function = value_function

  def search(self, state):
    """Returns ``(value, action)`` for the current player."""
    maximizing_player = state.current_player()
    if self.value_function is None:
      value_function = None
    else:
      value_function = lambda child: self.value_function(child, maximizing_player)
    return minimax.alpha_beta_search(
        self.game,
        state=state,
        value_function=value_function,
        maximum_depth=self.maximum_depth,
        maximizing_player_id=maximizing_player,
    )

  def step(self, state):
    """Returns the best action for the current player."""
    _, action = self.search(state)
    return action


def make_bot(game, maximum_depth=4, value_function=evaluate_state):
  """Creates an Ayo minimax bot."""
  return AyoMinimaxBot(game, maximum_depth, value_function)


def play_game(game, maximum_depth=4):
  """Plays one game with depth-limited minimax on both sides."""
  bots = [make_bot(game, maximum_depth), make_bot(game, maximum_depth)]
  state = game.new_initial_state()
  actions = []
  while not state.is_terminal():
    player = state.current_player()
    action = bots[player].step(state)
    actions.append(action)
    state.apply_action(action)
  return {"returns": state.returns(), "actions": actions, "state": state}
