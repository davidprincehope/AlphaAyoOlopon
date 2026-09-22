"""Depth-limited alpha-beta minimax for the Ayo Olopon game."""

from open_spiel.python.algorithms import minimax


def evaluate_state(state, maximizing_player):
  """Returns a simple Ayo evaluation from ``maximizing_player``'s view.

  Captured seeds are weighted most strongly because they determine the final
  result. Remaining seeds on each player's row provide a small positional
  signal during depth-limited searches.
  """
  opponent = 1 - maximizing_player
  total_seeds = float(state.total_seeds)
  own_row = sum(state.board[
      maximizing_player * state.num_houses_per_player:
      (maximizing_player + 1) * state.num_houses_per_player
  ])
  opponent_row = sum(state.board[
      opponent * state.num_houses_per_player:
      (opponent + 1) * state.num_houses_per_player
  ])
  captured_difference = state.captured[maximizing_player] - state.captured[opponent]
  row_difference = own_row - opponent_row
  return captured_difference / total_seeds + 0.25 * row_difference / total_seeds


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
