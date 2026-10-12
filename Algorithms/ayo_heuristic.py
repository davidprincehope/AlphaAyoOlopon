"""Feature-based heuristic evaluation for Ayo Olopon.

The functions in this module treat the supplied OpenSpiel state as read-only.
Hypothetical actions are evaluated through ``state.child(action)``.
"""

from dataclasses import dataclass
from typing import Mapping


_PLAYERS = (0, 1)
_DEFAULT_TERMINAL_VALUE = 10.0
_CAPTURE_SCALE = 48.0
_MOBILITY_SCALE = 6.0
_SEED_SCALE = 48.0


def _validate_player(player: int) -> None:
  if player not in _PLAYERS:
    raise ValueError(f"player must be 0 or 1, got {player!r}")


def _opponent(player: int) -> int:
  _validate_player(player)
  return 1 - player


def capture_advantage(state, player: int) -> int:
  """Returns captured seeds for ``player`` minus the opponent's score."""
  _validate_player(player)
  opponent = 1 - player
  return int(state.captured[player] - state.captured[opponent])


def immediate_tactical_potential(state, player: int) -> int:
  """Returns the best immediate captured-score gain from ``player``'s view.

  The player to move is determined by the state. If the opponent is to move,
  the best opponent gain is negated. The score delta includes any terminal
  collection caused by the candidate action, matching the game implementation.
  """
  _validate_player(player)
  if state.is_terminal():
    return 0

  to_move = int(state.current_player())
  before = int(state.captured[to_move])
  gains = [
      int(state.child(action).captured[to_move] - before)
      for action in state.legal_actions()
  ]
  best_gain = max(gains, default=0)
  return best_gain if to_move == player else -best_gain


def mobility(state, player: int) -> int:
  """Returns signed legal-action count from ``player``'s perspective."""
  _validate_player(player)
  if state.is_terminal():
    return 0
  count = len(state.legal_actions())
  return count if int(state.current_player()) == player else -count


def seed_control(state, player: int) -> int:
  """Returns seeds in ``player``'s row minus seeds in the opponent's row."""
  _validate_player(player)
  houses = state.num_houses_per_player
  own_start = player * houses
  opponent_start = (1 - player) * houses
  own = sum(state.board[own_start : own_start + houses])
  opponent = sum(state.board[opponent_start : opponent_start + houses])
  return int(own - opponent)


def extract_features(state, player: int) -> dict[str, int]:
  """Returns all raw features without mutating ``state``."""
  _validate_player(player)
  return {
      "capture": capture_advantage(state, player),
      "tactical": immediate_tactical_potential(state, player),
      "mobility": mobility(state, player),
      "seed": seed_control(state, player),
  }


@dataclass(frozen=True)
class HeuristicWeights:
  """Weights for the normalized C/T/M/S feature combination."""

  capture: float = 1.0
  tactical: float = 1.0
  mobility: float = 1.0
  seed: float = 1.0

  def as_mapping(self) -> Mapping[str, float]:
    return {
        "capture": self.capture,
        "tactical": self.tactical,
        "mobility": self.mobility,
        "seed": self.seed,
    }


def normalize_weight_vector(weights) -> tuple[float, ...]:
  """Normalizes a non-zero weight vector by its largest absolute value."""
  values = tuple(float(value) for value in weights)
  if not values or all(value == 0.0 for value in values):
    raise ValueError("weight vector must contain a non-zero value")
  scale = max(abs(value) for value in values)
  return tuple(round(value / scale, 12) for value in values)


def equivalent_weight_vectors(first, second) -> bool:
  """Returns whether vectors differ only by a positive scalar factor."""
  first_values = tuple(float(value) for value in first)
  second_values = tuple(float(value) for value in second)
  if len(first_values) != len(second_values):
    return False
  if any(value < 0 for value in first_values + second_values):
    return False
  try:
    return normalize_weight_vector(first_values) == normalize_weight_vector(second_values)
  except ValueError:
    return False


def _terminal_value(state, player: int, terminal_value: float) -> float | None:
  if not state.is_terminal():
    return None
  returns = state.returns()
  if returns[player] > 0:
    return float(terminal_value)
  if returns[player] < 0:
    return -float(terminal_value)
  return 0.0


def evaluate_state(
    state,
    player: int,
    weights: HeuristicWeights | Mapping[str, float] | None = None,
    terminal_value: float = _DEFAULT_TERMINAL_VALUE,
) -> float:
  """Returns the normalized weighted heuristic from ``player``'s view."""
  _validate_player(player)
  feature_weights = (
      weights.as_mapping()
      if isinstance(weights, HeuristicWeights)
      else dict(weights) if weights is not None else HeuristicWeights().as_mapping()
  )
  maximum_positional_score = sum(abs(float(feature_weights.get(name, 0.0)))
                               for name in ("capture", "tactical", "mobility", "seed"))
  if terminal_value <= maximum_positional_score:
    raise ValueError(
        "terminal_value must exceed the maximum positional score for the active weights"
    )
  terminal = _terminal_value(state, player, terminal_value)
  if terminal is not None:
    return terminal

  raw = extract_features(state, player)
  normalized = {
      "capture": raw["capture"] / _CAPTURE_SCALE,
      "tactical": raw["tactical"] / _CAPTURE_SCALE,
      "mobility": raw["mobility"] / _MOBILITY_SCALE,
      "seed": raw["seed"] / _SEED_SCALE,
  }
  return float(sum(feature_weights.get(name, 0.0) * value
                   for name, value in normalized.items()))


def candidate_heuristic(name: str):
  """Returns one of the equal-weight candidate evaluator callables."""
  feature_names = {
      "H_C": ("capture",),
      "H_CT": ("capture", "tactical"),
      "H_CTM": ("capture", "tactical", "mobility"),
      "H_CTMS": ("capture", "tactical", "mobility", "seed"),
  }
  if name not in feature_names:
    raise ValueError(f"unknown candidate heuristic: {name!r}")
  selected = feature_names[name]

  def evaluate(state, player: int) -> float:
    weights = {feature: 1.0 for feature in selected}
    return evaluate_state(state, player, weights=weights)

  evaluate.__name__ = name
  return evaluate


H_C = candidate_heuristic("H_C")
H_CT = candidate_heuristic("H_CT")
H_CTM = candidate_heuristic("H_CTM")
H_CTMS = candidate_heuristic("H_CTMS")
