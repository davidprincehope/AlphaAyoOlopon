"""Rule-compatible transfer features for the Ayo Olopon implementation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from Algorithms import ayo_heuristic


def winning_proximity(state, player: int) -> float:
  """Nonlinear proximity to the implemented 25-seed winning threshold."""
  if state.is_terminal():
    return 0.0
  opponent = 1 - player
  own = float(state.captured[player])
  other = float(state.captured[opponent])
  return (own / 25.0) ** 2 - (other / 25.0) ** 2


def _child_capture_deltas(state) -> tuple[int, list[tuple[int, object]]]:
  """Returns the moving player and each legal action's child state."""
  if state.is_terminal():
    return 0, []
  to_move = int(state.current_player())
  before = int(state.captured[to_move])
  children = [
      (int(child.captured[to_move] - before), child)
      for action in state.legal_actions()
      for child in (state.child(action),)
  ]
  return to_move, children


def capture_option_exposure(state, player: int) -> float:
  """Fraction of legal actions that capture, signed for ``player``."""
  if state.is_terminal():
    return 0.0
  to_move, children = _child_capture_deltas(state)
  ratio = sum(delta > 0 for delta, _ in children) / len(children) if children else 0.0
  return ratio if to_move == player else -ratio


def odu_potential(state, player: int) -> int:
  """Rule-neutral full-lap reserve differential between player rows."""
  houses = state.num_houses_per_player
  own_start = player * houses
  other_start = (1 - player) * houses
  own = sum(int(seeds) // 12 for seeds in state.board[own_start:own_start + houses])
  other = sum(int(seeds) // 12 for seeds in state.board[other_start:other_start + houses])
  return own - other


def extract_transfer_features(state, player: int) -> dict[str, float]:
  """Returns P, X and normalized O without mutating ``state``."""
  if player not in (0, 1):
    raise ValueError(f"player must be 0 or 1, got {player!r}")
  return {
      "proximity": winning_proximity(state, player),
      "exposure": capture_option_exposure(state, player),
      "odu": odu_potential(state, player) / 4.0,
  }


@dataclass(frozen=True)
class TransferWeights:
  """Weights for baseline C/T/M plus one or more transferred features."""

  proximity: float = 0.0
  exposure: float = 0.0
  odu: float = 0.0


def evaluate_transfer_state(
    state,
    player: int,
    weights: TransferWeights | Mapping[str, float] | None = None,
    terminal_value: float | None = None,
) -> float:
  """Evaluates the baseline plus transferred features from ``player``'s view."""
  transfer_weights = (
      {"proximity": weights.proximity, "exposure": weights.exposure, "odu": weights.odu}
      if isinstance(weights, TransferWeights)
      else {"proximity": 0.0, "exposure": 0.0, "odu": 0.0}
      if weights is None else dict(weights)
  )
  active = {"capture": 4.0, "tactical": 2.0, "mobility": 0.5, **transfer_weights}
  max_positional = sum(abs(float(value)) for value in active.values())
  W = max(10.0, 1.0 + max_positional) if terminal_value is None else float(terminal_value)
  if W <= max_positional:
    raise ValueError("terminal value must exceed the active positional magnitude")
  if state.is_terminal():
    returns = state.returns()
    return W if returns[player] > 0 else -W if returns[player] < 0 else 0.0

  baseline = ayo_heuristic.evaluate_state(
      state, player,
      weights={"capture": 4.0, "tactical": 2.0, "mobility": 0.5},
      terminal_value=W,
  )
  transfer = extract_transfer_features(state, player)
  return baseline + sum(transfer_weights.get(name, 0.0) * value
                        for name, value in transfer.items())


def candidate_weights(feature: str, weight: float) -> TransferWeights:
  if feature not in ("proximity", "exposure", "odu"):
    raise ValueError(feature)
  return TransferWeights(**{feature: float(weight)})
