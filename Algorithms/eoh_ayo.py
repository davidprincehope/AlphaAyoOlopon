"""Fixed one-ply greedy interface for EoH Ayo candidates.

Candidate code receives only the immutable StateView below. The game state and
action selection remain owned by this module.
"""
from dataclasses import dataclass
import math
from types import MappingProxyType


@dataclass(frozen=True)
class StateView:
  board: tuple[int, ...]
  captured: tuple[int, int]
  current_player: int
  num_houses_per_player: int
  total_seeds: int
  legal_actions: tuple[int, ...]
  terminal: bool
  returns: tuple[float, float]


NONTERMINAL_LIMIT = 10.0
TERMINAL_UTILITY = 11.0


def state_view(state):
  """Build a detached, immutable view; repetition history is intentionally hidden."""
  terminal = bool(state.is_terminal())
  returns = tuple(float(x) for x in state.returns()) if terminal else (0.0, 0.0)
  return StateView(tuple(map(int, state.board)), tuple(map(int, state.captured)),
                   int(state.current_player()) if not terminal else -1,
                   int(state.num_houses_per_player), int(state.total_seeds),
                   tuple(map(int, state.legal_actions())) if not terminal else (),
                   terminal, returns)


def greedy_action(state, evaluator):
  """Choose max successor value from the original actor's perspective.

  Legal action order is preserved for ties (Ayo actions are ascending).
  The returned tuple is (action, trace), useful for reproducible logs.
  """
  player = int(state.current_player())
  rows = []
  for action in state.legal_actions():
    child = state.child(action)
    if child.is_terminal():
      utility = float(child.returns()[player])
      score = math.copysign(TERMINAL_UTILITY, utility) if utility else 0.0
    else:
      raw = float(evaluator(state_view(child), player))
      if not math.isfinite(raw) or abs(raw) > NONTERMINAL_LIMIT:
        raise ValueError("candidate score must be finite and within [-10, 10]")
      score = raw
    rows.append({"action": int(action), "score": score,
                 "terminal": bool(child.is_terminal()), "successor": state_view(child)})
  if not rows:
    raise ValueError("cannot choose an action from a terminal/no-legal-action state")
  best = max(row["score"] for row in rows)
  return next(row["action"] for row in rows if row["score"] == best), rows


def fixture_evaluator(view, player_id):
  """Known-good development fixture: captured lead plus own-row seed control."""
  opp = 1 - player_id
  n = view.num_houses_per_player
  own = sum(view.board[player_id*n:(player_id+1)*n])
  other = sum(view.board[opp*n:(opp+1)*n])
  return max(-10.0, min(10.0, (view.captured[player_id]-view.captured[opp])/48.0
                         + (own-other)/96.0))
