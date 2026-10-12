"""Seat-swapped initial-position comparison at search depths 1, 2, and 3.

Depth counts individual Ayo actions. At each search node, the root policy
maximizes its frozen evaluator and the opponent minimizes it. The search is a
small, fixed-depth policy wrapper; it does not tune heuristics or call the
repository's separate Minimax agent.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
import time
from pathlib import Path

import pyspiel

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
  sys.path.insert(0, str(ROOT))

from experiments.handcrafted_vs_eoh import compare as shared
from Model.ayo_olopon import ayo_olopon  # noqa: F401 - registers game


OUTPUT_DIR = Path(__file__).resolve().parent
MAX_ACTIONS = 300
SEARCH_DEPTHS = (1, 2, 3)


def _terminal_score(state, perspective: int) -> float:
  result = float(state.returns()[perspective])
  return math.copysign(11.0, result) if result else 0.0


def _search(state, perspective: int, heuristic: str, plies_left: int,
            nodes: list[int]) -> float:
  """Minimax backup for one fixed evaluator, counting plies from this node."""
  if state.is_terminal():
    nodes[0] += 1
    return _terminal_score(state, perspective)
  if plies_left == 0:
    nodes[0] += 1
    return shared.evaluate(state, perspective, heuristic)

  legal = state.legal_actions()
  if not legal:
    nodes[0] += 1
    return shared.evaluate(state, perspective, heuristic)

  values = [
      _search(state.child(action), perspective, heuristic, plies_left - 1, nodes)
      for action in legal
  ]
  return max(values) if int(state.current_player()) == perspective else min(values)


def choose_action(state, heuristic: str, depth: int) -> tuple[int, dict]:
  """Choose a legal root action with a deterministic depth-limited search."""
  if depth not in SEARCH_DEPTHS:
    raise ValueError(f"depth must be one of {SEARCH_DEPTHS}")
  if state.is_terminal():
    raise ValueError("cannot select an action from a terminal state")

  perspective = int(state.current_player())
  nodes = [0]
  action_values = []
  for action in state.legal_actions():
    child = state.child(action)
    score = _search(child, perspective, heuristic, depth - 1, nodes)
    action_values.append({"action": int(action), "score": float(score)})
  best = max(item["score"] for item in action_values)
  chosen = next(item["action"] for item in action_values if item["score"] == best)
  return chosen, {
      "actor": perspective,
      "depth_plies": depth,
      "root_action_values": action_values,
      "leaf_or_terminal_nodes": nodes[0],
  }


def _play(depth: int, assignment: tuple[str, str], match_id: str) -> dict:
  game = pyspiel.load_game("ayo_olopon", {"enable_cycle_reporting": True})
  state = game.new_initial_state()
  starting_state = {
      "board": [int(value) for value in state.board],
      "captured": [int(value) for value in state.captured],
      "current_player": int(state.current_player()),
  }
  starting_hash = hashlib.sha256(
      json.dumps(starting_state, sort_keys=True, separators=(",", ":")).encode()
  ).hexdigest()
  policy_for_player = {0: assignment[0], 1: assignment[1]}
  policy_time = {name: 0.0 for name in shared.PLAYERS}
  decisions = {name: 0 for name in shared.PLAYERS}
  actions = []

  while not state.is_terminal() and len(actions) < MAX_ACTIONS:
    player = int(state.current_player())
    policy = policy_for_player[player]
    started = time.perf_counter()
    action, search = choose_action(state, policy, depth)
    policy_time[policy] += time.perf_counter() - started
    decisions[policy] += 1
    actions.append({"player": player, "policy": policy, "action": action,
                    "search": search})
    state.apply_action(action)

  truncated = not state.is_terminal()
  if truncated:
    differential = int(state.captured[0] - state.captured[1])
    returns = [1.0, -1.0] if differential > 0 else [-1.0, 1.0] if differential < 0 else [0.0, 0.0]
    termination_reason = "action_limit"
  else:
    returns = [float(value) for value in state.returns()]
    report = state.last_relay_report
    if report and report.get("reason") in {"repeated_relay_state", "relay_lap_limit_exceeded"}:
      termination_reason = "repetition"
    elif returns == [0.0, 0.0]:
      termination_reason = "terminal_draw"
    else:
      termination_reason = "terminal_win"

  policy_returns = {name: returns[assignment.index(name)] for name in shared.PLAYERS}
  return {
      "match_id": match_id,
      "depth_plies": depth,
      "starting_position": starting_state,
      "starting_state_sha256": starting_hash,
      "player_0_policy": assignment[0],
      "player_1_policy": assignment[1],
      "policy_returns": policy_returns,
      "policy_outcome": {
          name: "win" if value > 0 else "loss" if value < 0 else "draw"
          for name, value in policy_returns.items()
      },
      "termination_reason": termination_reason,
      "truncated": truncated,
      "game_length": len(actions),
      "final_captured": [int(value) for value in state.captured],
      "decisions": decisions,
      "policy_seconds": policy_time,
      "actions": actions,
  }


def _summarize(matches: list[dict]) -> dict:
  result = {}
  for depth in SEARCH_DEPTHS:
    games = [match for match in matches if match["depth_plies"] == depth]
    policies = {}
    for name in shared.PLAYERS:
      outcomes = [match["policy_outcome"][name] for match in games]
      policies[name] = {
          "games": len(games),
          "wins": outcomes.count("win"),
          "draws": outcomes.count("draw"),
          "losses": outcomes.count("loss"),
          "score_rate": sum(match["policy_returns"][name] * 0.5 + 0.5 for match in games) / len(games),
          "decision_count": sum(match["decisions"][name] for match in games),
          "policy_seconds": sum(match["policy_seconds"][name] for match in games),
      }
    seat_results = {}
    for name in shared.PLAYERS:
      seat_results[name] = {}
      for seat in (0, 1):
        subset = [match for match in games
                  if (match["player_0_policy"] == name) == (seat == 0)]
        outcomes = [match["policy_outcome"][name] for match in subset]
        seat_results[name][str(seat)] = {
            "games": len(subset),
            "wins": outcomes.count("win"),
            "draws": outcomes.count("draw"),
            "losses": outcomes.count("loss"),
        }
    result[str(depth)] = {
        "games": len(games),
        "termination_reasons": {
            reason: sum(match["termination_reason"] == reason for match in games)
            for reason in ("terminal_win", "terminal_draw", "repetition", "action_limit")
        },
        "policies": policies,
        "by_policy_seat": seat_results,
    }
  return result


def run(output_dir: Path = OUTPUT_DIR) -> dict:
  matches = []
  for depth in SEARCH_DEPTHS:
    for seat_index, assignment in enumerate((shared.PLAYERS, tuple(reversed(shared.PLAYERS)))):
      match_id = f"initial-depth{depth}-seat{seat_index}"
      matches.append(_play(depth, assignment, match_id))

  output_dir.mkdir(parents=True, exist_ok=True)
  log_path = output_dir / "initial_matches.jsonl"
  with log_path.open("w", encoding="utf-8", newline="\n") as stream:
    for match in matches:
      stream.write(json.dumps(match, sort_keys=True, separators=(",", ":")) + "\n")
  result = {
      "experiment": "handcrafted_vs_eoh_initial_position",
      "status": "completed",
      "starting_position": "standard Ayo initial state: 4 seeds in every pit; player 0 moves first",
      "starting_state_sha256": matches[0]["starting_state_sha256"],
      "handcrafted_policy": "H_CTM weights capture=4, tactical=2, mobility=0.5; terminal=+/-10",
      "eoh_policy": "H_star_ayo.py; candidate g2-m2-05",
      "search_definition": "depth counts actions; maximize on the root policy turn, minimize on opponent turns, evaluate heuristic at depth cutoff",
      "depths_plies": list(SEARCH_DEPTHS),
      "seat_swaps_per_depth": 1,
      "games_per_depth": 2,
      "game_length_limit": MAX_ACTIONS,
      "action_limit_score": "captured-seed differential; equal differential is draw",
      "randomness": "none; deterministic policies and legal-action-order tie breaks",
      "games": len(matches),
      "match_log": log_path.name,
      "summary": _summarize(matches),
  }
  (output_dir / "initial_results.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
  return result


if __name__ == "__main__":
  print(json.dumps(run(), indent=2))
