"""Compare the frozen handcrafted and EoH Ayo heuristics.

The one-ply mode is greedy over actual successors. The two-ply mode adds one
opponent reply and backs up the minimum heuristic value before maximizing the
root action. This is a depth-two heuristic policy, not the repository's
general Minimax agent and not part of EoH candidate selection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
  sys.path.insert(0, str(ROOT))

import pyspiel

from Algorithms import ayo_heuristic
from Algorithms import eoh_ayo
from Algorithms import H_star_ayo
from Model.ayo_olopon import ayo_olopon  # noqa: F401 - registers game
from experiments.heuristic import run_experiment as frozen_protocol


OUTPUT_DIR = Path(__file__).resolve().parent
OPENING_PATH = ROOT / "experiments" / "heuristic" / "openings" / "opening_positions.json"
OPENING_SHA256 = "971dba4079f260d733418fed6f008130b56778fd947584f8292669b2c92b2735"
MAX_ACTIONS = 300
PLAYERS = ("handcrafted", "eoh")
HANDCRAFTED_WEIGHTS = {"capture": 4.0, "tactical": 2.0, "mobility": 0.5}


def _hash_file(path: Path) -> str:
  return hashlib.sha256(path.read_bytes()).hexdigest()


def _terminal_score(state, perspective: int) -> float:
  """Map exact terminal return to a value strictly beyond nonterminal range."""
  result = float(state.returns()[perspective])
  return math.copysign(11.0, result) if result else 0.0


def evaluate(state, perspective: int, heuristic: str) -> float:
  """Evaluate a nonterminal state from a fixed player's perspective."""
  if state.is_terminal():
    return _terminal_score(state, perspective)
  if heuristic == "handcrafted":
    return ayo_heuristic.evaluate_state(
        state,
        perspective,
        weights=HANDCRAFTED_WEIGHTS,
        terminal_value=10.0,
    )
  if heuristic == "eoh":
    return float(H_star_ayo.evaluate_state(eoh_ayo.state_view(state), perspective))
  raise ValueError(f"unknown heuristic {heuristic!r}")


def choose_action(state, heuristic: str, depth: int) -> tuple[int, dict]:
  """Return a deterministic action and scores for all legal root actions.

  `depth=1` evaluates the immediate successor. `depth=2` evaluates each
  opponent reply and assumes the opponent chooses the reply least favorable
  to the root player. Ties preserve OpenSpiel's legal action order.
  """
  if depth not in (1, 2):
    raise ValueError("depth must be 1 (greedy) or 2 (one opponent reply)")
  if state.is_terminal():
    raise ValueError("cannot select an action from a terminal state")

  actor = int(state.current_player())
  rows = []
  for action in state.legal_actions():
    successor = state.child(action)
    reply_values = []
    if depth == 1 or successor.is_terminal():
      score = evaluate(successor, actor, heuristic)
    else:
      replies = successor.legal_actions()
      if not replies:
        score = evaluate(successor, actor, heuristic)
      else:
        for reply in replies:
          grandchild = successor.child(reply)
          reply_values.append({
              "action": int(reply),
              "score": evaluate(grandchild, actor, heuristic),
              "terminal": bool(grandchild.is_terminal()),
          })
        score = min(item["score"] for item in reply_values)
    rows.append({
        "action": int(action),
        "score": float(score),
        "terminal_successor": bool(successor.is_terminal()),
        "reply_values": reply_values,
    })

  best_score = max(row["score"] for row in rows)
  chosen = next(row["action"] for row in rows if row["score"] == best_score)
  return chosen, {"actor": actor, "depth": depth, "actions": rows}


def play_game(opening: dict, assignment: tuple[str, str], depth: int,
              match_id: str) -> dict:
  game = pyspiel.load_game("ayo_olopon", {"enable_cycle_reporting": True})
  state = frozen_protocol.state_from_snapshot(game, opening)
  policies = {0: assignment[0], 1: assignment[1]}
  decisions = {name: 0 for name in PLAYERS}
  policy_seconds = {name: 0.0 for name in PLAYERS}
  actions = []

  while not state.is_terminal() and len(actions) < MAX_ACTIONS:
    player = int(state.current_player())
    name = policies[player]
    started = time.perf_counter()
    action, trace = choose_action(state, name, depth)
    policy_seconds[name] += time.perf_counter() - started
    decisions[name] += 1
    actions.append({"player": player, "policy": name, "action": action,
                    "decision": trace})
    state.apply_action(action)

  truncated = not state.is_terminal()
  if truncated:
    difference = int(state.captured[0] - state.captured[1])
    returns = [1.0, -1.0] if difference > 0 else [-1.0, 1.0] if difference < 0 else [0.0, 0.0]
    reason = "action_limit"
  else:
    returns = [float(value) for value in state.returns()]
    report = state.last_relay_report
    if report and report.get("reason") in {"repeated_relay_state", "relay_lap_limit_exceeded"}:
      reason = "repetition"
    elif returns == [0.0, 0.0]:
      reason = "terminal_draw"
    else:
      reason = "terminal_win"

  policy_returns = {name: returns[assignment.index(name)] for name in PLAYERS}
  result = {
      "match_id": match_id,
      "opening_id": int(opening["opening_id"]),
      "opening_hash": opening["state_hash"],
      "depth": depth,
      "player_0_policy": assignment[0],
      "player_1_policy": assignment[1],
      "policy_returns": policy_returns,
      "policy_outcome": {
          name: "win" if value > 0 else "loss" if value < 0 else "draw"
          for name, value in policy_returns.items()
      },
      "termination_reason": reason,
      "truncated": truncated,
      "game_length": len(actions),
      "final_captured": [int(value) for value in state.captured],
      "decisions": decisions,
      "policy_seconds": policy_seconds,
      "actions": actions,
  }
  return result


def _new_stats():
  return {name: {"wins": 0, "draws": 0, "losses": 0, "score_sum": 0.0,
                 "games": 0, "decision_count": 0, "policy_seconds": 0.0}
          for name in PLAYERS}


def summarize(matches: list[dict]) -> dict:
  summaries = {}
  for depth in (1, 2):
    subset = [match for match in matches if match["depth"] == depth]
    stats = _new_stats()
    by_seat = {name: {"0": {"wins": 0, "draws": 0, "losses": 0},
                      "1": {"wins": 0, "draws": 0, "losses": 0}}
               for name in PLAYERS}
    for match in subset:
      for name in PLAYERS:
        outcome = match["policy_outcome"][name]
        outcome_key = {"win": "wins", "draw": "draws", "loss": "losses"}[outcome]
        stats[name][outcome_key] += 1
        stats[name]["score_sum"] += match["policy_returns"][name] * 0.5 + 0.5
        stats[name]["games"] += 1
        stats[name]["decision_count"] += match["decisions"][name]
        stats[name]["policy_seconds"] += match["policy_seconds"][name]
        seat = "0" if match["player_0_policy"] == name else "1"
        by_seat[name][seat][outcome_key] += 1
    for name in PLAYERS:
      stats[name]["score_rate"] = stats[name]["score_sum"] / stats[name]["games"] if stats[name]["games"] else None
    summaries[str(depth)] = {
        "label": "greedy_successor" if depth == 1 else "one_opponent_reply",
        "games": len(subset),
        "termination_reasons": {
            reason: sum(match["termination_reason"] == reason for match in subset)
            for reason in ("terminal_win", "terminal_draw", "repetition", "action_limit")
        },
        "policies": stats,
        "by_policy_seat": by_seat,
    }
  return summaries


def run(opening_path: Path = OPENING_PATH, output_dir: Path = OUTPUT_DIR,
        match_limit: int | None = None) -> dict:
  if _hash_file(opening_path) != OPENING_SHA256:
    raise ValueError("frozen opening file hash mismatch")
  openings = [item for item in frozen_protocol.load_frozen_openings(opening_path)
              if item["split"] == "development"]
  matches = []
  for depth in (1, 2):
    for opening in openings:
      for seat_assignment, assignment in enumerate((PLAYERS, tuple(reversed(PLAYERS)))):
        match_id = f"depth{depth}-open{opening['opening_id']:03d}-assign{seat_assignment}"
        matches.append(play_game(opening, assignment, depth, match_id))
        if match_limit is not None and len(matches) >= match_limit:
          break
      if match_limit is not None and len(matches) >= match_limit:
        break
    if match_limit is not None and len(matches) >= match_limit:
      break

  output_dir.mkdir(parents=True, exist_ok=True)
  matches_path = output_dir / "matches.jsonl"
  with matches_path.open("w", encoding="utf-8", newline="\n") as stream:
    for match in matches:
      stream.write(json.dumps(match, sort_keys=True, separators=(",", ":")) + "\n")
  results = {
      "experiment": "handcrafted_vs_eoh_ayo",
      "status": "pilot" if match_limit is not None else "completed",
      "opening_file_sha256": OPENING_SHA256,
      "split_used": "development_only",
      "selection_and_verification_positions_opened": False,
      "handcrafted_policy": "H_CTM weights capture=4, tactical=2, mobility=0.5; terminal=+/-10",
      "eoh_policy": "H_star_ayo.py; candidate g2-m2-05",
      "depths": {"1": "greedy immediate successor", "2": "one opponent reply; max root action of min reply evaluator"},
      "tie_break": "first action in OpenSpiel legal-action order",
      "game_length_limit": MAX_ACTIONS,
      "action_limit_score": "captured-seed differential; equal differential is draw",
      "randomness": "none; both policies and all tie breaks are deterministic",
      "games": len(matches),
      "match_log": str(matches_path.relative_to(ROOT)),
      "summary": summarize(matches),
  }
  (output_dir / "results.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
  return results


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--pilot-matches", type=int, help="run the first N matches only")
  parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
  args = parser.parse_args()
  result = run(output_dir=args.output_dir, match_limit=args.pilot_matches)
  print(json.dumps({"status": result["status"], "games": result["games"],
                    "summary": result["summary"]}, indent=2))


if __name__ == "__main__":
  main()
