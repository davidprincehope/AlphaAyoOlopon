"""Benchmark random play against the one-ply greedy H* policy.

The benchmark starts every game from the standard initial position and balances
the seats: half the games put H* first and half put random first.  A fixed
master seed makes the random policy reproducible.  The game log is JSONL so a
long run can be inspected while it is executing and resumed/validated without
loading all games into memory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
  sys.path.insert(0, str(ROOT))

import pyspiel

from Algorithms import H_star_ayo
from Algorithms import eoh_ayo
from Model.ayo_olopon import ayo_olopon  # noqa: F401 - registers the game


POLICIES = ("GREEDY_HSTAR", "RAND")
DEFAULT_SEED = 20260923
DEFAULT_GAMES = 10_000
DEFAULT_MAX_ACTIONS = 300


def _policy_action(state, policy: str, rng: random.Random) -> int:
  if policy == "RAND":
    return int(rng.choice(list(state.legal_actions())))
  if policy == "GREEDY_HSTAR":
    action, _ = eoh_ayo.greedy_action(
        state,
        lambda view, player: H_star_ayo.evaluate_state(view, player),
    )
    return int(action)
  raise ValueError(f"unknown policy: {policy!r}")


def play_game(game_id: int, assignment: tuple[str, str], seed: int,
              max_actions: int = DEFAULT_MAX_ACTIONS) -> dict:
  """Play one game and return a compact, auditable record."""
  game = pyspiel.load_game("ayo_olopon")
  state = game.new_initial_state()
  rng = random.Random(seed)
  actions = []
  policy_decisions = Counter()
  started = time.perf_counter()

  while not state.is_terminal() and len(actions) < max_actions:
    player = int(state.current_player())
    policy = assignment[player]
    action = _policy_action(state, policy, rng)
    if action not in state.legal_actions():
      raise RuntimeError(f"policy {policy} selected illegal action {action}")
    state.apply_action(action)
    actions.append(int(action))
    policy_decisions[policy] += 1

  if state.is_terminal():
    returns = [float(value) for value in state.returns()]
    report = state.last_relay_report
    if report and report.get("reason") in {
        "repeated_relay_state", "relay_lap_limit_exceeded"
    }:
      termination = "repetition"
    elif returns == [0.0, 0.0]:
      termination = "terminal_draw"
    else:
      termination = "terminal_win"
    truncated = False
  else:
    # The cap is a safety valve, not a natural draw.  Score the position so
    # the benchmark remains total, while reporting this outcome separately.
    difference = int(state.captured[0] - state.captured[1])
    returns = ([1.0, -1.0] if difference > 0 else
               [-1.0, 1.0] if difference < 0 else [0.0, 0.0])
    termination = "action_limit"
    truncated = True

  policy_returns = {
      policy: returns[seat] for seat, policy in enumerate(assignment)
  }
  return {
      "game_id": game_id,
      "seed": seed,
      "player_0_policy": assignment[0],
      "player_1_policy": assignment[1],
      "policy_returns": policy_returns,
      "policy_outcome": {
          policy: ("win" if value > 0 else "loss" if value < 0 else "draw")
          for policy, value in policy_returns.items()
      },
      "returns_by_player": returns,
      "termination_reason": termination,
      "truncated": truncated,
      "game_length": len(actions),
      "final_captured": [int(value) for value in state.captured],
      "policy_decisions": dict(policy_decisions),
      "elapsed_seconds": time.perf_counter() - started,
  }


def _empty_policy_stats() -> dict:
  return {policy: {
      "games": 0, "wins": 0, "draws": 0, "losses": 0,
      "score_sum": 0.0, "natural_draws": 0, "repetitions": 0,
      "action_limits": 0, "game_lengths": [], "seats": {"0": 0, "1": 0},
  } for policy in POLICIES}


def summarize(records: list[dict]) -> dict:
  stats = _empty_policy_stats()
  terminations = Counter(record["termination_reason"] for record in records)
  for record in records:
    for seat, policy in enumerate(
        (record["player_0_policy"], record["player_1_policy"])):
      outcome = record["policy_outcome"][policy]
      item = stats[policy]
      item["games"] += 1
      item[{"win": "wins", "draw": "draws", "loss": "losses"}[outcome]] += 1
      item["score_sum"] += {"win": 1.0, "draw": 0.5, "loss": 0.0}[outcome]
      item["natural_draws"] += int(record["termination_reason"] == "terminal_draw")
      item["repetitions"] += int(record["termination_reason"] == "repetition")
      item["action_limits"] += int(record["termination_reason"] == "action_limit")
      item["game_lengths"].append(record["game_length"])
      item["seats"][str(seat)] += 1
  for item in stats.values():
    item["score_rate"] = item["score_sum"] / item["games"] if item["games"] else None
    item["win_rate"] = item["wins"] / item["games"] if item["games"] else None
    item["average_game_length"] = (
        sum(item["game_lengths"]) / len(item["game_lengths"])
        if item["game_lengths"] else None)
    del item["game_lengths"]
  return {"games": len(records), "termination_reasons": dict(terminations),
          "policies": stats}


def run(games: int = DEFAULT_GAMES, seed: int = DEFAULT_SEED,
        max_actions: int = DEFAULT_MAX_ACTIONS, output_dir: Path | None = None) -> dict:
  if games < 1:
    raise ValueError("games must be positive")
  if games % 2:
    raise ValueError("games must be even so seats can be balanced")
  output_dir = (output_dir or Path(__file__).resolve().parent / "results").resolve()
  output_dir.mkdir(parents=True, exist_ok=True)
  log_path = output_dir / f"random_vs_greedy_hstar_{games}.jsonl"
  records = []
  assignments = (("GREEDY_HSTAR", "RAND"), ("RAND", "GREEDY_HSTAR"))
  with log_path.open("w", encoding="utf-8", newline="\n") as stream:
    for game_id in range(games):
      pair_id = game_id // 2
      assignment = assignments[game_id % 2]
      # The pair shares a seed, making seat comparisons easier to audit.
      game_seed = seed + pair_id
      record = play_game(game_id, assignment, game_seed, max_actions)
      records.append(record)
      stream.write(json.dumps(record, sort_keys=True) + "\n")

  manifest = {
      "experiment": "random_vs_greedy_hstar",
      "status": "completed",
      "created_at_utc": datetime.now(timezone.utc).isoformat(),
      "games": games,
      "master_seed": seed,
      "max_actions": max_actions,
      "starting_position": "standard_initial_state",
      "seat_balance": {"GREEDY_HSTAR": games // 2, "RAND": games // 2},
      "greedy_policy": "one-ply max successor value using Algorithms/H_star_ayo.py",
      "tie_break": "first action in OpenSpiel legal-action order",
      "action_limit_score": "captured-seed differential; reported as action_limit, not natural_draw",
      "log_file": str(log_path.relative_to(ROOT)),
      "log_sha256": hashlib.sha256(log_path.read_bytes()).hexdigest(),
      "summary": summarize(records),
  }
  result_path = output_dir / f"random_vs_greedy_hstar_{games}.json"
  result_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
  manifest["result_file"] = str(result_path.relative_to(ROOT))
  return manifest


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--games", type=int, default=DEFAULT_GAMES)
  parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
  parser.add_argument("--max-actions", type=int, default=DEFAULT_MAX_ACTIONS)
  parser.add_argument("--output-dir", type=Path)
  args = parser.parse_args()
  result = run(args.games, args.seed, args.max_actions, args.output_dir)
  print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
  main()
