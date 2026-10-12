"""Frozen one-ply greedy gameplay fitness for Ayo EoH candidates."""
from __future__ import annotations

import hashlib
import json
import random
import sys
import time
import builtins
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "experiments"), str(Path(__file__).resolve().parent)]
import pyspiel
from Model.ayo_olopon import ayo_olopon  # register game
from Algorithms.eoh_ayo import greedy_action, state_view
from candidate_validator import validate
from heuristic.run_experiment import load_frozen_openings, state_from_snapshot

OPENINGS = ROOT / "experiments/heuristic/openings/opening_positions.json"
MANIFEST = ROOT / "experiments/heuristic/raw_results_corrected/corrected_experiment_manifest.json"
PANEL_WEIGHTS = [0.18] * 5 + [0.10]
GAME_SEED = 20260922
MAX_ACTIONS = 300


def compile_candidate(source: str):
  allowed = {name: getattr(builtins, name) for name in
             ("float", "int", "min", "max", "sum", "abs", "len")}
  namespace = {"__builtins__": allowed}
  exec(compile(source, "<eoh_candidate>", "exec"), namespace, namespace)
  return namespace["evaluate_state"]


def snapshot_views(positions, *, successor=False):
  game = pyspiel.load_game("ayo_olopon", {"enable_cycle_reporting": True})
  out = []
  for pos in positions:
    state = state_from_snapshot(game, pos)
    for action in state.legal_actions() if successor else [None]:
      base = state.child(action) if successor else state
      actor = int(state.current_player())
      out.append((state_view(base), actor))
  return out


def run_match(position, candidate_id, candidate_fn, opponent_id, opponent_fn, candidate_seat, game_seed):
  game = pyspiel.load_game("ayo_olopon", {"enable_cycle_reporting": True})
  state = state_from_snapshot(game, position)
  rng = random.Random(game_seed)
  actions = []
  started = time.perf_counter()
  while not state.is_terminal() and len(actions) < MAX_ACTIONS:
    player = int(state.current_player())
    policy = candidate_fn if player == candidate_seat else opponent_fn
    if policy is None:
      action = rng.choice(state.legal_actions())
    else:
      action, _ = greedy_action(state, policy)
    state.apply_action(action)
    actions.append({"player": player, "action": int(action)})

  if state.is_terminal():
    returns = state.returns()
    report = state.last_relay_report
    reason = ("repetition" if report and report.get("reason") in
              {"repeated_relay_state", "relay_lap_limit_exceeded"} else
              "terminal_draw" if returns == [0.0, 0.0] else "terminal_win")
    truncated = False
  else:
    diff = state.captured[0] - state.captured[1]
    returns = [1.0, -1.0] if diff > 0 else [-1.0, 1.0] if diff < 0 else [0.0, 0.0]
    reason, truncated = "action_limit", True
  value = float(returns[candidate_seat])
  outcome = "win" if value > 0 else "loss" if value < 0 else "draw"
  return {
      "match_id": f"{candidate_id}-o{position['opening_id']:03d}-s{candidate_seat}-{opponent_id}",
      "candidate_id": candidate_id, "opponent_id": opponent_id,
      "opening_id": position["opening_id"], "opening_hash": position["state_hash"],
      "candidate_seat": candidate_seat, "seed": game_seed,
      "actions": actions, "outcome": outcome, "score": (value + 1.0) / 2.0,
      "termination_reason": reason, "truncated": truncated,
      "game_length": len(actions), "elapsed_seconds": time.perf_counter() - started,
  }


def evaluate(candidate, panel, positions, match_path, *, split):
  """Evaluate once on every position x seat x fixed panel opponent."""
  src = candidate["source"]
  fn = compile_candidate(src)
  raw_views = snapshot_views(positions[:min(4, len(positions))], successor=True)
  fixture_views = [(view, player) for view, _ in raw_views for player in (0, 1)]
  validation = validate(src, fixture_views)
  if not validation["ok"]:
    raise ValueError(f"candidate {candidate['candidate_id']} failed validation: {validation}")
  totals = {opp["candidate_id"]: {k: 0 for k in ("wins", "draws", "losses", "truncations", "games", "score_sum", "seconds", "plies")} for opp in panel}
  totals["RND"] = {k: 0 for k in ("wins", "draws", "losses", "truncations", "games", "score_sum", "seconds", "plies")}
  with Path(match_path).open("a", encoding="utf-8") as out:
    for opening in positions:
      for seat in (0, 1):
        for opponent in [*panel, {"candidate_id": "RND", "evaluator": None}]:
          seed = GAME_SEED + opening["opening_id"] * 1000 + seat * 100 + (int(opponent["candidate_id"].split("-")[-1]) if opponent["candidate_id"].startswith("anchor-") else 999)
          result = run_match(opening, candidate["candidate_id"], fn,
                             opponent["candidate_id"], opponent["evaluator"], seat, seed)
          result["split"] = split
          out.write(json.dumps(result, separators=(",", ":")) + "\n")
          t = totals[opponent["candidate_id"]]
          t["games"] += 1
          t["score_sum"] += result["score"]
          t["seconds"] += result["elapsed_seconds"]
          t["plies"] += result["game_length"]
          t["truncations"] += int(result["truncated"])
          t[{"win": "wins", "draw": "draws", "loss": "losses"}[result["outcome"]]] += 1
  scores = {opp: row["score_sum"] / row["games"] for opp, row in totals.items()}
  weighted = sum(w * scores[opp["candidate_id"]] for w, opp in zip(PANEL_WEIGHTS[:5], panel)) + PANEL_WEIGHTS[-1] * scores["RND"]
  return {"candidate_id": candidate["candidate_id"], "by_opponent": totals,
          "score_rate": weighted, "fitness": 1.0 - weighted,
          "panel_weights": {**{o["candidate_id"]: 0.18 for o in panel}, "RND": 0.10},
          "validation": validation, "split": split}


def main():
  openings = load_frozen_openings(OPENINGS, MANIFEST)
  dev = [o for o in openings if o["split"] == "development"]
  print(json.dumps({"opening_sha256": hashlib.sha256(OPENINGS.read_bytes()).hexdigest(),
                    "dev_positions": len(dev), "selection_openings_count_only": 30,
                    "panel_weight_sum": sum(PANEL_WEIGHTS)}, indent=2))

if __name__ == "__main__":
  main()
