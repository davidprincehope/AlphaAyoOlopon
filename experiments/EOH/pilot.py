"""Reproducible local throughput pilot; makes no LLM calls."""
from __future__ import annotations
import hashlib, json, random, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import pyspiel
from Model.ayo_olopon import ayo_olopon  # registers game
from Algorithms.eoh_ayo import fixture_evaluator, greedy_action, state_view
from candidate_validator import validate
from heuristic.run_experiment import load_frozen_openings, state_from_snapshot

OPENINGS = ROOT / "experiments/heuristic/openings/opening_positions.json"
MANIFEST = ROOT / "experiments/heuristic/raw_results_corrected/corrected_experiment_manifest.json"

def main():
  raw = OPENINGS.read_bytes(); digest = hashlib.sha256(raw).hexdigest()
  frozen = load_frozen_openings(OPENINGS, MANIFEST)
  dev = [p for p in frozen if p["split"] == "development"]
  game = pyspiel.load_game("ayo_olopon", {"enable_cycle_reporting": True})
  state = state_from_snapshot(game, dev[0])
  worked_actor = int(state.current_player())
  views = [(state_view(state.child(action)), player)
           for action in state.legal_actions() for player in (0, 1)]
  candidate = '''def evaluate_state(state_view, player_id):\n    opponent = 1 - player_id\n    n = state_view.num_houses_per_player\n    own = sum(state_view.board[player_id*n:(player_id+1)*n])\n    other = sum(state_view.board[opponent*n:(opponent+1)*n])\n    return max(-10.0, min(10.0, state_view.captured[player_id] - state_view.captured[opponent] + 0.1*(own-other)))\n'''
  validation = validate(candidate, views)
  if not validation["ok"]: raise RuntimeError(validation)
  example_action, example_rows = greedy_action(state, fixture_evaluator)
  worked_example = [{"action": row["action"],
                     "successor_board": list(row["successor"].board),
                     "successor_captured": list(row["successor"].captured),
                     "successor_current_player": row["successor"].current_player,
                     "capture_lead_for_actor": row["successor"].captured[worked_actor] - row["successor"].captured[1-worked_actor],
                     "row_seed_delta_for_actor": sum(row["successor"].board[worked_actor*6:(worked_actor+1)*6]) - sum(row["successor"].board[(1-worked_actor)*6:(2-worked_actor)*6]),
                     "score": row["score"], "terminal": row["terminal"]}
                    for row in example_rows]
  # Fixed workload: score every legal successor repeatedly to estimate eval throughput.
  start = time.perf_counter(); calls = 0
  for _ in range(100):
    for view, player in views:
      fixture_evaluator(view, player); calls += 1
  eval_seconds = time.perf_counter() - start
  # Two paired-seat games on one development opening against seeded random legal play.
  games = []
  for candidate_seat in (0, 1):
    state = state_from_snapshot(game, dev[0]); rng = random.Random(20260922 + candidate_seat)
    action_log = []; start_game = time.perf_counter()
    while not state.is_terminal() and len(action_log) < 300:
      player = int(state.current_player())
      if player == candidate_seat:
        action, _ = greedy_action(state, fixture_evaluator)
      else:
        action = rng.choice(state.legal_actions())
      state.apply_action(action); action_log.append({"player": player, "action": int(action)})
    ret = state.returns() if state.is_terminal() else [0.0, 0.0]
    games.append({"opening_id": dev[0]["opening_id"], "opening_hash": dev[0]["state_hash"],
                  "candidate_seat": candidate_seat, "seed": 20260922+candidate_seat,
                  "outcome": "win" if ret[candidate_seat] > 0 else "loss" if ret[candidate_seat] < 0 else "draw",
                  "termination_reason": "terminal" if state.is_terminal() else "action_limit",
                  "actions": action_log, "seconds": time.perf_counter()-start_game})
  output = {"pilot": "local_fixture_not_llm_generated", "opening_file_sha256": digest,
            "opening_count": len(frozen), "development_count": len(dev), "selection_count": 30,
            "candidate_validation": validation, "eval_calls": calls, "eval_seconds": eval_seconds,
            "eval_calls_per_second": calls/eval_seconds,
            "worked_example": {"opening_id": dev[0]["opening_id"],
                               "actor": worked_actor,
                               "chosen_action": example_action, "successors": worked_example},
            "paired_games": games}
  out = Path(__file__).resolve().parent / "pilot_results.json"
  out.write_text(json.dumps(output, indent=2))
  print(json.dumps({k:v for k,v in output.items() if k != "paired_games"}, indent=2))

if __name__ == "__main__": main()
