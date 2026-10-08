"""Queue plain MCTS only after the neural checkpoint comparison succeeds."""
from datetime import datetime, timezone
import json

import modal
from experiments.alpha_zero import modal_evaluate as base
from experiments.alpha_zero.modal_common import validate_name

app = modal.App("alpha-ayo-vanilla-mcts-benchmark")


@app.function(image=base.image, cpu=base.CPU, memory=base.MEMORY_MIB,
              volumes={str(base.RESULTS_ROOT): base.results_volume},
              timeout=24 * 60 * 60, retries=0, min_containers=0, buffer_containers=0,
              max_containers=1, single_use_containers=True)
def baseline_remote(evaluation_name: str, after_evaluation: str, after_call: str):
    validate_name(evaluation_name)
    validate_name(after_evaluation)
    root = base.RESULTS_ROOT / evaluation_name
    root.mkdir(parents=True, exist_ok=False)
    queue = {"status": "waiting", "after_evaluation": after_evaluation, "after_call": after_call,
             "created_at_utc": datetime.now(timezone.utc).isoformat(), "gpu": None}
    def save():
        (root / "queue.manifest.json").write_text(json.dumps(queue, indent=2) + "\n")
        base.results_volume.commit()
    save()
    print(json.dumps({"event": "waiting_for_alphazero", **queue}), flush=True)
    try:
        result = modal.FunctionCall.from_id(after_call).get(timeout=20 * 60 * 60)
        if result["status"] != "completed":
            raise RuntimeError("The AlphaZero comparison did not complete successfully")
        base.results_volume.reload()
        previous = json.loads((base.RESULTS_ROOT / after_evaluation / "comparison.manifest.json").read_text())
        if previous["status"] != "completed" or previous["completed_budgets"] != [64, 128]:
            raise RuntimeError("The saved AlphaZero comparison is incomplete")
        from Algorithms.alpha_zero import game as adapter  # Registers the same finite-horizon game; no model.
        from experiments.checkpoint_strength.opening_dataset import load_dataset
        from experiments.checkpoint_strength.mcts_baseline import run_baseline
        import pyspiel
        prior_report = json.loads((base.RESULTS_ROOT / after_evaluation / "128/results.jsonl").read_text().splitlines()[0])
        prior_manifest = json.loads((base.RESULTS_ROOT / after_evaluation / "128/results.manifest.json").read_text())
        config = previous["configuration_64"]
        game = pyspiel.load_game(prior_manifest["game"])
        dataset = load_dataset(config["openings"], game)
        if (dataset.content_sha256 != previous["opening_content_sha256"]
                or dataset.content_sha256 != prior_report["opening_dataset"]["content_sha256"]
                or config["games"] != 2 * len(dataset.openings)
                or config["opponent"] != {"name": "GREEDY_HSTAR", "params": {}}):
            raise ValueError("Plain MCTS must reuse the complete identical benchmark and opponent")
        queue.update(status="running")
        save()
        def progress(event):
            if event["event"] != "game_completed" or event["completed_games"] % 20 == 0:
                base.results_volume.commit()
        status = run_baseline(game, dataset, root / "benchmark", seed=config["seed"],
                              uct_c=prior_manifest["uct_c"], on_progress=progress)
        queue.update(status="completed", completed_games=status["completed_games"])
        save()
        return queue
    except BaseException as exc:
        queue.update(status="failed", error=repr(exc))
        save()
        raise
    finally:
        base.results_volume.commit()


@app.local_entrypoint()
def main(evaluation_name: str, after_evaluation: str, after_call: str):
    validate_name(evaluation_name)
    validate_name(after_evaluation)
    call = baseline_remote.spawn(evaluation_name, after_evaluation, after_call)
    print(json.dumps({"event": "baseline_queued", "evaluation_name": evaluation_name,
                      "function_call": call.object_id, "after_call": after_call,
                      "budgets": [64, 128], "positions": 200, "games_per_budget": 400,
                      "total_games": 800, "gpu": None}), flush=True)
