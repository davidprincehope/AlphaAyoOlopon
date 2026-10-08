"""Run the same fixed-opening experiment at 64, then 128 simulations on one GPU."""
from dataclasses import asdict, replace
from datetime import datetime, timezone
import json
from pathlib import Path

import modal
from experiments.alpha_zero import modal_evaluate as base
from experiments.alpha_zero.modal_common import ROOT, validate_name


app = modal.App("alpha-ayo-checkpoint-budget-comparison")


def reusable_reports(previous, run, config, dataset_hash):
    """Reuse only whole checkpoints from an identical interrupted experiment."""
    from experiments.alpha_zero.offline_evaluate import EvaluationConfig, source_hash
    manifest = json.loads((previous / "comparison.manifest.json").read_text())
    evaluation = json.loads((previous / "64/results.manifest.json").read_text())
    if (EvaluationConfig(**manifest["configuration_64"]) != config
            or manifest["opening_content_sha256"] != dataset_hash
            or manifest["completed_budgets"] or manifest["current_simulations"] != 64
            or evaluation["run_manifest_sha256"] != source_hash(run / "manifest.json")):
        raise ValueError("Recovery requires the identical interrupted 64-simulation experiment")
    for name, expected in evaluation["code_sha256"].items():
        if source_hash(ROOT / name) != expected:
            raise ValueError(f"Evaluation source changed since the interrupted run: {name}")
    reports = [json.loads(line) for line in (previous / "64/results.jsonl").read_text().splitlines()]
    steps = [row["learner_round"] for row in reports]
    if len(set(steps)) != len(steps) or not set(steps).issubset(config.checkpoints):
        raise ValueError("Invalid checkpoint selection in saved reports")
    for row in reports:
        path = run / "inference-checkpoints" / Path(row["checkpoint_source"]).name
        if (row["status"] != "completed" or row["games"] != config.games
                or len(row["matches"]) != config.games
                or row["opening_dataset"]["content_sha256"] != dataset_hash
                or source_hash(path) != row["checkpoint_sha256"]):
            raise ValueError("Incomplete or incompatible saved checkpoint report")
    return reports


def comparison_rows(low, high):
    high_by_step = {row["learner_round"]: row for row in high}
    if len(high_by_step) != len(high) or {r["learner_round"] for r in low} != set(high_by_step) or len(low) != len(high):
        raise ValueError("Budget comparison must contain the same unique checkpoints")
    rows = []
    for a in low:
        b = high_by_step[a["learner_round"]]
        if a["opening_dataset"] != b["opening_dataset"]:
            raise ValueError("Budget comparison must use identical opening datasets and order")
        rows.append({"learner_round": a["learner_round"],
                     "win_rate_64": a["win_rate"], "win_rate_128": b["win_rate"],
                     "win_rate_change_percentage_points": 100 * (b["win_rate"] - a["win_rate"]),
                     "score_rate_64": a["score_rate"], "score_rate_128": b["score_rate"],
                     "score_rate_change_percentage_points": 100 * (b["score_rate"] - a["score_rate"]),
                     "learner_by_seat_64": a["by_seat"], "learner_by_seat_128": b["by_seat"],
                     "first_player_advantage_64": a["first_player_advantage"],
                     "first_player_advantage_128": b["first_player_advantage"]})
    return rows


@app.function(image=base.image, gpu=base.GPU, cpu=base.CPU, memory=base.MEMORY_MIB,
              volumes={str(base.CHECKPOINT_ROOT): base.source_volume.with_mount_options(read_only=True),
                       str(base.RESULTS_ROOT): base.results_volume},
              timeout=24 * 60 * 60, retries=0, min_containers=0, buffer_containers=0,
              max_containers=1, single_use_containers=True)
def compare_remote(run_name: str, evaluation_name: str, configuration_64: dict, configuration_128: dict,
                   resume_from: str = ""):
    import jax
    from experiments.alpha_zero.offline_evaluate import EvaluationConfig, run_evaluation
    from experiments.checkpoint_strength.opening_dataset import load_dataset
    from Algorithms.alpha_zero.config import Settings
    import pyspiel

    validate_name(run_name)
    validate_name(evaluation_name)
    low, high = EvaluationConfig(**configuration_64), EvaluationConfig(**configuration_128)
    if low.simulations != 64 or high != replace(low, simulations=128) or low.openings is None:
        raise ValueError("Configurations must differ only in 64 versus 128 simulations and use a saved dataset")
    if not any(device.platform == "gpu" for device in jax.devices()):
        raise RuntimeError("The budget comparison requires a JAX GPU backend")
    run, root = base.CHECKPOINT_ROOT / run_name, base.RESULTS_ROOT / evaluation_name
    # Refuse reuse before any experiment begins; historical outputs stay intact.
    root.mkdir(parents=True, exist_ok=False)
    source = json.loads((run / "manifest.json").read_text())
    dataset = load_dataset(low.openings, pyspiel.load_game(Settings(**source["settings"]).game_string))
    reused = reusable_reports(base.RESULTS_ROOT / validate_name(resume_from), run, low,
                              dataset.content_sha256) if resume_from else []
    status = {"status": "running", "run_name": run_name, "evaluation_name": evaluation_name,
              "started_at_utc": datetime.now(timezone.utc).isoformat(), "budget_order": [64, 128],
              "configuration_64": asdict(low), "configuration_128": asdict(high),
              "opening_content_sha256": dataset.content_sha256,
              "current_simulations": 64, "completed_budgets": [], "completed_checkpoints": 0,
              "completed_games": 0, "gpu": base.GPU}
    if resume_from:
        status.update(resume_from=resume_from, reused_checkpoints=[r["learner_round"] for r in reused],
                      completed_checkpoints=len(reused), completed_games=sum(r["games"] for r in reused))
    def save():
        path = root / "comparison.manifest.json"
        temporary = root / "comparison.manifest.json.tmp"
        temporary.write_text(json.dumps(status, indent=2) + "\n")
        temporary.replace(path)
    save()
    base.results_volume.commit()
    print(json.dumps({"event": "comparison_started", **status}), flush=True)
    try:
        for config in (low, high):
            budget = config.simulations
            status.update(current_simulations=budget)
            save()
            print(json.dumps({"event": "budget_started", "simulations": budget}), flush=True)
            def checkpoint_done(report):
                if report["opening_dataset"]["content_sha256"] != dataset.content_sha256:
                    raise RuntimeError("Opening dataset changed during the comparison")
                status["completed_checkpoints"] += 1
                status["completed_games"] += report["games"]
                save()
                base.results_volume.commit()
                print(json.dumps({"event": "seat_advantage", "simulations": budget,
                                  "learner_round": report["learner_round"],
                                  "learner_by_seat": report["by_seat"],
                                  "first_player_advantage": report["first_player_advantage"]}), flush=True)
            if budget == 64 and resume_from:
                remaining = [step for step in low.checkpoints if step not in status["reused_checkpoints"]]
                new_reports = []
                if remaining:
                    recovery_output = root / "64-recovery/results.jsonl"
                    run_evaluation(run, replace(low, checkpoints=remaining), recovery_output,
                                   on_checkpoint=checkpoint_done)
                    new_reports = [json.loads(line) for line in recovery_output.read_text().splitlines()]
                merged = sorted(reused + new_reports, key=lambda r: low.checkpoints.index(r["learner_round"]))
                comparison_rows(merged, merged)  # Verify the complete selection has no duplicates.
                if [r["learner_round"] for r in merged] != low.checkpoints:
                    raise RuntimeError("Recovery did not finish every requested checkpoint")
                target = root / "64"
                target.mkdir()
                (target / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in merged))
                (target / "results.games.jsonl").write_text("".join(
                    json.dumps({"learner_round": r["learner_round"], **m}) + "\n"
                    for r in merged for m in r["matches"]))
                (target / "results.manifest.json").write_text(json.dumps({
                    "status": "completed", "config": asdict(low), "resume_from": resume_from,
                    "reused_checkpoints": status["reused_checkpoints"],
                    "recovery_artifacts": "../64-recovery", "completed_checkpoints": len(merged),
                    "completed_games": sum(r["games"] for r in merged),
                    "opening_content_sha256": dataset.content_sha256}, indent=2) + "\n")
            else:
                run_evaluation(run, config, root / str(budget) / "results.jsonl",
                               on_checkpoint=checkpoint_done)
            status["completed_budgets"].append(budget)
            save()
            base.results_volume.commit()
        reports = [[json.loads(line) for line in (root / str(b) / "results.jsonl").read_text().splitlines()]
                   for b in (64, 128)]
        (root / "budget_comparison.json").write_text(json.dumps(comparison_rows(*reports), indent=2) + "\n")
        status.update(status="completed", current_simulations=None,
                      finished_at_utc=datetime.now(timezone.utc).isoformat())
        save()
        print(json.dumps({"event": "comparison_completed", **status}), flush=True)
        return status
    except BaseException as exc:
        status.update(status="failed", error=repr(exc))
        save()
        raise
    finally:
        base.results_volume.commit()
        print(json.dumps({"event": "comparison_exiting", "results_committed": True,
                          "single_use_container": True, "artifacts": str(root)}), flush=True)


@app.local_entrypoint()
def main(run_name: str = "a100-fresh-64-200-20261003", evaluation_name: str = "",
         background: bool = False, resume_from: str = ""):
    from experiments.alpha_zero.offline_evaluate import load_config
    validate_name(run_name)
    evaluation_name = validate_name(evaluation_name or datetime.now(timezone.utc).strftime("mixed-v2-budgets-%Y%m%dT%H%M%SZ"))
    low = load_config(ROOT / "experiments/checkpoint_strength/configs/progression_mixed_v2_64.json")
    high = load_config(ROOT / "experiments/checkpoint_strength/configs/progression_mixed_v2_128.json")
    if resume_from:
        validate_name(resume_from)
    args = (run_name, evaluation_name, asdict(low), asdict(high), resume_from)
    print(json.dumps({"event": "comparison_submission", "run_name": run_name,
                      "evaluation_name": evaluation_name, "budget_order": [64, 128],
                      "resume_from": resume_from,
                      "configuration_64": asdict(low), "configuration_128": asdict(high)}), flush=True)
    if background:
        call = compare_remote.spawn(*args)
        print(f"Submitted sequential comparison; function call: {call.object_id}", flush=True)
    else:
        print(json.dumps(compare_remote.remote(*args)), flush=True)
