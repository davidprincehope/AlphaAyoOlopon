"""Finite offline progression evaluation on Modal, with read-only checkpoints."""
import json
import os
from pathlib import Path

import modal
from experiments.alpha_zero.modal_common import ROOT, build_image, gpu_request, validate_name

GPU = gpu_request("ALPHAZERO_EVAL_GPU", default="L4", allowed=("L4", "A100-40GB", "A100-80GB"))
CPU = float(os.environ.get("ALPHAZERO_EVAL_CPU", "2"))
MEMORY_MIB = int(os.environ.get("ALPHAZERO_EVAL_MEMORY_MIB", "8192"))
if CPU <= 0 or MEMORY_MIB <= 0:
    raise ValueError("Evaluation CPU and memory requests must be positive")
app = modal.App("alpha-ayo-alphazero-evaluation")
source_volume = modal.Volume.from_name("alpha-ayo-alphazero-runs")
results_volume = modal.Volume.from_name("alpha-ayo-alphazero-evaluations", create_if_missing=True)
image = build_image({"ALPHAZERO_EVAL_GPU": GPU, "ALPHAZERO_EVAL_CPU": str(CPU),
                     "ALPHAZERO_EVAL_MEMORY_MIB": str(MEMORY_MIB)},
                    extra_dirs=("experiments/checkpoint_strength",))
CHECKPOINT_ROOT = Path("/checkpoints")
RESULTS_ROOT = Path("/evaluations")


@app.function(image=image, gpu=GPU, cpu=CPU, memory=MEMORY_MIB,
              volumes={str(CHECKPOINT_ROOT): source_volume.with_mount_options(read_only=True),
                       str(RESULTS_ROOT): results_volume},
              timeout=24 * 60 * 60, retries=0, min_containers=0, buffer_containers=0,
              max_containers=1, single_use_containers=True)
def evaluate_remote(run_name: str, evaluation_name: str, configuration: dict):
    from dataclasses import asdict
    import jax
    from experiments.alpha_zero.offline_evaluate import EvaluationConfig, run_evaluation

    validate_name(run_name)
    validate_name(evaluation_name)
    config = EvaluationConfig(**configuration)
    devices = jax.devices()
    if not any(device.platform == "gpu" for device in devices):
        raise RuntimeError(f"Modal evaluation requires a JAX GPU backend; found {devices}")
    print(json.dumps({"event": "modal_evaluation_started", "run": run_name,
                      "evaluation": evaluation_name, "gpu_request": GPU, "cpu_request": CPU,
                      "memory_mib": MEMORY_MIB, "devices": [str(d) for d in devices],
                      "source_mount": "read_only", "config": asdict(config)}), flush=True)
    output = RESULTS_ROOT / evaluation_name / "results.jsonl"
    try:
        result = run_evaluation(CHECKPOINT_ROOT / run_name, config, output,
                                on_checkpoint=lambda report: results_volume.commit())
        return result
    finally:
        results_volume.commit()
        print(json.dumps({"event": "modal_evaluation_exiting", "artifacts": str(output.parent),
                          "results_committed": True, "single_use_container": True}), flush=True)


@app.local_entrypoint()
def main(run_name: str = "a100-fresh-64-200-20261003", evaluation_name: str = "",
         config: str = "", checkpoints: str = "", games: int = 0, simulations: int = 0,
         opponent: str = "", opponent_config: str = "", seed: int = -1,
         background: bool = False):
    from dataclasses import asdict, replace
    from datetime import datetime, timezone
    from experiments.alpha_zero.offline_evaluate import load_config, parse_steps

    validate_name(run_name)
    evaluation_name = validate_name(evaluation_name or datetime.now(timezone.utc).strftime("phase1-%Y%m%dT%H%M%SZ"))
    settings = load_config(Path(config) if config else ROOT / "experiments/alpha_zero/configs/progression_phase1.json")
    overrides = {}
    if checkpoints:
        overrides["checkpoints"] = parse_steps(checkpoints.split())
    if games != 0:
        overrides["games"] = games
    if simulations != 0:
        overrides["simulations"] = simulations
    if seed != -1:
        overrides["seed"] = seed
    if opponent and opponent_config:
        raise ValueError("Choose --opponent or --opponent-config")
    if opponent:
        overrides["opponent"] = {"name": opponent, "params": {}}
    if opponent_config:
        overrides["opponent"] = json.loads(Path(opponent_config).read_text(encoding="utf-8"))
    settings = replace(settings, **overrides)
    arguments = (run_name, evaluation_name, asdict(settings))
    print(json.dumps({"event": "modal_evaluation_submission", "run": run_name,
                      "evaluation": evaluation_name, "config": asdict(settings),
                      "artifacts": f"alpha-ayo-alphazero-evaluations /{evaluation_name}"}), flush=True)
    if background:
        call = evaluate_remote.spawn(*arguments)
        print(f"Submitted offline evaluation; function call: {call.object_id}", flush=True)
        print("Monitor evaluate_remote in the Modal dashboard or use modal app logs <app-id> --follow.", flush=True)
    else:
        result = evaluate_remote.remote(*arguments)
        print(json.dumps({"status": result["status"], "completed_checkpoints": result["completed_checkpoints"],
                          "completed_games": result["completed_games"], "artifacts": result["artifacts"]}), flush=True)
