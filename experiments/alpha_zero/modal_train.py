"""Run local self-play actors and a learner together on one Modal A100."""

import json
import os
from pathlib import Path

import modal
from experiments.alpha_zero.modal_common import REMOTE_ROOT, ROOT, build_image, gpu_request, validate_name


GPU = gpu_request("ALPHAZERO_MODAL_GPU")
CPU = float(os.environ.get("ALPHAZERO_MODAL_CPU", "30"))
MEMORY_MIB = int(os.environ.get("ALPHAZERO_MODAL_MEMORY_MIB", "65536"))
USE_MPS = os.environ.get("ALPHAZERO_MODAL_MPS", "1") == "1"
if CPU <= 0 or MEMORY_MIB <= 0:
    raise ValueError("CPU and memory requests must be positive")

app = modal.App("alpha-ayo-alphazero")
volume = modal.Volume.from_name("alpha-ayo-alphazero-runs", create_if_missing=True)
image = build_image({"ALPHAZERO_MODAL_GPU": GPU, "ALPHAZERO_MODAL_CPU": str(CPU),
                     "ALPHAZERO_MODAL_MEMORY_MIB": str(MEMORY_MIB),
                     "ALPHAZERO_MODAL_MPS": "1" if USE_MPS else "0"})


@app.function(image=image, gpu=GPU, cpu=CPU, memory=MEMORY_MIB,
              volumes={"/runs": volume}, timeout=24 * 60 * 60,
              max_containers=1, single_use_containers=True)
def train_remote(run_name: str, settings: dict, max_steps: int, actors: int,
                 evaluation_games: int, resume: bool):
    import subprocess
    import shutil
    import sys
    import time

    from Algorithms.alpha_zero.config import Settings

    validate_name(run_name)
    Settings(**settings)  # Validate before creating run files.
    if actors < 1 or max_steps < 1 or evaluation_games < 0 or evaluation_games % 2:
        raise ValueError("actors/max_steps must be positive; evaluation_games must be even and nonnegative")
    output = Path("/runs") / run_name
    if not resume and output.exists():
        raise FileExistsError(f"Run already exists: {output}; use --resume or a new --run-name")
    if resume and not output.is_dir():
        raise FileNotFoundError(f"No run to resume: {output}")
    config_file = Path("/tmp/alpha-zero-settings.json")
    config_file.write_text(json.dumps(settings), encoding="utf-8")
    command = [sys.executable, "-m", "experiments.alpha_zero.modal_worker",
               "--max-steps", str(max_steps), "--actors", str(actors),
               "--evaluation-games", str(evaluation_games)]
    command += (["--resume", str(output)] if resume else
                ["--config", str(config_file), "--output", str(output)])
    print(f"Resources: {GPU}, {CPU} physical CPU cores, {MEMORY_MIB} MiB RAM; {actors} local actors", flush=True)
    print(f"Run: {output}; target total learner step: {max_steps}", flush=True)
    started = time.perf_counter()
    mps = None
    try:
        environment = dict(os.environ)
        if USE_MPS:
            mps = shutil.which("nvidia-cuda-mps-control")
            if mps is None:
                raise RuntimeError("CUDA MPS control tool is missing; set ALPHAZERO_MODAL_MPS=0 to explicitly disable MPS")
            for key, path in (("CUDA_MPS_PIPE_DIRECTORY", "/tmp/alpha-ayo-mps"),
                              ("CUDA_MPS_LOG_DIRECTORY", "/tmp/alpha-ayo-mps-log")):
                Path(path).mkdir(exist_ok=True)
                environment[key] = path
            subprocess.run([mps, "-d"], check=True, env=environment, timeout=30)
            print("CUDA MPS controller started before any JAX GPU initialization", flush=True)
        subprocess.run(command, check=True, cwd=REMOTE_ROOT, env=environment)
        rows = [json.loads(line) for line in (output / "learner.jsonl").read_text().splitlines() if line]
        # Resume retains earlier history. The last session identifies this call.
        session = rows[-1]["session_id"]
        rounds = [{"step": row["step"], "gradient_updates": row["gradient_updates"],
                   **row["timing"]} for row in rows
                  if row.get("session_id") == session and "timing" in row]
        summary = {"run": run_name, "gpu_request": GPU, "physical_cpu_request": CPU,
                   "memory_mib_request": MEMORY_MIB, "actors": actors,
                   "evaluation_games": evaluation_games,
                   "cuda_mps_requested": USE_MPS,
                   "worker_wall_seconds": time.perf_counter() - started, "rounds": rounds,
                   "timing_note": "Round phases exclude initial model startup, full replay snapshots, and actor shutdown; worker wall includes them. Compare process_elapsed_seconds between rounds for end-to-end cadence."}
        summary["hardware"] = json.loads(Path("/tmp/alpha-zero-hardware.json").read_text())
        (output / f"modal-summary-{session}.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(summary, indent=2), flush=True)
        return summary
    finally:
        try:
            volume.commit()
        finally:
            if mps is not None:
                subprocess.run([mps], input="quit\n", text=True, env=environment,
                               timeout=30, check=False)


@app.local_entrypoint()
def main(run_name: str = "", config: str = "", max_steps: int = 3,
         actors: int = 28, evaluation_games: int = 0, resume: bool = False,
         background: bool = False):
    from datetime import datetime, timezone
    from dataclasses import asdict, replace
    from Algorithms.alpha_zero.config import load_settings

    if resume and not run_name:
        raise ValueError("--resume requires --run-name")
    run_name = validate_name(run_name or datetime.now(timezone.utc).strftime("a100-%Y%m%dT%H%M%SZ"))
    if max_steps < 1:
        raise ValueError("--max-steps must be positive; use finite resumable sessions")
    settings = asdict(replace(load_settings(Path(config) if config else ROOT / "experiments/alpha_zero/configs/starter.json"),
                              actors=actors, max_steps=max_steps, evaluation_games=evaluation_games))
    if resume and config:
        raise ValueError("Resume restores the saved settings; omit --config")
    if background:
        # --detach controls app lifetime; spawn also removes ownership by the
        # local synchronous caller. Return normally once submission is accepted.
        call = train_remote.spawn(run_name, settings, max_steps, actors, evaluation_games, resume)
        print(f"Submitted {run_name}; function call: {call.object_id}")
        print(f"Artifacts: Modal Volume alpha-ayo-alphazero-runs /{run_name}")
        print("Submission complete. Monitor progress in the Modal app dashboard.")
        return
    summary = train_remote.remote(run_name, settings, max_steps, actors, evaluation_games, resume)
    print(f"Persisted in Modal Volume alpha-ayo-alphazero-runs: /{run_name}")
    print(json.dumps(summary, indent=2))
