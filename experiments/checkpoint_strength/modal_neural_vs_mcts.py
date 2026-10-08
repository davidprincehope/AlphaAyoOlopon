"""Finite frozen-checkpoint-versus-MCTS experiments on Modal."""
from dataclasses import asdict, replace
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess

import modal
from experiments.alpha_zero import modal_evaluate as base
from experiments.alpha_zero.modal_common import ROOT, validate_name

app = modal.App("alpha-ayo-neural-vs-mcts")

SUITE_CASES = (
    ("puct64-mcts64", "c600_vs_mcts_64.json"),
    ("puct128-mcts128", "c600_vs_mcts_128.json"),
    ("puct64-mcts128", "c600_64_vs_mcts_128.json"),
    ("policy-mcts64", "c600_policy_vs_mcts_64.json"),
)


def suite_configurations(checkpoint):
    from experiments.checkpoint_strength.neural_vs_mcts import NeuralVsMCTSConfig
    return {name: replace(NeuralVsMCTSConfig(**json.loads(
        (ROOT / "experiments/checkpoint_strength/configs" / filename).read_text())), checkpoint=checkpoint)
        for name, filename in SUITE_CASES}


def stop_mps(control, environment):
    if control is None:
        return
    try:
        subprocess.run([control], input="quit\n", text=True, env=environment,
                       timeout=2, check=False)
    except (subprocess.TimeoutExpired, OSError) as exc:
        print(json.dumps({"event": "mps_cleanup_at_container_exit", "detail": type(exc).__name__}), flush=True)


@app.function(image=base.image, gpu=base.GPU, cpu=base.CPU, memory=base.MEMORY_MIB,
              volumes={str(base.CHECKPOINT_ROOT): base.source_volume.with_mount_options(read_only=True),
                       str(base.RESULTS_ROOT): base.results_volume},
              timeout=60 * 60, retries=0, min_containers=0, buffer_containers=0,
              max_containers=2, single_use_containers=True)
def compare_remote(run_name: str, evaluation_name: str, configuration: dict, case_name: str = ""):
    validate_name(run_name)
    validate_name(evaluation_name)
    from experiments.checkpoint_strength.neural_vs_mcts import NeuralVsMCTSConfig, run_neural_vs_mcts
    config = NeuralVsMCTSConfig(**configuration)
    output = base.RESULTS_ROOT / evaluation_name
    if case_name:
        output = output.joinpath(*(validate_name(part) for part in case_name.split("/")))
    mps = None
    environment = dict(os.environ)
    try:
        if config.workers > 1:
            mps = shutil.which("nvidia-cuda-mps-control")
            if mps is None:
                raise RuntimeError("CUDA MPS control tool is missing")
            for key, path in (("CUDA_MPS_PIPE_DIRECTORY", "/tmp/ayo-eval-mps"),
                              ("CUDA_MPS_LOG_DIRECTORY", "/tmp/ayo-eval-mps-log")):
                Path(path).mkdir(exist_ok=True)
                environment[key] = os.environ[key] = path
            subprocess.run([mps, "-d"], check=True, env=environment, timeout=30)
        import jax
        devices = jax.devices()
        if not any(d.platform == "gpu" for d in devices):
            raise RuntimeError("Modal comparison requires a JAX GPU backend")
        hardware = {"gpu_request": base.GPU, "physical_cpu_request": base.CPU,
                    "memory_mib_request": base.MEMORY_MIB, "cuda_mps": config.workers > 1,
                    "jax_devices": [{"id": str(d), "kind": d.device_kind} for d in devices],
                    "visible_logical_cpus": os.cpu_count()}
        quota = Path("/sys/fs/cgroup/cpu.max")
        hardware["cpu_cgroup_quota"] = quota.read_text().strip() if quota.exists() else None
        print(json.dumps({"event": "modal_neural_vs_mcts_started", "evaluation_name": evaluation_name,
                          "hardware": hardware, "source_mount": "read_only"}), flush=True)
        return run_neural_vs_mcts(base.CHECKPOINT_ROOT / run_name, config, output,
                                  on_progress=lambda manifest: base.results_volume.commit(), hardware=hardware)
    finally:
        try:
            base.results_volume.commit()
        finally:
            stop_mps(mps, environment)


@app.function(image=base.image, cpu=1, memory=2048,
              volumes={str(base.RESULTS_ROOT): base.results_volume},
              timeout=2 * 60 * 60, retries=0, single_use_containers=True)
def finalize_suite(evaluation_name, checkpoint, calls):
    root = base.RESULTS_ROOT / validate_name(evaluation_name)
    root.mkdir(parents=True, exist_ok=True)
    status = {"status": "running", "checkpoint": checkpoint, "total_games": 1600,
              "games_per_case": 400, "cases": list(calls), "completed_cases": [],
              "started_at_utc": datetime.now(timezone.utc).isoformat()}
    def save():
        temporary = root / "suite.manifest.json.tmp"
        temporary.write_text(json.dumps(status, indent=2) + "\n")
        temporary.replace(root / "suite.manifest.json")
        base.results_volume.commit()
    save()
    rows = []
    checkpoint_hash = None
    dataset_hash = None
    try:
        for name, call in calls.items():
            result = modal.FunctionCall.from_id(call).get()
            if result["status"] != "completed" or result["games"] != 400 or result["config"]["checkpoint"] != checkpoint:
                raise RuntimeError(f"Incomplete suite case: {name}")
            base.results_volume.reload()
            manifest = json.loads((root / name / "results.manifest.json").read_text())
            current_hash = manifest["checkpoints"][0]["checkpoint_sha256"]
            current_dataset = result["opening_dataset"]["content_sha256"]
            checkpoint_hash = checkpoint_hash or current_hash
            dataset_hash = dataset_hash or current_dataset
            if current_hash != checkpoint_hash or current_dataset != dataset_hash:
                raise RuntimeError("Suite cases use different checkpoints or opening datasets")
            policy = f"C{checkpoint}"
            rows.append({"case": name, "config": result["config"],
                         "neural": result["policies"][policy], "mcts": result["policies"]["MCTS"],
                         "score_interval": result["score_intervals"][policy],
                         "by_seat": result["by_seat"][policy]})
            status["completed_cases"].append(name)
            save()
        (root / "suite.results.json").write_text(json.dumps(rows, indent=2) + "\n")
        status.update(status="completed", completed_games=1600, checkpoint_sha256=checkpoint_hash,
                      opening_dataset_sha256=dataset_hash,
                      finished_at_utc=datetime.now(timezone.utc).isoformat())
        save()
        return status
    except BaseException as exc:
        status.update(status="failed", error=repr(exc))
        save()
        raise


@app.function(image=base.image, cpu=1, memory=2048,
              volumes={str(base.RESULTS_ROOT): base.results_volume},
              timeout=6 * 60 * 60, retries=0, single_use_containers=True)
def finalize_progression(evaluation_name, plan, calls):
    from experiments.checkpoint_strength.mcts_progression import result_row, validate_curve
    root = base.RESULTS_ROOT / validate_name(evaluation_name)
    root.mkdir(parents=True, exist_ok=True)
    rows = {case: [] for case in plan["cases"]}
    status = {"experiment": plan["experiment"], "status": "running", "total_games": 20800,
              "total_checkpoint_cases": 52, "completed_games": 0, "completed_checkpoint_cases": 0,
              "completed_cases": [], "checkpoints": plan["checkpoints"],
              "started_at_utc": datetime.now(timezone.utc).isoformat()}
    def save():
        temporary = root / "progression.manifest.json.tmp"
        temporary.write_text(json.dumps(status, indent=2) + "\n")
        temporary.replace(root / "progression.manifest.json")
        base.results_volume.commit()
    (root / "experiment.plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    save()
    try:
        expected = {f"{case}/c{step:06d}" for case in rows for step in plan["checkpoints"]}
        if set(calls) != expected:
            raise ValueError("Progression call IDs must cover all four conditions and 13 checkpoints")
        for key, call_id in calls.items():
            case, component = key.split("/")
            step = int(component[1:])
            result = modal.FunctionCall.from_id(call_id).get()
            base.results_volume.reload()
            manifest = json.loads((root / case / component / "results.manifest.json").read_text())
            row = result_row(case, step, result, manifest, plan)
            rows[case].append(row)
            rows[case].sort(key=lambda item: item["learner_round"])
            (root / case / "progression.results.jsonl").write_text(
                "".join(json.dumps(item) + "\n" for item in rows[case]))
            status["completed_cases"].append(key)
            status["completed_checkpoint_cases"] += 1
            status["completed_games"] += 400
            save()
        for curve in rows.values():
            validate_curve(curve)
        (root / "progression.results.json").write_text(json.dumps(rows, indent=2) + "\n")
        status.update(status="completed", finished_at_utc=datetime.now(timezone.utc).isoformat())
        save()
        return status
    except BaseException as exc:
        status.update(status="failed", error=repr(exc))
        save()
        raise


@app.local_entrypoint()
def main(run_name: str = "a100-fresh-64-200-20261003", evaluation_name: str = "",
         config: str = "", background: bool = False, checkpoint: int = 0, suite: bool = False,
         progression: bool = False):
    from experiments.checkpoint_strength.neural_vs_mcts import NeuralVsMCTSConfig
    validate_name(run_name)
    evaluation_name = validate_name(evaluation_name or datetime.now(timezone.utc).strftime("neural-vs-mcts-%Y%m%dT%H%M%SZ"))
    if progression:
        if not background or checkpoint or config or suite:
            raise ValueError("Progression requires --background and no individual checkpoint/config/suite")
        from experiments.checkpoint_strength.mcts_progression import make_plan
        plan = make_plan(ROOT / "runs/alpha_zero/modal-a100-600-20261003", run_name, evaluation_name)
        if any(c["workers"] > base.CPU for rows in plan["cases"].values() for c in rows.values()):
            raise ValueError("Request at least one physical CPU core per worker")
        # Verify the copied source before any evaluation calls are submitted.
        for relative, expected in [("manifest.json", plan["training_manifest_sha256"]), *[
            (f"inference-checkpoints/step-{int(step):06d}.npz", digest)
            for step, digest in plan["checkpoint_sha256"].items()]]:
            import hashlib
            content = b"".join(base.source_volume.read_file(f"/{run_name}/{relative}"))
            if hashlib.sha256(content).hexdigest() != expected:
                raise ValueError(f"Remote/local source mismatch: {relative}")
        try:
            exists = bool(list(base.results_volume.iterdir(f"/{evaluation_name}", recursive=False)))
        except (FileNotFoundError, modal.exception.NotFoundError):
            exists = False
        if exists:
            raise FileExistsError(f"Remote evaluation already exists: {evaluation_name}")
        local = ROOT / "runs/checkpoint_strength" / evaluation_name
        local.mkdir(parents=True, exist_ok=True)
        if (local / "launch.json").exists():
            raise FileExistsError(local / "launch.json")
        (local / "experiment.plan.json").write_text(json.dumps(plan, indent=2) + "\n")
        calls = {}
        launch = {"evaluation_name": evaluation_name, "run_name": run_name, "app_id": app.app_id,
                  "profile": os.environ.get("MODAL_PROFILE"), "gpu": base.GPU,
                  "max_gpu_containers": 2, "cpu_per_gpu": base.CPU, "memory_mib_per_gpu": base.MEMORY_MIB,
                  "results_volume": "alpha-ayo-alphazero-evaluations", "total_games": 20800,
                  "function_calls": calls, "finalize_call": None,
                  "submitted_at_utc": datetime.now(timezone.utc).isoformat()}
        def save_launch():
            temporary = local / "launch.json.tmp"
            temporary.write_text(json.dumps(launch, indent=2) + "\n")
            temporary.replace(local / "launch.json")
        save_launch()
        for step in plan["checkpoints"]:
            for case, rows in plan["cases"].items():
                key = f"{case}/c{step:06d}"
                calls[key] = compare_remote.spawn(run_name, evaluation_name, rows[str(step)], key).object_id
                save_launch()
        collector = finalize_progression.spawn(evaluation_name, plan, calls)
        launch["finalize_call"] = collector.object_id
        save_launch()
        print(json.dumps({"event": "mcts_progression_submitted", "app_id": app.app_id,
                          "evaluation_name": evaluation_name, "checkpoint_cases": len(calls),
                          "total_games": 20800, "finalize_call": collector.object_id}), flush=True)
        return
    if suite:
        if not background or checkpoint < 1 or config:
            raise ValueError("Suite mode requires --background, a positive --checkpoint, and no --config")
        configurations = suite_configurations(checkpoint)
        if any(settings.workers > base.CPU for settings in configurations.values()):
            raise ValueError("Request at least one physical CPU core per worker")
        local = ROOT / "runs/checkpoint_strength" / evaluation_name
        if (local / "launch.json").exists():
            raise FileExistsError(local / "launch.json")
        local.mkdir(parents=True, exist_ok=True)
        for name, settings in configurations.items():
            (local / f"config-{name}.json").write_text(json.dumps(asdict(settings), indent=2) + "\n")
        calls = {name: compare_remote.spawn(run_name, evaluation_name, asdict(settings), name).object_id
                 for name, settings in configurations.items()}
        collector = finalize_suite.spawn(evaluation_name, checkpoint, calls)
        launch = {"evaluation_name": evaluation_name, "checkpoint": checkpoint, "run_name": run_name,
                  "app_id": app.app_id, "profile": os.environ.get("MODAL_PROFILE"),
                  "gpu": base.GPU, "max_gpu_containers": 2, "workers_per_case": 8,
                  "results_volume": "alpha-ayo-alphazero-evaluations", "games_per_case": 400,
                  "total_games": 1600, "cases": {name: asdict(settings) for name, settings in configurations.items()},
                  "function_calls": {**calls, "finalize": collector.object_id},
                  "submitted_at_utc": datetime.now(timezone.utc).isoformat()}
        (local / "launch.json").write_text(json.dumps(launch, indent=2) + "\n")
        print(json.dumps({"event": "mcts_suite_submitted", **launch}), flush=True)
        return
    path = Path(config) if config else ROOT / "experiments/checkpoint_strength/configs/c600_vs_mcts_64.json"
    settings = NeuralVsMCTSConfig(**json.loads(path.read_text(encoding="utf-8")))
    if checkpoint:
        settings = replace(settings, checkpoint=checkpoint)
    if settings.workers > base.CPU:
        raise ValueError("Request at least one physical CPU core per worker with ALPHAZERO_EVAL_CPU")
    print(json.dumps({"event": "neural_vs_mcts_submission", "run_name": run_name,
                      "evaluation_name": evaluation_name, "configuration": asdict(settings),
                      "gpu": base.GPU, "cpu": base.CPU, "memory_mib": base.MEMORY_MIB}), flush=True)
    args = (run_name, evaluation_name, asdict(settings))
    if background:
        call = compare_remote.spawn(*args)
        print(f"Submitted C{settings.checkpoint} versus MCTS; function call: {call.object_id}", flush=True)
    else:
        print(json.dumps(compare_remote.remote(*args)), flush=True)
