"""Execute and verify the approved le4capture suite strictly one case at a time."""
from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import modal
from experiments.alpha_zero import modal_evaluate as base
from experiments.alpha_zero.modal_common import ROOT, build_image, validate_name
from experiments.checkpoint_strength.le4capture_suite import save_json, sha, validate_plan

app = modal.App("alpha-ayo-le4capture-sequential-suite")
cpu_image = build_image({"JAX_PLATFORMS": "cpu"}, extra_dirs=("experiments/checkpoint_strength",))
volumes = {str(base.CHECKPOINT_ROOT): base.source_volume.with_mount_options(read_only=True),
           str(base.RESULTS_ROOT): base.results_volume}


@app.function(image=base.image, gpu=base.GPU, cpu=base.CPU, memory=base.MEMORY_MIB,
              volumes=volumes, timeout=60*60, retries=0, max_containers=1,
              min_containers=0, buffer_containers=0, single_use_containers=True)
def gpu_case(source_run, evaluation_name, case):
    validate_name(source_run)
    validate_name(evaluation_name)
    for part in case["id"].split("/"):
        validate_name(part)
    output = base.RESULTS_ROOT / evaluation_name / case["id"]
    from experiments.checkpoint_strength.le4capture_recovery import prepare_case_output
    if prepare_case_output(base.RESULTS_ROOT/evaluation_name, case):
        return {"status": "completed", "games": 400, "case_id": case["id"]}
    base.results_volume.commit()
    source = base.CHECKPOINT_ROOT / source_run
    config, kind = case["config"], case["kind"]
    control = None
    environment = dict(os.environ)
    try:
        if config.get("workers", 1) > 1:
            control = shutil.which("nvidia-cuda-mps-control")
            if control is None:
                raise RuntimeError("CUDA MPS control is missing")
            for key, directory in (("CUDA_MPS_PIPE_DIRECTORY", "/tmp/ayo-suite-mps"), ("CUDA_MPS_LOG_DIRECTORY", "/tmp/ayo-suite-mps-log")):
                Path(directory).mkdir(exist_ok=True)
                environment[key] = os.environ[key] = directory
            subprocess.run([control, "-d"], check=True, env=environment, timeout=30)
        import jax
        devices = jax.devices()
        if not any(device.platform == "gpu" for device in devices):
            raise RuntimeError("Neural evaluation requires a GPU backend")
        hardware = {"gpu_request": base.GPU, "physical_cpu_request": base.CPU, "memory_mib_request": base.MEMORY_MIB,
                    "cuda_mps": control is not None, "jax_devices": [{"id": str(d), "kind": d.device_kind} for d in devices]}
        progress = lambda manifest: base.results_volume.commit()
        if kind == "greedy_eoh":
            from experiments.checkpoint_strength.greedy_progression import GreedyCheckpointConfig, run_checkpoint
            result = run_checkpoint(source, GreedyCheckpointConfig(**config), output, on_progress=progress, hardware=hardware)
        elif kind == "neural_vs_mcts":
            from experiments.checkpoint_strength.neural_vs_mcts import NeuralVsMCTSConfig, run_neural_vs_mcts
            result = run_neural_vs_mcts(source, NeuralVsMCTSConfig(**config), output, on_progress=progress, hardware=hardware)
        elif kind == "head_to_head":
            from experiments.checkpoint_strength.head_to_head import HeadToHeadConfig, run_head_to_head
            result = run_head_to_head(source, HeadToHeadConfig(**config), output, on_progress=progress)
        else:
            raise ValueError("Unknown neural case")
        return {"status": result["status"], "games": result["games"], "case_id": case["id"]}
    finally:
        base.results_volume.commit()
        if control:
            try:
                subprocess.run([control], input="quit\n", text=True, env=environment, timeout=2, check=False)
            except (subprocess.TimeoutExpired, OSError):
                pass


@app.function(image=cpu_image, cpu=2, memory=4096, volumes=volumes, timeout=60*60,
              retries=0, max_containers=1, min_containers=0, buffer_containers=0,
              single_use_containers=True)
def cpu_case(plan, case):
    validate_plan(plan)
    from experiments.checkpoint_strength.le4capture_recovery import prepare_case_output
    if prepare_case_output(base.RESULTS_ROOT/plan["evaluation_name"], case):
        return {"status": "completed", "games": 400, "case_id": case["id"]}
    base.results_volume.commit()
    from experiments.alpha_zero import offline_evaluate as offline
    from experiments.checkpoint_strength.opening_dataset import load_dataset
    from experiments.checkpoint_strength.mcts_baseline import run_baseline
    game = offline.pyspiel.load_game(plan["game"])
    dataset = load_dataset(case["config"]["openings"], game)
    def progress(event):
        if event.get("completed_games", 0) % 20 == 0 or event["event"] != "game_completed":
            base.results_volume.commit()
    try:
        result = run_baseline(game, dataset, base.RESULTS_ROOT / plan["evaluation_name"] / case["id"],
                              budgets=(case["config"]["simulations"],), seed=plan["seed"], uct_c=plan["uct_c"], on_progress=progress)
        return {"status": result["status"], "games": result["completed_games"], "case_id": case["id"]}
    finally:
        base.results_volume.commit()


@app.function(image=cpu_image, cpu=1, memory=4096, volumes=volumes, timeout=24*60*60,
              retries=0, max_containers=1, min_containers=0, buffer_containers=0,
              single_use_containers=True)
def sequential_suite(plan, resume=False):
    validate_plan(plan)
    from experiments.checkpoint_strength.le4capture_suite import case_files, sha, validate_case
    from experiments.checkpoint_strength.le4capture_recovery import load_saved_suite, pending_call_id
    root = base.RESULTS_ROOT / validate_name(plan["evaluation_name"])
    base.results_volume.reload()
    if (root / "suite.manifest.json").exists():
        status, rows = load_saved_suite(root, plan, explicit_resume=resume,
                                       recover_case=lambda folder, case, plan: validate_case(folder, case, plan)[0])
        if status["status"] == "completed":
            return status
        status.setdefault("recoveries", []).append({"previous_status": status["status"],
            "previous_error": status.get("error"), "at_utc": datetime.now(timezone.utc).isoformat()})
        status.update(status="running")
        status.pop("error", None)
    else:
        root.mkdir(parents=True, exist_ok=True)
        status = {"status": "running", "evaluation_name": plan["evaluation_name"], "total_cases": plan["total_cases"], "total_games": plan["total_games"],
                  "completed_cases": [], "completed_games": 0, "active_case": None, "active_call_id": None,
                  "started_at_utc": datetime.now(timezone.utc).isoformat(), "max_simultaneous_experiments": 1}
        save_json(root / "experiment.plan.json", plan)
        rows = []
    status.setdefault("orchestration_revisions", []).append({
        name: sha(ROOT/name) for name in ("experiments/checkpoint_strength/modal_le4capture_suite.py",
                                        "experiments/checkpoint_strength/le4capture_recovery.py")})
    def save():
        save_json(root / "suite.results.json", rows)
        save_json(root / "suite.manifest.json", status)
        base.results_volume.commit()
    save()
    try:
        for case in plan["cases"]:
            if case["id"] in status["completed_cases"]:
                continue
            saved_call = pending_call_id(status, case)
            status.update(active_case=case["id"], active_call_id=saved_call)
            folder = root / case["id"]
            manifest_name, _, _ = case_files(case)
            existing = json.loads((folder/manifest_name).read_text()) if (folder/manifest_name).exists() else {}
            if existing.get("status") != "completed":
                call = (modal.FunctionCall.from_id(saved_call) if saved_call else
                        cpu_case.spawn(plan, case) if case["kind"] == "mcts_vs_greedy_eoh" else
                        gpu_case.spawn(plan["source_run"], plan["evaluation_name"], case))
                status["active_call_id"] = call.object_id
                save()
                print(json.dumps({"event": "case_started", "case_id": case["id"], "function_call_id": call.object_id}), flush=True)
                result = call.get()
                if result["status"] != "completed" or result["games"] != 400:
                    raise RuntimeError("Case did not complete: " + case["id"])
                base.results_volume.reload()
            row, verification = validate_case(folder, case, plan)
            save_json(folder / "verification.json", verification)
            rows.append(row)
            status["completed_cases"].append(case["id"])
            status["completed_games"] += 400
            save()
            print(json.dumps({"event": "case_verified", "case_id": case["id"], "policies": row["policies"],
                              "completed_games": status["completed_games"], "total_games": plan["total_games"]}), flush=True)
        status.update(status="completed", active_case=None, active_call_id=None, finished_at_utc=datetime.now(timezone.utc).isoformat())
        save()
        return status
    except KeyboardInterrupt:
        status.update(status="interrupted", error="KeyboardInterrupt()", interrupted_at_utc=datetime.now(timezone.utc).isoformat())
        save()
        raise
    except BaseException as exc:
        status.update(status="failed", error=repr(exc), failed_at_utc=datetime.now(timezone.utc).isoformat())
        save()
        raise


@app.local_entrypoint()
def main(plan_file: str, background: bool = False, evaluation_name: str = "", resume: bool = False):
    plan = validate_plan(json.loads(Path(plan_file).read_text(encoding="utf-8")))
    if evaluation_name:
        plan = dict(plan, evaluation_name=validate_name(evaluation_name))
    local = ROOT / "runs/checkpoint_strength" / validate_name(plan["evaluation_name"])
    local.mkdir(parents=True, exist_ok=True)
    if (local / "launch.json").exists() and not resume:
        raise FileExistsError(local / "launch.json")
    if resume:
        if json.loads((local/"experiment.plan.json").read_text()) != plan:
            raise ValueError("Recovery plan differs")
        previous = local/"launch.json"
        previous_launch = json.loads(previous.read_text())
        inventory = subprocess.run([sys.executable, "-m", "modal", "app", "list", "--json"],
                                   capture_output=True, text=True, timeout=30, check=True)
        previous_app = next((item for item in json.loads(inventory.stdout) if item["app_id"] == previous_launch["app_id"]), None)
        if previous_app is None or previous_app["state"] != "stopped" or int(previous_app["tasks"]) != 0:
            raise ValueError("Stop the previous controller before explicit recovery")
        archive = local/('launch-before-recovery-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')+'.json')
        archive.write_bytes(previous.read_bytes())
    from experiments.checkpoint_strength.le4capture_launch import snapshot_sources
    snapshot_sources(plan, local)
    save_json(local / "experiment.plan.json", plan)
    launch = {"evaluation_name": plan["evaluation_name"], "source_run": plan["source_run"], "profile": os.environ.get("MODAL_PROFILE"),
              "app_id": app.app_id, "results_volume": "alpha-ayo-alphazero-evaluations", "total_games": plan["total_games"],
              "max_simultaneous_experiments": 1, "submitted_at_utc": datetime.now(timezone.utc).isoformat()}
    launch.update(resume=resume, orchestration_code_sha256={
        name: sha(ROOT/name) for name in (
            "experiments/checkpoint_strength/modal_le4capture_suite.py", "experiments/checkpoint_strength/le4capture_recovery.py")})
    if resume:
        launch["previous_launch"] = archive.name
    if background:
        call = sequential_suite.spawn(plan, resume=resume)
        launch["function_call_id"] = call.object_id
        save_json(local / "launch.json", launch)
        print(json.dumps({"event": "sequential_suite_submitted", **launch}), flush=True)
    else:
        save_json(local / "launch.json", launch)
        print(json.dumps(sequential_suite.remote(plan, resume=resume)), flush=True)
