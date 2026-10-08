"""Run a direct frozen-checkpoint comparison on Modal with read-only models."""
from dataclasses import asdict
from datetime import datetime, timezone
import json

import modal
from experiments.alpha_zero import modal_evaluate as base
from experiments.alpha_zero.modal_common import ROOT, validate_name

app = modal.App("alpha-ayo-checkpoint-head-to-head")


@app.function(image=base.image, gpu=base.GPU, cpu=base.CPU, memory=base.MEMORY_MIB,
              volumes={str(base.CHECKPOINT_ROOT): base.source_volume.with_mount_options(read_only=True),
                       str(base.RESULTS_ROOT): base.results_volume},
              timeout=24 * 60 * 60, retries=0, min_containers=0, buffer_containers=0,
              max_containers=1, single_use_containers=True)
def head_to_head_remote(run_name: str, evaluation_name: str, configuration: dict):
    import jax
    from experiments.checkpoint_strength.head_to_head import HeadToHeadConfig, run_head_to_head
    validate_name(run_name)
    validate_name(evaluation_name)
    if not any(device.platform == "gpu" for device in jax.devices()):
        raise RuntimeError("The Modal comparison requires a JAX GPU backend")
    print(json.dumps({"event": "modal_head_to_head_started", "run_name": run_name,
                      "evaluation_name": evaluation_name, "gpu": base.GPU,
                      "devices": [str(device) for device in jax.devices()],
                      "source_mount": "read_only"}), flush=True)
    try:
        return run_head_to_head(base.CHECKPOINT_ROOT / run_name, HeadToHeadConfig(**configuration),
                                base.RESULTS_ROOT / evaluation_name,
                                on_progress=lambda manifest: base.results_volume.commit())
    finally:
        base.results_volume.commit()


@app.local_entrypoint()
def main(run_name: str = "a100-fresh-64-200-20261003", evaluation_name: str = "",
         config: str = "", background: bool = False):
    from experiments.checkpoint_strength.head_to_head import HeadToHeadConfig
    validate_name(run_name)
    evaluation_name = validate_name(evaluation_name or datetime.now(timezone.utc).strftime("head-to-head-%Y%m%dT%H%M%SZ"))
    path = config or ROOT / "experiments/checkpoint_strength/configs/head_to_head_c50_c350_64.json"
    from pathlib import Path
    settings = HeadToHeadConfig(**json.loads(Path(path).read_text(encoding="utf-8")))
    print(json.dumps({"event": "head_to_head_submission", "run_name": run_name,
                      "evaluation_name": evaluation_name, "configuration": asdict(settings)}), flush=True)
    args = (run_name, evaluation_name, asdict(settings))
    if background:
        call = head_to_head_remote.spawn(*args)
        print(f"Submitted direct checkpoint comparison; function call: {call.object_id}", flush=True)
    else:
        print(json.dumps(head_to_head_remote.remote(*args)), flush=True)
