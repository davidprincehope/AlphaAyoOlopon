"""Fresh subprocess entrypoint for Modal's single-node AlphaZero job."""

import json
import multiprocessing
import os
from pathlib import Path
import time


def configure_process():
    # Each actor has its own JAX runtime. Apply before importing NumPy/JAX.
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "NUMEXPR_NUM_THREADS", "TF_NUM_INTRAOP_THREADS",
                 "TF_NUM_INTEROP_THREADS"):
        os.environ[name] = "1"
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    multiprocessing.set_start_method("spawn", force=True)


configure_process()
# Spawn children must register the game before unpickling worker arguments.
from experiments.alpha_zero import train  # noqa: E402


def install_compiled_inference():
    """Use a persistent compiled forward pass and one host transfer per leaf."""
    import jax
    from Algorithms.alpha_zero.inference import InferenceModel
    from open_spiel.python.algorithms.alpha_zero.model_linen import Model

    def inference(model, observation, mask):
        if not hasattr(model, "_modal_inference_view"):
            model._modal_inference_view = InferenceModel(model._state.params)
        view = model._modal_inference_view
        # Checkpoint loads replace the tree; never capture stale weights in JIT.
        view.params = model._state.params
        return jax.device_get(view.inference(observation, mask))

    Model.inference = inference


# Spawn workers re-import this module, so the optimization applies to every
# actor as well as the learner, before the worker target is invoked.
install_compiled_inference()


def install_immutable_actor_checkpoints():
    """Keep broadcast checkpoints readable while actors finish older games."""
    from uuid import uuid4
    from open_spiel.python.algorithms.alpha_zero.model_linen import Model

    original_save = Model.save_checkpoint

    def save_checkpoint(model, step, device=None):
        # The trainer reuses -1 between retained checkpoints. Orbax force-save
        # deletes that directory, racing actors restoring a queued update.
        # Publish each transient update once and retain it for this run so even
        # an actor finishing a long game can safely restore its queued version.
        checkpoint_id = f"live-{uuid4().hex}" if step == -1 else step
        return original_save(model, checkpoint_id, device=device)

    Model.save_checkpoint = save_checkpoint


install_immutable_actor_checkpoints()


def install_round_timing():
    """Measure actual phases, independent of upstream's synthetic first minute."""
    from open_spiel.python.utils import data_logger, file_logger

    clock = {"started": time.perf_counter()}
    original_print = file_logger.FileLogger.print
    original_write = data_logger.DataLoggerJsonLines.write

    def timed_print(logger, *args):
        # Actor logs do not go to stdout; only the learner enables this flag.
        if logger.also_to_stdout and args:
            now = time.perf_counter()
            if args[0] == "Collecting trajectories":
                clock["round_start"] = now
            elif args[0] == "Step:":
                clock["collection_end"] = now
            elif args[0] == "Checkpoint saved:":
                clock["learning_end"] = now
        return original_print(logger, *args)

    def timed_write(logger, data):
        if "gradient_updates" in data and "round_start" in clock:
            now = time.perf_counter()
            data = dict(data)
            data["timing"] = {
                "collection_seconds": clock["collection_end"] - clock["round_start"],
                "learning_and_model_checkpoint_seconds": clock["learning_end"] - clock["collection_end"],
                "export_and_evaluation_seconds": now - clock["learning_end"],
                "round_seconds": now - clock["round_start"],
                "process_elapsed_seconds": now - clock["started"],
            }
        return original_write(logger, data)

    file_logger.FileLogger.print = timed_print
    data_logger.DataLoggerJsonLines.write = timed_write


def main():
    import jax

    devices = jax.devices()
    if not any(device.platform == "gpu" for device in devices):
        raise RuntimeError(f"Modal training requires a JAX GPU backend; found {devices}")
    hardware = {"jax_devices": [{"id": str(d), "kind": d.device_kind} for d in devices],
                      "jax_version": jax.__version__,
                      "multiprocessing_start_method": multiprocessing.get_start_method(),
                      "visible_logical_cpus": os.cpu_count(),
                      "cpu_affinity_count": len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
                      "gpu_preallocation": os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]}
    quota = Path("/sys/fs/cgroup/cpu.max")
    hardware["cpu_cgroup_quota"] = quota.read_text().strip() if quota.exists() else None
    Path("/tmp/alpha-zero-hardware.json").write_text(json.dumps(hardware), encoding="utf-8")
    print(json.dumps(hardware), flush=True)
    install_round_timing()
    train.main()


if __name__ == "__main__":
    main()
