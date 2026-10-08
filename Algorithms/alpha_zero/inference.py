"""Policy/value inference and immutable round exports, without optimizer creation."""
import json
import os
from pathlib import Path
import uuid

import numpy as np


class InferenceModel:
    def __init__(self, params):
        import jax
        import jax.numpy as jnp
        from open_spiel.python.algorithms.alpha_zero.model_linen import AlphaZeroModel
        network = AlphaZeroModel("ayo_mlp", (15, 1, 1), 6, 256, 3)
        self.params = params
        @jax.jit
        def predict(params, observation, mask):
            logits, value = network.apply({"params": params}, observation.reshape((15, 1, 1)), False)
            logits = jnp.where(mask, logits, jnp.finfo(jnp.float32).min)
            return value, jax.nn.softmax(logits)
        self._predict = predict

    def inference(self, observation, legals_mask):
        import jax
        import jax.numpy as jnp
        # OpenSpiel search runs on the host. Transfer once per prediction rather
        # than indexing GPU policy scalars inside Python's search loop.
        return jax.device_get(self._predict(self.params, jnp.asarray(observation, dtype=jnp.float32),
                                           jnp.asarray(legals_mask, dtype=jnp.bool_)))


def export_round(run, step, params, manifest):
    from flax.traverse_util import flatten_dict
    root = Path(run) / "inference-checkpoints"
    root.mkdir(exist_ok=True)
    target = root / f"step-{step:06d}.npz"
    if target.exists():
        raise FileExistsError(target)
    temporary = root / f".pending-{uuid.uuid4().hex}.npz"
    payload = {"/".join(key): np.asarray(value) for key, value in flatten_dict(params).items()}
    payload["__metadata__"] = np.asarray(json.dumps({
        "format_version": 1, "training_step": step, "manifest": manifest}))
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, target)
    return target


def load_parameters(path):
    """Restore only params, including from historical full Orbax checkpoints."""
    import jax
    import jax.numpy as jnp
    from flax.traverse_util import unflatten_dict
    path = Path(path).resolve()
    if path.is_file():
        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(str(data["__metadata__"]))
            if metadata["format_version"] != 1:
                raise ValueError("Unsupported inference checkpoint version")
            return unflatten_dict({tuple(key.split("/")): jnp.asarray(data[key])
                                   for key in data.files if key != "__metadata__"})
    import orbax.checkpoint as ocp
    with ocp.PyTreeCheckpointer() as checkpointer:
        tree = checkpointer.metadata(str(path)).item_metadata.tree
        item = {"params": tree["params"]}
        sharding = jax.sharding.SingleDeviceSharding(jax.local_devices()[0])
        args = jax.tree.map(lambda _: ocp.ArrayRestoreArgs(sharding=sharding), item)
        return checkpointer.restore(str(path), args=ocp.args.PyTreeRestore(
            item=item, restore_args=args, partial_restore=True))["params"]


def checkpoint_path(run, step):
    run = Path(run).resolve()
    candidates = [run / "inference-checkpoints" / f"step-{step:06d}.npz",
                  run / f"checkpoint-{step}",
                  run / "training-checkpoints" / f"step-{step:06d}" / "model" / "checkpoint-0"]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"No retained checkpoint for learner round {step} in {run}")
