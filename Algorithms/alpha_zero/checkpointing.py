"""Versioned learner snapshots. Worker execution is intentionally not persisted."""

import hashlib
import json
import os
from pathlib import Path
import shutil
import uuid

import numpy as np

VERSION = 1
CONTRACT = ("architecture_version", "observation_version", "value_perspective",
            "observation_shape", "num_actions", "game")


def write_json(path, value):
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def save_buffer(path, buffer):
    import jax
    state = buffer.buffer_state
    arrays = {"capacity": np.asarray(buffer.max_size),
              "total_seen": np.asarray(buffer.total_seen),
              "rng": np.asarray(buffer._rng), "empty": np.asarray(state is None)}
    if state is not None:
        arrays.update(entry_index=np.asarray(state.entry_index), is_full=np.asarray(state.is_full))
        for i, leaf in enumerate(jax.tree.leaves(state.experience)):
            arrays[f"leaf_{i}"] = np.asarray(leaf)
    with Path(path).open("wb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())


def load_buffer(path, buffer, example):
    import jax
    import jax.numpy as jnp
    from open_spiel.python.algorithms.alpha_zero.replay_buffer import ReplayBufferState
    with np.load(path, allow_pickle=False) as data:
        capacity, total = int(data["capacity"]), int(data["total_seen"])
        if (capacity != buffer.max_size or total < 0 or data["rng"].shape != (2,)
                or data["rng"].dtype != np.dtype("uint32")):
            raise ValueError("Invalid replay capacity, counter, or RNG")
        buffer._rng = jnp.asarray(data["rng"])
        buffer._total_seen = jnp.asarray(total)
        if bool(data["empty"]):
            if total != 0:
                raise ValueError("Empty buffer has nonzero count")
            return
        index, full = int(data["entry_index"]), bool(data["is_full"])
        if index != total or full != (total >= capacity):
            raise ValueError("Inconsistent replay ring counters")
        leaves, structure = jax.tree.flatten(example)
        restored = []
        for i, leaf in enumerate(leaves):
            value = data[f"leaf_{i}"]
            if value.shape != (capacity, *leaf.shape) or value.dtype != leaf.dtype:
                raise ValueError(f"Incompatible replay leaf {i}")
            restored.append(jnp.asarray(value))
        buffer.buffer_state = ReplayBufferState(
            experience=jax.tree.unflatten(structure, restored), capacity=capacity,
            entry_index=jnp.asarray(index), is_full=jnp.asarray(full))


def validate_snapshot(path):
    path = Path(path).resolve()
    if not (path / "COMPLETE").is_file():
        raise ValueError(f"Not a complete training snapshot: {path}. Model-only checkpoints cannot resume training.")
    metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
    if metadata["format_version"] != VERSION:
        raise ValueError("Unsupported training snapshot version")
    for name, expected in metadata["sha256"].items():
        file = (path / name).resolve()
        if not file.is_relative_to(path) or not file.is_file() or digest(file) != expected:
            raise ValueError(f"Incomplete or corrupt snapshot file: {name}")
    if not {"learner.json", "replay.npz"}.issubset(metadata["sha256"]):
        raise ValueError("Snapshot lacks learner/replay state")
    if not any(name.startswith("model/checkpoint-0/") for name in metadata["sha256"]):
        raise ValueError("Snapshot lacks model state")
    learner = json.loads((path / "learner.json").read_text(encoding="utf-8"))
    if learner["completed_step"] != metadata["completed_step"]:
        raise ValueError("Snapshot step mismatch")
    return metadata, learner


def find_snapshot(run, step=None):
    root = Path(run).resolve() / "training-checkpoints"
    if step is not None:
        selected = root / f"step-{step:06d}"
        validate_snapshot(selected)
        return selected
    for selected in sorted(root.glob("step-*"), reverse=True):
        if (selected / "COMPLETE").is_file():
            validate_snapshot(selected)
            return selected
    raise ValueError("No complete training snapshot found; model-only checkpoints lack replay and learner state")


def save_snapshot(config, model_id, replay, evals, completed_step, total_trajectories):
    root = Path(config.path) / "training-checkpoints"
    root.mkdir(exist_ok=True)
    destination = root / f"step-{completed_step:06d}"
    if destination.exists():
        raise FileExistsError(destination)
    temporary = root / f".pending-{uuid.uuid4().hex}"
    temporary.mkdir()
    # Leave interrupted temporary writes for inspection; discovery ignores them.
    shutil.copytree(Path(config.path) / f"checkpoint-{model_id}", temporary / "model" / "checkpoint-0")
    save_buffer(temporary / "replay.npz", replay)
    for i, buffer in enumerate(evals):
        save_buffer(temporary / f"evaluation-{i}.npz", buffer)
    write_json(temporary / "learner.json", {
        "completed_step": completed_step, "total_trajectories": int(total_trajectories),
        "evaluation_levels": len(evals), "session_id": config.session_id})
    manifest = json.loads((Path(config.path) / "manifest.json").read_text(encoding="utf-8"))
    manifest["settings"] = {name: getattr(config, name, value)
                            for name, value in manifest["settings"].items()}
    metadata = {"format_version": VERSION, "completed_step": completed_step,
                "manifest": manifest, "worker_policy": "restart_fresh_games",
                "sha256": {p.relative_to(temporary).as_posix(): digest(p)
                           for p in temporary.rglob("*") if p.is_file()}}
    write_json(temporary / "metadata.json", metadata)
    write_json(temporary / "COMPLETE", {"completed_step": completed_step})
    validate_snapshot(temporary)
    temporary.rename(destination)
    pointer = root / f".latest-{uuid.uuid4().hex}.json"
    write_json(pointer, {"snapshot": destination.name, "completed_step": completed_step})
    os.replace(pointer, root / "latest.json")
    return destination


def restore_model(model, snapshot):
    original = model._path
    try:
        model._path = str(Path(snapshot) / "model")
        model.load_checkpoint(0)
    finally:
        model._path = original


def restore_learner(config, model, replay, evals):
    import jax.numpy as jnp
    from open_spiel.python.algorithms.alpha_zero import utils
    snapshot = Path(config.resume_snapshot)
    metadata, learner = validate_snapshot(snapshot)
    manifest = json.loads((Path(config.path) / "manifest.json").read_text(encoding="utf-8"))
    if any(metadata["manifest"][key] != manifest[key] for key in CONTRACT):
        raise ValueError("Incompatible snapshot model/game contract")
    if learner["evaluation_levels"] != len(evals):
        raise ValueError("Incompatible evaluation windows")
    example = utils.TrainInput(
        observation=jnp.zeros(config.observation_shape, dtype=jnp.float32),
        legals_mask=jnp.zeros(config.output_size, dtype=jnp.bool_),
        policy=jnp.zeros(config.output_size, dtype=jnp.float32),
        value=jnp.asarray(0, dtype=jnp.float32))
    load_buffer(snapshot / "replay.npz", replay, example)
    for i, buffer in enumerate(evals):
        load_buffer(snapshot / f"evaluation-{i}.npz", buffer, jnp.asarray(0, dtype=jnp.float32))
    restore_model(model, snapshot)
    return learner


def prepare_history(run, session, completed_step):
    """Keep original history and archive work newer than the restored snapshot."""
    run, session = Path(run), Path(session)
    history = run / "learner.jsonl"
    if history.exists():
        shutil.copy2(history, session / "previous-learner.jsonl")
        lines = history.read_text().splitlines()
        rows = []
        for i, line in enumerate(lines):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                if i != len(lines) - 1:
                    raise
                # A crash may leave a partially written final record. Original
                # bytes remain in previous-learner.jsonl for inspection.
        retained = [row for row in rows if row["step"] <= completed_step]
        temporary = session / "retained.jsonl"
        temporary.write_text("".join(json.dumps(row) + "\n" for row in retained), encoding="utf-8")
        os.replace(temporary, history)
    for checkpoint in run.glob("checkpoint-*"):
        suffix = checkpoint.name.removeprefix("checkpoint-")
        if suffix.lstrip("-").isdigit() and (int(suffix) < 0 or int(suffix) > completed_step):
            checkpoint.rename(session / checkpoint.name)
    evaluation = run / "evaluation.jsonl"
    if evaluation.exists():
        shutil.copy2(evaluation, session / "previous-evaluation.jsonl")
        retained = []
        for line in evaluation.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row["training_step"] <= completed_step:
                retained.append(row)
        temporary = session / "retained-evaluation.jsonl"
        temporary.write_text("".join(json.dumps(row) + "\n" for row in retained), encoding="utf-8")
        os.replace(temporary, evaluation)
    for checkpoint in (run / "inference-checkpoints").glob("step-*.npz"):
        if int(checkpoint.stem.removeprefix("step-")) > completed_step:
            archive = session / "inference-checkpoints"
            archive.mkdir(exist_ok=True)
            checkpoint.rename(archive / checkpoint.name)
