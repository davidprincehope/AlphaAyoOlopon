"""Shared Modal image and path validation for AlphaZero jobs."""
from pathlib import Path
import os
import re

import modal

REMOTE_ROOT = "/opt/alpha-ayo"
ROOT = Path(__file__).resolve().parents[2] if modal.is_local() else Path(REMOTE_ROOT)
DEFAULT_GPU = "A100-40GB"


def gpu_request(variable, *, default=DEFAULT_GPU, allowed=("A100-40GB", "A100-80GB")):
    """Validate the supported GPU request for each experiment workload."""
    gpu = os.environ.get(variable, default)
    if gpu not in allowed:
        raise ValueError(f"{variable} must be one of {', '.join(allowed)}")
    return gpu


def validate_name(name):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,100}", name):
        raise ValueError("Run name must be a single path component using letters, digits, _, ., or -")
    return name


def build_image(environment, *, extra_dirs=()):
    image = modal.Image.debian_slim(python_version="3.12")
    if not modal.is_local():
        return image
    image = (image.apt_install("libgomp1")
             .pip_install_from_requirements(str(ROOT / "requirements-alpha-zero.txt"))
             .pip_install("jax[cuda12]==0.11.2")
             .env({"PYTHONPATH": REMOTE_ROOT, "PYTHONUNBUFFERED": "1",
                   "JAX_PLATFORMS": "cuda", "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
                   "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                   **environment})
             .workdir(REMOTE_ROOT))
    # Ship the edited checkout, excluding run data, credentials, environments, and EOH.
    for local in ("Algorithms", "Model/ayo_olopon", "experiments/alpha_zero",
                  "experiments/agent_benchmark", "open_spiel/open_spiel/python", *extra_dirs):
        image = image.add_local_dir(ROOT / local, f"{REMOTE_ROOT}/{local}",
                                    ignore=["**/__pycache__/**", "**/*.pyc", "**/.git/**"])
    for local in ("Model/__init__.py", "experiments/__init__.py", "open_spiel/open_spiel/__init__.py"):
        if (ROOT / local).exists():
            image = image.add_local_file(ROOT / local, f"{REMOTE_ROOT}/{local}")
    return image
