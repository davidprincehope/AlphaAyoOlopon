"""Select the checked-out OpenSpiel sources and report their provenance."""

import importlib.metadata
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
OPEN_SPIEL_ROOT = ROOT / "open_spiel"


def use_repository_open_spiel():
    """Avoid the outer checkout directory shadowing open_spiel.python.

    pyspiel itself must still be installed/compiled for this interpreter.
    Fail instead of silently mixing an already-imported foreign Python package.
    """
    expected = OPEN_SPIEL_ROOT / "open_spiel"
    if not (expected / "python/algorithms/alpha_zero/model_linen.py").is_file():
        raise RuntimeError("The repository's JAX OpenSpiel checkout is missing")
    if "open_spiel" in sys.modules:
        package = sys.modules["open_spiel"]
        # The outer directory contains an __init__.py too. Extend its package
        # search path only if no Python submodules have already been imported.
        if expected not in [Path(p).resolve() for p in package.__path__]:
            if any(n.startswith("open_spiel.python") for n in sys.modules):
                raise RuntimeError("Import the AlphaZero runtime before open_spiel.python")
            package.__path__.insert(0, str(expected))
    if str(OPEN_SPIEL_ROOT) not in sys.path:
        sys.path.insert(0, str(OPEN_SPIEL_ROOT))


def provenance():
    versions = {}
    for name in ("open_spiel", "numpy", "jax", "jaxlib", "flax", "optax", "chex", "orbax-checkpoint"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    try:
        revision = subprocess.check_output(
            ["git", "-C", str(OPEN_SPIEL_ROOT), "rev-parse", "HEAD"], text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = "unknown"
    return {"python": sys.version, "packages": versions,
            "open_spiel_revision": revision, "open_spiel_source": str(OPEN_SPIEL_ROOT)}


def guard_worker_failures():
    """Make upstream queue polling fail when a worker dies, instead of hanging."""
    from open_spiel.python.utils import spawn

    class CheckedQueue:
        def __init__(self, queue, process):
            self._queue = queue
            self._process = process

        def __getattr__(self, name):
            return getattr(self._queue, name)

        def get_nowait(self):
            code = self._process.exitcode
            if code is not None and code != 0:
                raise RuntimeError(
                    f"AlphaZero worker exited with code {code}; inspect log-actor-* "
                    "and log-evaluator-* in the run directory"
                )
            return self._queue.get_nowait()

    class CheckedProcess(spawn.Process):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._queue = CheckedQueue(self._queue, self)

    spawn.Process = CheckedProcess
