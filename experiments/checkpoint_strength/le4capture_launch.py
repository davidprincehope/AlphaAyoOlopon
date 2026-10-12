"""Keep the exact local evaluation sources alongside a fresh suite launch."""
from pathlib import Path

from experiments.checkpoint_strength.le4capture_suite import ROOT, OPENINGS, sha


def snapshot_sources(plan, local, *, source_root=ROOT):
    source_root, local = Path(source_root), Path(local)
    expected = dict(plan["code_sha256"])
    expected[OPENINGS] = plan["opening_dataset_file_sha256"]
    expected["experiments/checkpoint_strength/le4capture_suite.py"] = plan["verification_code_sha256"]
    for name, digest in expected.items():
        if sha(source_root / name) != digest:
            raise ValueError("Frozen launch source differs: " + name)
    for name in expected:
        destination = local / "source-snapshot" / name
        content = (source_root / name).read_bytes()
        if destination.exists() and destination.read_bytes() != content:
            raise ValueError("Existing source snapshot differs: " + name)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(content)
