"""Export compact, source-hashed summaries of the completed C600 studies.

Requires the retained local runs. It never copies models, game traces, launch
metadata, cloud credentials, replay buffers, or optimizer snapshots.
"""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "docs/results/alpha_zero_600"
TRAINING = "runs/alpha_zero/modal-a100-600-20261003"
PROGRESSION = "runs/checkpoint_strength/checkpoint-mcts-progression-13x400-20261006-v1"
CONDITIONS = ("puct64-mcts64", "puct128-mcts128", "puct64-mcts128", "policy-mcts64")
CHECKPOINTS = [1, *range(50, 601, 50)]
REPORTS = (
    "checkpoint-head-to-head-c50-c350-64-20261005",
    "checkpoint-head-to-head-c350-c600-64-20261005",
    "checkpoint-c600-policy-vs-mcts-64-20261005",
    "checkpoint-c600-vs-mcts-64-20261005",
    "checkpoint-c600-64-vs-mcts-128-20261005",
    "checkpoint-c600-vs-mcts-128-20261005",
)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def relative_paths(value):
    """Remove the local machine's absolute checkout prefix, including in keys."""
    if isinstance(value, dict):
        return {relative_paths(key): relative_paths(item) for key, item in value.items()}
    if isinstance(value, list):
        return [relative_paths(item) for item in value]
    if isinstance(value, str):
        for prefix in (str(ROOT) + "\\", ROOT.as_posix() + "/"):
            if value.startswith(prefix):
                return value[len(prefix):].replace("\\", "/")
    return value


def main():
    manifest = json.loads((ROOT / PROGRESSION / "progression.manifest.json").read_text(encoding="utf-8"))
    verification = json.loads((ROOT / PROGRESSION / "verification.json").read_text(encoding="utf-8"))
    curves = json.loads((ROOT / PROGRESSION / "progression.results.json").read_text(encoding="utf-8"))
    if (manifest["status"] != "completed" or manifest["completed_games"] != 20800
            or verification["status"] != "passed" or verification["games_replayed"] != 20800
            or set(curves) != set(CONDITIONS)):
        raise ValueError("Publish only the complete, verified 20,800-game progression")
    for rows in curves.values():
        if [row["learner_round"] for row in rows] != CHECKPOINTS or any(row["games"] != 400 for row in rows):
            raise ValueError("Progression must contain all 13 checkpoints and 400 games per point")
        for row in rows:
            if row["score_rate"] != (row["wins"] + 0.5 * row["draws"]) / 400:
                raise ValueError("Progression outcome/score mismatch")
    for name, expected in verification["files"].items():
        if digest(ROOT / PROGRESSION / name) != expected:
            raise ValueError(f"Verified artifact changed: {name}")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    sources = {}

    def export(source, destination, transform=None):
        path = ROOT / source
        data = json.loads(path.read_text(encoding="utf-8"))
        if transform:
            data = transform(data)
        target = OUTPUT / destination
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(relative_paths(data), indent=2) + "\n", encoding="utf-8")
        sources[destination] = {"source": source, "source_sha256": digest(path),
                                "export_sha256": digest(target)}

    export(f"{TRAINING}/manifest.json", "training_manifest.json")
    learner_path = ROOT / TRAINING / "learner.jsonl"
    final = json.loads(learner_path.read_text(encoding="utf-8").splitlines()[-1])
    if final["step"] != 600:
        raise ValueError("Training summary must end at learner round 600")
    summary = {"completed_learner_rounds": final["step"], "gradient_updates": final["gradient_updates"],
               "collected_positions": final["total_states"], "self_play_trajectories": final["total_trajectories"],
               "final_loss": final["loss"], "source": f"{TRAINING}/learner.jsonl",
               "source_sha256": digest(learner_path)}
    (OUTPUT / "training_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    sources["training_summary.json"] = {"source": summary["source"], "source_sha256": summary["source_sha256"],
                                        "export_sha256": digest(OUTPUT / "training_summary.json")}

    for folder in REPORTS:
        export(f"runs/checkpoint_strength/{folder}/results.json", f"{folder}/results.json")
        if folder == "checkpoint-c600-vs-mcts-64-20261005":
            export(f"runs/checkpoint_strength/{folder}/results.manifest.json", f"{folder}/results.manifest.json")
    for name in ("progression.results.json", "progression.manifest.json", "verification.json", "experiment.plan.json"):
        export(f"{PROGRESSION}/{name}", name)
    (OUTPUT / "source_provenance.json").write_text(json.dumps({
        "normalization": "Local absolute checkout prefixes are replaced with repository-relative paths. "
                         "Training summary retains the final round's aggregate counts.",
        "sources": sources,
    }, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"exported_summaries": len(sources), "verified_progression_games": 20800,
                      "output": str(OUTPUT.relative_to(ROOT))}))


if __name__ == "__main__":
    main()
