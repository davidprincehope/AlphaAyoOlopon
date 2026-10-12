"""Manifest and replay checks for the selected le4capture MLP benchmark."""
from collections import defaultdict
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

from experiments.checkpoint_strength.opening_dataset import resolve_dataset_path

ROOT = Path(__file__).resolve().parents[2]
OPENINGS = "experiments/checkpoint_strength/dataset/dataset_le4capture_v3_200.json"
DATASET_HASH = "681ac741957726f17d3e29241330f76d4d0da80f7dbfd2e05605736ac68d0a9b"
CHECKPOINTS = (1, *range(50, 601, 50))
CONDITIONS = ("puct64-mcts64", "policy-mcts64")
EXCLUDED_CURVES = ["puct128-mcts128", "greedy128", "puct64-mcts128"]
SOURCE_RUN = "a100-fresh-64-200-20261003"
SEED = 20261003


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def case_specs(*, historical=False):
    """Current experiments; historical scope only validates the original exports."""
    shared = {"games": 400, "seed": SEED, "openings": OPENINGS}

    def greedy(step):
        return {"id": f"greedy64/c{step:06d}", "kind": "greedy_eoh",
                "config": {**shared, "checkpoint": step, "simulations": 64, "workers": 8}}

    def mcts(condition, step):
        budget = 128 if condition == "puct64-mcts128" else 64
        return {"id": f"{condition}/c{step:06d}", "kind": "neural_vs_mcts", "config": {
            **shared, "checkpoint": step, "simulations": 0 if condition == "policy-mcts64" else 64,
            "mcts_simulations": None if condition == "puct64-mcts64" else budget,
            "neural_mode": "policy" if condition == "policy-mcts64" else "puct", "workers": 8}}

    cases = [greedy(600), mcts("puct64-mcts64", 600), mcts("puct64-mcts128", 600), mcts("policy-mcts64", 600)]
    cases += [{"id": f"head-to-head/{a}-{b}", "kind": "head_to_head", "config": {
        **shared, "checkpoints": [a, b], "simulations": 64}} for a, b in ((50, 350), (350, 600))]
    cases += [{"id": f"mcts-vs-greedy/{budget}", "kind": "mcts_vs_greedy_eoh", "config": {
        **shared, "simulations": budget}} for budget in (64, 128)]
    cases += [greedy(step) for step in CHECKPOINTS if step != 600]
    conditions = ("puct64-mcts64", "puct64-mcts128", "policy-mcts64") if historical else CONDITIONS
    cases += [mcts(condition, step) for condition in conditions for step in CHECKPOINTS if step != 600]
    return cases


def validate_plan(plan):
    def normalized_cases(cases):
        return [{**case, "config": {**case["config"], "openings": str(
            resolve_dataset_path(case["config"]["openings"]))}} for case in cases]

    historical = plan["schema_version"] == 1
    expected = case_specs(historical=historical)
    excluded = ["puct128-mcts128", "greedy128"] if historical else EXCLUDED_CURVES
    if (plan["schema_version"] not in (1, 2)
            or normalized_cases(plan["cases"]) != normalized_cases(expected)
            or resolve_dataset_path(plan.get("opening_dataset", OPENINGS)) != resolve_dataset_path(OPENINGS)
            or plan["opening_dataset_sha256"] != DATASET_HASH
            or plan["source_run"] != SOURCE_RUN or plan["seed"] != SEED
            or plan["total_games"] != sum(case["config"]["games"] for case in expected)
            or plan["total_cases"] != len(expected)
            or plan["excluded_curves"] != excluded
            or set(plan["checkpoint_sha256"]) != {str(step) for step in CHECKPOINTS}):
        raise ValueError("Suite must contain exactly the approved cases and frozen benchmark")
    return plan


def make_plan(source, evaluation_name):
    from experiments.checkpoint_strength.greedy_progression import SOURCES
    source = Path(source)
    training = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    if training["architecture_version"] != "ayo_mlp_185863_v1":
        raise ValueError("The selected suite requires the original MLP")
    files = set(SOURCES) | {
        "Algorithms/ayo_mcts.py", "experiments/checkpoint_strength/neural_vs_mcts.py",
        "experiments/checkpoint_strength/modal_le4capture_suite.py",
        "experiments/checkpoint_strength/le4capture_recovery.py", "experiments/checkpoint_strength/mcts_baseline.py"}
    cases = case_specs()
    plan = {"schema_version": 2, "experiment": "original_mlp_le4capture_v3", "evaluation_name": evaluation_name,
            "source_run": SOURCE_RUN, "architecture": training["architecture_version"], "game": training["game"],
            "uct_c": training["settings"]["uct_c"], "source_manifest_sha256": sha(source / "manifest.json"),
            "checkpoint_sha256": {str(step): sha(source / f"inference-checkpoints/step-{step:06d}.npz") for step in CHECKPOINTS},
            "opening_dataset": OPENINGS, "opening_dataset_sha256": DATASET_HASH,
            "opening_dataset_file_sha256": sha(ROOT / OPENINGS), "seed": SEED,
            "cases": cases, "total_cases": len(cases), "total_games": sum(case["config"]["games"] for case in cases),
            "excluded_curves": EXCLUDED_CURVES, "max_simultaneous_experiments": 1,
            "code_sha256": {name: sha(ROOT / name) for name in sorted(files)},
            "verification_code_sha256": sha(Path(__file__))}
    return validate_plan(plan)


def selected_case(plan, case_id):
    return next(case for case in plan["cases"] if case["id"] == case_id)


def case_files(case):
    if case["kind"] == "mcts_vs_greedy_eoh":
        budget = case["config"]["simulations"]
        return "manifest.json", f"{budget}/results.json", f"{budget}/games.jsonl"
    return "results.manifest.json", "results.json", "results.games.jsonl"


def case_policies(case):
    kind, config = case["kind"], case["config"]
    if kind == "greedy_eoh":
        return "ALPHAZERO", "OPPONENT"
    if kind == "neural_vs_mcts":
        return f"C{config['checkpoint']}", "MCTS"
    if kind == "head_to_head":
        return tuple(f"C{step}" for step in config["checkpoints"])
    return "MCTS", "GREEDY_HSTAR"


def validate_case(folder, case, plan, *, source_root=ROOT):
    """Reconstruct the frozen openings and all outcomes from full legal histories."""
    from experiments.alpha_zero import offline_evaluate as offline
    from experiments.agent_benchmark.random_vs_greedy_hstar import summarize
    from experiments.checkpoint_strength.head_to_head import paired_score_interval
    from experiments.checkpoint_strength.opening_dataset import load_dataset
    import numpy as np

    folder = Path(folder)
    manifest_name, report_name, games_name = case_files(case)
    manifest = json.loads((folder / manifest_name).read_text(encoding="utf-8"))
    report = json.loads((folder / report_name).read_text(encoding="utf-8"))
    records = [json.loads(line) for line in (folder / games_name).read_text(encoding="utf-8").splitlines() if line]
    if (manifest["status"] != "completed" or report["status"] != "completed"
            or manifest["completed_games"] != 400 or len(records) != 400 or report["games"] != 400):
        raise ValueError(f"Incomplete case: {case['id']}")
    config, kind = case["config"], case["kind"]
    if kind != "mcts_vs_greedy_eoh":
        if manifest["config"] != config:
            raise ValueError("Case configuration differs from the frozen plan")
        if manifest["run_manifest_sha256"] != plan["source_manifest_sha256"]:
            raise ValueError("Training source manifest differs")
        checkpoint_entries = [manifest["checkpoint"]] if kind == "greedy_eoh" else manifest["checkpoints"]
        expected_steps = config.get("checkpoints", [config.get("checkpoint")])
        if [entry["learner_round"] for entry in checkpoint_entries] != expected_steps:
            raise ValueError("Incorrect checkpoint assignment")
        for entry in checkpoint_entries:
            if entry["checkpoint_sha256"] != plan["checkpoint_sha256"][str(entry["learner_round"])]:
                raise ValueError("Checkpoint export differs")
        if sha(folder / report_name) != manifest["summary_sha256"] or sha(folder / games_name) != manifest["games_sha256"]:
            raise ValueError("Result artifact hash differs")
    elif (manifest["budgets"] != [config["simulations"]] or manifest["completed_budgets"] != [config["simulations"]]
          or manifest["seed"] != SEED or manifest["uct_c"] != plan["uct_c"]):
        raise ValueError("Baseline budget or seed differs")
    if (str(offline.pyspiel.load_game(manifest["game"])) != str(offline.pyspiel.load_game(plan["game"]))
            or manifest["uct_c"] != plan["uct_c"]):
        raise ValueError("Rules or exploration constant differs")
    for name, expected in manifest["code_sha256"].items():
        if expected != plan["code_sha256"].get(name) or sha(Path(source_root) / name) != expected:
            raise ValueError("Evaluation source mismatch: " + name)
    game = offline.pyspiel.load_game(plan["game"])
    dataset = load_dataset(plan["opening_dataset"], game)
    provenance = report["opening_dataset"]
    if (dataset.content_sha256 != DATASET_HASH or provenance["content_sha256"] != DATASET_HASH
            or provenance["file_sha256"] != plan["opening_dataset_file_sha256"] or provenance["opening_count"] != 200):
        raise ValueError("Opening book differs")
    if [record["game_id"] for record in records] != list(range(400)):
        raise ValueError("Missing, unordered or duplicate game IDs")
    policies = case_policies(case)
    margins = {name: defaultdict(list) for name in policies}
    actions = 0
    for index, record in enumerate(records):
        opening = dataset.openings[index // 2]
        assignment = policies if index % 2 == 0 else policies[::-1]
        if (record["opening_pair_id"] != index // 2 or record["opening_id"] != opening.opening_id
                or record["opening_actions"] != list(opening.actions)
                or record["opening_state_sha256"] != opening.state_sha256 or record["seed"] != SEED + index // 2
                or (record["player_0_policy"], record["player_1_policy"]) != assignment):
            raise ValueError("Opening, seed, or seat assignment differs")
        state = opening.new_state(game)
        if abs(state.captured[0] - state.captured[1]) > 4:
            raise ValueError("Capture-gap filter differs")
        for seat, name in enumerate(assignment):
            margins[name][int(state.captured[seat] - state.captured[1-seat])].append(record)
        for action in record["continuation_actions"]:
            if state.is_terminal() or action not in state.legal_actions():
                raise ValueError("Illegal continuation action")
            state.apply_action(action)
            actions += 1
        returns = list(state.returns())
        if (not state.is_terminal() or returns != record["returns_by_player"]
                or list(state.captured) != record["final_captured"] or state.move_number() != record["game_length"]
                or bool(state.truncated) != record["truncated"]
                or len(record["continuation_actions"]) != record["continuation_length"]):
            raise ValueError("Terminal game replay differs")
        for seat, name in enumerate(assignment):
            expected = "win" if returns[seat] > 0 else "loss" if returns[seat] < 0 else "draw"
            if record["policy_returns"][name] != returns[seat] or record["policy_outcome"][name] != expected:
                raise ValueError("Recorded policy outcome differs from terminal returns")
    summary = summarize(records)
    claimed = report["summary"]["policies"] if kind in ("greedy_eoh", "mcts_vs_greedy_eoh") else report["policies"]
    for name in policies:
        for field in ("games", "wins", "draws", "losses", "win_rate", "score_rate"):
            if summary["policies"][name][field] != claimed[name][field]:
                raise ValueError("Recomputed outcome accounting differs")
    intervals = {name: paired_score_interval(records, name, SEED) for name in policies}
    for name in policies:
        old_interval = report.get("score_intervals", {}).get(name)
        if kind == "greedy_eoh" and name == "ALPHAZERO":
            old_interval = report["score_interval"]
        if old_interval and any(not np.isclose(intervals[name][field], old_interval[field], atol=1e-12, rtol=0)
                                for field in ("score_rate", "lower", "upper")):
            raise ValueError("Paired score interval differs")
    by_seat = {name: {f"P{seat}": summarize([r for r in records if r[f"player_{seat}_policy"] == name])["policies"][name]
                     for seat in (0, 1)} for name in policies}
    by_depth = {str(depth): summarize([r for r in records if r["opening_plies"] == depth])["policies"]
                for depth in sorted({r["opening_plies"] for r in records})}
    by_margin = {name: {str(margin): summarize(rows)["policies"][name] for margin, rows in sorted(groups.items())}
                 for name, groups in margins.items()}
    verification = {"status": "passed", "case_id": case["id"], "games_replayed": 400, "actions_replayed": actions,
                    "opening_pairs": 200, "opening_dataset_sha256": DATASET_HASH,
                    "files": {name: sha(folder / name) for name in (manifest_name, report_name, games_name)},
                    "checks": ["legal_terminal_replay", "source_and_artifact_hashes", "frozen_seeds_seats_and_prefixes",
                               "terminal_returns_captures_and_lengths", "outcomes_and_rates", "paired_confidence_intervals"]}
    row = {"case_id": case["id"], "kind": kind, "config": config, "games": 400, "policies": summary["policies"],
           "score_intervals": intervals, "by_seat": by_seat, "by_opening_depth": by_depth,
           "by_starting_capture_margin": by_margin, "termination_reasons": summary["termination_reasons"]}
    return row, verification


if __name__ == "__main__":
    import argparse
    from experiments.alpha_zero.modal_common import validate_name
    parser = argparse.ArgumentParser(description="Create a new le4capture evaluation plan from current sources")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--evaluation-name", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    save_json(args.output, make_plan(args.source, validate_name(args.evaluation_name)))
