"""Render the four progression charts when local game verification completes."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

from experiments.alpha_zero.modal_common import ROOT, validate_name


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-name", required=True)
    args = parser.parse_args()
    run = ROOT / "runs/checkpoint_strength" / validate_name(args.evaluation_name)
    output = ROOT / "visualizations/figures/mcts_learning_progression_13_checkpoints"
    deadline = time.monotonic() + 6 * 60 * 60
    status_path = run / "chart_status.json"
    def save(status, **extra):
        temporary = status_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps({"status": status, "updated_at_utc": datetime.now(timezone.utc).isoformat(),
                                        **extra}, indent=2) + "\n", encoding="utf-8")
        temporary.replace(status_path)
    save("waiting_for_verified_results", output=str(output))
    try:
        while time.monotonic() < deadline:
            verified = run / "verification.json"
            if verified.exists():
                try:
                    data = json.loads(verified.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    time.sleep(1)
                    continue
                if data.get("status") == "passed" and data.get("games_replayed") == 20800:
                    from visualizations.mcts_learning_progression import render
                    save("rendering", output=str(output))
                    render(run, output)
                    from PIL import Image
                    from xml.etree import ElementTree as ET
                    import zipfile
                    for path in output.glob("*.png"):
                        with Image.open(path) as picture:
                            if picture.size != (3300, 1950):
                                raise ValueError("Unexpected chart dimensions")
                            picture.verify()
                    for path in output.glob("*.svg"):
                        svg = ET.parse(path).getroot()
                        if len(list(svg.iter("{http://www.w3.org/2000/svg}text"))) < 20:
                            raise ValueError("SVG labels are not editable text")
                    with zipfile.ZipFile(output / "mcts_learning_progression_charts.zip") as archive:
                        if archive.testzip() is not None or len(archive.namelist()) != 11:
                            raise ValueError("Chart archive failed verification")
                    save("completed", figures=4, games=20800, export_checks="passed", output=str(output))
                    return
            time.sleep(15)
        raise TimeoutError("Verified results did not finish within six hours")
    except BaseException as exc:
        save("failed", error=repr(exc))
        raise


if __name__ == "__main__":
    main()
