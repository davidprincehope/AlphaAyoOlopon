"""Run the verified minimax depth experiment with aligned utilities.

The canonical command delegates to the source-preserved rerun implementation.
Action limits are unresolved and never adjudicated from a captured-seed lead.
"""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.minimax_depth_vs_random.rerun_experiment import main


if __name__ == "__main__":
    main()
