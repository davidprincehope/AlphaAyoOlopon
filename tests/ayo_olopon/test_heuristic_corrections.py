"""Tests for corrected experiment integrity and outcome classification."""

from pathlib import Path
from unittest import mock
import json

from absl.testing import absltest

from experiments.heuristic import run_experiment


class HeuristicCorrectionTest(absltest.TestCase):

  def test_frozen_openings_are_valid_and_split_without_duplicates(self):
    openings = run_experiment.load_frozen_openings()
    self.assertLen(openings, 100)
    self.assertLen([item for item in openings if item["split"] == "development"], 70)
    self.assertLen([item for item in openings if item["split"] == "verification"], 30)
    self.assertLen({item["state_hash"] for item in openings}, 100)

  def test_action_limit_is_not_reported_as_natural_draw(self):
    opening = run_experiment.load_frozen_openings()[0]
    with mock.patch.object(run_experiment, "GAME_LENGTH_LIMIT", 0):
      result = run_experiment.play_game(
          opening, {0: "RAND", 1: "RAND"}, game_seed=17
      )
    self.assertFalse(result["terminal"])
    self.assertEqual(result["termination_reason"], "action_limit")
    self.assertFalse(result["natural_draw"])
    self.assertTrue(result["truncated"])
    self.assertEqual(result["action_count"], 0)

  def test_correction_outputs_are_separate_from_original_results(self):
    base = Path(run_experiment.__file__).resolve().parent
    self.assertTrue((base / "raw_results").exists())
    self.assertTrue((base / "raw_results_corrected").exists())
    self.assertTrue((base / "raw_results" / "experiment_manifest.json").exists())
    self.assertTrue((base / "raw_results_corrected" / "corrected_experiment_manifest.json").exists())
    corrected = json.loads((base / "raw_results_corrected" / "selected_verification.json").read_text())
    required = {"terminal", "termination_reason", "winner", "natural_draw", "truncated", "action_count"}
    self.assertTrue(required.issubset(corrected["games"][0]))


if __name__ == "__main__":
  absltest.main()
