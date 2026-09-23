"""Offline API checks against the vendored, pinned official EoH package."""
import random
import unittest

from eoh.config import EoHConfig
from eoh.eoh.eoh import population_management
from eoh.eoh.evolution import Evolution, parent_selection


class PinnedEoHContractTest(unittest.TestCase):
  def test_all_five_paper_operators_have_prompts(self):
    evo = object.__new__(Evolution)
    evo.task = "Score Ayo states from the fixed player perspective."
    evo.template = "def evaluate_state(state_view, player_id):\n    return 0.0"
    evo._template_kind = "function"
    population = [{"algorithm": "simple idea", "code": evo.template}]
    self.assertIn("one sentence", evo._build_prompt("i1"))
    for op in ("e1", "e2"):
      self.assertTrue(evo._build_prompt(op, population))
    for op in ("m1", "m2", "m3"):
      self.assertTrue(evo._build_prompt(op, population[0]))
    self.assertIn("simplify", evo._build_prompt("m3", population[0]))

  def test_rank_parent_selection_and_elite_manager_contract(self):
    population = [{"algorithm": str(i), "code": str(i), "objective": float(i)} for i in range(5)]
    random.seed(2024)
    parents = parent_selection(population, 20)
    self.assertEqual(len(parents), 20)
    self.assertTrue(all(parent in population for parent in parents))
    merged = population + [{"algorithm": "duplicate score", "code": "different", "objective": 1.0},
                           {"algorithm": "worse", "code": "worse", "objective": 8.0}]
    elites = population_management(merged, 5)
    self.assertEqual([item["objective"] for item in elites], [0.0, 1.0, 2.0, 3.0, 4.0])
    self.assertEqual(sum(item["objective"] == 1.0 for item in elites), 1)

  def test_m3_can_be_enabled_in_a_pinned_config(self):
    ops = ["e1", "e2", "m1", "m2", "m3"]
    cfg = EoHConfig(pop_size=5, n_pop=3, operators=ops)
    self.assertEqual(cfg.operators, ops)
    self.assertEqual(cfg.operator_weights, [1.0] * 5)


if __name__ == "__main__":
  unittest.main()
