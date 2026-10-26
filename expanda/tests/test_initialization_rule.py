import random
import tempfile
import unittest
from pathlib import Path

import baseline_uncertainty as base
from transformer_ppo.initialization_rule import (
    InitializationRuleController,
    path_rule_score,
)


NORMALIZATION = {
    "source": "training-only",
    "configuration": "S0",
    "features": [
        "base_cost_per_teu", "base_emission_per_teu", "base_travel_time_h"],
    "minimum": [0.0, 0.0, 0.0],
    "maximum": [10.0, 100.0, 20.0],
    "test_instances_used": False,
}


def path(path_id, cost, emission, time):
    return base.Path(
        path_id, "A", "B", ["A", "B"], ["rail"], [],
        cost, emission, time)


class InitializationRuleTest(unittest.TestCase):
    def test_equal_weight_normalized_score(self):
        candidate = path(1, 5.0, 25.0, 20.0)
        self.assertAlmostEqual(path_rule_score(candidate, NORMALIZATION), 1.75)

    def test_selects_lowest_paths_and_preserves_base_random_stream(self):
        batch = base.Batch(0, "A", "B", 1.0, 0.0, 10.0)
        candidates = [
            path(1, 9.0, 90.0, 18.0),
            path(2, 1.0, 10.0, 2.0),
            path(3, 4.0, 20.0, 4.0),
        ]
        with tempfile.TemporaryDirectory() as temporary:
            controller = InitializationRuleController(
                Path(temporary), "run", "scenario", 100,
                normalizer=None, rule_normalization=NORMALIZATION,
                method="rule", instance={}, seeds={})
            random.seed(7)
            before = random.getstate()
            individual = controller.initial_individual(
                [batch], {("A", "B"): candidates}, max_paths=3)
            after_rule = random.getstate()
            random.setstate(before)
            k = random.randint(1, 3)
            random.sample(candidates, k)
            [random.random() for _ in range(k)]
            after_base = random.getstate()
            self.assertEqual(after_rule, after_base)
            selected = individual.od_allocations[("A", "B", 0)]
            self.assertEqual([item.path.path_id for item in selected], [2, 3, 1][:k])
            controller.close()


if __name__ == "__main__":
    unittest.main()
