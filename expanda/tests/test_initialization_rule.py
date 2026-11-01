import random
import tempfile
import unittest
from pathlib import Path

import baseline_uncertainty as base
from transformer_ppo.initialization_rule import (
    _directed_path_ranking,
    InitializationRuleController,
    path_rule_score,
)
from transformer_ppo.archive import ObjectiveNormalizer


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
        path_id, "A", "B", ["A", f"X{path_id}", "B"],
        ["rail", "rail"], [],
        cost, emission, time)


def option(candidate, probability, lateness):
    return base.ReliablePathOption(candidate, probability, lateness, {})


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
            try:
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
                self.assertEqual(
                    [item.path.path_id for item in selected],
                    [2, 3, 1][:k],
                )
            finally:
                controller.close()

    def test_joint_balanced_uses_all_five_path_signals(self):
        batch = base.Batch(0, "A", "B", 10.0, 0.0, 10.0)
        cheap_unreliable = path(1, 1.0, 1.0, 20.0)
        balanced = path(2, 2.0, 2.0, 4.0)
        clean_slow = path(3, 10.0, 10.0, 10.0)
        invalid_but_cheap = path(4, 0.0, 0.0, 0.0)
        ranked, effective = _directed_path_ranking(
            [cheap_unreliable, balanced, clean_slow, invalid_but_cheap], batch,
            [
                option(cheap_unreliable, 0.0, 20.0),
                option(balanced, 1.0, 0.0),
                option(clean_slow, 0.5, 5.0),
            ],
            "joint_balanced", 0)
        self.assertEqual(effective, "joint_balanced")
        self.assertEqual(ranked[0]["path"].path_id, 2)
        self.assertEqual(len(ranked[0]["scaled"]), 5)
        self.assertTrue(ranked[0]["structurally_feasible"])
        self.assertFalse(ranked[-1]["structurally_feasible"])

    def test_diverse_cycle_preserves_base_rng_and_rotates_profiles(self):
        batch = base.Batch(0, "A", "B", 10.0, 0.0, 10.0)
        candidates = [
            path(1, 1.0, 9.0, 9.0),
            path(2, 9.0, 1.0, 8.0),
            path(3, 5.0, 5.0, 1.0),
        ]
        options = {
            ("A", "B", 0): [
                option(candidates[0], 0.2, 8.0),
                option(candidates[1], 0.4, 5.0),
                option(candidates[2], 1.0, 0.0),
            ]
        }
        with tempfile.TemporaryDirectory() as temporary:
            controller = InitializationRuleController(
                Path(temporary), "run", "scenario", 100,
                normalizer=None, reliable_options=options,
                profile="diverse_cycle", method="rule",
                instance={}, seeds={})
            try:
                random.seed(13)
                before = random.getstate()
                for _ in range(4):
                    controller.initial_individual(
                        [batch], {("A", "B"): candidates}, max_paths=3)
                after_rule = random.getstate()
                random.setstate(before)
                for _ in range(4):
                    k = random.randint(1, 3)
                    random.sample(candidates, k)
                    [random.random() for _ in range(k)]
                self.assertEqual(after_rule, random.getstate())
                self.assertEqual(
                    set(controller.profile_counts),
                    {"joint_balanced", "cost", "emission",
                     "time_reliability"})
            finally:
                controller.close()

    def test_half_balanced_profile_directs_four_of_eight_individuals(self):
        batch = base.Batch(0, "A", "B", 10.0, 0.0, 10.0)
        candidates = [
            path(1, 1.0, 9.0, 9.0),
            path(2, 9.0, 1.0, 8.0),
            path(3, 5.0, 5.0, 1.0),
        ]
        options = {
            ("A", "B", 0): [
                option(candidates[0], 0.2, 8.0),
                option(candidates[1], 0.4, 5.0),
                option(candidates[2], 1.0, 0.0),
            ]
        }
        with tempfile.TemporaryDirectory() as temporary:
            controller = InitializationRuleController(
                Path(temporary), "run", "scenario", 100,
                normalizer=None, reliable_options=options,
                profile="joint_balanced_p050", method="rule",
                instance={}, seeds={})
            try:
                random.seed(17)
                before = random.getstate()
                for _ in range(8):
                    controller.initial_individual(
                        [batch], {("A", "B"): candidates}, max_paths=3)
                after_rule = random.getstate()
                random.setstate(before)
                for _ in range(8):
                    k = random.randint(1, 3)
                    random.sample(candidates, k)
                    [random.random() for _ in range(k)]
                self.assertEqual(after_rule, random.getstate())
                self.assertEqual(
                    controller.profile_counts["joint_balanced"], 4)
                self.assertEqual(
                    controller.profile_counts["base_passthrough"], 4)
            finally:
                controller.close()

    def test_hv_gate_selects_better_paired_initial_population(self):
        batch = base.Batch(0, "A", "B", 10.0, 0.0, 10.0)
        candidates = [
            path(1, 1.0, 9.0, 9.0),
            path(2, 9.0, 1.0, 8.0),
            path(3, 5.0, 5.0, 1.0),
        ]
        options = {
            ("A", "B", 0): [
                option(candidates[0], 0.2, 8.0),
                option(candidates[1], 0.4, 5.0),
                option(candidates[2], 1.0, 0.0),
            ]
        }
        normalizer = ObjectiveNormalizer(
            (0.0, 0.0, 0.0), (10.0, 10.0, 10.0))
        with tempfile.TemporaryDirectory() as temporary:
            controller = InitializationRuleController(
                Path(temporary), "run", "scenario", 100,
                normalizer=normalizer, reliable_options=options,
                profile="joint_balanced_hv_gate", method="rule",
                instance={}, seeds={})
            try:
                random.seed(19)
                directed = controller.initial_individual(
                    [batch], {("A", "B"): candidates}, max_paths=3)
                shadow = controller.initialization_counterfactual(directed)
                directed.objectives = (1.0, 1.0, 1.0)
                directed.feasible = True
                shadow.objectives = (5.0, 5.0, 5.0)
                shadow.feasible = True
                selected = controller.select_initial_population([directed])
                self.assertIs(selected[0], directed)
                self.assertEqual(controller.gate_selected, "directed")
                self.assertGreater(
                    controller.gate_directed_hv, controller.gate_base_hv)
                self.assertEqual(
                    controller.counterfactual_initialization_evaluations, 1)
            finally:
                controller.close()


if __name__ == "__main__":
    unittest.main()
