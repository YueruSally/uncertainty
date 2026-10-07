import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from transformer_ppo.archive import (
    ObjectiveNormalizer, RewardArchive, RewardConfig, exact_hv_3d,
    mutation_reward, nondominated,
)
from transformer_ppo.config import PPOConfig, SCENARIOS, TEST_CONFIGS, TRAIN_CONFIGS
from transformer_ppo.instance_catalog import _draw_quantity, _draw_release_time, _split_od_pool
from transformer_ppo.observations import (
    CANDIDATE_FEATURES, GLOBAL_FEATURES, PARENT_FEATURES,
    build_observation, candidate_tokens,
)
from transformer_ppo.statistics import formal_report, holm


class ArchiveTests(unittest.TestCase):
    def test_exact_hv_single_point(self):
        self.assertAlmostEqual(exact_hv_3d([(0.2, 0.3, 0.4)], (1., 1., 1.)),
                               .8 * .7 * .6)

    def test_archive_is_nondominated_and_deduplicated(self):
        archive = RewardArchive(ObjectiveNormalizer((0., 0., 0.), (10., 10., 10.),
                                                     reference=(1., 1., 1.)))
        first = archive.observe((2., 3., 4.), True, "a")
        duplicate = archive.observe((1., 1., 1.), True, "a")
        dominated = archive.observe((5., 6., 7.), True, "b")
        infeasible = archive.observe((1., 1., 1.), False, "c")
        self.assertGreater(first, 0.)
        self.assertEqual(duplicate, 0.)
        self.assertEqual(dominated, 0.)
        self.assertEqual(infeasible, 0.)
        self.assertEqual(len(archive.points), 1)

    def test_normalization_uses_instance_divisor(self):
        normalizer = ObjectiveNormalizer((0., 0., 0.), (10., 10., 10.)).with_divisor((2., 4., 5.))
        self.assertEqual(normalizer.transform((10., 20., 25.)), (.5, .5, .5))

    def test_reward_priority(self):
        config = RewardConfig()
        self.assertEqual(mutation_reward({"repair_success": False}, 1., config),
                         -config.repair_failure_penalty)
        self.assertEqual(mutation_reward({"repair_success": True,
            "mutation_reverted": False, "decision_changed": False}, 1., config),
            -config.no_change_penalty)
        self.assertGreater(mutation_reward({"repair_success": True,
            "mutation_reverted": False, "decision_changed": True,
            "feasible_after": True}, .005, config), 0.)


class ObservationTests(unittest.TestCase):
    def test_mask_and_operator_are_separate_from_list_position(self):
        rows = [
            {"target": {"batch_id": 2}, "eligible": False, "p_rule": 0.},
            {"target": {"batch_id": 7}, "eligible": True, "p_rule": 1.},
        ]
        values, eligible = candidate_tokens(rows, "add")
        self.assertEqual(eligible, [False, True])
        self.assertEqual(len(values[0]), len(values[1]))
        self.assertEqual(len(values[0]), len(CANDIDATE_FEATURES))
        self.assertNotIn(2., values[0])  # random/list batch IDs are not numeric features

    def test_complete_observation_matches_declared_dimensions(self):
        path = SimpleNamespace(arcs=[], base_cost_per_teu=2.,
                               base_emission_per_teu=3., base_travel_time_h=4.)
        allocation = SimpleNamespace(path=path, share=1.)
        individual = SimpleNamespace(
            od_allocations={("O", "D", 0): [allocation]}, objectives=(10., 20., 30.),
            feasible=True, rank=0)
        batch = SimpleNamespace(batch_id=0, origin="O", destination="D", quantity=5.)
        candidates = [{"target": {"batch_id": 0}, "eligible": True, "p_rule": 1.}]
        context = {"operator": "add", "K": 1, "deadline_window_alpha": 1.,
                   "generation": 0, "phase": "offspring", "feasible_before": True,
                   "violation_before": 0., "q90_cost_before": 10.,
                   "q90_emission_before": 20., "q90_makespan_before": 30.}
        observation = build_observation(
            individual, [batch], candidates, context, [individual], [],
            {"archive_size": 1, "archive_hv": .2},
            ObjectiveNormalizer((0., 0., 0.), (100., 100., 100.)), 100, 10, 5)
        self.assertEqual(len(observation["parent"][0]), len(PARENT_FEATURES))
        self.assertEqual(len(observation["candidates"][0]), len(CANDIDATE_FEATURES))
        self.assertEqual(len(observation["global"]), len(GLOBAL_FEATURES))


class CatalogTests(unittest.TestCase):
    def test_configuration_sets(self):
        self.assertEqual(len(TRAIN_CONFIGS), 9)
        self.assertEqual(len(TEST_CONFIGS), 12)
        self.assertEqual(SCENARIOS["S0"], (20, 1.0))
        self.assertEqual(SCENARIOS["S11"], (50, 2.0))

    def test_train_test_numeric_grids_are_disjoint(self):
        train_rng, test_rng = np.random.default_rng(1), np.random.default_rng(1)
        train_q = {_draw_quantity(train_rng, 81., 82., "train") for _ in range(100)}
        test_q = {_draw_quantity(test_rng, 81., 82., "test") for _ in range(100)}
        self.assertFalse(train_q & test_q)
        train_t = {_draw_release_time(train_rng, 48., 6., "train") for _ in range(100)}
        test_t = {_draw_release_time(test_rng, 48., 6., "test") for _ in range(100)}
        self.assertFalse(train_t & test_t)

    def test_od_split_is_disjoint_and_reproducible(self):
        pool = [("O", f"D{i}") for i in range(20)]
        a, b = _split_od_pool(pool, 123)
        a2, b2 = _split_od_pool(list(reversed(pool)), 123)
        self.assertEqual((a, b), (a2, b2))
        self.assertFalse(set(a) & set(b))


class ConfigTests(unittest.TestCase):
    def test_default_config_validates_and_digests(self):
        config = PPOConfig()
        config.validate()
        self.assertEqual(config.digest(), config.digest())

    def test_invalid_attention_width_fails(self):
        with self.assertRaises(ValueError):
            PPOConfig(hidden_dim=127, attention_heads=4).validate()


class StatisticsTests(unittest.TestCase):
    def test_holm_is_monotone_in_sorted_p_values(self):
        adjusted = holm([.03, .001, .02])
        self.assertEqual(adjusted[1], .003)
        self.assertGreaterEqual(adjusted[0], adjusted[2])

    @unittest.skipUnless(importlib.util.find_spec("scipy"), "SciPy not installed")
    def test_formal_report_keeps_configurations_separate(self):
        rows = []
        for configuration in TEST_CONFIGS:
            for index in range(10):
                instance = f"test-{configuration}-{index:03d}"
                for method, value in (
                        ("transformer", .8), ("mlp", .7),
                        ("rule", .6), ("random", .5)):
                    rows.append({"configuration": configuration,
                                 "instance_id": instance,
                                 "method": method, "oos_hv": value})
        report = formal_report(rows)
        self.assertEqual(len(report["families"]["primary_transformer_vs_rule"]), 12)
        self.assertEqual(
            report["families"]["primary_transformer_vs_rule"][0]["wins"], 10)

    @unittest.skipUnless(importlib.util.find_spec("scipy"), "SciPy not installed")
    def test_duplicate_method_instance_is_rejected(self):
        rows = []
        for configuration in TEST_CONFIGS:
            for index in range(10):
                instance = f"test-{configuration}-{index:03d}"
                for method in ("transformer", "mlp", "rule", "random"):
                    rows.append({"configuration": configuration,
                                 "instance_id": instance,
                                 "method": method, "oos_hv": .5})
        rows.append(dict(rows[0]))
        with self.assertRaises(ValueError):
            formal_report(rows)


if __name__ == "__main__":
    unittest.main()
