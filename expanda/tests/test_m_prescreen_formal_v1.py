import unittest

from experiments.run_m_prescreen_formal_v1 import (
    BALANCED, BASE, INSTANCES, OOS_SCENARIO_COUNT, SEEDS, SUPERVISED,
    frozen_methods, hierarchical_interval, summarize_optimization)
from transformer_ppo.instance_catalog import canonical_digest


PPO = "TRANSFORMER_K8_PPO_U002"
METHODS = (BASE, BALANCED, SUPERVISED, PPO)


def manifest():
    value = {"protocol": "m-prescreen-formal-method-freeze-v1",
        "supervised": {"method": SUPERVISED},
        "transformer_ppo": {"method": PPO, "ppo_update": 2},
        "training_instances": 200, "selection_instances": 20,
        "final_test_opened": False}
    value["manifest_digest"] = canonical_digest(value)
    return value


def run(instance, seed, method, gain):
    pairing = {"instance": instance, "seed": seed, "scenario": "same",
               "initial_population": [1, 2]}
    hv = .5*(1+gain/100)
    row = {"instance": instance, "algorithm_seed": seed,
        "strategy": method, "final_hypervolume": hv,
        "evaluation_budget": 400, "real_objective_evaluation_count": 400,
        "ppo_updates": 0, "training_feedback_evaluations": 0,
        "counterfactual_evaluation_count": 0,
        "candidate_audit_evaluations": 0, "stop_reason": "evaluation_budget",
        "optimisation_seconds": 1.0, "pairing": pairing}
    if method != BASE:
        row["k8_scale_search"] = {"method": method, "pool_events": 10,
            "hidden_candidate_objective_evaluations": 0}
    return row


class PrescreenFormalTests(unittest.TestCase):
    def test_frozen_methods_include_base_and_exclude_simple(self):
        self.assertEqual(frozen_methods(manifest()), METHODS)
        self.assertNotIn("K8_SIMPLE", METHODS)
        self.assertEqual(len(INSTANCES), 10)
        self.assertEqual(len(SEEDS), 10)
        self.assertEqual(OOS_SCENARIO_COUNT, 5000)

    def test_manifest_rejects_supervised_checkpoint_as_ppo(self):
        value = manifest(); value["transformer_ppo"]["ppo_update"] = 0
        identity = dict(value); identity.pop("manifest_digest")
        value["manifest_digest"] = canonical_digest(identity)
        with self.assertRaisesRegex(ValueError, "formal method freeze"):
            frozen_methods(value)

    def test_hierarchical_interval_is_deterministic(self):
        values = {"a": [1, 2, 3], "b": [4, 5, 6]}
        self.assertEqual(hierarchical_interval(values, 9),
                         hierarchical_interval(values, 9))

    def test_optimization_summary_compares_ppo_with_supervised(self):
        gains = {BASE: -2, BALANCED: 0, SUPERVISED: 3, PPO: 4}
        rows = [run(name, seed, method, gains[method])
                for name in INSTANCES for seed in SEEDS for method in METHODS]
        result = summarize_optimization(rows, METHODS)
        self.assertTrue(result["pairing_exact"])
        self.assertGreater(result["ppo_minus_supervised"]
                           ["equal_instance_mean_percent"], 0)
        self.assertAlmostEqual(result["comparisons"][PPO]["vs_balanced"]
                               ["equal_instance_mean_percent"], 4.0)


if __name__ == "__main__":
    unittest.main()
