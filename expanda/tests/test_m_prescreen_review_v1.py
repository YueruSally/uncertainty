import unittest

from experiments.run_m_prescreen_review_v1 import (
    BALANCED, INSTANCES, METHODS, MODEL_EXECUTION_COMMIT, ORACLE, SEEDS,
    SIMPLE, TRANSFORMER, summary, summarize_optimization)


class PrescreenReviewTests(unittest.TestCase):
    @staticmethod
    def make_run(instance, seed, method, gain):
        baseline = .5
        return {"instance": instance, "algorithm_seed": seed,
            "strategy": method,
            "final_hypervolume": baseline*(1+gain/100),
            "optimisation_seconds": 1.0,
            "pairing": {"instance": instance, "seed": seed},
            "k8_final_hv_audit": {
                "oracle_additional_objective_evaluations": (
                    70 if method == ORACLE else 0)}}

    def test_optimization_summary_compares_same_instances_and_seeds(self):
        gain = {BALANCED: 0.0, SIMPLE: .5, TRANSFORMER: 3.0, ORACLE: 5.0}
        runs = [self.make_run(instance, seed, method, gain[method])
                for instance in INSTANCES for seed in SEEDS
                for method in METHODS]
        result = summarize_optimization(runs)
        self.assertTrue(result["pairing_exact"])
        self.assertAlmostEqual(result["transformer_minus_simple_points"], 2.5)
        self.assertEqual(result["transformer_exceeds_greedy_oracle_instances"], [])
        self.assertEqual(len(result["table"]), 12)

    def test_summary_preserves_per_instance_means(self):
        instances = [INSTANCES[0]]*2+[INSTANCES[1]]*2
        result = summary([1, 3, -1, 1], instances)
        self.assertEqual(result["mean"], 1.0)
        self.assertEqual(result["per_instance"][0]["mean"], 2.0)
        self.assertEqual(result["per_instance"][1]["mean"], 0.0)

    def test_frozen_execution_commit_is_exact(self):
        self.assertEqual(len(MODEL_EXECUTION_COMMIT), 40)
        self.assertTrue(MODEL_EXECUTION_COMMIT.startswith("3813704"))


if __name__ == "__main__":
    unittest.main()
