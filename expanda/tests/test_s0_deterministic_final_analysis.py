import unittest

from analyze_s0_deterministic_final_test import analyze
from s0_deterministic_test_plan import build_schedule


class S0DeterministicFinalAnalysisTest(unittest.TestCase):
    def test_averages_repeats_before_instance_inference(self):
        instances = [f"final-test-S0-{index:03d}" for index in range(1, 11)]
        schedule, _ = build_schedule(instances)
        rows = []
        offsets = {
            "BASE": 0.0,
            "RULE-I": 0.005,
            "I": 0.01,
            "C": 0.02,
            "M": 0.03,
            "ICM": 0.04,
        }
        for task in schedule:
            rows.append({
                **task,
                "instance_digest": task["instance_id"],
                "final_hypervolume": (
                    0.4
                    + instances.index(task["instance_id"]) * 0.01
                    + task["algorithm_repeat"] * 0.001
                    + offsets[task["method"]]),
                "final_front_size": 10,
                "real_objective_evaluation_count": 30_000,
                "optimisation_seconds": 60.0,
            })
        plan = {
            "plan_digest": "test",
            "schedule": schedule,
            "final_test_instances": {
                "instances": [{"instance_id": value} for value in instances],
            },
        }
        result = analyze(rows, plan)
        self.assertEqual(result["raw_run_count"], 420)
        self.assertEqual(result["descriptive"]["BASE"]["run_count"], 30)
        self.assertEqual(result["descriptive"]["I"]["run_count"], 90)
        self.assertAlmostEqual(
            result["primary_paired_vs_base"][0]["mean_hv_difference"], 0.01)
        self.assertAlmostEqual(
            result["secondary_i_vs_rule_i"]["mean_hv_difference"], 0.005)


if __name__ == "__main__":
    unittest.main()
