import unittest

from s0_deterministic_test_plan import (
    ALGORITHM_REPEATS,
    POLICY_SEEDS,
    build_schedule,
)


class S0DeterministicTestPlanTest(unittest.TestCase):
    def setUp(self):
        self.instances = [f"final-test-S0-{index:03d}" for index in range(1, 11)]
        self.schedule, self.loads = build_schedule(self.instances)

    def test_frozen_run_counts(self):
        counts = {}
        for row in self.schedule:
            counts[row["method"]] = counts.get(row["method"], 0) + 1
        self.assertEqual(len(self.schedule), 420)
        self.assertEqual(counts, {
            "BASE": 30,
            "RULE-I": 30,
            "I": 90,
            "C": 90,
            "M": 90,
            "ICM": 90,
        })

    def test_seeds_are_paired_without_duplicate_controls(self):
        for instance in self.instances:
            for repeat in range(1, ALGORITHM_REPEATS + 1):
                rows = [
                    row for row in self.schedule
                    if row["instance_id"] == instance
                    and row["algorithm_repeat"] == repeat
                ]
                self.assertEqual(len(rows), 14)
                self.assertEqual(len({row["algorithm_seed"] for row in rows}), 1)
                self.assertEqual(
                    len([row for row in rows if row["method"] == "BASE"]), 1)
                self.assertEqual(
                    len([row for row in rows if row["method"] == "RULE-I"]), 1)
                for method in ("I", "C", "M", "ICM"):
                    seeds = {
                        row["policy_seed"] for row in rows
                        if row["method"] == method
                    }
                    self.assertEqual(seeds, set(POLICY_SEEDS))

    def test_operational_shards_are_balanced_and_complete(self):
        self.assertEqual({row["execution_shard"] for row in self.schedule}, {0, 1})
        self.assertEqual(len({row["task_id"] for row in self.schedule}), 420)
        self.assertAlmostEqual(self.loads[0], self.loads[1])


if __name__ == "__main__":
    unittest.main()
