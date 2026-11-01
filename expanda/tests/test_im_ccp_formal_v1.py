import unittest
from pathlib import Path

from im_ccp_formal_plan import (
    ALGORITHM_SEEDS,
    EVALUATION_BUDGET,
    GENERATIONS,
    M_MINIMUM_GENERATION,
    METHODS,
    POPULATION,
    build_schedule,
    expected_rule_i_attempts,
)
from run_im_ccp_formal_shard_v1 import _command, _smoke_tasks


class ImCcpFormalPlanTest(unittest.TestCase):
    def test_schedule_is_exactly_three_balanced_paired_shards(self):
        instances = [f"final-test-S0-{index:03d}" for index in range(1, 11)]
        schedule = build_schedule(instances)
        self.assertEqual(len(schedule), 90)
        self.assertEqual({row["method"] for row in schedule}, set(METHODS))
        self.assertEqual(
            {row["algorithm_seed"] for row in schedule},
            set(ALGORITHM_SEEDS))
        for shard in range(3):
            rows = [row for row in schedule if row["execution_shard"] == shard]
            self.assertEqual(len(rows), 30)
            self.assertEqual(len({row["algorithm_seed"] for row in rows}), 1)
            for instance in instances:
                self.assertEqual(
                    {row["method"] for row in rows
                     if row["instance_id"] == instance},
                    set(METHODS))

    def test_formal_budget_and_scaled_controls_are_frozen(self):
        self.assertEqual(POPULATION, 250)
        self.assertEqual(GENERATIONS, 200)
        self.assertEqual(EVALUATION_BUDGET, 50_250)
        self.assertEqual(M_MINIMUM_GENERATION, 100)
        self.assertEqual(expected_rule_i_attempts(), 88)

    def test_m_command_disables_policy_and_uses_balanced_replace(self):
        plan = {
            "network": {"path": "/tmp/network.xlsx"},
            "normalization": {"path": "/tmp/normalization.json"},
            "final_test_instances": {"directory": "/tmp/instances"},
            "m_infrastructure_checkpoint": {"path": "/tmp/m.pt"},
        }
        task = {
            "instance_id": "final-test-S0-001",
            "algorithm_seed": ALGORITHM_SEEDS[0],
            "method": "M_DIRECTED_BALANCED",
        }
        command = _command(task, plan, Path("/tmp/out"), "cpu")
        joined = " ".join(command)
        self.assertIn("--model-choice-probability 0", joined)
        self.assertIn("--m-forced-operator replace", joined)
        self.assertIn("--m-forced-operator-min-generation 100", joined)
        self.assertIn("--m-forced-replace-selector joint_balanced", joined)
        self.assertIn("--evaluation-budget 50250", joined)
        self.assertIn("--pop 250 --gens 200", joined)

    def test_smoke_selects_one_exact_paired_triplet(self):
        instances = [f"final-test-S0-{index:03d}" for index in range(1, 11)]
        shard = [
            row for row in build_schedule(instances)
            if row["execution_shard"] == 0]
        selected = _smoke_tasks(shard)
        self.assertEqual(len(selected), 3)
        self.assertEqual(
            {row["method"] for row in selected}, set(METHODS))
        self.assertEqual(len({row["instance_id"] for row in selected}), 1)
        self.assertEqual(len({row["algorithm_seed"] for row in selected}), 1)


if __name__ == "__main__":
    unittest.main()
