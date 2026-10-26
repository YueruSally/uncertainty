import tempfile
import unittest
from pathlib import Path

from run_s0_deterministic_test_shard import (
    _command,
    _frozen_updates_are_zero,
)


class S0DeterministicTestShardTest(unittest.TestCase):
    def setUp(self):
        self.task = {
            "instance_id": "final-test-S0-001",
            "algorithm_seed": 93_000_000,
            "method": "BASE",
            "policy_seed": None,
        }
        self.plan = {
            "final_test_instances": {
                "directory": "final",
            },
            "normalization": {"path": "normalization.json"},
            "rule_initialization_normalization": {"path": "rule.json"},
            "checkpoints": {
                component: {
                    "301": {"path": f"{component.lower()}301.pt"},
                }
                for component in ("I", "C", "M")
            },
        }

    def test_commands_use_only_the_frozen_method_inputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            base = _command(self.task, self.plan, output, "cuda")
            self.assertIn("BASE", base)
            self.assertNotIn("--checkpoint", base)

            rule_task = {**self.task, "method": "RULE-I"}
            rule = _command(rule_task, self.plan, output, "cuda")
            self.assertIn("--initialization-rule-normalization", rule)

            i_task = {**self.task, "method": "I", "policy_seed": 301}
            initialization = _command(i_task, self.plan, output, "cuda")
            self.assertIn("--checkpoint", initialization)
            self.assertNotIn("--c-checkpoint", initialization)

            icm_task = {**self.task, "method": "ICM", "policy_seed": 301}
            combined = _command(icm_task, self.plan, output, "cuda")
            self.assertIn("--i-checkpoint", combined)
            self.assertIn("--c-checkpoint", combined)
            self.assertIn("--m-checkpoint", combined)

    def test_frozen_policies_must_not_update(self):
        self.assertTrue(_frozen_updates_are_zero("I", {"ppo_updates": 0}))
        self.assertTrue(_frozen_updates_are_zero(
            "ICM", {"ppo_updates": {"I": 0, "C": 0, "M": 0}}))
        self.assertFalse(_frozen_updates_are_zero("M", {"ppo_updates": 1}))


if __name__ == "__main__":
    unittest.main()
