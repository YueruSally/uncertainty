import unittest

from run_s0_base_convergence_pilot import control_arguments, selected_records


def _selection(rows):
    return {
        "configuration": "S0",
        "instance_count": 50,
        "test_instances_used": False,
        "optimisation_outcomes_used": False,
        "selected": rows,
    }


class BaseConvergencePilotTest(unittest.TestCase):
    def test_orders_the_three_training_roles(self):
        rows = [
            {"role": "high", "instance_id": "train-S0-049"},
            {"role": "low", "instance_id": "train-S0-031"},
            {"role": "median", "instance_id": "train-S0-021"},
        ]
        self.assertEqual(
            [row["role"] for row in selected_records(_selection(rows))],
            ["low", "median", "high"],
        )

    def test_rejects_test_instance_leakage(self):
        rows = [
            {"role": "low", "instance_id": "train-S0-031"},
            {"role": "median", "instance_id": "test-S0-001"},
            {"role": "high", "instance_id": "train-S0-049"},
        ]
        with self.assertRaisesRegex(ValueError, "non-training"):
            selected_records(_selection(rows))

    def test_builds_v2_component_arguments(self):
        self.assertEqual(
            control_arguments("BASE"),
            [],
        )
        self.assertEqual(
            control_arguments("I", checkpoint="i.pt", device="cuda"),
            ["--checkpoint", "i.pt", "--device", "cuda"],
        )
        self.assertEqual(
            control_arguments(
                "I", checkpoint="i.pt", device="cuda",
                rule_normalization="rule.json", rule_top_k=8),
            [
                "--checkpoint", "i.pt", "--device", "cuda",
                "--initialization-rule-normalization", "rule.json",
                "--initialization-rule-top-k", "8",
            ],
        )
        self.assertEqual(
            control_arguments(
                "C", checkpoint="c.pt", device="cuda",
                model_choice_probability=0.5),
            [
                "--checkpoint", "c.pt", "--device", "cuda",
                "--model-choice-probability", "0.5",
            ],
        )
        self.assertEqual(
            control_arguments(
                "RULE-I", rule_normalization="rule.json"),
            ["--initialization-rule-normalization", "rule.json"],
        )


if __name__ == "__main__":
    unittest.main()
