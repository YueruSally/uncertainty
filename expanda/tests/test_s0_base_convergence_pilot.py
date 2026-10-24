import unittest

from run_s0_base_convergence_pilot import selected_records


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


if __name__ == "__main__":
    unittest.main()
