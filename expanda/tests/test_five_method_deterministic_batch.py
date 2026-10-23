from pathlib import Path
import unittest

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from run_five_method_deterministic_batch import (  # noqa: E402
    METHODS, _analysis, _exact_sign_flip_p_value, _holm,
)


class FiveMethodBatchTests(unittest.TestCase):
    def test_exact_sign_flip_detects_uniform_positive_difference(self):
        self.assertAlmostEqual(_exact_sign_flip_p_value([1.0] * 10), 2.0 / 1024.0)

    def test_holm_is_monotone_in_sorted_p_value_order(self):
        adjusted = _holm([0.04, 0.001, 0.02, 0.5])
        self.assertEqual(adjusted, [0.08, 0.004, 0.06, 0.5])

    def test_analysis_uses_paired_test_instances_and_final_hv(self):
        rows = []
        for index in range(10):
            for method_index, method in enumerate(METHODS):
                rows.append({
                    "instance_id": f"test-S0-{index + 1:03d}",
                    "method": method,
                    "final_hypervolume": 0.5 + 0.01 * method_index,
                    "final_front_size": 5 + method_index,
                    "optimisation_seconds": 1.0,
                })
        result = _analysis(rows)
        self.assertEqual(result["descriptive"]["BASE"]["count"], 10)
        self.assertEqual(len(result["paired_vs_base"]), 4)
        self.assertEqual(result["paired_vs_base"][0]["wins"], 10)
        self.assertAlmostEqual(
            result["paired_vs_base"][0]["mean_hv_difference"], 0.01)


if __name__ == "__main__":
    unittest.main()
