import unittest

from evaluate_m_three_method_oos import _summary


class MThreeMethodOOSTest(unittest.TestCase):
    def test_paired_difference_summary_uses_matched_rows(self):
        rows = [
            {"instance": "A", "difference": 1.0},
            {"instance": "A", "difference": -0.5},
            {"instance": "B", "difference": 2.5},
        ]
        result = _summary(rows, "difference")
        self.assertEqual(result["run_count"], 3)
        self.assertAlmostEqual(result["mean"], 1.0)
        self.assertEqual(result["wins"], 2)
        self.assertEqual(result["losses"], 1)
        self.assertEqual(len(result["per_instance"]), 2)


if __name__ == "__main__":
    unittest.main()
