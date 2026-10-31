import unittest

from build_m_continuation_mean_dataset import _aggregate_records


class MContinuationMeanDatasetTests(unittest.TestCase):
    def test_mean_std_and_risk_adjusted_label(self):
        rows = [
            {"continuation_seed": seed, "final_percent": value}
            for seed, value in zip((1, 2, 3, 4), (2.0, 4.0, 6.0, 8.0))
        ]
        result = _aggregate_records(rows, 1.0)
        self.assertEqual(result["continuation_count"], 4)
        self.assertAlmostEqual(result["continuation_mean_final_percent"], 5.0)
        self.assertAlmostEqual(
            result["continuation_standard_deviation_final_percent"],
            2.23606797749979,
        )
        self.assertAlmostEqual(
            result["continuation_risk_adjusted_final_percent"],
            2.76393202250021,
        )
        self.assertEqual(result["continuation_positive_fraction"], 1.0)

    def test_duplicate_continuation_seed_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "duplicate continuation"):
            _aggregate_records([
                {"continuation_seed": 1, "final_percent": 2.0},
                {"continuation_seed": 1, "final_percent": 3.0},
            ], 1.0)


if __name__ == "__main__":
    unittest.main()
