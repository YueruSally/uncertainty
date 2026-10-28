import unittest

from analyze_s0_v2_component_validation import paired_summary


def _runs(values):
    return {
        ("low", index, 100 + index): {
            "final_hypervolume": value,
        }
        for index, value in enumerate(values, start=1)
    }


class AnalyzeS0V2ComponentValidationTest(unittest.TestCase):
    def test_paired_summary_reports_wins_and_losses(self):
        base = _runs([0.50, 0.60, 0.70])
        candidate = _runs([0.55, 0.60, 0.63])
        result = paired_summary("I", candidate, base)
        self.assertEqual(result["wins"], 1)
        self.assertEqual(result["ties"], 1)
        self.assertEqual(result["losses"], 1)
        self.assertAlmostEqual(result["mean_hv"], 0.5933333333333334)

    def test_rejects_unpaired_keys(self):
        with self.assertRaisesRegex(ValueError, "paired keys"):
            paired_summary("C", _runs([0.5, 0.6]), _runs([0.5]))


if __name__ == "__main__":
    unittest.main()
