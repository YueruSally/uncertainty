import unittest

from build_m_downstream_gate_dataset import (
    _paired_percent, _pearson, _progress_percent,
)


class MPlusThreeDatasetTests(unittest.TestCase):
    def test_progress_percent_uses_matched_archive_hv(self):
        reference = {7: {"archive_hv": 0.5}}
        candidate = {7: {"archive_hv": 0.55}}
        self.assertAlmostEqual(
            _progress_percent(reference, candidate, 7), 10.0)

    def test_progress_percent_requires_requested_generation(self):
        self.assertIsNone(
            _progress_percent(
                {7: {"archive_hv": 0.5}},
                {8: {"archive_hv": 0.6}},
                7,
            )
        )

    def test_paired_percent_rejects_zero_reference(self):
        self.assertIsNone(_paired_percent(0.0, 1.0))

    def test_pearson_reports_exact_linear_relation(self):
        self.assertAlmostEqual(_pearson([1, 2, 3], [2, 4, 6]), 1.0)


if __name__ == "__main__":
    unittest.main()
