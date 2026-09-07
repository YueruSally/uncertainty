import unittest
from pathlib import Path

import numpy as np

import baseline_uncertainty as base
import audit_ccp_stage_c as audit
import run_ev_ccp_oos_pilot as pilot


class CorrectedStageCSemanticsTests(unittest.TestCase):
    def test_stage_c_q90_is_exact_production_order_statistic(self):
        values = list(range(1, 11))
        summary = audit.objective_summary(values, .90)
        self.assertEqual(summary["q90_empirical_order_statistic"], 9.0)
        self.assertEqual(
            summary["q90_empirical_order_statistic"],
            base.empirical_ccp_quantile(np.asarray(values), .90))

    def test_old_numpy_higher_definition_is_not_reintroduced(self):
        values = np.arange(1.0, 5001.0)
        self.assertEqual(pilot.summarise(values)["q90"], 4500.0)
        self.assertNotEqual(
            pilot.summarise(values)["q90"],
            float(np.quantile(values, .90, method="higher")))

    def test_delta_reports_absolute_and_relative_change(self):
        self.assertEqual(audit.delta(110.0, 100.0), {
            "absolute": 10.0, "relative_fraction": .1})

    def test_future_pilot_source_does_not_label_punctuality_as_reliability(self):
        source = Path(pilot.__file__).read_text(encoding="utf-8")
        self.assertNotIn("oos_reliable_at_alpha", source)
        self.assertNotIn('"oos_reliable_counts"', source)
        self.assertIn("punctuality_diagnostic", source)


if __name__ == "__main__":
    unittest.main()
