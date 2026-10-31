import unittest

from audit_m_operator_interventions import _paired_percent, _spread_events


class MOperatorAuditTests(unittest.TestCase):
    def test_spread_events_is_outcome_independent_and_honours_generation(self):
        rows = [
            {
                "decision_kind": "base_mixture_choice",
                "mixture_event_index": event,
                "generation": generation,
                "parent_id": f"parent-{event}",
                "final_hypervolume": 1000 - event,
            }
            for event, generation in ((1, 2), (2, 10), (3, 11), (4, 15))
        ]
        selected = _spread_events(rows, 2, 10)
        self.assertEqual(
            [row["mixture_event_index"] for row in selected], [2, 4])

    def test_spread_events_rejects_insufficient_late_states(self):
        with self.assertRaisesRegex(ValueError, "only 0 usable"):
            _spread_events([], 2, 10)

    def test_paired_percent(self):
        self.assertAlmostEqual(_paired_percent(0.5, 0.55), 10.0)
        self.assertIsNone(_paired_percent(0.0, 0.55))


if __name__ == "__main__":
    unittest.main()
