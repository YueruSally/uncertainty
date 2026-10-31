import unittest

from collect_m_continuation_stability import (
    _paired_percent, _select_actions, _select_states,
)


class MutationContinuationStabilityTests(unittest.TestCase):
    def test_state_selection_enforces_oracle_and_instance_cap(self):
        audit = {"states": [
            {"instance": "S0_001", "algorithm_seed": 1, "event": 1,
             "audited_oracle_percent": 20.0, "actions": [{}]},
            {"instance": "S0_001", "algorithm_seed": 2, "event": 2,
             "audited_oracle_percent": 19.0, "actions": [{}]},
            {"instance": "S0_002", "algorithm_seed": 1, "event": 3,
             "audited_oracle_percent": 10.0, "actions": [{}]},
        ]}
        selected = _select_states(audit, 2, 5.0, 1)
        self.assertEqual(
            [row["instance"] for row in selected], ["S0_001", "S0_002"])

    def test_action_selection_uses_top_actions_and_bottom_contrast(self):
        state = {"instance": "S0_001", "algorithm_seed": 1, "event": 1,
                 "actions": [
                     {"candidate_index": index, "final_percent": value,
                      "selected_replacement_executed": True}
                     for index, value in enumerate((2.0, 9.0, -4.0, 5.0))
                 ]}
        selected = _select_actions(state, 3)
        self.assertEqual(
            [row["final_percent"] for row in selected], [9.0, 5.0, -4.0])

    def test_paired_percent(self):
        self.assertAlmostEqual(_paired_percent(0.5, 0.55), 10.0)
        self.assertIsNone(_paired_percent(0.0, 0.1))


if __name__ == "__main__":
    unittest.main()
