import unittest

from collect_m_continuation_stability import (
    _paired_percent, _select_actions, _select_actions_index_spread,
    _select_states, _select_states_stratified,
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

    def test_stratified_states_balance_instances_seeds_and_generations(self):
        states = []
        for instance in ("S0_001", "S0_002"):
            for seed in (1, 2, 3):
                for generation in (1, 4, 8, 12):
                    states.append({
                        "instance": instance,
                        "algorithm_seed": seed,
                        "event": generation * 10 + seed,
                        "generation": generation,
                        "audited_oracle_percent": 100 - generation,
                        "actions": [{}],
                    })
        selected = _select_states_stratified({"states": states}, 12)
        self.assertEqual(len(selected), 12)
        for instance in ("S0_001", "S0_002"):
            rows = [row for row in selected if row["instance"] == instance]
            self.assertEqual(len(rows), 6)
            self.assertEqual(
                {seed: sum(row["algorithm_seed"] == seed for row in rows)
                 for seed in (1, 2, 3)},
                {1: 2, 2: 2, 3: 2},
            )

    def test_index_spread_actions_ignore_final_outcomes(self):
        state = {
            "instance": "S0_001", "algorithm_seed": 1, "event": 1,
            "actions": [
                {
                    "candidate_index": index,
                    "final_percent": 1000.0 if index == 4 else -index,
                    "selected_replacement_executed": True,
                }
                for index in range(10)
            ],
        }
        selected = _select_actions_index_spread(state, 4)
        self.assertEqual(
            [row["candidate_index"] for row in selected], [0, 3, 6, 9])


if __name__ == "__main__":
    unittest.main()
