import unittest

from validate_m_consensus_actions import (
    _validation_summary, build_validation_manifest,
)


def _source_action(candidate_index):
    return {
        "candidate_index": candidate_index,
        "action_signature": f"replace:{candidate_index}",
        "final_percent": 0.0,
        "selected_replacement_executed": True,
    }


class MConsensusValidationTests(unittest.TestCase):
    def setUp(self):
        self.source_manifest = {
            "continuation_seeds": [32001, 32002, 32003, 32004],
            "states": [{
                "instance": "S0_002",
                "algorithm_seed": 777001,
                "event": 42,
                "generation": 12,
                "actions": [_source_action(3), _source_action(9)],
            }],
        }
        self.audit = {"states": [{
            "instance": "S0_002",
            "algorithm_seed": 777001,
            "event": 42,
            "generation": 12,
            "audited_oracle_percent": 4.5,
            "actions": [_source_action(3), _source_action(9)],
        }]}
        self.consensus = {
            "completed": True,
            "selected_state_count": 1,
            "predictions": [{
                "state_key": "S0_002:777001:42",
                "generation": 12,
                "intervene": True,
                "selected_candidate_id": 9,
            }],
        }

    def test_freezes_exact_consensus_candidate_and_new_seeds(self):
        manifest = build_validation_manifest(
            self.consensus,
            self.source_manifest,
            self.audit,
            [33001, 33002],
        )
        self.assertEqual(manifest["continuation_seeds"], [33001, 33002])
        self.assertEqual(len(manifest["states"]), 1)
        action = manifest["states"][0]["actions"][0]
        self.assertEqual(action["candidate_index"], 9)
        self.assertEqual(action["selection_role"], "top_1")

    def test_rejects_reused_continuation_seed(self):
        with self.assertRaisesRegex(ValueError, "reuse source seeds"):
            build_validation_manifest(
                self.consensus,
                self.source_manifest,
                self.audit,
                [32001, 33001],
            )

    def test_rejects_candidate_not_in_frozen_source_state(self):
        self.consensus["predictions"][0]["selected_candidate_id"] = 99
        with self.assertRaisesRegex(ValueError, "uniquely executable"):
            build_validation_manifest(
                self.consensus,
                self.source_manifest,
                self.audit,
                [33001, 33002],
            )

    def test_validation_summary_applies_predeclared_safety_thresholds(self):
        action_rows = []
        records = []
        for index in range(5):
            values = [1.5] * 8
            action_rows.append({
                "instance": f"S0_{index + 2:03d}",
                "algorithm_seed": 777001,
                "event": index + 1,
                "candidate_index": index,
                "mean_final_percent": 1.5,
                "final_percent_each": values,
                "positive_fraction": 1.0,
                "negative_fraction": 0.0,
            })
            records.extend({"final_percent": value} for value in values)
        result = _validation_summary({
            "selected_state_count": 5,
            "selected_action_count": 5,
            "continuation_seed_count": 8,
            "scenario_exact": True,
            "forced_execution_exact": True,
            "pre_intervention_state_replay_exact": True,
            "action_stability": action_rows,
            "records": records,
        })
        self.assertTrue(result["admission_pass"])
        self.assertEqual(result["paired_action_continuation_count"], 40)
        self.assertEqual(result["negative_action_mean_fraction"], 0.0)


if __name__ == "__main__":
    unittest.main()
