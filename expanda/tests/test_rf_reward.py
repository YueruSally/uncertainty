import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import joblib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from transformer_ppo.config import OPERATORS
from transformer_ppo.observations import CANDIDATE_FEATURES, GLOBAL_FEATURES, PARENT_FEATURES
from transformer_ppo.rf_reward import (
    FEATURE_NAMES, RFRewardCollectionPolicy, RFRewardPolicy, feature_values,
)


def observation(candidate_first=(1.0, 2.0)):
    candidates = []
    for value in candidate_first:
        row = [0.0] * len(CANDIDATE_FEATURES)
        row[0] = value
        row[CANDIDATE_FEATURES.index("p_rule")] = 0.5
        candidates.append(row)
    return {
        "parent": [[1.0] * len(PARENT_FEATURES), [2.0] * len(PARENT_FEATURES)],
        "candidates": candidates,
        "eligible": [True] * len(candidates),
        "operator_id": 0,
        "global": [0.0] * len(GLOBAL_FEATURES),
    }


def candidates():
    return [
        {"eligible": True, "selection_probability": 0.5, "p_rule": 0.5},
        {"eligible": True, "selection_probability": 0.5, "p_rule": 0.5},
    ]


class FirstColumnModel:
    def predict(self, matrix):
        return np.asarray(matrix)[:, 0]


class RFFeatureTests(unittest.TestCase):
    def test_fixed_feature_schema_is_finite_and_contains_rule_probability(self):
        values = feature_values(observation(), 0)
        self.assertEqual(len(values), len(FEATURE_NAMES))
        self.assertTrue(np.all(np.isfinite(values)))
        self.assertIn("action_p_rule", FEATURE_NAMES)
        self.assertEqual(len([x for x in FEATURE_NAMES if x.startswith("operator_")]),
                         len(OPERATORS))

    def test_collection_logs_only_the_executed_action_with_exact_propensity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            policy = RFRewardCollectionPolicy(root, policy_seed=7, uniform_mixture=.5)
            policy.observation_provider = lambda *_: observation()
            decision = policy.choose(candidates(), {})
            policy.record_sample({
                "event_id": "e1", "instance_id": "train-S0-001",
                "instance_digest": "digest", "operator": "add", "generation": 0,
                "phase": "offspring", "reward": .25, "reward_hv_delta": .0025,
                "reward_archive_hv_before": .1, "reward_archive_hv_after": .1025,
                "selection_probability": decision.probabilities[decision.chosen_index],
                "mutation_success": False, "repair_success": False,
                "mutation_reverted": True, "decision_changed": False,
                "effective_mutation": False, "feasible_after": False,
                "ccp_evaluation_count": 11,
            })
            policy.observe_outcome(.25)
            policy.finish()
            rows = [json.loads(line) for line in
                    (root / "rf_reward_samples.jsonl").read_text().splitlines()]
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["candidate_index"], decision.chosen_index)
            self.assertAlmostEqual(rows[0]["logging_propensity"], .5)
            self.assertFalse(rows[0]["effective_mutation"])
            self.assertFalse(rows[0]["repair_success"])

    def test_frozen_policy_uses_reward_argmax_with_rule_tie_break(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rf.joblib"
            joblib.dump({
                "schema_version": 1,
                "target": "ppo_immediate_mutation_reward",
                "feature_names": FEATURE_NAMES,
                "model": FirstColumnModel(),
                "training_data_digest": "d",
                "rf_seed": 1,
            }, path)
            policy = RFRewardPolicy(path)
            policy.observation_provider = lambda *_: observation()
            decision = policy.choose(candidates(), {})
            self.assertEqual(decision.chosen_index, 1)
            self.assertEqual(decision.probabilities, [0.0, 1.0])


if __name__ == "__main__":
    unittest.main()
