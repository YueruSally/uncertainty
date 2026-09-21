import unittest
from unittest.mock import Mock

import numpy as np

from learning_mutation import MutationTarget
from learning_policy import (EligibleRuleLocationPolicy, LearningLocationPolicy,
                             RandomLocationPolicy)


def candidate(probability, eligible, batch_id):
    return {
        "target": MutationTarget(batch_id), "selection_probability": probability,
        "eligible": eligible, "batch_quantity": 1., "path_count": 1,
        "share": 1., "path_id": None, "path_cost_per_teu": 1.,
        "path_emission_per_teu": 1., "path_nominal_time_h": 1.,
        "from_node": None, "to_node": None, "current_mode": None,
        "alternative_mode_count": 0,
    }


class LocationPolicyTests(unittest.TestCase):
    def setUp(self):
        self.rows = [candidate(.5, False, 1), candidate(.25, True, 2),
                     candidate(.25, True, 3)]
        self.context = {
            "operator": "add", "generation": 1, "phase": "offspring",
            "feasible_before": True, "violation_before": 0.,
            "q90_cost_before": 1., "q90_emission_before": 1.,
            "q90_makespan_before": 1.,
        }

    def test_random_preserves_baseline_probabilities(self):
        decision = RandomLocationPolicy().choose(self.rows, self.context)
        self.assertEqual(decision.probabilities, [.5, .25, .25])
        self.assertAlmostEqual(sum(decision.probabilities), 1.)
        self.assertTrue(decision.exploration)

    def test_rule_conditions_baseline_on_eligibility(self):
        rng = Mock()
        rng.random.return_value = .5
        rng.choices.return_value = [1]
        policy = EligibleRuleLocationPolicy(epsilon=.1, rng=rng)
        decision = policy.choose(self.rows, self.context)
        for actual, expected in zip(decision.probabilities, [.05, .475, .475]):
            self.assertAlmostEqual(actual, expected)
        self.assertIn(decision.chosen_index, (1, 2))
        self.assertFalse(decision.metadata["eligible_fallback"])
        self.assertFalse(decision.exploration)

    def test_rule_falls_back_when_no_target_is_eligible(self):
        rows = [candidate(.75, False, 1), candidate(.25, False, 2)]
        decision = EligibleRuleLocationPolicy(epsilon=.1).choose(rows, self.context)
        self.assertEqual(decision.probabilities, [.75, .25])
        self.assertTrue(decision.metadata["eligible_fallback"])

    def test_learning_logs_epsilon_mixture_probability(self):
        policy = LearningLocationPolicy.__new__(LearningLocationPolicy)
        policy.epsilon = .1
        policy.model_path = None
        policy.artifact = {}
        policy.rng = Mock()
        policy.rng.random.return_value = .5
        policy.rng.choices.return_value = [2]
        policy.pipeline = Mock()
        policy.pipeline.classes_ = np.array([0, 1])
        policy.pipeline.predict_proba.return_value = np.array([[.8, .2], [.1, .9]])
        decision = policy.choose(self.rows, self.context)
        self.assertEqual(decision.chosen_index, 2)
        self.assertAlmostEqual(decision.probabilities[0], .05)
        self.assertAlmostEqual(decision.probabilities[1], .18863636363636366)
        self.assertAlmostEqual(decision.probabilities[2], .7613636363636364)
        self.assertAlmostEqual(sum(decision.probabilities), 1.)
        self.assertFalse(decision.exploration)
        self.assertIsNone(decision.scores[0])
        self.assertAlmostEqual(decision.scores[2], .9)
        self.assertAlmostEqual(decision.metadata["score_weight_mass"], .275)


if __name__ == "__main__":
    unittest.main()
