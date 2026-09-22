from pathlib import Path
import random
import tempfile
import unittest

import joblib
import numpy as np

from learning_policy import SegmentObjectiveMixturePolicy


class ConstantPipeline:
    classes_ = np.array([0, 1])

    def __init__(self, probabilities):
        self.probabilities = probabilities

    def predict_proba(self, frame):
        values = np.asarray(self.probabilities[:len(frame)], dtype=float)
        return np.column_stack([1.0 - values, values])


def candidate(eligible=True, probability=.5):
    return {
        "selection_probability": probability, "eligible": eligible,
        "batch_id": 1, "allocation_index": 0, "arc_index": 0, "path_id": 0,
        "batch_quantity": 1., "path_count": 1, "share": 1.,
        "path_cost_per_teu": 1., "path_emission_per_teu": 1.,
        "path_nominal_time_h": 1., "alternative_mode_count": 1,
        "from_node": "A", "to_node": "B", "current_mode": "road",
    }


class SegmentLearningPolicyTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.model = Path(self.tmp.name) / "model.joblib"
        heads = {
            name: ConstantPipeline([.8, .2]) for name in
            ("cost_improved", "emission_improved", "makespan_improved")}
        heads["capacity_worsened"] = ConstantPipeline([.1, .9])
        joblib.dump({
            "schema_version": 2, "deployment_status": "pilot_only",
            "pipelines": heads, "policy": {
                "guided_operator": "mode",
                "objective_heads": ["cost_improved", "emission_improved", "makespan_improved"],
                "risk_head": "capacity_worsened",
            }}, self.model)

    def tearDown(self):
        self.tmp.cleanup()

    def test_mode_uses_objective_benefit_and_capacity_risk(self):
        policy = SegmentObjectiveMixturePolicy(self.model, epsilon=0., rng=random.Random(1))
        decision = policy.choose([candidate(), candidate()], {
            "operator": "mode", "candidate_count": 2})
        self.assertTrue(decision.metadata["model_applied"])
        self.assertGreater(decision.scores[0], decision.scores[1])
        self.assertGreater(decision.probabilities[0], decision.probabilities[1])
        self.assertIn(decision.metadata["objective_head"], policy.objective_heads)

    def test_non_mode_uses_rule_eligible_distribution(self):
        policy = SegmentObjectiveMixturePolicy(self.model, epsilon=0., rng=random.Random(2))
        decision = policy.choose([candidate(False), candidate(True)], {"operator": "replace"})
        self.assertFalse(decision.metadata["model_applied"])
        self.assertEqual(decision.probabilities, [0., 1.])

    def test_unpromoted_model_is_rejected(self):
        artifact = joblib.load(self.model)
        artifact["deployment_status"] = "offline_validation_only"
        joblib.dump(artifact, self.model)
        with self.assertRaisesRegex(ValueError, "promoted"):
            SegmentObjectiveMixturePolicy(self.model)


if __name__ == "__main__":
    unittest.main()
