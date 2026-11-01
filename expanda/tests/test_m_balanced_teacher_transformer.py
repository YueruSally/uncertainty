from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from train_m_balanced_teacher_transformer import _instance_key, _teacher_target
from transformer_ppo.observations import CANDIDATE_PATH_V4_FEATURES


class BalancedTeacherTargetTest(unittest.TestCase):
    def test_instance_key_accepts_rollout_and_directory_spellings(self):
        self.assertEqual(_instance_key("train-S0-011"), "S0_011")
        self.assertEqual(_instance_key("S0-011"), "S0_011")
        self.assertEqual(_instance_key("S0_011"), "S0_011")

    def test_teacher_uses_joint_quantity_weighted_balanced_score(self):
        names = list(CANDIDATE_PATH_V4_FEATURES)

        def candidate(quantity, cost, emission, time, lateness, on_time):
            row = np.zeros(len(names), dtype=float)
            values = {
                "batch_quantity": quantity,
                "replacement_delta_cost_per_teu": cost,
                "replacement_delta_emission_per_teu": emission,
                "replacement_delta_nominal_time_h": time,
                "replacement_max_lateness_h": lateness,
                "replacement_on_time_probability": on_time,
            }
            for name, value in values.items():
                row[names.index(name)] = value
            return row.tolist()

        row = {
            "observation": {
                "candidates": [
                    candidate(100, -1, -1, -1, 0, 1),
                    candidate(1, -2, -2, -2, 0, 1),
                ],
                "eligible": [True, True],
            },
            "candidates": [
                {"target": {"batch_id": 1}, "path_id": "A"},
                {"target": {"batch_id": 2}, "path_id": "B"},
            ],
        }
        selected, scores, legal = _teacher_target(row, names)
        self.assertEqual(legal, [0, 1])
        self.assertEqual(selected, 0)
        self.assertLess(scores[0], scores[1])

    def test_teacher_ignores_ineligible_candidate(self):
        names = list(CANDIDATE_PATH_V4_FEATURES)
        candidates = np.zeros((3, len(names)), dtype=float)
        candidates[:, names.index("replacement_on_time_probability")] = [1, 0, 1]
        row = {
            "observation": {
                "candidates": candidates.tolist(),
                "eligible": [True, False, True],
            },
            "candidates": [
                {"target": {"batch_id": 1}, "path_id": "A"},
                {"target": {"batch_id": 2}, "path_id": "B"},
                {"target": {"batch_id": 3}, "path_id": "C"},
            ],
        }
        selected, _, legal = _teacher_target(row, names)
        self.assertEqual(legal, [0, 2])
        self.assertIn(selected, legal)


if __name__ == "__main__":
    unittest.main()
