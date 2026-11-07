import copy
import unittest

import numpy as np

from experiments.run_i_k8_residual_search_v1 import (
    BALANCED, BASE, INSTANCES, METHODS, ORACLE, PREVIOUS_TRANSFORMER,
    RESIDUAL, SEEDS, SIMPLE, summarize)
from i_k8_residual import (
    DIRECTED_ORIGINAL_INDEX, MODEL_ORDER, gate_decisions, original_index,
    reorder_row, score_gate)


def state(instance="validation-001"):
    labels = [{"hv_contribution": value}
              for value in (.1, .3, .2, .4, .25, .35, .28, .31)]
    return {
        "instance": instance,
        "trajectory_seed": 1,
        "event_index": 2,
        "candidate_features": [[float(index)] for index in range(8)],
        "candidate_batch_tokens": [[[float(index)]] for index in range(8)],
        "candidate_fingerprints": [f"f{index}" for index in range(8)],
        "candidate_roles": [f"r{index}" for index in range(8)],
        "labels": labels,
    }


class IResidualTests(unittest.TestCase):
    def test_reorder_places_directed_at_fixed_zero_residual_index(self):
        row = state()
        ordered = reorder_row(row)
        self.assertEqual(MODEL_ORDER[0], DIRECTED_ORIGINAL_INDEX)
        self.assertEqual(ordered["candidate_features"][0], [1.0])
        self.assertEqual(ordered["candidate_roles"][0], "r1")
        self.assertEqual(original_index(0), 1)
        self.assertEqual(original_index(1), 0)
        self.assertEqual(row["candidate_features"][0], [0.0])

    def test_nonpositive_conservative_score_falls_back_to_directed(self):
        predictions = np.zeros((3, 1, 8), dtype=float)
        predictions[:, 0, 2] = (.2, -.1, .1)
        selected, _, _, conservative = gate_decisions(
            predictions, uncertainty_weight=2.0, threshold=0.0)
        self.assertEqual(int(selected[0]), 0)
        self.assertLess(float(conservative[0, 2]), 0.0)
        self.assertEqual(original_index(selected[0]), 1)

    def test_positive_lower_bound_switches_and_scores_true_residual(self):
        predictions = np.zeros((3, 1, 8), dtype=float)
        predictions[:, 0, 3] = (.11, .12, .13)
        result = score_gate(
            [state()], predictions, uncertainty_weight=1.0, threshold=.05)
        decision = result["decisions"][0]
        self.assertEqual(decision["selected_model_index"], 3)
        self.assertEqual(decision["selected_original_index"], 3)
        self.assertAlmostEqual(decision["net_gain_vs_directed"], .1)
        self.assertEqual(result["switch_fraction"], 1.0)

    def test_summary_requires_gain_over_directed_and_simple(self):
        rows = []
        factors = {
            BASE: 1.0, BALANCED: 1.10, SIMPLE: 1.11,
            PREVIOUS_TRANSFORMER: 1.105, ORACLE: 1.20, RESIDUAL: 1.12,
        }
        for instance in INSTANCES:
            for seed in SEEDS:
                for method in METHODS:
                    rows.append({
                        "instance": instance,
                        "algorithm_seed": seed,
                        "strategy": method,
                        "final_hypervolume": factors[method],
                        "scenario_digest": "same",
                        "pairing": {"same": True},
                        "optimisation_seconds": 1.0,
                        "real_objective_evaluation_count": 400,
                        "initialization_k8_hidden_candidate_objective_evaluations": (
                            140 if method == ORACLE else 0),
                    })
        result = summarize(copy.deepcopy(rows))
        self.assertTrue(result["development_search_pass"])
        self.assertGreater(
            result["methods"][RESIDUAL]["vs_directed"]["mean"], 0)
        self.assertGreater(
            result["methods"][RESIDUAL]["vs_simple"]["mean"], 0)


if __name__ == "__main__":
    unittest.main()
