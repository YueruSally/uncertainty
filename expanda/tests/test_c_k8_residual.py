import copy
import unittest

import numpy as np

from c_k8_residual import (
    gate_decisions, model_order, original_index, reorder_row, score_gate,
    simple_index)
from experiments.run_c_k8_residual_search_v1 import (
    BASE, DIRECTED, INSTANCES, METHODS, ORACLE, PREVIOUS_TRANSFORMER,
    RESIDUAL, SEEDS, SIMPLE, summarize)


def state(instance="validation-001", scores=None):
    scores = scores or (4.0, 2.0, 3.0, 1.0, 5.0, 6.0, 7.0, 8.0)
    labels = [{"hv_contribution": value}
              for value in (.1, .2, .3, .4, .25, .35, .28, .31)]
    return {
        "instance": instance, "trajectory_seed": 1, "event_index": 2,
        "candidate_features": [[float(score), float(index)]
                               for index, score in enumerate(scores)],
        "candidate_batch_tokens": [[[float(index)]] for index in range(8)],
        "candidate_fingerprints": [[f"l{index}", f"r{index}"]
                                   for index in range(8)],
        "candidate_roles": [f"r{index}" for index in range(8)],
        "labels": labels, "simple_selected_index": 3,
    }


class CResidualTests(unittest.TestCase):
    def test_simple_decision_is_recomputed_and_moved_to_zero(self):
        row = state(); ordered = reorder_row(row)
        self.assertEqual(simple_index(row), 3)
        self.assertEqual(model_order(row)[0], 3)
        self.assertEqual(ordered["candidate_features"][0][0], 1.0)
        self.assertEqual(ordered["simple_selected_index"], 0)
        self.assertEqual(original_index(row, 0), 3)
        self.assertEqual(row["candidate_features"][0][0], 4.0)

    def test_simple_tie_break_prefers_original_index_one(self):
        row = state(scores=(3, 1, 4, 1, 5, 6, 7, 8))
        row["simple_selected_index"] = 1
        self.assertEqual(simple_index(row), 1)

    def test_nonpositive_lower_bound_falls_back_to_simple(self):
        predictions = np.zeros((3, 1, 8), dtype=float)
        predictions[:, 0, 2] = (.2, -.1, .1)
        selected, _, _, conservative = gate_decisions(predictions, 2.0, 0.0)
        self.assertEqual(int(selected[0]), 0)
        self.assertLess(float(conservative[0, 2]), 0.0)
        self.assertEqual(original_index(state(), selected[0]), 3)

    def test_positive_lower_bound_switches_and_scores_residual(self):
        predictions = np.zeros((3, 1, 8), dtype=float)
        predictions[:, 0, 1] = (.11, .12, .13)
        result = score_gate([state()], predictions, 1.0, .05)
        decision = result["decisions"][0]
        self.assertEqual(decision["simple_original_index"], 3)
        self.assertEqual(decision["selected_original_index"], 0)
        self.assertAlmostEqual(decision["net_gain_vs_simple"], -.3)

    def test_summary_requires_positive_bounds_over_base_and_simple(self):
        factors = {BASE: 1.0, DIRECTED: 1.01, SIMPLE: 1.04,
                   PREVIOUS_TRANSFORMER: 1.045, ORACLE: 1.10,
                   RESIDUAL: 1.06}
        rows = []
        for instance in INSTANCES:
            for seed in SEEDS:
                for method in METHODS:
                    rows.append({
                        "instance": instance, "algorithm_seed": seed,
                        "strategy": method,
                        "final_hypervolume": factors[method],
                        "scenario_digest": "same", "pairing": {"same": True},
                        "optimisation_seconds": 1.0,
                        "real_objective_evaluation_count": 400,
                        "crossover_k8_hidden_candidate_objective_evaluations": (
                            140 if method == ORACLE else 0),
                    })
        result = summarize(copy.deepcopy(rows))
        self.assertTrue(result["development_search_pass"])
        self.assertGreater(result["methods"][RESIDUAL]["vs_base"]["mean"], 0)
        self.assertGreater(result["methods"][RESIDUAL]["vs_simple"]["mean"], 0)


if __name__ == "__main__":
    unittest.main()
