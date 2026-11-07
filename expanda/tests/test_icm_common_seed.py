import copy
import unittest

from experiments.run_icm_common_seed_v1 import (
    BASE, C_HEURISTIC, C_TRANSFORMER, ICM_TRANSFORMER, INSTANCES,
    I_HEURISTIC, I_TRANSFORMER, METHODS, M_HEURISTIC, M_TRANSFORMER,
    SEEDS, summarize)


class ICMCommonSeedTests(unittest.TestCase):
    def test_summary_compares_all_components_on_one_grid(self):
        factors = {
            BASE: 1.0, I_HEURISTIC: 1.10, I_TRANSFORMER: 1.12,
            C_HEURISTIC: 1.03, C_TRANSFORMER: 1.05,
            M_HEURISTIC: 1.08, M_TRANSFORMER: 1.11,
            ICM_TRANSFORMER: 1.15,
        }
        rows = []
        for instance in INSTANCES:
            for seed in SEEDS:
                for method in METHODS:
                    rows.append({
                        "instance": instance, "algorithm_seed": seed,
                        "strategy": method,
                        "scenario_digest": "same", "pairing": {"same": True},
                        "final_hypervolume": factors[method],
                        "optimisation_seconds": 1.0,
                        "real_objective_evaluation_count": 400,
                    })
        result = summarize(copy.deepcopy(rows))
        self.assertEqual(result["best_single_component"], I_TRANSFORMER)
        self.assertGreater(
            result["component_comparisons"][I_TRANSFORMER]["summary"]["mean"], 0)
        self.assertGreater(result["combined_vs_best_single"]["mean"], 0)


if __name__ == "__main__":
    unittest.main()
