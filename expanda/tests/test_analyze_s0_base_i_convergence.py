import unittest

from analyze_s0_base_i_convergence import pair_curves, summarise


class AnalyzeS0BaseIConvergenceTests(unittest.TestCase):
    def test_pairs_same_seed_and_budget(self):
        base = [{
            "role": role, "instance_id": f"train-S0-00{index}", "repeat": "1",
            "algorithm_seed": str(6 + index),
            "real_objective_evaluation_count": "250", "archive_hv": "0.4",
            "runtime_seconds": "10.0",
        } for index, role in enumerate(("low", "median", "high"), 1)]
        learned = [{**row, "archive_hv": "0.5", "runtime_seconds": "12.0"}
                   for row in base]
        pairs = pair_curves(base, learned, (250,))
        self.assertEqual(pairs[0]["outcome"], "win")
        self.assertAlmostEqual(pairs[0]["percent_delta_vs_base"], 25.0)
        summary = summarise(pairs, (250,))
        overall = [row for row in summary if row["role"] == "overall"][0]
        self.assertEqual(overall["wins"], 3)

    def test_rejects_unpaired_seed(self):
        base = [{
            "role": "low", "instance_id": "train-S0-001", "repeat": "1",
            "algorithm_seed": "7", "real_objective_evaluation_count": "250",
            "archive_hv": "0.4", "runtime_seconds": "10.0",
        }]
        learned = [{**base[0], "algorithm_seed": "8", "archive_hv": "0.5"}]
        with self.assertRaisesRegex(ValueError, "unpaired"):
            pair_curves(base, learned, (250,))


if __name__ == "__main__":
    unittest.main()
