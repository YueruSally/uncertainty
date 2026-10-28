import unittest

from analyze_s0_v2_i_ablation import _choose


class AnalyzeS0V2IAblationTest(unittest.TestCase):
    def test_near_tie_uses_lower_variability_then_smaller_k(self):
        summaries = [
            {"setting": "top4", "candidate_mean_hv": 0.8000,
             "base_mean_hv": 0.78, "wins": 6, "losses": 3,
             "paired_delta_standard_deviation": 0.010},
            {"setting": "top8", "candidate_mean_hv": 0.8010,
             "base_mean_hv": 0.78, "wins": 6, "losses": 3,
             "paired_delta_standard_deviation": 0.008},
            {"setting": "top16", "candidate_mean_hv": 0.8005,
             "base_mean_hv": 0.78, "wins": 6, "losses": 3,
             "paired_delta_standard_deviation": 0.008},
            {"setting": "all", "candidate_mean_hv": 0.7990,
             "base_mean_hv": 0.78, "wins": 5, "losses": 4,
             "paired_delta_standard_deviation": 0.009},
        ]
        selected, _, eligible = _choose(
            summaries, {"top4": 4, "top8": 8, "top16": 16, "all": None},
            0.002)
        self.assertEqual(selected, "top8")
        self.assertEqual(set(eligible), {"top4", "top8", "top16"})

    def test_falls_back_to_all_when_top_k_fails_stability_guard(self):
        summaries = [
            {"setting": "top4", "candidate_mean_hv": 0.801,
             "base_mean_hv": 0.802, "wins": 3, "losses": 6,
             "paired_delta_standard_deviation": 0.001},
            {"setting": "top8", "candidate_mean_hv": 0.800,
             "base_mean_hv": 0.802, "wins": 4, "losses": 5,
             "paired_delta_standard_deviation": 0.002},
            {"setting": "top16", "candidate_mean_hv": 0.799,
             "base_mean_hv": 0.802, "wins": 4, "losses": 5,
             "paired_delta_standard_deviation": 0.003},
            {"setting": "all", "candidate_mean_hv": 0.798,
             "base_mean_hv": 0.802, "wins": 3, "losses": 6,
             "paired_delta_standard_deviation": 0.004},
        ]
        selected, reason, _ = _choose(
            summaries, {"top4": 4, "top8": 8, "top16": 16, "all": None},
            0.002)
        self.assertEqual(selected, "all")
        self.assertIn("do not force", reason)


if __name__ == "__main__":
    unittest.main()
