import unittest

from transformer_ppo.episode import (
    resolve_combined_graph_representation, resolve_uncertainty_profile,
)


class CCPUncertaintyProfileTests(unittest.TestCase):
    def test_transport_border_profile_matches_predeclared_parameters(self):
        profile = resolve_uncertainty_profile(
            "ccp", "transport_border_saa_v1")
        self.assertEqual(profile["scenario_count"], 200)
        self.assertEqual(profile["mode_time_cv"], {
            "road": 0.15, "rail": 0.10, "water": 0.20})
        self.assertEqual(profile["mode_time_max_factor"], {
            "road": None, "rail": None, "water": None})
        self.assertEqual(profile["border_delay_cv"], {
            "road": 0.50, "rail": 0.50, "water": 0.50})
        self.assertEqual(profile["border_delay_max_factor"], 3.5)
        self.assertEqual(profile["border_mean_source"], "node_border")
        self.assertFalse(profile["border_include_customs"])

    def test_explicit_scenario_count_overrides_profile_default(self):
        profile = resolve_uncertainty_profile(
            "ccp", "transport_border_saa_v1", scenario_count=500)
        self.assertEqual(profile["scenario_count"], 500)

    def test_legacy_ccp100_keeps_one_hundred_scenarios(self):
        profile = resolve_uncertainty_profile("ccp100", "legacy")
        self.assertEqual(profile["scenario_count"], 100)
        self.assertEqual(profile["border_mean_source"], "directional_csv")

    def test_deterministic_always_uses_one_scenario(self):
        profile = resolve_uncertainty_profile(
            "deterministic", "transport_border_saa_v1", scenario_count=500)
        self.assertEqual(profile["scenario_count"], 1)

    def test_invalid_profile_inputs_are_rejected(self):
        with self.assertRaises(ValueError):
            resolve_uncertainty_profile("ccp", "missing")
        with self.assertRaises(ValueError):
            resolve_uncertainty_profile("ccp", "legacy", scenario_count=0)

    def test_component_specific_ordered_candidate_schemas_can_be_combined(self):
        self.assertEqual(
            resolve_combined_graph_representation({
                "I": "path_v2", "C": "path_v2", "M": "path_v4"}),
            "path_v2")
        with self.assertRaises(ValueError):
            resolve_combined_graph_representation({
                "I": "legacy", "M": "path_v4"})


if __name__ == "__main__":
    unittest.main()
