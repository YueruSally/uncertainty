import unittest

import numpy as np

import baseline_uncertainty as model


class FrozenScenarioTests(unittest.TestCase):
    def test_empirical_ccp_quantile_uses_order_statistic(self):
        values = np.arange(1.0, 11.0)
        self.assertEqual(model.empirical_ccp_quantile(values, 0.80), 8.0)
        self.assertEqual(model.empirical_ccp_quantile(values, 0.90), 9.0)
        self.assertEqual(model.empirical_ccp_quantile(values, 1.00), 10.0)

    def test_capped_lognormal_is_reproducible_and_mean_corrected(self):
        first = model.sample_capped_mean_one_lognormal(
            size=200000, cv=0.60, cap_factor=3.5,
            rng=np.random.default_rng(2026))
        second = model.sample_capped_mean_one_lognormal(
            size=200000, cv=0.60, cap_factor=3.5,
            rng=np.random.default_rng(2026))
        self.assertTrue(np.array_equal(first, second))
        self.assertAlmostEqual(float(np.mean(first)), 1.0, delta=0.005)
        self.assertLessEqual(float(np.max(first)), 3.5 + 1e-12)
        sigma = np.sqrt(np.log1p(0.60 ** 2))
        mu = model.calibrated_capped_lognormal_mu(0.60, 3.5)
        self.assertAlmostEqual(
            model.capped_lognormal_mean(mu, sigma, 3.5), 1.0, places=10)

    def test_same_candidate_has_identical_ccp_values(self):
        arc = model.Arc(
            from_node="A", to_node="B", mode="road",
            distance=60.0, capacity=1000.0,
            cost_per_teu_km=1.0, emission_per_teu_km=2.0,
            speed_kmh=60.0)
        path = model.Path(
            path_id=1, origin="A", destination="B",
            nodes=["A", "B"], modes=["road"], arcs=[arc],
            base_cost_per_teu=60.0,
            base_emission_per_teu=120.0,
            base_travel_time_h=1.0)
        batch = model.Batch(
            batch_id=1, origin="A", destination="B",
            quantity=1.0, ET=0.0, LT=5.0,
            penalty_per_teu_h=2.0, max_late_h=2.0)

        model.configure_scenario_set(
            arcs=[arc], border_delay_map={}, size=50,
            seed=1234, stochastic=True)

        def evaluate_once():
            individual = model.Individual(od_allocations={
                ("A", "B", 1): [model.PathAllocation(path=path, share=1.0)]
            })
            model.evaluate_individual(
                individual, [batch], [arc], {},
                waiting_cost_per_teu_h=0.0,
                wait_emis_g_per_teu_h=0.0)
            return individual

        first = evaluate_once()
        second = evaluate_once()
        self.assertEqual(first.objectives, second.objectives)
        self.assertEqual(first.penalty, second.penalty)
        self.assertTrue(first.feasible)
        self.assertEqual(first.objectives[0], 60.0)
        self.assertEqual(first.objectives[1], 120.0)


if __name__ == "__main__":
    unittest.main()
