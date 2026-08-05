import random
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


class FeasibilityFirstSearchTests(unittest.TestCase):
    @staticmethod
    def _path(path_id, from_node, to_node):
        arc = model.Arc(
            from_node=from_node, to_node=to_node, mode="road",
            distance=1.0, capacity=10.0,
            cost_per_teu_km=1.0, emission_per_teu_km=1.0,
            speed_kmh=1.0)
        return model.Path(
            path_id=path_id, origin="O", destination="D",
            nodes=[from_node, to_node], modes=["road"], arcs=[arc],
            base_cost_per_teu=1.0,
            base_emission_per_teu=1.0,
            base_travel_time_h=1.0)

    def test_capacity_aware_choice_coordinates_batches(self):
        batches = [
            model.Batch(1, "O", "D", 4.0, 0.0, 10.0),
            model.Batch(2, "O", "D", 4.0, 0.0, 10.0),
        ]
        shared_1 = self._path(1, "S", "T")
        shared_2 = self._path(2, "S", "T")
        alt_1 = self._path(3, "A", "B")
        alt_2 = self._path(4, "C", "D")

        def option(path, arc_key):
            return model.ReliablePathOption(
                path=path, on_time_probability=1.0, max_lateness_h=0.0,
                resources={("arc", arc_key, 0): 4.0})

        reliable = {
            ("O", "D", 1): [
                option(shared_1, ("S", "T", "road")),
                option(alt_1, ("A", "B", "road")),
            ],
            ("O", "D", 2): [
                option(shared_2, ("S", "T", "road")),
                option(alt_2, ("C", "D", "road")),
            ],
        }
        capacities = {
            ("S", "T", "road"): 5.0,
            ("A", "B", "road"): 10.0,
            ("C", "D", "road"): 10.0,
        }
        random.seed(7)
        choice, excess = model.find_capacity_aware_choice(
            batches, reliable, capacities, restarts=5, iterations=20)

        self.assertIsNotNone(choice)
        self.assertEqual(excess, 0.0)
        self.assertFalse(choice[0] == 0 and choice[1] == 0)

    def test_node_without_positive_capacity_is_not_constrained(self):
        previous = dict(model.BORDER_CAPACITY)
        try:
            model.BORDER_CAPACITY["UnconstrainedPort"] = 0.0
            capacity = model._resource_available_capacity(
                ("node", "UnconstrainedPort", 0), {})
            self.assertEqual(capacity, float("inf"))
        finally:
            model.BORDER_CAPACITY.clear()
            model.BORDER_CAPACITY.update(previous)

    def test_structural_crossover_preserves_complete_single_path_genes(self):
        batch = model.Batch(1, "O", "D", 1.0, 0.0, 10.0)
        p1 = self._path(1, "A", "B")
        p2 = self._path(2, "C", "D")
        key = ("O", "D", 1)
        parent_1 = model.Individual({
            key: [model.PathAllocation(p1, 1.0)]})
        parent_2 = model.Individual({
            key: [model.PathAllocation(p2, 1.0)]})

        random.seed(11)
        for _ in range(20):
            child_1, child_2 = model.crossover_structural(
                parent_1, parent_2, [batch])
            self.assertEqual(len(child_1.od_allocations[key]), 1)
            self.assertEqual(len(child_2.od_allocations[key]), 1)
            self.assertEqual(child_1.od_allocations[key][0].share, 1.0)
            self.assertEqual(child_2.od_allocations[key][0].share, 1.0)

    def test_infeasible_dominance_uses_normalized_violation(self):
        lower_violation = model.Individual(
            objectives=(10.0, 10.0, 10.0), penalty=1e12,
            feasible=False, normalized_violation=0.01)
        higher_violation = model.Individual(
            objectives=(1.0, 1.0, 1.0), penalty=1.0,
            feasible=False, normalized_violation=0.02)

        self.assertTrue(model.dominates(lower_violation, higher_violation))
        self.assertFalse(model.dominates(higher_violation, lower_violation))


if __name__ == "__main__":
    unittest.main()
