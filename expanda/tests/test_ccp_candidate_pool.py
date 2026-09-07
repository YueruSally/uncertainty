import unittest
from unittest.mock import patch

import numpy as np

import baseline_uncertainty as base
import run_ccp_candidate_pool as pool


def individual(path_id, objective=(1.0, 1.0, 1.0)):
    via = f"V{path_id}"
    arc = base.Arc("A", via, "road", 1.0, 100.0, 1.0, 1.0, 1.0)
    path = base.Path(path_id, "A", "B", ["A", via], ["road"], [arc],
                     1.0, 1.0, 1.0)
    ind = base.Individual({("A", "B", 1): [base.PathAllocation(path, 1.0)]})
    ind.objectives = objective
    ind.feasible = ind.feasible_hard = True
    ind.normalized_violation = 0.0
    return ind


def two_path_individual(reverse=False, noisy_share=False):
    first = individual(1).od_allocations[("A", "B", 1)][0].path
    second = individual(2).od_allocations[("A", "B", 1)][0].path
    share_a = 0.6000000000000001 if noisy_share else 0.6
    allocations = [base.PathAllocation(first, share_a),
                   base.PathAllocation(second, 0.4)]
    if reverse:
        allocations.reverse()
    return base.Individual({("A", "B", 1): allocations})


class CandidatePoolWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.scenarios = base.ScenarioSet(
            3, 7, {("A", "B", "road"): np.array([1., 2., 3.])},
            {}, {}, {}, True)

    def test_three_optimisations_dedup_and_three_frozen_evaluations(self):
        shared = individual(1)
        distinct_equal_objectives = individual(2)
        optimise_calls = []
        evaluation_calls = []

        def optimise(scenario_id, singleton):
            optimise_calls.append((scenario_id, singleton.size))
            return [shared, distinct_equal_objectives] if scenario_id == 0 else [shared]

        def evaluate(ind, singleton):
            evaluation_calls.append((pool.decision_signature(ind),
                                     singleton.travel_multiplier[("A", "B", "road")][0]))
            value = float(singleton.travel_multiplier[("A", "B", "road")][0])
            return value, 4.0 - value, value * 10.0

        with patch.object(base, "crossover_hybrid",
                          side_effect=AssertionError("crossover during re-evaluation")), \
             patch.object(base, "mutate_fixed",
                          side_effect=AssertionError("mutation during re-evaluation")):
            records, occurrences, scenario_paretos = pool.run_candidate_pool_workflow(
                self.scenarios, optimise, evaluate)

        self.assertEqual(optimise_calls, [(0, 1), (1, 1), (2, 1)])
        self.assertEqual(len(scenario_paretos), 3)
        self.assertTrue(all(pareto for _, pareto in scenario_paretos))
        self.assertEqual(occurrences, 4)
        self.assertEqual(len(records), 2)
        self.assertEqual(len(evaluation_calls), 6)
        self.assertTrue(all(record.evaluation_count == 3 for record in records))
        self.assertEqual(records[0].ccp, (3.0, 3.0, 30.0))
        self.assertTrue(all(record.ccp_rank >= 0 and record.ev_rank >= 0
                            for record in records))

    def test_equal_objectives_do_not_deduplicate_distinct_decisions(self):
        records, occurrences = pool.pool_scenario_paretos(
            [(0, [individual(1), individual(2)])])
        self.assertEqual(occurrences, 2)
        self.assertEqual(len(records), 2)

    def test_signature_is_invariant_to_allocation_order(self):
        self.assertEqual(pool.decision_signature(two_path_individual()),
                         pool.decision_signature(two_path_individual(reverse=True)))

    def test_signature_normalises_harmless_share_noise(self):
        self.assertEqual(pool.decision_signature(two_path_individual()),
                         pool.decision_signature(
                             two_path_individual(noisy_share=True)))

    def test_stage_c_rejects_any_decision_mutation(self):
        record = pool.CandidateRecord("placeholder", individual(1))

        def mutating_evaluator(ind, scenario):
            ind.od_allocations[("A", "B", 1)][0].share = 0.5
            return 1.0, 2.0, 3.0

        with self.assertRaisesRegex(RuntimeError, "evaluation copy changed"):
            pool.re_evaluate_pool(
                [record], self.scenarios, mutating_evaluator, .9)

    def test_stage_c_records_matching_pre_and_post_signatures(self):
        record = pool.CandidateRecord("placeholder", individual(1))
        pool.re_evaluate_pool(
            [record], self.scenarios,
            lambda ind, scenario: (1.0, 2.0, 3.0), .9)
        self.assertEqual(record.stage_c_signature_before,
                         record.stage_c_signature_after)
        self.assertEqual(record.stage_c_signature_before,
                         pool.decision_signature(record.individual))

    def test_q90_200_is_180(self):
        self.assertEqual(base.empirical_ccp_quantile(np.arange(1, 201), .9), 180)

    def test_final_ranking_happens_after_reduction(self):
        records = [pool.CandidateRecord("a", individual(1)),
                   pool.CandidateRecord("b", individual(2))]
        calls = []
        original = pool.assign_ranks

        def checked(records_arg, attr, rank_attr):
            self.assertTrue(all(len(r.cost_s) == 3 for r in records_arg))
            self.assertTrue(all(np.isfinite(r.ccp).all() for r in records_arg))
            calls.append(attr)
            original(records_arg, attr, rank_attr)

        with patch.object(pool, "assign_ranks", side_effect=checked):
            pool.re_evaluate_pool(
                records, self.scenarios,
                lambda ind, scenario: (1.0, 2.0, 3.0), .9)
        self.assertEqual(calls, ["ccp", "ev"])


if __name__ == "__main__":
    unittest.main()
