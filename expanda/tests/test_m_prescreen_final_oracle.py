from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

import numpy as np

import baseline_uncertainty as base
from m_candidate_information import SPECIFIC_NAMES
from m_prescreen_final_oracle import (FinalOracleController,
    FinalOraclePolicy, simple_nominal_score)
from multi_instance_logging import fingerprint
from transformer_ppo.archive import RewardArchive, RewardConfig
import tests.test_m_operator_execution_consistency as fixtures


class FinalHVOracleTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.OperatorExecutionConsistencyTests()
        self.fixture.setUp()
        self.temp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp.cleanup()
        self.fixture.tearDown()

    def controller(self, method):
        policy = FinalOraclePolicy("fixture", 31, method)
        out = Path(self.temp.name) / method
        out.mkdir()
        controller = FinalOracleController(out,
            "fixture", "frozen100", 400, trainer=policy,
            archive=RewardArchive(self.fixture.normalizer),
            reward_config=RewardConfig(), batches=self.fixture.batches,
            generations=1000, pop_size=20, method=policy.name,
            instance={"instance_id": "fixture"},
            seeds={"algorithm": 31})
        controller.generation = 10
        controller.phase = "offspring"
        controller._active_mixture_event_index = 1
        return controller

    def test_simple_score_uses_only_three_nominal_deltas(self):
        low = np.zeros(len(SPECIFIC_NAMES))
        high = np.zeros(len(SPECIFIC_NAMES))
        for name in ("delta_cost", "delta_emission", "delta_nominal_time"):
            low[SPECIFIC_NAMES.index(name)] = -1.0
            high[SPECIFIC_NAMES.index(name)] = 1.0
        self.assertGreater(simple_nominal_score({"features": low}),
                           simple_nominal_score({"features": high}))

    def test_oracle_charges_selected_to_b400_and_others_as_diagnostic(self):
        f = self.fixture
        controller = self.controller("K8_ORACLE")
        try:
            f.parent.od_allocations[f.key] = [
                base.PathAllocation(f.paths[("A", "B")][0], 1.0)]
            population_member = deepcopy(f.parent)
            base._evaluate_individual_impl(population_member, f.batches,
                f.arcs, f.tt, 0.0, 0.0)
            controller.current_population = [population_member]
            child = deepcopy(f.parent)
            base.ACTIVE_MUTATION_LOGGER = controller
            try:
                result = controller._mutate_operator_only(child, f.batches,
                    f.paths, f.tt, f.lookup, f.arcs, 0.0, 0.0, f.options,
                    1, {})
            finally:
                base.ACTIVE_MUTATION_LOGGER = None
            self.assertEqual(result[0], "replace")
            self.assertEqual(controller.evaluations, 1)
            self.assertEqual(controller.oracle_pool_objective_evaluations,
                             controller.pool_candidates)
            self.assertEqual(controller.oracle_additional_objective_evaluations,
                             controller.pool_candidates-controller.pool_events)
            self.assertEqual(controller.cache_hits, 1)
            self.assertEqual(fingerprint(child),
                             controller.proposals[controller.selected_proposal]["fingerprint"])
            self.assertTrue(np.isfinite(child.objectives).all())
            summary = controller.summary()["k8_final_hv_audit"]
            self.assertEqual(summary["total_physical_objective_evaluations"],
                             1+summary["oracle_additional_objective_evaluations"])
        finally:
            controller.close()

    def test_balanced_and_simple_never_evaluate_hidden_candidates(self):
        f = self.fixture
        for method in ("BALANCED", "K8_SIMPLE"):
            controller = self.controller(method)
            try:
                f.parent.od_allocations[f.key] = [
                    base.PathAllocation(f.paths[("A", "B")][0], 1.0)]
                member = deepcopy(f.parent)
                base._evaluate_individual_impl(member, f.batches, f.arcs,
                    f.tt, 0.0, 0.0)
                controller.current_population = [member]
                base.ACTIVE_MUTATION_LOGGER = controller
                try:
                    controller._mutate_operator_only(deepcopy(f.parent),
                        f.batches, f.paths, f.tt, f.lookup, f.arcs, 0.0,
                        0.0, f.options, 1, {})
                finally:
                    base.ACTIVE_MUTATION_LOGGER = None
                self.assertEqual(controller.evaluations, 1)
                self.assertEqual(controller.oracle_pool_objective_evaluations, 0)
                self.assertEqual(controller.oracle_additional_objective_evaluations, 0)
            finally:
                controller.close()


if __name__ == "__main__":
    unittest.main()
