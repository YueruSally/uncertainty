import json
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import baseline_uncertainty as base
from learning_mutation import enumerate_targets, repair_after_mutation
from mutation_logging import MutationLogger, fingerprint


class LoggingTests(unittest.TestCase):
    def setUp(self):
        self.old_logger = base.ACTIVE_MUTATION_LOGGER
        self.old_scenarios = base.ACTIVE_SCENARIO_SET
        self.old_risk = base.RISK_METRIC
        self.old_cache = base._PATH_SCENARIO_CACHE
        self.arc = base.Arc("A", "B", "road", 60., 1000., 1., 2., 60.)
        self.path = base.path_from_arcs([self.arc], "A", "B")
        self.batch = base.Batch(batch_id=1, origin="A", destination="B", quantity=1., ET=0., LT=5.)
        self.key = ("A", "B", 1)
        self.ind = base.Individual(od_allocations={self.key: [base.PathAllocation(self.path, 1.)]})
        self.paths = {("A", "B"): [self.path]}
        self.lookup = {("A", "B", "road"): self.arc}
        self.tmp = tempfile.TemporaryDirectory()
        self.logger = MutationLogger(Path(self.tmp.name), "test", "frozen100", budget=20)
        base.ACTIVE_MUTATION_LOGGER = self.logger
        base.RISK_METRIC = "ccp"
        base.configure_scenario_set([self.arc], {}, size=100, seed=123, stochastic=True)

    def tearDown(self):
        base.ACTIVE_MUTATION_LOGGER = self.old_logger
        base.ACTIVE_SCENARIO_SET = self.old_scenarios
        base.RISK_METRIC = self.old_risk
        base._PATH_SCENARIO_CACHE = self.old_cache
        self.logger.close()
        self.tmp.cleanup()

    def evaluate(self, ind):
        base.evaluate_individual(ind, [self.batch], [self.arc], {}, 0., 0.)

    def test_equal_probabilities(self):
        self.assertEqual(base._FIXED_OP_PROBS, [.2] * 5)

    def test_exact_duplicate_reuses_evaluation(self):
        self.evaluate(self.ind)
        self.evaluate(self.ind)
        self.assertEqual(self.logger.evaluations, 1)
        self.assertEqual(len(self.ind.cost_s), 100)
        self.ind.od_allocations[self.key][0].share = .9
        self.evaluate(self.ind)
        self.assertEqual(self.logger.evaluations, 2)

    def test_failed_delete_is_logged_without_resampling(self):
        with patch.object(base, "sample_operator", return_value="del") as sample:
            op, ok = self.logger.mutate(self.ind, [self.batch], self.paths, {}, self.lookup,
                                        [self.arc], 0., 0.)
        self.assertEqual(op, "del")
        self.assertFalse(ok)
        sample.assert_called_once()
        self.assertEqual(self.logger.evaluations, 1)
        self.logger.flush_events([self.ind])
        event = json.loads((Path(self.tmp.name)/"mutation_events.jsonl").read_text())
        self.assertFalse(event["decision_changed"])
        self.assertEqual(event["delta_cost"], 0.)

    def test_single_path_share_mutation_is_not_effective(self):
        with patch.object(base, "sample_operator", return_value="mod"):
            self.logger.mutate(self.ind, [self.batch], self.paths, {}, self.lookup, [self.arc], 0., 0.)
        event = self.logger.pending[0][1]
        self.assertTrue(event["mutation_success"])
        self.assertFalse(event["decision_changed"])
        self.assertEqual(self.logger.evaluations, 1)

    def test_invalid_path_restores_before(self):
        before = deepcopy(self.ind)
        self.ind.od_allocations[self.key][0].path.nodes = ["A", "C"]
        result = repair_after_mutation(self.ind, before, [self.batch], self.paths, {}, self.lookup)
        self.assertTrue(result["mutation_reverted"])
        self.assertEqual(fingerprint(self.ind), fingerprint(before))

    def test_capacity_excess_is_not_repaired(self):
        self.arc.capacity = .1
        before = deepcopy(self.ind)
        result = repair_after_mutation(self.ind, before, [self.batch], self.paths, {}, self.lookup)
        self.evaluate(self.ind)
        self.assertFalse(result["mutation_reverted"])
        self.assertFalse(self.ind.feasible)
        self.assertGreater(self.ind.vio_breakdown["cap_excess"], 0)

    def test_target_probabilities_sum_to_one(self):
        for op in base.OPS:
            rows = enumerate_targets(self.ind, [self.batch], op, self.paths, {}, self.lookup)
            self.assertAlmostEqual(sum(r["selection_probability"] for r in rows), 1.)

    def test_single_path_mod_target_is_ineligible(self):
        rows = enumerate_targets(self.ind, [self.batch], "mod", self.paths, {}, self.lookup)
        self.assertEqual(len(rows), 1)
        self.assertFalse(rows[0]["eligible"])

    def test_budget_guard(self):
        self.logger.budget = 0
        with self.assertRaises(RuntimeError):
            self.evaluate(self.ind)

    def test_effective_move_has_separate_before_and_after_evaluations(self):
        alternative_arc = deepcopy(self.arc)
        alternative_arc.mode = "rail"
        alternative_arc.cost_per_teu_km = .5
        alternative_path = base.path_from_arcs([alternative_arc], "A", "B")
        # Use a road replacement with different cost to test evaluator attribution
        # independently from timetable sampling.
        alternative_arc.mode = "road"
        alternative_path.modes = ["road"]
        def replacement(ind, *args, **kwargs):
            ind.od_allocations[self.key] = [base.PathAllocation(alternative_path, 1.)]
            return True
        # Topology+shares fingerprint assumes immutable arc data during a run;
        # use a second valid topology to make the decision genuinely distinct.
        a = deepcopy(alternative_arc)
        b = deepcopy(alternative_arc)
        a.to_node = "C"
        b.from_node = "C"
        a.distance = b.distance = 30.
        alternative_path = base.path_from_arcs([a, b], "A", "B")
        lookup = {**self.lookup, ("A", "C", "road"): a, ("C", "B", "road"): b}
        with patch.object(base, "sample_operator", return_value="replace"), patch.object(base, "apply_mutation_op", side_effect=replacement):
            self.logger.mutate(self.ind, [self.batch], self.paths, {}, lookup, [self.arc, a, b], 0., 0.)
        row = self.logger.pending[0][1]
        self.assertEqual(self.logger.evaluations, 2)
        self.assertTrue(row["decision_changed"])
        self.assertTrue(row["raw_mutation_changed"])
        self.assertTrue(row["effective_mutation"])
        self.assertTrue(row["outcome_attributable_to_selected_target"])
        self.assertEqual(row["q90_cost_before"], 60.)
        self.assertEqual(row["q90_cost_after"], 30.)
        self.assertEqual(row["delta_cost"], 30.)

    def test_repair_only_change_is_not_an_effective_or_eligible_label(self):
        def repair_only(ind, before, *args):
            ind.od_allocations[self.key][0].share = .9
            return dict(repair_called=True, repair_success=True, repair_action_count=1,
                        duplicate_paths_merged=0, shares_removed=0, share_normalised=True,
                        missing_allocation_restored=0, invalid_path_detected=False,
                        mutation_reverted=False)

        with patch.object(base, "sample_operator", return_value="del"), \
                patch("mutation_logging.repair_after_mutation", side_effect=repair_only):
            self.logger.mutate(self.ind, [self.batch], self.paths, {}, self.lookup,
                               [self.arc], 0., 0.)
        row = self.logger.pending[0][1]
        self.assertFalse(row["raw_mutation_changed"])
        self.assertTrue(row["repair_changed_decision"])
        self.assertTrue(row["repair_only_change"])
        self.assertFalse(row["effective_mutation"])
        self.assertFalse(row["outcome_attributable_to_selected_target"])
        self.assertFalse(row["objective_label_eligible"])
        self.assertEqual(self.logger.total_effective, 0)

    def test_missing_allocation_is_restored(self):
        before = deepcopy(self.ind)
        self.ind.od_allocations = {}
        result = repair_after_mutation(self.ind, before, [self.batch], self.paths, {}, self.lookup)
        self.assertEqual(result["missing_allocation_restored"], 1)
        self.assertEqual(fingerprint(self.ind), fingerprint(before))

    def test_repair_is_idempotent_for_roundoff_level_share_sum(self):
        a = deepcopy(self.arc)
        b = deepcopy(self.arc)
        a.to_node = "C"
        b.from_node = "C"
        a.distance = b.distance = 30.
        alternative = base.path_from_arcs([a, b], "A", "B")
        self.ind.od_allocations[self.key] = [
            base.PathAllocation(self.path, .3),
            base.PathAllocation(alternative, .7000000000005),
        ]
        before = deepcopy(self.ind)
        lookup = {**self.lookup, ("A", "C", "road"): a, ("C", "B", "road"): b}
        result = repair_after_mutation(
            self.ind, before, [self.batch], self.paths, {}, lookup)
        self.assertEqual(result["repair_action_count"], 0)
        self.assertFalse(result["share_normalised"])
        self.assertEqual(fingerprint(self.ind), fingerprint(before))

    def test_repair_normalises_material_share_sum_error(self):
        a = deepcopy(self.arc)
        b = deepcopy(self.arc)
        a.to_node = "C"
        b.from_node = "C"
        a.distance = b.distance = 30.
        alternative = base.path_from_arcs([a, b], "A", "B")
        self.ind.od_allocations[self.key] = [
            base.PathAllocation(self.path, .3),
            base.PathAllocation(alternative, .6),
        ]
        before = deepcopy(self.ind)
        lookup = {**self.lookup, ("A", "C", "road"): a, ("C", "B", "road"): b}
        result = repair_after_mutation(
            self.ind, before, [self.batch], self.paths, {}, lookup)
        self.assertEqual(result["repair_action_count"], 1)
        self.assertTrue(result["share_normalised"])
        self.assertAlmostEqual(
            sum(x.share for x in self.ind.od_allocations[self.key]), 1.0)
        self.assertNotEqual(fingerprint(self.ind), fingerprint(before))


if __name__ == "__main__":
    unittest.main()
