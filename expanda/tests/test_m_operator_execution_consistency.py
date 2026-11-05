"""Real operator/CCP100 checks; no training and no server data required."""
from copy import deepcopy
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import baseline_uncertainty as base
from learning_mutation import valid_path
from multi_instance_logging import fingerprint
from multi_instance_policy import Decision
from transformer_ppo.archive import ObjectiveNormalizer, RewardArchive, RewardConfig, exact_hv_3d
from transformer_ppo.controller import PPOController
from tests.test_m_operator_scheduler_v6 import _Trainer


class OperatorExecutionConsistencyTests(unittest.TestCase):
    def setUp(self):
        self.saved = {name: getattr(base, name) for name in (
            "ACTIVE_SCENARIO_SET", "ACTIVE_MUTATION_LOGGER", "RISK_METRIC",
            "_PATH_SCENARIO_CACHE", "CONFIDENCE_COST", "CONFIDENCE_EMISSION", "CONFIDENCE_TIME")}
        self.py_rng, self.np_rng = random.getstate(), np.random.get_state()
        self.arcs = [base.Arc("A", "B", "road", 60., 1000., 1., 2., 60.),
                     base.Arc("A", "C", "road", 30., 1000., 1., 2., 60.),
                     base.Arc("C", "B", "road", 40., 1000., 1., 2., 60.),
                     base.Arc("A", "B", "rail", 50., 1000., .8, 1., 40.)]
        self.tt = {("A", "B", "rail"): [base.TimetableEntry("A", "B", "rail", 168., 0., 1.)]}
        paths = [base.path_from_arcs(arcs, "A", "B", path_id=index + 1, tt_dict=self.tt)
                 for index, arcs in enumerate(([self.arcs[0]], self.arcs[1:3], [self.arcs[3]]))]
        self.paths = {("A", "B"): paths}
        self.lookup = {(a.from_node, a.to_node, a.mode): a for a in self.arcs}
        self.batch = base.Batch(1, "A", "B", 1., 0., 5.)
        self.batches = [self.batch]
        self.key = ("A", "B", 1)
        self.parent = base.Individual({self.key: [base.PathAllocation(paths[0], .6),
                                                  base.PathAllocation(paths[1], .4)]})
        self.options = {self.key: [base.ReliablePathOption(p, 1., 0., {}) for p in paths]}
        base.ACTIVE_MUTATION_LOGGER = None
        base.RISK_METRIC = "ccp"
        base.CONFIDENCE_COST = base.CONFIDENCE_EMISSION = base.CONFIDENCE_TIME = .9
        base.configure_scenario_set(self.arcs, {}, size=100, seed=123, stochastic=True)
        self.scenarios = base.ACTIVE_SCENARIO_SET
        self.normalizer = ObjectiveNormalizer((0., 0., 0.), (1000., 1000., 10.), reference=(1.1, 1.1, 1.1))
        self.tmp = tempfile.TemporaryDirectory()
        self.trainer = _Trainer()
        self.controller = PPOController(
            Path(self.tmp.name), "consistency", "frozen100", 400,
            trainer=self.trainer, archive=RewardArchive(self.normalizer),
            reward_config=RewardConfig(), batches=self.batches, generations=20,
            pop_size=20, method=self.trainer.name, instance={"instance_id": "toy"},
            seeds={"algorithm": 17}, candidate_audit_states=1,
            candidate_audit_candidates=5, candidate_audit_replicates=3)

    def tearDown(self):
        self.controller.close()
        self.tmp.cleanup()
        for name, value in self.saved.items():
            setattr(base, name, value)
        random.setstate(self.py_rng)
        np.random.set_state(self.np_rng)

    def test_audit_matches_real_search_for_every_operator_and_preserves_state(self):
        c = self.controller
        population_member = deepcopy(self.parent)
        base._evaluate_individual_impl(population_member, self.batches, self.arcs, self.tt, 0., 0.)
        c.current_population = [population_member]
        c.archive.observe(population_member.objectives, population_member.feasible, fingerprint(population_member))
        points = [self.normalizer.transform(population_member.objectives)]
        candidates = c._operator_candidates(self.parent, self.batches, self.paths, self.tt, self.lookup, self.options)
        self.assertTrue(all(row["eligible"] for row in candidates))
        c._candidate_audit_pending = {
            "parent": deepcopy(self.parent), "candidates": candidates,
            "context": {"operator": "add"}, "archive_hv": c.archive.hypervolume,
            "archive_size": len(c.archive.points), "population_points": points,
            "observation": {"eligible": [True] * 5, "action_prior": [.2] * 5,
                            "candidates": [[float(i)] for i in range(5)], "global": [0.], "parent": [[0.]]}}
        self.trainer.pending = {"action": 4, "old_probabilities": [.2] * 5}
        state_py, state_np, state_control = random.getstate(), np.random.get_state(), c.control_rng.getstate()
        original_parent = fingerprint(self.parent)
        original_population = [(fingerprint(p), p.objectives) for p in c.current_population]
        original_archive = deepcopy(c.archive.summary())
        c._run_candidate_audit(self.batches, self.paths, self.tt, self.lookup, self.arcs, 0., 0., self.options, {})
        self.assertEqual(random.getstate(), state_py)
        np.testing.assert_equal(np.random.get_state(), state_np)
        self.assertEqual(c.control_rng.getstate(), state_control)
        self.assertEqual(fingerprint(self.parent), original_parent)
        self.assertEqual([(fingerprint(p), p.objectives) for p in c.current_population], original_population)
        self.assertEqual(c.archive.summary(), original_archive)
        self.assertEqual(c.evaluations, 0)
        self.assertIs(base.ACTIVE_SCENARIO_SET, self.scenarios)
        rows = [json.loads(line) for line in (Path(self.tmp.name) / "mutation_candidate_audit_outcomes.jsonl").read_text().splitlines()]
        self.assertEqual(len(rows), 15)
        for row in rows:
            with self.subTest(operator=row["operator"], repeat=row["replicate"]):
                seed = row["completion_seed"]
                candidate_index = row["candidate_id"]
                c.control_rng = random.Random(row["target_seed"])
                child = deepcopy(self.parent)

                def choose(*args, **kwargs):
                    self.trainer.pending = {"action": candidate_index}
                    self.trainer.sequence_policy_actions += 1
                    return Decision(candidate_index, [.2] * 5, [None] * 5,
                                    {"decision_kind": "model_choice"}), candidates

                def shadow_draw():
                    # Match operator RNG at the start of execution, after the
                    # real search's shadow Base batch/operator draws.
                    random.seed(seed)
                    np.random.seed(seed % (2**32 - 1))
                    return row["operator"]

                c.generation, c.phase = 0, "offspring"
                with patch.object(c, "_choose_operator", side_effect=choose), patch.object(base, "sample_operator", side_effect=shadow_draw):
                    operator, ok = c._mutate_operator_only(
                        child, self.batches, self.paths, self.tt, self.lookup,
                        self.arcs, 0., 0., self.options, 1, {})
                if not ok:
                    # NSGA-II still evaluates a crossed child after a failed mutation.
                    base._evaluate_individual_impl(child, self.batches, self.arcs, self.tt, 0., 0.)
                self.assertEqual(operator, row["operator"])
                self.assertEqual(ok, row["mutation_success"])
                self.assertEqual(fingerprint(child), row["fingerprint"])
                self.assertAlmostEqual(sum(a.share for a in child.od_allocations[self.key]), 1.0)
                self.assertTrue(all(a.share > 0 and valid_path(a.path, self.batch, self.tt, self.lookup)
                                    for a in child.od_allocations[self.key]))
                self.assertEqual(child.feasible, row["feasible"])
                np.testing.assert_allclose(child.objectives, row["objectives"], rtol=0, atol=1e-12)
                self.assertEqual(len(child.cost_s), 100)
                new_points = points + ([self.normalizer.transform(child.objectives)] if child.feasible else [])
                delta = exact_hv_3d(new_points, self.normalizer.reference) - exact_hv_3d(points, self.normalizer.reference)
                self.assertAlmostEqual(delta, row["population_delta_hv"], places=12)

    def test_capacity_infeasibility_is_identical_and_not_hidden_by_repair(self):
        for arc in self.arcs:
            arc.capacity = .01
        self.test_audit_matches_real_search_for_every_operator_and_preserves_state()
        rows = [json.loads(line) for line in (Path(self.tmp.name) / "mutation_candidate_audit_outcomes.jsonl").read_text().splitlines()]
        self.assertTrue(all(not row["feasible"] for row in rows))

    def test_failed_attempts_match_outer_search_evaluation_of_unchanged_child(self):
        with patch.object(base, "apply_mutation_op", return_value=False):
            self.test_audit_matches_real_search_for_every_operator_and_preserves_state()
        rows = [json.loads(line) for line in (Path(self.tmp.name) / "mutation_candidate_audit_outcomes.jsonl").read_text().splitlines()]
        self.assertTrue(all(not row["mutation_success"] for row in rows))
        self.assertTrue(all(row["fingerprint"] == fingerprint(self.parent) for row in rows))

    def test_illegal_delete_does_not_resample_or_mutate_parent(self):
        parent = deepcopy(self.parent)
        parent.od_allocations[self.key] = [base.PathAllocation(self.paths[("A", "B")][0], 1.)]
        before = fingerprint(parent)
        c = self.controller
        rows = c._operator_candidates(parent, self.batches, self.paths, self.tt, self.lookup, self.options)
        self.assertFalse(rows[1]["eligible"])
        with patch.object(base, "apply_mutation_op", side_effect=AssertionError("illegal action executed")):
            ok, batch, target, _ = c._execute_operator_action(
                parent, "del", self.batches, self.paths, self.tt, self.lookup, self.options)
        self.assertFalse(ok)
        self.assertIsNone(batch)
        self.assertIsNone(target)
        self.assertEqual(before, fingerprint(parent))

    def test_same_only_replace_mask_matches_executor(self):
        c = self.controller
        parent = deepcopy(self.parent)
        only_current = {self.key: self.options[self.key][:2]}
        rows = c._operator_candidates(parent, self.batches, self.paths, self.tt, self.lookup, only_current)
        self.assertFalse(rows[4]["eligible"])
        with patch.object(base, "apply_mutation_op", side_effect=AssertionError("unexecutable Replace")):
            ok, batch, _, _ = c._execute_operator_action(
                parent, "replace", self.batches, self.paths, self.tt, self.lookup, only_current)
        self.assertFalse(ok)
        self.assertIsNone(batch)


if __name__ == "__main__":
    unittest.main()
