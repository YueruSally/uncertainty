import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np

import baseline_uncertainty as base
from multi_instance_catalog import (SCENARIOS, TEST_ALPHAS, canonical_digest,
                                    delivery_deadline, prepare_batches, read_instance)
from multi_instance_mutation import candidates_for
from multi_instance_logging import MutationLogger
from multi_instance_policy import LearningPolicy, RandomPolicy, RulePolicy
from multi_instance_plan import jobs, paired_seeds
from multi_instance_oos import interval


def candidate(p, eligible, batch_id):
    return dict(selection_probability=p, eligible=eligible,
                target=dict(batch_id=batch_id), batch_quantity=2.)


class InstanceTests(unittest.TestCase):
    def setUp(self):
        self.raw = [base.Batch(i, "A", "B", 80.+i, 10., 110.) for i in range(4)]

    def test_alpha_is_deadline_window_factor(self):
        self.assertEqual([delivery_deadline(self.raw[0], a) for a in (.5, 1., 2.)],
                         [60., 110., 210.])
        self.assertEqual(delivery_deadline(base.Batch(0, "A", "B", 1., 0., 0.), .5), 1.)

    def test_fixed_instance_is_reproducible_and_new_test_od(self):
        a = prepare_batches(self.raw, 10, 2., 50)
        b = prepare_batches(self.raw, 10, 2., 50)
        self.assertEqual([vars(x) for x in a], [vars(x) for x in b])
        t = prepare_batches(self.raw, 5, .5, 51, test=True, od_pool=[("C", "D")])
        self.assertEqual({(x.origin, x.destination) for x in t}, {("C", "D")})
        self.assertFalse({x.quantity for x in t} & {x.quantity for x in self.raw})

    def test_manifest_digest_detects_change(self):
        record = dict(K=1, instance_id="S0", batches=[vars(self.raw[0])])
        record["instance_digest"] = canonical_digest(record)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "S0.json"
            path.write_text(json.dumps(record))
            self.assertEqual(read_instance(path)[0]["instance_id"], "S0")
            record["K"] = 2
            path.write_text(json.dumps(record))
            with self.assertRaises(ValueError):
                read_instance(path)


class PolicyTests(unittest.TestCase):
    def test_random_keeps_original_support_rule_removes_it(self):
        candidates = [candidate(.5, False, 0), candidate(.25, True, 1),
                      candidate(.25, True, 2)]
        self.assertEqual(RandomPolicy(random.Random(1)).choose(candidates).probabilities,
                         [.5, .25, .25])
        self.assertEqual(RulePolicy(random.Random(1)).choose(candidates).probabilities,
                         [0., .5, .5])

    def test_no_candidate_does_not_resample(self):
        rows = [candidate(.5, False, 0), candidate(.5, False, 1)]
        self.assertIsNone(RulePolicy().choose(rows).chosen_index)
        learning = LearningPolicy.__new__(LearningPolicy)
        learning.epsilon = .1
        learning.rng = random.Random(1)
        self.assertIsNone(learning.choose(rows).chosen_index)

    def test_learning_mixes_only_inside_mask_and_falls_back(self):
        rows = [candidate(.5, False, 0), candidate(.25, True, 1),
                candidate(.25, True, 2)]
        learning = LearningPolicy.__new__(LearningPolicy)
        learning.epsilon = .1
        learning.rng = random.Random(5)
        learning.artifact = {"target": "selection_survivor"}
        learning.pipeline = Mock()
        learning.pipeline.classes_ = np.array([0, 1])
        learning.pipeline.predict_proba.return_value = np.array([[.8, .2], [.1, .9]])
        decision = learning.choose(rows)
        self.assertEqual(decision.probabilities[0], 0.)
        self.assertAlmostEqual(sum(decision.probabilities), 1.)
        self.assertAlmostEqual(decision.metadata["tv_distance"],
                               .5*sum(abs(x-y) for x, y in
                                      zip(decision.probabilities, [0., .5, .5])))
        learning.pipeline.predict_proba.return_value = np.array([[1., 0.], [1., 0.]])
        self.assertEqual(learning.choose(rows).metadata["fallback"], "invalid_score_mass")

    def test_replace_mask_uses_reliable_options_not_path_library(self):
        arc = base.Arc("A", "B", "road", 50., 100., 1., 1., 50.)
        path = base.path_from_arcs([arc], "A", "B")
        batch = base.Batch(0, "A", "B", 2., 0., 10.)
        ind = base.Individual({("A", "B", 0): [base.PathAllocation(path, 1.)]})
        rows = candidates_for(ind, [batch], "replace", {("A", "B"): [path]}, {},
                              {("A", "B", "road"): arc}, {})
        self.assertFalse(rows[0]["eligible"])
        self.assertEqual(rows[0]["mask_reason"], "no_reliable_option")


class PlanTests(unittest.TestCase):
    def test_run_counts_and_pair_seeds(self):
        entries = [dict(instance_id=s, split="validation" if s in ("S8", "S9")
                        else "train") for s in SCENARIOS]
        entries += [dict(instance_id=f"T{i}", split="test") for i in range(1, 6)]
        plan = dict(entries=entries, audit=[("S0", "random"), ("S0", "rule"),
                                          ("S9", "random"), ("S9", "rule")])
        lock = dict(label="selection_survivor", epsilon=.1)
        counts = [len(jobs(plan, stage, lock)) for stage in
                  ("audit", "train", "validation_rule", "validation_labels",
                   "epsilon_zero", "test_controls", "test_learning")]
        self.assertEqual(counts, [4, 50, 10, 30, 10, 100, 50])
        self.assertEqual(sum(counts), 254)
        self.assertEqual(paired_seeds("T1", 1), paired_seeds("T1", 1))

    def test_master_prefix_is_exact_slice(self):
        master = base.ScenarioSet(size=5000, seed=1,
            travel_multiplier={("A", "B", "road"): np.arange(5000.)},
            border_delay_h={}, arc_border_event={}, border_event_mean_h={})
        tail = interval(master, 1000, 5000)
        self.assertEqual(tail.travel_multiplier[("A", "B", "road")][0], 1000.)
        self.assertEqual(tail.size, 4000)


class OutcomeLabelTests(unittest.TestCase):
    def test_rule_no_candidate_keeps_decision_and_logs_skip_only(self):
        arc = base.Arc("A", "B", "road", 50., 100., 1., 1., 50.)
        path = base.path_from_arcs([arc], "A", "B")
        batch = base.Batch(0, "A", "B", 2., 0., 10.)
        ind = base.Individual({("A", "B", 0): [base.PathAllocation(path, 1.)]})
        previous = (base.ACTIVE_MUTATION_LOGGER, base.ACTIVE_SCENARIO_SET,
                    base.RISK_METRIC, base._PATH_SCENARIO_CACHE)
        with tempfile.TemporaryDirectory() as tmp:
            logger = MutationLogger(Path(tmp), "r", "s", policy=RulePolicy(),
                                    instance={"instance_id": "S0", "K": 1,
                                              "deadline_window_alpha": 1.},
                                    seeds={"training": 1})
            try:
                base.ACTIVE_MUTATION_LOGGER = logger
                base.RISK_METRIC = "ccp"
                base.configure_scenario_set([arc], {}, size=100, seed=1,
                                            stochastic=True)
                with patch.object(base, "sample_operator", return_value="del"):
                    op, ok = logger.mutate(ind, [batch], {("A", "B"): [path]}, {},
                                           {("A", "B", "road"): arc}, [arc], 0., 0.)
                self.assertEqual((op, ok), ("del", False))
                self.assertEqual(logger.no_candidate, 1)
                self.assertEqual(len(logger.pending), 0)
            finally:
                logger.close()
                (base.ACTIVE_MUTATION_LOGGER, base.ACTIVE_SCENARIO_SET,
                 base.RISK_METRIC, base._PATH_SCENARIO_CACHE) = previous
            self.assertEqual((Path(tmp) / "mutation_outcomes.jsonl").read_text(), "")
            self.assertEqual(len((Path(tmp) / "mutation_skips.jsonl").read_text().splitlines()), 1)

    def test_combined_first_front_differs_from_environmental_survival(self):
        with tempfile.TemporaryDirectory() as tmp:
            logger = MutationLogger(Path(tmp), "r", "s")
            ind = base.Individual()
            ind.feasible, ind.rank = True, 0
            row = dict(phase="offspring", effective_mutation=True,
                       feasible_before=True, feasible_after=True,
                       outcome_attributable_to_selected_target=True,
                       finite_objective_label=True,
                       child_dominates_before=False, before_dominates_child=False,
                       q90_cost_before=10., q90_cost_after=9.,
                       q90_emission_before=10., q90_emission_after=11.,
                       q90_makespan_before=10., q90_makespan_after=10.)
            logger.pending.append((ind, row))
            logger.flush_events([])
            logger.close()
            actual = json.loads((Path(tmp) / "mutation_outcomes.jsonl").read_text())
            self.assertTrue(actual["first_front_member_label"])
            self.assertFalse(actual["selection_survivor"])
            self.assertEqual(actual["parent_relation"], "incomparable")


if __name__ == "__main__":
    unittest.main()
