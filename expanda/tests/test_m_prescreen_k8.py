from copy import deepcopy
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import baseline_uncertainty as base
from m_candidate_information import build_proposals
from m_prescreen_k8 import (CollectBalancedPolicy, CollectK8Controller,
    build_replace_pool, batch_tokens)
from multi_instance_logging import fingerprint
from transformer_ppo.archive import RewardArchive, RewardConfig
import tests.test_m_operator_execution_consistency as fixtures


class PrescreenK8Tests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.OperatorExecutionConsistencyTests()
        self.fixture.setUp()
        self.temp = tempfile.TemporaryDirectory()
        policy = CollectBalancedPolicy('fixture', 17)
        self.controller = CollectK8Controller(Path(self.temp.name), 'fixture',
            'frozen100', 400, trainer=policy,
            archive=RewardArchive(self.fixture.normalizer),
            reward_config=RewardConfig(), batches=self.fixture.batches,
            generations=1000, pop_size=20, method=policy.name,
            instance={'instance_id': 'fixture'}, seeds={'algorithm': 17})
        self.controller.generation = 10
        self.controller._active_mixture_event_index = 1

    def tearDown(self):
        self.controller.close()
        self.temp.cleanup()
        self.fixture.tearDown()

    def test_pool_is_evaluation_free_deterministic_and_balanced_first(self):
        f, c = self.fixture, self.controller
        parent_hash = fingerprint(f.parent)
        py_state, np_state = random.getstate(), np.random.get_state()
        operator_rows = c._operator_candidates(f.parent, f.batches,
            f.paths, f.tt, f.lookup, f.options)
        balanced = build_proposals(c, f.parent, f.batches, f.paths,
            f.tt, f.lookup, f.options, operator_rows)[4]
        with patch.object(base, 'evaluate_individual',
                side_effect=AssertionError('candidate was evaluated')), \
             patch.object(base, '_evaluate_individual_impl',
                side_effect=AssertionError('candidate was evaluated')):
            first = build_replace_pool(c, f.parent, f.batches,
                f.paths, f.tt, f.lookup, f.options)
            second = build_replace_pool(c, f.parent, f.batches,
                f.paths, f.tt, f.lookup, f.options)
        self.assertEqual(first[0]['fingerprint'], balanced['fingerprint'])
        self.assertEqual(first[0]['source'], 'balanced')
        self.assertEqual([p['fingerprint'] for p in first],
                         [p['fingerprint'] for p in second])
        self.assertEqual(len({p['fingerprint'] for p in first}), len(first))
        self.assertLessEqual(len(first), 8)
        self.assertEqual(fingerprint(f.parent), parent_hash)
        self.assertEqual(random.getstate(), py_state)
        np.testing.assert_equal(np.random.get_state(), np_state)
        self.assertTrue(all(np.isfinite(batch_tokens(p['child'], f.batches)).all()
                            for p in first))

    def test_balanced_execution_copies_frozen_child(self):
        f, c = self.fixture, self.controller
        c.current_population = []
        decision, _ = c._choose_operator(f.parent, f.batches,
            f.paths, f.tt, f.lookup, f.options)
        self.assertEqual(decision.chosen_index, 0)
        target = deepcopy(f.parent)
        with patch.object(base, 'apply_mutation_op',
                side_effect=AssertionError('mutation redrawn')):
            c._execute_operator_action(target, 'replace', f.batches,
                f.paths, f.tt, f.lookup, f.options)
        self.assertEqual(fingerprint(target), c.proposals[0]['fingerprint'])
        self.assertEqual(c.execution_checks, 1)

    def test_saved_labels_charge_only_alternatives_outside_search_budget(self):
        f, c = self.fixture, self.controller
        f.parent.od_allocations[f.key] = [base.PathAllocation(f.paths[('A', 'B')][0], 1.)]
        member = deepcopy(f.parent)
        base._evaluate_individual_impl(member, f.batches, f.arcs,
            f.tt, 0., 0.)
        c.current_population = [member]
        c.phase = 'offspring'
        child = deepcopy(f.parent)
        with patch.object(base, 'evaluate_individual',
                          wraps=base.evaluate_individual) as search_evaluate:
            result = c._mutate_operator_only(child, f.batches, f.paths,
                f.tt, f.lookup, f.arcs, 0., 0., f.options, 1, {})
        self.assertEqual(result[0], 'replace')
        self.assertEqual(len(c.feedback_rows), 1)
        row = c.feedback_rows[0]
        self.assertEqual(row['labels'][0]['fingerprint'], fingerprint(child))
        self.assertEqual(row['additional_objective_evaluations'],
                         len(row['labels'])-1)
        self.assertEqual(c.feedback_evaluations, len(row['labels'])-1)
        self.assertEqual(search_evaluate.call_count, 1)


if __name__ == '__main__':
    unittest.main()
