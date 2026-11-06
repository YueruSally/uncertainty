from copy import deepcopy
from pathlib import Path
import time
import unittest
from unittest.mock import patch

from experiments import expand_terminal_training_instances as coverage
from m_operator_offline import write_json, read_json, digest
from m_terminal_hv_ppo import load_checkpoint
from tests import test_m_terminal_feedback_v3 as fixtures


class TrainingCoverageTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.CompleteTerminalFeedbackTests()
        self.fixture.setUp()
        self.root, self.bundle = self.fixture.root, self.fixture.bundle
        self.rows = self.fixture.rows()
        for index, row in enumerate(self.rows):
            row['seed'] = index

    def tearDown(self):
        self.fixture.tearDown()

    def test_added_training_names_cannot_include_validation_or_development(self):
        plan = {'train_instances': ['train'], 'validation_instances': ['valid'], 'test_instances': ['test']}
        for name in ('train', 'valid', 'test'):
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, 'overlap'):
                coverage.validate_cohort(plan, self.root, [name])

    def test_renamed_duplicate_batches_are_not_additional_instances(self):
        data = self.root/'source.xlsx'
        data.write_text('test workbook identity')
        plan = {'train_instances': ['train'], 'validation_instances': ['valid'],
                'test_instances': ['test'], 'data': str(data)}
        def catalogue(path):
            name = Path(path).stem
            batches = [{'quantity': 1 if name in ('extra', 'valid') else 2}]
            return {'instance_id': name, 'source_data_sha256': digest(data),
                    'configuration': 'S0', 'batches': batches}, None
        with patch.object(coverage, 'read_instance', side_effect=catalogue):
            with self.assertRaisesRegex(ValueError, 'unique batch set'):
                coverage.validate_cohort(plan, self.root, ['extra'])

    def test_collects_all_legal_branches_with_unchanged_validation(self):
        old_rows = [deepcopy(self.rows[0]), deepcopy(self.rows[-1])]
        plan = {'extra_instances': ['extra'], 'seeds': [7], 'workers': 1,
                'train_instances': ['train', 'extra'], 'validation_instances': ['valid'],
                'bundle': str(self.bundle), 'maximum_new_searches': 5}
        write_json(self.root/'PLAN.json', plan)
        x = old_rows[0]['tokens']
        mask = [True, False, False, True, True]
        calls = []
        def batch(jobs, stage):
            values = []
            for name, seed, action in jobs:
                calls.append((name, seed, action))
                values.append(((name, seed, action), {'pairing': {'frozen': True},
                    'first_supported_input_sha256': 'same', 'real_objective_evaluation_count': 400,
                    'fixed_choices': 1, 'closed_loop_switches': int(action != 4),
                    'final_hypervolume': {0: .8, 3: 1.1, 4: 1.}[action]},
                    {'tokens': x, 'eligible': mask}))
            return values
        with patch.object(coverage, 'branch_runner', return_value=batch):
            rows = coverage.collect(plan, old_rows, self.root, time.monotonic()+10)
        self.assertEqual(calls, [('extra', 7, 4), ('extra', 7, 0), ('extra', 7, 3)])
        self.assertEqual(rows[:2], old_rows)
        self.assertEqual(rows[-1]['terminal_hvs'], [.8, None, None, 1.1, 1.])
        self.assertEqual(read_json(self.root/'DATASET.json')['new_objective_evaluations'], 1200)

    def test_fit_keeps_holdout_separate_and_marks_argmax_contract(self):
        plan = {'train_instances': ['train'], 'validation_instances': ['valid'],
                'bundle': str(self.bundle), 'candidate_epochs': [10, 20]}
        write_json(self.root/'PLAN.json', plan)
        write_json(self.root/'DATASET.json', {'rows': self.rows})
        report = coverage.fit(plan, self.rows, self.root)
        self.assertEqual(report['training_contexts'], 4)
        self.assertEqual(report['validation_contexts'], 4)
        self.assertEqual(report['new_ppo_updates'], 0)
        self.assertIn(report['selected']['epoch'], [10, 20])
        checkpoint = load_checkpoint(report['checkpoint'])
        self.assertEqual(checkpoint['execution_mode'], 'single-intervention-argmax')
        self.assertEqual(checkpoint['update'], 0)
        attempts = list((self.root/'training_attempts').glob('*.json'))
        self.assertEqual(len(attempts), 1)
        self.assertTrue(read_json(attempts[0])['completed'])


if __name__ == '__main__':
    unittest.main()
