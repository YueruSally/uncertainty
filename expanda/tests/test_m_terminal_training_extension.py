from pathlib import Path
from copy import deepcopy
import unittest
from unittest.mock import patch

from experiments import m_terminal_feedback_v3 as original
from experiments import extend_terminal_feedback_training as extension
from m_operator_offline import read_json, write_json, digest
from m_terminal_hv_ppo import load_checkpoint, save_checkpoint
from tests import test_m_terminal_feedback_v3 as fixtures


class TrainingExtensionTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.CompleteTerminalFeedbackTests()
        self.fixture.setUp()
        self.root, self.bundle = self.fixture.root, self.fixture.bundle
        self.rows = self.fixture.rows()
        for i, row in enumerate(self.rows):
            row['seed'] = i
        write_json(self.root/'DATASET.json', {'rows': self.rows})
        with patch.object(original, 'VALID_INSTANCES', ('valid',)), patch.object(original, 'EPOCHS', 10):
            self.old = original.fit_complete_feedback(self.bundle, self.rows, self.root, 'original-fixture')
        self.plan = {'train_instances': ['train'], 'validation_instances': ['valid']}
        self.out = self.root/'extension'
        self.out.mkdir()

    def tearDown(self):
        self.fixture.tearDown()

    def run_extension(self, old=None):
        return extension.fit(self.bundle, self.rows, self.plan, old or self.old,
                             self.root/'pretrained.pt', self.out, 'extension-fixture', candidates=(15, 20))

    def test_exact_adam_replay_then_extension_leaves_source_unchanged(self):
        before = digest(self.root/'pretrained.pt')
        result = self.run_extension()
        self.assertTrue(result['source_selected_weights_reproduced_exactly'])
        self.assertTrue(result['source_training_metrics_reproduced'])
        self.assertEqual(result['new_objective_evaluations'], 0)
        self.assertEqual(result['new_search_runs'], 0)
        self.assertEqual(result['new_ppo_updates'], 0)
        self.assertEqual(result['replayed_original_steps'], 10)
        self.assertEqual(result['additional_training_steps'], 10)
        self.assertEqual(digest(self.root/'pretrained.pt'), before)
        self.assertGreaterEqual(result['selected']['validation']['mean_expected_terminal_gain'],
                                self.old['selected']['validation']['mean_expected_terminal_gain'])
        self.assertTrue((self.out/'selected_pretrained.pt').exists())
        self.assertIn('not independent', result['qualification'])

    def test_corrupted_source_metrics_stop_before_extension(self):
        old = deepcopy(self.old)
        old['history'][0]['training']['mean_expected_terminal_gain'] += .01
        with self.assertRaisesRegex(ValueError, 'metrics did not reproduce'):
            self.run_extension(old)
        self.assertFalse((self.out/'selected_pretrained.pt').exists())

    def test_checkpoint_with_same_report_but_different_weights_is_rejected(self):
        path = self.root/'pretrained.pt'
        changed = load_checkpoint(path)
        name = next(iter(changed['model']))
        changed['model'][name] = changed['model'][name]+.01
        save_checkpoint(path, changed)
        with self.assertRaisesRegex(ValueError, 'weights did not reproduce exactly'):
            self.run_extension()

    def test_missing_branches_and_validation_overlap_are_rejected(self):
        changed = deepcopy(self.rows)
        changed[0]['terminal_hvs'][0] = None
        with self.assertRaisesRegex(ValueError, 'missing legal outcome'):
            extension.partition(changed, self.plan, self.bundle)
        bad_plan = {'train_instances': ['train'], 'validation_instances': ['train']}
        with self.assertRaisesRegex(ValueError, 'overlaps original'):
            extension.partition(self.rows, bad_plan, self.bundle)


if __name__ == '__main__':
    unittest.main()
