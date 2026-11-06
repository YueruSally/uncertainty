from copy import deepcopy
import unittest

import torch

from experiments.check_terminal_decoding import decisions, summarize


class DecodingCheckTests(unittest.TestCase):
    def test_balanced_ties_and_illegal_high_score(self):
        logits = torch.tensor([[0., 100., -1., 0., 0.], [-1., 100., -1., 1., 0.]])
        mask = torch.tensor([[True, False, True, True, True]]*2)
        probabilities, actions = decisions(logits, mask)
        self.assertEqual(actions.tolist(), [4, 3])
        self.assertTrue(bool((probabilities[:, 1] == 0).all()))
        self.assertTrue(torch.allclose(probabilities.sum(-1), torch.ones(2)))

    def test_argmax_can_be_better_or_worse_than_sampling(self):
        class Policy:
            def eval(self):
                return self

            def __call__(self, x, mask):
                return torch.tensor([[0., -torch.inf, -torch.inf, 1., -1.]]).repeat(len(x), 1), None

        rows = [{'instance': 'v', 'seed': 7, 'tokens': [[0.]]*5,
                 'eligible': [True, False, False, True, True],
                 'terminal_hvs': [.5, None, None, 1.2, 1.]}]
        good = summarize(Policy(), rows)
        self.assertGreater(good['argmax_gain_percent'], good['softmax_expected_gain_percent'])
        self.assertTrue(good['all_instances_argmax_positive'])
        bad_rows = deepcopy(rows)
        bad_rows[0]['terminal_hvs'] = [1.2, None, None, .5, 1.]
        bad = summarize(Policy(), bad_rows)
        self.assertLess(bad['argmax_gain_percent'], bad['softmax_expected_gain_percent'])
        self.assertFalse(bad['all_instances_argmax_positive'])
        # Altering outcome labels cannot change the selected action or probabilities.
        for key in ('argmax_operator', 'probabilities'):
            self.assertEqual(good['decisions'][0][key], bad['decisions'][0][key])

    def test_rejects_nonfinite_legal_scores_and_absent_balanced(self):
        mask = torch.ones((1, 5), dtype=torch.bool)
        logits = torch.zeros((1, 5))
        logits[0, 0] = torch.nan
        with self.assertRaisesRegex(ValueError, 'non-finite'):
            decisions(logits, mask)
        mask[0, 4] = False
        with self.assertRaisesRegex(ValueError, 'fallback'):
            decisions(torch.zeros((1, 5)), mask)


if __name__ == '__main__':
    unittest.main()
