import unittest
from unittest.mock import patch
from tempfile import TemporaryDirectory
from pathlib import Path

import numpy as np
import torch

from m_prescreen_surrogate import (CandidateObjectiveTransformer,
    choose_by_predicted_hv, fit_scaler, loss_value, pack_rows)
from transformer_ppo.archive import ObjectiveNormalizer
from train_m_prescreen_surrogate import fit_one


def rows():
    result = []
    for state in range(3):
        labels = []
        for action in range(8):
            labels.append({'feasible': action != 7,
                'objectives': None if action == 7 else
                    [20.+state-action, 30.+state+action, 4.+.1*action],
                'hv_contribution': .01*action, 'fingerprint': str(action),
                'source': 'balanced' if action == 0 else 'alternative'})
        result.append({'generation': 10+state, 'evaluation': 200+state,
            'population_points': [[.5,.5,.5],[.6,.4,.7]],
            'candidate_features': [[action, state, 1.] for action in range(8)],
            'candidate_batch_tokens': [[
                [action, batch, state, 1.] for batch in range(2)]
                for action in range(8)], 'labels': labels})
    return result


class PrescreenSurrogateTests(unittest.TestCase):
    def setUp(self):
        self.normalizer = ObjectiveNormalizer((0.,0.,0.), (100.,100.,10.),
                                              (1.1,1.1,1.1))
        self.rows = rows()
        self.scaler = fit_scaler(self.rows, self.normalizer)

    def test_tensor_contract_and_finite_training_loss(self):
        packed = pack_rows(self.rows, self.scaler, self.normalizer)
        model = CandidateObjectiveTransformer(packed[0].shape[-1],
            packed[1].shape[-1], packed[2].shape[-1])
        objective, feasibility = model(*packed[:4])
        self.assertEqual(tuple(objective.shape), (3,8,3))
        self.assertEqual(tuple(feasibility.shape), (3,8))
        loss = loss_value(objective, feasibility, *packed[4:])
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(any(parameter.grad is not None
                            for parameter in model.parameters()))

    def test_hv_decoding_keeps_balanced_tie_and_selects_better_point(self):
        prediction = np.full((2,8,3), .9)
        probability = np.ones((2,8))
        prediction[0,0] = [.4,.4,.4]
        prediction[0,1] = [.4,.4,.4]
        prediction[1,0] = [.6,.6,.6]
        prediction[1,3] = [.3,.3,.3]
        selected = choose_by_predicted_hv(self.rows[:2], prediction,
                                           probability, (1.1,1.1,1.1))
        np.testing.assert_array_equal(selected, [0,3])

    def test_scaler_uses_only_rows_passed_to_it(self):
        first = fit_scaler(self.rows[:2], self.normalizer)
        changed = rows()
        changed[-1]['candidate_features'][0][0] = 1e9
        second = fit_scaler(changed[:2], self.normalizer)
        self.assertEqual(first, second)

    def test_one_epoch_training_writes_loadable_checkpoint(self):
        with TemporaryDirectory() as folder, \
             patch('train_m_prescreen_surrogate.EPOCHS', 1), \
             patch('train_m_prescreen_surrogate.CHECKPOINT_EPOCHS', (1,)):
            model, report = fit_one(7, self.rows[:2], self.rows[2:],
                self.scaler, self.normalizer, Path(folder))
            self.assertEqual(report['selected']['epoch'], 1)
            self.assertTrue(Path(report['checkpoint']).is_file())
            self.assertEqual(len(model.state_dict()),
                             len(torch.load(report['checkpoint'],
                                            weights_only=True)['model']))


if __name__ == '__main__':
    unittest.main()
