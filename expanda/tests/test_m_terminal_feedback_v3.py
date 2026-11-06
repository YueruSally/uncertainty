from copy import deepcopy
import unittest
from unittest.mock import patch

import numpy as np
import torch

from experiments import m_terminal_feedback_v3 as m
from m_candidate_information import SPECIFIC_NAMES
from m_operator_offline import read_json, write_json, digest
from m_terminal_hv_ppo import checkpoint_model
from tests import test_m_terminal_hv_ppo as fixtures


class CompleteTerminalFeedbackTests(unittest.TestCase):
    def setUp(self):
        self.fixture=fixtures.TerminalHVTests()
        self.fixture.setUp()
        self.root=self.fixture.root
        self.bundle=self.fixture.bundle
        info=read_json(self.bundle)
        info['training_instances']=['train']
        write_json(self.bundle,info)

    def tearDown(self):
        self.fixture.tearDown()

    def test_fixed_branch_is_one_predeclared_action_then_balanced(self):
        p=m.FixedIntervention('fixture',5,3,self.bundle)
        summary=np.zeros((5,8))
        specific=np.zeros((5,len(SPECIFIC_NAMES)))
        mask=np.array([True,False,True,True,True])
        self.assertEqual(p.decide(summary,specific,mask).chosen_index,3)
        for _ in range(8):
            self.assertEqual(p.decide(summary,specific,mask).chosen_index,4)
        self.assertEqual(p.fixed_choices,1)
        self.assertEqual(p.switches,1)
        self.assertEqual(p.model_evaluations,0)
        self.assertEqual(p.transition['action'],3)
        other=m.FixedIntervention('fixture',5,1,self.bundle)
        with self.assertRaisesRegex(ValueError,'illegal'):
            other.decide(summary,specific,mask)

    def rows(self):
        return [{'instance':name,'tokens':np.zeros((5,self.fixture.dim)).tolist(),
            'eligible':[True,False,False,True,True],'terminal_hvs':[.8,None,None,1.1,1.]}
            for name in ('train','valid') for _ in range(4)]

    def test_expected_value_is_probability_weighted_not_hindsight_best(self):
        class ConstantPolicy:
            def eval(self):
                return self
            def __call__(self,x,mask):
                return torch.tensor([[0.,-torch.inf,-torch.inf,0.,0.]]).repeat(len(x),1),None
        score=m.expected_score(ConstantPolicy(),self.rows())
        self.assertAlmostEqual(score['mean_expected_terminal_gain'],(-.2+.1)/3,places=6)
        self.assertLess(score['mean_expected_terminal_gain'],0)

    def test_full_feedback_training_uses_actual_gain_and_has_no_ppo_updates(self):
        rows=self.rows()
        write_json(self.root/'DATASET.json',{'rows':rows})
        with patch.object(m,'VALID_INSTANCES',('valid',)),patch.object(m,'EPOCHS',10):
            report=m.fit_complete_feedback(self.bundle,rows,self.root,'fixture')
        self.assertEqual(report['new_ppo_updates'],0)
        initial=report['history'][0]['validation']['mean_expected_terminal_gain']
        self.assertGreater(report['selected']['validation']['mean_expected_terminal_gain'],initial)
        model,saved=checkpoint_model(self.root/'pretrained.pt',digest(self.root/'pretrained.pt'))
        self.assertEqual(saved['execution_mode'],'single-intervention')
        self.assertEqual(saved['update'],0)
        self.assertEqual(report['dataset_sha256'],digest(self.root/'DATASET.json'))

    def test_incomplete_labels_and_validation_leakage_are_rejected(self):
        rows=self.rows()
        rows[0]['terminal_hvs'][3]=None
        with patch.object(m,'VALID_INSTANCES',('valid',)):
            with self.assertRaisesRegex(ValueError,'all legal terminal outcomes'):
                m.fit_complete_feedback(self.bundle,rows,self.root,'fixture')
        rows=self.rows()
        with patch.object(m,'VALID_INSTANCES',('train',)):
            with self.assertRaisesRegex(ValueError,'overlaps'):
                m.fit_complete_feedback(self.bundle,rows,self.root,'fixture')

    def test_branch_pair_rejects_extra_budget_and_another_state(self):
        base={'pairing':{'scenarios':'same','initial':'same'}}
        branch={'pairing':deepcopy(base['pairing']),'first_supported_input_sha256':'same',
            'real_objective_evaluation_count':400,'fixed_choices':1,'closed_loop_switches':1}
        m.assert_terminal_pair(base,branch,'same')
        branch['real_objective_evaluation_count']=401
        with self.assertRaisesRegex(ValueError,'changed state, scenarios, budget'):
            m.assert_terminal_pair(base,branch,'same')


if __name__=='__main__':
    unittest.main()
