from copy import deepcopy
import unittest
from unittest.mock import patch

import numpy as np
import torch

from m_candidate_information import SPECIFIC_NAMES
from m_operator_offline import digest
from m_terminal_hv_ppo import (SingleInterventionPolicy, SingleInterventionBaseline, TerminalPolicy,
    load_checkpoint, save_checkpoint, optimizer_for, update_ppo)
from run_m_terminal_hv_ppo_v1 import report_search, verify_pair, METHODS
from tests import test_m_terminal_hv_ppo as fixtures


class SingleInterventionTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.TerminalHVTests()
        self.fixture.setUp()
        self.bundle, self.checkpoint = self.fixture.bundle, self.fixture.checkpoint
        saved = load_checkpoint(self.checkpoint)
        saved['execution_mode'] = 'single-intervention'
        save_checkpoint(self.checkpoint, saved)
        rng = np.random.default_rng(33)
        self.summary = rng.normal(size=(5,8))
        self.specific = rng.normal(size=(5,len(SPECIFIC_NAMES)))
        self.mask = np.array([True,False,True,True,True])

    def tearDown(self):
        self.fixture.tearDown()

    def policy(self, training):
        return SingleInterventionPolicy('fixture',7,self.bundle,self.checkpoint,digest(self.checkpoint),sample=training)

    def test_training_and_deployment_use_identical_single_sample(self):
        train, test = self.policy(True), self.policy(False)
        first = None
        for _ in range(6):
            a = train.decide(self.summary,self.specific,self.mask)
            b = test.decide(self.summary,self.specific,self.mask)
            self.assertEqual(a.chosen_index,b.chosen_index)
            self.assertEqual(a.probabilities,b.probabilities)
            if first is None:
                first = a
            else:
                self.assertEqual(a.chosen_index,4)
        self.assertEqual(train.model_evaluations,1)
        self.assertEqual(test.model_evaluations,1)
        self.assertEqual(len(train.transitions),1)
        self.assertLessEqual(train.switches,1)

    def test_sampled_balanced_consumes_opportunity_without_retry(self):
        policy = self.policy(True)
        logits = torch.tensor([[-1000.,-torch.inf,-1000.,-1000.,0.]])
        with patch.object(policy.model,'forward',return_value=(logits,torch.zeros(1))):
            choice=policy.decide(self.summary,self.specific,self.mask)
        self.assertEqual(choice.chosen_index,4)
        with patch.object(policy.model,'forward',side_effect=AssertionError('second attempt')):
            for _ in range(5):
                self.assertEqual(policy.decide(self.summary,self.specific,self.mask).chosen_index,4)
        self.assertEqual(len(policy.transitions),1)
        self.assertEqual(policy.switches,0)

    def test_no_alternative_defers_the_one_decision_and_input_is_paired(self):
        policy, baseline = self.policy(True), SingleInterventionBaseline('fixture',7)
        only_balanced = np.array([False,False,False,False,True])
        for mask in (only_balanced,self.mask):
            policy.decide(self.summary,self.specific,mask)
            baseline.decide(self.summary,self.specific,mask)
        self.assertEqual(policy.model_evaluations,1)
        self.assertEqual(policy.first_supported_input_sha256,baseline.first_supported_input_sha256)

    def test_single_checkpoint_cannot_be_deployed_as_repeated_decisions(self):
        with self.assertRaisesRegex(ValueError,'execution mode'):
            TerminalPolicy('fixture',7,self.bundle,self.checkpoint,digest(self.checkpoint))

    def test_single_choice_return_increases_probability_of_rewarded_action(self):
        model=self.fixture.model
        tokens=torch.tensor(np.concatenate([self.summary,self.specific],axis=1)[None],dtype=torch.float32)
        mask=torch.tensor([[False,False,False,True,True]])
        with torch.no_grad():
            logits,values=model(tokens,mask)
            distribution=torch.distributions.Categorical(logits=logits)
            before=float(distribution.probs[0,3])
        episodes=[]
        for action in (3,4)*4:
            transition={'tokens':tokens[0].tolist(),'eligible':mask[0].tolist(),'action':action,
                'old_log_probability':float(distribution.log_prob(torch.tensor([action]))[0]),
                'old_value':float(values[0])}
            episodes.append({'final_hv':1.1 if action==3 else 1.,'balanced_hv':1.,'transitions':[transition]})
        result=update_ppo(model,optimizer_for(model),episodes,1)
        with torch.no_grad():
            after=float(model(tokens,mask)[0].softmax(-1)[0,3])
        self.assertGreater(after,before)
        self.assertEqual(result['actions'],8)
        self.assertEqual(result['complete_episodes'],8)

    def test_search_admits_frozen_sampling_but_rejects_second_intervention(self):
        runs=[]
        for method in METHODS:
            runs.append({'instance':'fixture','algorithm_seed':1,'strategy':method,
                'pairing':{'initial':'same','scenarios':'same'},'real_objective_evaluation_count':400,
                'ppo_updates':0,'counterfactual_evaluation_count':0,'training_feedback_evaluations':0,
                'source_bundle_sha256':'same','candidate_contract':'concrete-proposal-v1',
                'minimum_control_generation':10,'final_hypervolume':1.,'sampling':method!=METHODS[0],
                'collect_for_training':False,'single_intervention':True,
                'learned_decisions':int(method!=METHODS[0]),'first_supported_input_sha256':'same',
                'closed_loop_switches':0,'optimisation_seconds':1.})
        plan={'single_intervention':True,'test_instances':['fixture'],'test_seeds':[1]}
        report=report_search(runs,plan)
        self.assertFalse(report['development_gain_established'])
        bad=deepcopy(runs)
        bad[-1]['learned_decisions']=2
        with self.assertRaisesRegex(ValueError,'count exceeded'):
            report_search(bad,plan)
        bad=deepcopy(runs)
        bad[-1]['first_supported_input_sha256']='different'
        with self.assertRaisesRegex(ValueError,'inputs differ'):
            verify_pair(bad)
        bad=deepcopy(runs)
        bad[-1]['final_hypervolume']=1.01
        with self.assertRaisesRegex(ValueError,'no-switch'):
            verify_pair(bad)


if __name__=='__main__':
    unittest.main()
