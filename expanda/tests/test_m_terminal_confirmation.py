from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

from m_operator_offline import write_json, digest
from m_terminal_hv_ppo import save_checkpoint
from validation.run_m_terminal_hv_confirm_v1 import freeze_development, hierarchical_summary, check_pair, METHODS


class TerminalConfirmationTests(unittest.TestCase):
    def test_no_final_test_when_development_did_not_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_json(root / 'PLAN.json', {'protocol': 'terminal-hv-ppo-v1'})
            write_json(root / 'QUICK_RESULT.json', {'completed': True, 'development_pass': False})
            write_json(root / 'SELECTED_POLICY.json', {'update': 1})
            (root / 'PIPELINE_COMPLETE').write_text('complete')
            with self.assertRaisesRegex(ValueError, 'has not established'):
                freeze_development(root)

    def test_selected_single_policy_retains_development_execution_mode(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            bundle=root/'BUNDLE.json'
            write_json(bundle,{'fixture_only':True})
            checkpoint=root/'selected.pt'
            metadata={'protocol':'terminal-hv-ppo-v1','execution_mode':'single-intervention',
                'update':4,'bundle_sha256':digest(bundle)}
            save_checkpoint(checkpoint,metadata)
            selected={'checkpoint':str(checkpoint),'sha256':digest(checkpoint),'update':4}
            plan={'protocol':'single-intervention-ppo-v2','single_intervention':True,
                'input_sha256':{str(bundle):digest(bundle)}}
            write_json(root/'PLAN.json',plan)
            write_json(root/'SELECTED_POLICY.json',selected)
            write_json(root/'QUICK_RESULT.json',{'completed':True,'development_pass':True,'selected':selected})
            (root/'PIPELINE_COMPLETE').write_text('complete')
            for protocol in ('single-intervention-ppo-v2','complete-terminal-feedback-v3'):
                with self.subTest(protocol=protocol):
                    plan['protocol']=protocol
                    write_json(root/'PLAN.json',plan)
                    self.assertEqual(freeze_development(root),(plan,selected,bundle))
            # A valid hash is insufficient if the checkpoint has different execution semantics.
            metadata['execution_mode']='repeated-decisions'
            save_checkpoint(checkpoint,metadata)
            selected['sha256']=digest(checkpoint)
            write_json(root/'SELECTED_POLICY.json',selected)
            write_json(root/'QUICK_RESULT.json',{'completed':True,'development_pass':True,'selected':selected})
            with self.assertRaisesRegex(ValueError,'execution mode/update/bundle'):
                freeze_development(root)

    def test_instance_resampling_keeps_between_instance_uncertainty(self):
        rows = []
        for name, value in [('a',1.2),('b',.9)]:
            for seed in range(20):
                rows.append({'instance':name,'seed':seed,'methods':{
                    'BASE':{'hv':1.},'HEURISTIC_BALANCED':{'hv':1.},'TRANSFORMER_PPO':{'hv':value}}})
        result=hierarchical_summary(rows,['a','b'],list(range(20)),2000)
        pooled=result['pooled_equal_instance']['ppo_vs_balanced']
        self.assertAlmostEqual(pooled['mean'],5.)
        self.assertLess(pooled['hierarchical_95ci_low'],0)
        self.assertGreater(pooled['hierarchical_95ci_high'],0)
        self.assertEqual(pooled['positive_instances'],1)

    def test_all_predeclared_pairs_required_and_reference_not_moved(self):
        row={'instance':'a','seed':7,'methods':{m:{'hv':1.} for m in METHODS}}
        with self.assertRaisesRegex(ValueError,'missing or duplicate'):
            hierarchical_summary([row],['a'],[7,8])
        with self.assertRaisesRegex(ValueError,'missing or duplicate'):
            hierarchical_summary([row,row],['a'],[7])
        row['methods']['BASE']['hv']=0
        with self.assertRaisesRegex(ValueError,'reference will not be moved'):
            hierarchical_summary([row],['a'],[7])

    def test_pairing_rejects_scenario_and_budget_changes(self):
        row={'pairing':{'scenarios':'same','initial_population':['one']},'real_objective_evaluation_count':400,
             'candidate_contract':'concrete-proposal-v1','ppo_updates':0}
        group={m:deepcopy(row) for m in METHODS}
        check_pair(group)
        group['TRANSFORMER_PPO']['pairing']['scenarios']='different'
        with self.assertRaisesRegex(ValueError,'different initial'):
            check_pair(group)
        group['TRANSFORMER_PPO']=deepcopy(row)
        group['TRANSFORMER_PPO']['real_objective_evaluation_count']=401
        with self.assertRaisesRegex(ValueError,'different initial'):
            check_pair(group)

    def test_single_confirmation_rejects_greedy_or_multiple_decisions(self):
        group={m:{'pairing':{'initial':'same','scenario':'same'},'real_objective_evaluation_count':400,
            'candidate_contract':'concrete-proposal-v1','single_intervention':True,
            'sampling':m=='TRANSFORMER_PPO','collect_for_training':False,
            'learned_decisions':int(m=='TRANSFORMER_PPO'),'first_supported_input_sha256':'same',
            'closed_loop_switches':0,'final_hypervolume':1.} for m in METHODS}
        check_pair(group)
        for field,value in [('sampling',False),('learned_decisions',2),('collect_for_training',True)]:
            bad=deepcopy(group)
            bad['TRANSFORMER_PPO'][field]=value
            with self.assertRaisesRegex(ValueError,'one frozen stochastic decision'):
                check_pair(bad)
        bad=deepcopy(group)
        bad['TRANSFORMER_PPO']['first_supported_input_sha256']='changed'
        with self.assertRaisesRegex(ValueError,'state differs'):
            check_pair(bad)
        bad=deepcopy(group)
        bad['TRANSFORMER_PPO']['final_hypervolume']=1.1
        with self.assertRaisesRegex(ValueError,'no-switch'):
            check_pair(bad)


if __name__=='__main__':
    unittest.main()
