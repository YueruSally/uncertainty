"""Synthetic audit checks: no search and no objective evaluations."""
from pathlib import Path
import contextlib
import copy
import csv
import io
import json
import shutil
import sys
import tempfile
import unittest
import numpy as np
import audit


class PairedChecks(unittest.TestCase):
    def fixture(self):
        plan={'instances':['a','b','c'],'seeds':list(range(10))}
        rows={}
        # The same baseline varies widely. Paired differences stay fixed.
        for i,n in enumerate(plan['instances']):
            for s in plan['seeds']:
                base=10+s+i;delta=(.1,.1,-.5)[i]
                for m,v in ((audit.INDEPENDENT,base),(audit.SHARED,base+delta),(audit.PPO,base+delta)):
                    rows[n,s,m]={'hv':v,'old_hv':v*2}
        return plan,rows

    def test_paired_constant_differences_and_concentrated_loss(self):
        plan,rows=self.fixture();overall,per,detail=audit.paired_hv(rows,plan,bootstrap=2000)
        self.assertAlmostEqual(overall[0]['平均HV差'],-.1)
        self.assertEqual((overall[0]['胜'],overall[0]['负']),(20,10))
        self.assertEqual(overall[0]['均值为负的实例数'],1)
        self.assertAlmostEqual(per[2]['去掉本实例后其余实例平均差'],.1)
        self.assertAlmostEqual(per[0]['95%区间下限'],.1)
        self.assertAlmostEqual(per[0]['95%区间上限'],.1)
        self.assertEqual(overall[1]['平'],30)
        self.assertEqual(overall[1]['95%区间下限'],0.)
        self.assertEqual(len(detail),120)
        self.assertAlmostEqual(overall[2]['平均HV差'],2*overall[0]['平均HV差'])

    def test_zero_reference_is_valid_and_missing_or_duplicate_pairs_fail(self):
        plan,rows=self.fixture();key=('a',0,audit.INDEPENDENT);rows[key]={'hv':0.,'old_hv':0.}
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'results').mkdir()
            items=[{'instance':n,'algorithm_seed':s,'method':m,**v} for (n,s,m),v in rows.items()]
            p=root/'results/PER_RUN.csv';audit.csv_write(p,items)
            self.assertEqual(audit.read_hv(root,plan)[key]['hv'],0.)
            audit.csv_write(p,items[:-1])
            with self.assertRaisesRegex(ValueError,'Missing'):audit.read_hv(root,plan)
            audit.csv_write(p,items+items[:1])
            with self.assertRaisesRegex(ValueError,'Duplicate'):audit.read_hv(root,plan)

    def test_hv_only_preserves_inputs_and_writes_absolute_differences(self):
        plan,rows=self.fixture()
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)/'source';out=Path(d)/'audit';(root/'results').mkdir(parents=True);(root/'terminal').mkdir()
            plan.update({'settings':{'population':20,'budget':400},'sha256':'fixture'})
            for p,value in [('PLAN.json',plan),('STATUS.json',{'completed':True}),
                            ('results/QUICK_RESULT.json',{'completed':True}),
                            ('results/INPUT_MANIFEST.json',{'plan_sha256':'fixture','files':{}}),
                            ('terminal/SELECTED.json',{'update':4})]:audit.write(root/p,value)
            audit.csv_write(root/'results/PER_RUN.csv',[{'instance':n,'algorithm_seed':s,'method':m,**v} for (n,s,m),v in rows.items()])
            before={str(p):audit.sha(p) for p in root.rglob('*') if p.is_file()}
            with contextlib.redirect_stdout(io.StringIO()):result=audit.execute(root,out,True)
            self.assertEqual(before,{str(p):audit.sha(p) for p in root.rglob('*') if p.is_file()})
            self.assertTrue(result['completed']);self.assertEqual(result['new_objective_evaluations'],0)
            self.assertEqual(result['new_searches'],0)
            with self.assertRaises(ValueError):audit.execute(root,out,True)


class ReplayChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'joint_icm_v1'))
        import torch
        from shared_policy import SharedPolicy
        from shared_model import SharedTransformer,fit_scaler,pack,order
        from joint_common import PACKAGE,PROTOCOL
        cls.torch=torch;cls.Policy=SharedPolicy;cls.pack=staticmethod(pack);cls.order=staticmethod(order)
        torch.set_num_threads(1);torch.manual_seed(1)
        cls.tmp=tempfile.TemporaryDirectory();cls.root=Path(cls.tmp.name);rng=np.random.default_rng(55)
        cls.rows={}
        for c in 'ICM':
            b=audit.read(PACKAGE/'assets'/c/'BUNDLE.json');s=(b['models']['200'] if c=='M' else b)['scaler']
            d=[len(s[k]['mean']) for k in ('batch','candidate','context')]
            cls.rows[c]=[{'candidate_batch_tokens':rng.normal(size=(8,3,d[0])).tolist(),
                'candidate_features':rng.normal(size=(8,d[1])).tolist(),'population_points':[[.1,.2,.3]],
                'slot_fraction':.2,'generation':11,'evaluation':220,
                'labels':[{'hv_contribution':j/100} for j in range(8)]} for _ in range(3)]
        scalers={c:fit_scaler(c,cls.rows[c]) for c in 'ICM'}
        dims={c:[len(scalers[c][k]['mean']) for k in ('batch','candidate','context')] for c in 'ICM'}
        for folder,bias in [('shared',1.),('ppo',-1.)]:
            dest=cls.root/folder;dest.mkdir();paths=[]
            for i in range(3):
                model=SharedTransformer(dims,16)
                with torch.no_grad():model.gain_head.weight.zero_();model.gain_head.bias.fill_(bias)
                p=dest/f'member_{i}.pt';paths.append(p.name)
                torch.save({'model':model.state_dict(),'dimensions':dims,'width':16},p)
            bundle={'protocol':PROTOCOL,'normalization_sha256':audit.sha(PACKAGE/'assets/reward_normalization.json'),
                    'checkpoints':paths,'checkpoint_sha256':{p:audit.sha(dest/p) for p in paths},'scalers':scalers,
                    'selected':{'gates':{c:{'lambda':0.,'threshold_scaled':0.} for c in 'ICM'}},
                    'validation_instances':{c:['valid'] for c in 'ICM'},'files':{}}
            audit.write(dest/'BUNDLE.json',bundle)
        cls.shared=SharedPolicy(cls.root/'shared');cls.ppo=SharedPolicy(cls.root/'ppo')

    @classmethod
    def tearDownClass(cls):cls.tmp.cleanup()

    def trajectory(self,stochastic):
        policy=self.Policy(self.root/'shared',stochastic,33,True)
        for c in 'ICM':
            for row in self.rows[c]:policy.choose(c,row)
        p=self.root/'events.npz';policy.save_events(p)
        with np.load(p,allow_pickle=False) as a:return {k:a[k].copy() for k in a.files}

    def test_actual_deployment_replay_and_owner_probability(self):
        for stochastic in (False,True):
            a=self.trajectory(stochastic)
            rows=audit.replay_arrays(a,self.shared,self.ppo,self.shared,stochastic,{'pool_source':'fixture'})
            self.assertEqual(len(rows),3)
            for r in rows:
                self.assertEqual(r['different_choice_fraction'],1.)
                self.assertEqual(r['shared_rule_choices'],0);self.assertEqual(r['ppo_rule_choices'],3)
                self.assertLess(r['owner_logp_max_error'],2e-5)
            same=audit.replay_arrays(a,self.shared,self.shared,self.shared,stochastic,{})
            self.assertTrue(all(r['different_choices']==0 for r in same))

    def test_corrupted_owning_probabilities_rejected(self):
        a=self.trajectory(True);a['I_logp']+=1.
        with self.assertRaisesRegex(ValueError,'owning policy'):
            audit.replay_arrays(a,self.shared,self.ppo,self.shared,True,{})

    def test_validation_labels_not_inputs_and_known_one_step_gain(self):
        for c in 'ICM':
            x=self.pack(c,self.rows[c],self.shared.bundle['scalers'][c]);a={c+'_'+k:t.numpy() for k,t in zip(('batch','candidate','context'),x)}
            labels=np.asarray([[r['labels'][j]['hv_contribution'] for j in self.order(c,r)] for r in self.rows[c]])
            a[c+'_label_hv']=labels
            r=audit.replay_arrays(a,self.shared,self.ppo,None,False,{})[0]
            self.assertAlmostEqual(r['recorded_one_step_mean_hv_difference'],float((labels[:,0]-labels[:,1]).mean()))
            self.assertIsNone(r['owner_logp_max_error'])
            a[c+'_label_hv']*=1000
            other=audit.replay_arrays(a,self.shared,self.ppo,None,False,{})[0]
            self.assertEqual(other['different_choices'],r['different_choices'])

    def test_gates_scalers_and_changed_actor_weights(self):
        gates,weights=audit.gate_audit(self.shared,self.ppo)
        self.assertTrue(all(r['same_gate'] for r in gates));self.assertGreater(weights['changed_actor_tensors'],0)
        old=copy.deepcopy(self.ppo.bundle['scalers']);self.ppo.bundle['scalers']['I']['clip']=1.
        try:
            with self.assertRaisesRegex(ValueError,'scalers'):audit.gate_audit(self.shared,self.ppo)
        finally:self.ppo.bundle['scalers']=old

    def test_replay_inventory_keeps_validation_training_and_missing_development_separate(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);out=root/'audit';out.mkdir()
            files={}
            for c,protocol in [('I','i-k8-scale-data-v1'),('C','c-k8-scale-data-v1'),('M','m-prescreen-scale-data-v1')]:
                collection=root/('collection_'+c);folder=collection/'runs/valid';folder.mkdir(parents=True)
                audit.write(collection/'PLAN.json',{'protocol':protocol})
                audit.write(folder/'FEEDBACK.json',{'rows':self.rows[c]})
                for p in (collection/'PLAN.json',folder/'FEEDBACK.json'):files[str(p)]=audit.sha(p)
            models={}
            for name,src in [('shared_model','shared'),('terminal/checkpoints/update_000','shared'),('terminal/checkpoints/update_001','ppo')]:
                dest=root/name;shutil.copytree(self.root/src,dest)
                b=audit.read(dest/'BUNDLE.json');b['files']=files;audit.write(dest/'BUNDLE.json',b);models[name]=dest
            shared=models['shared_model'];ppo=models['terminal/checkpoints/update_001']
            audit.write(shared/'READY.json',{'plan_sha256':'fixture','bundle_sha256':audit.sha(shared/'BUNDLE.json')})
            audit.write(root/'terminal/SELECTED.json',{'update':1,'model':str(ppo),'bundle_sha256':audit.sha(ppo/'BUNDLE.json')})
            plan={'sha256':'fixture','instances':['screen'],'seeds':[99],'train_instances':['train'],
                  'runtime_files':{},'environment':{'torch':str(self.torch.__version__),'numpy':str(np.__version__)}}
            counts={c:{'calls':3,'rule_choices':0,'switches':3,'feature_values':1,'clipped_values':0} for c in 'ICM'}
            for method,model in [(audit.SHARED,shared),(audit.PPO,ppo)]:
                folder=root/'screen/screen/seed_99'/method;folder.mkdir(parents=True)
                audit.write(folder/'RUN_COMPLETE.json',{'completed':True,'instance':'screen','algorithm_seed':99,'strategy':method,
                            'stochastic':False,'plan_sha256':'fixture','model_bundle_sha256':audit.sha(model/'BUNDLE.json'),'shared_policy_counts':counts})
            rollout=root/'terminal/rollouts/u001_0_train';rollout.mkdir(parents=True)
            np.savez_compressed(rollout/'TRAJECTORY.npz',**self.trajectory(True))
            audit.write(rollout/'RUN_COMPLETE.json',{'completed':True,'instance':'train','algorithm_seed':100,'stochastic':True,
                        'plan_sha256':'fixture','model_bundle_sha256':audit.sha(models['terminal/checkpoints/update_000']/'BUNDLE.json'),
                        'shared_policy_counts':counts})
            with contextlib.redirect_stdout(io.StringIO()):result=audit.replay(root,plan,out,{})
            self.assertFalse(result['complete_development_candidate_replay_available'])
            self.assertEqual(result['development_runs_with_full_inputs'],0)
            self.assertEqual(result['validation_pool_instances_per_component'],{c:1 for c in 'ICM'})
            self.assertEqual({r['候选来源'] for r in result['same_pool_summary']},{'training','validation'})
            self.assertTrue(all(r['不同选择比例']==1. for r in result['same_pool_summary']))


if __name__=='__main__':unittest.main()
