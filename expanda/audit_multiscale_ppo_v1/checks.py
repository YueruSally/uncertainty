"""Synthetic saved-artifact tests. No transport search or model training."""
import contextlib
import copy
import io
import tempfile
import unittest
from pathlib import Path
import numpy as np
import audit


class Checks(unittest.TestCase):
    def test_exact_kl_entropy_and_masks(self):
        p=np.array([[.8,.2,0.],[.1,.9,0.]])
        same=audit.compare_probabilities(p,p)
        np.testing.assert_array_equal(same['kl_nats'],[0.,0.])
        np.testing.assert_array_equal(same['top1_changed'],[0.,0.])
        q=np.array([[.2,.8,0.],[.2,.8,0.]])
        r=audit.compare_probabilities(p,q)
        self.assertAlmostEqual(r['kl_nats'][0],.8*np.log(4)+.2*np.log(.25))
        np.testing.assert_array_equal(r['top1_changed'],[1.,0.])
        with self.assertRaises(ValueError):audit.compare_probabilities(p,np.array([[.7,.2,.1],[.1,.9,0.]]))

    def test_sampling_and_signed_advantages(self):
        ids=audit.pool_indices(1000,128,'trajectory-1/I')
        np.testing.assert_array_equal(ids,audit.pool_indices(1000,128,'trajectory-1/I'))
        self.assertEqual(len(set(ids)),128)
        self.assertEqual(len(audit.pool_indices(1000,0,'x')),1000)
        stats=audit.advantage_stats(.5,[.1,.5,.9])
        self.assertAlmostEqual(stats['advantage_mean'],0.)
        self.assertAlmostEqual(stats['positive_fraction'],1/3)

    def test_existing_pairs_not_mean_only(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);folder=root/'screen'/'screen-fixture'/'seed_1'
            r=dict(completed=True,instance='screen-fixture',seed=1,scenario_digest='scenario',
                   instance_digest='instance',plan_sha256='plan',model_bundle_sha256='u0',
                   final_hypervolume=.5,final_front_raw=[[1,2,3]],evaluation_order_digest='sequence',
                   real_objective_evaluation_count=100,selection={'I':{'calls':1}})
            for arm in ('ICM_SHARED_SUPERVISED','ICM_TERMINAL_PPO'):
                p=folder/arm/'RUN_COMPLETE.json';p.parent.mkdir(parents=True);audit.write(p,r)
            plan={'screen':[{'instance_id':'screen-fixture'}],'settings':{'screen_seeds':[1,2]}}
            rows=audit.audit_finished_pairs(root,plan,{'bundle_sha256':'u0'})
            self.assertEqual(len(rows),1);self.assertTrue(rows[0]['all_checked_exact'])
            r['evaluation_order_digest']='different'
            audit.write(folder/'ICM_TERMINAL_PPO/RUN_COMPLETE.json',r)
            self.assertFalse(audit.audit_finished_pairs(root,plan,{'bundle_sha256':'u0'})[0]['all_checked_exact'])

    def test_full_audit_from_synthetic_saved_artifacts(self):
        import sys
        sys.path.insert(0,str(audit.EXPANDA/'joint_icm_multiscale_terminal_v1'))
        from policy import TerminalPolicy,snapshot
        from bridge import SharedTransformer,save_checkpoint,canonical_digest,FEATURE
        from prepare import training_schedule
        import torch
        torch.set_num_threads(1);torch.manual_seed(1)
        with tempfile.TemporaryDirectory() as d:
            base=Path(d);root=base/'experiment';root.mkdir();source=base/'supervision';shared=source/'shared_model';shared.mkdir(parents=True)
            dimensions={c:[3,2,4] for c in 'ICM'}
            gates={c:dict(lambda_value=0.,threshold_scaled=0.,disabled=False) for c in 'ICM'}
            for g in gates.values():g['lambda']=g.pop('lambda_value')
            checkpoints=[]
            for i in range(3):
                p=shared/f'member_{i}.pt';save_checkpoint(p,{'model':SharedTransformer(dimensions,8).state_dict()});checkpoints.append(p.name)
            bundle=dict(feature_version=FEATURE,dimensions=dimensions,settings={'width':8},scalers={},normalization={},
                        target_scales={},selected={'gates':gates},checkpoints=checkpoints,
                        checkpoint_sha256={p:audit.sha(shared/p) for p in checkpoints})
            audit.write(shared/'BUNDLE.json',bundle)
            initial=TerminalPolicy(shared)
            for u in (0,4,8):
                p=copy.deepcopy(initial)
                if u:
                    with torch.no_grad():
                        for model in p.models:model.gain_head.bias.add_(u*.05)
                snapshot(p,root/'terminal/checkpoints'/f'update_{u:03d}')
            entries=[dict(instance_id=f'train-{i}',configuration='S0',development_role='training') for i in range(6)]
            entries += [dict(instance_id=f'valid-{i}',configuration='S0',development_role='validation') for i in range(2)]
            plan=dict(source=str(source),entries=entries,screen=[],files={},settings=dict(
                configurations=['S0'],screen_seeds=[1],ppo=dict(updates=8,episodes_per_update=1,training_seed_base=123)))
            plan['sha256']=canonical_digest(plan);audit.write(root/'PLAN.json',plan)
            schedule=training_schedule(plan);audit.write(root/'terminal/TRAINING_SCHEDULE.json',schedule)
            audit.write(root/'terminal/SELECTED.json',{'update':0,'bundle_sha256':audit.sha(root/'terminal/checkpoints/update_000/BUNDLE.json')})
            rng=np.random.default_rng(7)
            for j in schedule:
                folder=root/'terminal/rollouts'/f'u{j["update"]:03d}'/j['instance'];folder.mkdir(parents=True)
                arrays={}
                for c in 'ICM':
                    for k,shape in [('batch',(6,8,2,3)),('candidate',(6,8,2)),('context',(6,4))]:
                        arrays[c+'_'+k]=rng.normal(size=shape).astype(np.float32)
                    arrays[c+'_marker']=np.zeros((6,8,1),np.float32);arrays[c+'_marker'][:,0]=1
                    arrays[c+'_mask']=np.ones((6,8),bool);arrays[c+'_mask'][:,-1]=False
                    arrays[c+'_action']=np.zeros(6,int);arrays[c+'_value']=np.linspace(0,1,6)
                np.savez_compressed(folder/'TRAJECTORY.npz',**arrays)
                audit.write(folder/'RUN_COMPLETE.json',dict(completed=True,stochastic=True,phase='training',
                    plan_sha256=plan['sha256'],seed=j['seed'],instance=j['instance'],configuration='S0',
                    trajectory_sha256=audit.sha(folder/'TRAJECTORY.npz'),final_hypervolume=.5))
                p=root/'terminal/checkpoints'/f'update_{j["update"]:03d}'/'UPDATE.json';p.parent.mkdir(exist_ok=True)
                audit.write(p,dict(update=j['update'],optimizer_steps=2,kl_early_stop=False,
                    last_minibatch_kl=.001,on_policy_logp_max_error=0,actions=18,complete_episodes=1))
            for u in (0,4,8):
                for name in ('valid-0','valid-1'):
                    p=root/'terminal/validation'/f'u{u:03d}'/name/'RUN_COMPLETE.json';p.parent.mkdir(parents=True)
                    audit.write(p,dict(completed=True,seed=42,scenario_digest='same',final_hypervolume=.5-u*.001))
            before={str(p):audit.sha(p) for p in root.rglob('*') if p.is_file()}
            with contextlib.redirect_stdout(io.StringIO()):result=audit.execute(root,base/'audit',4)
            self.assertEqual(before,{str(p):audit.sha(p) for p in root.rglob('*') if p.is_file()})
            self.assertTrue(result['shared_update0_actor_tensors_exact'])
            self.assertEqual(result['shared_update0_replayed_choice_differences'],0)
            self.assertEqual(len(result['policy_change']),6)
            self.assertTrue(all(r['sampled_pools']==32 for r in result['policy_change']))
            self.assertTrue(all(r['kl_nats']>0 for r in result['policy_change']))
            self.assertEqual(result['new_searches'],0)


if __name__=='__main__':unittest.main()
