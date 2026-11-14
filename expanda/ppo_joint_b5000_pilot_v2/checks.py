"""Synthetic/unit and real small-engine checks; none are experimental evidence."""
from policy_pilot import *
import contextlib
import importlib.util
import subprocess
import shutil
import tempfile
import unittest
from unittest.mock import patch

spec=importlib.util.spec_from_file_location('checked_pilot',HERE/'run.py')
runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)


class UnitChecks(unittest.TestCase):
    def test_advantage_normalization_is_episode_equal_not_action_count_weighted(self):
        arrays=[{c+'_value':np.zeros(n) for c in COMPONENTS} for n in (2,100)]
        result=advantage_statistics(arrays,[-1.,1.])
        for c in COMPONENTS:
            self.assertEqual(result[c]['mean'],0.);self.assertEqual(result[c]['std'],1.)
            self.assertEqual(result[c]['normalized_mean'],0.)
            self.assertEqual(result[c]['normalized_second_moment'],1.)

    def test_pairing_rejects_scenarios_and_missing_pairs(self):
        a=dict(instance='a',configuration='S0',seed=1,instance_digest='i',scenario_digest='s',
            evaluation_budget=5000,completed=True,final_hypervolume=.5)
        b=dict(a,final_hypervolume=.6)
        result=runner.paired([b],[a],draws=20)
        self.assertAlmostEqual(result['mean_paired_percent'],20.)
        with self.assertRaises(ValueError):runner.paired([dict(b,scenario_digest='other')],[a],draws=2)
        with self.assertRaises(ValueError):runner.paired([b],[],draws=2)
        zero=runner.paired([b],[dict(a,final_hypervolume=0.)],draws=2)
        self.assertIsNone(zero['mean_paired_percent'])

    def test_protocol_does_not_select_from_confirmation(self):
        p=read_json(HERE/'protocol.json')
        self.assertEqual(p['configurations'],['S0','S4','S8'])
        self.assertEqual(p['new_searches'],2+2*3*3+3*3*3+3*3*3)
        self.assertEqual(p['new_objective_evaluations_maximum'],p['new_searches']*5000)
        self.assertEqual(p['episodes_per_update'],len(p['configurations']))

    def test_exact_categorical_kl_and_masks(self):
        p=torch.tensor([[.2,.8,0.]]);q=torch.tensor([[.4,.6,0.]])
        self.assertAlmostEqual(categorical_kl(p,q).item(),.2*np.log(.5)+.8*np.log(4/3),places=6)
        self.assertEqual(categorical_kl(p,p).item(),0.)

    def test_critic_loss_does_not_change_actor_representation(self):
        torch.manual_seed(4);m=DetachedCritic({c:(2,3,4) for c in COMPONENTS},width=8)
        x=(torch.randn(2,8,5,2),torch.randn(2,8,3),torch.randn(2,4),torch.randn(2,8,1))
        _,v=m('I',*x);(v-1.).square().mean().backward()
        for name,p in m.named_parameters():
            if not name.startswith('value_head.'):
                self.assertTrue(p.grad is None or not torch.count_nonzero(p.grad),name)
        self.assertGreater(float(m.value_head.weight.grad.abs().sum()),0.)

    def test_union_report_uses_all_budgets_and_does_not_change_source(self):
        def row(point):
            return dict(instance='a',configuration='S0',seed=1,instance_digest='i',scenario_digest='s',
                evaluation_budget=5000,completed=True,final_hypervolume=.2,optimisation_seconds=1,
                model_bundle_sha256='model',budget_fronts=[dict(evaluation=b,hv=.2,front_raw=[p])
                    for b,p in zip((400,1000,5000),([9.,9.,9.],[4.,4.,4.],point))])
        groups={'共享监督_最高分':[row([2.,2.,2.])],'PPO_最高分':[row([1.,1.,1.])]}
        original=copy.deepcopy(groups)
        with tempfile.TemporaryDirectory() as d:
            with contextlib.redirect_stdout(__import__('io').StringIO()):runner.union_report(groups,Path(d))
            bounds=read_json(Path(d)/'INSTANCE_BOUNDS.json')['a']
            self.assertEqual(bounds['minimum'],[1.,1.,1.]);self.assertEqual(bounds['maximum'],[9.,9.,9.])
            q=read_json(Path(d)/'QUICK_RESULT.json')
            stat=q['comparisons']['PPO_最高分/共享监督_最高分/5000']
            self.assertAlmostEqual(stat['mean_hv'],1.331)
            self.assertTrue((Path(d)/'00_配对结果_实例联合归一化.csv').exists())
        self.assertEqual(groups,original)


class EngineChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1);cls.temp=tempfile.TemporaryDirectory(prefix='ppo-pilot-v2-check-');cls.root=Path(cls.temp.name)
        cls.source=cls.root/'source';s=read_json(SUP/'protocol.json')
        s.update(population=20,budget=400,states={'I':6,'C':3,'M':6},milestones=[100,200,400],
            epochs=1,checkpoint_epochs=[1],training_seeds=[11,12,13],configurations=['S0'],batch_size=4)
        cls.settings=s;info=read_json(EXPANDA/'formal_icm_s0_v1/assets/search_instances/prescreen-search-S0-001.json')
        info.pop('instance_digest');info.update(instance_id='training-fixture',development_role='training');info['instance_digest']=canonical_digest(info)
        cls.train=copy.deepcopy(info);write_json(cls.source/'instances/training-fixture.json',info);write_json(cls.root/'settings.json',s)
        code="import sys;from pathlib import Path;sys.path.insert(0,sys.argv[1]);from bootstrap import read_json;from collection import work;r=Path(sys.argv[2]);o=Path(sys.argv[4]);o.mkdir(parents=True);work(r,read_json(Path(sys.argv[3])),'training-fixture',12345,sys.argv[5],o,'fixture')"
        for arm in ('COLLECT_I','COLLECT_CM'):
            with (cls.root/(arm+'.log')).open('w') as log:
                args=[sys.executable,'-c',code,str(SUP),str(cls.source),str(cls.root/'settings.json'),str(cls.source/'collection/training-fixture'/arm),arm]
                r=subprocess.run(args,stdout=log,stderr=subprocess.STDOUT)
            if r.returncode:raise RuntimeError((cls.root/(arm+'.log')).read_text()[-5000:])
        info.pop('instance_digest');info.update(instance_id='validation-fixture',development_role='validation');info['instance_digest']=canonical_digest(info)
        cls.valid=copy.deepcopy(info);write_json(cls.source/'instances/validation-fixture.json',info)
        for arm in ('COLLECT_I','COLLECT_CM'):
            d=read_json(cls.source/'collection/training-fixture'/arm/'DATA.json');d.update(instance='validation-fixture',role='validation')
            for r in d['rows']:r['instance']='validation-fixture'
            write_json(cls.source/'collection/validation-fixture'/arm/'DATA.json',d)
        from data import load_dataset
        from training import fit
        folders=[cls.source/'collection'/n/arm for n in ('training-fixture','validation-fixture') for arm in ('COLLECT_I','COLLECT_CM')]
        rows,norm=load_dataset(folders,50000);cls.norm=norm;cls.rows=rows
        with (cls.root/'fit.log').open('w') as log,contextlib.redirect_stdout(log):
            fit(rows,norm,s,cls.source/'shared_model',{},dict(test_fixture=True))
        # Test-only gate activation exercises all three branches; never exported as research evidence.
        b=read_json(cls.source/'shared_model/BUNDLE.json')
        for g in b['selected']['gates'].values():g.update(disabled=False,**{'lambda':0.,'threshold_scaled':0.})
        write_json(cls.source/'shared_model/BUNDLE.json',b)
        policy=PilotPolicy(cls.source/'shared_model');p=read_json(HERE/'protocol.json')
        p.update(configurations=['S0'],updates=1,episodes_per_update=1,optimizer_steps=3,minibatch_per_component=4,
            training_seed_base=12345,target_kl=1.,temperatures={c:.5 for c in COMPONENTS})
        policy.bundle['pilot']=p;cls.p=p;cls.model=cls.root/'checkpoints/update_000';snapshot(policy,cls.model)
        cls.plan=dict(settings=copy.deepcopy(s),pilot=p,normalization=norm,sha256='fixture',files={},entries=[cls.train,cls.valid],
            training=[cls.train],tuning=[cls.valid],confirmation=[cls.valid])
        cls.plan['settings']['ppo']={'validation_seeds':[12345]}
        write_json(cls.root/'fixture_plan.json',cls.plan)
        for entry in cls.plan['entries']:write_json(cls.root/'instances'/(entry['instance_id']+'.json'),entry)
        from search_b5000 import run_search
        task=dict(instance='validation-fixture',seed=12345,arm='ICM_SHARED_SUPERVISED',phase='validation',sample=False,
            model=str(cls.source/'shared_model'),bundle_sha256=digest(cls.source/'shared_model/BUNDLE.json'),normalization=norm)
        old=cls.root/'old_validation';old.mkdir()
        write_json(cls.root/'old_task.json',task)
        code="import sys;from pathlib import Path;sys.path.insert(0,sys.argv[1]);from search_b5000 import run_search,read_json;r=Path(sys.argv[2]);run_search(r/'instances/validation-fixture.json',read_json(r/'settings.json'),read_json(r/'old_task.json'),r/'old_validation','fixture')"
        with (cls.root/'old.log').open('w') as log:
            done=subprocess.run([sys.executable,'-c',code,str(runner.B5),str(cls.root)],stdout=log,stderr=subprocess.STDOUT)
        if done.returncode:raise RuntimeError((cls.root/'old.log').read_text()[-5000:])
        cls.original=read_json(old/'RUN_COMPLETE.json')
        t=runner.make_task(cls.plan,cls.valid,12345,cls.model,'reproduction',False,False,0)
        cls.replay=cls.worker(t,cls.root/'runs/reproduction/u000/greedy/validation-fixture/seed_12345')
        t=runner.make_task(cls.plan,cls.train,12345,cls.model,'training',True,True,1)
        cls.trajectory_folder=cls.root/'runs/training/u001/training-fixture/seed_12345';cls.rollout=cls.worker(t,cls.trajectory_folder)
        t=runner.make_task(cls.plan,cls.train,12345,cls.model,'training_reference',False,False,1)
        cls.baseline=cls.worker(t,cls.root/'runs/training_reference/u001/training-fixture/seed_12345')
        print('Pilot v2 engine checks: '+str(cls.root),flush=True)

    @classmethod
    def worker(cls,t,folder):
        folder.mkdir(parents=True)
        taskfile=folder.with_suffix('.task.json');write_json(taskfile,t)
        code="import sys,importlib.util;from pathlib import Path;sys.path.insert(0,sys.argv[1]);s=importlib.util.spec_from_file_location('pilot_test_worker',Path(sys.argv[1])/'run.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m);r=Path(sys.argv[2]);m.run_worker(r,m.read_json(r/'fixture_plan.json'),m.read_json(Path(sys.argv[3])),Path(sys.argv[4]))"
        with folder.with_suffix('.log').open('w') as log:
            done=subprocess.run([sys.executable,'-c',code,str(HERE),str(cls.root),str(taskfile),str(folder)],stdout=log,stderr=subprocess.STDOUT)
        if done.returncode:raise RuntimeError(folder.with_suffix('.log').read_text()[-5000:])
        return read_json(folder/'RUN_COMPLETE.json')

    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()

    def test_u0_greedy_matches_shared_full_run_and_all_snapshots(self):
        for key in ('final_front_raw','final_hypervolume','evaluation_order_digest','real_objective_evaluation_count','selection'):
            self.assertEqual(self.original[key],self.replay[key],key)
        self.assertEqual([(r['evaluation'],r['front_raw']) for r in self.original['budget_fronts']],
            [(r['evaluation'],r['front_raw']) for r in self.replay['budget_fronts']])

    def test_training_pair_uses_same_instance_scenario_budget(self):
        runner.check_pair(self.rollout,self.baseline)
        self.assertIn(self.rollout['real_objective_evaluation_count'],(399,400))
        self.assertEqual(self.rollout['hidden_candidate_objective_evaluations'],0)
        self.assertTrue(self.rollout['trajectory_sha256'])
        with np.load(self.trajectory_folder/'TRAJECTORY.npz') as data:
            for c in COMPONENTS:
                self.assertGreater(len(data[c+'_action']),0)
                self.assertTrue(np.allclose(data[c+'_prob'].sum(-1),1))

    def test_paired_ppo_changes_actor_balances_components_and_can_restore_optimizer(self):
        policy=PilotPolicy(self.model);reference=PilotPolicy(self.model);optimizer=optimizer_for(policy,self.p)
        path=self.trajectory_folder/'TRAJECTORY.npz'
        # Controlled rewards test the optimizer, not claimed search performance.
        metrics=update(policy,reference,[(path,-.2),(path,.3)],self.p,optimizer,123)
        self.assertEqual(metrics['optimizer_steps'],3)
        self.assertEqual(metrics['component_minibatches'],dict(I=3,C=3,M=3))
        self.assertGreater(metrics['actor_parameter_l2_change'],0.)
        self.assertLess(metrics['on_policy_logp_max_error'],3e-5)
        for v in metrics['advantage'].values():
            self.assertAlmostEqual(v['normalized_mean'],0.)
            self.assertAlmostEqual(v['normalized_second_moment'],1.)
        dst=self.root/'optimizer_roundtrip';snapshot(policy,dst,optimizer,metrics)
        restored=PilotPolicy(dst);other=optimizer_for(restored,self.p);other.load_state_dict(load_checkpoint(dst/'optimizer.pt'))
        for a,b in zip(policy.models,restored.models):
            for key,v in a.state_dict().items():self.assertTrue(torch.equal(v,b.state_dict()[key]))
        self.assertEqual(set(other.state_dict()['state']),set(optimizer.state_dict()['state']))

    def test_temperature_calibration_is_train_states_only_and_preserves_greedy(self):
        temperatures,record=calibrate(self.source/'shared_model',[self.trajectory_folder/'TRAJECTORY.npz'],self.p)
        policy=PilotPolicy(self.model);shared=SharedPolicy(self.source/'shared_model')
        policy.bundle['pilot']['temperatures']=temperatures
        for c in COMPONENTS:
            self.assertIn(temperatures[c],self.p['temperature_grid'])
            self.assertGreater(record['components'][c]['states'],0)
            for row in self.rows[c]['validation']:self.assertEqual(policy.select(row),shared.select(row))

    def test_sampled_confirmation_is_not_training_and_does_not_save_training_data(self):
        freeze(self.root/'SELECTION.json',{'best_updated_checkpoint':1})
        t=runner.make_task(self.plan,self.valid,12345,self.model,'confirmation',True,False,0)
        folder=self.root/'runs/confirmation/u000/sampled/validation-fixture/seed_12345'
        r=self.worker(t,folder)
        self.assertTrue(r['stochastic']);self.assertEqual(r['phase'],'confirmation')
        self.assertIsNone(r['trajectory_sha256']);self.assertFalse((folder/'TRAJECTORY.npz').exists())

    def test_cli_scheduler_and_completed_job_resume(self):
        root=self.root/'cli_scheduler';root.mkdir()
        plan=copy.deepcopy(self.plan);plan.pop('sha256');plan['sha256']=canonical_digest(plan)
        write_json(root/'PLAN.json',plan)
        for entry in plan['entries']:write_json(root/'instances'/(entry['instance_id']+'.json'),entry)
        model=root/'checkpoints/update_000';model.parent.mkdir();shutil.copytree(self.model,model)
        t=runner.make_task(plan,self.valid,12345,model,'reproduction',False,False,0)
        folder=root/'runs/reproduction/u000/greedy/validation-fixture/seed_12345'
        rows=runner.scheduler.execute_jobs(root,plan,[(t,folder)],1,'unit/reproduction')
        self.assertEqual(rows[0]['final_front_raw'],self.original['final_front_raw'])
        stamp=(folder/'RUN_COMPLETE.json').stat().st_mtime_ns
        with patch.object(runner.scheduler.subprocess,'Popen',side_effect=AssertionError('Resume launched a duplicate worker')):
            resumed=runner.scheduler.execute_jobs(root,plan,[(t,folder)],1,'unit/resume')
        self.assertEqual(rows,resumed);self.assertEqual(stamp,(folder/'RUN_COMPLETE.json').stat().st_mtime_ns)

    def test_validation_instance_cannot_enter_training(self):
        t=runner.make_task(self.plan,self.valid,12345,self.model,'training',True,True,1)
        with self.assertRaisesRegex(ValueError,'training role'):
            runner.validate_task(self.root,self.plan,t,self.root/'unwanted')


if __name__=='__main__':unittest.main()
