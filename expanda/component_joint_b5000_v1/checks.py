"""Small actual engine runs and scheduler tests, not research evidence."""
from pathlib import Path
import sys
B5_DIR=Path(__file__).resolve().parent
TERMINAL_DIR=B5_DIR.parent/'joint_icm_multiscale_terminal_v1'
sys.path.insert(0,str(TERMINAL_DIR))
from bridge import *
import importlib.util
spec=importlib.util.spec_from_file_location('b5000_checked_runner',B5_DIR/'run.py')
runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
legacy=runner.legacy
from search_b5000 import ARMS
model_for,task=runner.model_for,runner.task
verify_reproduction,validation_score=runner.verify_reproduction,runner.validation_score
from search_b5000 import enabled_components
from policy import TerminalPolicy,snapshot
import tempfile
import subprocess
import unittest
from unittest.mock import patch


class Checks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1);cls.temp=tempfile.TemporaryDirectory(prefix='b5000-check-');cls.root=Path(cls.temp.name)
        cls.source=cls.root/'source';cls.s=read_json(SUP/'protocol.json')
        cls.s.update(population=20,budget=400,states={'I':6,'C':3,'M':6},milestones=[100,200,400],
                     epochs=1,checkpoint_epochs=[1],training_seeds=[11,12,13],configurations=['S0'],batch_size=4)
        info=read_json(EXPANDA/'formal_icm_s0_v1/assets/search_instances/prescreen-search-S0-001.json')
        info.pop('instance_digest');info.update(instance_id='training-fixture',development_role='training')
        info['instance_digest']=canonical_digest(info);cls.train=copy.deepcopy(info)
        write_json(cls.source/'instances/training-fixture.json',info);write_json(cls.root/'settings.json',cls.s)
        for arm in ('COLLECT_I','COLLECT_CM'):
            code="import sys;from pathlib import Path;sys.path.insert(0,sys.argv[1]);from bootstrap import read_json;from collection import work;root=Path(sys.argv[2]);out=Path(sys.argv[4]);out.mkdir(parents=True);work(root,read_json(Path(sys.argv[3])),'training-fixture',12345,sys.argv[5],out,'fixture')"
            cls.child(code,[SUP,cls.source,cls.root/'settings.json',cls.source/'collection/training-fixture'/arm,arm],cls.root/(arm+'.log'))
        info.pop('instance_digest');info.update(instance_id='validation-fixture',development_role='validation');info['instance_digest']=canonical_digest(info)
        cls.valid=copy.deepcopy(info);write_json(cls.source/'instances/validation-fixture.json',info)
        for arm in ('COLLECT_I','COLLECT_CM'):
            src=cls.source/'collection/training-fixture'/arm;dst=cls.source/'collection/validation-fixture'/arm
            d=read_json(src/'DATA.json');d.update(instance='validation-fixture',role='validation')
            for r in d['rows']:r['instance']='validation-fixture'
            write_json(dst/'DATA.json',d);r=read_json(src/'RUN_COMPLETE.json');r['feedback_sha256']=digest(dst/'DATA.json');write_json(dst/'RUN_COMPLETE.json',r)
        write_json(cls.source/'PLAN.json',{'sha256':'fixture'});write_json(cls.source/'instances/catalog.json',{'instances':[cls.train,cls.valid]})
        from data import load_dataset
        from training import fit
        folders=[cls.source/'collection'/n/arm for n in ('training-fixture','validation-fixture') for arm in ('COLLECT_I','COLLECT_CM')]
        rows,norm=load_dataset(folders,50000);cls.norm=norm
        fit(rows,norm,cls.s,cls.source/'shared_model',{}, {'full_budget_ready':True,'test_fixture':True})
        fit_independent(cls.source,cls.root/'independent_model')
        # Exercise active policies, even when this tiny synthetic fixture would
        # select disabled gates. These test-only bundles never become evidence.
        for folder in (cls.source/'shared_model',cls.root/'independent_model'):
            b=read_json(folder/'BUNDLE.json')
            for g in b['selected']['gates'].values():g.update(disabled=False,**{'lambda':0.,'threshold_scaled':0.})
            write_json(folder/'BUNDLE.json',b)
        snapshot(TerminalPolicy(cls.source/'shared_model'),cls.root/'update_000')
        info.pop('instance_digest');info.update(instance_id='screen-fixture',development_role='screen');info['instance_digest']=canonical_digest(info)
        cls.screen=copy.deepcopy(info);write_json(cls.root/'screen-fixture.json',info)
        cls.plan={'settings':copy.deepcopy(cls.s),'normalization':norm,'sha256':'fixture','source':str(cls.source),
                  'independent_model':str(cls.root/'independent_model'),'screen':[cls.screen],'entries':[cls.train,cls.valid]}
        cls.plan['settings'].update(screen_seeds=[12345],ppo=read_json(TERMINAL_DIR/'protocol.json')['ppo'])
        cls.plan['settings']['ppo']['validation_seeds']=[996001,996002]
        cls.results=[]
        for arm in ARMS:
            model=cls.root/'update_000' if arm=='ICM_TERMINAL_PPO' else model_for(cls.plan,arm,cls.root)
            t=task(cls.plan,'screen-fixture',12345,arm,'screen',model)
            out=cls.root/'screen_runs'/arm;specfile=out.with_suffix('.json')
            write_json(specfile,{'task':t,'settings':cls.s})
            code="import sys;from pathlib import Path;sys.path.insert(0,sys.argv[1]);from search_b5000 import read_json,run_search;x=read_json(Path(sys.argv[2]));out=Path(sys.argv[4]);out.mkdir(parents=True);run_search(Path(sys.argv[3]),x['settings'],x['task'],out,'fixture')"
            cls.child(code,[B5_DIR,specfile,cls.root/'screen-fixture.json',out],out.with_suffix('.log'))
            cls.results.append(read_json(out/'RUN_COMPLETE.json'))
        print('B5000 test artifacts '+str(cls.root),flush=True)

    @classmethod
    def child(cls,code,args,log):
        log.parent.mkdir(parents=True,exist_ok=True)
        with log.open('w') as f:r=subprocess.run([sys.executable,'-c',code]+list(map(str,args)),stdout=f,stderr=subprocess.STDOUT)
        if r.returncode:raise RuntimeError(log.read_text()[-8000:])

    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()

    def test_isolated_components_really_only_invoke_one_component(self):
        for r in self.results:
            active=enabled_components(r['arm'])
            for c in 'ICM':
                count=r['selection'][c]['calls']
                if c in active:self.assertGreater(count,0,(r['arm'],c))
                else:self.assertEqual(count,0,(r['arm'],c))
            self.assertIn(r['real_objective_evaluation_count'],(399,400))
            self.assertEqual(r['hidden_candidate_objective_evaluations'],0)
        self.assertEqual(len({r['scenario_digest'] for r in self.results}),1)

    def test_real_budget_fronts_recompute_and_u0_reproduction(self):
        from transformer_ppo.archive import exact_hv_3d
        from common import divisor
        n=legacy_learning.normalizer(self.norm).with_divisor(divisor(self.screen))
        for r in self.results:
            self.assertEqual([p['evaluation'] for p in r['budget_fronts']],[100,200,400])
            for p in r['budget_fronts']:
                self.assertAlmostEqual(exact_hv_3d([n.transform(x) for x in p['front_raw']],n.reference),p['hv'])
        shared=next(r for r in self.results if r['arm']=='ICM_SHARED_SUPERVISED')
        ppo=next(r for r in self.results if r['arm']=='ICM_TERMINAL_PPO')
        verify_reproduction(shared,ppo)
        bad=copy.deepcopy(ppo);bad['evaluation_order_digest']='bad'
        with self.assertRaises(ValueError):verify_reproduction(shared,bad)

    def test_multiseed_validation_pairs_by_instance_and_seed(self):
        initial=[dict(instance=i,seed=s,configuration=c,scenario_digest=i,final_hypervolume=1.)
                 for i,c in [('a','S0'),('b','S1')] for s in (1,2,3)]
        rows=[dict(r,final_hypervolume=1.+(.2 if r['configuration']=='S0' else -.1)) for r in initial]
        self.assertAlmostEqual(validation_score(rows,initial,['S0','S1']),.05)
        self.assertEqual(validation_score(rows,initial,['S0','S1']),validation_score(rows[::-1],initial[::-1],['S0','S1']))
        with self.assertRaises(ValueError):validation_score(rows[:-1],initial,['S0','S1'])
        rows[0]['scenario_digest']='wrong'
        with self.assertRaises(ValueError):validation_score(rows,initial,['S0','S1'])

    def test_cost_projection_uses_measured_arms(self):
        runner.estimate_cost(self.root,self.plan,self.results[:-1],2)
        e=read_json(self.root/'COST_ESTIMATE.json')
        self.assertEqual(e['remaining_frozen_panel_hours'],0)
        self.assertGreater(e['rough_remaining_search_hours'],0)

    def test_population250_budget5000(self):
        settings=dict(self.s,population=250,budget=5000,milestones=[400,1000,5000])
        t=task(self.plan,'screen-fixture',12345,'ICM_SHARED_SUPERVISED','screen',self.source/'shared_model')
        out=self.root/'full_parameter_smoke';specfile=out.with_suffix('.json')
        write_json(specfile,{'task':t,'settings':settings})
        code="import sys;from pathlib import Path;sys.path.insert(0,sys.argv[1]);from search_b5000 import read_json,run_search;x=read_json(Path(sys.argv[2]));out=Path(sys.argv[4]);out.mkdir(parents=True);run_search(Path(sys.argv[3]),x['settings'],x['task'],out,'fixture')"
        self.child(code,[B5_DIR,specfile,self.root/'screen-fixture.json',out],out.with_suffix('.log'))
        r=read_json(out/'RUN_COMPLETE.json')
        self.assertIn(r['real_objective_evaluation_count'],(4999,5000))
        self.assertEqual([p['evaluation'] for p in r['budget_fronts']],[400,1000,5000])
        self.assertEqual(r['hidden_candidate_objective_evaluations'],0)
        for c in 'ICM':self.assertGreater(r['selection'][c]['calls'],0)

    def test_complete_reporting_and_alias_not_fabricated_runtime(self):
        from reporting_b5000 import report
        q=report(self.results,self.plan,self.root/'report')
        self.assertTrue(q['completed']);self.assertEqual(len(q['comparisons']['400']),11)
        q=report(self.results,self.plan,self.root/'alias_report',reuse={'selected_update':0,'actual_reproduction_runs':1})
        import csv
        with (self.root/'alias_report/00_全部方法_冻结归一化.csv').open(encoding='utf-8-sig') as f:rows=list(csv.DictReader(f))
        own=[r for r in rows if r['方法']=='ICM 终局联合 PPO']
        self.assertTrue(all(not r['完整B5000搜索秒'] and '复用' in r['结果来源'] for r in own))
        bad=copy.deepcopy(self.results);bad[0]['scenario_digest']='wrong'
        with self.assertRaises(ValueError):report(bad,self.plan,self.root/'bad_report')

    def test_actual_new_terminal_workers_and_resume(self):
        root=self.root/'terminal_cli';root.mkdir();plan=copy.deepcopy(self.plan)
        plan['settings']['ppo'].update(updates=1,episodes_per_update=1,validation_checkpoints=[0,1])
        for r in plan['entries']:write_json(root/'instances'/(r['instance_id']+'.json'),r)
        plan['files']={};plan.pop('sha256');plan['sha256']=canonical_digest(plan);write_json(root/'PLAN.json',plan)
        selected=legacy.terminal(root,plan,1)
        self.assertEqual(selected['training_searches'],1);self.assertEqual(selected['validation_searches'],4)
        prior={str(p):p.stat().st_mtime_ns for p in root.rglob('RUN_COMPLETE.json')}
        with patch.object(legacy.subprocess,'Popen',side_effect=AssertionError('Do not repeat completed trajectories')):
            self.assertEqual(selected,legacy.terminal(root,plan,1))
        self.assertEqual(prior,{str(p):p.stat().st_mtime_ns for p in root.rglob('RUN_COMPLETE.json')})


class ProtocolChecks(unittest.TestCase):
    def test_prepare_frozen_split_and_resume(self):
        # Unit test of the manifest, not a substitute for domain/model loading
        # (the engine integration class above exercises those with real files).
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);original=root/'old';source=root/'source';out=root/'new'
            records=[]
            for cfg in ['S'+str(i) for i in range(9)]:
                for role,count in [('training',6),('validation',2),('screen',1)]:
                    for j in range(count):
                        name=f'{cfg}-{role}-{j}'
                        r=dict(instance_id=name,configuration=cfg,development_role=role,batch_signature=name)
                        records.append(r);write_json(original/'instances'/(name+'.json'),r)
            b=dict(normalization={},scalers={},target_scales={},checkpoint_sha256={},settings={'progress_denominator':50000})
            for role in ('training','validation'):
                b[role+'_instances']={c:[r['instance_id'] for r in records if r['development_role']==role] for c in 'ICM'}
            write_json(source/'shared_model/BUNDLE.json',b);write_json(original/'independent_model/BUNDLE.json',b)
            old=dict(files={},normalization={},source=str(source),
                settings=dict(population=250,scenario_count=100,progress_denominator=50000,
                              configurations=['S'+str(i) for i in range(9)],screen_seeds=[993001,993002,993003],ppo={}),
                entries=[r for r in records if r['development_role']!='screen'],screen=[r for r in records if r['development_role']=='screen'])
            old['sha256']=canonical_digest(old);write_json(original/'PLAN.json',old)
            def read_record(path):return read_json(path),None
            with patch.object(runner,'SharedPolicy'),patch.object(runner,'IndependentPolicy'),patch.object(runner,'read_instance',side_effect=read_record):
                p=runner.prepare(out,original)
                self.assertEqual(p,runner.prepare(out,original))
                self.assertEqual(p['settings']['budget'],5000)
                self.assertEqual(p['settings']['progress_denominator'],50000)
                self.assertEqual(p['settings']['maximum_validation_searches'],162)
                self.assertEqual(len(p['settings']['ppo']['validation_seeds']),3)
                self.assertEqual([read_json(out/'instances'/(r['instance_id']+'.json')) for r in records],records)
                self.assertEqual(read_json(original/'PLAN.json'),old)
                verify(p['files'])
                bad=copy.deepcopy(old);bad['screen'][0]['instance_id']=bad['entries'][0]['instance_id']
                bad.pop('sha256');bad['sha256']=canonical_digest(bad);write_json(original/'PLAN.json',bad)
                with self.assertRaisesRegex(ValueError,'Overlapping'):runner.prepare(root/'bad',original)


if __name__=='__main__':unittest.main()
