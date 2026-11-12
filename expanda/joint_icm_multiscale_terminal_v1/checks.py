"""Real small-budget integration checks; no research conclusions."""
from bridge import *
from policy import TerminalPolicy,snapshot,optimizer_for,update
from prepare import training_schedule
from reporting import union_hv,report
import copy
import importlib.util
import subprocess
import tempfile
import unittest
from unittest.mock import patch


def child(code,args,log):
    with log.open('w') as f:
        result=subprocess.run([sys.executable,'-c',code]+list(map(str,args)),stdout=f,stderr=subprocess.STDOUT)
    if result.returncode:raise RuntimeError(log.read_text()[-6000:])


class Checks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1);cls.tmp=tempfile.TemporaryDirectory(prefix='multi-terminal-check-');cls.root=Path(cls.tmp.name)
        cls.source=cls.root/'source';cls.source.mkdir();cls.s=read_json(SUP/'protocol.json')
        cls.s.update(population=20,budget=400,states={'I':6,'C':3,'M':6},milestones=[100,400],
                     epochs=1,checkpoint_epochs=[1],training_seeds=[11,12,13],configurations=['S0'],batch_size=4)
        info=read_json(EXPANDA/'formal_icm_s0_v1/assets/search_instances/prescreen-search-S0-001.json');info.pop('instance_digest')
        info['development_role']='training';info['instance_id']='training-fixture';info['instance_digest']=canonical_digest(info)
        cls.train=copy.deepcopy(info);write_json(cls.source/'instances/training-fixture.json',info)
        settings=cls.root/'settings.json';write_json(settings,cls.s)
        code="import sys;from pathlib import Path;sys.path.insert(0,sys.argv[1]);from bootstrap import read_json;from collection import work;root=Path(sys.argv[2]);out=Path(sys.argv[4]);out.mkdir(parents=True);work(root,read_json(Path(sys.argv[3])),'training-fixture',12345,sys.argv[5],out,'fixture')"
        for arm in ('COLLECT_I','COLLECT_CM'):
            out=cls.source/'collection/training-fixture'/arm
            child(code,[SUP,cls.source,settings,out,arm],cls.root/(arm+'.log'))
        info.pop('instance_digest');info['development_role']='validation';info['instance_id']='validation-fixture';info['instance_digest']=canonical_digest(info)
        cls.valid=copy.deepcopy(info);write_json(cls.source/'instances/validation-fixture.json',info)
        for arm in ('COLLECT_I','COLLECT_CM'):
            src=cls.source/'collection/training-fixture'/arm;dst=cls.source/'collection/validation-fixture'/arm
            d=read_json(src/'DATA.json');d.update(instance='validation-fixture',role='validation')
            for r in d['rows']:r['instance']='validation-fixture'
            write_json(dst/'DATA.json',d);r=read_json(src/'RUN_COMPLETE.json');r['feedback_sha256']=digest(dst/'DATA.json');write_json(dst/'RUN_COMPLETE.json',r)
        write_json(cls.source/'PLAN.json',{'sha256':'fixture'})
        write_json(cls.source/'instances/catalog.json',{'instances':[cls.train,cls.valid]})
        folders=[cls.source/'collection'/n/arm for n in ('training-fixture','validation-fixture') for arm in ('COLLECT_I','COLLECT_CM')]
        from data import load_dataset
        from training import fit
        rows,norm=load_dataset(folders,50000);cls.norm=norm;cls.rows=rows
        fit(rows,norm,cls.s,cls.source/'shared_model',{}, {'full_budget_ready':True,'test_fixture':True})
        fit_independent(cls.source,cls.root/'independent_model')
        p=TerminalPolicy(cls.source/'shared_model');snapshot(p,cls.root/'update_000')
        for gate in p.bundle['selected']['gates'].values():gate['disabled']=False
        snapshot(p,cls.root/'exploration_fixture')
        info.pop('instance_digest');info['instance_id']='screen-fixture';info['development_role']='screen';info['instance_digest']=canonical_digest(info)
        cls.screen=copy.deepcopy(info);write_json(cls.root/'screen-fixture.json',info)
        cls.spec={'settings':{**cls.s,**read_json(HERE2/'protocol.json')},'normalization':norm,'sha256':'fixture',
                  'source':str(cls.source),'screen':[cls.screen],'entries':[cls.train,cls.valid]}
        cls.spec['settings'].update(screen_seeds=[12345],configurations=['S0'])
        cls.results=[];cls.jobs=[]
        for arm in METHODS:
            model={'BASE':None,'ICM_HEURISTIC':None,'ICM_INDEPENDENT':cls.root/'independent_model',
                   'ICM_SHARED_SUPERVISED':cls.source/'shared_model','ICM_TERMINAL_PPO':cls.root/'update_000'}[arm]
            t={'instance':'screen-fixture','seed':12345,'arm':arm,'phase':'screen','sample':False,
               'model':str(model) if model else None,'bundle_sha256':digest(model/'BUNDLE.json') if model else None,'normalization':norm}
            cls.jobs.append((t,cls.root/arm));cls.results.append(cls.search(t,cls.root/'screen-fixture.json',cls.root/arm))
        cls.episodes=[]
        for i in range(2):
            t=dict(cls.jobs[-1][0],instance='training-fixture',phase='training',sample=True,seed=12345+i,
                   model=str(cls.root/'exploration_fixture'),bundle_sha256=digest(cls.root/'exploration_fixture/BUNDLE.json'))
            out=cls.root/f'episode{i}';r=cls.search(t,cls.source/'instances/training-fixture.json',out)
            cls.episodes.append((out/'TRAJECTORY.npz',r['final_hypervolume']))
        print('Integration artifacts: '+str(cls.root),flush=True)

    @classmethod
    def search(cls,t,instance,out):
        taskfile=out.with_suffix('.json');write_json(taskfile,{'task':t,'settings':cls.s})
        code="import sys;from pathlib import Path;sys.path.insert(0,sys.argv[1]);from bridge import read_json;from search import run_search;x=read_json(Path(sys.argv[2]));out=Path(sys.argv[4]);out.mkdir();run_search(Path(sys.argv[3]),x['settings'],x['task'],out,'fixture')"
        child(code,[HERE2,taskfile,instance,out],out.with_suffix('.log'))
        return read_json(out/'RUN_COMPLETE.json')

    @classmethod
    def tearDownClass(cls):cls.tmp.cleanup()

    def test_shared_zero_actor_exact_reproduction(self):
        shared=SharedPolicy(self.source/'shared_model');ppo=TerminalPolicy(self.root/'update_000')
        for c in COMPONENTS:
            for row in self.rows[c]['validation']:self.assertEqual(shared.select(row),ppo.select(row))
        a,b=self.results[-2:]
        self.assertEqual(a['final_front_raw'],b['final_front_raw'])
        self.assertEqual(a['evaluation_order_digest'],b['evaluation_order_digest'])

    def test_joint_real_budget_and_no_hidden_evaluations(self):
        for r in self.results:
            self.assertIn(r['real_objective_evaluation_count'],(399,400));self.assertEqual(r['hidden_candidate_objective_evaluations'],0)
            if r['arm']!='BASE':
                for c in COMPONENTS:self.assertGreater(r['selection'][c]['calls'],0,c)
                self.assertGreater(r['fixed_initial_slots_preserved'],0)
                self.assertGreater(r['deferred_c_pairs'],0)
        self.assertEqual(len({r['scenario_digest'] for r in self.results}),1)

    def test_independent_has_disjoint_parameters_same_scaling(self):
        p=IndependentPolicy(self.root/'independent_model');b=read_json(self.source/'shared_model/BUNDLE.json')
        for k in ('normalization','scalers','target_scales','training_instances','validation_instances'):self.assertEqual(p.bundle[k],b[k])
        m=p.models[0]
        ids=[{id(p) for p in m.networks[c].parameters()} for c in COMPONENTS]
        self.assertFalse(ids[0]&ids[1] or ids[1]&ids[2] or ids[0]&ids[2])

    def test_real_ppo_update_and_resume_snapshot(self):
        p=TerminalPolicy(self.root/'exploration_fixture');s=self.spec['settings']['ppo'];opt=optimizer_for(p,s)
        before={k:v.clone() for k,v in p.models[0].state_dict().items()}
        metric=update(p,self.episodes,s,opt,123)
        self.assertLess(metric['on_policy_logp_max_error'],3e-5);self.assertGreater(metric['optimizer_steps'],0)
        self.assertTrue(any(not torch.equal(v,p.models[0].state_dict()[k]) for k,v in before.items()))
        self.assertFalse(torch.equal(before['gain_head.weight'],p.models[0].state_dict()['gain_head.weight']))
        dst=self.root/'update_fixture';snapshot(p,dst,opt,metric)
        restored=TerminalPolicy(dst);restored_opt=optimizer_for(restored,s);restored_opt.load_state_dict(load_checkpoint(dst/'optimizer.pt'))
        for a,b in zip(p.models,restored.models):
            for k,v in a.state_dict().items():self.assertTrue(torch.equal(v,b.state_dict()[k]))
        self.assertEqual(set(opt.state_dict()['state']),set(restored_opt.state_dict()['state']))

    def test_protocol_schedule_and_zero_checkpoint_selection(self):
        spec=copy.deepcopy(self.spec);spec['settings']['configurations']=[f'S{i}' for i in range(9)];spec['entries']=[]
        for cfg in spec['settings']['configurations']:
            spec['entries'] += [{'configuration':cfg,'instance_id':cfg+f'-{i}','development_role':'training'} for i in range(6)]
        schedule=training_schedule(spec);self.assertEqual(len(schedule),32)
        self.assertEqual({r['configuration'] for r in schedule},set(spec['settings']['configurations']))
        self.assertTrue(all(sum(r['configuration']==c for r in schedule) in (3,4) for c in spec['settings']['configurations']))
        values=[{'update':0,'mean_difference_vs_shared':0},{'update':4,'mean_difference_vs_shared':-.1},{'update':8,'mean_difference_vs_shared':-.2}]
        self.assertEqual(max(values,key=lambda r:(r['mean_difference_vs_shared'],-r['update']))['update'],0)

    def test_new_development_instances_frozen_without_evaluation(self):
        from prepare import generate_screen
        import baseline_uncertainty as base
        root=self.root/'generation';source=root/'source';old=root/'old'
        write_json(source/'PLAN.json',{'source':str(old)})
        write_json(old/'instances/catalog.json',{'instances':[self.train,self.valid]})
        catalog={'train_od_pool':sorted({(b['origin'],b['destination']) for b in self.train['batches']}),
                 'test_od_pool':[],'source_data_sha256':digest(EXPANDA/'formal_icm_s0_v1/assets/data/data_expanded.xlsx')}
        catalog['catalog_digest']=canonical_digest(catalog);write_json(root/'original_catalog.json',catalog)
        s=copy.deepcopy(self.spec['settings']);s.update(configurations=[f'S{i}' for i in range(9)],
                source_catalog_default=str(root/'original_catalog.json'),source_catalog_digest=catalog['catalog_digest'])
        with patch.object(base,'evaluate_individual',side_effect=AssertionError('Instance generation must not evaluate objectives')):
            entries=generate_screen(root,source,s,[self.train,self.valid])
            self.assertEqual(entries,generate_screen(root,source,s,[self.train,self.valid]))
        self.assertEqual(len(entries),9);self.assertEqual(len({r['batch_signature'] for r in entries}),9)
        self.assertTrue(all(r['development_role']=='screen' and r['oos5000_seed'] is None for r in entries))

    def test_five_arm_report_union_and_pairs(self):
        q=report(self.results,self.spec,self.root/'report',{'update':0})
        self.assertEqual(q['screen_searches'],5)
        self.assertAlmostEqual(q['methods']['ICM_TERMINAL_PPO']['comparisons']['ICM_SHARED_SUPERVISED']['absolute']['mean'],0)
        bad=copy.deepcopy(self.results);bad[0]['scenario_digest']='wrong'
        with self.assertRaises(ValueError):report(bad,self.spec,self.root/'bad-report',{'update':0})
        rows=[dict(instance='a',final_front_raw=[],final_hypervolume=0),dict(instance='a',final_front_raw=[[1,2,3]],final_hypervolume=1)]
        normalized,bounds=union_hv(rows);self.assertEqual(normalized[0]['final_hypervolume'],0)
        self.assertEqual(bounds['a']['zero_range'],[True]*3)

    def test_terminal_cli_training_validation_and_reuse(self):
        # Execute actual worker subprocesses through the new orchestration path.
        spec=importlib.util.spec_from_file_location('terminal_runner_checked',HERE2/'run.py')
        runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
        root=self.root/'cli';root.mkdir();plan=copy.deepcopy(self.spec)
        plan['settings']['ppo'].update(updates=1,episodes_per_update=1,validation_checkpoints=[0,1])
        for r in plan['entries']:write_json(root/'instances'/(r['instance_id']+'.json'),r)
        plan['files']={};plan.pop('sha256');plan['sha256']=canonical_digest(plan);write_json(root/'PLAN.json',plan)
        selected=runner.terminal(root,plan,1)
        self.assertEqual(selected['training_searches'],1);self.assertEqual(selected['validation_searches'],2)
        # Disabled fixture gates force ties; checkpoint zero must be selectable.
        self.assertEqual(selected['update'],0)
        mtimes={str(p):p.stat().st_mtime_ns for p in root.rglob('RUN_COMPLETE.json')}
        with patch.object(runner.subprocess,'Popen',side_effect=AssertionError('Completed work must be reused')):
            self.assertEqual(selected,runner.terminal(root,plan,1))
        self.assertEqual(mtimes,{str(p):p.stat().st_mtime_ns for p in root.rglob('RUN_COMPLETE.json')})
        t=runner.task(plan,self.valid['instance_id'],plan['settings']['ppo']['validation_seed'],
                      'ICM_TERMINAL_PPO','validation',root/'terminal/checkpoints/update_000')
        runner.validate_task(root,plan,t,root/'terminal/validation/u000'/self.valid['instance_id'])
        with self.assertRaises(ValueError):
            runner.validate_task(root,plan,dict(t,phase='training',sample=True),root/'wrong')
        with self.assertRaises(ValueError):
            runner.validate_task(root,plan,dict(t,seed=t['seed']+1),root/'wrong')

    def test_c_preview_restores_rng_and_never_evaluates(self):
        from search import JointController
        from multi_instance_logging import fingerprint
        from types import SimpleNamespace
        from unittest.mock import patch
        import random
        import baseline_uncertainty as base
        ctl=JointController.__new__(JointController);ctl._feature_seconds=0.;ctl.deferred_pairs=0
        pairs=[(base.Individual(),base.Individual()) for _ in range(8)];c1,c2=pairs[0]
        ctl.pending_c={'pairs':pairs,'roles':['x']*8}
        draws=[]
        def preview(*args,**kwargs):draws.append(random.random())
        ctl.ch=SimpleNamespace(_paired_base_mutation=preview)
        ctl._state=lambda *args,**kwargs:(None,{'legal_mask':[True]*8})
        ctl._select=lambda *args:0
        state=random.getstate();npstate=np.random.get_state()
        with patch('search._pair_proxy_rows',return_value=[{'score':i} for i in range(8)]),patch('search._pair_candidate_features',return_value=[]),patch('search._pair_batch_tokens',return_value=[]),patch.object(base,'repair_missing_allocations'):
            ctl.process_crossover_pair(lambda *a: self.fail('Hidden evaluation'),c1,c2,[],{}, {},{},[],0,0)
        self.assertEqual(state,random.getstate());self.assertTrue(np.array_equal(npstate[1],np.random.get_state()[1]));self.assertEqual(len(draws),2)


if __name__=='__main__':unittest.main()
