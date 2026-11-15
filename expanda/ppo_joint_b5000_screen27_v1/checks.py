"""Synthetic protocol checks and exact engine/CLI replay; not experiment results."""
import contextlib
import copy
import importlib.util
import io
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('screen27_checks', HERE/'run.py')
r = importlib.util.module_from_spec(spec); spec.loader.exec_module(r)


def row(e, seed, arm, plan_sha, model_hash=None, value=.5):
    return dict(instance=e['instance_id'], configuration=e['configuration'], instance_digest=e['instance_digest'],
        seed=seed, arm=arm, phase='screen', completed=True, stochastic=False, ppo_updates=0,
        plan_sha256=plan_sha, model_bundle_sha256=model_hash, scenario_digest='scenario-'+e['configuration'],
        evaluation_budget=5000, real_objective_evaluation_count=5000, hidden_candidate_objective_evaluations=0,
        final_hypervolume=value, final_front_raw=[[1.,2.,3.]], optimisation_seconds=1.,
        budget_fronts=[dict(evaluation=b, hv=value, front_raw=[[1.,2.,3.]]) for b in r.BUDGETS])


class ProtocolChecks(unittest.TestCase):
    def test_exact_source_panel_and_fixed_model_hashes(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); original=root/'old'; source=root/'pilot'; out=root/'new'; supervision=root/'supervision'
            norm=dict(minimum=[0.]*3, maximum=[5.]*3, reference=[1.1]*3)
            r.write(supervision/'shared_model/BUNDLE.json', {'synthetic':'shared'})
            shared_hash=r.digest(supervision/'shared_model/BUNDLE.json')
            entries=[]
            for c in range(9):
                e=r.read(HERE.parent/'formal_icm_s0_v1/assets/search_instances/prescreen-search-S0-001.json')
                e.pop('instance_digest'); e.update(configuration='S'+str(c), instance_id=f'fixture-S{c}', development_role='screen')
                e['instance_digest']=r.canonical(e); entries.append(e); r.write(original/'instances'/(e['instance_id']+'.json'),e)
            settings=dict(population=250,budget=5000,scenario_count=100,progress_denominator=50000,
                          milestones=[400,1000,5000],screen_seeds=[993001,993002,993003])
            old=dict(settings=settings,screen=entries,source=str(supervision),normalization=norm,files={})
            old['sha256']=r.canonical(old); r.write(original/'PLAN.json',old); r.write(original/'STATUS.json',{'completed':True})
            for e in entries:
                for s in settings['screen_seeds']:
                    for arm in r.POOL_ARMS:
                        path=original/'screen'/e['instance_id']/f'seed_{s}'/arm/'RUN_COMPLETE.json'
                        r.write(path,row(e,s,arm,old['sha256'],shared_hash if arm=='ICM_SHARED_SUPERVISED' else None))
            plan=dict(source=str(original),source_plan_sha256=old['sha256'],normalization=norm,files={})
            plan['sha256']=r.canonical(plan); r.write(source/'PLAN.json',plan)
            r.write(source/'STATUS.json',dict(completed=True,selected_updated_checkpoint=1))
            r.write(source/'SELECTION.json',dict(best_updated_checkpoint=1,best_including_u0=0))
            model=source/'checkpoints/update_001'; model.mkdir(parents=True)
            (model/'fixture.pt').write_bytes(b'synthetic fixture; not model weights')
            bundle=dict(normalization=norm,settings=dict(progress_denominator=50000),checkpoint_sha256={'fixture.pt':r.digest(model/'fixture.pt')})
            r.write(model/'BUNDLE.json',bundle)
            with patch.object(r.p,'PilotPolicy',return_value=SimpleNamespace(bundle=bundle)):
                frozen=r.prepare(out,source); self.assertEqual(r.prepare(out,source),frozen)
                self.assertEqual(len(frozen['union_pool']),270)
                self.assertEqual([len(v) for v in frozen['references'].values()],[27,27])
                self.assertEqual(frozen['maximum_new_searches'],27)
                self.assertEqual(frozen['original_selection']['best_including_u0'],0)
                r.p.verify(frozen['files'])
                e=entries[0]; t=r.task(frozen,e,993001); folder=out/'runs'/e['instance_id']/'seed_993001/PPO_U001'
                r.validate_task(out,frozen,t,folder)
                for change in (dict(sample=True),dict(update=0),dict(seed=996001),dict(record=True)):
                    with self.assertRaises(ValueError):r.validate_task(out,frozen,dict(t,**change),folder)
                changed=Path(frozen['references']['SHARED_GREEDY'][0]); oldrow=r.read(changed)
                r.write(changed,dict(oldrow,scenario_digest='other'))
                with self.assertRaises(ValueError):r.prepare(out,source)

    def test_two_references_and_preview_do_not_claim_full_panel(self):
        e=dict(instance_id='fixture',configuration='S0',instance_digest='instance')
        groups={'BASE':[row(e,1,'BASE','x',value=.2),row(e,2,'BASE','x',value=.5)],
                'SHARED_GREEDY':[row(e,1,'shared','x',value=.3),row(e,2,'shared','x',value=.6)],
                'PPO_GREEDY':[row(e,1,'ppo','x',value=.4),row(e,2,'ppo','x',value=.6)]}
        plan=dict(model_bundle_sha256='fixed',qualification='synthetic',union_definition='synthetic union')
        with tempfile.TemporaryDirectory() as d,contextlib.redirect_stdout(io.StringIO()):
            out=Path(d);r.report(groups,out,plan,partial=True)
            q=r.read(out/'QUICK_RESULT.json')
            self.assertTrue(q['preview_only']);self.assertFalse(q['completed'])
            self.assertAlmostEqual(q['comparisons']['PPO_GREEDY/BASE/5000']['mean_paired_percent'],60.)
            self.assertAlmostEqual(q['comparisons']['PPO_GREEDY/SHARED_GREEDY/5000']['mean_paired_percent'],100/6)
            self.assertIsNone(q['comparisons']['PPO_GREEDY/BASE/5000']['absolute_ci_high'])
            r.report(groups,out,plan)
            q=r.read(out/'QUICK_RESULT.json');self.assertTrue(q['completed'])
            self.assertIsNotNone(q['comparisons']['PPO_GREEDY/SHARED_GREEDY/5000']['percent_ci_low'])

    def test_union_keeps_other_original_fronts_as_bounds_inputs(self):
        e=dict(instance_id='fixture',configuration='S0',instance_digest='instance')
        a=row(e,1,'BASE','x');other=row(e,1,'C_RULE','x');ppo=row(e,1,'ppo','x')
        other['budget_fronts'][0]['front_raw']=[[20.,20.,20.]]
        ppo['budget_fronts'][-1]['front_raw']=[[0.,0.,0.]]
        groups={'ORIGINAL_TEN_METHODS':[a,other],'PPO_GREEDY':[ppo]};before=copy.deepcopy(groups)
        normalized,bounds=r.audit.union_normalize(groups)
        self.assertEqual(bounds['fixture']['minimum'],[0.]*3);self.assertEqual(bounds['fixture']['maximum'],[20.]*3)
        self.assertAlmostEqual(normalized['PPO_GREEDY'][0]['final_hypervolume'],1.331)
        self.assertEqual(groups,before)


class EngineChecks(unittest.TestCase):
    def test_screen_cli_matches_exact_pilot_policy_and_resumes(self):
        # Existing fixture creates small real candidate data/models, all test-only.
        spec=importlib.util.spec_from_file_location('pilot_fixture',r.audit.PILOT/'checks.py')
        fixtures=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixtures)
        fixture=fixtures.EngineChecks
        fixture.setUpClass()
        try:
            root=fixture.root/'screen27';root.mkdir()
            entry=copy.deepcopy(fixture.valid);entry.pop('instance_digest');entry['development_role']='screen'
            entry['instance_digest']=r.canonical(entry);r.write(root/'instances'/(entry['instance_id']+'.json'),entry)
            settings=copy.deepcopy(fixture.settings);settings['screen_seeds']=[12345]
            model=fixture.model
            plan=dict(entries=[entry],model=str(model),model_bundle_sha256=r.digest(model/'BUNDLE.json'),
                settings=settings,normalization=fixture.norm,files={},scenarios={})
            t=r.task(plan,entry,12345)
            # Reference: original v2 worker, with only its task allowlist adapted to this test screen entry.
            plan['sha256']='reference';r.write(root/'reference_plan.json',plan);r.write(root/'task.json',t)
            reference=root/'reference';reference.mkdir()
            code="import importlib.util,sys;from pathlib import Path;sys.path.insert(0,sys.argv[1]);s=importlib.util.spec_from_file_location('reference_pilot',Path(sys.argv[1])/'run.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m);m.validate_task=lambda *a:None;r=Path(sys.argv[2]);m.run_worker(r,m.read_json(r/'reference_plan.json'),m.read_json(r/'task.json'),r/'reference')"
            with (root/'reference.log').open('w') as log:
                done=subprocess.run([sys.executable,'-c',code,str(r.audit.PILOT),str(root)],stdout=log,stderr=subprocess.STDOUT)
            self.assertEqual(done.returncode,0,(root/'reference.log').read_text()[-5000:])
            before=r.read(reference/'RUN_COMPLETE.json');plan.pop('sha256')
            plan['scenarios'][f'{entry["instance_id"]}/12345']=before['scenario_digest']
            plan['sha256']=r.canonical(plan);r.write(root/'PLAN.json',plan)
            folder=root/'runs'/entry['instance_id']/'seed_12345/PPO_U001'
            r.scheduler.HERE2=HERE
            with contextlib.redirect_stdout(io.StringIO()):after=r.scheduler.execute_jobs(root,plan,[(t,folder)],1,'test')[0]
            for key in ('final_front_raw','final_hypervolume','scenario_digest','real_objective_evaluation_count','evaluation_order_digest','selection'):
                self.assertEqual(before[key],after[key],key)
            self.assertEqual([(s['evaluation'],s['hv'],s['front_raw']) for s in before['budget_fronts']],
                             [(s['evaluation'],s['hv'],s['front_raw']) for s in after['budget_fronts']])
            self.assertFalse(after['stochastic']);self.assertIsNone(after['trajectory_sha256'])
            self.assertEqual(after['ppo_updates'],0)
            modified=(folder/'RUN_COMPLETE.json').stat().st_mtime_ns
            with contextlib.redirect_stdout(io.StringIO()):cached=r.scheduler.execute_jobs(root,plan,[(t,folder)],1,'resume')[0]
            self.assertEqual(after,cached);self.assertEqual(modified,(folder/'RUN_COMPLETE.json').stat().st_mtime_ns)
            r.validate_ppo(cached,plan)
        finally:fixture.tearDownClass()


if __name__=='__main__':unittest.main()
