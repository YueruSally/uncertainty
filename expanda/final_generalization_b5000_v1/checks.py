"""Protocol, disjointness, reporting and real frozen-engine replay checks."""
import contextlib
import copy
import importlib.util
import io
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('final_checks',HERE/'run.py')
r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)


class ProtocolChecks(unittest.TestCase):
    def test_fixed_grid_and_no_training(self):
        protocol=r.read(HERE/'protocol.json')
        r.validate_protocol(protocol)
        with self.assertRaises(ValueError):r.validate_protocol(dict(protocol,instances_per_configuration=1))
        self.assertEqual(len(protocol['configurations'])*protocol['instances_per_configuration']*len(protocol['algorithm_seeds'])*len(protocol['methods']),432)
        self.assertEqual(tuple(protocol['methods']),r.ARMS)
        self.assertEqual(protocol['primary_method'],'ICM_SHARED_SUPERVISED')
        self.assertEqual({r.SCENARIOS[c][0] for c in protocol['configurations'][:9]},{10,20,30})
        self.assertEqual({r.SCENARIOS[c][0] for c in protocol['configurations'][9:]},{50})

    def test_overlap_check_uses_batch_content_not_names(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);a=dict(batches=[dict(batch_id=1,quantity=20)],instance_id='old')
            r.write(root/'a/instances/a.json',a)
            r.write(root/'b/instances/catalog.json',{'instances':[dict(a,instance_id='renamed')]})
            r.write(root/'new/instances/new.json',dict(batches=[dict(batch_id=1,quantity=30)]))
            known,files=r.known_instances(root,root/'new')
            self.assertEqual(known,{r.signature(a)})
            self.assertEqual(len(files),2)

    def test_generate_freezes_36_disjoint_manifests_without_evaluation(self):
        with tempfile.TemporaryDirectory() as d:
            diag=Path(d);root=diag/'final';root.mkdir()
            sample=r.read(r.p.EXPANDA/'formal_icm_s0_v1/assets/search_instances/prescreen-search-S0-001.json')
            r.write(diag/'old/instances/sample.json',sample)
            workbook=r.p.EXPANDA/'formal_icm_s0_v1/assets/data/data_expanded.xlsx'
            catalog=dict(train_od_pool=sorted({(b['origin'],b['destination']) for b in sample['batches']}),
                         source_data_sha256=r.digest(workbook))
            catalog['catalog_digest']=r.canonical(catalog);r.write(diag/'source_catalog.json',catalog)
            old=dict(entries=[sample],settings=dict(source_catalog_default=str(diag/'source_catalog.json'),source_catalog_digest=catalog['catalog_digest']))
            protocol=r.read(HERE/'protocol.json')
            with patch.object(r,'_find_path_seed',side_effect=lambda seed,*a:seed),patch.object(r.p.base,'_evaluate_individual_impl',side_effect=AssertionError('No search allowed during generation')),contextlib.redirect_stdout(io.StringIO()):
                records=r.generate(root,old,protocol)
            self.assertEqual(len(records),36);self.assertEqual(len({r.signature(e) for e in records}),36)
            self.assertTrue(all(e['split']=='test' and e['development_role']=='final_test' for e in records))
            self.assertNotIn(r.signature(sample),{r.signature(e) for e in records})
            self.assertEqual({e['K'] for e in records if e['configuration']=='S11'},{50})
            for e in records:self.assertEqual(r.p.read_instance(root/'instances'/(e['instance_id']+'.json'))[0],e)

    def test_new_data_sampling_preserves_distribution_and_supports_all_sizes(self):
        data=r.p.EXPANDA/'formal_icm_s0_v1/assets/data/data_expanded.xlsx'
        b=r.p.base
        with contextlib.redirect_stdout(io.StringIO()):
            b.BORDER_EVENT_DEFINITIONS=b.load_border_event_definitions(b.DEFAULT_BORDER_EVENT_DATA_FILE)
            network=b.load_network_from_extended(str(data))
        raw=network[7];od=list({(x.origin,x.destination) for x in raw});signatures=set()
        for index,(cfg,(size,alpha)) in enumerate(r.SCENARIOS.items()):
            batches=r._batches(raw,od,size,alpha,180000000+100*index,'train',24.,2.)
            self.assertEqual(len(batches),size)
            self.assertTrue(all(int(round(x.quantity*100))%2==0 for x in batches))
            sig=r.canonical([r.asdict(x) for x in batches]);self.assertNotIn(sig,signatures);signatures.add(sig)

    def test_report_keeps_seen_size_and_large_size_separate(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);rows=[];entries=[]
            plan=dict(settings=dict(budget=5000,milestones=[400,1000,5000]),protocol=dict(algorithm_seeds=[1,2]),models={},
                normalization={},sha256='synthetic',entries=entries,primary_method='ICM_SHARED_SUPERVISED',
                normalization_reporting='synthetic',purpose='synthetic')
            for cfg in ('S0','S9'):
                e=dict(instance_id=cfg+'-fixture',instance_digest='instance-'+cfg,configuration=cfg);entries.append(e)
                for seed in (1,2):
                    for mi,arm in enumerate(r.ARMS):
                        t=r.task(plan,e,seed,arm);front=[[10.-mi,12.-mi,14.-mi]]
                        folder=root/'runs'/e['instance_id']/f'seed_{seed}'/arm
                        r.write(folder/'final_feasible_nondominated.json',[])
                        v=dict(instance=e['instance_id'],instance_digest=e['instance_digest'],configuration=cfg,seed=seed,arm=arm,
                            completed=True,phase='final_test',stochastic=False,ppo_updates=0,trajectory_sha256=None,scenario_digest='scenarios-'+cfg,
                            evaluation_budget=5000,real_objective_evaluation_count=5000,hidden_candidate_objective_evaluations=0,
                            plan_sha256='synthetic',task_sha256=r.canonical(t),model_bundle_sha256=None,final_front_raw=front,
                            budget_fronts=[dict(evaluation=b,front_raw=front,hv=.4) for b in r.BUDGETS],final_hypervolume=.4,
                            optimisation_seconds=5.,peak_memory_mib=2.,hostname='fixture',final_solutions_sha256=r.digest(folder/'final_feasible_nondominated.json'))
                        rows.append(v)
            r.report(rows,root,plan,draws=20)
            import csv
            with (root/'results_union/01_终点结果.csv').open(encoding='utf-8-sig') as f:table=list(csv.DictReader(f))
            self.assertEqual(len(table),8)
            self.assertEqual({v['范围'] for v in table},{'S0-S8_新批次','S9-S11_规模泛化'})
            self.assertEqual({v['参照'] for v in table},{'Base'})
            self.assertTrue((root/'fronts/FINAL_SOLUTIONS.jsonl.gz').exists())


class EngineChecks(unittest.TestCase):
    def test_four_frozen_methods_match_existing_engine_and_resume(self):
        # Build a small synthetic model once. These fixture runs are not paper evidence.
        spec=importlib.util.spec_from_file_location('final_fixture',r.s.audit.PILOT/'checks.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);fixture=module.EngineChecks
        fixture.setUpClass()
        try:
            root=fixture.root/'final';root.mkdir()
            entry=copy.deepcopy(fixture.valid);entry.pop('instance_digest');entry.update(split='test',development_role='final_test')
            entry['instance_digest']=r.canonical(entry);r.write(root/'instances'/(entry['instance_id']+'.json'),entry)
            plan=dict(settings=fixture.settings,protocol=dict(algorithm_seeds=[12345]),entries=[entry],normalization=fixture.norm,
                models=dict(ICM_SHARED_SUPERVISED=str(fixture.source/'shared_model'),ICM_TERMINAL_PPO=str(fixture.model)),files={})
            plan['sha256']=r.canonical(plan);r.write(root/'PLAN.json',plan)
            references={'ICM_SHARED_SUPERVISED':fixture.original,'ICM_TERMINAL_PPO':fixture.replay}
            for arm in ('BASE','C_RULE'):
                task=dict(instance=fixture.valid['instance_id'],seed=12345,arm=arm,phase='validation',sample=False,
                          model=None,bundle_sha256=None,normalization=fixture.norm)
                taskfile=fixture.root/(arm+'.task.json');r.write(taskfile,task)
                folder=fixture.root/arm;folder.mkdir()
                code="import sys;from pathlib import Path;sys.path.insert(0,sys.argv[1]);from search_b5000 import run_search,read_json;r=Path(sys.argv[2]);run_search(r/'instances/validation-fixture.json',read_json(r/'settings.json'),read_json(Path(sys.argv[3])),Path(sys.argv[4]),'reference')"
                with folder.with_suffix('.log').open('w') as log:
                    q=subprocess.run([sys.executable,'-c',code,str(r.p.B5),str(fixture.root),str(taskfile),str(folder)],stdout=log,stderr=subprocess.STDOUT)
                self.assertEqual(q.returncode,0,folder.with_suffix('.log').read_text()[-4000:]);references[arm]=r.read(folder/'RUN_COMPLETE.json')
            jobs=[]
            for arm in r.ARMS:
                t=r.task(plan,entry,12345,arm);out=root/'runs'/entry['instance_id']/'seed_12345'/arm;jobs.append((t,out))
                r.validate_task(root,plan,t,out)
                with self.assertRaises(ValueError):r.validate_task(root,plan,dict(t,sample=True),out)
                with self.assertRaises(ValueError):r.validate_task(root,plan,dict(t,seed=999),out)
            r.scheduler.HERE2=HERE
            with contextlib.redirect_stdout(io.StringIO()):actual=r.scheduler.execute_jobs(root,plan,jobs,1,'unit/final_test')
            for row in actual:
                ref=references[row['arm']]
                for key in ('final_front_raw','final_hypervolume','scenario_digest','real_objective_evaluation_count','evaluation_order_digest','selection'):
                    self.assertEqual(row[key],ref[key],(row['arm'],key))
                self.assertEqual([(x['evaluation'],x['hv'],x['front_raw']) for x in row['budget_fronts']],
                    [(x['evaluation'],x['hv'],x['front_raw']) for x in ref['budget_fronts']])
                self.assertEqual(row['torch_num_threads'],1);self.assertEqual(row['phase'],'final_test');r.validate_run(row,plan)
            r.validate_panel(actual,plan)
            with self.assertRaises(ValueError):r.validate_panel(actual[:-1],plan)
            bad=copy.deepcopy(actual);bad[0]['scenario_digest']='changed'
            with self.assertRaises(ValueError):r.validate_panel(bad,plan)
            before={(out/'RUN_COMPLETE.json'): (out/'RUN_COMPLETE.json').stat().st_mtime_ns for _,out in jobs}
            with contextlib.redirect_stdout(io.StringIO()):cached=r.scheduler.execute_jobs(root,plan,jobs,1,'unit/resume')
            self.assertEqual(cached,actual)
            self.assertTrue(all(path.stat().st_mtime_ns==stamp for path,stamp in before.items()))
        finally:fixture.tearDownClass()


if __name__=='__main__':unittest.main()
