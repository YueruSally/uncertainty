"""Bounded local correctness checks, separate from research instances/results."""
import copy
import contextlib
from dataclasses import asdict
from pathlib import Path
import tempfile
import unittest

import run
from run import read_json, write_json, canonical_digest


class ReplicationChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory(prefix='b400-recheck-tests-')
        cls.root=Path(cls.temp.name)
        cls.config=copy.deepcopy(run.CONFIG)
        cls.config.update(instance_count=1,algorithm_seeds=[961001])
        # Fixture reuses a known instance, never claims to be a new research instance.
        sample=read_json(run.PACKAGE/'assets/search_instances/prescreen-search-S0-001.json')
        sample.pop('instance_digest');sample['development_role']='b400_replication';sample['instance_digest']=canonical_digest(sample)
        cls.name=sample['instance_id']
        write_json(cls.root/'instances'/f'{cls.name}.json',sample)
        cat={'instances':[{'instance_id':cls.name,'instance_digest':sample['instance_digest']}],
             'config_digest':canonical_digest(cls.config)}
        cat['catalog_digest']=canonical_digest(cat);write_json(cls.root/'instances/catalog.json',cat)
        run.prepare_models(cls.root)
        cls.plan=run.make_plan(cls.root,cat,cls.config)
        with (cls.root/'test.log').open('w') as log,contextlib.redirect_stdout(log):
            cls.result=run.execute_jobs(cls.root,cls.plan,workers=1)
        cls.rows=read_json(cls.root/'ALL_RUNS.json')
        print('Seven-method B400 fixture:',cls.root,flush=True)

    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()

    def test_seven_methods_budget_and_pairing(self):
        self.assertEqual(len(self.rows),7)
        self.assertEqual(len({r['scenario_digest'] for r in self.rows}),1)
        for r in self.rows:
            self.assertEqual(r['real_objective_evaluation_count'],400)
            self.assertEqual(r['new_training_steps'],0)
            self.assertEqual(r['new_ppo_updates'],0)
        self.assertEqual(self.result['search_objective_evaluations'],2800)
        for m in run.RULES:
            self.assertIn('vs_rule',self.result['methods'][m])

    def test_unchanged_models_scalers_and_gate(self):
        for c in 'ICM':
            original=read_json(run.PACKAGE/'assets'/c/'BUNDLE.json')
            relocated=read_json(self.root/'models'/f'{c}_BUNDLE.json')
            a=original['models']['200'] if c=='M' else original
            b=relocated['models']['200'] if c=='M' else relocated
            self.assertEqual(list(a.pop('checkpoint_sha256').values()),list(b.pop('checkpoint_sha256').values()))
            a.pop('checkpoints');b.pop('checkpoints')
            self.assertEqual(original,relocated)

    def test_resume_without_new_runs(self):
        stamps={str(p):p.stat().st_mtime_ns for p in (self.root/'runs').rglob('RUN_COMPLETE.json')}
        with (self.root/'resume.log').open('w') as log,contextlib.redirect_stdout(log):
            result=run.execute_jobs(self.root,self.plan,workers=1)
        self.assertEqual(result['methods'],self.result['methods'])
        self.assertEqual(stamps,{str(p):p.stat().st_mtime_ns for p in (self.root/'runs').rglob('RUN_COMPLETE.json')})

    def test_zero_reference_and_pairing_rejection(self):
        folder=self.root/'synthetic';folder.mkdir()
        rows=copy.deepcopy(self.rows)
        for r in rows:
            if r['strategy']=='BASE':r['final_hypervolume']=0
        result=run.report(folder,self.plan,rows)
        self.assertIsNone(result['methods'][run.engine.I_TRANSFORMER]['vs_base']['percent']['mean'])
        self.assertIsNotNone(result['methods'][run.engine.I_TRANSFORMER]['vs_base']['absolute']['mean'])
        rows[0]['scenario_digest']='wrong'
        with self.assertRaises(ValueError):run.report(folder,self.plan,rows)

    def test_frozen_plan_and_duplicate_grid(self):
        cat=read_json(self.root/'instances/catalog.json')
        self.assertEqual(run.make_plan(self.root,cat,self.config),self.plan)
        changed=copy.deepcopy(self.config);changed['algorithm_seeds']=[961002]
        with self.assertRaises(ValueError):run.make_plan(self.root,cat,changed)
        with self.assertRaises(ValueError):run.report(self.root,self.plan,self.rows+self.rows[:1])

    def test_generator_and_catalog_integrity(self):
        fixture=self.root/'generator';fixture.mkdir()
        sample=read_json(run.PACKAGE/'assets/search_instances/prescreen-search-S0-001.json')
        ods=sorted({(b['origin'],b['destination']) for b in sample['batches']})
        source={'train_od_pool':ods,'test_od_pool':[['reserved','test']],
                'source_data_sha256':run.digest(run.PACKAGE/'assets/data/data_expanded.xlsx'),'instances':[]}
        source['catalog_digest']=canonical_digest(source)
        path=fixture/'source.json';write_json(path,source)
        cfg=dict(self.config,source_catalog_digest=source['catalog_digest'])
        with (fixture/'prepare.log').open('w') as log,contextlib.redirect_stdout(log):
            cat=run.prepare_instances(fixture,path,fixture/'no-prior',cfg)
        self.assertEqual(len(cat['instances']),1)
        self.assertFalse(cat['final_test_instances_used'])
        self.assertEqual(cat,run.prepare_instances(fixture,path,fixture/'no-prior',cfg))
        name=cat['instances'][0]['instance_id'];record=read_json(fixture/'instances'/f'{name}.json')
        self.assertEqual(record['K'],20);self.assertIsNone(record['oos5000_seed'])
        record['ccp100_seed']+=1;write_json(fixture/'instances'/f'{name}.json',record)
        with self.assertRaises(ValueError):run.verify_catalog(fixture,cfg)


if __name__=='__main__':unittest.main()
