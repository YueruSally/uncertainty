"""Release verification fixtures only; these files are never shipped in the package."""
import contextlib
import copy
import gzip
import importlib.util
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import unittest
import zipfile

import package as p
sys.path.insert(0,str(p.EXPANDA/'component_recheck_b400_icm_v1'))
spec=importlib.util.spec_from_file_location('combined_checks',p.EXPANDA/'component_recheck_b400_icm_v1/checks.py')
fixtures=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixtures)
e=fixtures.e


class ReleaseChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixtures.CombinedChecks.setUpClass();f=fixtures.CombinedChecks;cls.fixture=f
        cls.root=f.root/'portable';cls.root.mkdir()
        design=dict(instances=[f.name],algorithm_seeds=[f.seed],methods=list(e.ALL_METHODS),
                    environment=f.old['environment'],settings=f.old['config'],protocol=e.PROTOCOL)
        p.payload(cls.root,f.out,design)
        cls.result=cls.root/'results';cls.result.mkdir();cls.records=[]
        for m in e.ALL_METHODS:
            base=f.source if m in e.legacy.METHODS else f.out
            folder=base/'runs'/f.name/f'seed_{f.seed}'/m
            cls.records.append({k:e.read_json(folder/file) for k,file in (
                ('run','RUN_COMPLETE.json'),('configuration','configuration.json'),('front','final_feasible_nondominated.json'))})
        with gzip.open(cls.result/'raw_runs.jsonl.gz','wt') as output:
            for r in cls.records:output.write(json.dumps(r)+'\n')
        cls.manifest()
        cls.call('--recompute','--out',str(f.root/'report'))
        for file in ('RESULT_TABLE.csv','PER_RUN.csv','PER_INSTANCE.csv','QUICK_RESULT.json','INSTANCE_BOUNDS.json'):
            shutil.copy2(f.root/'report'/file,cls.result/file)
        cls.manifest()

    @classmethod
    def manifest(cls):
        files={str(path.relative_to(cls.root)):e.digest(path) for path in cls.root.rglob('*') if path.is_file()
               and path!=cls.root/'assets/manifest.json' and '__pycache__' not in path.parts and 'runtime_inputs' not in path.parts}
        e.write_json(cls.root/'assets/manifest.json',dict(files=files))

    @classmethod
    def call(cls,*args):
        proc=subprocess.run([sys.executable,str(cls.root/'run.py'),*args],cwd=str(cls.fixture.root),
                            stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
        if proc.returncode:raise RuntimeError(proc.stdout[-6000:])
        return proc.stdout

    @classmethod
    def tearDownClass(cls):fixtures.CombinedChecks.tearDownClass()

    def test_all_nine_methods_exactly_replay(self):
        f=self.fixture
        for m,record in zip(e.ALL_METHODS,self.records):
            out=f.root/('replay-'+m)
            self.call('--run','--instance',f.name,'--seed',str(f.seed),'--method',m,'--out',str(out))
            folder=out/'runs'/f.name/f'seed_{f.seed}'/m
            result=e.read_json(folder/'RUN_COMPLETE.json')
            self.assertEqual(result['final_hypervolume'],record['run']['final_hypervolume'],m)
            self.assertEqual(result['scenario_digest'],record['run']['scenario_digest'],m)
            self.assertEqual(e.read_json(folder/'final_feasible_nondominated.json'),record['front'],m)

    def test_hv_verification_no_search(self):
        self.assertIn('no search or training',self.call('--verify'))
        self.assertFalse((self.root/'new_runs').exists())

    def test_complete_archive_with_900_synthetic_rows(self):
        # Synthetic statistics fixture, never presented as research results.
        base=self.fixture.root/'assembly';old=base/'old';source=base/'extension'
        cfg=copy.deepcopy(e.legacy.CONFIG)
        names=[f'fixture-{i}' for i in range(10)]
        original=dict(config=cfg,instances=names,runs=700,environment=self.fixture.old['environment'])
        original['sha256']=e.canonical_digest(original);e.write_json(old/'PLAN.json',original)
        extension=dict(config={**cfg,'protocol':e.PROTOCOL,'methods':list(e.ALL_METHODS)},instances=names,runs=900,new_runs=200,
            source_root=str(old),source_plan_sha256=original['sha256'],source_plan_file_sha256=e.digest(old/'PLAN.json'),
            original_result_files={},runtime_files={})
        extension['sha256']=e.canonical_digest(extension);e.write_json(source/'PLAN.json',extension)
        shutil.copytree(self.fixture.source/'models',old/'models')
        sample=e.read_json(self.fixture.source/'instances'/f'{self.fixture.name}.json')
        for n in names:e.write_json(old/'instances'/f'{n}.json',sample)
        e.write_json(old/'instances/catalog.json',dict(instances=names))
        for n in names:
            for s in cfg['algorithm_seeds']:
                for index,m in enumerate(e.ALL_METHODS):
                    folder=(old if m in e.legacy.METHODS else source)/'runs'/n/f'seed_{s}'/m
                    point=[1+.03*index+.01*(s-cfg['algorithm_seeds'][0])]*3
                    norm=dict(minimum=[0]*3,maximum=[10]*3,objective_divisor=[1]*3,reference=list(e.hv.REFERENCE))
                    r=dict(completed=True,instance=n,algorithm_seed=s,strategy=m,evaluation_budget=400,
                        plan_sha256=original['sha256'] if m in e.legacy.METHODS else extension['sha256'],
                        real_objective_evaluation_count=400,scenario_digest=n,optimisation_seconds=1,
                        final_hypervolume=e.hv.exact_hv_3d([[v/10 for v in point]],e.hv.REFERENCE))
                    e.write_json(folder/'RUN_COMPLETE.json',r)
                    e.write_json(folder/'configuration.json',dict(scenario_digest=n,normalization=norm))
                    e.write_json(folder/'final_feasible_nondominated.json',[dict(training_feasible=True,source_run=n,
                        optimisation_objectives=dict(zip(e.hv.OBJECTIVES,point)))])
        with contextlib.redirect_stdout(io.StringIO()):
            e.hv.execute(source,source.parent/(source.name+'_hv_union_v1'))
            p.build(source,base/'package.zip')
        with zipfile.ZipFile(base/'package.zip') as archive:
            names=archive.namelist()
            self.assertIn('ICM_B400_900_Compact/results/raw_runs.jsonl.gz',names)
            self.assertFalse(any('/tests/' in n or '__pycache__' in n or n.endswith('.log') for n in names))
            archive.extractall(base/'moved')
        moved=base/'moved/ICM_B400_900_Compact'
        proc=subprocess.run([sys.executable,str(moved/'run.py'),'--verify'],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
        self.assertEqual(proc.returncode,0,proc.stdout[-6000:])
        self.assertIn('All 900 original HV',proc.stdout)

    def test_tamper_detection(self):
        path=self.root/'assets/design.json';before=path.read_bytes()
        try:
            path.write_bytes(before+b' ')
            with self.assertRaisesRegex(RuntimeError,'checksum mismatch'):self.call('--verify')
        finally:path.write_bytes(before)


if __name__=='__main__':unittest.main()
