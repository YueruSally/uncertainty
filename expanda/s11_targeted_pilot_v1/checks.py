"""Pilot-only scope, precise stop targeting, end-to-end execution and reuse."""
from pathlib import Path
import contextlib,copy,json,os,subprocess,sys,tempfile,unittest
import pilot
from pause_large import matching_processes,pause

class ScopeChecks(unittest.TestCase):
    def test_small_scope_preserves_algorithm_and_full_run_budget(self):
        small=pilot.read_json(pilot.PILOT/'protocol.json');large=pilot.read_json(pilot.ROOT/'protocol.json')
        changed={k for k in small if small[k]!=large.get(k)}
        permitted={'protocol','calibration_per_configuration','train_per_configuration','validation_per_configuration',
            'screen_per_configuration','algorithm_seeds','instance_master_seed','ccp_seed_base','collection_seed_base',
            'invocation_seconds','pilot_only','automatic_expansion','parent_version','acceptance'}
        self.assertEqual(changed,permitted)
        self.assertEqual(small['population'],250);self.assertEqual(small['budget'],50000)
        self.assertEqual(small['progress_denominator'],50000);self.assertEqual(small['configurations'],['S11'])
        self.assertEqual(len(small['training_seeds']),3)
        n=small['train_per_configuration']+small['validation_per_configuration']
        full=small['calibration_per_configuration']*7+n*2+small['screen_per_configuration']*len(small['algorithm_seeds'])*7
        self.assertEqual((full,n),(27,3));self.assertFalse(small['automatic_expansion'])

    def test_pause_matching_excludes_other_jobs_and_users(self):
        root=Path('/tmp/large');script=Path('/tmp/s11_targeted_v1/run.py')
        lines=[f'10 501 1 python3 -u {script} --stage all --out {root}',
            f'11 501 10 /usr/bin/python3 {script} --worker --out {root}',
            f'12 501 1 python3 {script} --out /tmp/large-extra',
            f'13 502 1 python3 {script} --out {root}',
            f'14 501 1 grep python3 {script} --out {root}',
            f'15 501 1 python3 /tmp/small/pilot.py --out {root}']
        r=matching_processes('\n'.join(lines),root,script,501)
        self.assertEqual(set(r),{10,11});self.assertFalse(r[10]['worker']);self.assertTrue(r[11]['worker'])

    def test_pause_only_terminates_created_target_process(self):
        with tempfile.TemporaryDirectory(prefix='s11-stop-fixture-') as tmp:
            root=Path(tmp);script=root/'run.py';script.write_text('import time\ntime.sleep(60)\n')
            target=root/'target';target.mkdir();(target/'keep.txt').write_text('completed results')
            a=subprocess.Popen([sys.executable,str(script),'--out',str(target)])
            b=subprocess.Popen([sys.executable,str(script),'--out',str(root/'unrelated')])
            try:
                pause(target,script,timeout=5)
                a.wait(timeout=5);self.assertIsNone(b.poll());self.assertEqual((target/'keep.txt').read_text(),'completed results')
            finally:
                for p in (a,b):
                    if p.poll() is None:p.terminate()
                    p.wait(timeout=5)

class PipelineChecks(unittest.TestCase):
    def test_real_s11_small_fixture_full_pipeline_and_reuse(self):
        with tempfile.TemporaryDirectory(prefix='s11-pilot-check-') as tmp:
            root=Path(tmp);sample=pilot.read_json(pilot.ROOT.parent/'formal_icm_s0_v1/assets/search_instances/prescreen-search-S0-001.json')
            src=dict(source_data_sha256=sample['source_data_sha256'],train_od_pool=sorted({(b['origin'],b['destination']) for b in sample['batches']}),test_od_pool=[['held-out-a','held-out-b']],instances=[])
            src['catalog_digest']=pilot.canonical_digest(src);pilot.write_json(root/'source.json',src)
            s=pilot.read_json(pilot.PILOT/'protocol.json')
            s.update(budget=400,population=20,m_start_generation=0,states={'I':6,'C':3,'M':6},milestones=[100,400],epochs=1,checkpoint_epochs=[1],train_per_configuration=1,source_catalog_digest=src['catalog_digest'])
            pilot.write_json(root/'fixture.json',s)
            code="""from pathlib import Path
import sys
sys.path.insert(0,sys.argv[1])
import pilot
real=pilot.read_json
settings=real(Path(sys.argv[2])/'fixture.json')
pilot.read_json=lambda p:settings if Path(p)==pilot.PILOT/'protocol.json' else real(p)
sys.argv=[str(pilot.PILOT/'pilot.py'),'--stage','all','--workers','2','--out',str(Path(sys.argv[2])/'out'),'--source-catalog',str(Path(sys.argv[2])/'source.json')]
pilot.main()
"""
            for iteration in range(2):
                log=root/f'run-{iteration}.log'
                with log.open('w') as f:r=subprocess.run([sys.executable,'-c',code,str(pilot.PILOT),str(root)],stdout=f,stderr=subprocess.STDOUT,timeout=480)
                self.assertEqual(r.returncode,0,log.read_text()[-10000:])
                result=pilot.read_json(root/'out/QUICK_RESULT.json')
                self.assertTrue(result['completed']);self.assertTrue(result['pilot_only'])
                self.assertFalse(result['automatic_expansion_started']);self.assertFalse(result['final_test_opened'])
                self.assertNotIn('all_components_met_reliable_gain',result)
                self.assertEqual(result['searches'],14);self.assertEqual(set(result['pilot_direction']),set('ICM'))
                self.assertTrue((root/'out/00_小试方向及耗时.csv').exists())
                for c in 'ICM':
                    self.assertEqual(len(result['pilot_direction'][c]['per_instance']),2)
                    model=pilot.read_json(root/'out/models'/c/'BUNDLE.json')
                    self.assertFalse(set(model['training_instances'])&set(model['validation_instances']))
                if iteration==0:frozen=pilot.digest(root/'out/models/FROZEN.json')
                else:
                    self.assertEqual(pilot.digest(root/'out/models/FROZEN.json'),frozen)
                    self.assertNotIn('[S11-TARGET] start ',log.read_text())

if __name__=='__main__':unittest.main()
