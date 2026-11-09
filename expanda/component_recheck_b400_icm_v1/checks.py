import contextlib
import copy
import io
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import extend as e
from controller import SimpleSelector
from runtime_common import read_json,write_json


class CombinedChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory(prefix='b400-icm-check-');cls.root=Path(cls.temp.name)
        cls.source=cls.root/'old';cls.out=cls.root/'extension';cls.out.mkdir()
        sample=read_json(e.legacy.PACKAGE/'assets/search_instances/prescreen-search-S0-001.json')
        sample.pop('instance_digest');sample['development_role']='b400_replication';sample['instance_digest']=e.canonical_digest(sample)
        cls.name=sample['instance_id'];cls.seed=961001
        write_json(cls.source/'instances'/f'{cls.name}.json',sample)
        config=copy.deepcopy(e.legacy.CONFIG);config.update(instance_count=1,algorithm_seeds=[cls.seed])
        cat={'instances':[{'instance_id':cls.name,'instance_digest':sample['instance_digest']}],
             'config_digest':e.canonical_digest(config)}
        cat['catalog_digest']=e.canonical_digest(cat);write_json(cls.source/'instances/catalog.json',cat)
        e.legacy.prepare_models(cls.source);cls.old=e.legacy.make_plan(cls.source,cat,config)
        with (cls.root/'old.log').open('w') as f,contextlib.redirect_stdout(f):
            e.legacy.execute_jobs(cls.source,cls.old,1)
        cls.original_hashes={str(p):e.digest(p) for p in cls.source.rglob('*') if p.is_file()}
        cls.plan=e.make_plan(cls.source,cls.out,cls.old)
        cls.rows=[]
        for method in e.ADDED:
            folder=cls.out/'runs'/cls.name/f'seed_{cls.seed}'/method;folder.mkdir(parents=True)
            log=cls.root/(method+'.log')
            with log.open('w') as f:
                # Reject any network loading in the heuristic arm.
                code="import sys;from pathlib import Path;sys.path.insert(0,sys.argv[1]);import extend as e;"
                if method==e.RULE:
                    code+="\ndef prohibited(*a,**kw):raise AssertionError('Rule arm loaded a learned model')\ne.FrozenScaleRanker=prohibited\nimport combined\ncombined.FrozenIResidualGate=prohibited\ncombined.FrozenCResidualGate=prohibited\n"
                code+="e.worker(Path(sys.argv[2]),sys.argv[3],int(sys.argv[4]),sys.argv[5],Path(sys.argv[6]),sys.argv[7])"
                proc=subprocess.run([sys.executable,'-c',code,str(e.HERE),str(cls.out),cls.name,str(cls.seed),method,str(folder),cls.plan['sha256']],stdout=f,stderr=subprocess.STDOUT)
            if proc.returncode:raise RuntimeError(log.read_text()[-6000:])
            cls.rows.append(read_json(folder/'RUN_COMPLETE.json'))
        # Replay the unmodified historical learned-ICM worker in a cold process.
        folder=cls.root/'original-icm';folder.mkdir();log=cls.root/'original-icm.log'
        code="import sys;from pathlib import Path;sys.path.insert(0,sys.argv[1]);import extend as e;from types import SimpleNamespace;root=Path(sys.argv[2]);e.legacy.engine.run_worker(SimpleNamespace(data=e.legacy.PACKAGE/'assets/data/data_expanded.xlsx',instance=root/'instances'/sys.argv[3],out=Path(sys.argv[4]),seed=961001,method=e.LEARN,plan_sha='reference',normalization=e.legacy.PACKAGE/'assets/reward_normalization.json',i_bundle=root/'models/I_BUNDLE.json',c_bundle=root/'models/C_BUNDLE.json',m_bundle=root/'models/M_BUNDLE.json'))"
        with log.open('w') as f:proc=subprocess.run([sys.executable,'-c',code,str(e.HERE),str(cls.source),cls.name+'.json',str(folder)],stdout=f,stderr=subprocess.STDOUT)
        if proc.returncode:raise RuntimeError(log.read_text()[-6000:])
        cls.reference=read_json(folder/'RUN_COMPLETE.json')
        cls.reference_front=read_json(folder/'final_feasible_nondominated.json')
        print('ICM real-search fixtures:',cls.root,flush=True)

    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()

    def test_budget_scenarios_and_selectors(self):
        b=read_json(self.source/'runs'/self.name/f'seed_{self.seed}'/'BASE/RUN_COMPLETE.json')
        for method,row in zip(e.ADDED,self.rows):
            e.validate(row,method,self.seed,self.plan['sha256'])
            self.assertEqual(row['real_objective_evaluation_count'],400)
            self.assertEqual(row['scenario_digest'],b['scenario_digest'])
            self.assertEqual(row['combined_hidden_candidate_objective_evaluations'],0)
            self.assertEqual(row['new_ppo_updates'],0)
            self.assertGreater(row['component_summaries']['C']['crossover_k8_eligible_pool_events'],0)
            self.assertGreater(row['component_summaries']['M']['k8_scale_search']['pool_events'],0)

    def test_learned_icm_matches_historical_worker(self):
        self.assertEqual(self.rows[1]['final_hypervolume'],self.reference['final_hypervolume'])
        front=read_json(self.out/'runs'/self.name/f'seed_{self.seed}'/e.LEARN/'final_feasible_nondominated.json')
        self.assertEqual(front,self.reference_front)

    def test_rule_adapter_is_exact_simple_and_no_weights(self):
        row={'candidate_features':[[float(i)] for i in range(8)]}
        self.assertEqual(SimpleSelector().select(row),0)
        row['candidate_features']=[[0.] for _ in range(8)]
        self.assertEqual(SimpleSelector().select(row),1)
        self.assertEqual(self.rows[0]['learned_components'],[])
        self.assertEqual(self.rows[0]['component_summaries']['C']['deferred_selector'],'deterministic Simple; no network')

    def test_source_untouched_and_resumable_plan(self):
        self.assertEqual(self.original_hashes,{str(p):e.digest(p) for p in self.source.rglob('*') if p.is_file()})
        self.assertEqual(e.make_plan(self.source,self.out,self.old),self.plan)

    def test_resume_does_not_rerun_completed_searches(self):
        stamps={str(p):p.stat().st_mtime_ns for p in (self.out/'runs').rglob('RUN_COMPLETE.json')}
        with mock.patch.object(e,'finish',return_value={'fixture':True}) as finish,contextlib.redirect_stdout(io.StringIO()):
            e.execute_jobs(self.out,self.plan,1)
        self.assertEqual(len(finish.call_args[0][2]),2)
        self.assertEqual(stamps,{str(p):p.stat().st_mtime_ns for p in (self.out/'runs').rglob('RUN_COMPLETE.json')})

if __name__=='__main__':unittest.main()
