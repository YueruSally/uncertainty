"""Local verification; synthetic records are test fixtures, never experiment output."""
import copy
import csv
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import run as task

SOURCE=Path(os.environ.get('ICM_240_FIXTURE','/Users/yueru/uncertainty/output/ICM_I_C_M_Complete_Reproduction_r3/raw_results'))


@unittest.skipUnless((SOURCE/'PLAN.json').exists(), 'Set ICM_240_FIXTURE to original 240-run results')
class ExtensionChecks(unittest.TestCase):
    def make_plan(self,out):
        old=task.read_json(SOURCE/'PLAN.json')
        # Local test machine versions differ; production CLI still enforces exact versions.
        with patch.object(task.torch,'__version__',old['torch_version']),patch.object(task.numpy,'__version__',old['numpy_version']):
            return task.prepare(SOURCE,out,3)

    def test_real_combined_worker_and_resume_validation(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d)/'new';out.mkdir();plan=self.make_plan(out)
            n=plan['instances'][0];s=plan['config']['algorithm_seeds'][0]
            folder=out/'runs'/n/f'seed_{s}'/task.RULE;folder.mkdir(parents=True)
            # Run a genuine local Base: cross-platform scenario floating-point hashes can differ.
            baseline=Path(d)/'local_base'
            task.run_episode(task.PACKAGE/'assets/data/data_expanded.xlsx',
                task.PACKAGE/'assets/search_instances'/f'{n}.json',baseline,None,
                task._normalizer(task.PACKAGE/'assets/reward_normalization.json'),algorithm_seed=s,
                evaluation_budget=400,pop_size=20,generations=1000,evaluation_mode='ccp',component='BASE',
                scenario_count=100,ccp_confidence=.9,uncertainty_profile='transport_border_saa_v1')
            local_config=task.read_json(baseline/'configuration.json')
            pairing=task.validate_pairing
            with patch.object(task,'validate_pairing',side_effect=lambda cfg,base: pairing(cfg,local_config)):
                task.worker(out,n,s,task.RULE,folder,plan['sha256'])
            wrong=copy.deepcopy(local_config);wrong['scenario_digest']='wrong'
            with self.assertRaises(ValueError):pairing(local_config,wrong)
            row=task.read_json(folder/'RUN_COMPLETE.json')
            task.validate(row,task.RULE,s,plan['sha256'])
            self.assertEqual(row['real_objective_evaluation_count'],400)
            self.assertEqual(row['combined_hidden_candidate_objective_evaluations'],0)
            damaged=copy.deepcopy(row);damaged['component_summaries']['C']['crossover_k8_selection_mode']='transformer'
            with self.assertRaises(ValueError):task.validate(damaged,task.RULE,s,plan['sha256'])

    def test_full_union_statistics_and_frozen_plan(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d)/'new';out.mkdir();plan=self.make_plan(out)
            self.assertEqual(self.make_plan(out),plan)
            for n in plan['instances']:
                for s in plan['config']['algorithm_seeds']:
                    old=SOURCE/'runs'/n/f'seed_{s}'/'BASE'
                    folder=out/'runs'/n/f'seed_{s}'/task.RULE;folder.mkdir(parents=True)
                    for filename in ('configuration.json','final_feasible_nondominated.json'):
                        (folder/filename).write_bytes((old/filename).read_bytes())
                    row=task.read_json(old/'RUN_COMPLETE.json')
                    row.update(strategy=task.RULE,plan_sha256=plan['sha256'])
                    task.write_json(folder/'RUN_COMPLETE.json',row)
            target=Path(d)/'statistics'
            summary=task.report.execute(out,target)
            self.assertEqual(summary['runs'],270)
            self.assertEqual(summary['methods'][task.RULE]['vs_base']['absolute']['ties'],30)
            for f,count in [('STATISTICS.csv',13),('PER_INSTANCE_STATISTICS.csv',39),('00_组件对比及运行时间.csv',5),('PER_RUN.csv',270)]:
                with (target/f).open(encoding='utf-8-sig') as h:self.assertEqual(len(list(csv.DictReader(h))),count)
            comparison=summary['methods']['I_C_M_K8_TRANSFORMER']['vs_rule']
            self.assertEqual(sum(comparison['absolute'][k] for k in ('wins','ties','losses')),30)
            wrong=copy.deepcopy(plan);wrong['config']['algorithm_seeds'][0]=961001
            task.write_json(out/'PLAN.json',wrong)
            with self.assertRaises(ValueError):task.report.hv.load_runs(out)

    def test_reject_expanded_source_and_wrong_environment(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d);old=task.read_json(SOURCE/'PLAN.json')
            with patch.object(task.torch,'__version__','invalid'):
                with self.assertRaisesRegex(ValueError,'versions'):task.prepare(SOURCE,out,3)
            with patch.object(task.torch,'__version__',old['torch_version']),patch.object(task.numpy,'__version__',old['numpy_version']):
                with self.assertRaisesRegex(ValueError,'parallelism'):task.prepare(SOURCE,out,1)

if __name__=='__main__':unittest.main()
