from pathlib import Path
from itertools import product
from contextlib import redirect_stdout
import copy
import io
import json
import math
import random
import tempfile
import unittest
import audit


class Checks(unittest.TestCase):
    def test_hv_known_union_and_boundary(self):
        self.assertAlmostEqual(audit.hv([[0,.5,.5],[.5,0,0]], [1,1,1]), .625)
        self.assertEqual(audit.hv([[1,0,0]], [1,1,1]), 0)
        self.assertEqual(audit.hv([], [1,1,1]), 0)
        self.assertAlmostEqual(audit.hv([[0,0,0],[.5,.5,.5],[0,0,0]], [1,1,1]), 1)
        with self.assertRaises(ValueError): audit.hv([[math.nan,0,0]])

    def test_hv_against_exhaustive_grid_cell_union(self):
        rng = random.Random(56)
        for _ in range(30):
            points = [[rng.randrange(-2,5)/4 for _ in range(3)] for _ in range(6)]
            axes = [sorted({p[d] for p in points if p[d] < 1}|{1.}) for d in range(3)]
            intervals = [list(zip(a,a[1:])) for a in axes]
            expected = 0.
            for cells in product(*intervals):
                center = [sum(c)/2 for c in cells]
                if any(all(p[d] <= center[d] for d in range(3)) for p in points):
                    expected += math.prod(b-a for a,b in cells)
            self.assertAlmostEqual(audit.hv(points,[1,1,1]), expected)

    def test_best_compares_best_with_best_and_counts_all_cost(self):
        rows = []
        for m in audit.METHODS:
            values = [.4,.6,.5] if m=='BASE' else [.3,.7,.3]
            for seed,h in enumerate(values):
                rows.append(dict(instance='A',configuration='S0',seed=seed,arm=m,union_hv=h,
                                 real_objective_evaluation_count=50000,optimisation_seconds=10))
        stats = audit.best_of_seeds(rows,50000)
        row = stats[1]
        self.assertEqual(row['total_evaluation_cap'],150000)
        self.assertEqual(row['total_seconds'],30)
        self.assertAlmostEqual(row['best_vs_base_percent'],100*(.7-.6)/.6)
        paired = next(r for r in audit.comparisons(rows,'union_hv') if r['method']==audit.METHODS[1] and r['reference']=='BASE')
        self.assertLess(paired['mean_paired_percent'],0)
        with self.assertRaises(ValueError): audit.best_of_seeds(rows[:-1],50000)
        self.assertIsNone(audit.percent(.5,0))

    def make_fixture(self, directory):
        root = directory/'source'; report = directory/'report'; source = directory/'supervision'
        def save(path,obj):
            path.parent.mkdir(parents=True,exist_ok=True);audit.write(path,obj)
        norm = dict(minimum=[0.,0.,0.],maximum=[1.,1.,1.],reference=[1.1]*3)
        entries = [dict(instance_id='A',configuration='S0'),dict(instance_id='B',configuration='S1')]
        settings = dict(budget=50000,screen_seeds=[11,12,13],milestones=[400,50000])
        plan = dict(settings=settings,source=str(source),screen=entries,normalization=norm)
        plan['sha256'] = audit.canonical(plan);save(root/'PLAN.json',plan)
        rows = []
        for i,entry in enumerate(entries):
            name = entry['instance_id'];cfg = entry['configuration']
            save(root/'instances'/(name+'.json'),dict(instance_digest='instance-'+name,batches=[dict(quantity=1.,ET=0.,LT=1.)]))
            points = []
            group = []
            for s in settings['screen_seeds']:
                for j,m in enumerate(audit.METHODS):
                    p = [.1+i*.1+(s-11)*.01+j*.02, .2+j*.01, .3+(s-11)*.02]
                    h = math.prod(1.1-x for x in p)
                    r = dict(instance=name,configuration=cfg,seed=s,arm=m,completed=True,
                        plan_sha256=plan['sha256'],instance_digest='instance-'+name,
                        phase='screen',stochastic=False,hidden_candidate_objective_evaluations=0,ppo_updates=0,
                        evaluation_budget=50000,real_objective_evaluation_count=50000,
                        scenario_digest='scenario-'+name,final_front_raw=[p],final_hypervolume=h,
                        selection={c:dict(calls=10,fallbacks=4,switches=6,unsupported=0) for c in 'ICM'},
                        milestone_hv=[dict(evaluation=n,hv=h) for n in (400,50000)],optimisation_seconds=10.)
                    folder = root/'screen'/name/f'seed_{s}'/m
                    save(folder/'RUN_COMPLETE.json',r)
                    save(folder/'final_archive_lineage.json',[dict(optimisation_objectives=dict(zip(('cost','emission','makespan'),p)))])
                    group.append(r);points.append(p)
            lo=[min(p[d] for p in points) for d in range(3)];hi=[max(p[d] for p in points) for d in range(3)]
            for r in group:
                p = r['final_front_raw'][0]
                u = math.prod(1.1-(p[d]-lo[d])/(hi[d]-lo[d]) for d in range(3))
                rows.append(dict(r,union_hv=u,final_hypervolume=u))
        save(report/'PAIRED_RUNS.json',rows)
        q = dict(completed=True,methods={m:dict(comparisons={}) for m in audit.METHODS})
        for r in audit.comparisons(rows,'union_hv'):
            q['methods'][r['method']]['comparisons'][r['reference']] = {'percent':{'mean':r['mean_paired_percent']}}
        save(report/'QUICK_RESULT.json',q)
        for folder in (source/'shared_model',root/'independent_model'):
            b = dict(checkpoint_sha256={},normalization=norm,
                training_instances={c:['training'] for c in 'ICM'},validation_instances={c:['validation'] for c in 'ICM'},
                settings=settings,coverage={},training_states={},validation_states={},selected={},target_scales={},loss='rank',feature_version='fixture')
            save(folder/'BUNDLE.json',b)
        return root,report

    def test_end_to_end_no_source_mutation_and_report_mismatch_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp);root,report=self.make_fixture(base)
            before={str(p):audit.digest(p) for p in base.rglob('*.json')}
            with redirect_stdout(io.StringIO()): result=audit.execute(root,report,base/'audit')
            self.assertEqual(result['checked_runs'],24)
            self.assertEqual(result['new_objective_evaluations'],0)
            self.assertTrue(all(audit.digest(Path(p))==sha for p,sha in before.items()))
            data=audit.read(report/'PAIRED_RUNS.json');data[0]['final_hypervolume']+=.1
            audit.write(report/'PAIRED_RUNS.json',data)
            with redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError,'HV recomputation'):
                audit.execute(root,report,base/'bad')

    def test_pairing_and_budget_invalid_runs_fail(self):
        for key,value in [('scenario_digest','wrong'),('real_objective_evaluation_count',400),('hidden_candidate_objective_evaluations',1)]:
            with self.subTest(key=key), tempfile.TemporaryDirectory() as tmp:
                base=Path(tmp);root,report=self.make_fixture(base)
                p=root/'screen/A/seed_11/BASE/RUN_COMPLETE.json'
                r=audit.read(p);r[key]=value;audit.write(p,r)
                with redirect_stdout(io.StringIO()), self.assertRaises(ValueError):audit.execute(root,report,base/'audit')


if __name__ == '__main__': unittest.main()
