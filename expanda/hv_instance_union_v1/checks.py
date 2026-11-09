import unittest
import contextlib
import io
import tempfile
from pathlib import Path
import recompute as r

class HVChecks(unittest.TestCase):
    def test_expanded_plan_integrity(self):
        plan={'config':{'protocol':'frozen-old-components-b400-recheck-v1','methods':list(r.METHODS[:-1]),
              'algorithm_seeds':list(range(10)),'population_size':20,'evaluation_budget':400},
              'instances':[f'i{i}' for i in range(10)],'runs':700}
        plan['sha256']=r.hashlib.sha256(r.json.dumps(plan,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
        design=r.experiment_design(plan,Path('unused'))
        self.assertEqual(len(design['methods']),7)
        self.assertEqual(design['plan_sha'],plan['sha256'])
        plan['runs']=699
        with self.assertRaises(ValueError):r.experiment_design(plan,Path('unused'))

    def test_full_700_run_fixture_and_missing_front(self):
        with tempfile.TemporaryDirectory(prefix='hv-700-fixture-') as temp:
            source=Path(temp)/'source';out=Path(temp)/'output'
            plan={'config':{'protocol':'frozen-old-components-b400-recheck-v1','methods':list(r.METHODS[:-1]),
                  'algorithm_seeds':list(range(10)),'population_size':20,'evaluation_budget':400},
                  'instances':[f'i{i}' for i in range(10)],'runs':700}
            plan['sha256']=r.hashlib.sha256(r.json.dumps(plan,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
            r.write_json(source/'PLAN.json',plan)
            last=None
            for name in plan['instances']:
                for seed in range(10):
                    for index,method in enumerate(r.METHODS[:-1]):
                        folder=source/'runs'/name/f'seed_{seed}'/method
                        point=[1+.02*seed+.03*index]*3
                        hv=r.exact_hv_3d([[v/10 for v in point]],r.REFERENCE)
                        r.write_json(folder/'RUN_COMPLETE.json',dict(completed=True,instance=name,algorithm_seed=seed,
                            strategy=method,plan_sha256=plan['sha256'],real_objective_evaluation_count=400,
                            scenario_digest=name,final_hypervolume=hv,optimisation_seconds=1))
                        r.write_json(folder/'configuration.json',{'scenario_digest':name,'normalization':{
                            'minimum':[0]*3,'maximum':[10]*3,'objective_divisor':[1]*3,'reference':list(r.REFERENCE)}})
                        last=folder/'final_feasible_nondominated.json'
                        r.write_json(last,[{'training_feasible':True,'source_run':name,'optimisation_objectives':dict(zip(r.OBJECTIVES,point))}])
            with contextlib.redirect_stdout(io.StringIO()):result=r.execute(source,out)
            self.assertEqual(result['runs'],700);self.assertEqual(result['method_count'],7)
            self.assertEqual(result['max_old_hv_replay_error'],0)
            self.assertNotIn('I_C_M_K8_TRANSFORMER',result['methods'])
            self.assertEqual(result['new_objective_evaluations'],0)
            b=r.read_json(out/'INSTANCE_BOUNDS.json')['instances']['i0']
            self.assertEqual(b['front_point_count_with_duplicates'],70)
            last.unlink()
            with self.assertRaises(FileNotFoundError):r.load_runs(source)

    def test_bounds_use_union_not_global_front(self):
        # Each run has its own valid front. The poorer run still defines maxima.
        bounds=r.bounds_of([[(1.,2.,3.)],[(4.,5.,6.)]])
        self.assertEqual(bounds['minimum'],[1.,2.,3.])
        self.assertEqual(bounds['maximum'],[4.,5.,6.])
        self.assertEqual(r.normalize([(1.,2.,3.),(4.,5.,6.)],bounds),[(0.,0.,0.),(1.,1.,1.)])

    def test_hv_analytic_and_union_overlap(self):
        self.assertAlmostEqual(r.exact_hv_3d([(0,0,0)],r.REFERENCE),1.331)
        self.assertAlmostEqual(r.exact_hv_3d([(1,1,1)],r.REFERENCE),.001)
        # Two boxes, exact inclusion/exclusion.
        a=(.2,.6,.4);b=(.6,.2,.4)
        expected=.9*.5*.7+.5*.9*.7-.5*.5*.7
        self.assertAlmostEqual(r.exact_hv_3d([a,b],r.REFERENCE),expected)

    def test_unit_conversion_cancels_in_shared_bounds(self):
        fronts=[[(2.,3.,4.),(4.,2.,3.)],[(3.,5.,2.)]]
        bound=r.bounds_of(fronts)
        factors=(100, .001, 24)
        changed=[[tuple(p[j]*factors[j] for j in range(3)) for p in f] for f in fronts]
        second=r.bounds_of(changed)
        for a,b in zip(fronts,changed):
            self.assertAlmostEqual(r.exact_hv_3d(r.normalize(a,bound),r.REFERENCE),r.exact_hv_3d(r.normalize(b,second),r.REFERENCE))

    def test_constant_dimension_and_empty_front(self):
        b=r.bounds_of([[(1,2,3)],[(1,5,6)]])
        self.assertEqual(b['constant_axes'],[0])
        self.assertEqual(r.normalize([(1,5,6)],b),[(0,1,1)])
        self.assertEqual(r.exact_hv_3d([],r.REFERENCE),0)
        with self.assertRaises(ValueError):r.bounds_of([[],[]])

    def test_paired_ratio_not_ratio_of_means(self):
        rows=[]
        for seed,b,v in ((1,1.,2.),(2,10.,10.)):
            rows.extend([dict(instance='a',algorithm_seed=seed,method='BASE',hv=b),dict(instance='a',algorithm_seed=seed,method='M',hv=v)])
        summary=r.compare(rows,'M','BASE',1)
        self.assertAlmostEqual(summary['percent']['mean'],50.)
        rows[0]['hv']=0
        self.assertIsNone(r.compare(rows,'M','BASE',1)['percent']['mean'])

if __name__=='__main__':unittest.main()
