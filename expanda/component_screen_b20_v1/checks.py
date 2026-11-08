"""Small correctness checks, never used for component selection."""
from common import *
import tempfile,unittest,contextlib,copy,subprocess
import numpy as np
from worker import work
from learning import fit_normalization,train_component,FrozenPolicy,labels,pack,scaler,decide
from report import comparison

def isolated_work(root,s,name,seed,arm,out,sha):
    spec=out.parent/(out.name+'-job.json');write_json(spec,{'root':str(root),'settings':s,'name':name,'seed':seed,'arm':arm,'out':str(out),'sha':sha})
    code="import sys;from pathlib import Path;sys.path.insert(0,sys.argv[1]);from worker import work;from common import read_json;x=read_json(Path(sys.argv[2]));work(Path(x['root']),x['settings'],x['name'],x['seed'],x['arm'],Path(x['out']),x['sha'])"
    log=out.parent/(out.name+'-process.log')
    with log.open('w') as f:
        result=subprocess.run([sys.executable,'-c',code,str(ROOT),str(spec)],stdout=f,stderr=subprocess.STDOUT)
    if result.returncode:raise RuntimeError(log.read_text()[-6000:])

import torch

class Correctness(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1);cls.temp=tempfile.TemporaryDirectory(prefix='b20-check-');cls.root=Path(cls.temp.name)
        cls.s=read_json(ROOT/'protocol.json');cls.s.update(population=20,budget=400,progress_denominator=400,states={'I':6,'C':3,'M':6},milestones=[100,400],epochs=1,checkpoint_epochs=[1],configurations=['S0'],train_per_configuration=1,validation_per_configuration=1)
        x=read_json(ROOT.parent/'formal_icm_s0_v1/assets/search_instances/prescreen-search-S0-001.json');x.pop('instance_digest');x['development_role']='training';x['instance_digest']=canonical_digest(x);cls.info=x;cls.name=x['instance_id']
        write_json(cls.root/'instances'/f'{cls.name}.json',x)
        cls.outputs={}
        with (cls.root/'worker.log').open('w') as log,contextlib.redirect_stdout(log):
            for arm in ['COLLECT_CM','COLLECT_I']:
                out=cls.root/arm;out.mkdir();isolated_work(cls.root,cls.s,cls.name,12345,arm,out,'test');cls.outputs[arm]=read_json(out/'RUN_COMPLETE.json')
            # Both diagnostic controllers use the same old normalization only for
            # smoke comparison; this fixture does not generate research models.
            cls.norm=read_json(ROOT.parent/'formal_icm_s0_v1/assets/reward_normalization.json');write_json(cls.root/'models/HV_NORMALIZATION.json',cls.norm)
            x=dict(x);x.pop('instance_digest');x['development_role']='screen';x['instance_digest']=canonical_digest(x);write_json(cls.root/'instances'/f'{cls.name}.json',x)
            for arm in ['BASE','I_RULE','C_RULE','M_RULE']:
                out=cls.root/arm;out.mkdir();isolated_work(cls.root,cls.s,cls.name,12345,arm,out,'test');cls.outputs[arm]=read_json(out/'RUN_COMPLETE.json')
        print(f'Correctness artifacts: {cls.root}',flush=True)
    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()
    def test_base_trajectory_unchanged_by_audits(self):
        a,b=self.outputs['BASE'],self.outputs['COLLECT_CM']
        self.assertEqual(a['evaluation_order_digest'],b['evaluation_order_digest']);self.assertEqual(a['final_hypervolume'],b['final_hypervolume'])
        self.assertEqual([r['decision_fingerprint'] for r in read_json(self.root/'BASE/final_archive_lineage.json')],[r['decision_fingerprint'] for r in read_json(self.root/'COLLECT_CM/final_archive_lineage.json')])
    def test_cost_and_phase_accounting(self):
        for arm in ('BASE','I_RULE','C_RULE','M_RULE'):
            r=self.outputs[arm];self.assertIn(r['real_objective_evaluation_count'],(399,400));self.assertEqual(r['hidden_candidate_objective_evaluations'],0)
            self.assertEqual(r['milestone_hv'][-1]['hv'],r['final_hypervolume'])
        for arm,components in [('COLLECT_I','I'),('COLLECT_CM','CM')]:
            r=self.outputs[arm]
            for c in components:self.assertLessEqual(r['audit_evaluations'][c],384);self.assertTrue(all(r['audit_phase_counts'][c]))
    def test_i_prefix_and_rule_are_same(self):
        # Collection stops at initialization; deployment must begin with the
        # identical evaluated Directed prefix, including sample/unsampled slots.
        a=read_json(self.root/'COLLECT_I/DATA.json')
        self.assertEqual(self.outputs['COLLECT_I']['generations_completed'],0)
        # Compare the first unique evaluations rather than final search HV.
        self.assertEqual(self.outputs['COLLECT_I']['initial_evaluation_digest'],self.outputs['I_RULE']['initial_evaluation_digest'])
        self.assertGreater(len(a['rows']),0)
        self.assertEqual(a['rows'][0]['initialization_progress'],0)
    def test_train_only_norm_and_negative_residuals(self):
        paths=[self.root/'COLLECT_I',self.root/'COLLECT_CM'];n=fit_normalization(paths)
        self.assertFalse(n['objective_clipping']);self.assertEqual(n['source'],'training-only')
        row=read_json(self.root/'COLLECT_I/DATA.json')['rows'][0]
        y=labels(row,n);self.assertEqual(y[0],0);self.assertTrue(np.isfinite(y).all())
        d=read_json(paths[0]/'DATA.json');d['role']='validation';p=self.root/'invalid';write_json(p/'DATA.json',d)
        with self.assertRaises(ValueError):fit_normalization([p])
    def test_train_freeze_and_real_inference_all_components(self):
        for c in 'ICM':
            data=read_json(self.root/('COLLECT_I' if c=='I' else 'COLLECT_CM')/'DATA.json')['rows'];rows=[r for r in data if r['component']==c]
            # Vary token count to verify actual grouped training without padding.
            extra=copy.deepcopy(rows[0]);extra['candidate_batch_tokens']=[t[:10] for t in extra['candidate_batch_tokens']]
            tr=rows+[extra];valid=copy.deepcopy(rows)
            for r in valid:r['instance']='validation-fixture'
            bundle=train_component(c,tr,valid,self.norm,self.s,self.root/'models'/c)
            self.assertEqual(len(bundle['checkpoints']),3)
            policy=FrozenPolicy(self.root/'models'/c);self.assertIn(policy.select(rows[0]),range(8))
            out=self.root/(c+'_LEARN');out.mkdir()
            with (self.root/f'{c}-infer.log').open('w') as log,contextlib.redirect_stdout(log):isolated_work(self.root,self.s,self.name,12345,c+'_LEARN',out,'test')
            r=read_json(out/'RUN_COMPLETE.json');self.assertGreater(r['selection'][c]['calls'],0);self.assertEqual(r['hidden_candidate_objective_evaluations'],0)
    def test_valid_inherited_cache_and_audit_isolation(self):
        from controller import ScreenController
        import baseline_uncertainty as base
        from types import SimpleNamespace as NS
        folder=self.root/'cache-unit';folder.mkdir()
        ctl=ScreenController(folder,'fixture','scenario',10,settings=dict(self.s,milestones=[]),arm='BASE',
                             reliable_options={},batches=(),arcs=[],instance=self.info,seeds={'algorithm':1})
        def individual(share):
            x=base.Individual();x.od_allocations={('a','b',0):[NS(path=NS(arcs=[NS(from_node='a',to_node='b',mode='road')]),share=share)]};return x
        calls=[]
        def evaluate(x):calls.append(1);x.objectives=(1.,2.,3.);x.feasible=True
        try:
            a=individual(.5);ctl.evaluate(evaluate,a);ctl.evaluate(evaluate,copy.deepcopy(a));self.assertEqual(len(calls),1)
            b=copy.deepcopy(a);b.od_allocations[('a','b',0)][0].share=.6;ctl.evaluate(evaluate,b);self.assertEqual(len(calls),2)
            ctl.evaluate(evaluate,individual(.5));self.assertEqual(len(calls),3)
            before=len(ctl.cache);fresh=individual(.9)
            ctl._audit('I',[[fresh] for _ in range(8)],{},evaluate,(),{})
            self.assertEqual(ctl.evaluations,3);self.assertEqual(len(ctl.cache),before);self.assertEqual(ctl.audit_counts['I'],1)
            ctl.evaluate(evaluate,fresh);self.assertEqual(ctl.evaluations,4);self.assertEqual(len(calls),5)
            self.assertEqual(ctl.budget_reserve('offspring_pair'),2);self.assertEqual(ctl.budget_reserve('feasibility_boost',3),4)
        finally:ctl.close()
    def test_zero_base_and_pairing_statistics(self):
        rows=[]
        for c in ('S0','S1'):
            for seed in range(3):
                for method,h in [('BASE',1.),('I_LEARN',1.1)]:rows.append(dict(configuration=c,instance=c+'a',seed=seed,arm=method,final_hypervolume=h))
        result=comparison(rows,'I_LEARN','BASE',100);self.assertAlmostEqual(result['percent']['mean'],10)
        rows[0]['final_hypervolume']=0;result=comparison(rows,'I_LEARN','BASE',100)
        self.assertIsNone(result['percent']['mean']);self.assertEqual(result['reference_zero_pairs'],1)
    def test_instance_generator_all_batch_sizes_and_split_separation(self):
        from instances import prepare
        source={'source_data_sha256':self.info['source_data_sha256'],
                'train_od_pool':sorted({(b['origin'],b['destination']) for b in self.info['batches']}),
                'test_od_pool':[['reserved-test-origin','reserved-test-destination']], 'instances':[]}
        source['catalog_digest']=canonical_digest(source)
        sourcepath=self.root/'fixture_source.json';write_json(sourcepath,source)
        s=dict(self.s,source_catalog_digest=source['catalog_digest'],configurations=['S0','S1','S6'],train_per_configuration=1,validation_per_configuration=1,screen_per_configuration=1)
        root=self.root/'generator'
        with (self.root/'generator.log').open('w') as log,contextlib.redirect_stdout(log):cat=prepare(root,s,sourcepath)
        self.assertEqual(len(cat['instances']),9)
        self.assertEqual({r['K'] for r in cat['instances']},{10,20,30})
        self.assertEqual(len({r['batch_signature'] for r in cat['instances']}),9)
        self.assertEqual(prepare(root,s,sourcepath),cat)
        changed=dict(s,budget=401)
        with self.assertRaises(ValueError):prepare(root,changed,sourcepath)
    def test_pair_labels_and_negative_gain(self):
        norm={'minimum':[0,0,0],'maximum':[1,1,1],'reference':[1.1]*3}
        row={'divisor':[1,1,1],'population_raw':[], 'labels':[
            [{'objectives':[.4,.4,.4],'feasible':True}],
            [{'objectives':[.9,.9,.9],'feasible':True}],
            [{'objectives':[.2,.6,.4],'feasible':True},{'objectives':[.6,.2,.4],'feasible':True}]]}
        y=labels(row,norm);self.assertLess(y[1],0)
        from transformer_ppo.archive import exact_hv_3d
        self.assertAlmostEqual(y[2],exact_hv_3d([[.2,.6,.4],[.6,.2,.4]],[1.1]*3)-.7**3)
    def test_zz_resume_and_complete_report(self):
        import time
        from run import execute_jobs
        from report import report
        from shutil import copytree
        r=self.root/'resume';r.mkdir();copytree(self.root/'instances',r/'instances');copytree(self.root/'models',r/'models')
        s=dict(self.s,screen_per_configuration=1,algorithm_seeds=[12345]);record={'protocol':s};sha=canonical_digest(record);record['sha256']=sha;write_json(r/'PLAN.json',record)
        job=(self.name,12345,'BASE',r/'new-base')
        result=execute_jobs(r,s,[job],sha,time.monotonic()+120,1);stamp=(job[3]/'RUN_COMPLETE.json').stat().st_mtime_ns
        self.assertEqual(execute_jobs(r,s,[job],sha,time.monotonic()+120,1),result)
        self.assertEqual((job[3]/'RUN_COMPLETE.json').stat().st_mtime_ns,stamp)
        rows=[read_json(self.root/m/'RUN_COMPLETE.json') for m in METHODS]
        q=report(rows,s,r);self.assertTrue(q['completed']);self.assertTrue((r/'CONVERGENCE.png').exists());self.assertEqual(q['searches'],7)
    def test_gate_does_not_use_label(self):
        p=np.zeros((3,8));p[:,2]=1.;self.assertEqual(decide(p,[True]*8,0,.02),2)
        self.assertEqual(decide(p,[True,True,False,True,True,True,True,True],0,.02),0)
        self.assertEqual(decide(p,[True]*8,0,.02,True),0)
if __name__=='__main__':unittest.main()
