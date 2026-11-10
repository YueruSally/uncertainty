"""Real 50-batch small-budget checks; fixtures are never experiment results."""
from common import *
import contextlib,copy,tempfile,unittest,subprocess
import numpy as np
import torch
from learning import fit_normalization,train_component,FrozenPolicy,labels,context,scaler,loss
from controller import fixed_seed_count,guarded_reference,mutation_proxy_features
from report import comparison,union_hv,select_rules

def isolated(root,s,name,seed,arm,out):
    out.mkdir(parents=True,exist_ok=True)
    spec=out.parent/(out.name+'-job.json')
    write_json(spec,dict(root=str(root),s=s,name=name,seed=seed,arm=arm,out=str(out)))
    command="from pathlib import Path;import sys;sys.path.insert(0,sys.argv[1]);from worker import work;from common import read_json;x=read_json(Path(sys.argv[2]));work(Path(x['root']),x['s'],x['name'],x['seed'],x['arm'],Path(x['out']),'test')"
    log=out.parent/(out.name+'-test.log')
    with log.open('w') as f:r=subprocess.run([sys.executable,'-c',command,str(ROOT),str(spec)],stdout=f,stderr=subprocess.STDOUT,timeout=240)
    if r.returncode:raise RuntimeError(log.read_text()[-7000:])
    return read_json(out/'RUN_COMPLETE.json')

class S11Checks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1);cls.tmp=tempfile.TemporaryDirectory(prefix='s11-target-check-');cls.root=Path(cls.tmp.name)
        cls.s=read_json(ROOT/'protocol.json')
        cls.s.update(population=20,budget=400,states={'I':6,'C':3,'M':6},milestones=[100,400],epochs=1,checkpoint_epochs=[1],train_per_configuration=1,validation_per_configuration=1,screen_per_configuration=1,calibration_per_configuration=1)
        sample=read_json(ROOT.parent/'formal_icm_s0_v1/assets/search_instances/prescreen-search-S0-001.json')
        src=dict(source_data_sha256=sample['source_data_sha256'],train_od_pool=sorted({(b['origin'],b['destination']) for b in sample['batches']}),test_od_pool=[['unused-a','unused-b']],instances=[])
        src['catalog_digest']=canonical_digest(src);cls.s['source_catalog_digest']=src['catalog_digest']
        write_json(cls.root/'source.json',src)
        from instances import prepare
        with (cls.root/'prepare.log').open('w') as f,contextlib.redirect_stdout(f):cls.cat=prepare(cls.root,cls.s,cls.root/'source.json')
        cls.names={r['development_role']:r['instance_id'] for r in cls.cat['instances']}
        write_json(cls.root/'RULE_SELECTION.json',{'plan_sha256':'test','profiles':dict.fromkeys('ICM','GUARDED')})
        cls.norm=read_json(ROOT.parent/'formal_icm_s0_v1/assets/reward_normalization.json')
        write_json(cls.root/'models/HV_NORMALIZATION.json',cls.norm)
        cls.runs={};cls.folders={}
        for c in 'ICM':
            name=cls.names['training'];folder=cls.root/('COLLECT_'+c)
            cls.runs['COLLECT_'+c]=isolated(cls.root,cls.s,name,12345,'COLLECT_'+c,folder);cls.folders[c]=folder
        # Exact same instance only inside correctness fixtures; no resulting model is published.
        name=cls.names['training'];path=cls.root/'instances'/f'{name}.json';x=read_json(path);x.pop('instance_digest');x['development_role']='screen';x['instance_digest']=canonical_digest(x);write_json(path,x)
        for arm in ['BASE','I_RULE','C_RULE','M_RULE']:
            cls.runs[arm]=isolated(cls.root,cls.s,name,12345,arm,cls.root/arm)
        print('S11 correctness artifacts:',cls.root,flush=True)

    @classmethod
    def tearDownClass(cls):cls.tmp.cleanup()

    def test_real_fifty_batch_disjoint_instance_roles(self):
        self.assertEqual({r['K'] for r in self.cat['instances']},{50})
        self.assertEqual(len({r['batch_signature'] for r in self.cat['instances']}),4)
        for r in self.cat['instances']:
            x=read_json(self.root/'instances'/f'{r["instance_id"]}.json')
            self.assertEqual(x['deadline_window_alpha'],2.0)
        self.assertEqual(self.s['progress_denominator'],50000)

    def test_audit_does_not_change_component_rule_trajectory(self):
        for c in 'CM':
            a,b=self.runs['COLLECT_'+c],self.runs[c+'_RULE']
            self.assertEqual(a['evaluation_order_digest'],b['evaluation_order_digest'])
            self.assertEqual(a['final_front_raw'],b['final_front_raw'])
        self.assertEqual(self.runs['COLLECT_I']['initial_evaluation_digest'],self.runs['I_RULE']['initial_evaluation_digest'])

    def test_fixed_initial_solutions_and_budget(self):
        n=fixed_seed_count(self.s['population'])
        self.assertEqual(self.runs['BASE']['initial_fingerprints'][:n],self.runs['I_RULE']['initial_fingerprints'][:n])
        self.assertEqual(fixed_seed_count(250),162)
        for arm in ('BASE','I_RULE','C_RULE','M_RULE'):
            r=self.runs[arm];self.assertIn(r['real_objective_evaluation_count'],(399,400));self.assertEqual(r['hidden_candidate_objective_evaluations'],0)
            self.assertEqual(r['observed_objective_evaluations'],r['real_objective_evaluation_count'])
        for c in 'ICM':
            r=self.runs['COLLECT_'+c];self.assertGreater(r['audit_states'][c],0)
            self.assertLessEqual(r['audit_evaluations'][c],384);self.assertTrue(all(r['audit_phase_counts'][c]))
            self.assertEqual(r['observed_objective_evaluations'],r['real_objective_evaluation_count']+sum(r['audit_evaluations'].values()))
            if c=='I':self.assertEqual(r['real_objective_evaluation_count'],20)
            data=read_json(self.folders[c]/'DATA.json')
            self.assertTrue(all(row['progress']==row['evaluation']/50000 for row in data['rows']))
            self.assertTrue(all(len(p)==(2 if c=='C' else 1) for row in data['rows'] for p in row['labels']))

    def test_train_norm_before_features_and_train_only_scales(self):
        norm=fit_normalization(list(self.folders.values()));self.assertFalse(norm['objective_clipping'])
        r=read_json(self.folders['I']/'DATA.json')['rows'][0]
        transformed=context(r,norm);self.assertEqual(transformed.shape,(15,));self.assertTrue(np.isfinite(transformed).all())
        y=labels(r,norm);self.assertEqual(y[0],0);self.assertTrue(np.isfinite(y).all())
        data=read_json(self.folders['I']/'DATA.json');data['role']='validation';write_json(self.root/'forbidden/DATA.json',data)
        with self.assertRaises(ValueError):fit_normalization([self.root/'forbidden'])
        negative=copy.deepcopy(r);negative['population_raw']=[];negative['divisor']=[1,1,1]
        negative['labels']=[[{'feasible':True,'objectives':[.2,.2,.2]}]]+[[{'feasible':True,'objectives':[.8,.8,.8]}]]*7
        self.assertLess(labels(negative,dict(minimum=[0]*3,maximum=[1]*3,reference=[1.1]*3))[1],0)

    def test_loss_zero_labels_calibrates_predictions(self):
        pred=torch.tensor([[0.]+[1.]*7],requires_grad=True);target=torch.zeros((1,8));mask=torch.ones((1,8),dtype=torch.bool)
        value=loss(pred,target,mask);self.assertGreater(float(value),0);value.backward();self.assertTrue((pred.grad[0,1:]>0).all())
        self.assertEqual(float(loss(torch.zeros((1,8)),target,mask)),0)

    def test_train_all_three_ensembles_and_frozen_execution(self):
        norm=fit_normalization(list(self.folders.values()));write_json(self.root/'models/HV_NORMALIZATION.json',norm)
        for c in 'ICM':
            rows=read_json(self.folders[c]/'DATA.json')['rows'];valid=copy.deepcopy(rows)
            for r in valid:r['instance']='validation-test-only'
            with (self.root/(c+'-train.log')).open('w') as f,contextlib.redirect_stdout(f):b=train_component(c,rows,valid,norm,self.s,self.root/'models'/c)
            self.assertEqual(len(b['checkpoints']),3);self.assertIn('smooth_l1',b['loss'])
            policy=FrozenPolicy(self.root/'models'/c)
            index=policy.select(rows[0]);self.assertTrue(rows[0]['legal_mask'][index])
            no_labels={k:v for k,v in rows[0].items() if k!='labels'};self.assertEqual(policy.select(no_labels),index)
            out=isolated(self.root,self.s,self.names['training'],12345,c+'_LEARN',self.root/(c+'_LEARN'))
            self.assertGreater(out['selection'][c]['calls'],0);self.assertEqual(out['hidden_candidate_objective_evaluations'],0)
        with self.assertRaises(ValueError):train_component('I',rows,rows,norm,self.s,self.root/'badtrain')

    def test_guard_uses_only_proxy_and_can_reject(self):
        b=dict(cost=100.,emission=100.,time=100.,missing=0.,capacity_excess=0.,lateness=0.)
        self.assertEqual(guarded_reference([b,dict(b,cost=101.)],1,0,.02),1)
        for key in ('cost','emission','time','missing','capacity_excess','lateness'):
            self.assertEqual(guarded_reference([b,dict(b,**{key:200.})],1,0,.02),0)

    def test_mutation_roles_distinguish_actual_base_from_balanced(self):
        keys=('cost','emission','time','capacity_excess','lateness','missing','scaled_cost','scaled_emission','scaled_time','scaled_capacity_excess','scaled_lateness','score')
        f=mutation_proxy_features([dict.fromkeys(keys,0.) for _ in range(8)])
        self.assertEqual(f[0][-4:],[0.,1.,0.,0.]);self.assertEqual(f[1][-4:],[1.,0.,0.,0.])
        self.assertEqual(f[2][-4:],[0.,0.,1.,0.])

    def test_report_bounds_pairing_and_no_forced_pass(self):
        rows=[]
        for i in range(2):
            for seed in range(2):
                for arm,p in [('BASE',.4),('I_RULE',.5),('I_LEARN',.6)]:
                    rows.append(dict(configuration='S11',instance=str(i),seed=seed,arm=arm,final_hypervolume=123.,final_front_raw=[[p,p,p]]))
        recomputed,bounds=union_hv(rows);self.assertEqual(bounds['0']['minimum'],[.4]*3)
        self.assertLess(comparison(recomputed,'I_LEARN','I_RULE',100)['percent']['high'],0)
        rows[0]['final_front_raw']=[]
        recomputed,_=union_hv(rows);self.assertIsNone(comparison(recomputed,'I_RULE','BASE',100)['percent']['mean'])

    def test_resume_reuses_and_refuses_modified_labels(self):
        from run import completed,execute_jobs
        folder=self.folders['C'];self.assertIsNotNone(completed(folder,'test'))
        import time
        job=(self.names['training'],12345,'COLLECT_C',folder);stamp=(folder/'RUN_COMPLETE.json').stat().st_mtime_ns
        execute_jobs(self.root,self.s,[job],'test',time.monotonic()+5,1)
        self.assertEqual((folder/'RUN_COMPLETE.json').stat().st_mtime_ns,stamp)
        with self.assertRaises(ValueError):completed(folder,'other-plan')

class PipelineChecks(unittest.TestCase):
    def test_whole_pipeline_and_resume_at_real_population_size(self):
        # Same real S11 generator and production population. Only the budget,
        # instances, epochs and sampling cap are reduced for implementation QA.
        with tempfile.TemporaryDirectory(prefix='s11-pipeline-') as tmp:
            root=Path(tmp);sample=read_json(ROOT.parent/'formal_icm_s0_v1/assets/search_instances/prescreen-search-S0-001.json')
            src=dict(source_data_sha256=sample['source_data_sha256'],train_od_pool=sorted({(b['origin'],b['destination']) for b in sample['batches']}),test_od_pool=[['unused-a','unused-b']],instances=[])
            src['catalog_digest']=canonical_digest(src);write_json(root/'source.json',src)
            s=read_json(ROOT/'protocol.json');s.update(budget=600,population=250,m_start_generation=0,
                states={'I':6,'C':3,'M':6},milestones=[400,600],epochs=1,checkpoint_epochs=[1],
                calibration_per_configuration=1,train_per_configuration=1,validation_per_configuration=1,screen_per_configuration=1,
                algorithm_seeds=[971001],source_catalog_digest=src['catalog_digest'])
            write_json(root/'fixture.json',s)
            code="""import sys
from pathlib import Path
sys.path.insert(0,sys.argv[1])
import run
actual=run.read_json
settings=actual(Path(sys.argv[2])/'fixture.json')
run.read_json=lambda path:settings if Path(path)==run.ROOT/'protocol.json' else actual(path)
sys.argv=[str(run.ROOT/'run.py'),'--stage','all','--out',str(Path(sys.argv[2])/'out'),'--workers','2','--source-catalog',str(Path(sys.argv[2])/'source.json')]
run.main()
"""
            for repeat in range(2):
                log=root/f'pipeline-{repeat}.log'
                with log.open('w') as f:r=subprocess.run([sys.executable,'-c',code,str(ROOT),str(root)],stdout=f,stderr=subprocess.STDOUT,timeout=480)
                self.assertEqual(r.returncode,0,log.read_text()[-8000:])
                report=read_json(root/'out/QUICK_RESULT.json');self.assertTrue(report['completed'])
                self.assertEqual(report['searches'],7);self.assertEqual(set(report['acceptance']),set('ICM'))
                self.assertFalse(report['final_test_opened']);self.assertEqual(report['new_ppo_updates'],0)
                if repeat==0:before=digest(root/'out/models/FROZEN.json')
                else:
                    self.assertEqual(digest(root/'out/models/FROZEN.json'),before)
                    self.assertNotIn('[S11-TARGET] start ',log.read_text())

if __name__=='__main__':unittest.main()
