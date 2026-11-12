"""Implementation checks only; tiny fixtures are not research outcomes."""
from bootstrap import *
import unittest
import tempfile
import copy
import subprocess
from unittest.mock import patch
import numpy as np
import torch
from shared import SharedTransformer,SharedPolicy,load_checkpoint
from training import fit,balanced_groups
from data import load_dataset,validate_rows


def isolated(root,settings,name,arm,out):
    job=out.parent/(out.name+'_job.json')
    write_json(job,{'root':str(root),'settings':settings,'name':name,'arm':arm,'out':str(out)})
    code="import sys;from pathlib import Path;sys.path.insert(0,sys.argv[1]);from bootstrap import read_json;from collection import work;x=read_json(Path(sys.argv[2]));work(Path(x['root']),x['settings'],x['name'],12345,x['arm'],Path(x['out']),'fixture')"
    log=out.parent/(out.name+'.log')
    with log.open('w') as f:
        result=subprocess.run([sys.executable,'-c',code,str(HERE),str(job)],stdout=f,stderr=subprocess.STDOUT)
    if result.returncode:raise RuntimeError(log.read_text()[-6000:])


class Checks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.temp=tempfile.TemporaryDirectory(prefix='multi-shared-check-');cls.root=Path(cls.temp.name)
        s=read_json(HERE/'protocol.json');s.update(population=20,budget=400,progress_denominator=50000,
            states={'I':6,'C':3,'M':6},milestones=[100,400],epochs=2,checkpoint_epochs=[1,2],
            configurations=['S0','S1'],training_seeds=[21,22,23],batch_size=4)
        cls.s=s
        x=read_json(EXPANDA/'formal_icm_s0_v1/assets/search_instances/prescreen-search-S0-001.json')
        x.pop('instance_digest');x['development_role']='training';x['instance_digest']=canonical_digest(x)
        cls.info=copy.deepcopy(x);cls.name=x['instance_id'];write_json(cls.root/'instances'/(cls.name+'.json'),x)
        cls.runs={}
        for arm in ('COLLECT_I','COLLECT_CM'):
            out=cls.root/arm;out.mkdir();isolated(cls.root,s,cls.name,arm,out)
            cls.runs[arm]=read_json(out/'RUN_COMPLETE.json')
        norm=read_json(EXPANDA/'formal_icm_s0_v1/assets/reward_normalization.json')
        write_json(cls.root/'models/HV_NORMALIZATION.json',norm)
        x.pop('instance_digest');x['development_role']='screen';x['instance_digest']=canonical_digest(x)
        write_json(cls.root/'instances'/(cls.name+'.json'),x)
        for arm in ('BASE','I_RULE'):
            out=cls.root/arm;out.mkdir();isolated(cls.root,s,cls.name,arm,out)
            cls.runs[arm]=read_json(out/'RUN_COMPLETE.json')
        train=[];valid=[]
        for arm in ('COLLECT_I','COLLECT_CM'):
            original=read_json(cls.root/arm/'DATA.json');train.append(cls.root/arm)
            duplicate=copy.deepcopy(original);duplicate['role']='validation';duplicate['instance']='validation-fixture'
            for r in duplicate['rows']:r['instance']='validation-fixture'
            path=cls.root/(arm+'_valid');write_json(path/'DATA.json',duplicate);valid.append(path)
        cls.rows,cls.norm=load_dataset(train+valid,50000)
        # Different configuration lengths exercise grouping without padding.
        for c in COMPONENTS:
            for role in ('training','validation'):
                extra=copy.deepcopy(cls.rows[c][role])
                for r in extra:
                    r['instance']+='-small';r['configuration']='S1'
                    r['candidate_batch_tokens']=[a[:10] for a in r['candidate_batch_tokens']]
                cls.rows[c][role]+=extra
        cls.coverage={'full_budget_ready':False,'test_fixture':True}
        cls.model=cls.root/'shared'
        fit(cls.rows,cls.norm,s,cls.model,{},cls.coverage)

    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()

    def test_base_trajectory_and_audit_isolation(self):
        a,b=self.runs['BASE'],self.runs['COLLECT_CM']
        self.assertEqual(a['evaluation_order_digest'],b['evaluation_order_digest'])
        self.assertEqual(a['final_hypervolume'],b['final_hypervolume'])
        self.assertEqual(a['hidden_candidate_objective_evaluations'],0)
        self.assertIn(b['real_objective_evaluation_count'],(399,400))
        for c in 'CM':self.assertTrue(all(b['audit_phase_counts'][c]))

    def test_fixed_seed_prefix_and_i_rule_match(self):
        from collection import fixed_seed_count
        a,b=self.runs['COLLECT_I'],self.runs['I_RULE']
        self.assertEqual(a['fixed_initial_slots_preserved'],fixed_seed_count(20))
        self.assertEqual(a['initial_evaluation_digest'],b['initial_evaluation_digest'])
        self.assertEqual(a['generations_completed'],0)
        rows=read_json(self.root/'COLLECT_I/DATA.json')['rows']
        self.assertTrue(all(r['initialization_progress']>=fixed_seed_count(20)/20 for r in rows))
        self.assertTrue(all(a['audit_phase_counts']['I']))

    def test_progress_and_train_only_scaling(self):
        for c in COMPONENTS:
            for r in self.rows[c]['training']:
                self.assertEqual(r['progress'],r['evaluation']/50000)
        self.assertFalse(self.norm['objective_clipping'])
        with self.assertRaises(ValueError):legacy_learning.fit_normalization([self.root/'COLLECT_I_valid'])
        changed=copy.deepcopy(self.rows)
        changed['I']['validation'][0]['candidate_features'][0][0]=1e15
        a=legacy_learning.scaler(self.rows['I']['training'],self.norm)
        b=legacy_learning.scaler(changed['I']['training'],self.norm)
        self.assertEqual(a,b)

    def test_shared_backbone_and_config_balance(self):
        b=read_json(self.model/'BUNDLE.json');model=SharedTransformer(b['dimensions'],16)
        groups=balanced_groups(self.rows,self.s['configurations'])
        self.assertEqual(len(groups),6)
        for c in COMPONENTS:
            model.zero_grad();r=self.rows[c]['training'][:1]
            t=legacy_learning.pack(r,b['scalers'][c],self.norm)
            pred=model(c,*t);pred[:,1:].sum().backward()
            self.assertIsNotNone(model.batch_encoder.layers[0].self_attn.in_proj_weight.grad)
            self.assertIsNotNone(model.candidate_encoder.layers[0].self_attn.in_proj_weight.grad)
            self.assertTrue(torch.equal(pred[:,0],torch.zeros_like(pred[:,0])))

    def test_mask_signed_label_and_reference(self):
        norm={'minimum':[0,0,0],'maximum':[1,1,1],'reference':[1.1]*3}
        row={'divisor':[1,1,1],'population_raw':[], 'labels':[
            [{'objectives':[.4,.4,.4],'feasible':True}],
            [{'objectives':[.9,.9,.9],'feasible':True}],
            [{'objectives':[.2,.6,.4],'feasible':True},{'objectives':[.6,.2,.4],'feasible':True}]]}
        y=legacy_learning.labels(row,norm);self.assertEqual(y[0],0);self.assertLess(y[1],0)
        pred=np.zeros((3,8));pred[:,1]=100
        self.assertEqual(legacy_learning.decide(pred,[True,False]+[True]*6,0,0),0)
        self.assertTrue(np.isfinite(legacy_learning.context(self.rows['I']['training'][0],self.norm)).all())

    def test_frozen_inference_and_resume(self):
        policy=SharedPolicy(self.model)
        for c in COMPONENTS:
            for r in self.rows[c]['validation']:
                chosen=policy.select(r);self.assertTrue(r['legal_mask'][chosen])
        stamp=digest(self.model/'BUNDLE.json')
        fit(self.rows,self.norm,self.s,self.model,{},self.coverage)
        self.assertEqual(stamp,digest(self.model/'BUNDLE.json'))
        with self.assertRaises(ValueError):fit(self.rows,self.norm,dict(self.s,epochs=3),self.model,{},self.coverage)

    def test_epoch_resume_matches_uninterrupted(self):
        import training
        target=self.root/'resume_model';original=training.save_checkpoint
        def stop(path,value):
            original(path,value)
            if path.name=='resume_21.pt' and value['epoch']==1:raise InterruptedError('fixture interruption')
        with patch.object(training,'save_checkpoint',stop):
            with self.assertRaises(InterruptedError):fit(self.rows,self.norm,self.s,target,{},self.coverage)
        fit(self.rows,self.norm,self.s,target,{},self.coverage)
        for seed in self.s['training_seeds']:
            a=load_checkpoint(self.model/f'seed_{seed}_epoch_2.pt')['model']
            b=load_checkpoint(target/f'seed_{seed}_epoch_2.pt')['model']
            for key in a:self.assertTrue(torch.equal(a[key],b[key]),key)

    def test_malformed_rows_fail(self):
        r=copy.deepcopy(self.rows['C']['training'][:1]);r[0]['labels'][0]=r[0]['labels'][0][:1]
        with self.assertRaises(ValueError):validate_rows(r,self.info,'C',400)

    def test_supervisor_worker_resume_and_immutable_runtime(self):
        # Exercise the actual command-line worker rather than only work().
        import importlib.util
        spec=importlib.util.spec_from_file_location('multi_pipeline',HERE/'run.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        root=self.root/'pipeline';root.mkdir()
        instance=root/'instances'/(self.name+'.json');write_json(instance,self.info)
        plan={'protocol':self.s,'files':{**runtime_files(),str(instance):digest(instance)},'instances':[self.name]}
        plan['sha256']=canonical_digest(plan);write_json(root/'PLAN.json',plan)
        out=root/'collection'/self.name/'COLLECT_I'
        # worker seeds must equal the manifest; no arbitrary seed substitution.
        jobs=[(self.name,self.info['algorithm_seed'],'COLLECT_I',out)]
        module.execute_jobs(root,self.s,jobs,plan['sha256'],1)
        stamp=(out/'RUN_COMPLETE.json').stat().st_mtime_ns
        module.execute_jobs(root,self.s,jobs,plan['sha256'],1)
        self.assertEqual(stamp,(out/'RUN_COMPLETE.json').stat().st_mtime_ns)
        changed=read_json(instance);changed['algorithm_seed']+=1;changed.pop('instance_digest')
        changed['instance_digest']=canonical_digest(changed);write_json(instance,changed)
        with self.assertRaises(ValueError):verify(plan['files'])


if __name__=='__main__':unittest.main()
