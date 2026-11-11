"""Correctness fixtures only: real S11 paths, no claimed experiment performance."""
from bootstrap import *
import contextlib,copy,io,random,subprocess,tempfile,unittest
from types import SimpleNamespace
from collections import namedtuple
from unittest.mock import patch
import numpy as np
import torch
import baseline_uncertainty as base
from repair_controller import coordinate_capacity
from fit import informative_weighted_loss,signal_check


def isolated(root,s,name,seed,arm,out,sha,old=False):
    out.mkdir(parents=True,exist_ok=True)
    job=out.parent/(out.name+'-job.json')
    write_json(job,dict(root=str(root),settings=s,name=name,seed=seed,arm=arm,out=str(out),sha=sha))
    code="""from pathlib import Path
import sys
sys.path.insert(0,sys.argv[1])
from bootstrap import *
import worker
if sys.argv[3]!='old':
    from repair_controller import RepairController
    worker.ScreenController=RepairController
x=read_json(Path(sys.argv[2]));s=x['settings']
if x['arm'].startswith('COLLECT_') and x['arm'][-1] in s.get('label_caps',{}):s=dict(s,extra_label_budget_per_component_instance=s['label_caps'][x['arm'][-1]])
worker.work(Path(x['root']),s,x['name'],x['seed'],x['arm'],Path(x['out']),x['sha'])
"""
    log=out.parent/(out.name+'-fixture.log')
    with log.open('w') as f:r=subprocess.run([sys.executable,'-c',code,str(NEW),str(job),'old' if old else 'new'],stdout=f,stderr=subprocess.STDOUT,timeout=300)
    if r.returncode:raise RuntimeError(log.read_text()[-8000:])
    return read_json(out/'RUN_COMPLETE.json')


class RepairChecks(unittest.TestCase):
    def test_nominal_repair_reduces_conflict_without_evaluation_or_global_rng(self):
        key=('a','b',1);batch=SimpleNamespace(origin='a',destination='b',batch_id=1)
        PathFixture=namedtuple('PathFixture','nodes modes arcs')
        p=PathFixture(nodes=('a','b'),modes=('road',),arcs=())
        q=PathFixture(nodes=('a','c','b'),modes=('road','road'),arcs=())
        resource=('arc',('a','b','road'),0)
        options={key:[SimpleNamespace(path=p,resources={resource:10.}),SimpleNamespace(path=q,resources={})]}
        ind=base.Individual();ind.od_allocations[key]=[base.PathAllocation(p,.6),base.PathAllocation(p,.4)]
        before=random.getstate();before_np=np.random.get_state()
        with patch.object(base,'_evaluate_individual_impl',side_effect=AssertionError('objective not allowed')), \
             patch.object(base,'simulate_path_over_scenarios',side_effect=AssertionError('scenario not allowed')):
            child,score=coordinate_capacity(ind,[batch],options,{('a','b','road'):4.},random.Random(1),2,8)
        self.assertEqual(score,0.)
        self.assertAlmostEqual(sum(a.share for a in child.od_allocations[key]),1.)
        self.assertEqual(ind.od_allocations[key][0].path,p)
        self.assertEqual(random.getstate(),before)
        self.assertTrue(all(np.array_equal(a,b) if isinstance(a,np.ndarray) else a==b for a,b in zip(np.random.get_state(),before_np)))

    def test_weighted_loss_keeps_negative_and_tied_examples(self):
        target=torch.zeros(3,8);target[0,1]=1.;target[1,1]=-1.
        pred=torch.ones(3,8,requires_grad=True);mask=torch.ones(3,8,dtype=torch.bool)
        value=informative_weighted_loss(pred,target,mask);value.backward()
        self.assertGreater(float(pred.grad[1,1]),0.)
        self.assertGreater(float(pred.grad[2,1]),0.)
        self.assertTrue(torch.isfinite(pred.grad).all())

    def test_zero_training_signal_stops_even_with_positive_validation(self):
        with tempfile.TemporaryDirectory(prefix='s11-signal-fixture-') as tmp:
            root=Path(tmp);catalog={'instances':[]}
            for name,role in [('t','training'),('v','validation')]:
                catalog['instances'].append(dict(instance_id=name,development_role=role))
                row={'divisor':[1.,1.,1.],'population_raw':[[1.,1.,1.]],'legal_mask':[True]*8,
                     'labels':[[{'feasible':True,'objectives':[1.,1.,1.]}] for _ in range(8)]}
                if role=='validation':row['labels'][1][0]['objectives']=[.5,.5,.5]
                write_json(root/'collection'/name/'COLLECT_I/DATA.json',dict(
                    instance=name,role=role,divisor=[1.,1.,1.],feasible_raw=[[1.,1.,1.]],rows=[row]))
            with contextlib.redirect_stdout(io.StringIO()):
                passed=signal_check(root,{'i_min_positive_training_instances':1},catalog)
            self.assertFalse(passed)
            report=read_json(root/'I_SIGNAL_CHECK.json')
            self.assertEqual(report['training_positive_instances'],0)
            self.assertEqual(report['records'][1]['positive_opportunity_states'],1)

    def test_declared_production_scope(self):
        s=read_json(NEW/'protocol.json');n=s['train_per_configuration']+s['validation_per_configuration']
        searches=n+s['screen_per_configuration']*len(s['algorithm_seeds'])*7
        self.assertEqual((searches,n),(20,6))
        self.assertEqual((s['population'],s['budget'],s['progress_denominator']),(250,50000,50000))
        self.assertEqual(n*sum(s['label_caps'].values()),6912)
        self.assertFalse(s['ppo']);self.assertFalse(s['automatic_expansion'])


class PipelineChecks(unittest.TestCase):
    def test_real_s11_pipeline_source_frozen_collection_parity_and_resume(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory(prefix='s11-repair-fixture-') as tmp:
            root=Path(tmp);source=root/'old';source.mkdir()
            s=read_json(NEW/'protocol.json')
            s.update(population=20,budget=400,states={'I':6,'C':3,'M':6},label_caps={'I':48,'M':48},
                     milestones=[100,400],epochs=1,checkpoint_epochs=[1],train_per_configuration=1,
                     validation_per_configuration=1,screen_per_configuration=1,i_min_positive_training_instances=1,m_start_generation=0)
            sample=read_json(NEW.parent/'formal_icm_s0_v1/assets/search_instances/prescreen-search-S0-001.json')
            cat=dict(source_data_sha256=sample['source_data_sha256'],train_od_pool=sorted({(b['origin'],b['destination']) for b in sample['batches']}),
                     test_od_pool=[['unseen-a','unseen-b']],instances=[])
            cat['catalog_digest']=canonical_digest(cat);write_json(root/'source_catalog.json',cat)
            s['source_catalog_digest']=cat['catalog_digest']
            old=dict(s,instance_master_seed=124000000,ccp_seed_base=126000000,collection_seed_base=127000000)
            from instances import prepare
            with contextlib.redirect_stdout(io.StringIO()):oldcat=prepare(source,old,root/'source_catalog.json')
            from common import sources
            plan=dict(protocol=old,files={str(p.relative_to(NEW.parent)):digest(p) for p in sources()})
            plan['sha256']=canonical_digest(plan);write_json(source/'PLAN.json',plan)
            write_json(source/'RULE_SELECTION.json',dict(plan_sha256=plan['sha256'],profiles=dict.fromkeys('ICM','GUARDED')))
            paths={}
            for role in ('training','validation'):
                row=next(r for r in oldcat['instances'] if r['development_role']==role)
                path=source/'collection'/row['instance_id']/'COLLECT_C';paths[role]=path
                isolated(source,old,row['instance_id'],row['algorithm_seed'],'COLLECT_C',path,plan['sha256'],old=True)
            norm=learning.fit_normalization([paths['training']])
            tr=read_json(paths['training']/'DATA.json')['rows'];va=read_json(paths['validation']/'DATA.json')['rows']
            with contextlib.redirect_stdout(io.StringIO()):learning.train_component('C',tr,va,norm,old,source/'models/C')
            write_json(source/'models/FROZEN.json',dict(plan_sha256=plan['sha256'],files={str(p.relative_to(source)):digest(p) for p in source.rglob('*') if p.is_file()}))
            before={str(p):digest(p) for p in source.rglob('*') if p.is_file()}
            from pipeline import execute
            log=root/'pipeline-fixture.log'
            try:
                with log.open('w') as f,contextlib.redirect_stdout(f):execute(root/'new',s,source,root/'source_catalog.json',2)
                report=read_json(root/'new/QUICK_RESULT.json')
            except Exception:
                print(log.read_text()[-10000:]);raise
            self.assertTrue(report['completed']);self.assertFalse(report['C_retrained'])
            self.assertEqual(report['searches'],7)
            self.assertEqual(before,{str(p):digest(p) for p in source.rglob('*') if p.is_file()})
            c_bundle=read_json(source/'models/C/BUNDLE.json')
            for name in ['BUNDLE.json']+c_bundle['checkpoints']:
                self.assertEqual(digest(source/'models/C'/name),digest(root/'new/models/C'/name))
            newcat=read_json(root/'new/instances/catalog.json')
            row=next(r for r in newcat['instances'] if r['development_role']=='training')
            collection=root/'new/collection'/row['instance_id']/'COLLECT_I'
            recorded=read_json(collection/'RUN_COMPLETE.json')
            sha=read_json(root/'new/PLAN.json')['sha256']
            # Separate fixture with the same instance/seed, but no side labels.
            # The experiment tree and its frozen instance roles are untouched.
            parity=root/'parity';info=read_json(root/'new/instances'/f'{row["instance_id"]}.json')
            info['development_role']='screen';info.pop('instance_digest')
            info['instance_digest']=canonical_digest(info)
            write_json(parity/'instances'/f'{row["instance_id"]}.json',info)
            write_json(parity/'RULE_SELECTION.json',read_json(root/'new/RULE_SELECTION.json'))
            write_json(parity/'models/HV_NORMALIZATION.json',read_json(root/'new/models/HV_NORMALIZATION.json'))
            direct=isolated(parity,s,row['instance_id'],row['algorithm_seed'],'I_RULE',root/'repeat',sha)
            self.assertEqual(recorded['initial_fingerprints'],direct['initial_fingerprints'])
            source_data=read_json(collection/'DATA.json')
            self.assertGreater(sum(x['feasible'] for r in source_data['rows'] for pair in r['labels'] for x in pair),0)
            rows=read_json(root/'new/PAIRED_RUNS.json')
            by={r['arm']:r for r in rows}
            from controller import fixed_seed_count
            fixed=fixed_seed_count(20)
            self.assertEqual(by['BASE']['initial_fingerprints'][:fixed],by['I_RULE']['initial_fingerprints'][:fixed])
            for r in rows:
                self.assertIn(r['real_objective_evaluation_count'],(399,400))
                self.assertEqual(r['hidden_candidate_objective_evaluations'],0)
                self.assertEqual(r['observed_objective_evaluations'],r['real_objective_evaluation_count'])
            frozen=digest(root/'new/models/FROZEN.json')
            with log.open('w') as f,contextlib.redirect_stdout(f):execute(root/'new',s,source,root/'source_catalog.json',2)
            self.assertEqual(frozen,digest(root/'new/models/FROZEN.json'))
            self.assertNotIn('[S11-TARGET] start ',log.read_text())


if __name__=='__main__':unittest.main()
