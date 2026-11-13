"""Scheduling/report provenance checks; no transport searches or model training."""
from pathlib import Path
import copy
import csv
import importlib.util
import json
import signal
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import drain
import resume


class DrainChecks(unittest.TestCase):
    def simulation(self, fail=False, timeout=False, externally_stopped=False):
        clock=[0.];signals=[];states=[]
        parent={'pid':100,'ppid':1,'state':'T' if externally_stopped else 'S','start':'start'}
        child={'pid':101,'ppid':100,'state':'R','start':'child'}
        def send(sig):
            signals.append((100,sig))
            if sig==signal.SIGSTOP:parent['state']='T'
            if sig==signal.SIGCONT:parent['state']='S'
        def inspect(pid):
            return dict(parent) if pid==100 else dict(child,state='Z' if clock[0]>=4 else 'R')
        def write(r):
            states.append(r)
            if fail:raise RuntimeError('simulated I/O failure')
        def sleep(t):clock[0]+=t
        def execute():
            drain.finish_workers(SimpleNamespace(send=send),parent,lambda:[child],write,
                1 if timeout else 10,inspect=inspect,now=lambda:clock[0],sleep=sleep)
        return execute,signals,states,clock

    def test_workers_finish_without_any_worker_signal(self):
        execute,signals,states,clock=self.simulation();execute()
        self.assertGreaterEqual(clock[0],4)
        self.assertEqual(signals,[(100,signal.SIGSTOP),(100,signal.SIGTERM),(100,signal.SIGCONT)])
        self.assertEqual(states[-1]['stage'],'old_supervisor_retiring')

    def test_handoff_failure_resumes_old_dispatcher(self):
        for kw in ({'fail':True},{'timeout':True}):
            execute,signals,states,clock=self.simulation(**kw)
            with self.assertRaises((RuntimeError,TimeoutError)):execute()
            self.assertEqual(signals,[(100,signal.SIGSTOP),(100,signal.SIGCONT)])

    def test_external_stop_not_taken_over(self):
        execute,signals,states,clock=self.simulation(externally_stopped=True)
        with self.assertRaises(RuntimeError):execute()
        self.assertFalse(signals)

    def test_exact_experiment_process_matching(self):
        root=Path('/tmp/my_experiment');script=Path('/repo/run.py')
        r={'args':['python3',str(script),'--out',str(root)]}
        self.assertTrue(drain.matches(r,script,root,False))
        self.assertFalse(drain.matches(r,script,root,True))
        self.assertFalse(drain.matches(r,script,Path('/tmp/other'),False))
        self.assertFalse(drain.matches({'args':['python3','/other/run.py','--out',str(root)]},script,root,False))


class ReportChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runner,cls.reporting=resume.load_runtime()

    def fixture(self,root):
        plan={'screen':[{'instance_id':'a'}], 'settings':{'screen_seeds':[1,2,3],
            'configurations':['S0'],'budget':400,'milestones':[400]}}
        found=[]
        for seed in (1,2,3):
            for j,arm in enumerate(resume.UNIQUE):
                row={'instance':'a','configuration':'S0','seed':seed,'arm':arm,'phase':'screen',
                     'stochastic':False,'hidden_candidate_objective_evaluations':0,'ppo_updates':0,
                     'evaluation_budget':400,'real_objective_evaluation_count':400,
                     'scenario_digest':'scenario','instance_digest':'instance','final_hypervolume':.5+j/100,
                     'final_front_raw':[[10-j*.1,20-seed*.1,30-j*.1]],
                     'evaluation_order_digest':str(j),'optimisation_seconds':4.+j,'peak_memory_mib':10,
                     'selection':{c:{'calls':10,'fallbacks':5,'switches':5,'unsupported':0} for c in 'ICM'},
                     'milestone_hv':[{'evaluation':400,'hv':.5+j/100}]}
                p=root/'screen'/'a'/f'seed_{seed}'/arm/'RUN_COMPLETE.json'
                self.runner.write_json(p,row);found.append((row,p))
            if seed<3:
                row=copy.deepcopy(found[-1][0]);row.update(arm=resume.PPO,optimisation_seconds=8.)
                p=root/'screen'/'a'/f'seed_{seed}'/resume.PPO/'RUN_COMPLETE.json'
                self.runner.write_json(p,row);found.append((row,p))
        return plan,found

    def test_alias_not_fabricated_as_measured_run_or_runtime(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)/'source';out=Path(d)/'results';plan,found=self.fixture(root)
            prior={str(p):self.runner.digest(p) for r,p in found}
            q=resume.write_report(found,root,out,plan,{'update':0},self.runner,self.reporting)
            self.assertEqual(q['physically_completed_screen_searches'],14)
            self.assertEqual(q['distinct_policy_searches'],12)
            self.assertEqual(q['skipped_duplicate_searches'],1)
            self.assertEqual(q['screen_objective_evaluations'],14*400)
            self.assertEqual(q['aliases'][resume.PPO]['measured_reproduction_mean_seconds'],8.)
            with (out/'00_联合训练对比.csv').open(encoding='utf-8-sig') as f:table=list(csv.DictReader(f))
            self.assertEqual(len(table),5)
            self.assertEqual(table[-1]['平均搜索时间(秒)'],'')
            self.assertEqual(table[-1]['相对Base提升(%)'],table[-2]['相对Base提升(%)'])
            with (out/'PER_RUN.csv').open(encoding='utf-8-sig') as f:per=list(csv.DictReader(f))
            self.assertEqual(len(per),12)
            self.assertFalse((root/'screen/a/seed_3'/resume.PPO/'RUN_COMPLETE.json').exists())
            self.runner.verify(prior)

    def test_identity_failure_never_silently_reused(self):
        with tempfile.TemporaryDirectory() as d:
            plan,found=self.fixture(Path(d))
            altered=copy.deepcopy(found)
            next(r for r,p in altered if r['arm']==resume.PPO)['final_hypervolume']+=.01
            with self.assertRaises(ValueError):resume.check_reproductions(altered)
            with self.assertRaises(ValueError):resume.check_reproductions([(r,p) for r,p in found if r['seed']==1])

    def test_resumption_only_schedules_unique_policies_and_preserves_completed_files(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)/'source';out=Path(d)/'continuation';plan,found=self.fixture(root)
            sup=Path(d)/'supervised'
            for folder in (sup/'shared_model',root/'independent_model',root/'terminal/checkpoints/update_000'):
                self.runner.write_json(folder/'BUNDLE.json',{'synthetic_test':True})
            selected={'update':0,'model':str(root/'terminal/checkpoints/update_000'),
                      'bundle_sha256':self.runner.digest(root/'terminal/checkpoints/update_000/BUNDLE.json')}
            plan.update(source=str(sup),normalization={},files={})
            plan['sha256']=self.runner.canonical_digest(plan)
            self.runner.write_json(root/'PLAN.json',plan)
            all_jobs=resume.jobs(root,plan,selected,self.runner)
            bykey={(r['instance'],r['seed'],r['arm']):(r,p) for r,p in found}
            skipped=None
            for task,folder in all_jobs:
                key=task['instance'],task['seed'],task['arm']
                if key not in bykey:continue
                r,p=bykey[key]
                r.update(completed=True,plan_sha256=plan['sha256'],task_sha256=self.runner.canonical_digest(task))
                self.runner.write_json(p,r)
                if key==('a',3,'BASE'):
                    skipped=(r,p);p.unlink()
            before={str(p):self.runner.digest(p) for r,p in found if p.exists()}
            def schedule(source,frozen,wanted,workers,stage):
                self.assertEqual(len(wanted),12)
                self.assertTrue(all(t['arm'] in resume.UNIQUE for t,p in wanted))
                missing=[(t,p) for t,p in wanted if not (p/'RUN_COMPLETE.json').exists()]
                self.assertEqual([(t['seed'],t['arm']) for t,p in missing],[(3,'BASE')])
                self.runner.write_json(skipped[1],skipped[0])
            a=SimpleNamespace(source=root,out=out,workers=2,take_over=False,recover=False)
            old_signals={s:signal.getsignal(s) for s in (signal.SIGTERM,signal.SIGINT)}
            try:
                with patch.object(resume,'identity',return_value=(plan,selected)),\
                     patch.object(resume,'load_runtime',return_value=(self.runner,self.reporting)),\
                     patch.object(self.runner,'execute_jobs',side_effect=schedule):
                    resume.execute(a)
            finally:
                for s,h in old_signals.items():signal.signal(s,h)
            self.runner.verify(before)
            q=self.runner.read_json(out/'results/QUICK_RESULT.json')
            self.assertEqual(q['new_screen_searches_this_invocation'],1)
            self.assertEqual(q['new_objective_evaluations_this_invocation'],400)
            self.assertTrue(self.runner.read_json(out/'STATUS.json')['completed'])
            self.assertFalse(self.runner.read_json(root/'STATUS.json')['completed'])


if __name__=='__main__':unittest.main()
