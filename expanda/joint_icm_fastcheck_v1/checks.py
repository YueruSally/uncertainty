"""Actual bounded searches plus isolation, frozen-model, pairing and resume checks."""
from bridge_fast import *
import unittest
from unittest.mock import patch
import shutil
import platform


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


class Checks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture=module('full_fixture_for_fastcheck',FULL/'checks.py').Checks
        cls.fixture.setUpClass();cls.base=cls.fixture.root;cls.source=cls.fixture.source
        cls.runner=module('fastcheck_runner_test',FAST/'run.py');cls.old=cls.base/'full-running';cls.old.mkdir()
        plan=copy.deepcopy(cls.fixture.spec);plan['files']={}
        plan['environment']={'python':platform.python_version(),'torch':str(torch.__version__),'numpy':str(np.__version__)}
        plan.pop('sha256');plan['sha256']=canonical_digest(plan);write_json(cls.old/'PLAN.json',plan)
        write_json(cls.old/'STATUS.json',{'completed':False,'stage':'optimizer/u002'})
        shutil.copytree(cls.base/'independent_model',cls.old/'independent_model')
        from policy import snapshot
        snapshot(TerminalPolicy(cls.source/'shared_model'),cls.old/'terminal/checkpoints/update_000',metrics={'update':0})
        snapshot(TerminalPolicy(cls.source/'shared_model'),cls.old/'terminal/checkpoints/update_002',metrics={
            'update':2,'complete_episodes':4,'optimizer_steps':1,'kl_early_stop':False,'terminal_returns':[.5]*4,'last_minibatch_kl':0.})
        cls.protocol=read_json(FAST/'protocol.json');cls.protocol.update(configurations=['S0'],population=20,budget=400)
        cls.root=cls.base/'fast-check';cls.root.mkdir()
        cls.before={str(p):digest(p) for p in cls.old.rglob('*') if p.is_file()}
        b=read_json(cls.source/'shared_model/BUNDLE.json')
        with patch.object(cls.runner,'verify_shared',return_value=(b,[cls.fixture.train,cls.fixture.valid])):
            cls.result=cls.runner.execute(cls.root,cls.source,cls.old,cls.protocol,2)
        cls.plan=read_json(cls.root/'PLAN.json')

    @classmethod
    def tearDownClass(cls):cls.fixture.tearDownClass()

    def test_actual_four_arm_search_no_updates_and_source_unchanged(self):
        self.assertEqual(self.result['completed_searches'],4)
        self.assertEqual(self.result['new_ppo_updates'],0)
        self.assertEqual(self.plan['settings']['progress_denominator'],50000)
        self.assertEqual(self.before,{str(p):digest(p) for p in self.old.rglob('*') if p.is_file()})
        self.assertFalse((self.root/'terminal').exists())
        for p in (self.root/'runs').rglob('RUN_COMPLETE.json'):
            r=read_json(p);self.assertEqual(r['evaluation_budget'],400);self.assertIn(r['real_objective_evaluation_count'],(399,400))
            self.assertEqual(r['hidden_candidate_objective_evaluations'],0);self.assertFalse(r['stochastic'])

    def test_repeat_never_restarts_completed_searches(self):
        before={str(p):p.stat().st_mtime_ns for p in (self.root/'runs').rglob('RUN_COMPLETE.json')}
        b=read_json(self.source/'shared_model/BUNDLE.json')
        with patch.object(self.runner,'verify_shared',return_value=(b,[self.fixture.train,self.fixture.valid])),patch.object(full.subprocess,'Popen',side_effect=AssertionError('No duplicate search')):
            self.runner.execute(self.root,self.source,self.old,self.protocol,2)
        self.assertEqual(before,{str(p):p.stat().st_mtime_ns for p in (self.root/'runs').rglob('RUN_COMPLETE.json')})

    def test_task_scope_seed_model_and_budget_rejected(self):
        entry=self.plan['panel'][0];task,path=self.runner.make_task(self.root,self.plan,entry,'ICM_TERMINAL_PPO')
        self.runner.validate_worker(self.root,self.plan,task,path)
        for change in ({'phase':'screen'},{'sample':True},{'instance':'training-fixture'},{'seed':task['seed']+1},{'bundle_sha256':'wrong'}):
            with self.subTest(change=change),self.assertRaises(ValueError):self.runner.validate_worker(self.root,self.plan,dict(task,**change),path)
        from reporting_fast import report
        rows=[read_json(p) for p in (self.root/'runs').rglob('RUN_COMPLETE.json')]
        bad=copy.deepcopy(rows);bad[0]['scenario_digest']='changed'
        with self.assertRaises(ValueError):report(bad,self.plan,self.root/'bad-report',self.protocol['methods'])
        with self.assertRaises(ValueError):report(rows[:-1],self.plan,self.root/'bad-report',self.protocol['methods'])
        bad=copy.deepcopy(rows);bad[0]['evaluation_budget']=50000
        with self.assertRaises(ValueError):report(bad,self.plan,self.root/'bad-report',self.protocol['methods'])

    def test_missing_model_preserves_full_process_and_partial_results(self):
        plan=copy.deepcopy(self.plan);plan['fast']['ppo_update']=7
        before=read_json(self.old/'STATUS.json');write_json(self.old/'STATUS.json',{'completed':False,'paused':True})
        try:
            with patch.object(self.runner.os,'kill',side_effect=AssertionError('Do not signal full job')),self.assertRaises(full.Paused):
                self.runner.binding(self.root,plan,'ICM_TERMINAL_PPO')
        finally:write_json(self.old/'STATUS.json',before)
        self.assertTrue((self.root/'results/00_当前小试对比.csv').exists())


if __name__=='__main__':unittest.main()
