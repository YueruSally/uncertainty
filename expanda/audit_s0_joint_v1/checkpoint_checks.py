"""Validate paired checkpoint summaries without running training/search."""
from pathlib import Path
import json
import tempfile
import unittest
from checkpoints import inspect


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))


class Checks(unittest.TestCase):
    def fixture(self,root,kind):
        old=kind=='s0';names=['a','b','c']
        plan={'settings':{'ppo':{'updates':4,'validation_checkpoints':[0,4]}},'validation_instances':names,
              'entries':[{'instance_id':n,'development_role':'validation','configuration':('S0' if n=='a' else 'S1')} for n in names]}
        write(root/'PLAN.json',plan)
        for u in (0,4):
            ck=root/'terminal/checkpoints'/f'update_{u:03d}';write(ck/'BUNDLE.json',{})
            if u:write(ck/'UPDATE.json',{'complete_episodes':4,'optimizer_steps':8,'kl_early_stop':False,'terminal_returns':[.2,.3,.4,.5],'approximate_kl':.001})
            for n,base,gain in zip(names,[.4,.5,.6],[.3,-.1,-.1]):
                p=root/'terminal/validation'/(f'u{u:03d}_{n}' if old else f'u{u:03d}/{n}')/'RUN_COMPLETE.json'
                write(p,{'completed':True,'instance':n,'stochastic':False,'final_hypervolume':base+(gain if u else 0),
                         'algorithm_seed':1,'seed':1,'scenario_digest':n})
        write(root/'terminal/SELECTED.json',{'update':4})

    def test_s0_pairing_and_completed_selection(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);self.fixture(root,'s0');r=inspect(root,'s0')
            self.assertAlmostEqual(r['validation'][1]['configuration_equal_mean_delta_vs_update0'],.1/3)
            self.assertEqual(r['validation'][1]['negative_instances'],2)
            write(root/'terminal/SELECTED.json',{'update':0})
            with self.assertRaisesRegex(ValueError,'selection'):inspect(root,'s0')

    def test_multiscale_config_equal_not_instance_equal(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);self.fixture(root,'multi');r=inspect(root,'multi')
            self.assertAlmostEqual(r['validation'][1]['configuration_equal_mean_delta_vs_update0'],.1)
            p=root/'terminal/validation/u004/a/RUN_COMPLETE.json';r=json.loads(p.read_text());r['seed']=2;write(p,r)
            with self.assertRaisesRegex(ValueError,'pairing'):inspect(root,'multi')

    def test_partial_panel_not_reported_as_complete_mean(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);self.fixture(root,'s0')
            (root/'terminal/SELECTED.json').unlink();(root/'terminal/validation/u004_c/RUN_COMPLETE.json').unlink()
            r=inspect(root,'s0')['validation'][1]
            self.assertEqual(r['completed_validation_instances'],2)
            self.assertIsNone(r['configuration_equal_mean_delta_vs_update0'])


if __name__=='__main__':unittest.main()
