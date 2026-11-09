"""Cold-process correctness fixtures; not research results."""
import copy
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import check
from check import ROOT, read_json, write_json, canonical_digest


def isolated(root, settings, name, arm, out, original=False):
    out.mkdir(parents=True)
    spec = out.parent / (out.name + '-job.json')
    write_json(spec, dict(root=str(root), settings=settings, name=name, arm=arm, out=str(out)))
    if original:
        code = "from worker import work;work(Path(x['root']),x['settings'],x['name'],12345,x['arm'],Path(x['out']),'fixture')"
    else:
        code = "from check import run_worker;run_worker(Path(x['root']),Path(x['out']),x['name'],12345,x['arm'],'fixture',x['settings'])"
    preamble = "import sys;from pathlib import Path;sys.path.insert(0,sys.argv[1]);sys.path.insert(0,sys.argv[2]);from common import read_json;x=read_json(Path(sys.argv[3]));"
    log = out.parent / (out.name + '-process.log')
    with log.open('w') as stream:
        proc = subprocess.run([sys.executable, '-c', preamble + code, str(ROOT), str(check.HERE), str(spec)], stdout=stream, stderr=subprocess.STDOUT)
    if proc.returncode:
        raise RuntimeError(log.read_text()[-8000:])
    return read_json(out / 'RUN_COMPLETE.json')


class SmallCheck(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(prefix='b20-smallcheck-')
        cls.root = Path(cls.tmp.name)
        cls.settings = read_json(ROOT / 'protocol.json')
        cls.settings.update(population=20, budget=400, progress_denominator=400, milestones=[100,400])
        info = read_json(ROOT.parent / 'formal_icm_s0_v1/assets/search_instances/prescreen-search-S0-001.json')
        info.pop('instance_digest'); info['development_role'] = 'screen'; info['instance_digest'] = canonical_digest(info)
        cls.name = info['instance_id']
        write_json(cls.root / 'instances' / (cls.name + '.json'), info)
        write_json(cls.root / 'models/HV_NORMALIZATION.json', read_json(ROOT.parent / 'formal_icm_s0_v1/assets/reward_normalization.json'))
        cls.old = {}
        for arm in ('BASE','I_RULE'):
            out = cls.root / 'screen' / cls.name / 'seed_12345' / arm
            cls.old[arm] = isolated(cls.root, cls.settings, cls.name, arm, out, True)
        cls.rows = [isolated(cls.root, cls.settings, cls.name, arm, cls.root / arm) for arm in check.ARMS]
        # Production population with only four post-initialization evaluations.
        settings = dict(cls.settings, population=250, budget=254, progress_denominator=20000, milestones=[250,254])
        cls.large = [isolated(cls.root, settings, cls.name, arm, cls.root / ('pop250-' + arm)) for arm in check.ARMS]
        print('Correctness fixtures:', cls.root, flush=True)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_fixed_prefix_size(self):
        self.assertEqual(check.fixed_count(250),162)
        self.assertEqual(check.fixed_count(20),12)

    def test_original_search_replay(self):
        a = self.rows[1]; old = self.old['I_RULE']
        self.assertEqual(a['evaluation_order_digest'],old['evaluation_order_digest'])
        self.assertEqual(a['final_hypervolume'],old['final_hypervolume'])
        self.assertEqual(self.rows[0]['initial_evaluation_digest'],self.old['BASE']['initial_evaluation_digest'])

    def test_seed_preservation_and_budget(self):
        for rows,n in ((self.rows,12),(self.large,162)):
            b,a,k = rows
            self.assertEqual(k['initialization']['fingerprints'][:n],b['initialization']['fingerprints'][:n])
            self.assertEqual(len({r['scenario_digest'] for r in rows}),1)
            for r in rows:
                self.assertEqual(r['hidden_candidate_objective_evaluations'],0)
                self.assertEqual(r['new_training_steps'],0)
            self.assertEqual(b['real_objective_evaluation_count'],b['initialization']['population'])
        for r in self.rows[1:]:
            self.assertIn(r['real_objective_evaluation_count'],(399,400))
        for r in self.large[1:]:
            self.assertIn(r['real_objective_evaluation_count'],(253,254))
        # The suspected collapse is observed, not assumed in the report.
        print('Pop250 initialization diversity:', [(r['arm'],r['initialization']['unique_individuals'],r['initialization']['largest_duplicate_count']) for r in self.large],flush=True)

    def test_saved_gate_selection(self):
        folder = self.root / 'gate-fixture'
        disabled = dict(epoch=10, **{'lambda':0}, threshold_scaled=0, disabled=True, mean_gain=0, switch_fraction=0, per_configuration={'S0':0})
        enabled = dict(disabled, disabled=False, mean_gain=-.01, switch_fraction=.4)
        report = dict(choices=[enabled,disabled],selected=disabled)
        bundle = dict(gate=disabled,training_states=10,validation_states=5)
        write_json(folder/'models/C/VALIDATION.json',report);write_json(folder/'models/C/BUNDLE.json',bundle)
        audit = check.crossover_audit(folder)
        self.assertTrue(audit['selection_reproduced_exactly'])
        self.assertEqual(audit['enabled_choices_with_positive_validation_gain'],0)
        # A disabled selection despite an available positive choice must fail.
        report['choices'][0]['mean_gain']=.1
        write_json(folder/'models/C/VALIDATION.json',report)
        with self.assertRaises(ValueError):check.crossover_audit(folder)

    def test_summary_and_mismatch_rejection(self):
        out = self.root/'summary';out.mkdir()
        write_json(out/'C_GATE_AUDIT.json',{'fixture':True})
        plan = {'cases':[dict(configuration='S0',instance=self.name,seed=12345)]}
        result = check.summarize(self.root,out,plan,self.rows)
        self.assertTrue(result['original_I_RULE_replayed_exactly'])
        self.assertTrue(result['base_fixed_slots_preserved_exactly'])
        altered = copy.deepcopy(self.rows);altered[1]['evaluation_order_digest']='wrong'
        with self.assertRaises(ValueError):check.summarize(self.root,out,plan,altered)

    def test_plan_resume_and_frozen_input_rejection(self):
        write_json(self.root/'PLAN.json',{'protocol':dict(self.settings,population=250,budget=20000),
                   'sha256':'fixture','files':{str((ROOT/'common.py').relative_to(ROOT.parent)):check.digest(ROOT/'common.py')}})
        write_json(self.root/'instances/catalog.json',{'instances':[dict(instance_id=self.name,configuration='S0',development_role='screen')]})
        write_json(self.root/'models/C/BUNDLE.json',{'fixture':True})
        write_json(self.root/'models/C/VALIDATION.json',{'fixture':True})
        norm = self.root/'models/HV_NORMALIZATION.json'
        write_json(self.root/'models/FROZEN.json',{'plan_sha256':'fixture','files':{'models/HV_NORMALIZATION.json':check.digest(norm)}})
        out = self.root/'frozen-plan';out.mkdir()
        with patch.object(check,'CONFIGURATIONS',('S0',)),patch.object(check,'SEEDS',(12345,)):
            first = check.freeze_plan(self.root,out)
            self.assertEqual(first,check.freeze_plan(self.root,out))
            saved = norm.read_bytes()
            try:
                norm.write_text('{}')
                with self.assertRaises(ValueError):check.freeze_plan(self.root,out)
            finally:norm.write_bytes(saved)


if __name__=='__main__':unittest.main()
