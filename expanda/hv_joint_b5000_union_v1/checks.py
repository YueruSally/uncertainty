"""Geometry, provenance, and reporting checks; no experiment execution."""
import contextlib
import copy
import io
import itertools
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import recompute as h


def fixture():
    plan = dict(sha256='frozen', settings=dict(budget=5000, population=250,
        milestones=list(h.BUDGETS), screen_seeds=[1, 2, 3]),
        screen=[dict(instance_id='one', configuration='S0', instance_digest='instance')])
    report = dict(completed=True, preview_only=False, ppo_alias=None)
    rows = []
    for a, arm in enumerate(h.ARMS[:-1]):
        for seed in [1, 2, 3]:
            snaps = [dict(evaluation=n, actual_evaluations=n, hv=1.2,
                front_raw=[[10+a+seed, 20+2*a+seed, 30+3*a+seed]], controller_seconds=n/100)
                for n in h.BUDGETS]
            snaps[-1]['front_raw'].append([0, 100, 50])
            rows.append(dict(completed=True, instance='one', configuration='S0', instance_digest='instance',
                scenario_digest='scenario', seed=seed, arm=arm, phase='screen', stochastic=False,
                plan_sha256='frozen', hidden_candidate_objective_evaluations=0, ppo_updates=0,
                evaluation_budget=5000, real_objective_evaluation_count=5000, optimisation_seconds=60,
                budget_fronts=snaps, final_front_raw=snaps[-1]['front_raw'], final_hypervolume=1.2,
                model_bundle_sha256=arm, selection={}, evaluation_order_digest='eval-order'))
    return rows, plan, report


class Checks(unittest.TestCase):
    def test_hv_against_inclusion_exclusion(self):
        rng = np.random.default_rng(2026)
        for count in range(1, 7):
            points = rng.random((count, 3)).tolist()
            expected = 0
            for size in range(1, count+1):
                for subset in itertools.combinations(points, size):
                    expected += (-1)**(size+1)*np.prod(np.asarray(h.REF)-np.max(subset, axis=0))
            self.assertAlmostEqual(h.hv3(points), expected, places=12)
        self.assertEqual(h.hv3([]), 0)
        self.assertAlmostEqual(h.hv3([[0, 0, 0]]), 1.331)
        self.assertAlmostEqual(h.hv3([[0, 0, 0], [1, 1, 1], [0, 0, 0]]), 1.331)

    def test_bounds_include_every_budget_and_dominated_union_points(self):
        rows, plan, report = fixture()
        rows, _ = h.validate_panel(rows, plan, report)
        per, bounds = h.renormalize(rows)
        self.assertEqual(bounds['one']['minimum'][0], 0)  # only the late node supplied this minimum
        self.assertEqual(bounds['one']['maximum'][1], 100)
        self.assertEqual(bounds['one']['maximum'][0], 22)  # retain points dominated across different runs
        a = next(r for r in per if r['arm'] == 'BASE' and r['seed'] == 1 and r['evaluation'] == 400)
        z = (np.array([11, 21, 31])-np.array(bounds['one']['minimum']))/np.array(bounds['one']['effective_width'])
        self.assertAlmostEqual(a['hv_union'], np.prod(np.array(h.REF)-z))
        self.assertTrue(all(0 <= r['hv_01'] <= 1 for r in per))

    def test_constant_and_empty_fronts(self):
        rows, plan, report = fixture()
        for r in rows:
            for s in r['budget_fronts']:
                s['front_raw'] = [[4, 4, 4]]
            r['final_front_raw'] = [[4, 4, 4]]
        validated, _ = h.validate_panel(rows, plan, report)
        per, bounds = h.renormalize(validated)
        self.assertEqual(bounds['one']['constant_objectives'], [0, 1, 2])
        self.assertTrue(all(abs(r['hv_01']-1) < 1e-12 for r in per))
        for r in validated:
            for s in r['budget_fronts']:
                s['front_raw'] = []
        per, bounds = h.renormalize(validated)
        self.assertTrue(bounds['one']['empty_union'])
        self.assertTrue(all(r['hv_01'] == 0 for r in per))
        c = h.paired([r for r in per if r['evaluation'] == 400], 'I_LEARN', 'BASE', 1000)
        self.assertIsNone(c['mean_percent'])
        self.assertEqual(c['zero_reference_pairs'], 3)
        self.assertEqual(c['mean_absolute_hv01'], 0)

    def test_pairing_and_incomplete_panels_rejected(self):
        rows, plan, report = fixture()
        variants = [rows[:-1], rows+[rows[0]]]
        for change in ('scenario', 'front', 'finite', 'model', 'count'):
            changed = copy.deepcopy(rows)
            if change == 'scenario': changed[0]['scenario_digest'] = 'wrong'
            if change == 'front': changed[0]['budget_fronts'].pop()
            if change == 'finite': changed[0]['budget_fronts'][0]['front_raw'][0][0] = float('nan')
            if change == 'model': changed[0]['model_bundle_sha256'] = 'different'
            if change == 'count': changed[0]['budget_fronts'][0]['actual_evaluations'] = 401
            variants.append(changed)
        for changed in variants:
            with self.assertRaises(ValueError):
                h.validate_panel(changed, plan, report)

    def test_alias_checks_and_no_invented_runtime(self):
        rows, plan, report = fixture()
        for r in list(rows):
            if r['arm'] == 'ICM_SHARED_SUPERVISED' and r['seed'] in (1, 2):
                rows.append(dict(copy.deepcopy(r), arm='ICM_TERMINAL_PPO', model_bundle_sha256='u0'))
        report.update(selected_ppo_update=0, ppo_alias=dict(selected_update=0, alias_of='ICM_SHARED_SUPERVISED',
            actual_reproduction_runs=2, reused_rows=1, checked_keys=[dict(instance='one', seed=s) for s in (1, 2)]))
        validated, methods = h.validate_panel(rows, plan, report)
        per, _ = h.renormalize(validated)
        summary, _ = h.tables(per, methods, report['ppo_alias'], 1000)
        for r in summary:
            if r['方法'] == h.NAMES['ICM_TERMINAL_PPO']:
                self.assertIsNone(r['平均完整B5000运行秒'])
        self.assertTrue(all(r['full_run_seconds'] is None for r in per if r['result_source'].startswith('u0_alias')))
        rows[-1]['budget_fronts'][0]['front_raw'][0][0] += 1
        with self.assertRaisesRegex(ValueError, 'snapshot mismatch'):
            h.validate_panel(rows, plan, report)

    def test_relative_hv_preserves_gains(self):
        rows, plan, report = fixture()
        validated, methods = h.validate_panel(rows, plan, report)
        per, _ = h.renormalize(validated)
        sample = [r for r in per if r['evaluation'] == 5000]
        a = h.paired(sample, 'I_LEARN', 'BASE', 1000)
        b = h.paired([dict(r, hv_01=r['hv_union']) for r in sample], 'I_LEARN', 'BASE', 1000)
        for key in ('mean_percent', 'ci_low', 'ci_high'):
            self.assertAlmostEqual(a[key], b[key], places=11)

    def test_end_to_end_idempotent_and_source_unchanged(self):
        rows, plan, report = fixture()
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            source = Path(tmp)/'source'
            folder = source/'supervised_results'
            folder.mkdir(parents=True)
            for p, data in [(source/'PLAN.json', plan), (folder/'QUICK_RESULT.json', report), (folder/'PAIRED_RUNS.json', rows)]:
                p.write_bytes(h.json_bytes(data))
            before = {p: p.read_bytes() for p in source.rglob('*.json')}
            out = h.execute(source, draws=1000)
            self.assertEqual(out, h.execute(source, draws=1000))
            q = json.loads((out/'QUICK_RESULT.json').read_bytes())
            self.assertEqual(len(q['summary']), 30)
            self.assertFalse(q['all_methods_including_ppo_complete'])
            self.assertEqual(before, {p: p.read_bytes() for p in source.rglob('*.json')})
            with self.assertRaisesRegex(ValueError, 'outside'):
                h.execute(source, out=source/'new-report', draws=1000)
            (out/'PER_RUN.csv').write_text('changed')
            with self.assertRaisesRegex(ValueError, 'output changed'):
                h.execute(source, draws=1000)


if __name__ == '__main__':
    unittest.main()
