"""Supplement verification; synthetic fixtures are not experimental evidence."""
import contextlib
import copy
import importlib.util
import io
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('base_report_checks', HERE / 'run.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def record(instance='fixture-S0', seed=1, hv=.5, point=5.):
    return dict(completed=True, instance=instance, configuration='S0', seed=seed,
        instance_digest='instance', scenario_digest='scenario', evaluation_budget=5000,
        real_objective_evaluation_count=5000, hidden_candidate_objective_evaluations=0,
        final_hypervolume=hv, final_front_raw=[[point]*3], optimisation_seconds=1.,
        budget_fronts=[dict(evaluation=b, hv=hv, front_raw=[[point]*3]) for b in runner.BUDGETS])


class StatisticalChecks(unittest.TestCase):
    def test_prepare_reuses_only_exact_panel_and_freezes_output(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); source = root / 'pilot'; output = root / 'supplement'; old = root / 'old'
            norm = dict(minimum=[0.]*3, maximum=[10.]*3, reference=[1.1]*3)
            original_plan = dict(normalization=norm)
            original_plan['sha256'] = runner.canonical(original_plan)
            runner.write(old / 'PLAN.json', original_plan)
            entries = []; paths = []; files = {}; settings = dict(population=250, budget=5000,
                scenario_count=100, progress_denominator=50000, milestones=list(runner.BUDGETS),
                ppo=dict(validation_seeds=[1, 2, 3]))
            for u in (0, 1): runner.write(source / f'checkpoints/update_{u:03d}/BUNDLE.json', {'synthetic': u})
            for config in ('S0', 'S4', 'S8'):
                entry = runner.read(HERE.parent / 'formal_icm_s0_v1/assets/search_instances/prescreen-search-S0-001.json')
                entry.pop('instance_digest')
                entry.update(instance_id='fixture-' + config, configuration=config, development_role='validation')
                entry['instance_digest'] = runner.canonical(entry); entries.append(entry)
                runner.write(source / 'instances' / (entry['instance_id'] + '.json'), entry)
                for seed in (1, 2, 3):
                    row = record(entry['instance_id'], seed)
                    row.update(configuration=config, instance_digest=entry['instance_digest'], stochastic=False,
                               arm='ICM_TERMINAL_PPO', plan_sha256=original_plan['sha256'])
                    p = old / 'validation' / entry['instance_id'] / f'seed_{seed}/RUN_COMPLETE.json'
                    runner.write(p, row); paths.append(str(p)); files[str(p)] = runner.digest(p)
                    # Exactly one matching Base is reusable; unrelated old screen keys are ignored.
                    target = old / 'screen' / (entry['instance_id'] if config == 'S0' and seed == 1 else 'other-' + config) / f'seed_{seed}/BASE/RUN_COMPLETE.json'
                    runner.write(target, dict(row, arm='BASE'))
            source_plan = dict(source=str(old), source_plan_sha256=original_plan['sha256'], normalization=norm,
                files=files, settings=settings, confirmation=entries, validation_baseline_files=paths)
            source_plan['sha256'] = runner.canonical(source_plan)
            runner.write(source / 'PLAN.json', source_plan)
            runner.write(source / 'STATUS.json', dict(completed=True, selected_updated_checkpoint=1))
            runner.write(source / 'SELECTION.json', dict(best_updated_checkpoint=1))
            for entry in entries:
                for seed in (1, 2, 3):
                    for u, deployment in ((0, 'sampled'), (1, 'sampled'), (1, 'greedy')):
                        row = record(entry['instance_id'], seed)
                        row.update(configuration=entry['configuration'], instance_digest=entry['instance_digest'],
                            stochastic=deployment == 'sampled', plan_sha256=source_plan['sha256'],
                            model_bundle_sha256=runner.digest(source / f'checkpoints/update_{u:03d}/BUNDLE.json'))
                        runner.write(source / f'runs/confirmation/u{u:03d}' / deployment / entry['instance_id'] / f'seed_{seed}/RUN_COMPLETE.json', row)
            plan = runner.prepare(output, source)
            self.assertEqual(len(plan['reused_base']), 1)
            self.assertEqual(sum(map(len, plan['sources'].values())), 36)
            self.assertEqual(plan, runner.prepare(output, source))
            changed = Path(plan['sources']['PPO_GREEDY'][0])
            row = runner.read(changed); row['scenario_digest'] = 'wrong'; runner.write(changed, row)
            with self.assertRaises(ValueError): runner.prepare(output, source)

    def test_pairing_uses_each_base_not_ratio_of_means_or_best_run(self):
        refs = [record(seed=1, hv=.1), record(seed=2, hv=1.)]
        own = [record(seed=1, hv=.2), record(seed=2, hv=1.)]
        stat = runner.pilot.paired(own, refs, draws=10)
        self.assertAlmostEqual(stat['mean_paired_percent'], 50.)
        self.assertEqual((stat['wins'], stat['ties'], stat['losses']), (1, 1, 0))
        with self.assertRaises(ValueError):
            runner.pilot.paired(own, [dict(r, scenario_digest='other') for r in refs], draws=10)
        with self.assertRaises(ValueError): runner.pilot.paired(own, refs[:1], draws=10)

    def test_zero_base_never_uses_small_constant(self):
        stat = runner.pilot.paired([record(hv=.3)], [record(hv=0.)], draws=10)
        self.assertIsNone(stat['mean_paired_percent'])
        self.assertEqual(stat['mean_absolute_difference'], .3)
        self.assertEqual(runner.pilot.hv.interval(stat['per_run'], 'percent', 10), (None, None))

    def test_union_contains_base_all_seeds_budgets_and_preserves_source(self):
        groups = {m: [record()] for m in runner.NAMES}
        groups['BASE'][0]['budget_fronts'][0]['front_raw'] = [[10.]*3]
        groups['BASE'].append(record(seed=2, point=2.))
        original = copy.deepcopy(groups)
        normalized, bounds = runner.union_normalize(groups)
        self.assertEqual(groups, original)
        self.assertEqual(bounds['fixture-S0']['minimum'], [2.]*3)
        self.assertEqual(bounds['fixture-S0']['maximum'], [10.]*3)
        self.assertEqual(bounds['fixture-S0']['reference'], [1.1]*3)
        self.assertAlmostEqual(normalized['BASE'][1]['final_hypervolume'], 1.331)
        self.assertAlmostEqual(normalized['BASE'][0]['budget_fronts'][0]['hv'], .001)

    def test_empty_front_zero_span_and_nonfinite_handling(self):
        groups = {'BASE': [record(point=1.)]}
        norm, _ = runner.union_normalize(groups)
        self.assertAlmostEqual(norm['BASE'][0]['final_hypervolume'], 1.331)
        for p in groups['BASE'][0]['budget_fronts']: p['front_raw'] = []
        norm, _ = runner.union_normalize(groups)
        self.assertEqual(norm['BASE'][0]['final_hypervolume'], 0.)
        groups['BASE'][0]['budget_fronts'][0]['front_raw'] = [[float('nan')]*3]
        with self.assertRaises(ValueError): runner.union_normalize(groups)

    def test_report_is_explicitly_relative_to_base(self):
        groups = {m: [record(hv=.6)] for m in runner.NAMES}
        groups['BASE'] = [record(hv=.5)]
        plan = dict(selected_update=1, qualification='synthetic test')
        with tempfile.TemporaryDirectory() as d, contextlib.redirect_stdout(io.StringIO()):
            runner.report(groups, Path(d), plan)
            q = runner.read(Path(d) / 'QUICK_RESULT.json')
            stat = q['comparisons']['PPO_GREEDY/BASE/5000']
            self.assertAlmostEqual(stat['mean_paired_percent'], 20.)
            self.assertEqual(q['comparisons']['PPO_GREEDY/SHARED_GREEDY/5000']['mean_paired_percent'], 0.)
            self.assertEqual(q['reference_method'], 'BASE')

    def test_validation_rejects_wrong_budget_scenario_and_front(self):
        row = record()
        e = dict(instance_id=row['instance'], instance_digest='instance', configuration='S0')
        settings = dict(budget=5000, milestones=list(runner.BUDGETS))
        runner.validate_run(row, e, 1, settings, 'scenario')
        for changed in (dict(scenario_digest='wrong'), dict(real_objective_evaluation_count=400),
                        dict(final_front_raw=[]), dict(hidden_candidate_objective_evaluations=1)):
            with self.assertRaises(ValueError): runner.validate_run(dict(row, **changed), e, 1, settings, 'scenario')


class EngineChecks(unittest.TestCase):
    def test_base_cli_matches_original_engine_and_resume_reuses_record(self):
        expanda = HERE.parent
        with tempfile.TemporaryDirectory(prefix='base-report-check-') as d:
            root = Path(d)
            info = runner.read(expanda / 'formal_icm_s0_v1/assets/search_instances/prescreen-search-S0-001.json')
            info.pop('instance_digest')
            info.update(instance_id='validation-fixture', development_role='validation')
            info['instance_digest'] = runner.canonical(info)
            runner.write(root / 'instances/validation-fixture.json', info)
            settings = runner.read(runner.pilot.SUP / 'protocol.json')
            settings.update(population=20, budget=400, milestones=[100, 200, 400],
                            ppo={'validation_seeds': [12345]})
            norm = runner.read(expanda / 'formal_icm_s0_v1/assets/reward_normalization.json')
            norm = {k: norm[k] for k in ('minimum', 'maximum', 'reference')}
            plan = dict(settings=settings, normalization=norm, files={}, entries=[info])
            task = runner.base_task(plan, info, 12345)
            runner.write(root / 'settings.json', settings); runner.write(root / 'task.json', task)
            original = root / 'original'; original.mkdir()
            code = "import sys;from pathlib import Path;sys.path.insert(0,sys.argv[1]);from search_b5000 import run_search,read_json;r=Path(sys.argv[2]);run_search(r/'instances/validation-fixture.json',read_json(r/'settings.json'),read_json(r/'task.json'),r/'original','original')"
            with (root / 'original.log').open('w') as log:
                p = subprocess.run([sys.executable, '-c', code, str(runner.pilot.B5), str(root)], stdout=log, stderr=subprocess.STDOUT)
            self.assertEqual(p.returncode, 0, (root / 'original.log').read_text()[-5000:])
            before = runner.read(original / 'RUN_COMPLETE.json')
            plan['scenarios'] = {'validation-fixture/12345': before['scenario_digest']}
            plan['sha256'] = runner.canonical(plan); runner.write(root / 'PLAN.json', plan)
            folder = root / 'runs/validation-fixture/seed_12345/BASE'
            with contextlib.redirect_stdout(io.StringIO()):
                rows = runner.scheduler.execute_jobs(root, plan, [(task, folder)], 1, 'test')
            after = rows[0]
            for key in ('final_front_raw', 'final_hypervolume', 'scenario_digest', 'evaluation_order_digest',
                        'real_objective_evaluation_count', 'selection'):
                self.assertEqual(before[key], after[key], key)
            self.assertEqual([(s['evaluation'], s['front_raw'], s['hv']) for s in before['budget_fronts']],
                             [(s['evaluation'], s['front_raw'], s['hv']) for s in after['budget_fronts']])
            self.assertFalse(any(after['selection'][c]['calls'] for c in 'ICM'))
            self.assertIsNone(after['model_bundle_sha256'])
            self.assertEqual(after['hidden_candidate_objective_evaluations'], 0)
            modified = (folder / 'RUN_COMPLETE.json').stat().st_mtime_ns
            with contextlib.redirect_stdout(io.StringIO()):
                cached = runner.scheduler.execute_jobs(root, plan, [(task, folder)], 1, 'resume')
            self.assertEqual(cached, rows)
            self.assertEqual((folder / 'RUN_COMPLETE.json').stat().st_mtime_ns, modified)
            with self.assertRaises(ValueError):
                runner.worker(root, plan, dict(task, arm='ICM_TERMINAL_PPO'), folder)


if __name__ == '__main__': unittest.main()
