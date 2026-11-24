#!/usr/bin/env python3
"""S6 exact reproduction gate, nine fresh S7 searches, and read-only S6--S8 merge.

All scientific operations call the unchanged section68 code at 1b95b10.
This entry point only selects inputs, schedules frozen workers and joins reports.
"""
import argparse
import copy
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
from statistics import mean, stdev
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace

sys.dont_write_bytecode = True
EXP = Path(__file__).resolve().parent
FROZEN = EXP / 'section68_ev_ccp_worst_v1'
ORIGINAL_COMMIT = '1b95b109957f24b343c153f2f627c4c98f7dc0bd'
PRIOR_ENVIRONMENT_COMMIT = '26ac7559c97bba1e15274eb0195b6aca2d1e3597'
SEEDS = [993001, 993002, 993003]
DIAG = EXP / 'transformer_ppo_runs/diagnostics'


def git(*args):
    return subprocess.check_output(['git', *args], cwd=EXP)


def frozen_import():
    # Check bytes before importing the frozen computation modules.
    files = {}
    for name in ('preflight.py', 'engine.py', 'launch.py'):
        path = FROZEN / name
        relative = 'expanda/section68_ev_ccp_worst_v1/' + name
        original = git('show', ORIGINAL_COMMIT + ':' + relative)
        if path.read_bytes() != original:
            raise ValueError('Frozen computation file differs from 1b95b10: ' + str(path))
        if git('show', PRIOR_ENVIRONMENT_COMMIT + ':' + relative) != original:
            raise ValueError('Prior environment had different computation code: ' + relative)
        files[str(path)] = hashlib.sha256(original).hexdigest()
    sys.path.insert(0, str(FROZEN))
    import launch
    import preflight
    return launch, preflight, files


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def checked_plan(run, root):
    plan = run.read(root / 'PLAN.json')
    if (plan['output'] != str(root)
            or run.canonical({k: v for k, v in plan.items() if k != 'sha256'}) != plan['sha256']):
        raise ValueError('Plan identity/hash differs: ' + str(root))
    run.verify(plan['files'])
    return plan


def checked_result(run, root, plan, task):
    folder = run.completed_task(root, task, plan)
    if folder is None:
        raise ValueError('Missing completed search: ' + str(task))
    meta = run.read(folder / 'COMPLETE.json')
    result = run.read(folder / 'search/RUN_COMPLETE.json')
    config = run.read(folder / 'search/configuration.json')
    selected = next(s for s in plan['selected'] if s['instance'] == task['instance'])
    ev = task['model'] == 'EV'
    tag = 'expected_input' if ev else 'optimization_100'
    pack, pack_meta = run.load_scenarios(root / 'scenarios' / task['configuration'] / tag)
    count = result['real_objective_evaluation_count']
    if (count not in (4999, 5000) or meta['actual_evaluations'] != count
            or result['stop_reason'] != {4999: 'atomic_pair_budget_remainder', 5000: 'evaluation_cap'}[count]):
        raise ValueError('Actual evaluations/stop reason differ from original policy: ' + str(task))
    if (meta.get('task') != task or not meta.get('completed')
            or result['arm'] != 'BASE' or result['enabled_components']
            or result['model_bundle_sha256'] is not None
            or any(result['selection'][c]['calls'] for c in 'ICM')
            or config['evaluation_mode'] != ('deterministic' if ev else 'ccp')
            or config['training_size'] != (1 if ev else 100)
            or config['uncertainty']['scenario_count'] != (1 if ev else 100)
            or config['alpha'] != (1. if task['model'] == 'WORST' else .9)
            or config['population'] != 250 or config['evaluation_budget'] != 5000
            or config['seeds']['algorithm'] != task['seed']
            or config['instance_digest'] != selected['instance_digest']
            or config['scenario_digest'] != pack_meta['scenario_digest']
            or pack.size != (1 if ev else 100) or pack.stochastic != (not ev)):
        raise ValueError('Saved worker settings differ: ' + str(task))
    if ev and (any(not run.np.array_equal(a, run.np.ones(1)) for a in pack.travel_multiplier.values())
               or any(not run.np.array_equal(a, [pack.border_event_mean_h[k]])
                      for k, a in pack.border_delay_h.items())):
        raise ValueError('EV did not use constant mean inputs')
    checks = meta['checks']
    if (not checks['all_front_objectives_reproduced'] or checks['maximum_absolute_error'] != 0
            or not checks['worst_not_less_than_ccp'] or not checks['batch_arrivals_match_makespan']
            or checks['independent_realizations'] != 5000):
        raise ValueError('Frozen re-evaluation checks did not pass: ' + str(task))
    return folder, dict(**task, actual_evaluations=count, stop_reason=result['stop_reason'],
        search_seconds=meta['search_seconds'], reevaluation_seconds=checks['reevaluation_seconds'])


def freeze_sources(run, source, output, frozen_files):
    old = checked_plan(run, source)
    if (old['commit']['stdout'] != ORIGINAL_COMMIT
            or old['protocol'] != 'section68-ev-ccp-worst-v1'
            or len(old['tasks']) != 18):
        raise ValueError('Wrong source experiment/version')
    status = run.read(source / 'STATUS.json')
    if not status.get('completed') or status.get('done') != 18:
        raise ValueError('S6/S8 source experiment is incomplete')
    if (old['settings']['population'], old['settings']['budget'], old['settings']['scenario_count'],
            old['settings']['confidence'], old['settings']['screen_seeds']) != (250, 5000, 100, .9, SEEDS):
        raise ValueError('Unexpected original search settings')
    expected = {(c, s, m) for c in ('S6', 'S8') for s in SEEDS for m in run.MODELS}
    if {(t['configuration'], t['seed'], t['model']) for t in old['tasks']} != expected:
        raise ValueError('Wrong original task coverage')
    # This is the previously authorised continuation record, not a call to its code.
    policy_path = Path(status['budget_policy'])
    policy = run.read(policy_path)
    if (policy['commit']['stdout'] != PRIOR_ENVIRONMENT_COMMIT
            or policy['plan_sha256'] != old['sha256']
            or policy['accepted_actual_counts'] != [4999, 5000]):
        raise ValueError('Prior environment commit/budget acceptance record differs')
    env = run.environment_review(Path(old['preflight']),
        old.get('environment_policy', {}).get('accepted_memtotal_pair'))
    old_env = run.read(source / 'ENVIRONMENT.json')
    if any(env[k] != old_env[k] for k in ('current', 'packages', 'current_affinity')):
        raise ValueError('Current environment differs from S6/S8: ' + json.dumps(
            {k: {'current': env[k], 'original': old_env[k]}
             for k in ('current', 'packages', 'current_affinity') if env[k] != old_env[k]}))
    run.save(output / 'ENVIRONMENT.json', env)
    pins = dict(old['files'])
    for p in (source / 'PLAN.json', source / 'STATUS.json', policy_path):
        pins[str(p)] = run.sha(p)
    lookup, counts = {}, []
    for task in old['tasks']:
        folder, row = checked_result(run, source, old, task)
        lookup[(task['configuration'], task['seed'], task['model'])] = folder
        counts.append(row)
        pins[str(folder / 'COMPLETE.json')] = run.sha(folder / 'COMPLETE.json')
    head = git('rev-parse', 'HEAD').decode().strip()
    relative = 'expanda/' + Path(__file__).name
    if git('show', head + ':' + relative) != Path(__file__).read_bytes():
        raise ValueError('Supplementary entry is not identical to its committed version')
    version = dict(original_experiment_commit=ORIGINAL_COMMIT,
        confirmed_prior_environment_commit=PRIOR_ENVIRONMENT_COMMIT,
        supplementary_entry_commit=head, launch_checkout_commit=head,
        supplementary_entry_sha256=run.sha(__file__),
        frozen_computation_files=frozen_files, source_plan_sha256=old['sha256'],
        prior_environment_evidence=str(policy_path),
        source_root=str(source), new_searches_planned=10, s7_searches=9, reproduction_searches=1,
        computing_logic='Unmodified 1b95b10 preflight/engine/launch and original frozen dependency hashes',
        budget_policy='Unchanged Base upper limit 5000; actual counts 4999 or 5000, no padding',
        selection='Smallest-ID S7 validation instance; S6 reproduction is excluded from final averages')
    run.save(output / 'VERSIONS.json', version)
    run.save(output / 'SOURCE_HASHES.json', pins)
    run.csv_write(output / 'ORIGINAL_ACTUAL_EVALUATIONS.csv', counts)
    return old, lookup, pins, version


def prepare_reproduction(run, source, old, output, version):
    root = output / 'S6_REPRODUCTION'
    root.mkdir()
    for name in ('instances', 'scenarios', 'runs', 'logs'):
        (root / name).mkdir()
    selected = copy.deepcopy(next(s for s in old['selected'] if s['configuration'] == 'S6'))
    instance = root / 'instances' / (selected['instance'] + '.json')
    shutil.copy2(selected['local_instance'], instance)
    selected['local_instance'] = str(instance)
    shutil.copytree(source / 'scenarios/S6', root / 'scenarios/S6')
    files = dict(old['files'])
    files.update({str(p): run.sha(p) for p in root.rglob('*') if p.is_file()})
    files[str(output / 'VERSIONS.json')] = run.sha(output / 'VERSIONS.json')
    files[str(Path(__file__))] = run.sha(__file__)
    task = dict(configuration='S6', instance=selected['instance'], seed=993001, model='CCP')
    plan = {k: copy.deepcopy(old[k]) for k in
            ('settings', 'normalization', 'uncertainty', 'environment_policy', 'preflight')}
    plan.update(protocol='section68-s6-reproduction-v1', output=str(root), selected=[selected],
                tasks=[task], files=files, versions=version, final_searches=1)
    plan['sha256'] = run.canonical(plan)
    run.save(root / 'PLAN.json', plan)
    return root, plan


def schedule(run, root, plan, output, stage, workers):
    """Orchestration only: every child executes the unchanged frozen worker."""
    pending = list(range(len(plan['tasks'])))
    running, completed = [], []
    last = 0.
    try:
        while pending or running:
            while pending and len(running) < workers:
                i = pending.pop(0)
                task = plan['tasks'][i]
                if run.task_dir(root, task).exists():
                    raise ValueError('Fresh searches only; task directory already exists')
                attempt = run.task_dir(root, task) / ('attempt_' + run.stamp())
                log = root / 'logs' / (f'{i:02d}_' + run.stamp() + '.log')
                stream = log.open('x')
                process = subprocess.Popen([sys.executable, '-B', '-u', str(FROZEN / 'launch.py'),
                    '--worker', '--out', str(root), '--task-index', str(i), '--attempt', str(attempt)],
                    stdout=stream, stderr=subprocess.STDOUT, cwd=EXP, start_new_session=True)
                running.append((i, process, stream, log, attempt))
                print('[S7] Start', task, 'pid=', process.pid, 'log=', log, flush=True)
            for item in list(running):
                i, process, stream, log, attempt = item
                if process.poll() is not None:
                    stream.close(); running.remove(item)
                    if process.returncode:
                        failure = attempt / 'FAILURE.json'
                        detail = run.read(failure)['error'] if failure.exists() else log.read_text(errors='replace')[-4000:]
                        raise RuntimeError('Frozen worker stopped: ' + detail + '; log=' + str(log))
                    _, row = checked_result(run, root, plan, plan['tasks'][i])
                    completed.append(row)
            if time.monotonic() - last >= 15:
                run.status(output, completed=False, stage=stage, done=len(completed), total=len(plan['tasks']),
                    running=[dict(task=plan['tasks'][i], pid=p.pid, log=str(log)) for i, p, _, log, _ in running])
                last = time.monotonic()
            if pending or running:
                time.sleep(1)
    finally:
        for _, process, stream, _, _ in running:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill(); process.wait()
            stream.close()
    return completed


def compare_reproduction(run, original, reproduced, output):
    a_path = original / 'search/final_feasible_nondominated.json'
    b_path = reproduced / 'search/final_feasible_nondominated.json'
    a, b = run.read(a_path), run.read(b_path)
    signature = lambda front: sorted((run.decision_id(r), tuple(run.objectives(r))) for r in front)
    report = dict(original_file=str(a_path), reproduced_file=str(b_path),
        original_sha256=run.sha(a_path), reproduced_sha256=run.sha(b_path),
        original_solutions=len(a), reproduced_solutions=len(b),
        exact_decisions_and_objectives=signature(a) == signature(b),
        exact_full_front_records=a == b, tolerance=0,
        original_evaluations=run.read(original / 'COMPLETE.json')['actual_evaluations'],
        reproduced_evaluations=run.read(reproduced / 'COMPLETE.json')['actual_evaluations'])
    report['passed'] = (report['exact_decisions_and_objectives'] and report['exact_full_front_records']
                        and report['original_evaluations'] == report['reproduced_evaluations'])
    run.save(output / 'S6_REPRODUCTION_CHECK.json', report)
    if not report['passed']:
        raise ValueError('S6 CCP seed 993001 is not an exact reproduction; S7 was not started')
    print('[S7] PASS S6 CCP 993001: exact full front and actual evaluation count', flush=True)


def prepare_s7(run, pre, old, output, version):
    original_preflight = run.read(Path(old['preflight']) / 'PREFLIGHT.json')
    audit = output / 'S7_PREFLIGHT'
    # Only the target selection changes. All preflight calculations are frozen.
    targets = pre.TARGETS
    pre.TARGETS = {'S7': (30, 1.)}
    try:
        result = pre.main(SimpleNamespace(out=audit,
            table5=Path(original_preflight['sources'][0]), ppo=Path(original_preflight['sources'][1])))
    finally:
        pre.TARGETS = targets
    if result:
        raise ValueError('S7 frozen preflight failed; see S7_PREFLIGHT/PREFLIGHT.json')
    p = run.read(audit / 'PREFLIGHT.json')
    if (len(p['selected']) != 1 or p['selected'][0]['configuration'] != 'S7'
            or p['selected'][0]['K'] != 30 or p['selected'][0]['deadline_window_alpha'] != 1.
            or p['selected'][0]['search_seeds'] != SEEDS or p['uncertainty'] != old['uncertainty']):
        raise ValueError('S7 preflight specifications differ')
    root = output / 'S7_RUNS'
    root.mkdir()
    # prepare() has a hard-coded 18-search report count. Correct metadata only
    # before the first immutable PLAN write; no scientific parameters change.
    original_save = run.save
    def save_metadata(path, obj):
        if Path(path) == root / 'PLAN.json':
            if (obj['settings'] != old['settings'] or obj['normalization'] != old['normalization']
                    or obj['uncertainty'] != old['uncertainty'] or len(obj['tasks']) != 9):
                raise ValueError('Prepared S7 settings differ from original')
            obj.pop('sha256')
            obj.update(protocol='section68-s7-supplement-v1', final_searches=9, pilot_task_indices=[],
                       stop_on_nonexact_pilot_budget=False, versions=version,
                       accepted_actual_counts=[4999, 5000], original_plan_sha256=old['sha256'])
            for f in (Path(__file__), output / 'VERSIONS.json', output / 'S6_REPRODUCTION_CHECK.json'):
                obj['files'][str(f)] = run.sha(f)
            obj['sha256'] = run.canonical(obj)
        original_save(path, obj)
    run.save = save_metadata
    try:
        plan = run.prepare(SimpleNamespace(preflight=audit,
            accept_memtotal_pair=old['environment_policy']['accepted_memtotal_pair']), root)
    finally:
        run.save = original_save
    if checked_plan(run, root) != plan:
        raise ValueError('Prepared plan round-trip mismatch')
    selected = plan['selected'][0]
    if (selected['independent_provenance']['action'] == 'new independent stream'
            and selected['independent_seed'] != 368500007):
        raise ValueError('New S7 independent seed differs from original seed formula')
    run.save(output / 'SELECTED_S7.json', selected)
    return root, plan


def merge_reports(run, old_root, old, s7_root, s7, output, version, reproduction_counts):
    task_sources, folders, counts = [], {}, []
    for root, plan in ((old_root, old), (s7_root, s7)):
        for task in plan['tasks']:
            folder, count = checked_result(run, root, plan, task)
            key = (task['configuration'], task['seed'], task['model'])
            if key in folders:
                raise ValueError('Duplicate search in combined report')
            folders[key] = folder
            counts.append(count)
            task_sources.append(dict(task=task, complete_file=str(folder / 'COMPLETE.json'),
                                     complete_sha256=run.sha(folder / 'COMPLETE.json')))
    expected = {(c, s, m) for c in ('S6', 'S7', 'S8') for s in SEEDS for m in run.MODELS}
    if set(folders) != expected:
        raise ValueError('Expected 27 unique searches for three configurations')
    combined = output / 'COMBINED_S6_S7_S8'
    combined.mkdir()
    tasks = [dict(configuration=c, seed=s, model=m,
                  instance=next(t['task']['instance'] for t in task_sources if
                                (t['task']['configuration'], t['task']['seed'], t['task']['model']) == (c, s, m)))
             for c in ('S6', 'S7', 'S8') for s in SEEDS for m in run.MODELS]
    manifest = dict(protocol='section68-s6-s7-s8-merge-v1', tasks=tasks, task_sources=task_sources,
                    versions=version, old_plan_sha256=old['sha256'], s7_plan_sha256=s7['sha256'])
    manifest['sha256'] = run.canonical(manifest)
    run.save(combined / 'PLAN.json', manifest)
    # Reuse the exact original export calculations, resolving each task to its
    # original completed folder. No source folder is written or re-evaluated.
    resolver = run.completed_task
    run.completed_task = lambda _root, t, _plan: folders[(t['configuration'], t['seed'], t['model'])]
    try:
        report, _ = run.export(combined, manifest, list(range(27)), 'RESULTS')
    finally:
        run.completed_task = resolver
    routes = read_csv(report / 'ROUTE_METRICS.csv')
    split = {(r['configuration'], r['seed'], r['model'], r['plan_type']): float(r['value'])
             for r in routes if r['metric'] == 'split_batch_fraction' and r['item'] == 'all'}
    per_seed = read_csv(report / 'PER_SEED.csv')
    if len(per_seed) != 54 or len(split) != 54:
        raise ValueError('Missing chosen-plan summary or split-batch metric')
    metrics = ('cost_mean', 'cost_p90', 'emission_p90', 'makespan_p90', 'punctuality_percent',
               'makespan_mean', 'emission_mean')
    detailed = []
    for r in per_seed:
        key = (r['configuration'], r['seed'], r['model'], r['plan_type'])
        detailed.append({k: r[k] for k in ('configuration', 'instance', 'seed', 'model', 'plan_type')})
        detailed[-1].update({k: float(r[k]) for k in metrics})
        detailed[-1]['split_batch_percent'] = 100 * split[key]
    means = read_csv(report / 'SEED_MEANS.csv')
    main = []
    for r in means:
        matches = [d for d in detailed if all(d[k] == r[k] for k in ('configuration', 'instance', 'model', 'plan_type'))]
        if len(matches) != 3 or {int(d['seed']) for d in matches} != set(SEEDS):
            raise ValueError('Missing three common seeds at merge')
        row = {k: r[k] for k in ('configuration', 'instance', 'model', 'plan_type', 'seeds')}
        row.update({k: float(r[k]) for k in metrics})
        row['split_batch_percent'] = mean(d['split_batch_percent'] for d in matches)
        main.append(row)
    timings = []
    for cfg in ('S6', 'S7', 'S8'):
        for model in run.MODELS:
            group = [r for r in counts if r['configuration'] == cfg and r['model'] == model]
            timings.append(dict(configuration=cfg, instance=group[0]['instance'], model=model, searches=3,
                search_seconds_mean=mean(r['search_seconds'] for r in group),
                search_seconds_sd=stdev(r['search_seconds'] for r in group)))
    run.csv_write(report / 'MAIN_RESULTS_PER_SEED.csv', detailed)
    run.csv_write(report / 'MAIN_RESULTS_SEED_MEANS.csv', main)
    run.csv_write(report / 'SEARCH_TIME_SEED_MEANS.csv', timings)
    run.csv_write(report / 'ACTUAL_EVALUATIONS_ALL27.csv', counts)
    run.csv_write(report / 'REPRODUCTION_RUNTIME.csv', reproduction_counts)
    lines = ['6.8 S6、S7、S8 合并结果；各行均为3个共同种子的平均值。',
             '原冻结版本：' + ORIGINAL_COMMIT, '上一轮环境提交：' + PRIOR_ENVIRONMENT_COMMIT,
             '新增入口提交：' + version['supplementary_entry_commit'],
             'S6 CCP 993001完整前沿精确复现通过；复现运行不重复计入下表。',
             'EV为平均输入下1次评价；CCP/最坏情况使用100个优化场景；独立复评5000个场景。',
             '停止规则：预算上限5000，实际次数见ACTUAL_EVALUATIONS_ALL27.csv。',
             '表1：配置 | 模型 | 方案 | 均值成本 | P90成本 | P90排放 | P90完工时间(h) | 准时率(%) | 平均完工时间(h) | 平均排放 | 拆分批次(%)']
    for r in main:
        lines.append(' | '.join([r['configuration'], r['model'], r['plan_type']] +
            [f'{r[k]:.6f}' for k in (*metrics, 'split_batch_percent')]))
    lines.append('表2：配置 | 模型 | 方案 | 均值成本变化(%) | P90完工时间变化(%) | 准时率变化(百分点)，参照EV')
    for r in read_csv(report / 'PAIRED_SEED_MEANS.csv'):
        lines.append(' | '.join([r['configuration'], r['model'], r['plan_type']] +
            [f'{float(r[k]):.6f}' for k in ('cost_mean_change_percent', 'makespan_p90_change_percent', 'punctuality_change_percentage_points')]))
    lines.append('搜索时间：配置 | 模型 | 秒（均值±样本标准差）')
    for r in timings:
        lines.append(f"{r['configuration']} | {r['model']} | {r['search_seconds_mean']:.2f} ± {r['search_seconds_sd']:.2f}")
    lines.append('实际评价次数：' + json.dumps(counts, ensure_ascii=False))
    lines.append('沿用原始路线指标定义；本次未修改铁路口岸分类。上述拆分比例直接读取原split_batch_fraction。')
    (report / 'COMBINED_REPORT_CN.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    run.save(report / 'SUPPLEMENT_MANIFEST.json', dict(versions=version,
        files={str(p): run.sha(p) for p in report.iterdir() if p.is_file()},
        source_files={s['complete_file']: s['complete_sha256'] for s in task_sources}))
    return report, counts


def main(args):
    if not 1 <= args.workers <= 2:
        raise ValueError('At most two parallel searches')
    root, source = args.out.resolve(), args.source.resolve()
    if root.exists() or source == root or source in root.parents or root in source.parents:
        raise ValueError('Use a new output directory separate from the existing experiment')
    run, pre, frozen_files = frozen_import()
    root.mkdir(parents=True)
    try:
        old, lookup, pins, version = freeze_sources(run, source, root, frozen_files)
        run.status(root, completed=False, stage='S6_reproduction', done=0, total=1)
        repro_root, repro = prepare_reproduction(run, source, old, root, version)
        reproduction_counts = schedule(run, repro_root, repro, root, 'S6_reproduction', 1)
        reproduced, _ = checked_result(run, repro_root, repro, repro['tasks'][0])
        compare_reproduction(run, lookup[('S6', 993001, 'CCP')], reproduced, root)
        run.verify(pins)
        s7_root, s7 = prepare_s7(run, pre, old, root, version)
        run.status(root, completed=False, stage='S7_searches', done=0, total=9,
                   selected=s7['selected'][0], reproduction_passed=True)
        schedule(run, s7_root, s7, root, 'S7_searches', args.workers)
        run.verify(pins)
        run.verify(s7['files'])
        report, counts = merge_reports(run, source, old, s7_root, s7, root, version, reproduction_counts)
        run.verify(pins)
        run.status(root, completed=True, stage='all', done=9, total=9, results=str(report),
            reproduction_passed=True, new_searches=10, combined_searches=27,
            selected=s7['selected'][0], actual_evaluation_counts=counts,
            reproduction_actual_evaluation_counts=reproduction_counts)
        print('[S7] COMPLETE', report, flush=True)
    except BaseException as exc:
        run.save(root / ('FAILURE_' + run.stamp() + '.json'), dict(error=repr(exc), traceback=traceback.format_exc()))
        run.status(root, completed=False, stage='stopped', error=str(exc))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=DIAG / 'section68_ev_ccp_worst_runs_v3')
    parser.add_argument('--out', type=Path, default=DIAG / 'section68_s7_supplement_v1')
    parser.add_argument('--workers', type=int, default=2)
    main(parser.parse_args())
