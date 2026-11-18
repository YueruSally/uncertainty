#!/usr/bin/env python3
"""Add one new S0-S8 instance per configuration to the frozen single-member panel."""
import argparse
from collections import deque
from dataclasses import asdict
import fcntl
import os
from pathlib import Path
import platform
import shutil
import signal
import socket
from statistics import mean, stdev
import subprocess
import sys
import tempfile
import time

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
DIAG = EXP / 'transformer_ppo_runs/diagnostics'
sys.path.insert(0, str(EXP))
import icm_single_member_panel_b5000 as previous

NAMES = previous.NAMES
ARMS = tuple(NAMES)
CONFIGS = [f'S{i}' for i in range(9)]
SEEDS = [993001, 993002, 993003]
GENERATION = dict(instance_seed_base=240000000, scenario_seed_base=242000000,
                  instances_per_configuration=1, version=1)
STOP = None


def kind(arm):
    if arm == previous.SHARED:
        return 'shared'
    if arm == previous.PPO:
        return 'ppo'
    return 'independent' if arm in previous.INDEPENDENT else None


def checked_plan(s, path):
    p = s.read(path)
    if s.canonical({k: v for k, v in p.items() if k != 'sha256'}) != p['sha256']:
        raise ValueError('Plan checksum changed: ' + str(path))
    return p


def runtime_files(s, original, panel):
    folders = ('component_joint_b5000_v1', 'component_screen_b20_v1', 'joint_icm_multiscale_v1',
               'joint_icm_multiscale_terminal_v1', 'ppo_joint_b5000_pilot_v2',
               'ppo_joint_b5000_screen27_v1', 'ppo_joint_b5000_base_report_v1', 'hv_joint_b5000_union_v1')
    paths = [HERE/'run.py', EXP/'icm_single_member_b5000.py', EXP/'icm_single_member_panel_b5000.py',
             EXP/'formal_icm_s0_v1/assets/data/data_expanded.xlsx']
    paths += [p for folder in folders for p in (EXP/folder).glob('*') if p.suffix in ('.py', '.json')]
    paths += [p for p in (EXP/'formal_icm_s0_v1/src').rglob('*') if p.suffix in ('.py', '.csv')]
    frozen = {**original['files'], **panel['source_code']}
    result = {}
    for path in paths:
        if path.name == 'checks.py' or any('test' in part.lower() for part in path.relative_to(EXP).parts):
            continue
        value = s.digest(path)
        if str(path) in frozen and value != frozen[str(path)]:
            raise ValueError('Historical runtime changed: ' + str(path))
        result[str(path)] = value
    return result


def collect_old(s, source, panel, original_root, shared_root, models):
    records = []
    for entry in panel['entries']:
        name = entry['instance_id']
        if s.read(original_root/'instances'/(name+'.json')) != entry:
            raise ValueError('Original instance changed: ' + name)
        for seed in SEEDS:
            for arm in ARMS:
                if arm == previous.SHARED:
                    path = shared_root/'runs'/name/f'seed_{seed}/RUN_COMPLETE.json'
                    expected_plan = panel['shared_single_plan']
                elif arm in previous.RULES:
                    path = original_root/'screen'/name/f'seed_{seed}'/arm/'RUN_COMPLETE.json'
                    expected_plan = panel['source_plan']
                else:
                    path = source/'runs'/name/f'seed_{seed}'/arm/'RUN_COMPLETE.json'
                    expected_plan = panel['sha256']
                row = s.read(path)
                s.audit.validate_run(row, entry, seed, panel['settings'], panel['scenarios'][f'{name}/{seed}'])
                model_hash = models[kind(arm)]['bundle_sha256'] if kind(arm) else None
                if (row['arm'] != arm or row['stochastic'] or row['phase'] != 'screen'
                        or row['plan_sha256'] != expected_plan or row['model_bundle_sha256'] != model_hash):
                    raise ValueError('Not the frozen single-member run: ' + str(path))
                if str(path) in panel['reused'] and s.digest(path) != panel['reused'][str(path)]:
                    raise ValueError('Previously reused run changed: ' + str(path))
                records.append(dict(path=str(path), sha256=s.digest(path), instance=name, seed=seed,
                                    arm=arm, seconds=row['optimisation_seconds']))
    if len(records) != 297:
        raise ValueError('Expected all 297 original physical runs')
    return records


def generate_instances(s, root, panel, original, known_paths):
    import baseline_uncertainty as base
    from transformer_ppo.instance_catalog import _batches, _find_path_seed, SCENARIOS
    settings = panel['settings']
    catalog_path = Path(settings['source_catalog_default'])
    catalog = s.read(catalog_path)
    if (catalog['catalog_digest'] != settings['source_catalog_digest'] or
            s.canonical({k: v for k, v in catalog.items() if k != 'catalog_digest'}) != catalog['catalog_digest']):
        raise ValueError('Original OD split changed')
    workbook = EXP/'formal_icm_s0_v1/assets/data/data_expanded.xlsx'
    if s.digest(workbook) != catalog['source_data_sha256']:
        raise ValueError('Original transport workbook changed')
    od = [tuple(x) for x in catalog['train_od_pool']]
    if not od or set(od) & {tuple(x) for x in catalog['test_od_pool']}:
        raise ValueError('Invalid OD split')
    # Read instance manifests only, never their outcomes, when excluding prior instances.
    known = set()
    for path in sorted(known_paths):
        obj = s.read(path)
        entries = [obj] if 'batches' in obj else obj.get('instances', [])
        known.update(s.canonical(e['batches']) for e in entries if 'batches' in e)
    known.update(s.canonical(e['batches']) for e in panel['entries'] + original['entries'])
    base.BORDER_EVENT_DEFINITIONS = base.load_border_event_definitions(base.DEFAULT_BORDER_EVENT_DATA_FILE)
    nodes, regions, _, _, _, arcs, timetables, raw, *_ = base.load_network_from_extended(str(workbook))
    tt, lookup = base.build_timetable_dict(timetables), base.build_arc_lookup(arcs)
    records = []
    for i, cfg in enumerate(CONFIGS):
        sample = next(e for e in panel['entries'] if e['configuration'] == cfg)
        k, alpha = SCENARIOS[cfg]
        if (sample['K'], sample['deadline_window_alpha']) != (k, alpha):
            raise ValueError('Configuration definition changed: ' + cfg)
        seed = GENERATION['instance_seed_base'] + 100*i
        release = sample['release_time_definition']
        batches = _batches(raw, od, k, alpha, seed, 'train', release['maximum_h'], release['step_h'])
        signature = s.canonical([asdict(b) for b in batches])
        if signature in known:
            raise ValueError('New instance duplicates an existing one; no automatic redraw: ' + cfg)
        known.add(signature)
        row = {key: sample[key] for key in ('schema_version', 'alpha_definition', 'release_time_definition',
                                           'quantity_definition', 'od_pool')}
        row.update(instance_id=f'joint-confirm-{cfg}-002', configuration=cfg, K=k,
                   deadline_window_alpha=alpha, split='train', development_role='screen',
                   instance_seed=seed, path_seed=_find_path_seed(seed+1000000, nodes, regions, arcs, batches, tt, lookup),
                   algorithm_seed=SEEDS[0], ccp100_seed=GENERATION['scenario_seed_base']+i, oos5000_seed=None,
                   source_data_sha256=s.digest(workbook), source_catalog_digest=catalog['catalog_digest'],
                   batch_signature=signature, batches=[asdict(b) for b in batches])
        row['instance_digest'] = s.canonical(row)
        path = root/'instances'/(row['instance_id']+'.json')
        s.freeze(path, row)
        s.p.read_instance(path)
        records.append(row)
        print('[SECOND] frozen', row['instance_id'], 'zero objective evaluations', flush=True)
    return records, catalog_path


def prepare(s, root, source):
    if (root/'PLAN.json').exists():
        plan = checked_plan(s, root/'PLAN.json')
        if plan['source'] != str(source):
            raise ValueError('Output belongs to a different source')
        s.p.verify(plan['files'])
        s.p.verify({r['path']: r['sha256'] for r in plan['reused']})
        return plan
    panel = checked_plan(s, source/'PLAN.json')
    if not s.read(source/'STATUS.json').get('completed'):
        raise ValueError('The original single-member experiment is incomplete')
    original_root = source.parent/'ccp_component_joint_b5000_v1'
    shared_root = source.parent/'ccp_icm_shared_single_991001_b5000_v1'
    original = checked_plan(s, original_root/'PLAN.json')
    shared = s.read(shared_root/'PLAN.json')
    if (original['sha256'] != panel['source_plan'] or s.canonical(shared) != panel['shared_single_plan']
            or original['settings'] != panel['settings'] or original['normalization'] != panel['normalization']
            or original['screen'] != panel['entries'] or panel['seed'] != 991001 or panel['ppo_update'] != 1):
        raise ValueError('Original panel provenance differs')
    st = panel['settings']
    if (tuple(st[k] for k in ('population', 'budget', 'scenario_count', 'pool_size', 'progress_denominator'))
            != (250, 5000, 100, 8, 50000) or st['confidence'] != .9 or st['m_start_generation'] != 10
            or st['screen_seeds'] != SEEDS or st['milestones'] != [400, 1000, 5000]
            or sorted(e['configuration'] for e in panel['entries']) != CONFIGS):
        raise ValueError('Unexpected search protocol')
    files = runtime_files(s, original, panel)
    models = {}
    for key, src in dict(shared=shared_root/'one_member', **{k: Path(v) for k, v in panel['models'].items()}).items():
        bundle = s.read(src/'BUNDLE.json')
        expected = shared['single_model'] if key == 'shared' else panel['hashes'][key]
        if (s.digest(src/'BUNDLE.json') != expected or len(bundle['checkpoints']) != 1
                or bundle['normalization'] != panel['normalization']):
            raise ValueError('Wrong single-member model: ' + key)
        name = bundle['checkpoints'][0]
        if Path(name).name != name or s.digest(src/name) != bundle['checkpoint_sha256'][name]:
            raise ValueError('Wrong model weights: ' + key)
        dest = root/'models'/key
        dest.mkdir(parents=True, exist_ok=True)
        for filename in ('BUNDLE.json', name):
            target = dest/filename
            if not target.exists():
                shutil.copyfile(src/filename, target)
            if s.digest(target) != s.digest(src/filename):
                raise ValueError('Copied model changed: ' + str(target))
            files[str(target)] = s.digest(target)
        models[key] = dict(path=str(dest), bundle_sha256=expected, checkpoint=name)
    reused = collect_old(s, source, panel, original_root, shared_root, models)
    print('[SECOND] verified 297 existing runs; model members=1; PPO u1 unchanged', flush=True)
    known_paths = {p.resolve() for p in source.parent.glob('*/instances/*.json')
                   if root not in p.resolve().parents}
    known_paths.add(Path(st['source_catalog_default']))
    s.freeze(root/'GENERATION.json', dict(GENERATION, source_plan_sha256=panel['sha256']))
    entries, catalog = generate_instances(s, root, panel, original, known_paths)
    files.update({str(p): s.digest(p) for p in [source/'PLAN.json', original_root/'PLAN.json', shared_root/'PLAN.json',
                  root/'GENERATION.json', catalog, *[root/'instances'/(e['instance_id']+'.json') for e in entries]]})
    plan = dict(source=str(source), source_plan_sha256=panel['sha256'], settings=st, normalization=panel['normalization'],
                original_entries=panel['entries'], entries=entries, models=models, reused=reused, files=files,
                known_instance_manifest_hashes={str(p): s.digest(p) for p in sorted(known_paths)},
                training_seed=991001, ppo_update=1, training_updates=0, new_searches=297, reused_searches=297,
                final_test_opened=False, purpose='Second development instance per S0-S8 configuration; fixed policies',
                environment=dict(host=socket.gethostname(), python=sys.version, platform=platform.platform()))
    plan['sha256'] = s.canonical(plan)
    s.freeze(root/'PLAN.json', plan)
    return plan


def task_for(s, plan, job):
    arm, name, seed = job
    model = Path(plan['models'][kind(arm)]['path']) if kind(arm) else None
    return s.scheduler.task(plan, name, seed, arm, 'screen', model, False)


def folder(root, job):
    arm, name, seed = job
    return root/'runs'/name/f'seed_{seed}'/arm


def validate_new(s, plan, job, row):
    arm, name, seed = job
    entry = next(e for e in plan['entries'] if e['instance_id'] == name)
    s.audit.validate_run(row, entry, seed, plan['settings'])
    task = task_for(s, plan, job)
    if (row['plan_sha256'] != plan['sha256'] or row['task_sha256'] != s.canonical(task)
            or row['model_bundle_sha256'] != task['bundle_sha256'] or row['arm'] != arm
            or row['stochastic'] or row['phase'] != 'screen'):
        raise ValueError('New result does not match its frozen task: ' + str(job))


def worker(s, root, plan, job):
    import search_b5000 as search
    arm, name, seed = job
    if arm not in ARMS or seed not in SEEDS or name not in {e['instance_id'] for e in plan['entries']}:
        raise ValueError('Unscheduled worker')
    dest = folder(root, job)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.with_suffix('.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if dest.exists():
            validate_new(s, plan, job, s.read(dest/'RUN_COMPLETE.json'))
            return
        attempt = Path(tempfile.mkdtemp(prefix=arm+'.attempt-', dir=dest.parent))
        if arm == previous.PPO:
            search.TerminalPolicy = s.p.PilotPolicy
        row = search.run_search(root/'instances'/(name+'.json'), plan['settings'], task_for(s, plan, job), attempt, plan['sha256'])
        validate_new(s, plan, job, row)
        attempt.rename(dest)


def report(s, root, plan, jobs, destination, include_old=False, preview=False):
    groups = {m: [] for m in ARMS}
    if include_old:
        for r in plan['reused']:
            if s.digest(Path(r['path'])) != r['sha256']:
                raise ValueError('Reused record changed: ' + r['path'])
            groups[r['arm']].append(s.read(Path(r['path'])))
    for job in jobs:
        row = s.read(folder(root, job)/'RUN_COMPLETE.json')
        validate_new(s, plan, job, row)
        groups[job[0]].append(row)
    # All 11 arms, all three seeds and all three snapshots enter each instance's bounds.
    for arm, rows in groups.items():
        if len(rows) != len(groups['BASE']):
            raise ValueError('Incomplete paired method set: ' + arm)
        for entry in {r['instance'] for r in rows}:
            if {r['seed'] for r in rows if r['instance'] == entry} != set(SEEDS):
                raise ValueError('Reporting requires all three seeds of each included instance')
    normalized, bounds = s.audit.union_normalize(groups)
    output, direct, endpoint = [], [], []
    for budget in (400, 1000, 5000):
        at = {m: [dict(r, metric=next(b['hv'] for b in r['budget_fronts'] if b['evaluation'] == budget)) for r in runs]
              for m, runs in normalized.items()}
        for arm, ref in [(m, 'BASE') for m in ARMS] + [(previous.PPO, previous.SHARED)]:
            # Vectorized, hierarchical resampling uses the same existing interval routine.
            baseline = {(r['instance'], r['seed']): r for r in at[ref]}
            pairs = []
            for r in at[arm]:
                b = baseline[r['instance'], r['seed']]
                s.p.check_pair(r, b)
                diff = r['metric'] - b['metric']
                pairs.append(dict(instance=r['instance'], configuration=r['configuration'], seed=r['seed'],
                                  own=r['metric'], reference=b['metric'], absolute=diff,
                                  percent=100*diff/b['metric'] if b['metric'] > 0 else None))
            lo, hi = s.p.hv.interval(pairs, 'percent', 3000)
            alo, ahi = s.p.hv.interval(pairs, 'absolute', 3000)
            row = dict(评价次数=budget, 方法=NAMES[arm], 参照=NAMES[ref], 配对数=len(pairs),
                       平均HV=s.p.equal_mean(pairs, 'own'),
                       平均配对提升百分比=s.p.equal_mean(pairs, 'percent') if all(p['percent'] is not None for p in pairs) else '',
                       提升区间下限=lo, 提升区间上限=hi, 平均HV差=s.p.equal_mean(pairs, 'absolute'),
                       HV差区间下限=alo, HV差区间上限=ahi, 胜='', 平='', 负='',
                       搜索平均秒='', 搜索标准差秒='', 搜索总秒='', 推理平均秒='', 推理标准差秒='')
            if budget == 5000:
                times = [r['optimisation_seconds'] for r in at[arm]]
                inference = [r['inference_seconds'] for r in at[arm]]
                row.update(胜=sum(p['absolute'] > 1e-12 for p in pairs) if arm != ref else '—',
                           平=sum(abs(p['absolute']) <= 1e-12 for p in pairs) if arm != ref else '—',
                           负=sum(p['absolute'] < -1e-12 for p in pairs) if arm != ref else '—',
                           搜索平均秒=mean(times), 搜索标准差秒=stdev(times), 搜索总秒=sum(times),
                           推理平均秒=mean(inference), 推理标准差秒=stdev(inference))
                endpoint += [dict(method=arm, reference_method=ref, **p) for p in pairs]
            (output if ref == 'BASE' else direct).append(row)
    s.write(destination/'INSTANCE_BOUNDS.json', bounds)
    s.p.csv_write(destination/'RESULTS.csv', output)
    s.p.csv_write(destination/'PPO_vs_SHARED.csv', direct)
    s.p.csv_write(destination/'ENDPOINT_PAIRS.csv', endpoint)
    s.write(destination/'REPORT.json', dict(completed=True, preview_only=preview, plan_sha256=plan['sha256'],
            pairs_per_method=len(groups['BASE']), configurations=sorted({r['configuration'] for r in groups['BASE']}),
            reference='BASE', normalization='Per instance across 11 methods, all three seeds and three budgets; ref=(1.1,1.1,1.1); no HV volume division',
            interval='95% hierarchical bootstrap; configurations equally weighted, resample instances then seeds within configuration; 3000 draws',
            endpoint_wtl_tolerance=1e-12, early_wtl_exported=False, training_updates=0))
    print('[SECOND] report ready:', destination/'RESULTS.csv', flush=True)


def execute(s, root, plan, workers):
    jobs = [(arm, e['instance_id'], seed) for e in plan['entries'] for seed in SEEDS for arm in ARMS]
    done, todo, running = {}, deque(), {}
    cost = {arm: mean(r['seconds'] for r in plan['reused'] if r['arm'] == arm) for arm in ARMS}
    for job in jobs:
        saved = folder(root, job)/'RUN_COMPLETE.json'
        if saved.exists():
            row = s.read(saved)
            validate_new(s, plan, job, row)
            done[job] = row
        else:
            todo.append(job)
    s.write(root/'COST_ESTIMATE.json', dict(previous_mean_seconds=cost, workers=workers, new_searches=297,
            remaining_searches=len(todo), estimated_search_hours=sum(cost[j[0]] for j in todo)/workers/3600,
            excludes='Process startup, instance variability, machine contention and reports'))
    print(f'[SECOND] {len(done)}/297 new searches complete; 297 old runs reused; workers={workers}; '
          f'estimated search hours={sum(cost[j[0]] for j in todo)/workers/3600:.2f}', flush=True)
    (root/'logs').mkdir(exist_ok=True)
    started, heartbeat, last_reported = time.monotonic(), 0, -1
    try:
        while todo or running:
            if STOP:
                raise InterruptedError(STOP)
            while todo and len(running) < workers:
                job = todo.popleft()
                arm, name, seed = job
                log = root/'logs'/f'{name}_{seed}_{arm}_{time.time_ns()}.log'
                with log.open('w') as stream:
                    process = subprocess.Popen([sys.executable, '-u', str(HERE/'run.py'), '--out', str(root),
                                                '--worker', arm, name, str(seed)], cwd=EXP, stdout=stream,
                                                stderr=subprocess.STDOUT, start_new_session=True)
                running[job] = (process, time.monotonic(), log)
                print(f'[SECOND] start {name}/{seed}/{arm} PID={process.pid} log={log}', flush=True)
            for job, (process, begin, log) in list(running.items()):
                code = process.poll()
                if code is None:
                    if time.monotonic()-begin > plan['settings']['task_timeout_seconds']:
                        raise TimeoutError('Task timeout; preserve attempts: ' + str(log))
                    continue
                if code:
                    raise RuntimeError(f'Worker exit {code}: {log}')
                row = s.read(folder(root, job)/'RUN_COMPLETE.json')
                validate_new(s, plan, job, row)
                for other, saved in done.items():
                    if other[1:] == job[1:] and saved['scenario_digest'] != row['scenario_digest']:
                        raise ValueError('CCP realizations differ across methods')
                done[job] = row
                del running[job]
                print(f'[SECOND] complete {len(done)}/297', *job, f"seconds={row['optimisation_seconds']:.1f}", flush=True)
            now = time.monotonic()
            if now-heartbeat >= 30:
                remaining = [j for j in jobs if j not in done]
                status = dict(completed=False, stage='search', done=len(done), total=297, reused_searches=297,
                              workers=workers, elapsed_seconds=int(now-started),
                              rough_remaining_hours=sum(cost[j[0]] for j in remaining)/workers/3600,
                              running=[dict(arm=j[0], instance=j[1], seed=j[2], pid=p.pid, seconds=int(now-t), log=str(l))
                                       for j, (p, t, l) in running.items()])
                s.write(root/'STATUS.json', status)
                print(f'[SECOND] heartbeat {len(done)}/297 running={len(running)}', flush=True)
                heartbeat = now
            # Preview only complete instances: all 3 seeds and 11 methods share frozen bounds.
            complete_names = {e['instance_id'] for e in plan['entries'] if
                              all((a, e['instance_id'], seed) in done for seed in SEEDS for a in ARMS)}
            if len(complete_names) > last_reported and complete_names:
                report(s, root, plan, [j for j in jobs if j[1] in complete_names], root/'preview', preview=True)
                last_reported = len(complete_names)
            if running:
                time.sleep(1)
    finally:
        # Only child Popen handles created by this invocation are terminated on interruption.
        for p, _, _ in running.values():
            if p.poll() is None:
                p.terminate()
        for p, _, _ in running.values():
            try:
                p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                p.kill()
                p.wait()
    report(s, root, plan, jobs, root/'new_instance_only')
    report(s, root, plan, jobs, root/'combined54', include_old=True)
    s.write(root/'STATUS.json', dict(completed=True, stage='all', new_searches=297, reused_searches=297,
            total_searches=594, instances_per_configuration=2, pairs_per_method=54, methods=11,
            training_updates=0, training_seed=991001, ppo_update=1, workers=workers,
            wall_seconds_this_invocation=time.monotonic()-started, final_test_opened=False))
    print('[SECOND] COMPLETE', root/'combined54/RESULTS.csv', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, default=DIAG/'ccp_icm_single_member_second_instance_b5000_v1')
    p.add_argument('--source', type=Path, default=DIAG/'ccp_icm_single_member_panel_b5000_v1')
    p.add_argument('--workers', type=int, choices=(1, 2), default=2)
    p.add_argument('--prepare-only', action='store_true')
    p.add_argument('--worker', nargs=3, help=argparse.SUPPRESS)
    a = p.parse_args()
    root, source = a.out.resolve(), a.source.resolve()
    if root == source or root in source.parents or source in root.parents:
        raise ValueError('Use a separate output directory; historical files are read-only')
    s = previous.previous.load_module(EXP)
    if a.worker:
        plan = checked_plan(s, root/'PLAN.json')
        s.p.verify(plan['files'])
        worker(s, root, plan, (a.worker[0], a.worker[1], int(a.worker[2])))
        return
    root.mkdir(parents=True, exist_ok=True)
    def stop(signum, frame):
        global STOP
        STOP = f'Received signal {signum}'
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, stop)
    with (root/'.launch.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            print('[SECOND] checking frozen single-member source, models and raw records', flush=True)
            plan = prepare(s, root, source)
            if a.prepare_only:
                print('[SECOND] READY; 9 new instances, no objective evaluations', flush=True)
                return
            execute(s, root, plan, a.workers)
        except Exception as exc:
            s.write(root/'STATUS.json', dict(completed=False, paused=isinstance(exc, InterruptedError),
                    stage='stopped', reason=str(exc), resume='Repeat the same command; completed runs reused'))
            raise


if __name__ == '__main__':
    main()
