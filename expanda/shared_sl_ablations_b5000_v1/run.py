#!/usr/bin/env python3
"""Two frozen single-member Shared SL ablations on the existing 54 development pairs."""
import argparse
from collections import Counter
import copy
import fcntl
import os
from pathlib import Path
import shutil
import signal
from statistics import mean, stdev
import sys
import time

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
EXP = HERE.parent
DIAG = EXP/'transformer_ppo_runs/diagnostics'
sys.path.insert(0, str(EXP))
from icm_single_member_b5000 import load_module
from policies import RandomReplacement, policy_seed, zero_threshold_policy

s = load_module(EXP)
p = s.p
read, write, digest, canonical, freeze = s.read, s.write, s.digest, s.canonical, s.freeze
SHARED, RULE = 'ICM_SHARED_SUPERVISED', 'ICM_HEURISTIC'
RANDOM, ZERO = 'ICM_RANDOM_MATCHED', 'ICM_SHARED_ZERO_THRESHOLD'
NEW = (RANDOM, ZERO)
CONTROLS = ('BASE', RULE, SHARED)
NAMES = dict(BASE='Base', ICM_HEURISTIC='规则 ICM',
             ICM_SHARED_SUPERVISED='共享监督 ICM（单模型）',
             ICM_RANDOM_MATCHED='随机替换 ICM（匹配设定概率）',
             ICM_SHARED_ZERO_THRESHOLD='共享监督 ICM（阈值为零）')
CONFIGS = [f'S{i}' for i in range(9)]
SEEDS = [993001, 993002, 993003]
BUDGETS = (400, 1000, 5000)
VOLUME = 1.331
EXPECTED_COUNTS = {'I': (4752, 4661, 91), 'C': (62404, 27032, 35372), 'M': (21875, 7189, 14686)}


def checked(path):
    obj = read(path)
    if canonical({k: v for k, v in obj.items() if k != 'sha256'}) != obj['sha256']:
        raise ValueError('Plan checksum changed: '+str(path))
    return obj


def copy_frozen(src, dst, files):
    files[str(src)] = digest(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists():
        tmp = dst.with_suffix(dst.suffix+'.copying')
        shutil.copyfile(src, tmp)
        tmp.replace(dst)
    if digest(dst) != files[str(src)]:
        raise ValueError('Frozen copy changed: '+str(dst))
    files[str(dst)] = digest(dst)


def pairs_of(rows):
    return {(r['instance'], r['seed']) for r in rows}


def validate_panel(rows, entries, settings):
    expected = {(e['instance_id'], seed) for e in entries for seed in SEEDS}
    if len(rows) != 54 or pairs_of(rows) != expected:
        raise ValueError('Expected the exact 18-instance / 54-pair panel')
    by = {e['instance_id']: e for e in entries}
    for row in rows:
        s.audit.validate_run(row, by[row['instance']], row['seed'], settings)


def prepare(root, source):
    if (root/'PLAN.json').exists():
        plan = checked(root/'PLAN.json')
        if plan['source'] != str(source):
            raise ValueError('Resume source differs')
        p.verify(plan['files'])
        return plan
    source_plan = checked(source/'PLAN.json')
    status = read(source/'STATUS.json')
    if not status.get('completed') or source_plan['model_members'] != 1:
        raise ValueError('Expected completed single-member S0-S8 source')
    settings = copy.deepcopy(source_plan['settings'])
    if (tuple(settings[k] for k in ('population', 'budget', 'scenario_count', 'pool_size',
                                    'progress_denominator', 'm_start_generation')) != (250, 5000, 100, 8, 50000, 10)
            or settings['confidence'] != .9 or settings['screen_seeds'] != SEEDS
            or settings['milestones'] != list(BUDGETS)):
        raise ValueError('Search settings differ from the frozen experiment')
    entries = source_plan['development']
    if (len(entries) != 18 or len({e['instance_id'] for e in entries}) != 18
            or Counter(e['configuration'] for e in entries) != {c: 2 for c in CONFIGS}
            or any(e['development_role'] != 'screen' for e in entries)):
        raise ValueError('Expected two existing development instances per S0-S8 configuration')
    files = {}
    # Verify frozen transport/runtime dependencies, without reading unrelated training traces.
    for filename, expected in source_plan['files'].items():
        path = Path(filename)
        if path.suffix in ('.py', '.csv', '.xlsx') and EXP in path.parents:
            if digest(path) != expected:
                raise ValueError('Historical runtime/input changed: '+filename)
            files[filename] = expected
    for path in (source/'PLAN.json', source/'STATUS.json', HERE/'run.py', HERE/'policies.py', HERE/'run.sh'):
        files[str(path)] = digest(path)
    origin = Path(source_plan['shared_model'])
    bundle = read(origin/'BUNDLE.json')
    if (bundle['checkpoints'] != ['seed_991001_epoch_20.pt']
            or bundle['normalization'] != source_plan['normalization']):
        raise ValueError('Wrong shared member or scaling')
    for c, threshold in zip('ICM', (.05, 0., 0.)):
        gate = bundle['selected']['gates'][c]
        if gate['threshold_scaled'] != threshold or gate['disabled']:
            raise ValueError('The established gates are I=.05, C=0, M=0, all enabled')
    p.verify({str(origin/name): value for name, value in bundle['checkpoint_sha256'].items()})
    model = root/'shared_model'
    for name in ['BUNDLE.json']+bundle['checkpoints']:
        if digest(origin/name) != source_plan['files'][str(origin/name)]:
            raise ValueError('Shared artifact changed since source run')
        copy_frozen(origin/name, model/name, files)
    for entry in entries:
        src = source/'instances'/(entry['instance_id']+'.json')
        if read(src) != entry:
            raise ValueError('Instance changed')
        copy_frozen(src, root/'instances'/src.name, files)
    groups = {}
    # Retain the complete historical 11-method normalization pool; only 3 controls are displayed.
    for arm, names in source_plan['source_records'].items():
        rows = []
        for name in names:
            path = Path(name)
            if digest(path) != source_plan['files'][name]:
                raise ValueError('Source run changed: '+name)
            files[name] = source_plan['files'][name]
            row = read(path)
            if row['arm'] != arm or row['stochastic']:
                raise ValueError('Wrong reference method or deployment mode')
            rows.append(row)
        validate_panel(rows, entries, settings)
        groups[arm] = rows
    if len(groups) != 11 or not set(CONTROLS).issubset(groups):
        raise ValueError('Missing historical panel')
    controls = {(r['instance'], r['seed']): r for r in groups['BASE']}
    for rows in groups.values():
        for row in rows:
            p.check_pair(row, controls[row['instance'], row['seed']])
    if any(r['model_bundle_sha256'] != digest(origin/'BUNDLE.json') for r in groups[SHARED]):
        raise ValueError('Reused shared results do not use the frozen single member')
    counts, probabilities = {}, {}
    for c in 'ICM':
        count = {k: sum(r['selection'][c][k] for r in groups[SHARED])
                 for k in ('calls', 'switches', 'fallbacks')}
        if tuple(count[k] for k in ('calls', 'switches', 'fallbacks')) != EXPECTED_COUNTS[c]:
            raise ValueError('Source replacement counts differ from the confirmed 54-run audit: '+c)
        counts[c] = count
        probabilities[c] = count['switches']/count['calls']
    settings['invocation_seconds'] = 172800
    settings['task_timeout_seconds'] = 14400
    plan = dict(protocol='shared-sl-ablations-b5000-v1', source=str(source),
        source_plan_sha256=source_plan['sha256'], entries=entries, settings=settings,
        normalization=source_plan['normalization'], shared_model=str(model), files=files,
        source_records=source_plan['source_records'], replacement_counts=counts,
        replacement_probabilities=probabilities, original_gates=bundle['selected']['gates'],
        expected_scenarios={f"{r['instance']}/{r['seed']}": r['scenario_digest'] for r in groups['BASE']},
        methods=list(CONTROLS)+list(NEW), new_methods=list(NEW), new_searches=108,
        reused_comparison_searches=162, historical_front_pool_searches=594,
        model_members=1, training_member=991001, training_updates=0, hv_divisor=VOLUME,
        rule='Random: fixed per-component Bernoulli probability, uniform legal non-rule choice; private RNG. Zero threshold: greedy over legal candidates including rule at score zero; ties keep rule.',
        qualification='Development ablation: random probabilities fixed from existing 54 development runs; no outcome-based retuning. Probabilities match aggregate frequencies, not exact counts or state-conditioned intervention timing. No causal isolation of ranking alone; no independent-test claim.')
    plan['sha256'] = canonical(plan)
    freeze(root/'PLAN.json', plan)
    write(root/'FROZEN_DECISIONS.json', dict(probabilities=probabilities, source_counts=counts,
        source_gates=bundle['selected']['gates'], zero_threshold_gates={c: 0. for c in 'ICM'},
        direct_threshold_changes=['I'], matching='expected probability, not exact realized replacement rate'))
    return plan


def make_task(plan, entry, seed, variant):
    task = p.scheduler.task(plan, entry['instance_id'], seed, SHARED, 'screen', Path(plan['shared_model']))
    task.update(ablation=variant, ablation_seed=policy_seed(entry['instance_id'], seed))
    return task


def jobs_for(root, plan, entries=None):
    return [(make_task(plan, e, seed, variant),
             root/'runs'/e['instance_id']/f'seed_{seed}'/variant)
            for e in (entries or plan['entries']) for seed in SEEDS for variant in NEW]


def validate_new(row, task, plan):
    entry = next(e for e in plan['entries'] if e['instance_id'] == task['instance'])
    s.audit.validate_run(row, entry, task['seed'], plan['settings'],
                         plan['expected_scenarios'][f"{task['instance']}/{task['seed']}"])
    if (row['plan_sha256'] != plan['sha256'] or row['task_sha256'] != canonical(task)
            or row['arm'] != task['ablation'] or row['ablation'] != task['ablation']
            or row['model_bundle_sha256'] != task['bundle_sha256']):
        raise ValueError('Ablation result identity mismatch')
    for c in 'ICM':
        v = row['selection'][c]
        if v['calls'] != v['switches'] + v['fallbacks']:
            raise ValueError('Selection counters disagree')
    if task['ablation'] == RANDOM:
        for c in 'ICM':
            v, observed = row['random_selection'][c], row['selection'][c]
            if v['calls'] != observed['calls'] or v['switches'] != observed['switches']:
                raise ValueError('Random-policy and controller counters disagree')


def worker(root, plan, task, folder):
    if (task, folder) not in jobs_for(root, plan):
        raise ValueError('Task is outside the 108 frozen ablation searches')
    p.verify(plan['files'])
    import search_b5000 as search
    original, original_write = search.SharedPolicy, search.write_json
    holder = []
    def factory(model):
        if task['ablation'] == RANDOM:
            policy = RandomReplacement(plan['replacement_probabilities'], task['ablation_seed'])
        else:
            policy = zero_threshold_policy(original, model)
        holder.append(policy)
        return policy
    def defer_completion(path, value):
        # Publish completion only after the ablation identity and counters are validated.
        if Path(path).resolve() != (folder/'RUN_COMPLETE.json').resolve():
            return original_write(path, value)
    search.SharedPolicy, search.write_json = factory, defer_completion
    try:
        row = search.run_search(root/'instances'/(task['instance']+'.json'), plan['settings'],
                                task, folder, plan['sha256'])
    finally:
        search.SharedPolicy, search.write_json = original, original_write
    row.update(arm=task['ablation'], ablation=task['ablation'],
        engine_arm=SHARED, stochastic=(task['ablation'] == RANDOM),
        ablation_seed=task['ablation_seed'], training_updates=0,
        selection_time_includes_model_forward=(task['ablation'] == ZERO))
    if task['ablation'] == RANDOM:
        row['random_selection'] = holder[0].counts
        row['model_forward_calls'] = 0
        row['replacement_probabilities'] = plan['replacement_probabilities']
    else:
        row['effective_gates'] = holder[0].bundle['selected']['gates']
    validate_new(row, task, plan)
    write(folder/'RUN_COMPLETE.json', row)


def statistics(own, refs, budget, arm, reference):
    lookup = {(r['instance'], r['seed']): r for r in refs}
    pairs = []
    for r in own:
        b = lookup[r['instance'], r['seed']]
        p.check_pair(r, b)
        a = next(x['hv'] for x in r['budget_fronts'] if x['evaluation'] == budget)
        v = next(x['hv'] for x in b['budget_fronts'] if x['evaluation'] == budget)
        pairs.append(dict(instance=r['instance'], configuration=r['configuration'], seed=r['seed'],
                          own=a/VOLUME, reference=v/VOLUME, absolute=(a-v)/VOLUME,
                          percent=100*(a-v)/v if v > 0 else None))
    lo, hi = p.hv.interval(pairs, 'percent', 3000)
    alo, ahi = p.hv.interval(pairs, 'absolute', 3000)
    tol = 1e-12/VOLUME
    times = [next(x for x in r['budget_fronts'] if x['evaluation'] == budget).get('controller_seconds') for r in own]
    endpoint = [r['optimisation_seconds'] for r in own]
    selection = [r['inference_seconds'] for r in own]
    row = dict(评价次数=budget, 方法=NAMES[arm], 参照=NAMES[reference], 配对数=len(pairs),
        平均HV=p.equal_mean(pairs, 'own'), 平均配对提升百分比=p.equal_mean(pairs, 'percent') if all(v['percent'] is not None for v in pairs) else None,
        提升95区间下限=lo, 提升95区间上限=hi, 平均HV差=p.equal_mean(pairs, 'absolute'), HV差95区间下限=alo, HV差95区间上限=ahi,
        胜=sum(v['absolute'] > tol for v in pairs) if arm != reference else '—',
        平=sum(abs(v['absolute']) <= tol for v in pairs) if arm != reference else '—',
        负=sum(v['absolute'] < -tol for v in pairs) if arm != reference else '—',
        节点累计平均秒=mean(times) if all(v is not None for v in times) else None,
        节点累计标准差秒=stdev(times) if len(times)>1 and all(v is not None for v in times) else None,
        B5000搜索平均秒=mean(endpoint) if budget == 5000 else None,
        B5000搜索标准差秒=stdev(endpoint) if budget == 5000 else None,
        B5000候选选择平均秒=mean(selection) if budget == 5000 else None)
    return row, pairs


def report(root, plan, new_rows, destination, instances):
    expected = {(name, seed) for name in instances for seed in SEEDS}
    old = {arm: [row for path in paths if (row := read(Path(path)))['instance'] in instances]
           for arm, paths in plan['source_records'].items()}
    groups = dict(old)
    for arm in NEW:
        groups[arm] = [r for r in new_rows if r['arm'] == arm and r['instance'] in instances]
    for arm, rows in groups.items():
        if len(rows) != len(expected) or pairs_of(rows) != expected:
            raise ValueError('Incomplete report group: '+arm)
    normalized, bounds = s.audit.union_normalize(groups)
    destination.mkdir(parents=True, exist_ok=True)
    comparisons = [(a, 'BASE') for a in NAMES] + [(SHARED, RANDOM), (SHARED, ZERO), (RANDOM, RULE), (ZERO, RULE)]
    summary, direct, details, by_config = [], [], [], []
    for budget in BUDGETS:
        for arm, ref in comparisons:
            row, pairs = statistics(normalized[arm], normalized[ref], budget, arm, ref)
            (summary if ref == 'BASE' else direct).append(row)
            details.extend(dict(method=arm, reference_method=ref, evaluation=budget, **v) for v in pairs)
            for cfg in CONFIGS:
                own = [r for r in normalized[arm] if r['configuration'] == cfg]
                baseline = [r for r in normalized[ref] if r['configuration'] == cfg]
                if own:
                    row, _ = statistics(own, baseline, budget, arm, ref)
                    by_config.append(dict(配置=cfg, **row))
    rates = []
    for arm in (RULE, SHARED, RANDOM, ZERO):
        for c in 'ICM':
            rows = groups[arm]
            count = {k: sum(r['selection'][c][k] for r in rows) for k in ('calls', 'switches', 'fallbacks', 'unsupported')}
            rates.append(dict(method=arm, component=c, runs=len(rows), **count,
                actual_switch_percent=100*count['switches']/count['calls'] if count['calls'] else None,
                target_switch_percent=100*plan['replacement_probabilities'][c] if arm == RANDOM else None,
                no_legal_alternative=sum(r['random_selection'][c]['no_legal_alternative'] for r in rows) if arm == RANDOM else None,
                blocked_request=sum(r['random_selection'][c]['blocked_request'] for r in rows) if arm == RANDOM else None))
    for name, rows in [('RESULTS.csv', summary), ('DIRECT_COMPARISONS.csv', direct),
                       ('PER_PAIR.csv', details), ('PER_CONFIGURATION.csv', by_config), ('REPLACEMENT_RATES.csv', rates)]:
        p.csv_write(destination/name, rows)
    write(destination/'INSTANCE_BOUNDS.json', bounds)
    write(destination/'REPORT.json', dict(completed=True, preview=(len(instances) != 18),
        pairs_per_method=len(expected), plan_sha256=plan['sha256'], primary_reference='BASE',
        hv_divisor=VOLUME, hv_already_divided=True, hv_reference=[1.1]*3,
        normalization='Per instance union of 11 historical methods plus two ablations, all seeds and all three budgets; old reports untouched.',
        intervals='95% paired hierarchical bootstrap, instances then seeds within fixed equally weighted configurations, 3000 draws.',
        wtl_tolerance=1e-12/VOLUME, qualification=plan['qualification'],
        timing='Historical controls reused. Controller cumulative node times and full B5000 search times reported separately; random selection uses no neural forward but retains candidate feature construction.'))
    print('[ABLATION] REPORT', destination, flush=True)


def execute(args):
    root, source = args.out.resolve(), args.source.resolve()
    if root == source or root in source.parents or source in root.parents:
        raise ValueError('A separate output directory is required')
    root.mkdir(parents=True, exist_ok=True)
    p.torch.set_num_threads(1)
    with (root/'.pipeline.lock').open('a+') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit('This ablation is already running')
        def stop(signum, frame):
            raise p.scheduler.Paused(f'Received signal {signum}')
        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)
        saved_scheduler = (p.scheduler.HERE2, p.scheduler.valid_completion, p.scheduler.write_json)
        try:
            plan = prepare(root, source)
            jobs = jobs_for(root, plan)
            cached = [(t, p.scheduler.valid_completion(f, t, plan)) for t, f in jobs]
            for task, row in cached:
                if row is not None:
                    validate_new(row, task, plan)
            done = sum(row is not None for task, row in cached)
            print(f'[ABLATION] {done}/108 complete; reuse 162 control searches; workers={args.workers}', flush=True)
            print('[ABLATION] fixed replacement probabilities:', plan['replacement_probabilities'], flush=True)
            if args.prepare_only:
                write(root/'STATUS.json', dict(completed=False, stage='prepared', done=done, total=108, training_updates=0))
                return
            started = time.monotonic()
            p.scheduler.HERE2 = HERE
            scheduler_write = p.scheduler.write_json
            scheduler_completion = p.scheduler.valid_completion
            def audited_completion(folder, task, frozen_plan):
                row = scheduler_completion(folder, task, frozen_plan)
                if row is not None:
                    validate_new(row, task, frozen_plan)
                return row
            def global_status(path, state):
                if Path(path) == root/'STATUS.json' and 'running' in state:
                    state = dict(state)
                    state['stage_done'], state['stage_total'] = state['done'], state['total']
                    state['done'] = sum(audited_completion(f, t, plan) is not None for t, f in jobs)
                    state['total'] = 108
                    state['training_updates'] = 0
                    by_hash = {canonical(t): t['ablation'] for t, f in jobs}
                    state['running'] = [dict(v, ablation=by_hash[Path(v['log']).name.split('_')[0]])
                                        for v in state['running']]
                    print(f"[ABLATION] total {state['done']}/108; running={len(state['running'])}", flush=True)
                return scheduler_write(path, state)
            p.scheduler.valid_completion, p.scheduler.write_json = audited_completion, global_status
            all_rows = []
            # Finish both ablations on one entire configuration before moving on.
            # Preview always contains both methods and three seeds per instance.
            for cfg in CONFIGS:
                entries = [e for e in plan['entries'] if e['configuration'] == cfg]
                cfg_jobs = jobs_for(root, plan, entries)
                rows = p.scheduler.execute_jobs(root, plan, cfg_jobs, args.workers, f'ablation/{cfg}')
                for task, folder in cfg_jobs:
                    validate_new(read(folder/'RUN_COMPLETE.json'), task, plan)
                all_rows.extend(rows)
                names = {r['instance'] for r in all_rows}
                report(root, plan, all_rows, root/'preview', names)
                write(root/'STATUS.json', dict(completed=False, stage=f'{cfg}_complete',
                    done=len(all_rows), total=108, reused_comparison_searches=162, training_updates=0))
            p.verify(plan['files'])
            report(root, plan, all_rows, root/'results', {e['instance_id'] for e in plan['entries']})
            write(root/'STATUS.json', dict(completed=True, stage='all', done=108, total=108,
                new_searches=108, reused_comparison_searches=162, training_updates=0,
                original_results_modified=False, hv_already_divided=True, hv_divisor=VOLUME,
                wall_seconds_this_invocation=time.monotonic()-started))
            print('[ABLATION] COMPLETE', root/'results', flush=True)
        except p.scheduler.Paused as exc:
            write(root/'STATUS.json', dict(completed=False, paused=True, reason=str(exc), resume='Repeat the same command'))
            print('[ABLATION] PAUSED', exc, flush=True)
        except Exception as exc:
            write(root/'STATUS.json', dict(completed=False, failed=True, error=str(exc)))
            raise
        finally:
            p.scheduler.HERE2, p.scheduler.valid_completion, p.scheduler.write_json = saved_scheduler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=DIAG/'ccp_single_ppo_s0_s8_b5000_v1')
    parser.add_argument('--out', type=Path, default=DIAG/'ccp_shared_sl_ablations_b5000_v1')
    parser.add_argument('--workers', type=int, choices=(1, 2), default=2)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--task', type=Path)
    parser.add_argument('--run-out', type=Path)
    args = parser.parse_args()
    if args.worker:
        worker(args.out.resolve(), checked(args.out/'PLAN.json'), read(args.task), args.run_out.resolve())
    else:
        execute(args)


if __name__ == '__main__':
    main()
