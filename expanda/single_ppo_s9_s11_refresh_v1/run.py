#!/usr/bin/env python3
"""Evaluate the S0--S8-selected single PPO on the existing S9--S11 panel."""
import argparse
import copy
import fcntl
import importlib.util
from pathlib import Path
import shutil
import signal
from statistics import mean, stdev
import sys
import time

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
EXP = HERE.parent
DIAG = EXP / 'transformer_ppo_runs/diagnostics'


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    obj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj


large = module('refresh_large', EXP/'single_s9_s11_b5000_v1/run.py')
train = module('refresh_training', EXP/'single_ppo_s0_s8_b5000_v1/run.py')
engine, p, s = large.engine, large.p, large.s
read, write, digest, canonical, freeze = large.read, large.write, large.digest, large.canonical, large.freeze
SHARED, PPO = 'ICM_SHARED_SUPERVISED', 'ICM_TERMINAL_PPO'
REPORT_ARMS = ('BASE', SHARED, PPO)
DIVISOR = 1.331


def pair_keys(rows):
    return {(r['instance'], r['seed']) for r in rows}


def check_shared(old_folder, new_folder, norm):
    old = train.single(old_folder, norm)
    new = train.single(new_folder, norm)
    for key in ('selected', 'scalers', 'dimensions', 'settings', 'feature_version',
                'checkpoints', 'checkpoint_sha256', 'normalization'):
        if old.get(key) != new.get(key):
            raise ValueError('Cannot reuse shared policy: changed '+key)
    if old['checkpoints'] != ['seed_991001_epoch_20.pt']:
        raise ValueError('Expected the frozen shared member 991001, epoch 20')
    return old


def frozen_copy(src, dst, files):
    files[str(src)] = digest(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists():
        temporary = dst.with_suffix(dst.suffix+'.copying')
        shutil.copyfile(src, temporary)
        temporary.replace(dst)
    if digest(dst) != files[str(src)]:
        raise ValueError('Frozen copy differs: '+str(dst))
    files[str(dst)] = digest(dst)


def prepare(root, large_root, training_root):
    if (root/'PLAN.json').exists():
        plan = large.checked(root/'PLAN.json')
        if (plan['large_source'], plan['training_source']) != (str(large_root), str(training_root)):
            raise ValueError('Resume sources changed')
        p.verify(plan['files'])
        return plan
    old = large.checked(large_root/'PLAN.json')
    fitted = train.checked(training_root/'PLAN.json')
    if not all(read(folder/'STATUS.json').get('completed') for folder in (large_root, training_root)):
        raise ValueError('Both source experiments must be complete')
    large.validate_protocol(old['protocol'])
    train.validate_protocol(fitted['pilot'])
    selection = read(training_root/'SELECTION.json')
    scores = read(training_root/'TRAINING_PROGRESS.json')['validation']
    update = selection['selected_update']
    if update != train.select_checkpoint(scores):
        raise ValueError('Checkpoint does not match the recorded S0-S8 validation selection')
    if update == 0:
        raise ValueError('Validation selected unchanged u0: reuse shared results; no new PPO searches are needed')
    if fitted['model_members'] != 1 or old['normalization'] != fitted['normalization']:
        raise ValueError('Single-member / frozen training normalization mismatch')
    for key in ('population', 'budget', 'scenario_count', 'confidence', 'pool_size',
                'progress_denominator', 'm_start_generation', 'milestones'):
        if old['settings'][key] != fitted['settings'][key]:
            raise ValueError('Deployment setting differs: '+key)
    shared_folder = Path(old['models'][SHARED])
    check_shared(shared_folder, Path(fitted['shared_model']), old['normalization'])
    selected_folder = train.model_path(training_root, update)
    trained = train.single(selected_folder, old['normalization'])
    if (digest(selected_folder/'BUNDLE.json') != selection['bundle_sha256'] or
            trained.get('fine_tuning_provenance', {}).get('plan_sha256') != fitted['sha256']):
        raise ValueError('Selected PPO weights belong to a different training run')
    origin = train.single(Path(fitted['shared_model']))
    for key in ('selected', 'scalers', 'dimensions', 'feature_version'):
        if trained.get(key) != origin.get(key):
            raise ValueError('PPO gate/input definition changed: '+key)
    if trained.get('pilot', {}).get('configurations') != [f'S{i}' for i in range(9)]:
        raise ValueError('Selected PPO bundle does not document S0-S8 training')

    files = {}
    # Preserve runtime identity with both source experiments. Historical reports
    # and search outputs are never edited, and only the needed records are reused.
    for manifest in (old, fitted):
        for filename, expected in manifest['files'].items():
            path = Path(filename)
            if path.suffix in ('.py', '.csv', '.xlsx') and EXP in path.parents:
                if digest(path) != expected:
                    raise ValueError('Source runtime/input changed: '+filename)
                if filename in files and files[filename] != expected:
                    raise ValueError('Source runtimes disagree: '+filename)
                files[filename] = expected
    for path in (large_root/'PLAN.json', large_root/'STATUS.json', training_root/'PLAN.json',
                 training_root/'STATUS.json', training_root/'SELECTION.json',
                 training_root/'TRAINING_PROGRESS.json', HERE/'run.py'):
        files[str(path)] = digest(path)
    for folder in (shared_folder, Path(fitted['shared_model'])):
        bundle = read(folder/'BUNDLE.json')
        for name in ['BUNDLE.json']+bundle['checkpoints']:
            files[str(folder/name)] = digest(folder/name)
    for name in ['BUNDLE.json']+trained['checkpoints']:
        frozen_copy(selected_folder/name, root/'model'/name, files)
    entries = old['entries']
    if len(entries) != 9 or any(sum(e['configuration'] == cfg for e in entries) != 3 for cfg in large.CONFIGS):
        raise ValueError('Expected the existing 9 large instances')
    for entry in entries:
        src = large_root/'instances'/(entry['instance_id']+'.json')
        if read(src) != entry:
            raise ValueError('Instance manifest changed')
        frozen_copy(src, root/'instances'/src.name, files)
    sources = {arm: [] for arm in large.ARMS}
    for entry in entries:
        for seed in old['protocol']['algorithm_seeds']:
            paired = []
            for arm in large.ARMS:
                path = large_root/'runs'/entry['instance_id']/f'seed_{seed}'/arm/'RUN_COMPLETE.json'
                row = read(path)
                engine.validate_run(row, old)
                sources[arm].append(str(path))
                files[str(path)] = digest(path)
                paired.append(row)
            for row in paired[1:]:
                p.check_pair(row, paired[0])
    settings = copy.deepcopy(old['settings'])
    settings['invocation_seconds'] = 172800
    settings['task_timeout_seconds'] = 14400
    plan = dict(large_source=str(large_root), training_source=str(training_root), files=files,
        entries=entries, settings=settings, normalization=old['normalization'],
        protocol=old['protocol'], models={SHARED: str(shared_folder), PPO: str(root/'model')},
        selected_update=update, selection=selection, source_records=sources,
        maximum_new_searches=27, model_members=1, training_member=991001,
        training_configurations=[f'S{i}' for i in range(9)], new_training_updates=0,
        checkpoint_selection_on_large_results=False, historical_large_results_already_inspected=True,
        hv_divisor=DIVISOR, hv_reference=[1.1]*3,
        normalization_reporting='Per-instance union of saved four-method fronts and new PPO; all seeds and budgets; HV / 1.331.',
        qualification='Fixed checkpoint transfer to previously inspected S9-S11 instances. No new training or large-instance selection. Base/shared reused; Simple/old PPO used only to retain the historical normalization pool.')
    plan['sha256'] = canonical(plan)
    freeze(root/'PLAN.json', plan)
    return plan


def jobs_for(root, plan):
    return [(engine.task(plan, entry, seed, PPO),
             root/'runs'/entry['instance_id']/f'seed_{seed}'/PPO)
            for entry in plan['entries'] for seed in plan['protocol']['algorithm_seeds']]


def worker(root, plan, task, folder):
    if (task, folder) not in jobs_for(root, plan):
        raise ValueError('Only the 27 frozen PPO tasks are authorized')
    train.single(Path(plan['models'][PPO]), plan['normalization'])
    engine.worker(root, plan, task, folder)


def scale_summary(row):
    result = dict(row)
    for field in ('平均HV', '平均HV差', 'HV差区间下限', 'HV差区间上限'):
        if result[field] is not None:
            result[field] /= DIVISOR
    return result


def add_node_times(row, runs, budget):
    times = [next(point for point in r['budget_fronts'] if point['evaluation'] == budget).get('controller_seconds')
             for r in runs]
    row['节点累计平均秒'] = mean(times) if all(t is not None for t in times) else None
    row['节点累计标准差秒'] = stdev(times) if len(times)>1 and all(t is not None for t in times) else None
    return row


def report(root, plan, new_rows):
    expected = {(e['instance_id'], seed) for e in plan['entries'] for seed in plan['protocol']['algorithm_seeds']}
    if len(new_rows) != 27 or pair_keys(new_rows) != expected:
        raise ValueError('Incomplete or duplicate PPO panel')
    old = {arm: [read(Path(path)) for path in paths] for arm, paths in plan['source_records'].items()}
    for arm, runs in old.items():
        if len(runs) != 27 or pair_keys(runs) != expected:
            raise ValueError('Incomplete source panel: '+arm)
    reference = {(r['instance'], r['seed']): r for r in old['BASE']}
    for row in new_rows:
        engine.validate_run(row, plan)
        p.check_pair(row, reference[row['instance'], row['seed']])
    pool = {('HISTORICAL_PPO' if arm == PPO else arm): rows for arm, rows in old.items()}
    pool[PPO] = new_rows
    normalized, bounds = s.audit.union_normalize(pool)
    large.NAMES[PPO] = f'ICM 共享监督+PPO微调（S0-S8单模型 u{plan["selected_update"]}）'
    for mode, source in [('results_union', normalized), ('results_frozen', pool)]:
        folder = root/mode
        folder.mkdir(parents=True, exist_ok=True)
        summary, direct, configurations, details = [], [], [], []
        for budget in large.BUDGETS:
            comparisons = [(arm, 'BASE') for arm in REPORT_ARMS]+[(PPO, SHARED)]
            for arm, ref in comparisons:
                row, pairs = large.statistics(source[arm], source[ref], budget, arm, ref)
                add_node_times(row, source[arm], budget)
                (summary if ref == 'BASE' else direct).append(scale_summary(row))
                for pair in pairs:
                    scaled = dict(pair)
                    for key in ('own', 'reference', 'absolute'):
                        scaled[key] /= DIVISOR
                    details.append(scaled)
                for cfg in large.CONFIGS:
                    own = [r for r in source[arm] if r['configuration'] == cfg]
                    baseline = [r for r in source[ref] if r['configuration'] == cfg]
                    row, _ = large.statistics(own, baseline, budget, arm, ref)
                    add_node_times(row, own, budget)
                    configurations.append(dict(配置=cfg, **scale_summary(row)))
        for filename, rows in [('RESULTS.csv', summary), ('PPO_vs_SHARED.csv', direct),
                               ('PER_CONFIGURATION.csv', configurations), ('PER_PAIR.csv', details)]:
            p.csv_write(folder/filename, rows)
        write(folder/'REPORT.json', dict(plan_sha256=plan['sha256'], completed=True,
            primary_reference='BASE', direct_reference=SHARED, selected_update=plan['selected_update'],
            hv_divisor=DIVISOR, reference=[1.1]*3, hv_already_divided=True,
            normalization='instance union' if mode=='results_union' else 'frozen training bounds',
            wtl_tolerance_before_division=1e-12, wtl_tolerance_after_division=1e-12/DIVISOR,
            interval='95% hierarchical paired bootstrap within fixed configurations; instances then seeds; 3000 draws',
            time_definition='Node time starts at controller creation; full B5000 search time also includes earlier setup/finalization.',
            qualification=plan['qualification']))
    write(root/'results_union/INSTANCE_BOUNDS.json', bounds)
    write(root/'REUSE.json', dict(base_runs=27, shared_runs=27, other_old_records='normalization pool only',
        source_records=plan['source_records'], new_ppo_runs=27, new_training_updates=0))


def execute(args):
    root, old, trained = args.out.resolve(), args.large_source.resolve(), args.training_source.resolve()
    if any(root == src or root in src.parents or src in root.parents for src in (old, trained)):
        raise ValueError('Use a separate output directory')
    root.mkdir(parents=True, exist_ok=True)
    p.torch.set_num_threads(1)
    with (root/'.pipeline.lock').open('a+') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit('This refresh is already running')
        def stop(signum, frame):
            raise p.scheduler.Paused(f'Received signal {signum}')
        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)
        try:
            plan = prepare(root, old, trained)
            jobs = jobs_for(root, plan)
            done = sum(p.scheduler.valid_completion(folder, task, plan) is not None for task, folder in jobs)
            durations = [read(Path(path))['optimisation_seconds'] for path in plan['source_records'][PPO]]
            estimate = (27-done)*mean(durations)/args.workers/3600
            write(root/'COST_ESTIMATE.json', dict(completed_new_ppo=done, total_new_ppo=27,
                remaining_search_hours=estimate, workers=args.workers, prior_ppo_mean_seconds=mean(durations),
                qualification='Historical PPO timing; excludes source checks, reporting and load variation.'))
            print(f'[PPO-LARGE-REFRESH] only new PPO: {done}/27; reuse Base/shared; '
                  f'workers={args.workers}; remaining search estimate={estimate:.2f} h', flush=True)
            if args.prepare_only:
                write(root/'STATUS.json', dict(completed=False, stage='prepared', done=done, total=27, new_training_updates=0))
                return
            started = time.monotonic()
            p.scheduler.HERE2 = HERE
            rows = p.scheduler.execute_jobs(root, plan, jobs, args.workers, 'new_PPO_S9_S11')
            p.verify(plan['files'])
            report(root, plan, rows)
            write(root/'STATUS.json', dict(completed=True, stage='all', done=27, total=27,
                selected_update=plan['selected_update'], new_searches=27, reused_base=27, reused_shared=27,
                new_training_updates=0, hv_divisor=DIVISOR, historical_large_results_already_inspected=True,
                wall_seconds_this_invocation=time.monotonic()-started))
            print('[PPO-LARGE-REFRESH] COMPLETE; results_union/RESULTS.csv; PPO_vs_SHARED.csv', flush=True)
        except p.scheduler.Paused as exc:
            write(root/'STATUS.json', dict(completed=False, paused=True, reason=str(exc), resume='Repeat the same launch command'))
            print('[PPO-LARGE-REFRESH] PAUSED:', exc, flush=True)
        except Exception as exc:
            write(root/'STATUS.json', dict(completed=False, failed=True, error=str(exc)))
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--large-source', type=Path, default=DIAG/'ccp_single_s9_s11_b5000_v1')
    parser.add_argument('--training-source', type=Path, default=DIAG/'ccp_single_ppo_s0_s8_b5000_v1')
    parser.add_argument('--out', type=Path, default=DIAG/'ccp_single_ppo_s9_s11_refresh_v1')
    parser.add_argument('--workers', type=int, choices=(1, 2), default=2)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--task', type=Path)
    parser.add_argument('--run-out', type=Path)
    args = parser.parse_args()
    if args.worker:
        worker(args.out.resolve(), large.checked(args.out/'PLAN.json'), read(args.task), args.run_out.resolve())
    else:
        execute(args)


if __name__ == '__main__':
    main()
