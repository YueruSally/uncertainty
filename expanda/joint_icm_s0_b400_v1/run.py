#!/usr/bin/env python3
"""Bounded S0 joint supervision then terminal PPO, isolated from running S11 jobs."""
from common import *
import argparse
import fcntl
import os
import shutil
import signal
import socket
import subprocess
import time
from train_shared import discover, load_collections, train
from shared_policy import SharedPolicy
from terminal_ppo import records, ppo_update, snapshot
import shared_search

DIAG = HERE.parent / 'transformer_ppo_runs/diagnostics'


def prepare(a):
    root = a.out.resolve()
    if (root / 'PLAN.json').exists():
        p = read_json(root / 'PLAN.json')
        if p['settings'] != SETTINGS_B400:
            raise ValueError('Frozen settings changed')
        if p['environment'] != {'torch': str(torch.__version__), 'numpy': str(np.__version__)}:
            raise ValueError('Use the same Torch/NumPy environment')
        verify(p['files']); verify(p['runtime_files'])
        return p
    source = a.source.resolve()
    old = read_json(source / 'PLAN.json')
    if old['environment'] != {'torch': str(torch.__version__), 'numpy': str(np.__version__)}:
        raise ValueError('Use historical environment: ' + str(old['environment']))
    verify(old['runtime_files'])
    design, rows, files, _ = history(source)
    original = Path(old['source_root'])
    old_grid = read_json(original / 'PLAN.json')
    if any(old_grid[k] != v for k, v in {
        'population_size': 20, 'evaluation_budget': 400, 'scenario_count': 100,
        'ccp_confidence': .9, 'candidate_pool': 8}.items()):
        raise ValueError('Historical population/budget/scenarios differ')
    m_collection = discover(a.diagnostics, 'M', a.m_collection)
    mp = read_json(m_collection / 'PLAN.json')
    instances_root = Path(mp['generated_instances'])
    train_names = sorted(r['instance_id'] for r in mp['instances'] if r['development_role'] == 'train')
    valid_names = sorted(r['instance_id'] for r in mp['instances'] if r['development_role'] == 'validation')
    if len(train_names) != 200 or len(valid_names) != 20 or set(train_names) & set(valid_names):
        raise ValueError('Invalid training/validation catalog')
    files[str(m_collection / 'PLAN.json')] = digest(m_collection / 'PLAN.json')
    train_digests = set()
    for name in train_names + valid_names:
        path = instances_root / (name + '.json'); info, _ = read_instance(path)
        role = 'train' if name in train_names else 'validation'
        if info['configuration'] != 'S0' or info['development_role'] != role:
            raise ValueError('Wrong training/validation role: ' + name)
        if digest(path) != mp['input_sha256'][str(path)]:
            raise ValueError('Training instance hash changed')
        train_digests.add(info['instance_digest']); files[str(path)] = digest(path)
    for name in design['instances']:
        path = PACKAGE / 'assets/search_instances' / (name + '.json')
        info, _ = read_instance(path)
        ref = read_json(original / 'runs' / name / f'seed_{design["algorithm_seeds"][0]}' / 'BASE/configuration.json')
        if info['configuration'] != 'S0' or info['instance_digest'] in train_digests:
            raise ValueError('Screen/training overlap or wrong configuration')
        if info['instance_digest'] != ref['instance_digest']:
            raise ValueError('Screen instance differs from historical Base')
        files[str(path)] = digest(path)
    source_model = a.shared_source.resolve()
    folders = None
    if (source_model / 'BUNDLE.json').exists():
        b = read_json(source_model / 'BUNDLE.json')
        tr, va = check_split(b)
        if (tr, va) != (train_names, valid_names) or b['settings'] != SETTINGS:
            raise ValueError('Existing shared model has different training protocol')
        verify(b['files'])
        files.update(b['files'])
        files[str(source_model / 'BUNDLE.json')] = digest(source_model / 'BUNDLE.json')
        files.update({str(source_model / n): h for n, h in b['checkpoint_sha256'].items()})
    else:
        folders = {c: str(discover(a.diagnostics, c, getattr(a, c.lower() + '_collection'))) for c in COMPONENTS}
        data, label_files = load_collections({c: Path(p) for c, p in folders.items()})
        files.update(label_files)
        del data
        source_model = None
    for path in (PACKAGE / 'src/baseline_uncertainty.py', PACKAGE / 'assets/data/data_expanded.xlsx',
                 PACKAGE / 'src/data/border_crossing_events.csv', PACKAGE / 'assets/reward_normalization.json'):
        if digest(path) not in old_grid['input_sha256'].values():
            raise ValueError('Frozen transport engine or normalization differs: ' + str(path))
    p = {'settings': SETTINGS_B400, 'source': str(source), 'original': str(original),
         'instances': design['instances'], 'seeds': design['algorithm_seeds'],
         'train_instances': train_names, 'validation_instances': valid_names,
         'instances_root': str(instances_root), 'source_model': str(source_model) if source_model else None,
         'collections': folders, 'files': files, 'runtime_files': runtime_files(),
         'environment': {'torch': str(torch.__version__), 'numpy': str(np.__version__)},
         'supervised_settings': SETTINGS, 'created_on_host': socket.gethostname()}
    p['sha256'] = canonical_digest(p); freeze(root / 'PLAN.json', p)
    print('[S0-JOINT] frozen: 32 PPO training + 60 validation + 60 screen searches; B400 only', flush=True)
    return p


def run_job(root, stage, command, logfile):
    logfile.parent.mkdir(parents=True, exist_ok=True)
    with logfile.open('a') as log:
        proc = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        started = time.monotonic(); last = 0.
        try:
            while proc.poll() is None:
                elapsed = time.monotonic() - started
                if elapsed > SETTINGS_B400['worker_timeout_seconds']:
                    raise TimeoutError('Worker timeout; inspect ' + str(logfile))
                if elapsed >= last:
                    state = {'completed': False, 'stage': stage, 'pid': proc.pid,
                             'seconds': round(elapsed), 'log': str(logfile)}
                    write_json(root / 'STATUS.json', state)
                    print('[S0-JOINT] heartbeat ' + json.dumps(state), flush=True); last = elapsed + 30
                time.sleep(.5)
            if proc.returncode:
                raise RuntimeError(str(logfile) + '\n' + logfile.read_text(errors='replace')[-5000:])
        finally:
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGTERM)
                try: proc.wait(timeout=15)
                except subprocess.TimeoutExpired: os.killpg(proc.pid, signal.SIGKILL); proc.wait()


def shared_model(root, p):
    folder = root / 'shared_model'
    if (folder / 'BUNDLE.json').exists() and (folder / 'READY.json').exists():
        b = read_json(folder / 'BUNDLE.json')
        if read_json(folder / 'READY.json')['plan_sha256'] != p['sha256']:
            raise ValueError('Shared model belongs to another plan')
        check_split(b); verify({str(folder / n): h for n, h in b['checkpoint_sha256'].items()})
        if digest(folder / 'BUNDLE.json') != read_json(folder / 'READY.json')['bundle_sha256']:
            raise ValueError('Shared bundle changed')
        return folder
    if folder.exists():
        dest = root / 'interrupted' / ('shared_model_' + str(time.time_ns()))
        dest.parent.mkdir(exist_ok=True); folder.rename(dest)
    run_job(root, 'shared-supervision', [sys.executable, '-u', str(HERE / 'run.py'),
            '--out', str(root), '--worker', 'fit'], root / 'shared_training.log')
    return shared_model(root, p)


def identity(p, instance, seed, method, model, sample):
    return {'plan_sha256': p['sha256'], 'instance': instance, 'algorithm_seed': seed,
            'strategy': method, 'stochastic': sample, 'model_bundle_sha256': digest(model / 'BUNDLE.json')}


def rollout(root, p, instance, seed, model, folder, method, sample=False, role='screen', reference=None):
    expected = identity(p, instance, seed, method, model, sample)
    if (folder / 'RUN_COMPLETE.json').exists():
        r = read_json(folder / 'RUN_COMPLETE.json')
        if any(r.get(k) != v for k, v in expected.items()) or r['real_objective_evaluation_count'] != 400:
            raise ValueError('Resume identity/budget mismatch: ' + str(folder))
        if r['combined_hidden_candidate_objective_evaluations'] or not r['completed']:
            raise ValueError('Invalid completed search')
        if sample and not (folder / 'TRAJECTORY.npz').exists():
            raise ValueError('Missing PPO trajectory')
        print('[S0-JOINT] reused ' + str(folder.relative_to(root)), flush=True)
        return r
    if folder.exists():
        archive = root / 'interrupted' / (folder.name + '_' + str(time.time_ns()))
        archive.parent.mkdir(exist_ok=True); folder.rename(archive)
    task = {'instance': instance, 'seed': seed, 'model': str(model), 'folder': str(folder),
            'method': method, 'sample': sample, 'role': role,
            'reference': str(reference) if reference else None}
    task_file = root / 'tasks' / (canonical_digest(task) + '.json'); freeze(task_file, task)
    log = root / 'logs' / (task_file.stem + '.log')
    run_job(root, str(folder.relative_to(root)), [sys.executable, '-u', str(HERE / 'run.py'),
            '--out', str(root), '--worker', 'search', '--task', str(task_file)], log)
    r = rollout(root, p, instance, seed, model, folder, method, sample, role, reference)
    print(f'[S0-JOINT] complete {instance}/{seed}/{method} HV={r["final_hypervolume"]:.7g} seconds={r["optimisation_seconds"]:.1f}', flush=True)
    return r


def screen(root, p, model, method):
    for n in p['instances']:
        for seed in p['seeds']:
            ref = Path(p['original']) / 'runs' / n / f'seed_{seed}' / 'BASE/configuration.json'
            rollout(root, p, n, seed, model, root / 'screen' / n / f'seed_{seed}' / method,
                    method, reference=ref)


def terminal(root, p, supervised):
    dest = root / 'terminal'; dest.mkdir(exist_ok=True)
    config = SETTINGS_B400['ppo']; first = dest / 'checkpoints/update_000'
    if not first.exists(): snapshot(SharedPolicy(supervised), first)
    initial = SharedPolicy(first); original = SharedPolicy(supervised)
    if any(not torch.equal(x, y) for a, b in zip(initial.models, original.models)
           for x, y in zip(a.parameters(), b.parameters())):
        raise ValueError('Update zero differs from frozen shared supervision')
    history_rows = []; validation = []
    for update in range(config['updates'] + 1):
        checkpoint = dest / f'checkpoints/update_{update:03d}'
        if update:
            previous = dest / f'checkpoints/update_{update-1:03d}'; episodes = []
            for j in range(config['episodes_per_update']):
                idx = (update - 1) * config['episodes_per_update'] + j
                n = p['train_instances'][idx]; folder = dest / 'rollouts' / f'u{update:03d}_{j}_{n}'
                r = rollout(root, p, n, 974000 + idx, previous, folder, PPO, True, 'train')
                episodes.append(records(folder, r['final_hypervolume']))
            if not (checkpoint / 'BUNDLE.json').exists():
                policy = SharedPolicy(previous); torch.manual_seed(975000 + update)
                metrics = ppo_update(policy, episodes, config); snapshot(policy, checkpoint, metrics)
            SharedPolicy(checkpoint)
            history_rows.append({'update': update, **read_json(checkpoint / 'UPDATE.json')})
            print('[S0-PPO] ' + json.dumps(history_rows[-1]), flush=True)
        if update in config['validation_checkpoints']:
            hvs = []
            for n in p['validation_instances']:
                folder = dest / 'validation' / f'u{update:03d}_{n}'
                ref = dest / 'validation' / f'u000_{n}/configuration.json' if update else None
                r = rollout(root, p, n, 976001, checkpoint, folder, PPO, False, 'validation', ref)
                hvs.append(r['final_hypervolume'])
            base = validation[0]['hv'] if validation else hvs
            validation.append({'update': update, 'hv': hvs,
                               'mean_difference_vs_initial': float(np.mean(np.asarray(hvs) - base))})
            print('[S0-PPO-VALIDATION] ' + json.dumps(validation[-1]), flush=True)
        write_json(dest / 'TRAINING_PROGRESS.json', {'completed_update': update, 'training': history_rows, 'validation': validation})
    selected = choose_checkpoint(validation)
    write_json(dest / 'QUICK_RESULT.json', {'completed': True, 'selected': selected,
        'training': history_rows, 'validation': validation, 'training_searches': 32,
        'validation_searches': 60, 'completed_search_evaluations': 36800,
        'selection_uses_screen_instances': False, 'final_test_opened': False})
    chosen = dest / f'checkpoints/update_{selected["update"]:03d}'
    freeze(dest / 'SELECTED.json', {'update': selected['update'], 'model': str(chosen),
                                  'bundle_sha256': digest(chosen / 'BUNDLE.json')})
    return chosen


def worker(a, p):
    torch.set_num_threads(1); verify(p['runtime_files'])
    if a.worker == 'fit':
        folder = a.out / 'shared_model'
        if p['source_model']:
            source = Path(p['source_model']); b = read_json(source / 'BUNDLE.json')
            folder.mkdir(parents=True)
            for name in ['BUNDLE.json', 'PLAN.json', 'VALIDATION.json'] + b['checkpoints']:
                shutil.copy2(source / name, folder / name)
            print('[S0-JOINT] reused frozen S0 shared supervision; no new training', flush=True)
        else:
            data, files = load_collections({c: Path(f) for c, f in p['collections'].items()})
            train(data, folder, provenance={**files, **p['runtime_files']})
        b = read_json(folder / 'BUNDLE.json'); check_split(b)
        write_json(folder / 'READY.json', {'plan_sha256': p['sha256'], 'bundle_sha256': digest(folder / 'BUNDLE.json')})
        return
    task = read_json(a.task); role = task['role']; name = task['instance']; sample = task['sample']
    allowed = p['instances'] if role == 'screen' else p['train_instances'] if role == 'train' else p['validation_instances']
    if name not in allowed or sample != (role == 'train'):
        raise ValueError('Wrong rollout split')
    instance = (PACKAGE / 'assets/search_instances' if role == 'screen' else Path(p['instances_root'])) / (name + '.json')
    info, _ = read_instance(instance)
    if info['configuration'] != 'S0': raise ValueError('S0 only')
    shared_search.METHOD = task['method']
    started = time.monotonic()
    row = shared_search.run_one(instance, task['seed'], Path(task['model']), Path(task['folder']),
        p['sha256'], budget=400, population=20, generations=1000, stochastic=sample, record=sample,
        reference_config=Path(task['reference']) if task['reference'] else None)
    row.update(host=socket.gethostname(), invocation_seconds=time.monotonic() - started,
               purpose=role, search_model_updates=0)
    write_json(Path(task['folder']) / 'RUN_COMPLETE.json', row)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=DIAG / 'ccp_s0_joint_b400_v1')
    parser.add_argument('--source', type=Path, default=DIAG / 'ccp_icm_heuristic_270_v1')
    parser.add_argument('--shared-source', type=Path, default=DIAG / 'ccp_icm_formal_joint_v1/shared_model')
    parser.add_argument('--diagnostics', type=Path, default=DIAG)
    for c in COMPONENTS: parser.add_argument('--' + c.lower() + '-collection', type=Path)
    parser.add_argument('--stage', choices=['all', 'prepare', 'shared'], default='all')
    parser.add_argument('--worker', choices=['fit', 'search']); parser.add_argument('--task', type=Path)
    a = parser.parse_args(); a.out = a.out.resolve()
    if a.worker: worker(a, read_json(a.out / 'PLAN.json')); return
    for source in (a.source.resolve(), a.shared_source.resolve()):
        if a.out == source or a.out in source.parents or source in a.out.parents:
            raise ValueError('Output must be separate from historical sources')
    a.out.mkdir(parents=True, exist_ok=True); torch.set_num_threads(1)
    def interrupted(signum, frame): raise KeyboardInterrupt('Received signal ' + str(signum))
    for sig in (signal.SIGTERM, signal.SIGINT): signal.signal(sig, interrupted)
    with (a.out / '.pipeline.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            p = prepare(a)
            if a.stage == 'prepare':
                write_json(a.out / 'STATUS.json', {'completed_requested_stage': 'prepare', 'search_started': False})
                print('[S0-JOINT] PREPARE COMPLETE; zero new evaluations', flush=True); return
            model = shared_model(a.out, p); screen(a.out, p, model, SHARED)
            from results import report
            report(a.out, p, [SHARED], a.out / 'shared_report')
            print('[S0-JOINT] shared comparison ready: shared_report/00_联合训练对比.csv', flush=True)
            if a.stage == 'shared':
                write_json(a.out / 'STATUS.json', {'completed_requested_stage': 'shared', 'ppo_started': False})
                return
            selected = terminal(a.out, p, model); screen(a.out, p, selected, PPO)
            verify(p['files']); verify(p['runtime_files'])
            report(a.out, p, [SHARED, PPO], a.out / 'results')
            write_json(a.out / 'STATUS.json', {'completed': True, 'stage': 'all', 'final_test_opened': False})
            print('[S0-JOINT] COMPLETE; results/00_联合训练对比.csv', flush=True)
        except BaseException as exc:
            write_json(a.out / 'STATUS.json', {'completed': False, 'error': str(exc), 'resume': 'Repeat identical command; completed work reused'})
            raise

if __name__ == '__main__': main()
