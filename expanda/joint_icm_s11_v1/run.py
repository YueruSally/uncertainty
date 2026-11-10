#!/usr/bin/env python3
"""S11 frozen-model transfer: original eight arms and shared supervision in parallel.

All historical source/checkpoint files remain read-only. No S11 training or PPO.
"""
import argparse
import copy
from dataclasses import asdict
import fcntl
import os
from pathlib import Path
import resource
import signal
import socket
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
JOINT = HERE.parent / 'joint_icm_v1'
sys.path.insert(0, str(JOINT))
from joint_common import (PACKAGE, METHOD, COMPONENTS, canonical_digest, digest,
                          read_json, write_json, read_instance, verify, sources,
                          torch, np)
import experiment

DEFAULT_OUT = HERE.parent / 'transformer_ppo_runs/diagnostics/ccp_icm_s11_joint_v1'
DEFAULT_DIAG = DEFAULT_OUT.parent
DEFAULT_SHARED = DEFAULT_DIAG / 'ccp_icm_formal_joint_v1/shared_model'
ROLE = 'S0-trained frozen models on new S11 development instances; not independent final test'


def frozen(path, value):
    if path.exists():
        if read_json(path) != value:
            raise ValueError('Frozen input changed; preserve this output: ' + str(path))
    else:
        write_json(path, value)


def settings():
    old = read_json(PACKAGE / 'experiment.json')
    return {**old, 'protocol': 'icm-s11-frozen-transfer-v1', 'role': ROLE,
            'configuration': 'S11', 'batch_count': 50, 'deadline_window_alpha': 2.0,
            'instance_master_seed': 116000000, 'ccp_seed_base': 118000000,
            'oos_seed_base': 119000000,
            'oos_policy': 'reserved only; this launcher evaluates optimization scenarios, no automatic OOS',
            'shared_method': METHOD, 'training_scope': 'existing S0 labels only',
            'new_ppo_updates': 0}


def prepare(out, source_catalog):
    """One shared manifest; preparation is serialized even with separate terminals."""
    out.mkdir(parents=True, exist_ok=True)
    with (out / '.prepare.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        path = out / 'PLAN.json'
        if path.exists():
            p = read_json(path)
            if p['settings'] != settings():
                raise ValueError('S11 settings changed')
            verify(p['files'])
            if p['environment'] != {'torch': str(torch.__version__), 'numpy': str(np.__version__)}:
                raise ValueError('Use the same Torch and NumPy versions for both experiment lines')
            return p
        cfg = settings()
        source = Path(source_catalog or cfg['source_catalog_default']).expanduser().resolve()
        catalog = read_json(source)
        identity = dict(catalog); claimed = identity.pop('catalog_digest')
        data = PACKAGE / 'assets/data/data_expanded.xlsx'
        if (canonical_digest(identity) != claimed or claimed != cfg['expected_source_catalog_digest']
                or catalog['source_data_sha256'] != digest(data)):
            raise ValueError('Source catalog/workbook is not the original frozen split')
        od = [tuple(x) for x in catalog['train_od_pool']]
        if not od or set(od) & {tuple(x) for x in catalog['test_od_pool']}:
            raise ValueError('Invalid OD split')
        snapshot = out / 'source_catalog.json'
        frozen(snapshot, catalog)
        generation = {'settings': cfg, 'source_catalog_sha256': digest(source),
                      'generator_sha256': digest(Path(__file__)), 'data_sha256': digest(data)}
        frozen(out / 'GENERATION_PLAN.json', generation)
        import baseline_uncertainty as base
        from transformer_ppo.instance_catalog import _batches, _find_path_seed
        from transformer_ppo.config import SCENARIOS
        if SCENARIOS['S11'] != (50, 2.0):
            raise ValueError('S11 definition changed')
        base.BORDER_EVENT_DEFINITIONS = base.load_border_event_definitions(base.DEFAULT_BORDER_EVENT_DATA_FILE)
        network = base.load_network_from_extended(str(data))
        nodes, regions, _, _, _, arcs, timetables, raw, *_ = network
        timetable = base.build_timetable_dict(timetables); lookup = base.build_arc_lookup(arcs)
        sample = read_json(PACKAGE / 'assets/search_instances/prescreen-search-S0-001.json')
        release = sample['release_time_definition']
        known = {r.get('batch_signature') for r in catalog['instances']}
        prior_seeds = {r.get(k) for r in catalog['instances'] for k in ('instance_seed', 'ccp100_seed', 'oos5000_seed')}
        records = []
        for i in range(cfg['instance_count']):
            seed = cfg['instance_master_seed'] + i * 100
            if {seed, cfg['ccp_seed_base'] + i, cfg['oos_seed_base'] + i} & prior_seeds:
                raise ValueError('S11 seeds overlap source catalog')
            batches = _batches(raw, od, 50, 2.0, seed, 'train', release['maximum_h'], release['step_h'])
            batch_rows = [asdict(b) for b in batches]; signature = canonical_digest(batch_rows)
            if signature in known:
                raise ValueError('Duplicate instance; do not retry seeds based on results')
            known.add(signature)
            name = f'formal-transfer-S11-{i+1:03d}'
            row = {k: sample[k] for k in ('schema_version', 'alpha_definition', 'release_time_definition',
                                          'quantity_definition', 'od_pool')}
            row.update(instance_id=name, configuration='S11', K=50, deadline_window_alpha=2.0,
                       split='train', development_role='frozen_transfer_development', instance_seed=seed,
                       path_seed=_find_path_seed(seed + cfg['path_seed_offset'], nodes, regions, arcs, batches, timetable, lookup),
                       ccp100_seed=cfg['ccp_seed_base'] + i, oos5000_seed=cfg['oos_seed_base'] + i,
                       algorithm_seed=cfg['algorithm_seeds'][0], source_data_sha256=digest(data),
                       source_catalog_digest=claimed, batch_signature=signature, batches=batch_rows)
            row['instance_digest'] = canonical_digest(row)
            target = out / 'instances' / (name + '.json'); frozen(target, row)
            read_instance(target)
            records.append({k: row[k] for k in ('instance_id', 'instance_digest', 'batch_signature', 'ccp100_seed')})
            print(f'[S11-PREPARE] {i+1}/10 {name}; batches=50 alpha=2; zero objective evaluations', flush=True)
        cat = {'instances': records, 'configuration': 'S11', 'count': len(records),
               'final_test_instances_used': False, 'source_catalog_digest': claimed}
        cat['catalog_digest'] = canonical_digest(cat); frozen(out / 'instances/catalog.json', cat)
        files = {**sources(), str(Path(__file__).resolve()): digest(Path(__file__)),
                 str(PACKAGE / 'experiment.json'): digest(PACKAGE / 'experiment.json')}
        for c in COMPONENTS:
            original = PACKAGE / 'assets' / c / 'BUNDLE.json'
            bundle = copy.deepcopy(read_json(original))
            spec = bundle['models']['200'] if c == 'M' else bundle
            checkpoints, hashes = [], {}
            for old in spec['checkpoints']:
                ck = (PACKAGE / 'assets' / c / Path(old).name).resolve()
                sha = spec['checkpoint_sha256'][old]
                verify({str(ck): sha}); checkpoints.append(str(ck)); hashes[str(ck)] = sha
            spec['checkpoints'], spec['checkpoint_sha256'] = checkpoints, hashes
            target = out / 'runtime_inputs' / (c + '_BUNDLE.json'); frozen(target, bundle)
            files.update(hashes); files[str(original)] = digest(original)
            files[str(target)] = digest(target)
        for f in [snapshot, out / 'GENERATION_PLAN.json'] + sorted((out / 'instances').glob('*.json')):
            files[str(f)] = digest(f)
        plan = {'settings': cfg, 'instances': [r['instance_id'] for r in records],
                'algorithm_seeds': cfg['algorithm_seeds'], 'methods': list(experiment.METHODS),
                'shared_method': METHOD, 'original_searches': 2400, 'shared_searches': 300,
                'total_searches': 2700, 'files': files,
                'environment': {'torch': str(torch.__version__), 'numpy': str(np.__version__)}}
        frozen(path, plan)
        return plan


def shared_manifest(out, model, plan):
    from shared_policy import SharedPolicy
    model = model.resolve(); policy = SharedPolicy(model); bundle = policy.bundle
    train = set(sum(bundle['training_instances'].values(), []))
    valid = set(sum(bundle['validation_instances'].values(), []))
    if train & valid or set(plan['instances']) & (train | valid):
        raise ValueError('Training/validation overlap with S11 evaluation')
    if len(train) != 200 or len(valid) != 20:
        raise ValueError('Expected original S0 200/20 supervision split')
    source_plan = read_json(model / 'PLAN.json')
    for name in sorted(train | valid):
        matches = [Path(p) for p in source_plan['files'] if Path(p).name == name + '.json']
        if len(matches) != 1:
            raise ValueError('Missing training instance provenance: ' + name)
        f = matches[0]; verify({str(f): source_plan['files'][str(f)]})
        info, _ = read_instance(f)
        if info['configuration'] != 'S0':
            raise ValueError('This S11 transfer experiment requires S0-only supervision')
    manifest = {'model': str(model), 'bundle_sha256': digest(model / 'BUNDLE.json'),
                'checkpoint_sha256': {str(model / p): s for p, s in bundle['checkpoint_sha256'].items()},
                'scope': 'S0 supervised, S11 frozen evaluation; no S11 refitting'}
    frozen(out / 'SHARED_MODEL.json', manifest)
    return manifest


def run_worker(out, name, seed, method):
    plan = read_json(out / 'PLAN.json'); verify(plan['files'])
    if name not in plan['instances'] or seed not in plan['algorithm_seeds'] or method not in plan['methods'] + [METHOD]:
        raise ValueError('Worker is outside frozen grid')
    cfg = plan['settings']; sha = digest(out / 'PLAN.json')
    instance = out / 'instances' / (name + '.json')
    folder = out / 'runs' / name / f'seed_{seed}' / method
    if folder.exists():
        raise ValueError('Worker output must be fresh')
    if method == METHOD:
        from shared_search import run_one
        manifest = read_json(out / 'SHARED_MODEL.json'); model = Path(manifest['model'])
        verify({str(model / 'BUNDLE.json'): manifest['bundle_sha256'], **manifest['checkpoint_sha256']})
        result = run_one(instance, seed, model, folder, sha, cfg['evaluation_budget'], cfg['population_size'], cfg['maximum_generations'])
    else:
        experiment.SETTINGS = cfg
        a = argparse.Namespace(instance=instance, method=method, seed=seed, out=folder, plan_sha=sha,
            data=PACKAGE / 'assets/data/data_expanded.xlsx', normalization=PACKAGE / 'assets/reward_normalization.json',
            **{c.lower() + '_bundle': out / 'runtime_inputs' / (c + '_BUNDLE.json') for c in COMPONENTS})
        folder.mkdir(parents=True)
        experiment.run_worker(a); result = read_json(folder / 'RUN_COMPLETE.json')
    result.update(role=ROLE, configuration='S11', batch_count=50, deadline_window_alpha=2.0,
                  host=socket.gethostname(), shared_network=(method == METHOD),
                  peak_resident_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss /
                      (1024. ** 2 if sys.platform == 'darwin' else 1024.))
    write_json(folder / 'RUN_COMPLETE.json', result)


def jobs_for(plan, stage, smoke=False):
    methods = plan['methods'] if stage == 'original' else [METHOD]
    # First full-budget run of every arm is retained as part of the final grid.
    first = [(plan['instances'][0], plan['algorithm_seeds'][0], m) for m in methods]
    if smoke:
        return first
    seen = set(first)
    return first + [(n, s, m) for n in plan['instances'] for s in plan['algorithm_seeds']
                    for m in methods if (n, s, m) not in seen]


def validate_completion(out, plan, job):
    n, s, m = job; folder = out / 'runs' / n / f'seed_{s}' / m
    row = read_json(folder / 'RUN_COMPLETE.json'); cfg = read_json(folder / 'configuration.json')
    info, _ = read_instance(out / 'instances' / (n + '.json'))
    if (row.get('completed') is not True or row['plan_sha256'] != digest(out / 'PLAN.json')
            or (row['instance'], row['algorithm_seed'], row['strategy']) != job
            or row['real_objective_evaluation_count'] != plan['settings']['evaluation_budget']
            or row.get('configuration') != 'S11' or row.get('batch_count') != 50
            or cfg['instance_digest'] != info['instance_digest']):
        raise ValueError('Invalid/resumed S11 result: ' + str(folder))
    if m == METHOD:
        model = read_json(out / 'SHARED_MODEL.json')
        if row.get('model_bundle_sha256') != model['bundle_sha256'] or row.get('combined_hidden_candidate_objective_evaluations') != 0:
            raise ValueError('Shared model identity or budget mismatch')
    else:
        experiment.SETTINGS = plan['settings']
        experiment.validate_run(row, m, s, digest(out / 'PLAN.json'))
    # Same-instance/seed pairing can be checked irrespective of completion order.
    for other in folder.parent.glob('*/RUN_COMPLETE.json'):
        other_cfg = read_json(other.parent / 'configuration.json')
        for k in ('scenario_digest', 'instance_digest', 'normalization', 'seeds'):
            if cfg[k] != other_cfg[k]:
                raise ValueError('Method pairing mismatch: ' + k)
    if not (folder / 'final_feasible_nondominated.json').exists():
        raise ValueError('Missing final solutions')
    return row


def stop_children(active):
    for p, h, *_ in active.values():
        if p.poll() is None:
            os.killpg(p.pid, signal.SIGTERM)
    for p, h, *_ in active.values():
        try:
            p.wait(timeout=15)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, signal.SIGKILL); p.wait()
        h.close()


def ensure_shared(a, plan):
    """Reuse a completed shared model, or train on historical labels in this line."""
    model = a.shared_model.resolve(); model.mkdir(parents=True, exist_ok=True)
    waiting = time.monotonic()
    while not (model / 'BUNDLE.json').exists():
        # An existing training process may own the same directory. Wait, never duplicate it.
        with (model / '.train.lock').open('a+') as handle:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                print('[S11-SHARED] waiting for existing shared supervision; original searches continue', flush=True)
            else:
                from train_shared import discover, load_collections, train
                folders = {c: discover(a.diagnostics.resolve(), c) for c in COMPONENTS}
                rows, files = load_collections(folders)
                train(rows, model, provenance={**files, **sources()})
                break
        if time.monotonic() - waiting > a.wait_hours * 3600:
            raise TimeoutError('Shared supervision wait limit; original results are retained')
        time.sleep(30)
    return shared_manifest(a.out, model, plan)


def search_stage(a, plan):
    with (a.out / ('.' + a.stage + '.lock')).open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if a.stage == 'shared':
            ensure_shared(a, plan)
        pending = jobs_for(plan, a.stage, a.smoke); total = len(pending)
        active = {}; done = 0; last = 0.; start = time.monotonic(); completed_rows = []
        try:
            while pending or active:
                while pending and len(active) < a.workers:
                    job = pending.pop(0); n, s, m = job
                    folder = a.out / 'runs' / n / f'seed_{s}' / m
                    if (folder / 'RUN_COMPLETE.json').exists():
                        completed_rows.append(validate_completion(a.out, plan, job)); done += 1; continue
                    if folder.exists():
                        dest = a.out / 'interrupted' / f'{n}_{s}_{m}_{time.time_ns()}'
                        dest.parent.mkdir(exist_ok=True); folder.rename(dest)
                    log = a.out / 'logs' / f'{n}_{s}_{m}_{time.time_ns()}.log'
                    log.parent.mkdir(exist_ok=True); h = log.open('w')
                    cmd = [sys.executable, '-u', str(Path(__file__)), '--worker', '--out', str(a.out),
                           '--instance', n, '--seed', str(s), '--method', m]
                    p = subprocess.Popen(cmd, stdout=h, stderr=subprocess.STDOUT, start_new_session=True)
                    active[p.pid] = (p, h, time.monotonic(), log, job)
                    print(f'[S11-{a.stage.upper()}] start {n}/{s}/{m} PID={p.pid} log={log}', flush=True)
                for pid, (p, h, t, log, job) in list(active.items()):
                    if time.monotonic() - t > plan['settings']['per_run_timeout_seconds']:
                        raise TimeoutError('Full-budget worker timeout: ' + str(log))
                    if p.poll() is None:
                        continue
                    h.close(); del active[pid]
                    if p.returncode:
                        raise RuntimeError(str(log) + '\n' + log.read_text(errors='replace')[-6000:])
                    row = validate_completion(a.out, plan, job); completed_rows.append(row); done += 1
                    print(f'[S11-{a.stage.upper()}] complete {done}/{total} {job} HV={row["final_hypervolume"]:.7g} seconds={row["optimisation_seconds"]:.1f}', flush=True)
                if time.monotonic() - last >= 30:
                    status = {'stage': a.stage, 'finished': done, 'total': total, 'running': len(active),
                              'elapsed_seconds': time.monotonic() - start, 'smoke_only': a.smoke}
                    write_json(a.out / (a.stage.upper() + '_STATUS.json'), status)
                    print('[S11] ' + str(status), flush=True); last = time.monotonic()
                if active:
                    time.sleep(.5)
            verify(plan['files'])
            marker = a.stage.upper() + ('_SMOKE_COMPLETE.json' if a.smoke else '_COMPLETE.json')
            completion = {'completed': True, 'stage': a.stage, 'finished': total, 'total': total, 'running': 0,
                          'searches': total, 'smoke_only': a.smoke, 'plan_sha256': digest(a.out / 'PLAN.json'),
                          'host': socket.gethostname(), 'workers': a.workers}
            if a.smoke:
                completion['seconds_per_method'] = {r['strategy']: r['optimisation_seconds'] for r in completed_rows}
                completion['peak_resident_mib_per_method'] = {r['strategy']: r.get('peak_resident_mib') for r in completed_rows}
                completion['rough_full_grid_search_hours_at_these_durations'] = (
                    sum(completion['seconds_per_method'].values()) * len(plan['instances']) * len(plan['algorithm_seeds']) / 3600 / a.workers)
                completion['estimate_excludes'] = 'S11 instance variability, process overhead, supervision and OOS'
            write_json(a.out / marker, completion)
            write_json(a.out / (a.stage.upper() + '_STATUS.json'), completion)
        except BaseException as exc:
            write_json(a.out / (a.stage.upper() + '_STATUS.json'), {'completed': False, 'finished': done, 'error': str(exc)})
            raise
        finally:
            stop_children(active)


def report(out, plan):
    for stage in ('original', 'shared'):
        for job in jobs_for(plan, stage):
            validate_completion(out, plan, job)
    verify(plan['files'])
    from formal_report import union_report
    sha = digest(out / 'PLAN.json')
    comparisons = union_report(out, {'source': str(out), 'instances': plan['instances'],
                               'seeds': plan['algorithm_seeds'], 'sha256': sha, 'original_plan_sha': sha})
    write_json(out / 'QUICK_RESULT.json', {'completed': True, 'configuration': 'S11', 'batch_count': 50,
        'deadline_window_alpha': 2., 'searches': plan['total_searches'], 'comparisons': comparisons,
        'role': ROLE, 'oos_completed': False, 'new_ppo_updates': 0,
        'qualification': 'S0-trained models; S11 changes both batch count and delivery-window multiplier. Union HV is post-hoc reporting only.'})


def both(a, plan):
    if a.workers < 2:
        raise ValueError('--stage both needs at least two allocated workers; use separate stages for serial execution')
    active = {}
    with (a.out / '.both.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            for stage, workers in [('original', (a.workers + 1) // 2), ('shared', a.workers // 2)]:
                log = a.out / (stage + '.nohup.log'); h = log.open('a')
                cmd = [sys.executable, '-u', str(Path(__file__)), '--stage', stage, '--workers', str(workers),
                       '--out', str(a.out), '--shared-model', str(a.shared_model.resolve()),
                       '--diagnostics', str(a.diagnostics.resolve()), '--wait-hours', str(a.wait_hours)]
                if a.smoke:
                    cmd.append('--smoke')
                p = subprocess.Popen(cmd, stdout=h, stderr=subprocess.STDOUT, start_new_session=True)
                active[p.pid] = (p, h, stage, log)
                print(f'[S11] {stage} PID={p.pid} workers={workers} log={log}', flush=True)
            last = 0.
            while active:
                for pid, (p, h, stage, log) in list(active.items()):
                    if p.poll() is None:
                        continue
                    h.close(); del active[pid]
                    if p.returncode:
                        raise RuntimeError(f'{stage} failed; completed results retained\n' + log.read_text(errors='replace')[-6000:])
                if time.monotonic() - last > 30:
                    progress = {}
                    for stage in ('original', 'shared'):
                        path = a.out / (stage.upper() + '_STATUS.json')
                        if path.exists():
                            progress[stage] = read_json(path)
                    print('[S11-PARALLEL] ' + str(progress), flush=True); last = time.monotonic()
                if active:
                    time.sleep(1)
            if not a.smoke:
                report(a.out, plan)
            else:
                for stage in ('original', 'shared'):
                    print('[S11-COST] ' + str(read_json(a.out / (stage.upper() + '_SMOKE_COMPLETE.json'))), flush=True)
            print('[S11] requested stages complete; no PPO or independent-scenario reevaluation started', flush=True)
        finally:
            stop_children(active)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, default=DEFAULT_OUT)
    p.add_argument('--diagnostics', type=Path, default=DEFAULT_DIAG)
    p.add_argument('--shared-model', type=Path, default=DEFAULT_SHARED)
    p.add_argument('--source-catalog', type=Path)
    p.add_argument('--stage', choices=('prepare', 'original', 'shared', 'both', 'report'), default='both')
    p.add_argument('--workers', type=int, default=2, help='total worker slots for both lines; not per line')
    p.add_argument('--wait-hours', type=float, default=24.)
    p.add_argument('--smoke', action='store_true', help='one full-budget run per method, retained in the final grid')
    p.add_argument('--worker', action='store_true'); p.add_argument('--instance'); p.add_argument('--seed', type=int); p.add_argument('--method')
    a = p.parse_args(); a.out = a.out.resolve()
    if a.workers < 1 or a.wait_hours <= 0:
        p.error('positive workers/wait-hours required')
    def stop(*_):
        raise KeyboardInterrupt('Stopped; completed runs retained; use the same command to resume')
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, stop)
    if a.worker:
        run_worker(a.out, a.instance, a.seed, a.method); return
    plan = prepare(a.out, a.source_catalog)
    if a.stage == 'prepare':
        return
    if a.stage == 'report':
        report(a.out, plan)
    elif a.stage == 'both':
        both(a, plan)
    else:
        search_stage(a, plan)


if __name__ == '__main__':
    main()
