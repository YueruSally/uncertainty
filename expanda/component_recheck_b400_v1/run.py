#!/usr/bin/env python3
"""Frozen old-policy replication on ten fresh S0 development instances."""
import argparse
import csv
from dataclasses import asdict
import fcntl
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import time
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
PACKAGE = HERE.parent / 'formal_icm_s0_v1'
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '1'
os.environ['MPLBACKEND'] = 'Agg'
sys.path.insert(0, str(PACKAGE / 'src/support'))
sys.path.insert(0, str(PACKAGE / 'src'))
from runtime_common import read_json, write_json, digest, paired_summary
from transformer_ppo.instance_catalog import canonical_digest, read_instance, _batches, _find_path_seed
import experiment as engine

METHODS = engine.METHODS[:-1]
RULES = {engine.I_TRANSFORMER: engine.I_HEURISTIC,
         engine.C_TRANSFORMER: engine.C_HEURISTIC,
         engine.M_TRANSFORMER: engine.M_HEURISTIC}
CONFIG = {
    'protocol': 'frozen-old-components-b400-recheck-v1',
    'instance_count': 10, 'configuration': 'S0',
    'algorithm_seeds': list(range(961001, 961011)), 'methods': list(METHODS),
    'population_size': 20, 'evaluation_budget': 400, 'maximum_generations': 1000,
    'scenario_count': 100, 'ccp_confidence': .9, 'candidate_pool': 8,
    'instance_master_seed': 126000000, 'path_seed_offset': 1000000,
    'ccp_seed_base': 128000000,
    'source_catalog_digest': '9d4f557c7f61919c88a21df42dc6a2b56b36bcfdb812f8f45a81f9c8bf752087',
    'training': False, 'ppo': False, 'final_test_opened': False,
    'independent_oos': False,
    'selection': 'Report equal-instance paired HV changes vs Base and matching rule; no automatic changes or PPO.',
    'model_version': 'Frozen historical I residual, C residual, M size200 ensemble; no B20 retraining or fixed-seed-preservation patch.',
}


def freeze(path, value):
    if path.exists() and read_json(path) != value:
        raise ValueError(f'Frozen file changed; preserve existing results: {path}')
    write_json(path, value)


def prepare_models(root):
    """Relocate only checkpoint paths; keep model/scalers/gates unchanged."""
    paths = {}
    for c in 'ICM':
        bundle = read_json(PACKAGE / 'assets' / c / 'BUNDLE.json')
        spec = bundle['models']['200'] if c == 'M' else bundle
        if len(spec['checkpoints']) != 3:
            raise ValueError('Expected three historical checkpoints per component')
        names, hashes = [], {}
        for old in spec['checkpoints']:
            path = (PACKAGE / 'assets' / c / Path(old).name).resolve()
            if digest(path) != spec['checkpoint_sha256'][old]:
                raise ValueError(f'Historical weight changed: {path}')
            names.append(str(path)); hashes[str(path)] = digest(path)
        spec['checkpoints'], spec['checkpoint_sha256'] = names, hashes
        paths[c] = root / 'models' / f'{c}_BUNDLE.json'
        freeze(paths[c], bundle)
    return paths


def verify_catalog(root, config):
    cat = read_json(root / 'instances/catalog.json')
    identity = dict(cat); claimed = identity.pop('catalog_digest')
    if canonical_digest(identity) != claimed or cat['config_digest'] != canonical_digest(config):
        raise ValueError('Instance catalog/config changed')
    if len(cat['instances']) != config['instance_count']:
        raise ValueError('Incomplete instance catalog')
    seen = set()
    for item in cat['instances']:
        row, _ = read_instance(root / 'instances' / (item['instance_id'] + '.json'))
        if row['instance_digest'] != item['instance_digest'] or row['batch_signature'] in seen:
            raise ValueError('Changed or duplicate instance')
        if canonical_digest(row['batches']) != row['batch_signature'] or row['development_role'] != 'b400_replication':
            raise ValueError('Invalid new development instance')
        seen.add(row['batch_signature'])
    return cat


def prepare_instances(root, source, diagnostics, config=CONFIG):
    if (root / 'instances/catalog.json').exists():
        return verify_catalog(root, config)
    original = read_json(source)
    identity = dict(original); claimed = identity.pop('catalog_digest')
    data = PACKAGE / 'assets/data/data_expanded.xlsx'
    if claimed != canonical_digest(identity) or claimed != config['source_catalog_digest']:
        raise ValueError('Original OD split/catalog changed')
    if original['source_data_sha256'] != digest(data):
        raise ValueError('Source workbook changed')
    train_od = [tuple(p) for p in original['train_od_pool']]
    if not train_od or set(train_od) & {tuple(p) for p in original['test_od_pool']}:
        raise ValueError('Invalid train/test OD split')
    catalogs = {source.resolve(), (PACKAGE / 'assets/search_instances/catalog.json').resolve()}
    for folder in (diagnostics, PACKAGE / 'assets'):
        if folder.exists():
            catalogs.update(p.resolve() for p in folder.rglob('catalog.json') if root not in p.resolve().parents)
    known, prior_seeds = set(), set()
    for path in sorted(catalogs):
        for row in read_json(path).get('instances', []):
            if not isinstance(row, dict): continue
            if row.get('batch_signature'): known.add(row['batch_signature'])
            for key in ('instance_seed', 'ccp100_seed', 'oos5000_seed'):
                if row.get(key) is not None: prior_seeds.add(row[key])
    freeze(root / 'GENERATION_PLAN.json', {
        'config_digest': canonical_digest(config),
        'prior_catalog_sha256': {str(p): digest(p) for p in sorted(catalogs)},
        'generator_sha256': digest(Path(__file__))})
    import baseline_uncertainty as base
    base.BORDER_EVENT_DEFINITIONS = base.load_border_event_definitions(base.DEFAULT_BORDER_EVENT_DATA_FILE)
    nodes, regions, _, _, _, arcs, timetables, raw, *_ = base.load_network_from_extended(str(data))
    tt = base.build_timetable_dict(timetables); lookup = base.build_arc_lookup(arcs)
    sample = read_json(PACKAGE / 'assets/search_instances/prescreen-search-S0-001.json')
    records = []
    for index in range(config['instance_count']):
        seed = config['instance_master_seed'] + index * 100
        ccp = config['ccp_seed_base'] + index
        if seed in prior_seeds or ccp in prior_seeds:
            raise ValueError('New seed namespace overlaps previous instances')
        batches = _batches(raw, train_od, 20, 1., seed, 'train',
                           sample['release_time_definition']['maximum_h'], sample['release_time_definition']['step_h'])
        signature = canonical_digest([asdict(b) for b in batches])
        if signature in known: raise ValueError('Duplicate instance; no automatic seed retry')
        known.add(signature)
        name = f'recheck-b400-S0-{index+1:03d}'
        record = {k: sample[k] for k in ('schema_version','configuration','K','deadline_window_alpha',
                  'alpha_definition','release_time_definition','quantity_definition','od_pool')}
        record.update(instance_id=name, split='train', development_role='b400_replication',
                      instance_seed=seed, path_seed=_find_path_seed(seed + config['path_seed_offset'], nodes, regions, arcs, batches, tt, lookup),
                      ccp100_seed=ccp, oos5000_seed=None, algorithm_seed=config['algorithm_seeds'][0],
                      source_data_sha256=digest(data), source_catalog_digest=claimed,
                      batch_signature=signature, batches=[asdict(b) for b in batches])
        record['instance_digest'] = canonical_digest(record)
        freeze(root / 'instances' / (name + '.json'), record)
        records.append({k: record[k] for k in ('instance_id','instance_seed','path_seed','ccp100_seed','batch_signature','instance_digest')})
        print(f'[B400-PREPARE] {index+1}/{config["instance_count"]}; zero objective evaluations', flush=True)
    cat = {'config_digest': canonical_digest(config), 'instances': records, 'configuration':'S0',
           'purpose':'fresh batches in historical training OD domain; development replication, not final test',
           'final_test_instances_used':False, 'source_data_sha256':digest(data)}
    cat['catalog_digest'] = canonical_digest(cat)
    freeze(root / 'instances/catalog.json', cat)
    return verify_catalog(root, config)


def make_plan(root, cat, config=CONFIG):
    import platform
    import numpy as np
    import torch
    files = list((PACKAGE/'src').rglob('*.py')) + list((PACKAGE/'src/data').glob('*.csv'))
    files += [PACKAGE/'assets/data/data_expanded.xlsx', PACKAGE/'assets/reward_normalization.json', Path(__file__)]
    files += list((root/'models').glob('*.json')) + list((root/'instances').glob('*.json'))
    if (root/'GENERATION_PLAN.json').exists():files.append(root/'GENERATION_PLAN.json')
    for c in 'ICM':
        files.append(PACKAGE/'assets'/c/'BUNDLE.json')
        files += list((PACKAGE/'assets'/c).glob('*.pt'))
    value = {'config':config,'instances':[r['instance_id'] for r in cat['instances']],
             'environment':{'python':platform.python_version(),'numpy':str(np.__version__),'torch':str(torch.__version__)},
             'runs':len(cat['instances'])*len(config['algorithm_seeds'])*len(METHODS),
             'files':{str(p.resolve()):digest(p) for p in sorted(set(files))},
             'maximum_completed_search_evaluations':len(cat['instances'])*len(config['algorithm_seeds'])*len(METHODS)*400,
             'zero_reference_policy':'Percent aggregate undefined if any reference HV is zero; report all-pair absolute difference.'}
    value['sha256'] = canonical_digest(value)
    freeze(root/'PLAN.json',value)
    return value


def validate(row, method, seed, sha):
    # A zero-HV or no-eligible-action run remains an outcome, not a reason to drop it.
    import math
    if (not row.get('completed') or row['strategy'] != method or row['algorithm_seed'] != seed
        or row['plan_sha256'] != sha or row['real_objective_evaluation_count'] != 400
        or row['evaluation_budget'] != 400 or row.get('ppo_updates',0)
        or not math.isfinite(row['final_hypervolume']) or row['final_hypervolume'] < 0):
        raise ValueError('Invalid B400 run')
    hidden = [v for k,v in row.items() if 'hidden_candidate_objective_evaluations' in k]
    hidden.append(row.get('k8_scale_search',{}).get('hidden_candidate_objective_evaluations',0))
    if any(v != 0 for v in hidden): raise ValueError('Unexpected hidden objective evaluation')
    if method in (engine.I_HEURISTIC,engine.I_TRANSFORMER) and row.get('initialization_k8_pool_events') != 20:
        raise ValueError('Initialization controller did not cover 20 original slots')


def worker(root, name, seed, method, folder, sha):
    plan = read_json(root/'PLAN.json')
    if sha != plan['sha256'] or name not in plan['instances'] or seed not in plan['config']['algorithm_seeds'] or method not in METHODS:
        raise ValueError('Worker not in frozen grid')
    for path, expected in plan['files'].items():
        if digest(Path(path)) != expected:
            raise ValueError(f'Frozen code/input changed during execution: {path}')
    engine.SETTINGS = plan['config']
    engine.validate_run = validate
    args = SimpleNamespace(data=PACKAGE/'assets/data/data_expanded.xlsx', instance=root/'instances'/f'{name}.json',
        out=folder, seed=seed, method=method, plan_sha=sha, normalization=PACKAGE/'assets/reward_normalization.json',
        i_bundle=root/'models/I_BUNDLE.json', c_bundle=root/'models/C_BUNDLE.json', m_bundle=root/'models/M_BUNDLE.json')
    engine.run_worker(args)


def report(root, plan, rows):
    keyed = {(r['instance'],r['algorithm_seed'],r['strategy']):r for r in rows}
    expected = {(n,s,m) for n in plan['instances'] for s in plan['config']['algorithm_seeds'] for m in METHODS}
    if set(keyed) != expected or len(keyed) != len(rows): raise ValueError('Missing/duplicate grid rows')
    for name,seed,_ in expected:
        b = keyed[name,seed,'BASE']
        for m in METHODS:
            r = keyed[name,seed,m]; validate(r,m,seed,plan['sha256'])
            if r['scenario_digest'] != b['scenario_digest'] or r.get('pairing') != b.get('pairing'):
                raise ValueError('Scenario or seed pairing failed')
    def compare(method, ref, seed):
        pairs=[]
        for n in plan['instances']:
            for s in plan['config']['algorithm_seeds']:
                v,b = keyed[n,s,method]['final_hypervolume'],keyed[n,s,ref]['final_hypervolume']
                pairs.append({'instance':n,'algorithm_seed':s,'absolute':v-b,'percent':100*(v/b-1) if b>0 else None})
        absolute=paired_summary(pairs,'absolute',seed)
        percent=paired_summary(pairs,'percent',seed)
        if percent['undefined_pairs']:
            percent.update(mean=None,hierarchical_bootstrap_95ci_low=None,hierarchical_bootstrap_95ci_high=None)
        return {'reference':ref,'percent':percent,'absolute':absolute}
    methods={};table=[]
    for i,m in enumerate(METHODS):
        selected=[r for r in rows if r['strategy']==m]
        vsbase=compare(m,'BASE',962000+i); vsrule=compare(m,RULES[m],963000+i) if m in RULES else None
        seconds=statistics.fmean(r['optimisation_seconds'] for r in selected)
        methods[m]={'vs_base':vsbase,'vs_rule':vsrule,'mean_seconds':seconds,
                    'mean_hv':statistics.fmean(r['final_hypervolume'] for r in selected),
                    'actual_evaluations':sum(r['real_objective_evaluation_count'] for r in selected)}
        p=vsbase['percent']
        table.append(dict(method=m,mean_percent_vs_base=p['mean'],ci_low=p['hierarchical_bootstrap_95ci_low'],
              ci_high=p['hierarchical_bootstrap_95ci_high'],mean_absolute_vs_base=vsbase['absolute']['mean'],
              rule=RULES.get(m),mean_percent_vs_rule=vsrule['percent']['mean'] if vsrule else None,
              rule_ci_low=vsrule['percent']['hierarchical_bootstrap_95ci_low'] if vsrule else None,
              rule_ci_high=vsrule['percent']['hierarchical_bootstrap_95ci_high'] if vsrule else None,
              wins=vsbase['absolute']['wins'],ties=vsbase['absolute']['ties'],losses=vsbase['absolute']['losses'],mean_seconds=seconds))
    with (root/'RESULT_TABLE.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
    result={'completed':True,'reference_method':'BASE','design':plan['config'],'searches':len(rows),'methods':methods,
            'qualification':'S0/pop20/B400 development replication only; unchanged old initialization rewrites all slots. No inference about B20000 or independent final-test generalization.',
            'search_objective_evaluations':sum(r['real_objective_evaluation_count'] for r in rows),
            'hidden_candidate_objective_evaluations':0,'new_training_steps':0,'new_ppo_updates':0,
            'interrupted_attempts':len(list((root/'interrupted').glob('*'))),
            'cost_qualification':'Completed-run costs only; preserved interrupted attempts may add cost.',
            'conclusion':'FROZEN_B400_REPLICATION_COMPLETE',
            'next_action':'Review paired Base and rule comparisons; no automatic tuning, larger-budget search or PPO.'}
    write_json(root/'QUICK_RESULT.json',result);write_json(root/'ALL_RUNS.json',rows)
    return result


def execute_jobs(root,plan,workers=1):
    jobs=[(n,s,m) for n in plan['instances'] for s in plan['config']['algorithm_seeds'] for m in METHODS]
    active={};rows=[];started=time.monotonic();last=0
    def interrupt(*_): raise KeyboardInterrupt('Interrupted; resume with same command')
    old_handlers={sig:signal.signal(sig,interrupt) for sig in (signal.SIGTERM,signal.SIGINT)}
    try:
        while jobs or active:
            if time.monotonic()-started>28800:raise TimeoutError('Eight-hour invocation limit; completed runs reusable')
            while jobs and len(active)<workers:
                n,s,m=jobs.pop(0);folder=root/'runs'/n/f'seed_{s}'/m
                if (folder/'RUN_COMPLETE.json').exists():
                    row=read_json(folder/'RUN_COMPLETE.json');validate(row,m,s,plan['sha256'])
                    if row['instance']!=n:raise ValueError('Wrong resumed instance')
                    rows.append(row);continue
                if folder.exists():
                    dest=root/'interrupted'/f'{n}_{s}_{m}_{time.time_ns()}';dest.parent.mkdir(exist_ok=True);folder.rename(dest)
                folder.mkdir(parents=True)
                logfile=root/'logs'/f'{n}_{s}_{m}_{time.time_ns()}.log';logfile.parent.mkdir(exist_ok=True)
                handle=logfile.open('w')
                proc=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),'--out',str(root),'--worker',
                    '--instance',n,'--seed',str(s),'--method',m,'--run-out',str(folder),'--sha',plan['sha256']],stdout=handle,stderr=subprocess.STDOUT)
                active[str(folder)]=(proc,handle,time.monotonic(),logfile)
                print(f'[B400] start {n}/{s}/{m} PID={proc.pid}',flush=True)
            for folder,(proc,handle,began,logfile) in list(active.items()):
                if time.monotonic()-began>900:raise TimeoutError(f'Worker exceeded 15 minutes: {logfile}')
                if proc.poll() is None:continue
                handle.close();del active[folder]
                if proc.returncode:raise RuntimeError(f'Worker failed: {logfile}\n'+logfile.read_text(errors='replace')[-4000:])
                row=read_json(Path(folder)/'RUN_COMPLETE.json');rows.append(row)
                print(f'[B400] complete {len(rows)}/{plan["runs"]} {row["strategy"]} HV={row["final_hypervolume"]:.7g} seconds={row["optimisation_seconds"]:.1f}',flush=True)
            if time.monotonic()-last>=30:
                status={'completed':False,'finished':len(rows),'total':plan['runs'],'running':len(active),'elapsed_seconds':round(time.monotonic()-started)}
                write_json(root/'STATUS.json',status);print(f'[B400] heartbeat {len(rows)}/{plan["runs"]} running={len(active)}',flush=True);last=time.monotonic()
            if active:time.sleep(.2)
        result=report(root,plan,rows)
        write_json(root/'STATUS.json',{'completed':True,'finished':len(rows),'total':plan['runs'],'elapsed_seconds':time.monotonic()-started})
        print('[B400] COMPLETE; see RESULT_TABLE.csv and QUICK_RESULT.json',flush=True)
        return result
    except BaseException as exc:
        write_json(root/'STATUS.json',{'completed':False,'finished':len(rows),'total':plan['runs'],'error':str(exc)})
        raise
    finally:
        for proc,handle,*_ in active.values():
            if proc.poll() is None:proc.terminate()
        for proc,handle,*_ in active.values():
            try:proc.wait(timeout=10)
            except subprocess.TimeoutExpired:proc.kill();proc.wait()
            handle.close()
        for sig,handler in old_handlers.items():signal.signal(sig,handler)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,default=HERE.parent/'transformer_ppo_runs/diagnostics/ccp_component_recheck_b400_v1')
    p.add_argument('--source-catalog',type=Path,default=Path('/home/lunet/cokt2/uncertainty-transformer-ppo/expanda/transformer_ppo_runs/instances/catalog.json'))
    p.add_argument('--workers',type=int,choices=(1,2),default=1)
    p.add_argument('--prepare',action='store_true')
    p.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    p.add_argument('--instance');p.add_argument('--seed',type=int);p.add_argument('--method',choices=METHODS)
    p.add_argument('--run-out',type=Path);p.add_argument('--sha')
    a=p.parse_args();root=a.out.resolve()
    if a.worker:
        worker(root,a.instance,a.seed,a.method,a.run_out,a.sha);return
    root.mkdir(parents=True,exist_ok=True)
    with (root/'.pipeline.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        prepare_models(root)
        cat=prepare_instances(root,a.source_catalog.resolve(),HERE.parent/'transformer_ppo_runs/diagnostics')
        plan=make_plan(root,cat)
        print(f'[B400] frozen 9 checkpoints; {plan["runs"]} searches; no training/PPO; workers={a.workers}',flush=True)
        if not a.prepare:execute_jobs(root,plan,a.workers)

if __name__=='__main__':main()
