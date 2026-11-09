"""Portable frozen nine-arm B400 reproduction. Default action only verifies files."""
import argparse
import csv
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parent
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[key]='1'
os.environ['MPLBACKEND']='Agg'
sys.path.insert(0,str(ROOT/'src/support'))
sys.path.insert(0,str(ROOT/'src'))
from runtime_common import read_json,write_json,digest,_normalizer
from transformer_ppo.instance_catalog import canonical_digest
import hv_report as hv


def verify():
    manifest=read_json(ROOT/'assets/manifest.json')
    for rel,sha in manifest['files'].items():
        path=(ROOT/rel).resolve()
        if ROOT not in path.parents or digest(path)!=sha:
            raise ValueError(f'Package checksum mismatch: {rel}')
    print(f'[PACKAGE] {len(manifest["files"])} files verified',flush=True)
    return read_json(ROOT/'assets/design.json')


def bundles():
    paths={}
    for c in 'ICM':
        value=read_json(ROOT/'assets/models'/c/'BUNDLE.json')
        spec=value['models']['200'] if c=='M' else value
        names=[];hashes={}
        for old in spec['checkpoints']:
            path=(ROOT/'assets/models'/c/Path(old).name).resolve()
            if digest(path)!=spec['checkpoint_sha256'][old]:raise ValueError(f'Weight changed: {path}')
            names.append(str(path));hashes[str(path)]=digest(path)
        spec['checkpoints']=names;spec['checkpoint_sha256']=hashes
        paths[c]=ROOT/'runtime_inputs'/f'{c}_BUNDLE.json'
        write_json(paths[c],value)
    return paths


def read_records(path):
    with gzip.open(path,'rt',encoding='utf-8') as f:
        for line in f:yield json.loads(line)


def load_saved_runs(source):
    design=read_json(ROOT/'assets/design.json')
    records=list(read_records(source/'raw_runs.jsonl.gz'))
    expected={(n,s,m) for n in design['instances'] for s in design['algorithm_seeds'] for m in design['methods']}
    keyed={};rows=[];scenarios={}
    for record in records:
        r,c,front=record['run'],record['configuration'],record['front']
        key=(r['instance'],r['algorithm_seed'],r['strategy'])
        if key in keyed or key not in expected:raise ValueError('Unexpected or duplicate saved run')
        if (not r['completed'] or r['real_objective_evaluation_count']!=400 or r['evaluation_budget']!=400
                or r['scenario_digest']!=c['scenario_digest']):raise ValueError('Invalid saved run')
        keyed[key]=r;scenarios.setdefault(key[:2],set()).add(r['scenario_digest'])
        points=[]
        for item in front:
            if item['training_feasible'] is not True or item['source_run']!=r['instance']:
                raise ValueError('Invalid saved feasible front')
            p=tuple(float(item['optimisation_objectives'][k]) for k in hv.OBJECTIVES)
            if not all(math.isfinite(v) for v in p):raise ValueError('Nonfinite objective')
            points.append(p)
        if set(hv.nondominated(points))!=set(points):raise ValueError('Front contains dominated points')
        norm=c['normalization'];lo=norm['minimum'];hi=norm['maximum'];div=norm['objective_divisor']
        if tuple(norm['reference'])!=hv.REFERENCE:raise ValueError('Reference changed')
        old=hv.exact_hv_3d([[(p[j]/div[j]-lo[j])/(hi[j]-lo[j]) for j in range(3)] for p in points],hv.REFERENCE)
        if not math.isclose(old,r['final_hypervolume'],rel_tol=1e-10,abs_tol=1e-12):raise ValueError('Old HV replay failed')
        rows.append(dict(instance=key[0],algorithm_seed=key[1],method=key[2],points=points,
                         old_hv=r['final_hypervolume'],replayed_old_hv=old,seconds=r['optimisation_seconds']))
    if set(keyed)!=expected or any(len(v)!=1 for v in scenarios.values()):raise ValueError('Incomplete/unpaired run grid')
    order={key:i for i,key in enumerate((n,s,m) for n in design['instances'] for s in design['algorithm_seeds'] for m in design['methods'])}
    rows.sort(key=lambda r:order[r['instance'],r['algorithm_seed'],r['method']])
    plan={**design,'base_bootstrap_seed':962000,'rule_bootstrap_seed':963000,
          'protocol':'frozen-old-components-b400-icm-extension-v1'}
    return plan,rows,{str((source/'raw_runs.jsonl.gz').resolve()):digest(source/'raw_runs.jsonl.gz'),
                      str(ROOT/'assets/design.json'):digest(ROOT/'assets/design.json')}


def recompute(source,out):
    # Reuse original normalization, HV and bootstrap implementation unchanged.
    hv.load_runs=load_saved_runs
    return hv.execute(source,out)


def validate_run(r,method,seed,sha):
    if (r.get('completed') is not True or r['strategy']!=method or r['algorithm_seed']!=seed
            or r['plan_sha256']!=sha or r['real_objective_evaluation_count']!=400 or r['evaluation_budget']!=400
            or not math.isfinite(r['final_hypervolume']) or r['final_hypervolume']<0 or r.get('ppo_updates',0)):
        raise ValueError('Invalid replay run')
    hidden=[v for k,v in r.items() if 'hidden_candidate_objective_evaluations' in k]
    hidden.append(r.get('k8_scale_search',{}).get('hidden_candidate_objective_evaluations',0))
    if any(v!=0 for v in hidden):raise ValueError('Hidden objective evaluation')


def worker(a,design):
    import torch
    import experiment as engine
    torch.set_num_threads(1);b=bundles()
    settings=dict(population_size=20,evaluation_budget=400,maximum_generations=1000,scenario_count=100,ccp_confidence=.9)
    engine.SETTINGS=settings;engine.validate_run=validate_run
    sha=digest(ROOT/'assets/manifest.json')
    common=dict(data=ROOT/'assets/data/data_expanded.xlsx',instance=ROOT/'assets/instances'/f'{a.instance}.json',
                out=a.out,seed=a.seed,method=a.method,plan_sha=sha,
                normalization=ROOT/'assets/reward_normalization.json',i_bundle=b['I'],c_bundle=b['C'],m_bundle=b['M'])
    if a.method!='I_C_M_K8_HEURISTIC':
        engine.run_worker(SimpleNamespace(**common))
    else:
        from combined_rule import CombinedRuleController
        from mutation import ScaleSearchPolicy
        from transformer_ppo.archive import RewardArchive,RewardConfig
        from transformer_ppo.episode import run_episode
        policy=ScaleSearchPolicy(a.instance,a.seed,'HEURISTIC_BALANCED',predictor=None)
        def factory(*args,**kwargs):
            norm=kwargs.pop('normalizer');reliable=kwargs.pop('reliable_options');batches_=kwargs.pop('batches');arcs=kwargs.pop('arcs')
            return CombinedRuleController(*args,trainer=policy,archive=RewardArchive(norm),reward_config=RewardConfig(),
                batches=batches_,generations=1000,pop_size=20,graph_observation={},reliable_options=reliable,arcs=arcs,**kwargs)
        result=run_episode(common['data'],common['instance'],a.out,None,_normalizer(common['normalization']),
            algorithm_seed=a.seed,evaluation_budget=400,pop_size=20,generations=1000,evaluation_mode='ccp',component='BASE',
            scenario_count=100,ccp_confidence=.9,uncertainty_profile='transport_border_saa_v1',
            inference_controller_factories={'BASE':factory},method_override=a.method)
        result.update(strategy=a.method,instance=a.instance,algorithm_seed=a.seed,plan_sha256=sha,
                      scenario_digest=read_json(a.out/'configuration.json')['scenario_digest'],new_training_steps=0,new_ppo_updates=0)
        validate_run(result,a.method,a.seed,sha);write_json(a.out/'RUN_COMPLETE.json',result)
    record=read_json(a.out/'RUN_COMPLETE.json')
    ref=next(r['run'] for r in read_records(ROOT/'results/raw_runs.jsonl.gz') if
             (r['run']['instance'],r['run']['algorithm_seed'],r['run']['strategy'])==(a.instance,a.seed,a.method))
    if record['scenario_digest']!=ref['scenario_digest']:raise ValueError('Replay scenarios differ from original')


def replay(a,design):
    import platform,numpy,torch
    actual=dict(python=platform.python_version(),numpy=str(numpy.__version__),torch=str(torch.__version__))
    if actual!=design['environment'] and not a.allow_environment_change:
        raise ValueError(f'Use original environment {design["environment"]}; current {actual}. --allow-environment-change permits a documented portability check.')
    jobs=[(n,s,m) for n in design['instances'] for s in design['algorithm_seeds'] for m in design['methods']
          if (a.instance is None or n==a.instance) and (a.seed is None or s==a.seed) and (a.method is None or m==a.method)]
    if not jobs:raise ValueError('Selection is outside the frozen grid')
    if a.out.resolve()==(ROOT/'results').resolve():raise ValueError('Preserve packaged results')
    a.out.mkdir(parents=True,exist_ok=True)
    identity=dict(package_sha256=digest(ROOT/'assets/manifest.json'),environment=actual,jobs=jobs)
    marker=a.out/'REPLAY_PLAN.json'
    if marker.exists() and read_json(marker)!=json.loads(json.dumps(identity)):raise ValueError('Replay plan changed; use a new output')
    write_json(marker,identity);bundles()
    for i,(n,s,m) in enumerate(jobs,1):
        folder=a.out/'runs'/n/f'seed_{s}'/m
        if (folder/'RUN_COMPLETE.json').exists():
            validate_run(read_json(folder/'RUN_COMPLETE.json'),m,s,identity['package_sha256'])
            print(f'[REPLAY] {i}/{len(jobs)} reused {n}/{s}/{m}',flush=True);continue
        if folder.exists():raise ValueError(f'Incomplete attempt preserved: {folder}; use a new output directory')
        folder.mkdir(parents=True)
        logfile=a.out/'logs'/f'{n}_{s}_{m}.log';logfile.parent.mkdir(exist_ok=True)
        with logfile.open('w') as log:
            p=subprocess.run([sys.executable,'-u',str(ROOT/'run.py'),'--worker','--instance',n,'--seed',str(s),
                '--method',m,'--out',str(folder)],stdout=log,stderr=subprocess.STDOUT,timeout=900)
        if p.returncode:raise RuntimeError(f'Worker failed: {logfile}\n'+logfile.read_text(errors='replace')[-6000:])
        print(f'[REPLAY] {i}/{len(jobs)} complete {n}/{s}/{m}',flush=True)
    if len(jobs)==900:
        with gzip.open(a.out/'raw_runs.jsonl.gz','wt',encoding='utf-8') as f:
            for n,s,m in jobs:
                folder=a.out/'runs'/n/f'seed_{s}'/m
                r={key:read_json(folder/file) for key,file in [('run','RUN_COMPLETE.json'),('configuration','configuration.json'),('front','final_feasible_nondominated.json')]}
                f.write(json.dumps(r,separators=(',',':'),allow_nan=False)+'\n')
        recompute(a.out,a.out.parent/(a.out.name+'_hv'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    g=p.add_mutually_exclusive_group();g.add_argument('--verify',action='store_true');g.add_argument('--recompute',action='store_true');g.add_argument('--run',action='store_true')
    p.add_argument('--out',type=Path);p.add_argument('--method');p.add_argument('--instance');p.add_argument('--seed',type=int)
    p.add_argument('--allow-environment-change',action='store_true');p.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    a=p.parse_args();design=verify()
    if a.worker:worker(a,design);return
    if a.run:
        a.out=(a.out or ROOT/'new_runs').resolve();replay(a,design)
    elif a.recompute:recompute(ROOT/'results',(a.out or ROOT/'recomputed_hv').resolve())
    else:
        with tempfile.TemporaryDirectory(prefix='b400-package-verify-') as t:
            out=Path(t)/'hv';recompute(ROOT/'results',out)
            for name in ('RESULT_TABLE.csv','PER_INSTANCE.csv','PER_RUN.csv'):
                with (out/name).open(encoding='utf-8-sig') as f:x=list(csv.DictReader(f))
                with (ROOT/'results'/name).open(encoding='utf-8-sig') as f:y=list(csv.DictReader(f))
                if x!=y:raise ValueError(f'Saved statistics changed: {name}')
        bundles();print(f'[PACKAGE] All {len(design["instances"])*len(design["algorithm_seeds"])*len(design["methods"])} original HV values and three result tables reproduced; no search or training.',flush=True)

if __name__=='__main__':main()
