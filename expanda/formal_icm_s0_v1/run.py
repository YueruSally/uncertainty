#!/usr/bin/env python3
"""Frozen S0 formal component selection. Use --smoke first, then --run (resumable)."""
import argparse
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '1'
os.environ['MPLBACKEND'] = 'Agg'
sys.path.insert(0, str(ROOT/'src/support'))
sys.path.insert(0, str(ROOT/'src'))

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(assets):
    missing = []
    required = [assets / 'reward_normalization.json',
                assets / 'search_instances/catalog.json']
    required += [assets / f'search_instances/prescreen-search-S0-{i:03d}.json'
                 for i in range(1, 4)]
    required += [assets / c / 'BUNDLE.json' for c in ('I', 'C', 'M')]
    missing.extend(str(p) for p in required if not p.is_file())
    if missing:
        return missing, {}
    bundles = {}
    for component in ('I', 'C', 'M'):
        original = json.loads((assets / component / 'BUNDLE.json').read_text())
        relocated = copy.deepcopy(original)
        spec = relocated['models']['200'] if component == 'M' else relocated
        if len(spec['checkpoints']) != 3:
            raise ValueError(f'{component}: expected three frozen checkpoints')
        names, hashes = [], {}
        for old in spec['checkpoints']:
            path = assets / component / Path(old).name
            if not path.is_file():
                missing.append(str(path))
                continue
            expected = spec['checkpoint_sha256'][old]
            if digest(path) != expected:
                raise ValueError(f'Checkpoint hash mismatch: {path}')
            names.append(str(path.resolve()))
            hashes[str(path.resolve())] = expected
        spec['checkpoints'], spec['checkpoint_sha256'] = names, hashes
        bundles[component] = relocated
    if missing:
        return missing, {}
    prepared = ROOT / 'runtime_inputs'
    prepared.mkdir(exist_ok=True)
    paths = {}
    for component, bundle in bundles.items():
        path = prepared / f'{component}_BUNDLE.json'
        payload = json.dumps(bundle, sort_keys=True, indent=2) + '\n'
        if not path.exists() or path.read_text() != payload:
            path.write_text(payload)
        paths[component] = path
    return [], paths


def verify(bundles):
    sys.path.insert(0, str(ROOT / 'src/support'))
    sys.path.insert(0, str(ROOT / 'src'))
    import numpy as np
    import torch
    from initialization import FrozenIResidualGate
    from crossover import FrozenCResidualGate
    from mutation import FrozenScaleRanker
    torch.set_num_threads(1)
    policies = [FrozenIResidualGate(json.loads(bundles['I'].read_text()), digest),
                FrozenCResidualGate(json.loads(bundles['C'].read_text()), digest),
                FrozenScaleRanker(json.loads(bundles['M'].read_text()), digest)]
    for policy in policies:
        scaler = policy.scaler
        batch = torch.zeros(1, 8, 2, len(scaler['batch']['mean']))
        features = torch.zeros(1, 8, len(scaler['candidate']['mean']))
        context = torch.zeros(1, len(scaler['context']['mean']))
        marker = torch.zeros(1, 8, 1); marker[:, 0, 0] = 1
        with torch.inference_mode():
            for model in policy.models:
                values = model(batch, features, context, marker)
                assert values.shape == (1, 8) and torch.isfinite(values).all()
    print('通过：9 个冻结模型哈希、加载及前向计算；未运行搜索。', flush=True)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT/'experiment.json')
    parser.add_argument('--out', type=Path, default=ROOT/'formal_runs')
    parser.add_argument('--source-catalog', type=Path)
    modes=parser.add_mutually_exclusive_group()
    for name in ('verify','prepare','smoke','run','oos-worker'):
        modes.add_argument('--'+name, action='store_true')
    parser.add_argument('--instance')
    parser.add_argument('--seed', type=int)
    args=parser.parse_args(); args.config=args.config.resolve(); args.out=args.out.resolve()
    from runtime_common import read_json, write_json
    from formal_inputs import validate_config, prepare_instances, verify_catalog
    config=read_json(args.config); validate_config(config)
    if args.oos_worker:
        catalog=verify_catalog(ROOT/'assets/formal_instances',config,ROOT/'assets/data/data_expanded.xlsx')
        if args.instance not in [r['instance_id'] for r in catalog['instances']] or args.seed not in config['algorithm_seeds']:
            raise ValueError('OOS worker not in the frozen grid')
        from formal_oos import evaluate_seed
        evaluate_seed(ROOT,args.out,config,args.instance,args.seed)
        return 0
    args.out.mkdir(parents=True,exist_ok=True)
    with (args.out/'.pipeline.lock').open('w') as lock:
        try: fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError: raise RuntimeError('This output is already running. Do not start a duplicate process.')
        missing,bundles=prepare(ROOT/'assets')
        if missing: raise FileNotFoundError('Missing frozen inputs: '+str(missing))
        print('[FORMAL] 9 frozen checkpoint hashes verified; no PPO or refitting.',flush=True)
        if args.verify:
            verify(bundles); return 0
        if not (args.prepare or args.smoke or args.run):
            print('Use --smoke for eight full-budget checks, then --run for the remaining grid and OOS.',flush=True)
            return 0
        catalog=prepare_instances(ROOT,config,args.source_catalog)
        if args.prepare:
            print('[FORMAL] Fresh S0 catalog frozen; no objective evaluations.',flush=True); return 0
        if args.run and not (args.out/'SMOKE_REPORT.json').exists():
            raise ValueError('Run --smoke first to check all eight methods at the full budget and measure runtime.')
        command=[sys.executable,'-u',str(ROOT/'src/experiment.py'),'--config',str(args.config),
            '--data',str(ROOT/'assets/data/data_expanded.xlsx'),
            '--search-instances',str(ROOT/'assets/formal_instances'),
            '--normalization',str(ROOT/'assets/reward_normalization.json'),
            '--i-bundle',str(bundles['I']),'--c-bundle',str(bundles['C']),'--m-bundle',str(bundles['M']),
            '--out',str(args.out),'--workers',str(config['workers'])]
        if args.smoke:command.append('--smoke')
        subprocess.run(command,cwd=ROOT/'src',check=True)
        if args.smoke:
            result=read_json(args.out/'SMOKE_REPORT.json')
            print(json.dumps({k:result[k] for k in ('seconds_per_method','estimated_search_serial_hours','estimated_search_parallel_hours_ideal','estimate_excludes_oos_and_parallel_contention')},indent=2),flush=True)
            print('[FORMAL] Smoke complete. Continue unchanged with: python3 -u run.py --run',flush=True)
        else:
            from formal_oos import run_oos
            run_oos(ROOT,args.out,config,args.config)
        return 0

if __name__=='__main__':
    raise SystemExit(main())
