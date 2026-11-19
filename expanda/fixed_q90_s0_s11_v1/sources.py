"""Pin only immutable completed files. Never read source STATUS as authority."""
from pathlib import Path
import numpy as np
from common import (ARMS, CONFIGS, DATA, OBJECTIVES, canonical, checked_plan,
                    digest, freeze, read, verify)


class Pending(Exception):
    pass


def required(path):
    if not path.is_file():
        raise Pending(str(path))
    return read(path)  # Malformed completed input is an error, not a missing run.


def exact_decision_id(row):
    # Preserve full float precision. Do not merge merely equal objectives or
    # the historical fingerprints rounded to twelve share decimals.
    blocks = []
    for block in row['decision']:
        allocs = [dict(nodes=a['nodes'], modes=a['modes'], share=a['share'])
                  for a in block['allocations']]
        blocks.append(dict(origin=block['origin'], destination=block['destination'],
            batch_id=block['batch_id'], allocations=sorted(allocs, key=canonical)))
    return canonical(sorted(blocks, key=canonical))


def prepare(cfg, source, large, out, runtime):
    ci = CONFIGS.index(cfg)
    root = source if ci < 9 else large
    plan_path = root / 'PLAN.json'
    required(plan_path)
    plan = checked_plan(plan_path)
    for name, sha in runtime.items():
        if name in plan.get('files', {}) and sha != plan['files'][name]:
            raise ValueError('Evaluator differs from the source runtime: ' + name)
    if ci < 9:
        name = f'joint-confirm-{cfg}-001'
        entry = next(e for e in plan['original_entries'] if e['instance_id'] == name)
        seeds = [993001, 993002, 993003]
        candidates = {(r['instance'], r['seed'], r['arm']): r for r in plan['reused']}
        paths = [(arm, seed, Path(candidates[name, seed, arm]['path'])) for seed in seeds for arm in ARMS]
        expected = {str(p): candidates[name, seed, arm]['sha256'] for arm, seed, p in paths}
        model = Path(plan['models']['shared']['path'])
        model_sha = plan['models']['shared']['bundle_sha256']
    else:
        name = f'single-large-{cfg}-001'
        entry = next(e for e in plan['entries'] if e['instance_id'] == name)
        seeds = plan['protocol']['algorithm_seeds']
        if len(seeds) != 3 or len(set(seeds)) != 3:
            raise ValueError('Expected three original algorithm seeds')
        paths = [(arm, seed, root/'runs'/name/f'seed_{seed}'/arm/'RUN_COMPLETE.json'
                 ) for seed in seeds for arm in ARMS]
        expected = {}
        model = Path(plan['models']['ICM_SHARED_SUPERVISED'])
        model_sha = plan['model_metadata']['ICM_SHARED_SUPERVISED']['bundle_sha256']
    if (canonical({k: v for k, v in entry.items() if k != 'instance_digest'}) != entry['instance_digest']
            or entry['configuration'] != cfg or entry['source_data_sha256'] != digest(DATA)):
        raise ValueError('Instance/workbook identity failed: ' + name)
    files = {str(plan_path): digest(plan_path)}
    bundle = read(model/'BUNDLE.json')
    checkpoints = bundle['checkpoints']
    if (digest(model/'BUNDLE.json') != model_sha or len(checkpoints) != 1
            or checkpoints[0] != 'seed_991001_epoch_20.pt'):
        raise ValueError('Expected frozen single Shared SL member 991001, epoch 20')
    for f in [model/'BUNDLE.json', model/checkpoints[0]]:
        files[str(f)] = digest(f)
    if files[str(model/checkpoints[0])] != bundle['checkpoint_sha256'][checkpoints[0]]:
        raise ValueError('Shared checkpoint hash mismatch')
    previous = out.parent / 'S0' / 'INPUTS.json'
    if previous.is_file() and read(previous)['model_bundle_sha256'] != model_sha:
        raise ValueError('Selected configurations used different Shared SL policies')
    solutions, members, source_configs = {}, [], []
    scenario_hashes = set()
    for arm, seed, path in paths:
        run = required(path)
        if not run.get('completed'):
            raise Pending(str(path))
        folder = path.parent
        config = required(folder/'configuration.json')
        rows = required(folder/'final_feasible_nondominated.json')
        for f in [path, folder/'configuration.json', folder/'final_feasible_nondominated.json']:
            files[str(f)] = digest(f)
        if str(path) in expected and files[str(path)] != expected[str(path)]:
            raise ValueError('Historical run changed: ' + str(path))
        if (run['instance'] != name or run['configuration'] != cfg or run['seed'] != seed
                or run['arm'] != arm or run['instance_digest'] != entry['instance_digest']
                or run['evaluation_budget'] != 5000 or run.get('stochastic', False)
                or not 4999 <= run['real_objective_evaluation_count'] <= 5000
                or run.get('model_bundle_sha256') != (model_sha if arm != 'BASE' else None)):
            raise ValueError('Unexpected source run protocol: ' + str(path))
        if (config['population'] != 250 or config['evaluation_budget'] != 5000
                or config['training_size'] != 100 or config['alpha'] != .9
                or config['evaluation_mode'] != 'ccp' or config['instance_id'] != name
                or config['instance_digest'] != entry['instance_digest']
                or config['method'] != arm or config['seeds']['algorithm'] != seed
                or config['seeds']['training'] != entry['ccp100_seed']
                or config['seeds']['path'] != entry['path_seed']
                or config['scenario_digest'] != run['scenario_digest']):
            raise ValueError('Unexpected saved evaluation configuration: ' + str(path))
        if 'final_solutions_sha256' in run and run['final_solutions_sha256'] != files[str(folder/'final_feasible_nondominated.json')]:
            raise ValueError('Saved solutions hash differs')
        raw = [[r['optimisation_objectives'][obj] for obj in OBJECTIVES] for r in rows]
        if not raw or sorted(raw) != sorted(run['final_front_raw']):
            raise ValueError('Saved decisions do not match completed final front: ' + str(path))
        for row in rows:
            if not row['training_feasible'] or not np.isfinite(list(row['optimisation_objectives'].values())).all():
                raise ValueError('Invalid original feasible-front solution')
            key = exact_decision_id(row)
            if key in solutions:
                a = [solutions[key]['optimisation_objectives'][o] for o in OBJECTIVES]
                b = [row['optimisation_objectives'][o] for o in OBJECTIVES]
                if not np.allclose(a, b, rtol=1e-10, atol=1e-8):
                    raise ValueError('Identical decisions have different original objectives')
            else:
                solutions[key] = dict(row, exact_decision_id=key)
            members.append(dict(configuration=cfg, instance=name, method=arm, algorithm_seed=seed,
                solution_id=row['source_solution_id'], exact_decision_id=key))
        scenario_hashes.add(run['scenario_digest'])
        source_configs.append(config)
    if len(scenario_hashes) != 1 or any(c['uncertainty'] != source_configs[0]['uncertainty'] for c in source_configs):
        raise ValueError('The six source fronts do not share one optimization scenario set')
    new_seeds = [280000000 + 100*ci + i for i in range(21)]
    # Known optimization seeds may not overlap any of the new random streams.
    known_seeds = {e['ccp100_seed'] for e in plan.get('entries', []) + plan.get('original_entries', [])}
    if known_seeds & set(new_seeds):
        raise ValueError('New and original scenario seeds overlap')
    result = dict(configuration=cfg, instance=entry, source_root=str(root), files=files,
        model_bundle_sha256=model_sha, source_runs=6, unique_plans=len(solutions),
        source_config=source_configs[0], scenario_digest=next(iter(scenario_hashes)),
        new_seeds=new_seeds, member_rows=members,
        decisions=[solutions[key] for key in sorted(solutions)])
    result['sha256'] = canonical(result)
    verify(files)
    freeze(out/'INPUTS.json', result)
    freeze(out/'instance.json', entry)
    return result
