#!/usr/bin/env python3
"""Collect a frozen K8 Replace panel under CCP100/pop20/B400."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import torch

from m_operator_offline import digest, read_json, write_json
from m_prescreen_k8 import CollectBalancedPolicy, CollectK8Controller, MAX_STATES
from run_learning_components_deterministic import _normalizer
from transformer_ppo.episode import run_episode
from transformer_ppo.instance_catalog import canonical_digest, read_instance

SOURCE = Path(__file__).resolve().parent
PROTOCOL = SOURCE/'experiments/m_prescreen_protocol_v1.json'


def log(message):
    print(f'[M-K8-COLLECT] {message}', flush=True)


def worker(args):
    torch.set_num_threads(1)
    instance, _ = read_instance(args.instance)
    policy = CollectBalancedPolicy(instance['instance_id'], args.seed)
    holders = []

    def factory(*a, **kw):
        controller = CollectK8Controller(*a, **kw)
        holders.append(controller)
        return controller

    result = run_episode(args.data, args.instance, args.out, policy,
        _normalizer(args.normalization), args.seed, evaluation_budget=400,
        pop_size=20, generations=1000, evaluation_mode='ccp', component='M',
        scenario_count=100, ccp_confidence=.9,
        uncertainty_profile='transport_border_saa_v1',
        inference_controller_factories={'M': factory})
    rows = holders[0].feedback_rows
    extra = holders[0].feedback_evaluations
    if (result['real_objective_evaluation_count'] != 400 or
            result['ppo_updates'] or result['counterfactual_evaluation_count'] or
            not rows or len(rows) > MAX_STATES or
            extra != sum(row['additional_objective_evaluations'] for row in rows) or
            not result['selected_candidate_execution_checks']):
        raise RuntimeError('K8 collection budget, execution or feedback mismatch')
    if any(row['candidate_sources'][0] != 'balanced' or
           not 1 < len(row['labels']) <= 8 or
           len(row['labels']) != len(row['candidate_batch_tokens']) or
           len({label['fingerprint'] for label in row['labels']}) != len(row['labels'])
           for row in rows):
        raise RuntimeError('K8 data pool has invalid or duplicate candidates')
    write_json(args.out/'FEEDBACK.json', {'rows': rows,
        'additional_objective_evaluations': extra})
    result.update(protocol=('m-prescreen-k8-collection-v1'
        if args.design == 'surrogate-v1' else
        'm-prescreen-k8-ranker-validation-v2'),
        algorithm_seed=args.seed, instance_id=instance['instance_id'],
        plan_sha256=args.plan_sha,
        feedback_sha256=digest(args.out/'FEEDBACK.json'),
        feedback_state_count=len(rows),
        total_objective_evaluations_including_training_feedback=400+extra)
    write_json(args.out/'RUN_COMPLETE.json', result)
    log(f"complete {instance['instance_id']} seed={args.seed} states={len(rows)} extra={extra}")


def validate_run(folder, instance, seed, plan_hash, expected_protocol):
    result = read_json(folder/'RUN_COMPLETE.json')
    panel = read_json(folder/'FEEDBACK.json')
    config = read_json(folder/'configuration.json')
    if (result.get('protocol') != expected_protocol or
            result.get('instance_id') != instance or
            result.get('algorithm_seed') != seed or
            result.get('plan_sha256') != plan_hash or
            result.get('real_objective_evaluation_count') != 400 or
            result.get('feedback_sha256') != digest(folder/'FEEDBACK.json') or
            result.get('feedback_state_count') != len(panel['rows']) or
            result.get('training_feedback_evaluations') !=
                panel['additional_objective_evaluations'] or
            config.get('instance_id') != instance or
            config.get('evaluation_mode') != 'ccp' or
            config.get('population') != 20 or
            config.get('evaluation_budget') != 400 or
            config.get('training_size') != 100 or
            config.get('alpha') != .9 or
            config.get('uncertainty', {}).get('name') !=
                'transport_border_saa_v1' or
            config.get('uncertainty', {}).get('scenario_count') != 100 or
            not config.get('scenario_digest')):
        raise ValueError(f'invalid completed collection run: {folder}')
    return result


def run_task(task):
    root, name, seed, plan_sha, args = task
    folder = root/'runs'/name/f'seed_{seed}'
    expected_protocol = ('m-prescreen-k8-collection-v1'
        if args.design == 'surrogate-v1' else
        'm-prescreen-k8-ranker-validation-v2')
    if (folder/'RUN_COMPLETE.json').exists():
        return validate_run(folder, name, seed, plan_sha, expected_protocol)
    if folder.exists():
        interrupted = root/'interrupted_runs'/name/f'seed_{seed}_{int(time.time_ns())}'
        interrupted.parent.mkdir(parents=True, exist_ok=True)
        folder.rename(interrupted)
        log(f'quarantined incomplete {name} seed={seed}: {interrupted}')
    folder.parent.mkdir(parents=True, exist_ok=True)
    folder.mkdir()
    log(f'start {name} seed={seed}; log={root/"logs"/f"{name}_{seed}.log"}')
    command = [sys.executable, '-u', str(Path(__file__).resolve()), '--worker',
        '--instance', str(args.instances/(name+'.json')),
        '--data', str(args.data), '--normalization', str(args.normalization),
        '--out', str(folder), '--seed', str(seed), '--plan-sha', plan_sha,
        '--design', args.design]
    env = {**os.environ, 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
           'OPENBLAS_NUM_THREADS': '1'}
    started = time.monotonic()
    with (root/'logs'/f'{name}_{seed}.log').open('w') as handle:
        process = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT,
            env=env, timeout=300, check=False)
    if process.returncode != 0:
        raise RuntimeError(f'{name} seed={seed} failed exit={process.returncode}; '
                           f'see {root/"logs"/f"{name}_{seed}.log"}')
    log(f'{name} seed={seed} elapsed={time.monotonic()-started:.1f}s')
    return validate_run(folder, name, seed, plan_sha, expected_protocol)


def execute(args):
    protocol = read_json(PROTOCOL)
    final_catalog = read_json(args.final_test_catalog)
    identity = dict(final_catalog)
    identity.pop('catalog_digest', None)
    final_plan = protocol['final_test']
    if (canonical_digest(identity) != final_catalog.get('catalog_digest') or
            final_catalog.get('catalog_digest') != final_plan['catalog_digest'] or
            final_catalog.get('source_catalog_digest') !=
                final_plan['source_catalog_digest'] or
            final_catalog.get('source_data_sha256') !=
                final_plan['network_workbook_sha256'] or
            final_catalog.get('master_seed') != final_plan['master_seed'] or
            final_catalog.get('ccp_seed_base') != final_plan['ccp_seed_base'] or
            final_catalog.get('oos_seed_base') != final_plan['oos_seed_base'] or
            final_catalog.get('instance_count') != final_plan['instance_count'] or
            final_catalog.get('optimisation_outcomes_used') is not False or
            final_catalog.get('model_selection_outcomes_used') is not False):
        raise ValueError('final-test catalog is not the precommitted unopened panel')
    if args.design == 'surrogate-v1':
        names = (protocol['collection']['train_instances'] +
                 protocol['collection']['validation_instances'])
        seeds = protocol['collection']['collection_seeds']
        plan_protocol = 'm-prescreen-k8-collection-v1'
        roles = {'train_instances': protocol['collection']['train_instances'],
                 'validation_instances': protocol['collection']['validation_instances']}
        next_action = 'train a frozen objective surrogate on the saved K8 panel; do not open final-test outcomes'
    else:
        design = protocol['ranker_v2']
        names = design['validation_instances']
        seeds = design['validation_collection_seeds']
        plan_protocol = 'm-prescreen-k8-ranker-validation-v2'
        roles = {'train_instances': [],
                 'validation_instances': design['validation_instances']}
        next_action = 'train the precommitted direct-gain ranker using original training data and this new validation panel'
    inputs = [PROTOCOL, args.data, args.normalization,
        args.final_test_catalog,
        SOURCE/'m_prescreen_k8.py', SOURCE/'m_candidate_information.py',
        SOURCE/'transformer_ppo/controller.py',
        SOURCE/'data/border_crossing_events.csv']
    for name in names:
        path = args.instances/(name+'.json')
        instance, _ = read_instance(path)
        if instance['instance_id'] != name or instance['source_data_sha256'] != digest(args.data):
            raise ValueError(f'wrong instance or source workbook: {name}')
        inputs.append(path)
    plan = {'protocol': plan_protocol, 'design': args.design,
        'instances': names, 'seeds': seeds, 'scenario_count': 100,
        'population': 20, 'search_budget': 400,
        'states_per_run_cap': MAX_STATES,
        'maximum_extra_evaluations': len(names)*len(seeds)*MAX_STATES*7,
        'inputs_sha256': {str(p.resolve()): digest(p) for p in inputs}}
    root = args.out.resolve()
    root.mkdir(parents=True, exist_ok=True)
    if (root/'PLAN.json').exists() and read_json(root/'PLAN.json') != plan:
        raise ValueError('frozen K8 collection inputs changed; choose a new output')
    write_json(root/'PLAN.json', plan)
    plan_sha = digest(root/'PLAN.json')
    (root/'logs').mkdir(exist_ok=True)
    tasks = [(root, name, seed, plan_sha, args) for name in names for seed in seeds]
    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        pending = {executor.submit(run_task, task): task for task in tasks}
        for future in as_completed(pending):
            results.append(future.result())
            write_json(root/'STATUS.json', {'completed_runs': len(results),
                'expected_runs': len(tasks),
                'states': sum(r['feedback_state_count'] for r in results),
                'additional_objective_evaluations': sum(
                    r['training_feedback_evaluations'] for r in results)})
    summary = {'completed': True, 'run_count': len(results),
        **roles,
        'state_count': sum(r['feedback_state_count'] for r in results),
        'additional_objective_evaluations': sum(
            r['training_feedback_evaluations'] for r in results),
        'search_objective_evaluations': 400*len(results),
        'interrupted_run_directories': sum(1 for _ in (
            root/'interrupted_runs').rglob('seed_*')) if (
            root/'interrupted_runs').exists() else 0,
        'uncommitted_interrupted_evaluations_in_total': False,
        'candidate_count_distribution': {str(n): sum(len(row['labels']) == n
            for name in names for seed in seeds for row in read_json(
                root/'runs'/name/f'seed_{seed}'/'FEEDBACK.json')['rows'])
            for n in range(2,9)},
        'new_ppo_updates': 0,
        'next_action': next_action}
    write_json(root/'QUICK_RESULT.json', summary)
    (root/'COLLECTION_COMPLETE').write_text(plan_sha+'\n')
    log(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--instances', type=Path)
    parser.add_argument('--data', type=Path,
        default=SOURCE/'data/data_expanded.xlsx')
    parser.add_argument('--normalization', type=Path)
    parser.add_argument('--final-test-catalog', type=Path)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--design', choices=('surrogate-v1', 'ranker-v2'),
                        default='surrogate-v1')
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--instance', type=Path)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--plan-sha')
    args = parser.parse_args()
    if args.worker:
        worker(args)
    else:
        if not all((args.instances, args.normalization, args.final_test_catalog,
                    args.out)) or not 1 <= args.workers <= 2:
            parser.error('requires --instances, --normalization, '
                         '--final-test-catalog, --out and workers=1..2')
        lock_path = args.out.resolve().with_suffix('.lock')
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with lock_path.open('w') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            execute(args)


if __name__ == '__main__':
    main()
