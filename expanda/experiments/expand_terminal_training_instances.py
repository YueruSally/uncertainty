#!/usr/bin/env python3
"""One bounded training-coverage expansion; frozen CCP100/pop20/B400 branches.

Add six predeclared catalogue instances to the six existing training instances.
Keep both validation instances and all development/test cases out of training.
This is data collection plus offline fitting, not a deployed performance claim.
"""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
import csv
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

SOURCE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE))

import numpy as np
import torch
from experiments.m_terminal_feedback_v3 import assert_terminal_pair, expected_score
from experiments.extend_terminal_feedback_training import partition
from experiments.check_terminal_decoding import summarize
from m_operator_offline import read_json, write_json, digest
from m_terminal_hv_ppo import initialize, save_checkpoint, optimizer_for
from run_m_operator_closed_loop_v1 import load_run
from transformer_ppo.instance_catalog import read_instance, canonical_digest

EXTRA_INSTANCES = ('train-S0-011', 'train-S0-015', 'train-S0-017',
                   'train-S0-018', 'train-S0-019', 'train-S0-020')
SEEDS = tuple(range(902001, 902009))
CHECKPOINTS = (40, 80, 120, 200, 300, 400)
INVOCATION_SECONDS = 1800


def log(message):
    print(f'[M-COVERAGE] {message}', flush=True)


def validate_cohort(old_plan, instances, extra):
    reserved = set(old_plan['train_instances'] + old_plan['validation_instances'] + old_plan['test_instances'])
    if len(set(extra)) != len(extra) or set(extra) & reserved:
        raise ValueError('added training instances overlap the frozen training/validation/development sets')
    seen = set()
    for name in sorted(reserved):
        record, _ = read_instance(instances/(name+'.json'))
        seen.add(canonical_digest(record['batches']))
    new_files = []
    for name in extra:
        path = instances/(name+'.json')
        record, _ = read_instance(path)
        signature = canonical_digest(record['batches'])
        if (record['instance_id'] != name or record['source_data_sha256'] != digest(old_plan['data'])
                or record['configuration'] != 'S0' or signature in seen):
            raise ValueError(f'added instance identity/data/unique batch set mismatch: {name}')
        seen.add(signature)
        new_files.append(path)
    return new_files


def freeze(source, out, workers):
    old = read_json(source/'PLAN.json')
    previous = read_json(source/'QUICK_RESULT.json')
    data = read_json(source/'DATASET.json')
    extension = source/'training_extension_400_v1'
    comparator = read_json(extension/'decoding_check_v1/QUICK_RESULT.json')
    extension_plan = read_json(extension/'PLAN.json')
    decode_plan = read_json(extension/'decoding_check_v1/PLAN.json')
    extension_result = read_json(extension/'QUICK_RESULT.json')
    if (old['protocol'] != 'complete-terminal-feedback-v3' or not previous['completed']
            or not (source/'PIPELINE_COMPLETE').exists() or not comparator['completed']
            or data['plan_sha256'] != digest(source/'PLAN.json') or len(data['rows']) != old['contexts']):
        raise ValueError('requires completed source, training extension and frozen decoding check')
    for record in (old, extension_plan, decode_plan):
        for path, sha in record['input_sha256'].items():
            if digest(path) != sha:
                raise ValueError(f'source input changed: {path}')
    if digest(extension_result['checkpoint']) != extension_result['sha256']:
        raise ValueError('old comparator checkpoint changed')
    instances = Path(old['instances'])
    files = validate_cohort(old, instances, EXTRA_INSTANCES)
    # Keep the original held-out instances and original per-instance state count.
    partition(data['rows'], old, Path(old['bundle']))
    counts = Counter(r['instance'] for r in data['rows'])
    if set(counts.values()) != {len(SEEDS)}:
        raise ValueError('source contexts per instance differ from the frozen expansion design')
    fixed = [source/'PLAN.json', source/'DATASET.json', source/'QUICK_RESULT.json',
        extension/'PLAN.json', extension/'QUICK_RESULT.json',
        extension/'decoding_check_v1/PLAN.json', extension/'decoding_check_v1/QUICK_RESULT.json',
        Path(__file__), SOURCE/'experiments/m_terminal_feedback_v3.py',
        SOURCE/'experiments/extend_terminal_feedback_training.py', SOURCE/'experiments/check_terminal_decoding.py']
    plan = {'protocol': 'terminal-feedback-training-coverage-v1', 'source': str(source),
        'bundle': old['bundle'], 'data': old['data'], 'normalization': old['normalization'],
        'instances': old['instances'], 'extra_instances': list(EXTRA_INSTANCES), 'seeds': list(SEEDS),
        'train_instances': old['train_instances']+list(EXTRA_INSTANCES),
        'validation_instances': old['validation_instances'], 'test_instances': old['test_instances'],
        'new_contexts': len(EXTRA_INSTANCES)*len(SEEDS),
        'maximum_new_searches': 5*len(EXTRA_INSTANCES)*len(SEEDS),
        'maximum_new_objective_evaluations': 400*5*len(EXTRA_INSTANCES)*len(SEEDS),
        'search_design': {'scenario_count': 100, 'population_size': 20, 'evaluation_budget': 400,
                          'ccp_confidence': .9, 'single_fixed_intervention': True},
        'optimizer': {'steps': max(CHECKPOINTS), 'learning_rate': 1e-4, 'gradient_clip': 1.,
            'objective': 'same probability-weighted final-HV relative return, scaled using training data only'},
        'initialization': 'original pretraining bundle, identical to the 400-step comparison; no continuation from fitted 400-step weights',
        'candidate_epochs': list(CHECKPOINTS),
        'selection': 'largest mean validation softmax expected return; earlier exact ties',
        'candidate_decoding': 'argmax; exact ties prefer Balanced; unchanged .02 offset and .25 temperature',
        'admission': 'argmax conditional gain positive in each unchanged validation instance; then actual search remains necessary',
        'workers': workers, 'threads_per_worker': 1, 'invocation_seconds': INVOCATION_SECONDS,
        'source_training_and_search_cost': previous['training_summary'],
        'input_sha256': {str(p.resolve()): digest(p) for p in fixed+files}}
    if (out/'PLAN.json').exists() and read_json(out/'PLAN.json') != plan:
        raise ValueError('frozen training coverage plan changed')
    write_json(out/'PLAN.json', plan)
    return plan, data['rows'], comparator


def branch_runner(plan, out, deadline):
    sha = digest(out/'PLAN.json')
    stop = threading.Event()
    env = {**os.environ, 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1'}
    (out/'logs').mkdir(exist_ok=True)

    def branch(name, seed, action):
        folder = out/'branches'/name/f'seed_{seed}'/f'action_{action}'
        if not (folder/'RUN_COMPLETE.json').exists():
            if stop.is_set() or time.monotonic() >= deadline:
                raise TimeoutError('coverage invocation cap reached; completed branches retained')
            if folder.exists() and any(folder.iterdir()):
                interrupted = out/'interrupted_runs'/f'{name}_{seed}_{action}_{time.time_ns()}'
                interrupted.parent.mkdir(exist_ok=True)
                folder.rename(interrupted)
            command = [sys.executable, '-u', str(SOURCE/'experiments/m_terminal_feedback_v3.py'), '--worker',
                '--instance', str(Path(plan['instances'])/(name+'.json')), '--data', plan['data'],
                '--normalization', plan['normalization'], '--bundle', plan['bundle'], '--out', str(folder),
                '--seed', str(seed), '--action', str(action), '--plan-sha', sha]
            logfile = out/'logs'/f'{name}_{seed}_{action}_{time.time_ns()}.log'
            with logfile.open('w') as handle, subprocess.Popen(command, cwd=SOURCE, env=env,
                    stdout=handle, stderr=subprocess.STDOUT) as proc:
                start = last = time.monotonic()
                try:
                    while proc.poll() is None:
                        now = time.monotonic()
                        if stop.is_set() or now >= deadline or now-start > 300:
                            raise TimeoutError(f'branch cap reached: {logfile}')
                        if now-last >= 15:
                            log(f'running {name}/{seed}/action={action} pid={proc.pid} elapsed={now-start:.0f}s')
                            last = now
                        time.sleep(.2)
                finally:
                    if proc.poll() is None:
                        proc.kill()
                        proc.wait()
                if proc.returncode:
                    raise RuntimeError(f'branch failed: {logfile}\n'+'\n'.join(logfile.read_text(errors='replace').splitlines()[-20:]))
        row = load_run(folder, name, sha)
        if (row['fixed_action'] != action or row['algorithm_seed'] != seed
                or row['source_bundle_sha256'] != digest(plan['bundle'])
                or row['input_sha256'] != digest(folder/'INPUT.json')
                or row['fixed_choices'] != 1):
            raise ValueError('branch identity or frozen input mismatch')
        inputs = read_json(folder/'INPUT.json')
        if inputs['action'] != action or not inputs['eligible'][action]:
            raise ValueError('branch action or legality mismatch')
        return (name, seed, action), row, inputs

    def batch(jobs, stage):
        results = []
        with ThreadPoolExecutor(max_workers=plan['workers']) as pool:
            futures = [pool.submit(branch, *job) for job in jobs]
            try:
                for future in as_completed(futures):
                    result = future.result()
                    results.append(result)
                    write_json(out/'STATUS.json', {'stage': stage, 'completed': len(results), 'total': len(jobs)})
                    name, seed, action = result[0]
                    log(f'{stage} {len(results)}/{len(jobs)} {name}/{seed}/action={action}')
            except BaseException:
                stop.set()
                for future in futures:
                    future.cancel()
                raise
        return results
    return batch


def collect(plan, original_rows, out, deadline):
    batch = branch_runner(plan, out, deadline)
    grid = [(n, s) for n in plan['extra_instances'] for s in plan['seeds']]
    baselines = batch([(n, s, 4) for n, s in grid], 'baseline contexts')
    indexed, reference = {}, {}
    for (name, seed, _), row, inputs in baselines:
        assert_terminal_pair(row, row, row['first_supported_input_sha256'])
        if not np.isfinite(row['final_hypervolume']) or row['final_hypervolume'] <= 0:
            raise ValueError('new context has nonpositive baseline HV; do not drop it or move the reference')
        reference[name, seed] = row
        hvs = [None]*5
        hvs[4] = row['final_hypervolume']
        indexed[name, seed] = {'instance': name, 'seed': seed, 'tokens': inputs['tokens'],
            'eligible': inputs['eligible'], 'first_input_sha256': row['first_supported_input_sha256'], 'terminal_hvs': hvs}
    jobs = [(n, s, a) for n, s in grid for a in range(4) if indexed[n, s]['eligible'][a]]
    for (name, seed, action), row, inputs in batch(jobs, 'legal terminal alternatives'):
        context = indexed[name, seed]
        assert_terminal_pair(reference[name, seed], row, context['first_input_sha256'])
        np.testing.assert_array_equal(inputs['tokens'], context['tokens'])
        if inputs['eligible'] != context['eligible']:
            raise ValueError('paired branch mask changed')
        if not np.isfinite(row['final_hypervolume']):
            raise ValueError('non-finite branch outcome')
        context['terminal_hvs'][action] = row['final_hypervolume']
    rows = deepcopy(original_rows)+[indexed[n, s] for n, s in grid]
    partition(rows, plan, Path(plan['bundle']))
    counts = Counter(r['instance'] for r in rows)
    if set(counts.values()) != {len(plan['seeds'])}:
        raise ValueError('incomplete per-instance training panel')
    nsearch = len(grid)+len(jobs)
    if nsearch > plan['maximum_new_searches']:
        raise ValueError('training search cap exceeded')
    write_json(out/'DATASET.json', {'rows': rows, 'plan_sha256': digest(out/'PLAN.json'),
        'new_searches': nsearch, 'new_objective_evaluations': 400*nsearch})
    return rows


def fit(plan, rows, out):
    started = time.perf_counter()
    attempt_path = out/'training_attempts'/f'{time.time_ns()}.json'
    write_json(attempt_path, {'completed': False, 'completed_steps': 0})
    train, valid = partition(rows, plan, Path(plan['bundle']))
    x = torch.tensor([r['tokens'] for r in train], dtype=torch.float32)
    mask = torch.tensor([r['eligible'] for r in train], dtype=torch.bool)
    q = torch.tensor([[0. if value is None else value/r['terminal_hvs'][4]-1
                      for value in r['terminal_hvs']] for r in train], dtype=torch.float32)
    scale = max(float(q[mask].std(unbiased=False)), 1e-4)
    model = initialize(Path(plan['bundle']))
    optimizer = torch.optim.Adam(model.actors.parameters(), lr=1e-4)
    history, selected, weights = [], None, None
    for epoch in range(max(plan['candidate_epochs'])+1):
        if epoch:
            model.eval()
            loss = -(model(x, mask)[0].softmax(-1)*q/scale).sum(-1).mean()
            optimizer.zero_grad()
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.actors.parameters(), 1.)
            if not torch.isfinite(loss) or not torch.isfinite(norm):
                raise FloatingPointError('non-finite training-coverage optimization')
            optimizer.step()
        if epoch == 0 or epoch in plan['candidate_epochs']:
            metrics = {'epoch': epoch, 'training': expected_score(model, train), 'validation': expected_score(model, valid)}
            history.append(metrics)
            # Epoch zero is descriptive; candidates match the frozen 40..400 schedule.
            if epoch in plan['candidate_epochs'] and (selected is None or
                    metrics['validation']['mean_expected_terminal_gain'] > selected['validation']['mean_expected_terminal_gain']+1e-12):
                selected, weights = deepcopy(metrics), deepcopy(model.state_dict())
            write_json(attempt_path, {'completed': False, 'completed_steps': epoch,
                'elapsed_seconds_at_last_checkpoint': time.perf_counter()-started})
            log(f'epoch={epoch}/{max(plan["candidate_epochs"])} train={100*metrics["training"]["mean_expected_terminal_gain"]:+.4f}% '
                f'valid={100*metrics["validation"]["mean_expected_terminal_gain"]:+.4f}%')
    model.load_state_dict(weights)
    save_checkpoint(out/'selected_pretrained.pt', {'protocol': 'terminal-hv-ppo-v1',
        'execution_mode': 'single-intervention-argmax', 'plan_sha256': digest(out/'PLAN.json'),
        'bundle_sha256': digest(plan['bundle']), 'update': 0, 'feature_count': x.shape[-1],
        'ensemble_count': len(model.actors), 'model': weights, 'optimizer': optimizer_for(model).state_dict(),
        'complete_feedback_epoch': selected['epoch'], 'training_kind': 'expanded training coverage; no PPO',
        'qualification': 'argmax deployment has not yet been evaluated in independent search'})
    report = {'selected': selected, 'history': history, 'training_contexts': len(train),
        'validation_contexts': len(valid), 'training_only_reward_scale': scale,
        'elapsed_seconds': time.perf_counter()-started, 'optimizer_steps': max(plan['candidate_epochs']),
        'validation': summarize(model, valid), 'new_ppo_updates': 0,
        'plan_sha256': digest(out/'PLAN.json'), 'dataset_sha256': digest(out/'DATASET.json'),
        'checkpoint': str(out/'selected_pretrained.pt'), 'sha256': digest(out/'selected_pretrained.pt')}
    write_json(out/'TRAINING.json', report)
    write_json(attempt_path, {'completed': True, 'completed_steps': max(plan['candidate_epochs']),
        'elapsed_seconds': report['elapsed_seconds']})
    return report


def execute(source, out, workers):
    torch.set_num_threads(1)
    source, out = source.resolve(), out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    with (out/'.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan, old_rows, old_report = freeze(source, out, workers)
        if (out/'PIPELINE_COMPLETE').exists():
            result = read_json(out/'QUICK_RESULT.json')
            if digest(result['training']['checkpoint']) != result['training']['sha256']:
                raise ValueError('completed model changed')
            print((out/'RESULT_TABLE.csv').read_text(), flush=True)
            log('completed result retained; no new search or training')
            return
        deadline = time.monotonic()+INVOCATION_SECONDS
        log(f'new training instances={plan["extra_instances"]}; contexts={plan["new_contexts"]}; CCP100/pop20/B400')
        rows = collect(plan, old_rows, out, deadline)
        if (out/'TRAINING.json').exists():
            training = read_json(out/'TRAINING.json')
            if (training['plan_sha256'] != digest(out/'PLAN.json')
                    or training['dataset_sha256'] != digest(out/'DATASET.json')
                    or training['sha256'] != digest(training['checkpoint'])):
                raise ValueError('saved training result or data changed')
            log('committed training reused; no additional optimizer steps')
        else:
            training = fit(plan, rows, out)
        receipts = [read_json(p) for p in (out/'branches').glob('**/RUN_COMPLETE.json')]
        dataset = read_json(out/'DATASET.json')
        if len(receipts) != dataset['new_searches'] or any(r['real_objective_evaluation_count'] != 400 for r in receipts):
            raise ValueError('completed search cost does not match the predeclared panel')
        promising = training['validation']['all_instances_argmax_positive']
        interruptions = len(list((out/'interrupted_runs').glob('*')))
        attempts = [read_json(p) for p in (out/'training_attempts').glob('*.json')]
        result = {'completed': True, 'protocol': plan['protocol'], 'training': training,
            'source_validation': old_report['validation'], 'new_search_runs': len(receipts),
            'new_objective_evaluations': 400*len(receipts),
            'new_search_optimization_seconds': sum(r['optimisation_seconds'] for r in receipts),
            'old_experiment_cost': plan['source_training_and_search_cost'], 'new_ppo_updates': 0,
            'optimizer_attempts': attempts,
            'uncommitted_training_cost_may_exist': any(not a['completed'] for a in attempts),
            'interrupted_runs': interruptions, 'uncommitted_interruption_cost_excluded': interruptions > 0,
            'next_action': 'BOUNDED_ARGMAX_SEARCH_REQUIRED' if promising else 'EXPANDED_COVERAGE_DID_NOT_ESTABLISH_GAIN',
            'qualification': 'Development validation only; same validation instances reused. No new deployed search performance or PPO benefit is established.'}
        write_json(out/'QUICK_RESULT.json', result)
        with (out/'RESULT_TABLE.csv').open('w') as handle:
            writer = csv.writer(handle)
            writer.writerow(['training_panel', 'instance', 'softmax_expected_gain_percent', 'argmax_gain_percent', 'argmax_switch_fraction'])
            for label, report in [('ORIGINAL', old_report['validation']), ('EXPANDED', training['validation'])]:
                for row in report['per_instance']:
                    writer.writerow([label, row['instance'], row['softmax_expected_gain_percent'],
                        row['argmax_gain_percent'], row['argmax_switch_fraction']])
        (out/'PIPELINE_COMPLETE').write_text('training coverage complete; independent search still required\n')
        write_json(out/'STATUS.json', {'stage': 'complete', 'next_action': result['next_action']})
        print((out/'RESULT_TABLE.csv').read_text(), flush=True)
        log(f'COMPLETE new_searches={len(receipts)} new_evaluations={400*len(receipts)} {result["next_action"]}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--workers', type=int, choices=(1, 2), default=2)
    args = parser.parse_args()
    out = args.out or args.source.parent/'ccp_m_terminal_coverage_12inst_v1'
    try:
        execute(args.source, out, args.workers)
    except Exception as error:
        write_json(out/'STOPPED.json', {'reason': str(error), 'completed_branches_retained': True})
        raise


if __name__ == '__main__':
    main()
