#!/usr/bin/env python3
"""Bounded training-only screen of persistent Replace preference profiles.

Balanced is the unchanged executor and ranking. Alternatives add one extra
copy of cost, emission, or time-risk to its score from generation 10 onward.
This is a new action set, not a learned result, PPO, or an oracle deployment.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import fcntl
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import threading
import time

SOURCE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE))

import numpy as np
import torch
from m_candidate_information import CandidatePolicy, CandidateController
from m_operator_offline import digest, read_json, write_json
from multi_instance_logging import fingerprint
from run_learning_components_deterministic import _normalizer
from run_m_operator_closed_loop_v1 import load_run
from transformer_ppo.controller import PPOController
from transformer_ppo.episode import run_episode
from transformer_ppo.instance_catalog import read_instance, canonical_digest
from transformer_ppo.observations import joint_balanced_candidate_scores

PROFILES = ('BALANCED', 'COST', 'EMISSION', 'TIME_RISK')
WEIGHTS = {'BALANCED': [1, 1, 1], 'COST': [2, 1, 1],
           'EMISSION': [1, 2, 1], 'TIME_RISK': [1, 1, 2]}
INSTANCES = ('train-S0-002', 'train-S0-005', 'train-S0-009')
SEEDS = (903001, 903002, 903003)


def log(message):
    print(f'[M-PROFILE] {message}', flush=True)


def profile_index(rows, profile):
    """Legal target only; no objective evaluation, sampling, or child generation."""
    if profile not in PROFILES:
        raise ValueError('unknown Replace profile')
    if profile == 'BALANCED':
        return PPOController._joint_balanced_candidate_index(rows)
    indices = [i for i, row in enumerate(rows) if row.get('eligible')]
    if not indices:
        return None
    def finite(value, default=0.):
        try:
            value = float(value)
        except (TypeError, ValueError):
            return default
        return value if math.isfinite(value) else default
    raw = []
    for i in indices:
        row = rows[i]
        quantity = finite(row.get('batch_quantity'))
        raw.append([quantity*finite(row.get('replacement_delta_cost_per_teu')),
            quantity*finite(row.get('replacement_delta_emission_per_teu')),
            finite(row.get('replacement_delta_nominal_time_h')),
            finite(row.get('replacement_max_lateness_h'), 1e9),
            -finite(row.get('replacement_on_time_probability'))])
    raw = np.asarray(raw, dtype=float)
    spans = np.ptp(raw, axis=0)
    spans[spans <= 1e-15] = 1.
    scaled = (raw-raw.min(axis=0))/spans
    additional = {'COST': scaled[:, 0], 'EMISSION': scaled[:, 1],
                  'TIME_RISK': scaled[:, 2:5].mean(axis=1)}[profile]
    # Reuse the exact existing Balanced arithmetic and tie breaker.
    balanced = joint_balanced_candidate_scores(rows)
    scores = {i: balanced[i]+float(extra) for i, extra in zip(indices, additional)}
    return min(indices, key=lambda i: (scores[i],
        int(rows[i]['target'].batch_id), str(rows[i].get('path_id'))))


class ProfileController(CandidateController):
    def __init__(self, *args, **kwargs):
        self.profile = kwargs['trainer'].replace_profile
        self.profile_decisions = self.profile_deviations = 0
        self.first_profile_state = None
        super().__init__(*args, **kwargs)

    def _balanced_replace_target(self, ind, batches, reliable_options):
        if self.generation < 10:
            raise RuntimeError('profile must not alter the pre-G10 search')
        if self.first_profile_state is None:
            self.first_profile_state = {'generation': self.generation, 'evaluation': self.evaluations,
                'parent': fingerprint(ind),
                'population': [fingerprint(item) for item in self.current_population]}
        self.profile_decisions += 1
        if self.profile == 'BALANCED':
            return super()._balanced_replace_target(ind, batches, reliable_options)
        rows = self._operator_target_rows(ind, batches, 'replace', None, None, None, reliable_options)
        selected = profile_index(rows, self.profile)
        if selected is None:
            return None, None, {'eligible_candidate_count': 0, 'selector': self.profile}
        self.profile_deviations += int(selected != self._joint_balanced_candidate_index(rows))
        row = rows[selected]
        batch = next(b for b in batches if b.batch_id == row['target'].batch_id)
        return batch, row['target'], {'eligible_candidate_count': sum(bool(r.get('eligible')) for r in rows),
            'selector': self.profile, 'selected_candidate_index': selected,
            'selected_batch_id': int(batch.batch_id), 'selected_path_id': str(row.get('path_id'))}

    def summary(self):
        return {**super().summary(), 'replace_profile': self.profile,
            'profile_decisions': self.profile_decisions, 'profile_deviations': self.profile_deviations,
            'first_profile_state': self.first_profile_state}


def run_one(a):
    torch.set_num_threads(1)
    info, _ = read_instance(a.instance)
    policy = CandidatePolicy(info['instance_id'], a.seed, 'HEURISTIC_BALANCED')
    policy.replace_profile = a.profile
    result = run_episode(a.data, a.instance, a.out, policy, _normalizer(a.normalization), a.seed,
        evaluation_budget=400, pop_size=20, generations=1000, evaluation_mode='ccp', component='M',
        scenario_count=100, ccp_confidence=.9, uncertainty_profile='transport_border_saa_v1',
        inference_controller_factories={'M': ProfileController})
    result.update(strategy=a.profile, algorithm_seed=a.seed, plan_sha256=a.plan_sha,
        learning_enabled=False, new_training_steps=0,
        role='training-only action-set screen; no Transformer/PPO performance claim')
    validate_run(result)
    write_json(a.out/'RUN_COMPLETE.json', result)


def validate_run(row):
    if (row['real_objective_evaluation_count'] != 400 or row['evaluation_budget'] != 400
            or row.get('ppo_updates', 0) or row.get('counterfactual_evaluation_count', 0)
            or row.get('candidate_audit_evaluation_count', 0) or row.get('training_feedback_evaluations', 0)
            or row.get('forced_operator_candidate_evaluations', 0)
            or row['minimum_control_generation'] != 10):
        raise ValueError('profile run changed budget, schedule, or evaluated hidden alternatives')
    if not math.isfinite(row['final_hypervolume']) or row['final_hypervolume'] <= 0:
        raise ValueError('profile comparison requires positive finite HV')


def analyze(runs, instances=INSTANCES, seeds=SEEDS):
    expected = {(n, s, p) for n in instances for s in seeds for p in PROFILES}
    keyed = {(r['instance'], r['algorithm_seed'], r['strategy']): r for r in runs}
    if len(keyed) != len(runs) or set(keyed) != expected:
        raise ValueError('incomplete or duplicate frozen profile grid')
    matrix, table = {}, []
    for name in instances:
        matrix[name] = {}
        for seed in seeds:
            baseline = keyed[name, seed, 'BALANCED']
            gains = []
            for profile in PROFILES:
                row = keyed[name, seed, profile]
                validate_run(row)
                if row['replace_profile'] != profile:
                    raise ValueError('profile identity mismatch')
                if (row['pairing'] != baseline['pairing']
                        or row['first_profile_state'] != baseline['first_profile_state']):
                    raise ValueError('initial population, scenarios or pre-G10 state mismatch')
                gains.append(100*(row['final_hypervolume']/baseline['final_hypervolume']-1))
            matrix[name][seed] = gains
        for i, profile in enumerate(PROFILES):
            rows = [keyed[name, s, profile] for s in seeds]
            table.append({'instance': name, 'profile': profile,
                'mean_hv': statistics.fmean(r['final_hypervolume'] for r in rows),
                'mean_seconds': statistics.fmean(r['optimisation_seconds'] for r in rows),
                'mean_paired_percent_vs_balanced': statistics.fmean(matrix[name][s][i] for s in seeds)})
    # Select on two seeds, score on the third. Rotate without ever using the
    # held-out return to choose a profile. These remain TRAINING diagnostics.
    rotations, per_instance = [], []
    for name in instances:
        local = []
        for heldout in seeds:
            training = [s for s in seeds if s != heldout]
            means = [statistics.fmean(matrix[name][s][i] for s in training) for i in range(4)]
            choice = max(range(4), key=lambda i: (means[i], -i))  # Balanced wins exact ties.
            row = {'instance': name, 'heldout_seed': heldout, 'selection_seeds': training,
                'selected_profile': PROFILES[choice], 'heldout_gain_percent': matrix[name][heldout][choice]}
            rotations.append(row)
            local.append(row['heldout_gain_percent'])
        per_instance.append({'instance': name, 'cross_seed_gain_percent': statistics.fmean(local)})
    mean = statistics.fmean(r['cross_seed_gain_percent'] for r in per_instance)
    positive = sum(r['cross_seed_gain_percent'] > 0 for r in per_instance)
    screen = mean >= .5 and positive >= 2
    return {'completed': True, 'table': table, 'per_instance': per_instance,
        'cross_seed_selection': rotations, 'cross_seed_mean_gain_percent': mean,
        'screen_pass': screen, 'automatic_training': False,
        'conclusion': 'PROFILE_ACTION_SET_WORTH_LEARNING_TRIAL' if screen else 'STOP_PROFILE_ACTION_SET',
        'qualification': '36 training searches on three fixed instances; cross-seed selection uses known instance identity; not an independent test, neural policy, statistical proof, or deployment oracle',
        'new_searches': len(runs), 'new_objective_evaluations': sum(r['real_objective_evaluation_count'] for r in runs),
        'search_optimization_seconds_sum': sum(r['optimisation_seconds'] for r in runs),
        'new_training_steps': 0, 'new_ppo_updates': 0}


def frozen_plan(a):
    previous = read_json(a.source/'PLAN.json')
    if (previous['protocol'] != 'terminal-feedback-training-coverage-v1'
            or not (a.source/'PIPELINE_COMPLETE').exists()):
        raise ValueError('requires completed 12-instance coverage experiment')
    if not set(INSTANCES) <= set(previous['train_instances']) or set(INSTANCES) & set(
            previous['validation_instances']+previous['test_instances']):
        raise ValueError('screen must use predeclared training instances only')
    files = [a.source/'PLAN.json', a.source/'QUICK_RESULT.json', Path(__file__),
             Path(previous['data']), Path(previous['normalization']), SOURCE/'data/border_crossing_events.csv']
    seen = set()
    for name in INSTANCES:
        path = Path(previous['instances'])/(name+'.json')
        info, _ = read_instance(path)
        signature = canonical_digest(info['batches'])
        if (info['instance_id'] != name or info['source_data_sha256'] != digest(previous['data'])
                or info['configuration'] != 'S0' or signature in seen):
            raise ValueError('instance identity, workbook or unique batch-set mismatch')
        seen.add(signature)
        files.append(path)
    files += sorted(SOURCE.glob('*.py')) + sorted((SOURCE/'transformer_ppo').glob('*.py'))
    return {'protocol': 'persistent-replace-profile-screen-v1',
        'data': previous['data'], 'normalization': previous['normalization'], 'instances_root': previous['instances'],
        'instances': list(INSTANCES), 'seeds': list(SEEDS), 'profiles': list(PROFILES), 'weights': WEIGHTS,
        'scenario_count': 100, 'ccp_confidence': .9, 'population_size': 20, 'evaluation_budget': 400,
        'minimum_generation': 10, 'run_count': 36, 'maximum_completed_evaluations': 14400,
        'workers': a.workers, 'threads_per_worker': 1, 'invocation_seconds': 900, 'run_seconds': 120,
        'torch_version': str(torch.__version__), 'numpy_version': str(np.__version__),
        'automatic_retries': 0, 'screen_rule': 'cross-seed mean >=0.5 percent and positive in at least 2 of 3 training instances; no automatic neural training',
        'method_change': 'replace one-off operator switching with a fixed Replace ranking profile at every post-G10 mutation',
        'execution_contract': 'unchanged concrete-proposal-v1; same private proposal random streams and exact frozen child execution as terminal-feedback Balanced',
        'input_sha256': {str(p.resolve()): digest(p) for p in files}}


def execute(a):
    plan = frozen_plan(a)
    root = a.out
    if (root/'PLAN.json').exists() and read_json(root/'PLAN.json') != plan:
        raise ValueError('frozen inputs or protocol changed; retain old output')
    write_json(root/'PLAN.json', plan)
    if (root/'PIPELINE_COMPLETE').exists():
        log('already complete; no new searches; see RESULT_TABLE.csv and QUICK_RESULT.json')
        return
    sha = digest(root/'PLAN.json')
    started = time.monotonic()
    deadline = started+plan['invocation_seconds']
    stop, lock = threading.Event(), threading.Lock()
    status = {'stage': 'search', 'completed_runs': 0, 'total_runs': plan['run_count']}
    (root/'logs').mkdir(exist_ok=True)
    env = {**os.environ, 'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'}
    log(f"{plan['run_count']} searches; CCP100/pop20/B400; {len(plan['instances'])} training instances x {len(plan['seeds'])} seeds x 4 fixed profiles")

    def group(name, seed):
        rows = []
        offset = plan['seeds'].index(seed) % 3
        others = PROFILES[1:]
        order = ('BALANCED',)+others[offset:]+others[:offset]
        for profile in order:
            if stop.is_set() or time.monotonic() >= deadline:
                raise TimeoutError('screen stopped; completed results retained')
            folder = root/'runs'/name/f'seed_{seed}'/profile
            reused = (folder/'RUN_COMPLETE.json').exists()
            if not reused:
                if folder.exists() and any(folder.iterdir()):
                    destination = root/'interrupted_runs'/f'{name}_{seed}_{profile}_{time.time_ns()}'
                    destination.parent.mkdir(exist_ok=True)
                    folder.rename(destination)
                logfile = root/'logs'/f'{name}_{seed}_{profile}_{time.time_ns()}.log'
                command = [sys.executable, '-u', str(Path(__file__).resolve()), '--worker',
                    '--instance', str(Path(plan['instances_root'])/(name+'.json')), '--data', plan['data'],
                    '--normalization', plan['normalization'], '--out', str(folder), '--seed', str(seed),
                    '--profile', profile, '--plan-sha', sha]
                with logfile.open('w') as handle, subprocess.Popen(command, cwd=SOURCE, env=env,
                        stdout=handle, stderr=subprocess.STDOUT) as proc:
                    begin = heartbeat = time.monotonic()
                    try:
                        while proc.poll() is None:
                            now = time.monotonic()
                            if stop.is_set() or now >= deadline or now-begin >= plan['run_seconds']:
                                raise TimeoutError(f'run limit reached; see {logfile}')
                            if now-heartbeat >= 15:
                                log(f'running {name}/{seed}/{profile} pid={proc.pid} elapsed={now-begin:.0f}s')
                                heartbeat = now
                            time.sleep(.2)
                    finally:
                        if proc.poll() is None:
                            proc.kill()
                            proc.wait()
                    if proc.returncode:
                        raise RuntimeError(f'worker failed: {logfile}\n'+'\n'.join(logfile.read_text(errors='replace').splitlines()[-15:]))
            row = load_run(folder, name, sha)
            if row['strategy'] != profile or row['algorithm_seed'] != seed:
                raise ValueError('saved run identity mismatch')
            validate_run(row)
            rows.append(row)
            with lock:
                status['completed_runs'] += 1
                status['elapsed_seconds'] = time.monotonic()-started
                write_json(root/'STATUS.json', status)
                log(f"{status['completed_runs']}/{plan['run_count']} {'reused' if reused else 'complete'} {name}/{seed}/{profile} HV={row['final_hypervolume']:.7g} seconds={row['optimisation_seconds']:.2f} changed_targets={row['profile_deviations']}")
        return rows

    runs = []
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        futures = [pool.submit(group, name, seed) for seed in plan['seeds'] for name in plan['instances']]
        try:
            for future in as_completed(futures):
                runs.extend(future.result())
        except BaseException:
            stop.set()
            for future in futures:
                future.cancel()
            raise
    report = analyze(runs, instances=plan['instances'], seeds=plan['seeds'])
    report['elapsed_seconds_this_invocation'] = time.monotonic()-started
    report['interrupted_runs'] = len(list((root/'interrupted_runs').glob('*')))
    report['uncommitted_interruption_cost_excluded'] = bool(report['interrupted_runs'])
    report['timing_scope'] = 'two equal single-thread workers; exploratory concurrent time, not isolated timing benchmark'
    write_json(root/'COMPLETE.json', {**report, 'runs': runs})
    write_json(root/'QUICK_RESULT.json', report)
    with (root/'RESULT_TABLE.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(report['table'][0]))
        writer.writeheader()
        writer.writerows(report['table'])
    write_json(root/'STATUS.json', {**status, 'stage': 'complete', 'conclusion': report['conclusion']})
    (root/'PIPELINE_COMPLETE').write_text(report['conclusion']+'\n')
    print((root/'RESULT_TABLE.csv').read_text(), flush=True)
    log(f"COMPLETE cross_seed={report['cross_seed_mean_gain_percent']:+.4f}% {report['conclusion']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--workers', choices=(1, 2), default=2, type=int)
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--instance', type=Path)
    parser.add_argument('--data', type=Path)
    parser.add_argument('--normalization', type=Path)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--profile', choices=PROFILES)
    parser.add_argument('--plan-sha')
    a = parser.parse_args()
    a.out = a.out.resolve()
    if a.worker:
        if any(getattr(a, name) is None for name in ('instance', 'data', 'normalization', 'seed', 'profile', 'plan_sha')):
            parser.error('worker arguments incomplete')
        run_one(a)
        return
    if a.source is None:
        parser.error('--source is required')
    a.source = a.source.resolve()
    a.out.mkdir(parents=True, exist_ok=True)
    with (a.out/'.lock').open('w') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            execute(a)
        except Exception as exc:
            status = read_json(a.out/'STATUS.json') if (a.out/'STATUS.json').exists() else {}
            write_json(a.out/'STATUS.json', {**status, 'stage': 'stopped', 'error': str(exc)})
            log(f'STOPPED {exc}')
            raise


if __name__ == '__main__':
    main()
