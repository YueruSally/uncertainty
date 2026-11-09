#!/usr/bin/env python3
"""Bounded diagnostic of seed preservation and saved crossover gate selection.

Never changes the original protocol, weights, inputs or screen results.
"""
import argparse
from collections import Counter
import csv
import fcntl
from pathlib import Path
import signal
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'component_screen_b20_v1'))
from common import ROOT, read_json, write_json, digest, canonical_digest, read_instance
from controller import ScreenController
from learning import normalizer
from multi_instance_logging import fingerprint
from transformer_ppo.episode import run_episode
import baseline_uncertainty as base
import numpy as np
import torch

CONFIGURATIONS = ('S1', 'S0', 'S8')  # 10/tight, 20/normal, 30/loose; developer diagnosis.
SEEDS = (951001, 951002)
ARMS = ('BASE_INIT', 'I_ALL', 'I_KEEP_FIXED')


def fixed_count(population):
    seeds = int(round(population * base.FEASIBILITY_SEED_FRACTION))
    if base.FEASIBILITY_SEED_FRACTION > 0 and seeds == 0:
        seeds = 1
    seeds = min(population, max(0, seeds))
    remaining = max(0, population - seeds)
    each = max(1, remaining // 6) if remaining else 0
    return min(population, seeds + 3 * each)


class DiagnosticController(ScreenController):
    def __init__(self, *args, diagnostic_arm, **kwargs):
        self.diagnostic_arm = diagnostic_arm
        self.initial_record = None
        super().__init__(*args, arm='BASE' if diagnostic_arm == 'BASE_INIT' else 'I_RULE', **kwargs)
        self.keep_count = fixed_count(self.settings['population'])

    def has_budget(self, required):
        if self.diagnostic_arm == 'BASE_INIT' and self.generation >= 0:
            return False
        return super().has_budget(required)

    def initial_individual_from_base(self, ind, batches, path_lib, max_paths=3, slot_index=None):
        if self.diagnostic_arm == 'I_KEEP_FIXED' and slot_index < self.keep_count:
            self.slot = slot_index
            return ind
        return super().initial_individual_from_base(ind, batches, path_lib, max_paths, slot_index)

    def log_initial_population(self, population):
        super().log_initial_population(population)
        identities = [fingerprint(ind) for ind in population]
        counts = Counter(identities)
        self.initial_record = {
            'population': len(population), 'fixed_slots': self.keep_count,
            'fingerprints': identities,
            'unique_individuals': len(counts), 'largest_duplicate_count': max(counts.values()),
            'feasible_fraction': sum(ind.feasible for ind in population) / len(population),
            'initial_hv': self._hv(population),
            'search_evaluations': self.evaluations,
        }
        write_json(self.out / 'INITIALIZATION.json', self.initial_record)


def run_worker(source, out, name, seed, arm, sha, settings=None):
    torch.set_num_threads(1)
    settings = settings or read_json(source / 'PLAN.json')['protocol']
    info, _ = read_instance(source / 'instances' / (name + '.json'))
    if info['development_role'] != 'screen':
        raise ValueError('Diagnostic requires a development screen instance')
    holders = []
    def factory(*args, **kwargs):
        ctl = DiagnosticController(*args, settings=settings, diagnostic_arm=arm, **kwargs)
        holders.append(ctl)
        return ctl
    saved_clear = base.clear_inherited_evaluation_cache
    base.clear_inherited_evaluation_cache = lambda *individuals: None
    try:
        result = run_episode(
            ROOT.parent / 'formal_icm_s0_v1/assets/data/data_expanded.xlsx',
            source / 'instances' / (name + '.json'), out, None,
            normalizer(read_json(source / 'models/HV_NORMALIZATION.json')),
            algorithm_seed=seed, evaluation_budget=settings['budget'],
            pop_size=settings['population'], generations=1000000,
            evaluation_mode='ccp', component='BASE', scenario_count=settings['scenario_count'],
            ccp_confidence=settings['confidence'], uncertainty_profile='transport_border_saa_v1',
            inference_controller_factories={'BASE': factory}, method_override=arm)
    finally:
        base.clear_inherited_evaluation_cache = saved_clear
    ctl = holders[0]
    if arm != 'BASE_INIT' and ctl.evaluations not in (settings['budget'] - 1, settings['budget']):
        raise ValueError('Incomplete diagnostic search budget')
    if not ctl.initial_record or ctl.initial_record['population'] != settings['population']:
        raise ValueError('Incomplete initialization')
    result.update(instance=name, seed=seed, arm=arm, diagnostic_plan_sha256=sha,
                  scenario_digest=read_json(out / 'configuration.json')['scenario_digest'],
                  initialization=ctl.initial_record, new_training_steps=0,
                  new_ppo_updates=0, stop_reason=('initialization_only' if arm == 'BASE_INIT' else result['stop_reason']))
    write_json(out / 'RUN_COMPLETE.json', result)
    return result


def crossover_audit(source):
    bundle = read_json(source / 'models/C/BUNDLE.json')
    report = read_json(source / 'models/C/VALIDATION.json')
    choices = report['choices']
    selected = min(choices, key=lambda x: (-x['mean_gain'], x['switch_fraction'], x['epoch'], x['lambda'], x['threshold_scaled']))
    if selected != report['selected'] or selected != bundle['gate']:
        raise ValueError('Saved C gate does not reproduce its predeclared selection rule')
    enabled = [c for c in choices if not c['disabled']]
    best = min(enabled, key=lambda x: (-x['mean_gain'], x['switch_fraction'], x['epoch'], x['lambda'], x['threshold_scaled']))
    zero_gate = [c for c in enabled if c['lambda'] == 0 and c['threshold_scaled'] == 0]
    return {'selection_reproduced_exactly': True, 'selected': selected,
            'best_enabled_choice': best, 'zero_threshold_choices': zero_gate,
            'enabled_choices_with_positive_validation_gain': sum(c['mean_gain'] > 0 for c in enabled),
            'training_states': bundle['training_states'], 'validation_states': bundle['validation_states'],
            'new_objective_evaluations': 0, 'new_training_steps': 0,
            'qualification': 'Checks saved validation selection only; cannot establish whether representation or data caused failure.'}


def freeze_plan(source, out):
    old = read_json(source / 'PLAN.json')
    if old['protocol']['budget'] != 20000 or old['protocol']['population'] != 250:
        raise ValueError('This diagnostic requires original B20000/pop250 experiment')
    for relative, sha in old['files'].items():
        if digest(ROOT.parent / relative) != sha:
            raise ValueError(f'Original experiment code/input changed: {relative}')
    from run import verify_frozen
    frozen = verify_frozen(source)
    if frozen['plan_sha256'] != old['sha256']:
        raise ValueError('Frozen model assets belong to a different source plan')
    cat = read_json(source / 'instances/catalog.json')
    cases = []
    files = [source / 'PLAN.json', source / 'instances/catalog.json', source / 'models/HV_NORMALIZATION.json',
             source / 'models/C/BUNDLE.json', source / 'models/C/VALIDATION.json', Path(__file__)]
    for config in CONFIGURATIONS:
        names = sorted(r['instance_id'] for r in cat['instances'] if r['configuration'] == config and r['development_role'] == 'screen')
        name = names[0]
        files.append(source / 'instances' / (name + '.json'))
        for seed in SEEDS:
            cases.append({'configuration': config, 'instance': name, 'seed': seed})
            for arm in ('BASE', 'I_RULE'):
                files.append(source / 'screen' / name / f'seed_{seed}' / arm / 'RUN_COMPLETE.json')
    for record in cases:
        for arm in ('BASE', 'I_RULE'):
            r = read_json(source / 'screen' / record['instance'] / f'seed_{record["seed"]}' / arm / 'RUN_COMPLETE.json')
            if not r['completed'] or r['plan_sha256'] != old['sha256']:
                raise ValueError('Source run does not belong to frozen original experiment')
    value = {'source_plan': old['sha256'], 'cases': cases, 'arms': list(ARMS),
             'budget': 20000, 'population': 250, 'maximum_new_searches': 12,
             'initialization_only_runs': 6, 'maximum_new_solution_evaluations': 241500,
             'files': {str(p.resolve()): digest(p) for p in files},
             'criterion': 'Paired mean final HV difference KEEP_FIXED minus ALL, and wins; descriptive only, no automatic admission or refitting.',
             'purpose': 'Development diagnostic; not independent confirmation or proof of seed preservation causing all historical differences.'}
    value['sha256'] = canonical_digest(value)
    path = out / 'PLAN.json'
    if path.exists() and read_json(path) != value:
        raise ValueError('Diagnostic plan changed; preserve existing output')
    write_json(path, value)
    return value


def summarize(source, out, plan, rows):
    indexed = {(r['instance'], r['seed'], r['arm']): r for r in rows}
    paired = []
    table = []
    for case in plan['cases']:
        name, seed = case['instance'], case['seed']
        b, a, k = [indexed[name, seed, arm] for arm in ARMS]
        old = read_json(source / 'screen' / name / f'seed_{seed}' / 'I_RULE/RUN_COMPLETE.json')
        baseline = read_json(source / 'screen' / name / f'seed_{seed}' / 'BASE/RUN_COMPLETE.json')
        if len({r['scenario_digest'] for r in (b, a, k, old, baseline)}) != 1:
            raise ValueError('Scenario pairing failed')
        if a['evaluation_order_digest'] != old['evaluation_order_digest'] or a['final_hypervolume'] != old['final_hypervolume']:
            raise ValueError('Original I_RULE replay differs; stop before attributing difference to seed preservation')
        if b['initial_evaluation_digest'] != baseline['initial_evaluation_digest']:
            raise ValueError('Base initialization replay differs')
        n = b['initialization']['fixed_slots']
        if k['initialization']['fingerprints'][:n] != b['initialization']['fingerprints'][:n]:
            raise ValueError('Protected Base initialization slots were changed')
        delta = k['final_hypervolume'] - a['final_hypervolume']
        record = {**case, 'base_hv': baseline['final_hypervolume'], 'all_hv': a['final_hypervolume'],
                  'keep_fixed_hv': k['final_hypervolume'], 'keep_minus_all_hv': delta,
                  'keep_minus_base_hv': k['final_hypervolume'] - baseline['final_hypervolume']}
        paired.append(record)
        for r in (b, a, k):
            init = r['initialization']
            table.append({**case, 'arm': r['arm'], 'initial_unique': init['unique_individuals'],
                          'initial_largest_duplicate': init['largest_duplicate_count'],
                          'initial_feasible_fraction': init['feasible_fraction'], 'initial_hv': init['initial_hv'],
                          'final_hv': None if r['arm'] == 'BASE_INIT' else r['final_hypervolume'],
                          'evaluations': r['real_objective_evaluation_count'], 'seconds': r['optimisation_seconds']})
    with (out / 'RESULT_TABLE.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(table[0])); w.writeheader(); w.writerows(table)
    delta = np.array([r['keep_minus_all_hv'] for r in paired])
    q = {'completed': True, 'original_I_RULE_replayed_exactly': True, 'base_fixed_slots_preserved_exactly': True,
         'initialization': {'paired_runs': paired, 'mean_keep_minus_all_hv': float(delta.mean()),
                            'wins': int((delta > 1e-12).sum()), 'ties': int((np.abs(delta) <= 1e-12).sum()),
                            'losses': int((delta < -1e-12).sum())},
         'crossover': read_json(out / 'C_GATE_AUDIT.json'),
         'new_objective_evaluations': sum(r['real_objective_evaluation_count'] for r in rows),
         'interrupted_attempts': len(list((out / 'interrupted').glob('*'))),
         'cost_qualification': 'Evaluation total covers completed runs; partial failed/interrupted attempts are preserved separately and may add cost.',
         'new_training_steps': 0, 'new_ppo_updates': 0,
         'conclusion': 'SEED_PRESERVATION_DIRECTIONALLY_HELPFUL_IN_THIS_SMALL_PANEL' if delta.mean() > 0 else 'SEED_PRESERVATION_GAIN_NOT_ESTABLISHED_IN_THIS_SMALL_PANEL',
         'qualification': 'Six development pairs; no claim of statistical confirmation or universal cause. No model, gate, or prior result changed.',
         'next_action': 'Review initialization and gate evidence; no automatic large rerun or PPO.'}
    write_json(out / 'QUICK_RESULT.json', q)
    return q


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    default = ROOT.parent / 'transformer_ppo_runs/diagnostics'
    parser.add_argument('--source', type=Path, default=default / 'ccp_component_screen_b20_v1')
    parser.add_argument('--out', type=Path, default=default / 'ccp_component_screen_b20_smallcheck_v1')
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--instance'); parser.add_argument('--seed', type=int); parser.add_argument('--arm', choices=ARMS)
    parser.add_argument('--run-out', type=Path); parser.add_argument('--sha')
    args = parser.parse_args(); source = args.source.resolve(); out = args.out.resolve()
    if args.worker:
        plan = read_json(out / 'PLAN.json')
        if args.sha != plan['sha256'] or not any(c['instance'] == args.instance and c['seed'] == args.seed for c in plan['cases']):
            raise ValueError('Worker does not match frozen diagnostic plan')
        run_worker(source, args.run_out, args.instance, args.seed, args.arm, args.sha)
        return
    if source == out or source in out.parents:
        raise ValueError('Diagnostic output must be separate from the frozen source')
    out.mkdir(parents=True, exist_ok=True)
    lock = (out / 'pipeline.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    plan = freeze_plan(source, out)
    write_json(out / 'C_GATE_AUDIT.json', crossover_audit(source))
    jobs = [(c['instance'], c['seed'], arm) for c in plan['cases'] for arm in ARMS]
    active = {}; rows = []; last = 0; began = time.monotonic()
    def interrupted(*_):
        raise KeyboardInterrupt('Interrupted; completed runs remain reusable')
    signal.signal(signal.SIGTERM, interrupted); signal.signal(signal.SIGINT, interrupted)
    try:
        while jobs or active:
            if time.monotonic() - began > 7200:
                raise RuntimeError('Two-hour invocation limit; completed runs reusable')
            while jobs and len(active) < 2:
                name, seed, arm = jobs.pop(0)
                folder = out / 'runs' / name / f'seed_{seed}' / arm
                complete = folder / 'RUN_COMPLETE.json'
                if complete.exists():
                    row = read_json(complete)
                    if not row['completed'] or row['diagnostic_plan_sha256'] != plan['sha256'] or (row['instance'], row['seed'], row['arm']) != (name, seed, arm):
                        raise ValueError('Saved diagnostic run plan changed')
                    rows.append(row); continue
                if folder.exists():
                    saved = out / 'interrupted' / f'{name}_{seed}_{arm}_{time.time_ns()}'
                    saved.parent.mkdir(exist_ok=True); folder.rename(saved)
                folder.mkdir(parents=True)
                logpath = out / 'logs' / f'{name}_{seed}_{arm}_{time.time_ns()}.log'
                logpath.parent.mkdir(exist_ok=True); log = logpath.open('w')
                proc = subprocess.Popen([sys.executable, '-u', str(Path(__file__).resolve()), '--source', str(source),
                    '--out', str(out), '--worker', '--instance', name, '--seed', str(seed), '--arm', arm,
                    '--run-out', str(folder), '--sha', plan['sha256']], stdout=log, stderr=subprocess.STDOUT)
                active[str(folder)] = (proc, log, time.monotonic(), logpath)
                print(f'[SMALL-CHECK] start {name}/{seed}/{arm} log={logpath}', flush=True)
            for folder, (proc, log, start, logpath) in list(active.items()):
                if time.monotonic() - start > 2400:
                    raise RuntimeError(f'Worker timeout: {logpath}')
                if proc.poll() is None:continue
                log.close(); del active[folder]
                if proc.returncode:raise RuntimeError(f'Worker failed: {logpath}')
                row = read_json(Path(folder) / 'RUN_COMPLETE.json'); rows.append(row)
                print(f'[SMALL-CHECK] complete {len(rows)}/18 {row["instance"]}/{row["arm"]}', flush=True)
            if time.monotonic() - last >= 30:
                write_json(out / 'STATUS.json', {'completed': False, 'finished': len(rows), 'total': 18, 'running': len(active)})
                print(f'[SMALL-CHECK] heartbeat {len(rows)}/18 running={len(active)}', flush=True); last = time.monotonic()
            if active:time.sleep(1)
        q = summarize(source, out, plan, rows)
        write_json(out / 'STATUS.json', {'completed': True, 'conclusion': q['conclusion']})
        print('[SMALL-CHECK] COMPLETE ' + q['conclusion'], flush=True)
    except BaseException as e:
        write_json(out / 'STATUS.json', {'completed': False, 'error': str(e), 'finished': len(rows)})
        raise
    finally:
        for proc, log, *_ in active.values():
            if proc.poll() is None:proc.terminate()
        for proc, log, *_ in active.values():
            try:proc.wait(timeout=10)
            except subprocess.TimeoutExpired:proc.kill();proc.wait()
            log.close()
        lock.close()

if __name__ == '__main__':main()
