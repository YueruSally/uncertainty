#!/usr/bin/env python3
"""Bounded 40 -> 400 step check using frozen full-search branch labels only.

No search, new objective evaluation, PPO rollout, feature change or test-instance
selection occurs here. Reconstruct the original Adam trajectory rather than
silently restarting its moments at the old selected checkpoint.
"""
import argparse
import csv
from copy import deepcopy
import fcntl
import json
import math
from pathlib import Path
import sys
import time

SOURCE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE))

import torch
from experiments.m_terminal_feedback_v3 import expected_score
from m_operator_offline import read_json, write_json, digest
from m_terminal_hv_ppo import initialize, load_checkpoint, save_checkpoint, optimizer_for

ADDITIONAL_CHECKPOINTS = (80, 120, 200, 300, 400)
MAX_SECONDS = 120


def log(message):
    print(f'[M-EXTEND] {message}', flush=True)


def partition(rows, plan, bundle):
    names = set(plan['validation_instances'])
    if names & set(read_json(bundle)['training_instances']):
        raise ValueError('validation overlaps original pretraining instances')
    train = [r for r in rows if r['instance'] in plan['train_instances']]
    valid = [r for r in rows if r['instance'] in names]
    if not train or not valid or len(train)+len(valid) != len(rows):
        raise ValueError('frozen training/validation partition is incomplete or overlapping')
    if len({(r['instance'], r['seed']) for r in rows}) != len(rows):
        raise ValueError('duplicate decision context')
    for r in rows:
        hvs, mask = r['terminal_hvs'], r['eligible']
        if len(hvs) != 5 or len(mask) != 5 or not mask[4]:
            raise ValueError('invalid action schema or absent Balanced')
        if any((q is None) == bool(legal) for q, legal in zip(hvs, mask)):
            raise ValueError('missing legal outcome or outcome for illegal action')
        if hvs[4] <= 0 or any(not math.isfinite(v) for v in hvs if v is not None):
            raise ValueError('invalid frozen terminal outcome')
    return train, valid


def check_metric_replay(actual, expected):
    for part in ('training', 'validation'):
        for metric in ('mean_expected_terminal_gain', 'mean_switch_probability'):
            if not math.isclose(actual[part][metric], expected[part][metric], rel_tol=1e-7, abs_tol=1e-9):
                raise ValueError(f'original training metrics did not reproduce: {part}/{metric}')
        if actual[part]['per_instance_expected_gain'].keys() != expected[part]['per_instance_expected_gain'].keys():
            raise ValueError('original instance partition changed')
        for name, gain in expected[part]['per_instance_expected_gain'].items():
            if not math.isclose(actual[part]['per_instance_expected_gain'][name], gain, rel_tol=1e-7, abs_tol=1e-9):
                raise ValueError(f'original instance metric did not reproduce: {name}')


def fit(bundle, rows, source_plan, old, old_checkpoint, out, plan_sha, candidates=ADDITIONAL_CHECKPOINTS):
    started = time.perf_counter()
    train, valid = partition(rows, source_plan, bundle)
    x = torch.tensor([r['tokens'] for r in train], dtype=torch.float32)
    mask = torch.tensor([r['eligible'] for r in train], dtype=torch.bool)
    q = torch.tensor([[0. if v is None else v/r['terminal_hvs'][4]-1 for v in r['terminal_hvs']]
                      for r in train], dtype=torch.float32)
    scale = max(float(q[mask].std(unbiased=False)), 1e-4)
    if scale != old['training_only_reward_scale']:
        raise ValueError('original training-only reward scale changed')
    model = initialize(bundle)
    optimizer = torch.optim.Adam(model.actors.parameters(), lr=1e-4)
    old_model = load_checkpoint(old_checkpoint)
    previous = {r['epoch']: r for r in old['history']}
    old_limit = max(previous)
    if not candidates or min(candidates) <= old_limit:
        raise ValueError('extension checkpoints must follow the original training limit')
    selected = deepcopy(old['selected'])
    selected_state = deepcopy(old_model['model'])
    history, weight_replay = [], False
    for epoch in range(max(candidates)+1):
        if time.perf_counter()-started > MAX_SECONDS:
            raise TimeoutError('offline training time cap reached; no search was launched')
        if epoch:
            model.eval()
            logits, _ = model(x, mask)
            loss = -(logits.softmax(-1)*q/scale).sum(-1).mean()
            optimizer.zero_grad()
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.actors.parameters(), 1.)
            if not torch.isfinite(loss) or not torch.isfinite(norm):
                raise FloatingPointError('non-finite full-feedback optimization')
            optimizer.step()
        if epoch in previous or epoch in candidates:
            metrics = {'epoch': epoch, 'training': expected_score(model, train),
                       'validation': expected_score(model, valid)}
            if epoch in previous:
                check_metric_replay(metrics, previous[epoch])
            if epoch == old['selected']['epoch']:
                state = model.state_dict()
                weight_replay = (state.keys() == selected_state.keys()
                                 and all(torch.equal(state[k], selected_state[k]) for k in state))
                if not weight_replay:
                    raise ValueError('original selected weights did not reproduce exactly')
            if epoch >= old_limit:
                history.append(metrics)
                if metrics['validation']['mean_expected_terminal_gain'] > selected['validation']['mean_expected_terminal_gain']+1e-12:
                    selected, selected_state = deepcopy(metrics), deepcopy(model.state_dict())
                log(f'epoch={epoch}/{max(candidates)} train={100*metrics["training"]["mean_expected_terminal_gain"]:+.4f}% '
                    f'valid={100*metrics["validation"]["mean_expected_terminal_gain"]:+.4f}% '
                    f'valid_switch={100*metrics["validation"]["mean_switch_probability"]:.2f}%')
    if not weight_replay:
        raise ValueError('no authenticated replay of the old selected model')
    model.load_state_dict(selected_state)
    save_checkpoint(out/'selected_pretrained.pt', {**old_model, 'plan_sha256': plan_sha,
        'model': selected_state, 'optimizer': optimizer_for(model).state_dict(), 'update': 0,
        'complete_feedback_epoch': selected['epoch'],
        'training_kind': 'bounded full-feedback training extension; no new PPO'})
    gains = selected['validation']['per_instance_expected_gain']
    promising = (selected['validation']['mean_expected_terminal_gain'] > 0
                 and all(g > 0 for g in gains.values()))
    result = {'completed': True, 'source_selected_epoch': old['selected']['epoch'],
        'source_selected': old['selected'], 'selected': selected, 'history': history,
        'source_training_metrics_reproduced': True, 'source_selected_weights_reproduced_exactly': True,
        'training_contexts': len(train), 'validation_contexts': len(valid),
        'optimizer_steps_this_invocation': max(candidates), 'replayed_original_steps': old_limit,
        'additional_training_steps': max(candidates)-old_limit, 'new_objective_evaluations': 0,
        'new_search_runs': 0, 'new_ppo_updates': 0, 'elapsed_seconds': time.perf_counter()-started,
        'checkpoint': str(out/'selected_pretrained.pt'), 'sha256': digest(out/'selected_pretrained.pt'),
        'validation_positive_on_both_instances': promising,
        'next_action': 'BOUNDED_DEPLOYED_SEARCH_REQUIRED' if promising else 'NO_AUTOMATIC_SEARCH_OR_PPO',
        'qualification': 'Adaptive development follow-up on the same validation instances. Exact conditional branch expectations are not independent deployed performance or evidence of a PPO improvement.'}
    return result


def execute(source, out):
    torch.set_num_threads(1)
    source, out = source.resolve(), out.resolve()
    source_plan = read_json(source/'PLAN.json')
    old = read_json(source/'PRETRAINING.json')
    data = read_json(source/'DATASET.json')
    if (source_plan['protocol'] != 'complete-terminal-feedback-v3'
            or not (source/'PIPELINE_COMPLETE').exists()
            or source_plan['epochs'] != 40 or not old['completed']):
        raise ValueError('requires the completed frozen 40-step experiment')
    if (digest(source/'DATASET.json') != old['dataset_sha256']
            or digest(old['checkpoint']) != old['sha256']
            or data['plan_sha256'] != digest(source/'PLAN.json')):
        raise ValueError('source data or checkpoint changed')
    for path, sha in source_plan['input_sha256'].items():
        if digest(path) != sha:
            raise ValueError(f'frozen source input changed: {path}')
    files = [source/'PLAN.json', source/'PRETRAINING.json', source/'DATASET.json',
             Path(old['checkpoint']), Path(__file__)]
    plan = {'protocol': 'terminal-feedback-training-extension-400-v1',
        'source': str(source), 'source_plan_sha256': digest(source/'PLAN.json'),
        'steps': 400, 'candidate_epochs': [old['selected']['epoch'], *ADDITIONAL_CHECKPOINTS],
        'selection': 'largest mean validation expected terminal gain; earlier exact ties',
        'training': 'same original weights, data, instance split, normalized inputs, Adam 1e-4, full-information objective and gradient clip 1; replay Adam from step 0 and verify old selected weights',
        'new_objective_evaluations': 0, 'new_ppo_updates': 0, 'maximum_seconds': MAX_SECONDS,
        'input_sha256': {str(p): digest(p) for p in files}}
    out.mkdir(parents=True, exist_ok=True)
    with (out/'.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (out/'PLAN.json').exists() and read_json(out/'PLAN.json') != plan:
            raise ValueError('extension plan changed; preserve the existing output')
        write_json(out/'PLAN.json', plan)
        if (out/'QUICK_RESULT.json').exists():
            result = read_json(out/'QUICK_RESULT.json')
            if digest(result['checkpoint']) != result['sha256']:
                raise ValueError('completed extension checkpoint changed')
            log('completed result retained; no repeated training')
        else:
            result = fit(Path(source_plan['bundle']), data['rows'], source_plan, old,
                         Path(old['checkpoint']), out, digest(out/'PLAN.json'))
            write_json(out/'QUICK_RESULT.json', result)
        with (out/'RESULT_TABLE.csv').open('w') as handle:
            writer = csv.writer(handle)
            writer.writerow(['model', 'epoch', 'train_expected_gain_percent', 'validation_expected_gain_percent',
                             'validation_switch_probability_percent'])
            writer.writerow(['ALWAYS_BALANCED', '', 0, 0, 0])
            for label, record in [('SOURCE_SELECTED', result['source_selected']), ('EXTENDED_SELECTED', result['selected'])]:
                writer.writerow([label, record['epoch'], 100*record['training']['mean_expected_terminal_gain'],
                    100*record['validation']['mean_expected_terminal_gain'],
                    100*record['validation']['mean_switch_probability']])
        (out/'PIPELINE_COMPLETE').write_text('offline only; no search or PPO started\n')
        print((out/'RESULT_TABLE.csv').read_text(), flush=True)
        print(json.dumps({k: result[k] for k in ('source_selected_weights_reproduced_exactly',
            'new_objective_evaluations', 'new_search_runs', 'new_ppo_updates', 'elapsed_seconds',
            'validation_positive_on_both_instances', 'next_action')}, indent=2), flush=True)
        log(f'COMPLETE results={out}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    execute(args.source, args.out or args.source/'training_extension_400_v1')


if __name__ == '__main__':
    main()
