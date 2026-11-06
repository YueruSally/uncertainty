#!/usr/bin/env python3
"""Frozen softmax-vs-argmax comparison on already evaluated terminal branches.

Only the selected training-extension checkpoint is examined. No temperature,
threshold, checkpoint or test-instance sweep; no search or training is launched.
"""
import argparse
from collections import Counter
import csv
import fcntl
import json
from pathlib import Path
import sys
import time

SOURCE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE))

import numpy as np
import torch
from experiments.extend_terminal_feedback_training import partition
from m_operator_offline import read_json, write_json, digest, OPERATORS
from m_terminal_hv_ppo import checkpoint_model


def decisions(logits, eligible):
    """Decision computation must not have access to terminal outcome labels."""
    if (logits.ndim != 2 or logits.shape[1] != 5 or logits.shape != eligible.shape
            or not bool(eligible[:, 4].all())):
        raise ValueError('five-action logits need a legal Balanced fallback')
    masked = logits.masked_fill(~eligible, -torch.inf)
    if not bool(torch.isfinite(masked[eligible]).all()):
        raise ValueError('non-finite legal policy score')
    probabilities = masked.softmax(-1)
    maxima, actions = masked.max(-1)
    # Exact ties favour the frozen Balanced action; no fitted gate or threshold.
    actions = torch.where(masked[:, 4] == maxima, torch.full_like(actions, 4), actions)
    return probabilities, actions


def summarize(model, rows):
    x = torch.tensor([r['tokens'] for r in rows], dtype=torch.float32)
    eligible = torch.tensor([r['eligible'] for r in rows], dtype=torch.bool)
    model.eval()
    with torch.no_grad():
        probabilities, actions = decisions(model(x, eligible)[0], eligible)
    # All choices above are frozen BEFORE reading the labels below.
    q = torch.tensor([[0. if value is None else value/r['terminal_hvs'][4]-1
                       for value in r['terminal_hvs']] for r in rows], dtype=torch.float32)
    stochastic = (probabilities*q).sum(-1).numpy()
    greedy = q[torch.arange(len(rows)), actions].numpy()
    possible = q.masked_fill(~eligible, -torch.inf).max(-1).values.numpy()
    reports = []
    for name in sorted({r['instance'] for r in rows}):
        indices = [i for i, r in enumerate(rows) if r['instance'] == name]
        selected = actions[indices].tolist()
        reports.append({'instance': name, 'contexts': len(indices),
            'softmax_expected_gain_percent': 100*float(np.mean(stochastic[indices])),
            'argmax_gain_percent': 100*float(np.mean(greedy[indices])),
            'softmax_switch_probability': float((1-probabilities[indices, 4]).mean()),
            'argmax_switch_fraction': sum(a != 4 for a in selected)/len(indices),
            'argmax_actions': dict(Counter(OPERATORS[a] for a in selected)),
            'contexts_with_a_recorded_positive_alternative': int((possible[indices] > 1e-12).sum()),
            'recorded_hindsight_best_gain_percent': 100*float(np.mean(possible[indices]))})
    return {'contexts': len(rows), 'per_instance': reports,
        'softmax_expected_gain_percent': float(np.mean([r['softmax_expected_gain_percent'] for r in reports])),
        'argmax_gain_percent': float(np.mean([r['argmax_gain_percent'] for r in reports])),
        'softmax_switch_probability': float((1-probabilities[:, 4]).mean()),
        'argmax_switch_fraction': float((actions != 4).float().mean()),
        'all_instances_argmax_positive': all(r['argmax_gain_percent'] > 0 for r in reports),
        'decisions': [{'instance': r['instance'], 'seed': r['seed'],
            'argmax_operator': OPERATORS[int(actions[i])], 'probabilities': probabilities[i].tolist(),
            'argmax_gain_percent': 100*float(greedy[i]),
            'softmax_expected_gain_percent': 100*float(stochastic[i])} for i, r in enumerate(rows)]}


def execute(source):
    started = time.perf_counter()
    torch.set_num_threads(1)
    source = source.resolve()
    extension = source/'training_extension_400_v1'
    original_plan = read_json(source/'PLAN.json')
    plan = read_json(extension/'PLAN.json')
    saved_result = read_json(extension/'QUICK_RESULT.json')
    if (original_plan['protocol'] != 'complete-terminal-feedback-v3'
            or plan['protocol'] != 'terminal-feedback-training-extension-400-v1'
            or not saved_result['completed'] or not (extension/'PIPELINE_COMPLETE').exists()):
        raise ValueError('requires the completed frozen training extension')
    for frozen in (original_plan, plan):
        for path, sha in frozen['input_sha256'].items():
            if digest(path) != sha:
                raise ValueError(f'frozen input changed: {path}')
    if plan['source_plan_sha256'] != digest(source/'PLAN.json'):
        raise ValueError('extension source plan changed')
    data = read_json(source/'DATASET.json')
    if data['plan_sha256'] != digest(source/'PLAN.json'):
        raise ValueError('dataset belongs to a different plan')
    checkpoint = Path(saved_result['checkpoint'])
    model, metadata = checkpoint_model(checkpoint, saved_result['sha256'])
    if (metadata['plan_sha256'] != digest(extension/'PLAN.json') or metadata['update'] != 0
            or metadata['execution_mode'] != 'single-intervention'
            or metadata['bundle_sha256'] != digest(original_plan['bundle'])
            or metadata['complete_feedback_epoch'] != saved_result['selected']['epoch']):
        raise ValueError('selected model provenance/execution mismatch')
    train, valid = partition(data['rows'], original_plan, Path(original_plan['bundle']))
    out = extension/'decoding_check_v1'
    out.mkdir(exist_ok=True)
    files = [source/'DATASET.json', extension/'PLAN.json', extension/'QUICK_RESULT.json', checkpoint, Path(__file__)]
    frozen = {'protocol': 'terminal-decoding-check-v1', 'checkpoint': str(checkpoint),
        'selected_epoch': saved_result['selected']['epoch'],
        'arms': ['ALWAYS_BALANCED', 'EXISTING_SOFTMAX', 'ARGMAX_BALANCED_TIES'],
        'model_selection': 'none; use the already selected extension checkpoint only',
        'input_sha256': {str(p): digest(p) for p in files},
        'new_objective_evaluations': 0, 'new_training_steps': 0,
        'decision': 'argmax positive on each validation instance only motivates a separately frozen deployed search; no automatic search or PPO'}
    with (out/'.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (out/'PLAN.json').exists() and read_json(out/'PLAN.json') != frozen:
            raise ValueError('decoding check plan changed; retain existing results')
        write_json(out/'PLAN.json', frozen)
        reports = {part: summarize(model, rows) for part, rows in [('training', train), ('validation', valid)]}
        for part, report in reports.items():
            previous = saved_result['selected'][part]
            if not np.isclose(report['softmax_expected_gain_percent'],
                    100*previous['mean_expected_terminal_gain'], rtol=1e-7, atol=1e-7):
                raise ValueError('unchanged stochastic-policy expectation did not reproduce')
        promising = reports['validation']['all_instances_argmax_positive']
        qualification = ('The same development validation instances have already guided training. '
            'These are conditional terminal returns on stored branches, not independent search results. '
            'Hindsight best uses outcome labels and is diagnostic only, never a deployable result. '
            'Argmax is a new decoding rule, not a bug fix or PPO improvement.')
        write_json(out/'DECISIONS.json', {part: report['decisions'] for part, report in reports.items()})
        result = {'completed': True, 'selected_epoch': saved_result['selected']['epoch'],
            'checkpoint_sha256': saved_result['sha256'],
            **{part: {k: v for k, v in report.items() if k != 'decisions'} for part, report in reports.items()},
            'new_objective_evaluations': 0, 'new_search_runs': 0, 'new_training_steps': 0,
            'elapsed_seconds': time.perf_counter()-started,
            'next_action': 'FREEZE_ARGMAX_AND_CHECK_DEPLOYED_SEARCH' if promising else 'NO_ARGMAX_GAIN_ESTABLISHED',
            'qualification': qualification}
        write_json(out/'QUICK_RESULT.json', result)
        with (out/'RESULT_TABLE.csv').open('w') as handle:
            writer = csv.writer(handle)
            writer.writerow(['split', 'instance', 'contexts', 'softmax_expected_gain_percent',
                'argmax_gain_percent', 'softmax_switch_probability', 'argmax_switch_fraction'])
            for part, report in reports.items():
                for row in report['per_instance']:
                    writer.writerow([part, row['instance'], row['contexts'], row['softmax_expected_gain_percent'],
                        row['argmax_gain_percent'], row['softmax_switch_probability'], row['argmax_switch_fraction']])
        print((out/'RESULT_TABLE.csv').read_text(), flush=True)
        print(json.dumps({'validation': result['validation'], 'new_objective_evaluations': 0,
            'new_search_runs': 0, 'new_training_steps': 0, 'next_action': result['next_action'],
            'qualification': qualification}, indent=2), flush=True)
        print(f'[M-DECODE] COMPLETE {out}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    execute(parser.parse_args().source)


if __name__ == '__main__':
    main()
