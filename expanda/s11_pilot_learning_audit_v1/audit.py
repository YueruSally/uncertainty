#!/usr/bin/env python3
"""Read frozen S11 labels/checkpoints; no search, fitting, or gate changes."""
import argparse
from collections import defaultdict
import inspect
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 's11_targeted_v1'))
import learning as L
import numpy as np
import torch


def equal_mean(rows, values):
    groups = defaultdict(lambda: defaultdict(list))
    for row, value in zip(rows, values):
        groups[row['configuration']][row['instance']].append(float(value))
    return float(np.mean([np.mean([np.mean(x) for x in instances.values()])
                          for instances in groups.values()]))


def label_summary(rows, targets, scale):
    masks = np.asarray([r['legal_mask'] for r in rows], dtype=bool)
    alternatives = masks.copy()
    alternatives[:, 0] = False
    values = targets[alternatives]
    oracle = np.max(np.where(masks, targets, -np.inf), axis=1)
    differences = targets[:, :, None] - targets[:, None, :]
    pairs = (np.triu(np.ones((8, 8), dtype=bool), 1)[None]
             & masks[:, :, None] & masks[:, None, :])
    informative = pairs & (np.abs(differences / scale) > 1e-6)
    tol = 1e-12
    return {
        'states': len(rows), 'independent_instances': len({r['instance'] for r in rows}),
        'legal_alternatives': int(alternatives.sum()),
        'positive_alternatives': int(np.sum(values > tol)),
        'negative_alternatives': int(np.sum(values < -tol)),
        'zero_alternatives': int(np.sum(np.abs(values) <= tol)),
        'states_with_positive_alternative': int(np.sum(oracle > tol)),
        'states_with_ranking_signal': int(np.sum(informative.any(axis=(1, 2)))),
        'informative_candidate_pairs': int(informative.sum()),
        'states_all_legal_candidates_tied': int(np.sum(
            np.all((np.abs(targets) <= tol) | ~masks, axis=1))),
        'oracle_mean_residual_hv': equal_mean(rows, oracle),
        'absolute_nonzero_residual_median': float(np.median(np.abs(values[np.abs(values) > tol])))
            if np.any(np.abs(values) > tol) else 0.0,
        'zero_tolerance_raw_hv': tol,
        'ranking_tolerance_scaled': 1e-6,
    }


def decision_summary(rows, targets, predictions, scale, gate):
    choices = [L.decide(predictions[:, i], row['legal_mask'], gate['lambda'],
                        gate['threshold_scaled'] * scale, gate['disabled'])
               for i, row in enumerate(rows)]
    gains = targets[np.arange(len(rows)), choices]
    selected = np.asarray(choices) != 0
    masks = np.asarray([r['legal_mask'] for r in rows], dtype=bool)
    oracle = equal_mean(rows, np.max(np.where(masks, targets, -np.inf), axis=1))
    mean = equal_mean(rows, gains)
    return {
        'mean_residual_hv': mean, 'switches': int(selected.sum()),
        'switch_fraction': float(selected.mean()),
        'beneficial_switches': int(np.sum(selected & (gains > 1e-12))),
        'harmful_switches': int(np.sum(selected & (gains < -1e-12))),
        'neutral_switches': int(np.sum(selected & (np.abs(gains) <= 1e-12))),
        'oracle_fraction_captured': mean / oracle if oracle > 1e-12 else None,
        'per_instance': [
            {'instance': name,
             'mean_residual_hv': float(np.mean([g for r, g in zip(rows, gains) if r['instance'] == name]))}
            for name in sorted({r['instance'] for r in rows})],
    }


def feature_ranges(rows, bundle):
    scaler = bundle['scaler']
    result = {}
    for part in ('batch', 'candidate', 'context'):
        total = clipped = 0
        largest = 0.0
        for row in rows:
            x = (L.context(row, bundle['normalization']) if part == 'context'
                 else row['candidate_batch_tokens' if part == 'batch' else 'candidate_features'])
            z = np.abs((np.asarray(x) - scaler[part]['mean']) / scaler[part]['std'])
            if not np.isfinite(z).all():
                raise ValueError('Non-finite feature values')
            total += z.size
            clipped += int(np.sum(z > scaler['clip']))
            largest = max(largest, float(z.max()))
        result[part] = {'values': total, 'clipped_fraction': clipped / total, 'max_abs_z': largest}
    return result


def objective_coverage(rows, norm):
    feasible = outside = total = 0
    for row in rows:
        n = L.episode_normalizer(norm, row['divisor'])
        for legal, pair in zip(row['legal_mask'], row['labels']):
            if not legal:
                continue
            for child in pair:
                total += 1
                if child['feasible']:
                    feasible += 1
                    outside += int(any(x >= r for x, r in zip(n.transform(child['objectives']), n.reference)))
    return {'legal_candidate_children': total, 'feasible_children': feasible,
            'feasible_children_outside_hv_reference': outside}


def verify_source(source):
    plan = L.read_json(source / 'PLAN.json')
    frozen = L.read_json(source / 'models/FROZEN.json')
    if frozen['plan_sha256'] != plan['sha256']:
        raise ValueError('Frozen models and collection plan differ')
    for name, sha in frozen['files'].items():
        if L.digest(source / name) != sha:
            raise ValueError('Frozen input changed: ' + name)
    # These are the actual feature, target, inference and HV functions being used.
    for name in ('s11_targeted_v1/learning.py', 's11_targeted_v1/common.py',
                 'formal_icm_s0_v1/src/network.py',
                 'formal_icm_s0_v1/src/support/transformer_ppo/archive.py'):
        if L.digest(HERE.parent / name) != plan['files'][name]:
            raise ValueError('Diagnostic runtime differs from the frozen experiment: ' + name)
    return plan, frozen


def load_rows(source, component, role, catalog):
    result = []
    for item in catalog['instances']:
        if item['development_role'] != role:
            continue
        data = L.read_json(source / 'collection' / item['instance_id'] / ('COLLECT_' + component) / 'DATA.json')
        if data['role'] != role or data['instance'] != item['instance_id']:
            raise ValueError('Collection role mismatch')
        for row in data['rows']:
            if row['component'] != component or row['instance'] != item['instance_id']:
                raise ValueError('Collection identity mismatch')
            if len(row['legal_mask']) != 8 or not row['legal_mask'][0]:
                raise ValueError('Missing legal rule reference')
            result.append(row)
    if not result:
        raise ValueError('Missing ' + component + '/' + role)
    return result


def audit_component(source, component, settings, catalog):
    folder = source / 'models' / component
    bundle = L.read_json(folder / 'BUNDLE.json')
    validation = L.read_json(folder / 'VALIDATION.json')
    data = {role: load_rows(source, component, role, catalog) for role in ('training', 'validation')}
    for role, rows in data.items():
        if sorted({r['instance'] for r in rows}) != bundle[role + '_instances'] or len(rows) != bundle[role + '_states']:
            raise ValueError('Bundle and collection differ: ' + role)
    if set(bundle['training_instances']) & set(bundle['validation_instances']):
        raise ValueError('Training and validation overlap')
    targets = {role: np.stack([L.labels(r, bundle['normalization']) for r in rows]) for role, rows in data.items()}
    scale = bundle['scale']
    if not np.isclose(max(float(targets['training'].std()), 1e-9), scale, rtol=1e-9, atol=1e-14):
        raise ValueError('Training labels do not reproduce the stored scale')
    diagnostic_gate = {'lambda': 0., 'threshold_scaled': 0., 'disabled': False}
    enabled = [x for x in validation['choices'] if not x['disabled']]
    best_enabled = min(enabled, key=lambda x: (-x['mean_gain'], x['switch_fraction'],
                       x['epoch'], x['lambda'], x['threshold_scaled']))
    report = {
        'selected_gate': bundle['gate'], 'best_enabled_gate_from_original_validation': best_enabled,
        'actual_loss': bundle['loss'], 'label_scale': scale,
        'training_history': validation['history'], 'labels': {}, 'feature_ranges': {}, 'objective_coverage': {}, 'epochs': [],
    }
    for role, rows in data.items():
        report['labels'][role] = label_summary(rows, targets[role], scale)
        report['labels'][role]['per_phase'] = [
            {'phase': phase, **label_summary([rows[i] for i in ids], targets[role][ids], scale)}
            for phase in sorted({r['phase'] for r in rows})
            for ids in [[i for i, r in enumerate(rows) if r['phase'] == phase]]]
        report['feature_ranges'][role] = feature_ranges(rows, bundle)
        report['objective_coverage'][role] = objective_coverage(rows, bundle['normalization'])
        label_info = report['labels'][role]
        print('[S11-AUDIT] %s %s states=%s instances=%s positive-opportunity=%s all-tied=%s oracle=%+.8g' % (
            component, role, len(rows), label_info['independent_instances'],
            label_info['states_with_positive_alternative'], label_info['states_all_legal_candidates_tied'],
            label_info['oracle_mean_residual_hv']), flush=True)
    for epoch in settings['checkpoint_epochs']:
        models = []
        for seed in settings['training_seeds']:
            extra = {'weights_only': True} if 'weights_only' in inspect.signature(torch.load).parameters else {}
            ck = torch.load(folder / f'seed_{seed}_epoch_{epoch}.pt', map_location='cpu', **extra)
            model = L.CandidateGainTransformer(*ck['dimensions'], width=ck['width'])
            model.load_state_dict(ck['model_state']); model.eval(); models.append(model)
        entry = {'epoch': epoch}
        for role, rows in data.items():
            preds = L.predict(models, rows, bundle['scaler'], bundle['normalization'], scale)
            entry[role] = {'fixed_zero_threshold': decision_summary(rows, targets[role], preds, scale, diagnostic_gate)}
            if epoch == bundle['gate']['epoch']:
                entry[role]['deployed_gate'] = decision_summary(rows, targets[role], preds, scale, bundle['gate'])
            if role == 'validation':
                # Reproduce every original enabled validation choice; do not fit a new gate.
                error = switch_error = 0.0
                for gate in (g for g in enabled if g['epoch'] == epoch):
                    stats = decision_summary(rows, targets[role], preds, scale, gate)
                    error = max(error, abs(stats['mean_residual_hv'] - gate['mean_gain']))
                    switch_error = max(switch_error, abs(stats['switch_fraction'] - gate['switch_fraction']))
                entry['original_validation_gain_max_abs_replay_error'] = error
                entry['original_validation_switch_fraction_max_abs_replay_error'] = switch_error
        report['epochs'].append(entry)
        print('[S11-AUDIT] %s epoch=%s fixed-rule-threshold train=%+.8g valid=%+.8g (HV units)' % (
            component, epoch, entry['training']['fixed_zero_threshold']['mean_residual_hv'],
            entry['validation']['fixed_zero_threshold']['mean_residual_hv']), flush=True)
    return report


def execute(source, out):
    source = source.resolve(); out = out.resolve()
    if source == out or source in out.parents or out in source.parents:
        raise ValueError('Audit output must be separate from the source tree')
    torch.set_num_threads(1)
    started = time.perf_counter()
    plan, frozen = verify_source(source)
    catalog = L.read_json(source / 'instances/catalog.json')
    components = {c: audit_component(source, c, plan['protocol'], catalog) for c in 'ICM'}
    # Verify that neither the models nor collection files changed during this audit.
    _, after = verify_source(source)
    if frozen != after:
        raise ValueError('Frozen manifest changed during audit')
    result = {
        'completed': True, 'source': str(source), 'plan_sha256': plan['sha256'],
        'components': components, 'new_objective_evaluations': 0,
        'new_search_runs': 0, 'new_training_steps': 0, 'model_or_gate_changes': 0,
        'elapsed_seconds': time.perf_counter() - started,
        'qualification': 'All gains here are one-step residual HV under the frozen training normalizer, '
            'not final-search percentages. Oracle uses stored labels only. Fixed-zero-threshold '
            'predictions are diagnostic; no gate is changed. No screen labels are used. '
            'One validation instance cannot establish population-level generalization.',
    }
    out.mkdir(parents=True, exist_ok=True)
    L.write_json(out / 'AUDIT.json', result)
    compact = {c: {k: x[k] for k in ('selected_gate', 'best_enabled_gate_from_original_validation',
                                   'training_history', 'labels', 'epochs', 'feature_ranges', 'objective_coverage')}
               for c, x in components.items()}
    for x in compact.values():
        x['labels'] = {role: {k: v for k, v in summary.items() if k != 'per_phase'}
                       for role, summary in x['labels'].items()}
    L.write_json(out / 'QUICK_RESULT.json', {
        'components': compact, 'new_objective_evaluations': 0, 'new_training_steps': 0})
    for c, r in components.items():
        print('[S11-AUDIT] %s frozen_disabled=%s best_original_enabled_valid_gain=%+.8g' % (
            c, r['selected_gate']['disabled'], r['best_enabled_gate_from_original_validation']['mean_gain']), flush=True)
        for epoch in r['epochs']:
            losses = [h['training_loss'] for h in r['training_history'] if h['epoch'] == epoch['epoch']]
            replay = epoch['original_validation_gain_max_abs_replay_error']
            switch_replay = epoch['original_validation_switch_fraction_max_abs_replay_error']
            print('[S11-AUDIT] %s epoch=%s original_train_loss=%.8g validation_replay_gain_error=%.3g switch_error=%.3g' % (
                c, epoch['epoch'], np.mean(losses), replay, switch_replay), flush=True)
        for role in ('training', 'validation'):
            clips = {k: round(v['clipped_fraction'], 6) for k, v in r['feature_ranges'][role].items()}
            print('[S11-AUDIT] %s %s clipped=%s objective_coverage=%s' % (
                c, role, clips, r['objective_coverage'][role]), flush=True)
    print('[S11-AUDIT] COMPLETE zero new searches/evaluations/training; results=' + str(out), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    execute(args.source, args.out or args.source.with_name(args.source.name + '_learning_audit_v1'))
