#!/usr/bin/env python3
"""Train and evaluate the frozen K8 objective surrogate; no new searches."""
import argparse
from collections import Counter
import json
from pathlib import Path
import random
import time

import numpy as np
import torch

from m_operator_offline import digest, read_json, write_json
from m_prescreen_surrogate import (CandidateObjectiveTransformer,
    choose_by_predicted_hv, fit_scaler, loss_value, pack_rows, predict)
from run_learning_components_deterministic import _normalizer

SOURCE = Path(__file__).resolve().parent
PROTOCOL = SOURCE/'experiments/m_prescreen_protocol_v1.json'
TRAIN_SEEDS = (911001, 911002, 911003)
EPOCHS = 80
CHECKPOINT_EPOCHS = (10, 20, 40, 60, 80)
BATCH_SIZE = 16


def log(message):
    print(f'[M-K8-TRAIN] {message}', flush=True)


def load_rows(collection, expected_plan):
    plan = read_json(collection/'PLAN.json')
    if (plan.get('protocol') != 'm-prescreen-k8-collection-v1' or
            not (collection/'COLLECTION_COMPLETE').exists()):
        raise ValueError('requires completed K8 collection')
    rows, sources = [], {}
    for instance in expected_plan['collection']['train_instances'] + \
            expected_plan['collection']['validation_instances']:
        for seed in expected_plan['collection']['collection_seeds']:
            folder = collection/'runs'/instance/f'seed_{seed}'
            complete = read_json(folder/'RUN_COMPLETE.json')
            feedback = folder/'FEEDBACK.json'
            if (complete.get('feedback_sha256') != digest(feedback) or
                    complete.get('instance_id') != instance or
                    complete.get('algorithm_seed') != seed):
                raise ValueError(f'collection provenance mismatch: {folder}')
            value = read_json(feedback)
            for row in value['rows']:
                if (len(row['labels']) != 8 or
                        row['candidate_sources'][0] != 'balanced' or
                        len(row['candidate_features']) != 8 or
                        len(row['candidate_batch_tokens']) != 8):
                    raise ValueError('training requires exact eight-child states')
                row = dict(row)
                row['instance'] = instance
                row['trajectory_seed'] = seed
                rows.append(row)
            sources[str(feedback.resolve())] = digest(feedback)
    if len(rows) != read_json(collection/'QUICK_RESULT.json')['state_count']:
        raise ValueError('collection state count changed')
    return rows, sources


def split_batches(indices, rng):
    order = np.asarray(indices).copy()
    rng.shuffle(order)
    return [order[start:start+BATCH_SIZE]
            for start in range(0, len(order), BATCH_SIZE)]


def validation_loss(model, rows, scaler, normalizer):
    model.eval()
    values = []
    with torch.inference_mode():
        for start in range(0, len(rows), BATCH_SIZE):
            packed = pack_rows(rows[start:start+BATCH_SIZE], scaler, normalizer)
            objective, logits = model(*packed[:4])
            values.append((loss_value(objective, logits, *packed[4:]).item(), len(packed[0])))
    return sum(value*count for value, count in values)/sum(count for _, count in values)


def fit_one(seed, train_rows, validation_rows, scaler, normalizer, out):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    packed = pack_rows(train_rows[:1], scaler, normalizer)
    model = CandidateObjectiveTransformer(packed[0].shape[-1],
        packed[1].shape[-1], packed[2].shape[-1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    history, best = [], None
    rng = np.random.default_rng(seed)
    started = time.monotonic()
    for epoch in range(1, EPOCHS+1):
        model.train()
        total, count = 0., 0
        for indices in split_batches(np.arange(len(train_rows)), rng):
            batch = [train_rows[int(i)] for i in indices]
            values = pack_rows(batch, scaler, normalizer)
            objective, logits = model(*values[:4])
            loss = loss_value(objective, logits, *values[4:])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            total += loss.item()*len(batch)
            count += len(batch)
        if epoch in CHECKPOINT_EPOCHS:
            valid = validation_loss(model, validation_rows, scaler, normalizer)
            record = {'epoch': epoch, 'train_loss': total/count,
                      'validation_loss': valid}
            history.append(record)
            log(f'seed={seed} epoch={epoch}/{EPOCHS} train={total/count:.6f} valid={valid:.6f}')
            if best is None or (valid, epoch) < (best['validation_loss'], best['epoch']):
                best = {**record, 'state': {name: value.detach().cpu().clone()
                                            for name, value in model.state_dict().items()}}
    if best is None:
        raise RuntimeError('no checkpoint was evaluated')
    model.load_state_dict(best.pop('state'))
    checkpoint = out/f'seed_{seed}.pt'
    torch.save({'protocol': 'm-prescreen-objective-surrogate-v1',
        'seed': seed, 'epoch': best['epoch'], 'model': model.state_dict(),
        'batch_feature_count': packed[0].shape[-1],
        'candidate_feature_count': packed[1].shape[-1],
        'context_feature_count': packed[2].shape[-1]}, checkpoint)
    return model, {'seed': seed, 'selected': best, 'history': history,
        'checkpoint': str(checkpoint.resolve()),
        'checkpoint_sha256': digest(checkpoint),
        'elapsed_seconds': time.monotonic()-started}


def ensemble_predict(models, rows, scaler, normalizer):
    outputs = [predict(model, rows, scaler, normalizer) for model in models]
    return (np.mean([value[0] for value in outputs], axis=0),
            np.mean([value[1] for value in outputs], axis=0))


def score(rows, predicted, probability, normalizer):
    selected = choose_by_predicted_hv(rows, predicted, probability,
                                      normalizer.reference)
    records = []
    for row, action in zip(rows, selected):
        actual = float(row['labels'][int(action)]['hv_contribution'])
        balanced = float(row['labels'][0]['hv_contribution'])
        oracle = max(float(label['hv_contribution']) for label in row['labels'])
        records.append({'instance': row['instance'], 'trajectory_seed': row['trajectory_seed'],
            'generation': row['generation'], 'evaluation': row['evaluation'],
            'selected': int(action), 'selected_source': row['candidate_sources'][int(action)],
            'actual_selected_hv_contribution': actual,
            'balanced_hv_contribution': balanced,
            'net_gain': actual-balanced, 'oracle_gain': oracle-balanced})
    per_instance = []
    for instance in sorted({row['instance'] for row in records}):
        subset = [row for row in records if row['instance'] == instance]
        gains = np.asarray([row['net_gain'] for row in subset])
        losses = gains[gains < 0]
        per_instance.append({'instance': instance, 'states': len(subset),
            'mean_net_gain': float(gains.mean()),
            'switches': sum(row['selected'] != 0 for row in subset),
            'switch_fraction': float(np.mean([row['selected'] != 0 for row in subset])),
            'positive_gain_total': float(gains[gains > 0].sum()),
            'negative_gain_total': float(losses.sum()),
            'gain_p10': float(np.quantile(gains, .1)),
            'mean_oracle_gain': float(np.mean([row['oracle_gain'] for row in subset]))})
    gains = np.asarray([row['net_gain'] for row in records])
    return {'states': len(records), 'mean_net_gain': float(gains.mean()),
        'switches': sum(row['selected'] != 0 for row in records),
        'switch_fraction': float(np.mean([row['selected'] != 0 for row in records])),
        'positive_gain_total': float(gains[gains > 0].sum()),
        'negative_gain_total': float(gains[gains < 0].sum()),
        'gain_p10': float(np.quantile(gains, .1)),
        'action_counts': dict(Counter(str(row['selected']) for row in records)),
        'per_instance': per_instance, 'decisions': records}


def prediction_metrics(rows, predicted, probability, normalizer):
    actual, estimates, truth, estimated = [], [], [], []
    for row, state_prediction, state_probability in zip(
            rows, predicted, probability):
        for label, point, chance in zip(
                row['labels'], state_prediction, state_probability):
            truth.append(bool(label['feasible']))
            estimated.append(bool(chance >= .5))
            if label['feasible']:
                actual.append(normalizer.transform(label['objectives']))
                estimates.append(point)
    actual, estimates = np.asarray(actual), np.asarray(estimates)
    error = estimates-actual
    return {'feasible_labels': int(sum(truth)),
        'infeasible_labels': int(len(truth)-sum(truth)),
        'feasibility_accuracy': float(np.mean(np.asarray(truth) == estimated)),
        'normalized_objective_mae': np.mean(np.abs(error), axis=0).tolist(),
        'normalized_objective_rmse': np.sqrt(np.mean(error**2, axis=0)).tolist()}


def execute(args):
    protocol = read_json(PROTOCOL)
    training_design = protocol['supervised_training']
    if (training_design['width'] != 32 or training_design['layers'] != 2 or
            training_design['heads'] != 4 or
            training_design['training_seeds'] != list(TRAIN_SEEDS) or
            training_design['epochs'] != EPOCHS or
            training_design['checkpoint_epochs'] != list(CHECKPOINT_EPOCHS) or
            training_design['batch_size'] != BATCH_SIZE or
            training_design['learning_rate'] != 3e-4 or
            training_design['weight_decay'] != 1e-4):
        raise ValueError('implementation differs from frozen supervised design')
    collection = args.collection.resolve()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    rows, source_hashes = load_rows(collection, protocol)
    for path in (Path(__file__).resolve(),
                 SOURCE/'m_prescreen_surrogate.py', PROTOCOL):
        source_hashes[str(path)] = digest(path)
    train_names = set(protocol['collection']['train_instances'])
    validation_names = set(protocol['collection']['validation_instances'])
    train_rows = [row for row in rows if row['instance'] in train_names]
    validation_rows = [row for row in rows if row['instance'] in validation_names]
    if (not train_rows or not validation_rows or
            {row['instance'] for row in train_rows} != train_names or
            {row['instance'] for row in validation_rows} != validation_names):
        raise ValueError('instance-level train/validation split changed')
    normalizer = _normalizer(args.normalization)
    scaler = fit_scaler(train_rows, normalizer)
    frozen = {'protocol': 'm-prescreen-objective-surrogate-v1',
        'collection': str(collection), 'collection_plan_sha256': digest(collection/'PLAN.json'),
        'protocol_sha256': digest(PROTOCOL), 'normalization': str(args.normalization.resolve()),
        'normalization_sha256': digest(args.normalization),
        'train_instances': sorted(train_names), 'validation_instances': sorted(validation_names),
        'train_states': len(train_rows), 'validation_states': len(validation_rows),
        'train_seeds': list(TRAIN_SEEDS), 'epochs': EPOCHS,
        'checkpoint_epochs': list(CHECKPOINT_EPOCHS), 'batch_size': BATCH_SIZE,
        'learning_rate': 3e-4, 'weight_decay': 1e-4,
        'selection': 'ensemble mean objective prediction; feasibility>=0.5; exact predicted-population HV; first tie keeps Balanced',
        'scaler': scaler, 'source_sha256': source_hashes}
    if (out/'PLAN.json').exists() and read_json(out/'PLAN.json') != frozen:
        raise ValueError('frozen training inputs changed; use a new output directory')
    write_json(out/'PLAN.json', frozen)
    models, training = [], []
    for seed in TRAIN_SEEDS:
        model, report = fit_one(seed, train_rows, validation_rows,
                                scaler, normalizer, out)
        models.append(model)
        training.append(report)
    predicted, probability = ensemble_predict(models, validation_rows,
                                               scaler, normalizer)
    offline = score(validation_rows, predicted, probability, normalizer)
    diagnostics = prediction_metrics(validation_rows, predicted, probability,
                                     normalizer)
    passed = (offline['switches'] > 0 and
              all(row['mean_net_gain'] > 0 for row in offline['per_instance']))
    bundle = {**frozen, 'checkpoints': [row['checkpoint'] for row in training],
        'checkpoint_sha256': {row['checkpoint']: row['checkpoint_sha256']
                              for row in training},
        'selected_epochs': [row['selected']['epoch'] for row in training],
        'offline_gate_pass': passed}
    write_json(out/'BUNDLE.json', bundle)
    write_json(out/'TRAINING.json', {'completed': True, 'models': training})
    result = {'completed': True, 'new_objective_evaluations': 0,
        'new_searches': 0, 'new_ppo_updates': 0,
        'training_states': len(train_rows), 'validation_states': len(validation_rows),
        'selected_epochs': bundle['selected_epochs'], 'offline_validation': offline,
        'prediction_diagnostics': diagnostics,
        'offline_gate_pass': passed,
        'next_action': ('run bounded paired B400 search without PPO'
            if passed else 'stop before search; objective surrogate did not beat Balanced on both frozen validation instances')}
    write_json(out/'QUICK_RESULT.json', result)
    (out/'TRAINING_COMPLETE').write_text(digest(out/'BUNDLE.json')+'\n')
    log(json.dumps({key: result[key] for key in (
        'training_states', 'validation_states', 'selected_epochs',
        'offline_gate_pass', 'next_action')}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--collection', type=Path, required=True)
    parser.add_argument('--normalization', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    execute(args)


if __name__ == '__main__':
    main()
