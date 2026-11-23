#!/usr/bin/env python3
"""Repeat original full-validation selection with shared member 991001 only.

Read saved validation states/labels and epoch 10/20/30 weights. No searches,
objective evaluations, training, refitting, or changes to deployed bundles.
"""
import argparse
from collections import Counter
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '1'
EXP = Path(__file__).resolve().parent
DIAG = EXP / 'transformer_ppo_runs/diagnostics'
sys.path.insert(0, str(EXP / 'joint_icm_multiscale_v1'))
from bootstrap import COMPONENTS, legacy_learning, progress_row
from shared import SharedTransformer, load_checkpoint, predictions
from training import choose_gates
import numpy as np
import torch


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def write(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    tmp.replace(path)


def csv_write(path, rows):
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def policy_signature(selected):
    return (selected['epoch'], tuple((selected['gates'][c]['threshold_scaled'],
                                     selected['gates'][c]['disabled']) for c in COMPONENTS))


def run(a):
    started = time.monotonic()
    source, deployed, out = a.source.resolve(), a.deployed.resolve(), a.out.resolve()
    if any(out == p or out in p.parents or p in out.parents for p in (source, deployed)):
        raise ValueError('Use a separate report directory; source/model directories are read-only.')
    out.mkdir(parents=True, exist_ok=True)
    inputs = {}

    def pin(path, expected=None):
        path = Path(path).resolve()
        value = digest(path)
        if expected is not None and value != expected:
            raise ValueError('Input hash differs from training record: ' + str(path))
        inputs[str(path)] = value
        return value

    def status(stage, **extra):
        value = dict(completed=False, stage=stage, pid=os.getpid(),
                     elapsed_seconds=round(time.monotonic()-started, 1),
                     new_searches=0, new_objective_evaluations=0, training_updates=0, **extra)
        write(out/'STATUS.json', value)
        print('[SINGLE-VALID]', json.dumps(value, ensure_ascii=False), flush=True)

    with (out/'.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        status('checking_inputs')
        model_dir = source/'shared_model'
        for p in (model_dir/'BUNDLE.json', deployed/'BUNDLE.json', source/'instances/catalog.json'):
            pin(p)
        b, current = read(model_dir/'BUNDLE.json'), read(deployed/'BUNDLE.json')
        pin(model_dir/'PLAN.json', b['plan_sha256'])
        settings = b['settings']
        if (settings['checkpoint_epochs'] != [10, 20, 30]
                or settings['gate_threshold_scaled'] != [0., .02, .05]
                or settings['progress_denominator'] != 50000):
            raise ValueError('Unexpected original selection grid or progress scale.')
        if current['checkpoints'] != ['seed_991001_epoch_20.pt']:
            raise ValueError('Expected deployed single member 991001, epoch 20.')
        for key in ('normalization', 'scalers', 'target_scales', 'dimensions', 'plan_sha256',
                    'feature_version', 'validation_instances', 'training_instances'):
            if b[key] != current[key]:
                raise ValueError('Deployed model has different training provenance: ' + key)
        expected_current = (20, ((.05, False), (0., False), (0., False)))
        if policy_signature(current['selected']) != expected_current:
            raise ValueError('Deployed settings differ from epoch 20 / thresholds 0.05,0,0.')
        name = current['checkpoints'][0]
        pin(deployed/name, b['checkpoint_sha256'][name])
        if current['checkpoint_sha256'][name] != b['checkpoint_sha256'][name]:
            raise ValueError('Deployed weights differ from original member 991001.')
        # Check the implementations actually used, not unrelated later scripts.
        for rel in ('component_screen_b20_v1/learning.py', 'joint_icm_multiscale_v1/bootstrap.py',
                    'joint_icm_multiscale_v1/shared.py', 'joint_icm_multiscale_v1/training.py'):
            hashes = [h for p,h in b['files'].items() if p.endswith('/'+rel)]
            if len(hashes) != 1:
                raise ValueError('Missing original implementation hash: ' + rel)
            pin(EXP/rel, hashes[0])
        pin(Path(__file__))
        checkpoints = {}
        for epoch in settings['checkpoint_epochs']:
            path = model_dir/f'seed_991001_epoch_{epoch}.pt'
            pin(path, b['checkpoint_sha256'].get(path.name))
            ck = load_checkpoint(path)
            if ck['epoch'] != epoch or ck['plan_sha256'] != b['plan_sha256']:
                raise ValueError('Checkpoint does not belong to the original fit: ' + str(path))
            checkpoints[epoch] = ck['model']

        valid = set(b['validation_instances']['I'])
        if len(valid) != 18 or valid & set(b['training_instances']['I']):
            raise ValueError('Unexpected or overlapping validation split.')
        entries = [r for r in read(source/'instances/catalog.json')['instances'] if r['instance_id'] in valid]
        if len(entries) != 18 or len({r['instance_id'] for r in entries}) != 18:
            raise ValueError('Validation catalogue incomplete.')
        if (Counter(r['configuration'] for r in entries) != {c:2 for c in settings['configurations']}
                or any(r['development_role'] != 'validation' for r in entries)):
            raise ValueError('Expected two validation instances per configuration.')
        rows, targets = {c:[] for c in COMPONENTS}, {c:[] for c in COMPONENTS}
        cache_dir = out/'cache'
        cache_dir.mkdir(exist_ok=True)
        old_audit = EXP/'single_b5000_offline_review_v1/run.py'
        for entry in entries:
            instance = entry['instance_id']
            for arm in ('COLLECT_I', 'COLLECT_CM'):
                path = source/'collection'/instance/arm/'DATA.json'
                expected = [h for p,h in b['files'].items() if p.endswith(f'/collection/{instance}/{arm}/DATA.json')]
                if len(expected) != 1:
                    raise ValueError('Validation file not bound to model: ' + str(path))
                h = pin(path, expected[0]); data = read(path)
                if data['role'] != 'validation' or data['instance'] != instance:
                    raise ValueError('Non-validation data or mismatched identity: ' + str(path))
                rr = [progress_row(r, settings['progress_denominator']) for r in data['rows']]
                key = canonical(dict(data=h, normalization=b['normalization'],
                                     loss_source=digest(EXP/'component_screen_b20_v1/learning.py')))
                cached = cache_dir/('labels_'+key+'.json')
                ys = None
                old_cache = a.previous_review/'cache'/('labels_'+h+'.json')
                if cached.exists():
                    item = read(cached)
                    if item['key'] != key:
                        raise ValueError('Label cache provenance changed.')
                    ys = item['labels']; pin(cached)
                elif old_cache.exists() and old_audit.exists():
                    item = read(old_cache)
                    old_key = canonical(dict(data=h, normalization=b['normalization'], audit=digest(old_audit)))
                    if item.get('key') == old_key:
                        ys = item['labels']; pin(old_cache); pin(old_audit)
                if ys is None:
                    ys = []
                    for i,r in enumerate(rr):
                        ys.append(legacy_learning.labels(r, b['normalization']).tolist())
                        if i % 24 == 0:
                            status('reading_saved_labels', instance=instance, arm=arm, done=i, total=len(rr))
                if len(ys) != len(rr):
                    raise ValueError('Wrong cached label count.')
                write(cached, dict(key=key, labels=ys))
                for r,y in zip(rr,ys):
                    if r['instance'] != instance or r['configuration'] != entry['configuration']:
                        raise ValueError('Validation row identity mismatch.')
                    rows[r['component']].append(r); targets[r['component']].append(y)
            status('validation_instance_loaded', instance=instance)
        for c in COMPONENTS:
            counts = Counter(r['instance'] for r in rows[c])
            if (set(counts) != set(b['validation_instances'][c])
                    or set(counts.values()) != {settings['states'][c]}
                    or len(rows[c]) != b['validation_states'][c]):
                raise ValueError('Missing/duplicate validation states: ' + c)
            targets[c] = np.asarray(targets[c], dtype=float)
            if not np.isfinite(targets[c]).all():
                raise ValueError('Nonfinite saved labels.')

        torch.set_num_threads(1)
        candidates, grid = [], []
        for epoch, state in checkpoints.items():
            model = SharedTransformer(b['dimensions'], settings['width'])
            model.load_state_dict(state); model.eval()
            gates = {}
            for c in COMPONENTS:
                status('validation_predictions', epoch=epoch, component=c, states=len(rows[c]))
                # Exactly the original validation batching/grouping, one member only.
                pred = predictions([model], c, rows[c], b['scalers'][c], b['normalization'])
                gates[c] = choose_gates(rows[c], pred, targets[c], b['target_scales'][c], settings)
                # With one member, std=0: all original lambda choices are identical.
                for threshold, disabled in [(0.,False),(.02,False),(.05,False),(0.,True)]:
                    value = legacy_learning.validation_score(rows[c], pred, None, 0., threshold,
                              disabled=disabled, targets=targets[c]/b['target_scales'][c])
                    grid.append(dict(epoch=epoch, component=c, threshold=threshold, disabled=disabled,
                                     mean_scaled_gain=value['mean_gain'],
                                     mean_raw_gain=value['mean_gain']*b['target_scales'][c],
                                     switch_fraction=value['switch_fraction']))
            candidates.append(dict(epoch=epoch, gates=gates,
                score=float(np.mean([g['mean_gain'] for g in gates.values()])),
                switch_fraction=float(np.mean([g['switch_fraction'] for g in gates.values()]))))
        best = min(candidates, key=lambda r:(-r['score'],r['switch_fraction'],r['epoch']))
        same = policy_signature(best) == policy_signature(current['selected'])
        status('verifying_inputs_unchanged')
        for path,h in inputs.items():
            if digest(path) != h:
                raise ValueError('Input changed during audit: ' + path)
        result = dict(completed=True, training_seed=991001, selected=best,
                      current=current['selected'], matches_current_deployment=same,
                      candidates=candidates, validation_states={c:len(rows[c]) for c in COMPONENTS},
                      validation_scope='ALL saved validation states; no evaluation-budget filter',
                      prediction_mode='original validation batching, one member only',
                      new_searches=0, new_objective_evaluations=0, training_updates=0,
                      deployed=False, inputs=inputs)
        write(out/'SELECTION.json', result)
        csv_write(out/'GATE_CANDIDATES.csv', grid)
        lines = ['共享监督：seed 991001，原流程、全部已采集验证状态',
                 '验证状态数：'+str(result['validation_states']),
                 'epoch | 验证分数 | I阈值 | C阈值 | M阈值（关闭表示始终保留规则）']
        for item in candidates:
            values = ['关闭' if item['gates'][c]['disabled'] else str(item['gates'][c]['threshold_scaled']) for c in COMPONENTS]
            lines.append(f"{item['epoch']} | {item['score']:.12g} | "+' | '.join(values))
        lines += [f"选中 epoch：{best['epoch']}",
                  f"与当前 epoch20 / 0.05,0,0（均启用）一致：{'是' if same else '否'}",
                  '阈值单位是模型分数；验证分数是组件标签尺度标准化后的单步HV净增量。',
                  '没有搜索、方案评价、训练更新或模型替换。']
        (out/'REPORT.txt').write_text('\n'.join(lines)+'\n', encoding='utf-8')
        final = dict(completed=True, stage='complete', pid=os.getpid(),
                     elapsed_seconds=round(time.monotonic()-started, 1),
                     new_searches=0, new_objective_evaluations=0, training_updates=0,
                     deployed=False, matches_current_deployment=same)
        write(out/'STATUS.json', final)
        print('\n'+'\n'.join(lines), flush=True)
        print('[SINGLE-VALID] COMPLETE '+str(out), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, default=DIAG/'ccp_icm_multiscale_shared_v1')
    p.add_argument('--deployed', type=Path, default=DIAG/'ccp_icm_shared_single_991001_b5000_v1/one_member')
    p.add_argument('--out', type=Path, default=DIAG/'ccp_shared_single_full_validation_v1')
    p.add_argument('--previous-review', type=Path, default=DIAG/'ccp_single_b5000_offline_review_v1')
    run(p.parse_args())


if __name__ == '__main__':
    main()
