#!/usr/bin/env python3
"""Read-only steps 1--3: freeze 54-pair evidence, audit labels, revalidate one member.

No search, objective evaluator, optimizer, model export, or automatic deployment is
called. Saved objective labels are used only for offline HV arithmetic.
"""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import fcntl
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import shutil
import signal
import sys
import time

sys.dont_write_bytecode = True
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '1'
HERE = Path(__file__).resolve().parent
EXP = HERE.parent
DIAG = EXP / 'transformer_ppo_runs/diagnostics'
CONFIGS = [f'S{i}' for i in range(9)]
COMPONENTS = ('I', 'C', 'M')
BANDS = ('<=1000', '1000-5000', '>5000')
ROLES = ('training', 'validation')
ARM_KIND = dict(BASE=None, I_RULE=None, C_RULE=None, M_RULE=None, ICM_HEURISTIC=None,
                I_LEARN='independent', C_LEARN='independent', M_LEARN='independent',
                ICM_INDEPENDENT='independent', ICM_SHARED_SUPERVISED='shared', ICM_TERMINAL_PPO='ppo')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    tmp.replace(path)


def csv_write(path, rows):
    if not rows:
        return
    path = Path(path)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    tmp = path.with_suffix('.tmp')
    with tmp.open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    tmp.replace(path)


def checked_plan(path):
    p = read(path)
    if canonical({k: v for k, v in p.items() if k != 'sha256'}) != p['sha256']:
        raise ValueError('PLAN checksum mismatch: ' + str(path))
    return p


class Audit:
    def __init__(self, out):
        self.out = out
        self.started = time.monotonic()
        self.last = 0
        self.inputs = read(out/'INPUTS.json') if (out/'INPUTS.json').exists() else {}
        self.warnings = []
        self.label_workers = 2

    def pin(self, path, expected=None):
        path = Path(path).resolve()
        value = sha(path)
        old = self.inputs.get(str(path))
        if (expected is not None and value != expected) or (old is not None and value != old):
            raise ValueError('Frozen input changed: ' + str(path))
        self.inputs[str(path)] = value
        return value

    def snapshot(self, path, target):
        value = self.pin(path)
        dst = self.out/'snapshot'/target
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            shutil.copyfile(path, dst)
        if sha(dst) != value:
            raise ValueError('Snapshot differs: ' + str(dst))

    def progress(self, stage, done=None, total=None, force=False):
        if not force and time.monotonic() - self.last < 20:
            return
        self.last = time.monotonic()
        s = dict(completed=False, stage=stage, done=done, total=total, pid=os.getpid(),
                 elapsed_seconds=round(self.last-self.started, 1), new_searches=0,
                 new_objective_evaluations=0, training_updates=0)
        write(self.out/'STATUS.json', s)
        write(self.out/'INPUTS.json', self.inputs)
        print('[OFFLINE]', json.dumps(s, ensure_ascii=False), flush=True)


def freeze_evidence(a, root):
    plan = checked_plan(root/'PLAN.json')
    status = read(root/'STATUS.json')
    if not status.get('completed') or plan['training_seed'] != 991001:
        raise ValueError('Expected completed second-instance single-member panel, seed 991001')
    st = plan['settings']
    if (st['population'], st['budget'], st['scenario_count'], st['pool_size'], st['progress_denominator']) != (250, 5000, 100, 8, 50000):
        raise ValueError('Unexpected fixed search settings')
    if st['screen_seeds'] != [993001, 993002, 993003] or st['milestones'] != [400, 1000, 5000]:
        raise ValueError('Unexpected search seeds/checkpoints')
    entries = plan['original_entries'] + plan['entries']
    if Counter(e['configuration'] for e in entries) != Counter({c: 2 for c in CONFIGS}):
        raise ValueError('Expected 18 development instances, two per configuration')
    if len({e['instance_id'] for e in entries}) != 18 or len({e['batch_signature'] for e in entries}) != 18:
        raise ValueError('Development instances duplicated')
    for f in ('PLAN.json', 'STATUS.json'):
        a.snapshot(root/f, f)
    for folder in ('new_instance_only', 'combined54'):
        for p in sorted((root/folder).glob('*')):
            if p.is_file():
                a.snapshot(p, str(Path(folder)/p.name))
    if not (root/'combined54/RESULTS.csv').exists():
        raise ValueError('Missing combined54/RESULTS.csv')
    bundles = {}
    for kind, meta in plan['models'].items():
        folder = Path(meta['path'])
        a.pin(folder/'BUNDLE.json', meta['bundle_sha256'])
        b = read(folder/'BUNDLE.json')
        if len(b['checkpoints']) != 1 or b['normalization'] != plan['normalization']:
            raise ValueError('Expected frozen single-member model: ' + kind)
        for f in ['BUNDLE.json'] + b['checkpoints']:
            if f != 'BUNDLE.json':
                a.pin(folder/f, b['checkpoint_sha256'][f])
            a.snapshot(folder/f, str(Path('models')/kind/f))
        bundles[kind] = (folder, b)
    jobs = list(plan['reused'])
    for e in plan['entries']:
        for seed in st['screen_seeds']:
            for arm in ARM_KIND:
                jobs.append(dict(path=str(root/'runs'/e['instance_id']/f'seed_{seed}'/arm/'RUN_COMPLETE.json'),
                                 instance=e['instance_id'], seed=seed, arm=arm))
    if len(jobs) != 594:
        raise ValueError('Expected 594 saved physical runs')
    expected = {(e['instance_id'], seed, arm) for e in entries for seed in st['screen_seeds'] for arm in ARM_KIND}
    seen = set()
    manifest = []
    for i, job in enumerate(jobs):
        p = Path(job['path'])
        digest = a.pin(p, job.get('sha256'))
        r = read(p)
        key = (r['instance'], r['seed'], r['arm'])
        if key not in expected or key in seen or not r.get('completed'):
            raise ValueError('Invalid/duplicate/incomplete saved run: ' + str(p))
        seen.add(key)
        kind = ARM_KIND[r['arm']]
        model_hash = plan['models'][kind]['bundle_sha256'] if kind else None
        entry = next(e for e in entries if e['instance_id'] == r['instance'])
        if (r['model_bundle_sha256'] != model_hash or r['stochastic'] or
                r['instance_digest'] != entry['instance_digest'] or
                {x['evaluation'] for x in r['budget_fronts']} != {400, 1000, 5000}):
            raise ValueError('Saved search identity/model/milestones differ: ' + str(p))
        manifest.append(dict(path=str(p), sha256=digest, instance=key[0], seed=key[1], arm=key[2],
                             model_bundle_sha256=model_hash, source='reused' if i < len(plan['reused']) else 'new'))
        a.progress('1_freeze_saved_runs', i+1, len(jobs))
    csv_write(a.out/'SAVED_RUNS.csv', manifest)
    # Pin the historical runtime without executing any searches. Changed sources
    # are recorded, not mistaken for a retroactive change to already saved runs.
    runtime = []
    for filename, expected_hash in plan['files'].items():
        p = Path(filename)
        got = a.pin(p) if p.exists() else None
        runtime.append(dict(path=filename, expected=expected_hash, actual=got, matches=got == expected_hash))
    csv_write(a.out/'RUNTIME_PROVENANCE.csv', runtime)
    if any(not r['matches'] for r in runtime):
        a.warnings.append('部分历史运行文件缺失或哈希变化，见 RUNTIME_PROVENANCE.csv；已有结果未修改。')
    return plan, bundles


def band(e):
    return '<=1000' if e <= 1000 else '1000-5000' if e <= 5000 else '>5000'


def balanced_mean(rows, values):
    by = defaultdict(lambda: defaultdict(list))
    for r, v in zip(rows, values):
        by[r['configuration']][r['instance']].append(float(v))
    return float(np.mean([np.mean([np.mean(v) for v in inst.values()]) for inst in by.values()])) if by else None


def summarize(records, field=None):
    result = []
    for role in ROLES:
        for c in COMPONENTS:
            for cfg in ['ALL'] + CONFIGS:
                whole = [r for r in records if r['role'] == role and r['component'] == c and
                         (cfg == 'ALL' or r['configuration'] == cfg)]
                for phase in BANDS + ('<=5000', 'ALL'):
                    rr = [r for r in whole if phase == 'ALL' or (r['evaluation'] <= 5000 if phase == '<=5000' else band(r['evaluation']) == phase)]
                    item = dict(role=role, component=c, configuration=cfg, band=phase, states=len(rr),
                                instances=len({r['instance'] for r in rr}),
                                fraction_of_states=len(rr)/len(whole) if whole else None)
                    if not field:
                        for k in ('legal_alternatives', 'feasible_alternatives', 'nonzero_alternatives',
                                  'positive_opportunity', 'ranking_signal'):
                            item[k] = sum(r[k] for r in rr)
                        item['mean_best_raw_gain'] = balanced_mean(rr, [r['best_raw_gain'] for r in rr])
                    else:
                        sw = [r for r in rr if r[field+'_choice'] != 0]
                        item.update(switches=len(sw), switch_percent=100*len(sw)/len(rr) if rr else None,
                                    mean_raw_gain=balanced_mean(rr, [r[field+'_gain'] for r in rr]),
                                    better=sum(r[field+'_gain'] > 1e-12 for r in sw),
                                    tied=sum(abs(r[field+'_gain']) <= 1e-12 for r in sw),
                                    worse=sum(r[field+'_gain'] < -1e-12 for r in sw),
                                    prediction_mode='per_state_deployment' if role == 'validation' else 'batched_diagnostic',
                                    model=field)
                    result.append(item)
    return result


def label_worker(row, norm):
    """Pure HV arithmetic from saved outcomes; worker never receives an out path."""
    sys.path.insert(0, str(EXP/'joint_icm_multiscale_terminal_v1'))
    from bridge import legacy_learning as learning, torch
    torch.set_num_threads(1)
    return learning.labels(row, norm).tolist()


def coverage_first(a, sources):
    counts = Counter()
    instances = defaultdict(set)
    for i,(filename,digest) in enumerate(sorted(sources.items())):
        a.pin(filename,digest)
        d=read(filename)
        for r in d['rows']:
            for cfg in ('ALL',r['configuration']):
                for phase in (band(r['evaluation']), 'ALL') + (('<=5000',) if r['evaluation']<=5000 else ()):
                    key=(d['role'],r['component'],cfg,phase)
                    counts[key]+=1;instances[key].add(r['instance'])
        a.progress('2_coverage_inventory',i+1,len(sources))
    rows=[]
    for role in ROLES:
        for c in COMPONENTS:
            for cfg in ['ALL']+CONFIGS:
                for phase in BANDS+('<=5000','ALL'):
                    key=(role,c,cfg,phase);total=counts[role,c,cfg,'ALL']
                    rows.append(dict(role=role,component=c,configuration=cfg,band=phase,
                                     states=counts[key],instances=len(instances[key]),
                                     fraction_of_states=counts[key]/total if total else None))
    csv_write(a.out/'STATE_COUNTS.csv',rows)
    for r in rows:
        if r['configuration']=='ALL' and r['band']=='<=5000':
            print('[OFFLINE] coverage',r,flush=True)


def stage_data(a, bundles, plan, pool=None):
    sb = bundles['shared'][1]
    ib = bundles['independent'][1]
    for k in ('normalization', 'scalers', 'target_scales', 'training_instances', 'validation_instances'):
        if sb[k] != ib[k]:
            raise ValueError('Supervision sources differ: ' + k)
    sources = {p: h for p, h in sb['files'].items() if Path(p).name == 'DATA.json'}
    other = {p: h for p, h in ib['files'].items() if Path(p).name == 'DATA.json'}
    if sources != other:
        raise ValueError('Independent/shared label file lists differ')
    coverage_first(a,sources)
    train = set(sb['training_instances']['I'])
    valid = set(sb['validation_instances']['I'])
    dev = {e['instance_id'] for e in plan['entries']+plan['original_entries']}
    if train & valid or (train | valid) & dev or (len(train), len(valid)) != (54, 18):
        raise ValueError('Missing or overlapping instance roles')
    for c in COMPONENTS:
        if set(sb['training_instances'][c]) != train or set(sb['validation_instances'][c]) != valid:
            raise ValueError('Component split mismatch')
    policies = dict(shared=SharedPolicy(bundles['shared'][0]), independent=IndependentPolicy(bundles['independent'][0]))
    records, valid_rows, targets = [], {c: [] for c in COMPONENTS}, {c: [] for c in COMPONENTS}
    observed = Counter()
    for fi, (filename, expected_hash) in enumerate(sorted(sources.items())):
        a.pin(filename, expected_hash)
        d = read(filename)
        role, name, cfg = d['role'], d['instance'], d['configuration']
        if role not in ROLES or name not in (train if role == 'training' else valid) or cfg not in CONFIGS:
            raise ValueError('Unexpected label file role/identity: ' + filename)
        rows = [progress_row(r, sb['settings']['progress_denominator']) for r in d['rows']]
        cache = a.out/'cache'/('labels_'+expected_hash+'.json')
        cache_key = canonical(dict(data=expected_hash, normalization=sb['normalization'], audit=sha(__file__)))
        if cache.exists():
            saved = read(cache)
            if saved['key'] != cache_key:
                raise ValueError('Label cache provenance mismatch')
            ys = saved['labels']
        else:
            if pool is None:
                ys = []
                for ri, row in enumerate(rows):
                    ys.append(legacy_learning.labels(row, sb['normalization']).tolist())
                    a.progress(f'2_labels_{name}', ri+1, len(rows))
            else:
                ys = [None]*len(rows)
                jobs = {pool.submit(label_worker,{k:r[k] for k in ('divisor','population_raw','labels')},sb['normalization']):i
                        for i,r in enumerate(rows)}
                for finished,future in enumerate(as_completed(jobs),1):
                    ys[jobs[future]]=future.result()
                    a.progress(f'2_labels_{name}',finished,len(rows))
            write(cache, dict(key=cache_key, labels=ys))
        if len(rows) != len(ys):
            raise ValueError('Wrong label count')
        pred = {}
        for kind, policy in policies.items():
            pred[kind] = {}
            for c in COMPONENTS:
                sub = [r for r in rows if r['component'] == c]
                if sub:
                    pred[kind][c] = predictions(policy.models, c, sub, sb['scalers'][c], sb['normalization'])[0]
        local_index = Counter()
        for ri, (r, y) in enumerate(zip(rows, ys)):
            c = r['component']
            if c not in COMPONENTS or r['instance'] != name or r['configuration'] != cfg:
                raise ValueError('Label row identity differs')
            mask = np.asarray(r['legal_mask'], dtype=bool)
            if mask.shape != (8,) or not mask[0] or len(y) != 8 or not np.isfinite(y).all():
                raise ValueError('Invalid saved label/mask')
            y = np.asarray(y)
            alt = [j for j in range(1, 8) if mask[j]]
            scaled = y[mask]/sb['target_scales'][c]
            rec = dict(role=role, configuration=cfg, instance=name, component=c,
                       source=filename, source_row=ri, evaluation=r['evaluation'],
                       legal_alternatives=len(alt),
                       feasible_alternatives=sum(any(child['feasible'] for child in r['labels'][j]) for j in alt),
                       nonzero_alternatives=sum(abs(y[j]) > 1e-12 for j in alt),
                       positive_opportunity=int(any(y[j] > 1e-12 for j in alt)),
                       ranking_signal=int(float(np.ptp(scaled)) > 1e-6),
                       best_raw_gain=float(max(0., max((y[j] for j in alt), default=0.))))
            for kind, policy in policies.items():
                scores = pred[kind][c][local_index[c]]
                if not np.isfinite(scores).all() or scores[0] != 0:
                    raise ValueError('Invalid model scores/rule anchor')
                gate = policy.bundle['selected']['gates'][c]
                choice = legacy_learning.decide(scores[None], mask, gate['lambda'], gate['threshold_scaled'], gate['disabled'])
                actual = policy.select(r) if role == 'validation' else choice
                rec[kind+'_choice'] = int(actual)
                rec[kind+'_gain'] = float(y[actual])
                rec[kind+'_batch_deploy_mismatch'] = bool(actual != choice) if role == 'validation' else None
            if role == 'validation':
                valid_rows[c].append(r)
                targets[c].append(y)
            records.append(rec)
            observed[c, role] += 1
            local_index[c] += 1
        a.progress('2_saved_data_files', fi+1, len(sources), force=True)
        # Save partial summaries, so the first coverage output is visible early.
        csv_write(a.out/'STATE_COVERAGE.csv', summarize(records))
    for c in COMPONENTS:
        for role in ROLES:
            if observed[c, role] != sb[role+'_states'][c]:
                raise ValueError('Saved state count differs: ' + c+'/'+role)
    csv_write(a.out/'LABEL_SIGNAL.csv', summarize(records))
    csv_write(a.out/'SINGLE_POLICY_AUDIT.csv', summarize(records, 'shared')+summarize(records, 'independent'))
    csv_write(a.out/'PER_STATE.csv', records)
    write(a.out/'DATA_SPLITS.json', dict(training=sorted(train), validation=sorted(valid), development=sorted(dev),
          development_used_for_selection=False, label_files=sources, current_gate_clipping={k:p.range_checks for k,p in policies.items()}))
    return valid_rows, {c: np.asarray(v) for c,v in targets.items()}, records, sources


def gate_scores(rows, pred, targets, scale, threshold, disabled=False):
    choices = np.asarray([legacy_learning.decide(p[None], r['legal_mask'], 0., threshold, disabled)
                          for r,p in zip(rows, pred)], dtype=int)
    gains = targets[np.arange(len(rows)), choices]
    return dict(mean_raw_gain=balanced_mean(rows, gains), mean_scaled_gain=balanced_mean(rows, gains/scale),
                switch_fraction=float(np.mean(choices != 0)), threshold_scaled=threshold, disabled=disabled,
                lambda_value=0.)


def validation_selection(a, bundles, valid, targets, sources):
    candidates, selected, availability = [], {}, []
    root = Path(next(iter(sources))).parents[3]
    original = DIAG/'ccp_component_joint_b5000_v1/PLAN.json'
    a.pin(original)
    old = read(original)
    source_folders = dict(shared=root/'shared_model', independent=Path(old['independent_model']))
    for kind in ('shared', 'independent'):
        folder, b = bundles[kind]
        source = source_folders[kind]
        # The training PLAN binds retained, nonselected checkpoints to the fit.
        train_plan = source/'PLAN.json'
        if train_plan.exists():
            a.pin(train_plan, b['plan_sha256'])
        else:
            a.warnings.append(f'{kind} 训练 PLAN 缺失；只检查当前已冻结检查点。')
        per_epoch = []
        for epoch in b['settings']['checkpoint_epochs']:
            name = f'seed_991001_epoch_{epoch}.pt'
            p = folder/name if name in b['checkpoints'] else source/name
            ok = p.exists() and (name in b['checkpoints'] or train_plan.exists())
            availability.append(dict(model=kind, epoch=epoch, path=str(p), available=ok))
            if not ok:
                continue
            a.pin(p, b['checkpoint_sha256'].get(name))
            ck = load_checkpoint(p)
            if ck.get('plan_sha256') != b['plan_sha256'] or ck.get('epoch') != epoch:
                raise ValueError('Checkpoint training provenance mismatch: ' + str(p))
            model = (SharedTransformer if kind == 'shared' else IndependentBank)(b['dimensions'], b['settings']['width'])
            model.load_state_dict(ck['model'])
            model.eval()
            gates = {}
            for c in COMPONENTS:
                ids = [i for i,r in enumerate(valid[c]) if r['evaluation'] <= 5000]
                rows = [valid[c][i] for i in ids]
                expected = set(b['validation_instances'][c])
                if {r['instance'] for r in rows} != expected:
                    a.warnings.append(f'{kind}/{c} 在 B5000 内缺少验证实例覆盖，不提出完整模型选择。')
                    continue
                cache_key = canonical(dict(checkpoint=sha(p), component=c, data=sources,
                                           normalization=b['normalization'], scalers=b['scalers'], audit=sha(__file__)))
                cache = a.out/'cache'/('pred_'+cache_key+'.json')
                if cache.exists():
                    pred = np.asarray(read(cache)['scores'])
                else:
                    # Use the exact one-state inference path used at deployment;
                    # near-ties must not be decided by batch-size roundoff.
                    pp = []
                    for i,r in enumerate(rows):
                        pp.append(predictions([model], c, [r], b['scalers'][c], b['normalization'])[0,0].tolist())
                        a.progress(f'3_{kind}_epoch{epoch}_{c}', i+1, len(rows))
                    pred = np.asarray(pp)
                    write(cache, dict(scores=pp))
                options = [gate_scores(rows,pred,targets[c][ids],b['target_scales'][c],float(t))
                           for t in b['settings']['gate_threshold_scaled']]
                options.append(gate_scores(rows,pred,targets[c][ids],b['target_scales'][c],0.,True))
                gate = min(options,key=lambda g:(-g['mean_scaled_gain'],g['switch_fraction'],not g['disabled'],g['threshold_scaled']))
                gates[c] = gate
                for g in options:
                    candidates.append(dict(model=kind, epoch=epoch, component=c, states=len(rows),
                                           instances=len(expected), **g))
            if len(gates) == 3:
                per_epoch.append(dict(epoch=epoch, gates=gates,
                                      score=float(np.mean([g['mean_scaled_gain'] for g in gates.values()])),
                                      switch_fraction=float(np.mean([g['switch_fraction'] for g in gates.values()]))))
            csv_write(a.out/'GATE_CANDIDATES.csv', candidates)
            a.progress(f'3_{kind}_epoch{epoch}_complete', force=True)
        complete = len(per_epoch) == len(b['settings']['checkpoint_epochs'])
        best = min(per_epoch,key=lambda v:(-v['score'],v['switch_fraction'],v['epoch'])) if per_epoch else None
        changed = None if best is None else (best['epoch'] != b['selected']['epoch'] or any(
            best['gates'][c]['disabled'] != b['selected']['gates'][c]['disabled'] or
            best['gates'][c]['threshold_scaled'] != b['selected']['gates'][c]['threshold_scaled'] for c in COMPONENTS))
        selected[kind] = dict(complete_grid=complete, current_epoch=b['selected']['epoch'],
                             current_gates=b['selected']['gates'], best_available=best,
                             selection_complete=complete, proposed_change=changed,
                             deployed=False, scope='saved validation states with evaluation <= 5000 only')
        if not complete:
            a.warnings.append(f'{kind} 检查点/前期覆盖不完整；best_available 不能当作完整重选结果。')
    csv_write(a.out/'CHECKPOINT_AVAILABILITY.csv', availability)
    csv_write(a.out/'SUPERVISED_SELECTION.csv', [dict(model=k, complete_grid=v['complete_grid'],
        current_epoch=v['current_epoch'], proposed_epoch=v['best_available']['epoch'] if v['best_available'] else None,
        proposed_change=v['proposed_change'], deployed=False) for k,v in selected.items()])
    write(a.out/'PROPOSED_SELECTION.json', dict(models=selected, training_seed=991001,
          new_searches=0, training_updates=0, model_files_written=False,
          score='Within instance state mean, then configuration-equal instance means, then component-equal scaled label gain',
          qualification='Selection on reused validation states only; NOT independent evidence of final-search improvement.'))
    return selected


def ppo_provenance(a, bundles):
    b = bundles['ppo'][1]
    pilot = b.get('pilot', {})
    root = DIAG/'ccp_joint_ppo_b5000_pilot_v2'
    record = dict(deployed_member=b['checkpoints'], shared_pretraining_configurations=b['settings']['configurations'],
                  recorded_ppo_protocol=pilot, source_records={}, update_metrics=[])
    for name in ('PLAN.json', 'STATUS.json', 'SELECTION.json', 'TRAINING_PROGRESS.json'):
        p = root/name
        if p.exists():
            a.snapshot(p, 'ppo_provenance/'+name)
            record['source_records'][name] = read(p)
        else:
            a.warnings.append('PPO 溯源记录缺失：' + str(p))
    original_plan = record['source_records'].get('PLAN.json', {})
    if original_plan and original_plan.get('pilot', {}).get('configurations') != pilot.get('configurations'):
        raise ValueError('PPO bundle/protocol configuration mismatch')
    for p in sorted((root/'checkpoints').glob('update_*/UPDATE.json')):
        a.pin(p)
        record['update_metrics'].append(dict(path=str(p), metrics=read(p)))
    name = b['checkpoints'][0]
    matches = []
    for p in sorted((root/'checkpoints').glob('update_*/'+name)):
        if a.pin(p) == b['checkpoint_sha256'][name]:
            matches.append(p.parent.name)
    record['deployed_member_matching_checkpoints'] = matches
    if not matches:
        a.warnings.append('未找到与当前单模型 PPO 权重逐字节相同的源检查点，需补齐源文件。')
    pairs = []
    for p in sorted((root/'runs/training').glob('u*/*/seed_*/RUN_COMPLETE.json')):
        a.pin(p)
        r = read(p)
        update = int(p.parents[2].name[1:])
        ref = root/'runs/training_reference'/p.parents[2].name/r['instance']/f"seed_{r['seed']}"/'RUN_COMPLETE.json'
        if not ref.exists():
            a.warnings.append('PPO 配对参考搜索缺失：'+str(ref))
            continue
        a.pin(ref)
        v = read(ref)
        for key in ('instance', 'configuration', 'seed', 'scenario_digest', 'instance_digest'):
            if r[key] != v[key]:
                raise ValueError('PPO reward pair mismatch: '+key)
        if (not r.get('completed') or not v.get('completed') or not r['stochastic'] or v['stochastic']):
            raise ValueError('PPO saved training/reference flags differ')
        delta = r['final_hypervolume']-v['final_hypervolume']
        metric = next((m['metrics'] for m in record['update_metrics'] if m['metrics']['update']==update), {})
        saved = next((x for x in metric.get('reward_pairs', []) if x['instance']==r['instance'] and x['seed']==r['seed']), None)
        reward_ok = saved is not None and abs(saved['paired_return']-delta) <= 1e-12
        if saved is not None and not reward_ok:
            raise ValueError('Saved PPO reward is not the recorded paired HV difference')
        pairs.append(dict(update=update, instance=r['instance'], configuration=r['configuration'], seed=r['seed'],
                          policy_hv=r['final_hypervolume'], reference_hv=v['final_hypervolume'],
                          paired_return=delta, update_reward_verified=reward_ok))
    csv_write(a.out/'PPO_TRAINING_PAIRS.csv', pairs)
    record['actual_policy_trajectories'] = len(pairs)
    record['actual_training_configurations'] = sorted({r['configuration'] for r in pairs})
    record['actual_trajectories_per_update'] = dict(Counter(r['update'] for r in pairs))
    expected_count = pilot.get('updates',0)*pilot.get('episodes_per_update',0)
    if len(pairs) != expected_count or any(not r['update_reward_verified'] for r in pairs):
        a.warnings.append('PPO 原始轨迹或更新奖励记录不完整；实际覆盖仅按找到的记录报告。')
    record['historical_validation_is_not_new_single_member_validation'] = True
    write(a.out/'PPO_PROVENANCE.json', record)
    return record


def run(a, root):
    a.pin(__file__)
    a.progress('1_freeze', force=True)
    plan, bundles = freeze_evidence(a, root)
    # Check only the code actually used by the offline forward/label computation.
    needed = ('joint_icm_multiscale_v1/shared.py', 'joint_icm_multiscale_v1/bootstrap.py',
              'joint_icm_multiscale_terminal_v1/bridge.py', 'component_screen_b20_v1/learning.py',
              'component_screen_b20_v1/common.py')
    for rel in needed:
        p = EXP/rel
        expected = bundles['independent'][1]['files'].get(str(p)) or bundles['shared'][1]['files'].get(str(p))
        if expected is None:
            raise ValueError('Missing model/runtime provenance for '+str(p))
        a.pin(p, expected)
    global np, SharedPolicy, IndependentPolicy, SharedTransformer, IndependentBank
    global legacy_learning, predictions, progress_row, load_checkpoint
    sys.path.insert(0, str(EXP/'joint_icm_multiscale_terminal_v1'))
    from bridge import (np, torch, SharedPolicy, IndependentPolicy, SharedTransformer, IndependentBank,
                        legacy_learning, predictions, progress_row, load_checkpoint)
    torch.set_num_threads(1)
    # Also pin the imported HV implementation and normalization class's source.
    import inspect
    for obj in (legacy_learning.exact_hv_3d, legacy_learning.episode_normalizer):
        p = Path(inspect.getfile(obj)).resolve()
        expected = bundles['shared'][1]['files'].get(str(p))
        if expected is None:
            raise ValueError('Missing HV runtime provenance: '+str(p))
        a.pin(p, expected)
    if a.label_workers == 1:
        valid, targets, records, sources = stage_data(a, bundles, plan)
    else:
        with ProcessPoolExecutor(max_workers=a.label_workers,mp_context=multiprocessing.get_context('spawn')) as pool:
            valid, targets, records, sources = stage_data(a, bundles, plan, pool)
    selected = validation_selection(a, bundles, valid, targets, sources)
    ppo = ppo_provenance(a, bundles)
    # Detect accidental or concurrent changes to every file consumed by this run.
    a.progress('verify_inputs_unchanged', force=True)
    for p, expected in list(a.inputs.items()):
        a.pin(p, expected)
    write(a.out/'INPUTS.json', a.inputs)
    lines = ['# 单模型 B5000 离线核对（步骤1—3）', '',
             '原54对×11方法=594次搜索已记录哈希；结果、模型与阈值未修改。',
             '本次新增搜索0，新增方案评价0，参数更新0。保存的目标值用于重算标签，不调用运输方案评价器。', '',
             '| 组件 | 训练状态≤5000/全部 | 验证状态≤5000/全部 |', '|---|---:|---:|']
    for c in COMPONENTS:
        cells=[]
        for role in ROLES:
            rr=[r for r in records if r['component']==c and r['role']==role]
            cells.append(f"{sum(r['evaluation']<=5000 for r in rr)}/{len(rr)}")
        lines.append('| '+c+' | '+' | '.join(cells)+' |')
    lines += ['', '| 模型 | 当前轮次 | 前期验证建议轮次 | 网格完整 | 建议是否变化 |', '|---|---:|---:|---|---|']
    for k,v in selected.items():
        ep = v['best_available']['epoch'] if v['best_available'] else '不可选'
        lines.append(f"| {k} | {v['current_epoch']} | {ep} | {v['complete_grid']} | {v['proposed_change']} |")
    mismatch = sum(bool(r[k+'_batch_deploy_mismatch']) for r in records for k in ('shared','independent'))
    lines += ['', f'验证状态的批量/逐状态部署改选不一致次数：{mismatch}。重选使用逐状态推理结果。',
              '阈值单位是模型分数；标签净增益是单步HV差，不是最终HV百分比。',
              '训练状态仅用于描述数据覆盖，不参与此次轮次/阈值选择。固定成员991001；不评价或平均其他成员。',
              '建议仍需完整B5000验证搜索确认，本脚本不会自动启动；当前54对开发结果不参与选择。', '',
              'PPO的实际协议、训练更新记录、包含u0及仅更新后检查点的选择见 PPO_PROVENANCE.json。',
              f"实际找到的策略训练轨迹：{ppo['actual_policy_trajectories']}；配置：{', '.join(ppo['actual_training_configurations']) or '无完整记录'}。",
              '不能把监督预训练覆盖直接当作PPO轨迹覆盖；历史验证分也不能当作单成员重新验证分。', '',
              '详细输出：STATE_COVERAGE.csv、LABEL_SIGNAL.csv、SINGLE_POLICY_AUDIT.csv、',
              'GATE_CANDIDATES.csv、PROPOSED_SELECTION.json、CHECKPOINT_AVAILABILITY.csv。', '', '待处理事项：']
    lines += ['- '+w for w in a.warnings] or ['- 无文件缺失警告；这不等于已证明最终搜索收益。']
    (a.out/'REPORT_CN.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    state=dict(completed=True, stage='offline_steps_1_to_3', pid=os.getpid(),
               elapsed_seconds=time.monotonic()-a.started, frozen_runs=594, pairs_per_method=54,
               new_searches=0,new_objective_evaluations=0,training_updates=0,models_modified=False,
               checkpoint_grids_complete=all(v['complete_grid'] for v in selected.values()),
               warnings=a.warnings, automatic_further_experiments=False)
    write(a.out/'STATUS.json',state)
    print('\n'.join(lines),flush=True)
    print('[OFFLINE] COMPLETE',a.out,flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=DIAG/'ccp_icm_single_member_second_instance_b5000_v1')
    p.add_argument('--out',type=Path,default=DIAG/'ccp_single_b5000_offline_review_v1')
    p.add_argument('--label-workers',type=int,choices=(1,2),default=2)
    args=p.parse_args();root=args.source.resolve();out=args.out.resolve()
    if root==out or root in out.parents or out in root.parents or out.name in ('models','runs','combined54'):
        raise ValueError('Use a separate sibling audit output, never an experiment/model directory')
    out.mkdir(parents=True,exist_ok=True)
    with (out/'.lock').open('a') as lock:
        try:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit('此核对任务已在运行；请查看 pipeline.nohup.log。')
        a=Audit(out)
        a.label_workers=args.label_workers
        # Never allow an audit output inside any prior experiment/model tree.
        if out.name in ('ccp_icm_single_member_panel_b5000_v1', 'ccp_icm_multiscale_shared_v1'):
            raise ValueError('Output must be a new audit directory')
        def stop(signum, frame):
            raise KeyboardInterrupt(f'signal {signum}')
        signal.signal(signal.SIGTERM,stop)
        if (out/'STATUS.json').exists() and read(out/'STATUS.json').get('completed'):
            for filename,expected in list(a.inputs.items()): a.pin(filename,expected)
            print('已完成，原输入哈希未变：',out/'REPORT_CN.md');return
        try:
            run(a,root)
        except BaseException as e:
            write(out/'INPUTS.json',a.inputs)
            write(out/'STATUS.json',dict(completed=False,stage='interrupted' if isinstance(e,KeyboardInterrupt) else 'error',
                error=str(e),pid=os.getpid(),new_searches=0,training_updates=0,
                resume='Repeat the same command; label/prediction caches are reused.'))
            raise


if __name__=='__main__':
    main()
