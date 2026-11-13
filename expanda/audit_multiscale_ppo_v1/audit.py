#!/usr/bin/env python3
"""Read-only PPO diagnostics: saved candidate replay, advantages, and u0 identity.

No search, optimization, scheduler mutation, checkpoint selection or source write.
Replay samples are fixed by trajectory identity, never by reward or model output.
"""
from pathlib import Path
from collections import defaultdict
import argparse
import csv
import hashlib
import json
import os
import sys
import time

for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '1'
import numpy as np

HERE = Path(__file__).resolve().parent
EXPANDA = HERE.parent


def read(path):
    for attempt in range(3):
        try:
            return json.loads(path.read_text(encoding='utf-8'))
        except json.JSONDecodeError:
            if attempt == 2:
                raise
            time.sleep(.1)


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def csv_write(path, rows):
    if rows:
        with path.open('w', encoding='utf-8-sig', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)


def pool_indices(count, cap, identity):
    if cap == 0 or count <= cap:
        return np.arange(count)
    seed = int(hashlib.sha256(identity.encode()).hexdigest()[:16], 16)
    return np.sort(np.random.default_rng(seed).choice(count, cap, replace=False))


def compare_probabilities(p, q):
    """Exact forward KL on identical pools; zeros outside legal support stay zero."""
    p, q = np.asarray(p, float), np.asarray(q, float)
    if p.shape != q.shape or p.ndim != 2:
        raise ValueError('Mismatched probability arrays')
    if (not np.isfinite(p).all() or not np.isfinite(q).all()
            or (p < 0).any() or (q < 0).any()
            or not np.allclose(p.sum(1), 1) or not np.allclose(q.sum(1), 1)):
        raise ValueError('Invalid probabilities')
    if ((p > 0) != (q > 0)).any():
        raise ValueError('Different support/gates; finite KL comparison is invalid')
    lp = np.log(np.where(p > 0, p, 1.))
    lq = np.log(np.where(q > 0, q, 1.))
    return dict(kl_nats=np.maximum(0., (p * (lp-lq)).sum(1)),
                entropy_reference_nats=-(p*lp).sum(1),
                entropy_checkpoint_nats=-(q*lq).sum(1),
                top1_changed=(p.argmax(1) != q.argmax(1)).astype(float))


def advantage_stats(reward, old_values):
    values = np.asarray(old_values, float)
    if not np.isfinite(values).all() or not np.isfinite(reward):
        raise ValueError('Nonfinite return/value')
    advantage = reward-values
    return dict(actions=len(values), terminal_hv=float(reward),
                old_value_mean=float(values.mean()), advantage_mean=float(advantage.mean()),
                advantage_std=float(advantage.std()), advantage_min=float(advantage.min()),
                advantage_max=float(advantage.max()), positive_fraction=float((advantage>0).mean()))


def audit_finished_pairs(root, plan, selected):
    """Exact saved-record checks, not inference from similar mean HV."""
    rows = []
    for instance in plan['screen']:
        for seed in plan['settings']['screen_seeds']:
            folder = root/'screen'/instance['instance_id']/f'seed_{seed}'
            paths = [folder/arm/'RUN_COMPLETE.json' for arm in ('ICM_SHARED_SUPERVISED', 'ICM_TERMINAL_PPO')]
            if not all(p.exists() for p in paths):
                continue
            a,b = [read(p) for p in paths]
            if not a.get('completed') or not b.get('completed'):
                continue
            for key in ('instance', 'seed', 'scenario_digest', 'instance_digest', 'plan_sha256'):
                if a[key] != b[key]:
                    raise ValueError('Unpaired screen identity: '+str(folder))
            if b['model_bundle_sha256'] != selected['bundle_sha256']:
                raise ValueError('Wrong selected checkpoint in screen run')
            row = dict(instance=a['instance'], seed=seed,
                       final_hv_exact=a['final_hypervolume']==b['final_hypervolume'],
                       raw_front_exact=a['final_front_raw']==b['final_front_raw'],
                       evaluation_order_digest_exact=(a.get('evaluation_order_digest') is not None
                           and a['evaluation_order_digest']==b.get('evaluation_order_digest')),
                       evaluation_count_exact=a['real_objective_evaluation_count']==b['real_objective_evaluation_count'],
                       selection_counts_exact=a.get('selection') is not None and a['selection']==b.get('selection'))
            row['all_checked_exact'] = all(v for k,v in row.items() if k not in ('instance','seed'))
            rows.append(row)
    return rows


def execute(root, out, cap):
    if out == root or root in out.parents or out in root.parents:
        raise ValueError('Audit output must be separate from experiment source')
    plan = read(root/'PLAN.json')
    sys.path.insert(0, str(EXPANDA/'joint_icm_multiscale_terminal_v1'))
    from policy import TerminalPolicy
    from bridge import SharedPolicy, verify, canonical_digest
    import torch
    torch.set_num_threads(1)
    if canonical_digest({k:v for k,v in plan.items() if k!='sha256'}) != plan['sha256']:
        raise ValueError('Invalid frozen plan')
    verify(plan['files'])
    source = Path(plan['source'])/'shared_model'
    if out == source or source in out.parents or out in source.parents:
        raise ValueError('Audit output overlaps supervised source')
    policies = {u:TerminalPolicy(root/'terminal/checkpoints'/f'update_{u:03d}') for u in (0,4,8)}
    shared = SharedPolicy(source)
    for u,p in policies.items():
        for key in ('scalers','normalization','dimensions','target_scales','feature_version'):
            if p.bundle[key] != shared.bundle[key]:
                raise ValueError(f'Incompatible {key} at update {u}')
        if p.bundle['selected']['gates'] != shared.bundle['selected']['gates']:
            raise ValueError('Changed gate at update '+str(u))
    actor_exact = len(shared.models)==len(policies[0].models) and all(
        torch.equal(value, b.state_dict()[key])
        for a,b in zip(shared.models,policies[0].models)
        for key,value in a.state_dict().items())
    selected = read(root/'terminal/SELECTED.json')
    pairs = audit_finished_pairs(root, plan, selected)
    out.mkdir(parents=True, exist_ok=False)
    manifest = {str(root/'PLAN.json'):sha(root/'PLAN.json')}
    for folder in [source]+[root/'terminal/checkpoints'/f'update_{u:03d}' for u in (0,4,8)]:
        manifest[str(folder/'BUNDLE.json')] = sha(folder/'BUNDLE.json')
        manifest.update({str(folder/p):h for p,h in read(folder/'BUNDLE.json')['checkpoint_sha256'].items()})
    csv_write(out/'UPDATE0_REPRODUCTION.csv', pairs)
    records = []; replay = []; advantages = []; update_rows = []; identity_checks = []
    schedule = read(root/'terminal/TRAINING_SCHEDULE.json')
    from prepare import training_schedule
    if schedule != training_schedule(plan):
        raise ValueError('Training schedule changed')
    by_update = {}
    for u in range(1,plan['settings']['ppo']['updates']+1):
        path = root/'terminal/checkpoints'/f'update_{u:03d}'/'UPDATE.json'
        m = read(path); manifest[str(path)] = sha(path); by_update[u] = m
        update_rows.append({k:m.get(k) for k in ('update','optimizer_steps','kl_early_stop','last_minibatch_kl',
                          'on_policy_logp_max_error','actions','complete_episodes')})
    for number, job in enumerate(schedule, 1):
        u = job['update'];folder=root/'terminal/rollouts'/f'u{u:03d}'/job['instance']
        record_path=folder/'RUN_COMPLETE.json';r=read(record_path);path=folder/'TRAJECTORY.npz'
        if (not r['completed'] or not r['stochastic'] or r['phase']!='training'
                or r['plan_sha256']!=plan['sha256'] or r['seed']!=job['seed'] or r['instance']!=job['instance']):
            raise ValueError('Invalid training trajectory record')
        if sha(path)!=r['trajectory_sha256']:
            raise ValueError('Trajectory digest changed: '+str(path))
        manifest[str(path)] = r['trajectory_sha256']; manifest[str(record_path)] = sha(record_path)
        metadata = dict(training_update=u, instance=r['instance'], configuration=r['configuration'], seed=r['seed'])
        records.append({**metadata,'terminal_hv':r['final_hypervolume']})
        with np.load(path, allow_pickle=False) as arrays:
            for c in ('I','C','M'):
                if c+'_action' not in arrays:
                    continue
                n = len(arrays[c+'_action'])
                if not n:
                    continue
                advantages.append({**metadata,'component':c,**advantage_stats(r['final_hypervolume'],arrays[c+'_value'])})
                ids = pool_indices(n,cap,f'{u}/{r["instance"]}/{r["seed"]}/{c}')
                x = tuple(torch.tensor(arrays[c+'_'+k][ids],dtype=torch.float32)
                          for k in ('batch','candidate','context','marker'))
                mask = torch.tensor(arrays[c+'_mask'][ids],dtype=torch.bool)
                outputs = {v:[] for v in policies};supervised_choices=[]
                with torch.no_grad():
                    for start in range(0,len(ids),16):
                        chunk=tuple(t[start:start+16] for t in x);local_mask=mask[start:start+16]
                        for v,p in policies.items():
                            dist,_=p.distribution(c,chunk,local_mask);outputs[v].append(dist.probs.numpy())
                        pred=np.stack([m(c,*chunk).numpy() for m in shared.models])
                        # Actual SharedPolicy.select numpy gate/argmax arithmetic.
                        from bridge import legacy_learning
                        gate=shared.bundle['selected']['gates'][c]
                        supervised_choices.extend(legacy_learning.decide(pred[:,i],local_mask[i].numpy(),
                            gate['lambda'],gate['threshold_scaled'],gate['disabled']) for i in range(len(local_mask)))
                outputs={v:np.concatenate(parts) for v,parts in outputs.items()}
                identity_checks.append(dict(**metadata,component=c,pools=len(ids),
                    shared_vs_u0_top1_differences=int((np.asarray(supervised_choices)!=outputs[0].argmax(1)).sum())))
                for v in (4,8):
                    metrics=compare_probabilities(outputs[0],outputs[v])
                    replay.append({**metadata,'component':c,'checkpoint':v,'available_pools':n,'sampled_pools':len(ids),
                                   **{key:float(value.mean()) for key,value in metrics.items()}})
        print(f'[MULTI-AUDIT] {number}/{len(schedule)} saved training trajectories; zero new evaluations',flush=True)
        csv_write(out/'REPLAY_PER_TRAJECTORY.csv',replay)
    summary=[]
    for c in ('I','C','M'):
        for u in (4,8):
            group=[r for r in replay if r['component']==c and r['checkpoint']==u]
            if group:
                summary.append(dict(component=c,checkpoint=u,trajectories=len(group),
                    sampled_pools=sum(r['sampled_pools'] for r in group),
                    **{key:float(np.mean([r[key] for r in group])) for key in (
                        'kl_nats','entropy_reference_nats','entropy_checkpoint_nats','top1_changed')}))
    csv_write(out/'POLICY_CHANGE.csv',summary);csv_write(out/'ADVANTAGES.csv',advantages)
    csv_write(out/'UPDATE_LOG.csv',update_rows);csv_write(out/'SHARED_UPDATE0_CHOICES.csv',identity_checks)
    # These are descriptive differences, not a new checkpoint selection or seed CI.
    validation=[]
    for job in (r for r in plan['entries'] if r['development_role']=='validation'):
        name=job['instance_id'];ref=read(root/'terminal/validation/u000'/name/'RUN_COMPLETE.json')
        for u in (4,8):
            r=read(root/'terminal/validation'/f'u{u:03d}'/name/'RUN_COMPLETE.json')
            if not r['completed'] or (r['seed'],r['scenario_digest'])!=(ref['seed'],ref['scenario_digest']):
                raise ValueError('Broken validation pairing')
            validation.append(dict(instance=name,configuration=job['configuration'],seed=r['seed'],update=u,
                update0_hv=ref['final_hypervolume'],checkpoint_hv=r['final_hypervolume'],
                paired_absolute_hv_difference=r['final_hypervolume']-ref['final_hypervolume']))
    csv_write(out/'VALIDATION_PAIRED.csv',validation)
    result=dict(completed=True,selected_update=selected['update'],shared_update0_actor_tensors_exact=actor_exact,
        completed_screen_pairs_checked=len(pairs),all_checked_screen_pairs_exact=all(r['all_checked_exact'] for r in pairs) if pairs else None,
        shared_update0_replayed_choice_differences=sum(r['shared_vs_u0_top1_differences'] for r in identity_checks),
        policy_change=summary,training_updates=update_rows,new_searches=0,new_objective_evaluations=0,new_optimizer_steps=0,
        qualification=['Replay uses saved TRAINING candidates, not unseen validation/development trajectories.',
            'Fixed reward-independent sampling; summary weights trajectories equally within each component. KL is KL(u0 || checkpoint), in nats.',
            'No optimizer-step entropy/clip fraction was logged. End-checkpoint replay cannot reconstruct their history.',
            'Large policy change does not alone establish wrong rewards; small change does not alone establish clipping.',
            'One validation seed per instance does not estimate seed variability or prove update 0 is truly optimal.',
            'Return is normalized terminal HV, advantage is R minus recorded old critic value; no paired reference subtraction or advantage standardization.',
            'Identity checks exclude wall-clock time. Matching stored records is evidence on checked runs, not a proof on every possible future state.',
            'No pending jobs skipped, cloned, stopped or marked complete. Unexecuted rows must never be reported as measured runs.'])
    write(out/'QUICK_RESULT.json',result);write(out/'INPUT_HASHES.json',manifest)
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=EXPANDA/'transformer_ppo_runs/diagnostics/ccp_icm_multiscale_joint_v1')
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--max-pools',type=int,default=128,help='per component per training trajectory; 0 means all')
    a=parser.parse_args()
    if a.max_pools<0:parser.error('--max-pools must be nonnegative')
    execute(a.source.resolve(),a.out.resolve(),a.max_pools)


if __name__=='__main__':main()
