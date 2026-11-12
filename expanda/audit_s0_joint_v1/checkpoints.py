#!/usr/bin/env python3
"""Read completed PPO updates and paired validation returns without changing jobs."""
from pathlib import Path
from collections import defaultdict
import argparse
import csv
import io
import json
import math
import statistics
import time


def read(path):
    # A running parent may still be replacing a small JSON progress record.
    for attempt in range(3):
        try:return json.loads(path.read_text(encoding='utf-8'))
        except json.JSONDecodeError:
            if attempt==2:raise
            time.sleep(.2)


def inspect(root,kind):
    if not (root/'PLAN.json').exists():return {'source':str(root),'available':False}
    plan=read(root/'PLAN.json');old=kind=='s0'
    entries=plan.get('entries',[])
    names=plan['validation_instances'] if old else [r['instance_id'] for r in entries if r['development_role']=='validation']
    cfg={n:'S0' for n in names} if old else {r['instance_id']:r['configuration'] for r in entries if r['development_role']=='validation'}
    updates=[];validation=[];complete={};terminal=root/'terminal'
    for u in range(plan['settings']['ppo']['updates']+1):
        checkpoint=terminal/'checkpoints'/f'update_{u:03d}'
        if not (checkpoint/'BUNDLE.json').exists():continue
        metric=checkpoint/'UPDATE.json'
        if u and metric.exists():
            m=read(metric);values=m['terminal_returns']
            updates.append({'update':u,'episodes':m['complete_episodes'],'optimizer_steps':m['optimizer_steps'],
                'kl_early_stop':m['kl_early_stop'],'last_minibatch_kl':m.get('last_minibatch_kl',m.get('approximate_kl')),
                'training_terminal_mean':statistics.fmean(values),'training_terminal_min':min(values),'training_terminal_max':max(values),
                'qualification':'Different training instances/seeds across updates; these means are NOT a paired learning curve.'})
        if u not in plan['settings']['ppo']['validation_checkpoints']:continue
        runs={}
        for n in names:
            folder=terminal/'validation'/(f'u{u:03d}_{n}' if old else f'u{u:03d}/{n}')
            p=folder/'RUN_COMPLETE.json'
            if not p.exists():continue
            r=read(p)
            if not r['completed'] or r['instance']!=n or r['stochastic']:raise ValueError('Invalid validation record: '+str(p))
            hv=float(r['final_hypervolume'])
            if not math.isfinite(hv):raise ValueError('Nonfinite validation HV')
            runs[n]=r
        row={'update':u,'completed_validation_instances':len(runs),'expected_validation_instances':len(names),
             'configuration_equal_mean_delta_vs_update0':None,'positive_instances':None,'ties':None,'negative_instances':None}
        if set(runs)==set(names):
            complete[u]=runs
            if 0 not in complete:raise ValueError('Completed checkpoint validation without completed update0')
            by=defaultdict(list);deltas=[]
            for n,r in runs.items():
                b=complete[0][n];seed='algorithm_seed' if old else 'seed'
                if (r[seed],r['scenario_digest'])!=(b[seed],b['scenario_digest']):raise ValueError('Validation pairing changed: '+n)
                delta=r['final_hypervolume']-b['final_hypervolume'];deltas.append(delta);by[cfg[n]].append(delta)
            row.update(configuration_equal_mean_delta_vs_update0=statistics.fmean(statistics.fmean(v) for v in by.values()),
                positive_instances=sum(v>1e-12 for v in deltas),ties=sum(abs(v)<=1e-12 for v in deltas),negative_instances=sum(v< -1e-12 for v in deltas))
        validation.append(row)
    selected=read(terminal/'SELECTED.json') if (terminal/'SELECTED.json').exists() else None
    if selected:
        eligible=[r for r in validation if r['configuration_equal_mean_delta_vs_update0'] is not None]
        best=max(eligible,key=lambda r:(r['configuration_equal_mean_delta_vs_update0'],-r['update']))
        chosen=next((r for r in eligible if r['update']==selected['update']),None)
        if chosen is None or best['configuration_equal_mean_delta_vs_update0']-chosen['configuration_equal_mean_delta_vs_update0']>1e-12:
            raise ValueError('Stored checkpoint selection disagrees with paired validation')
    return {'source':str(root),'available':True,'training':updates,'validation':validation,'selected':selected,
            'qualification':'Frozen training normalization; complete validation panels only. No screen results used for selection. No searches or training started.'}


def print_table(title,rows):
    print('\n'+title,flush=True)
    if not rows:print('尚无已完成记录',flush=True);return
    stream=io.StringIO();w=csv.DictWriter(stream,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows);print(stream.getvalue(),flush=True)


def review(diag,multi_root):
    results={}
    for title,root,kind in [('S0 B400 已完成训练',diag/'ccp_s0_joint_b400_v1','s0'),('S0–S8 完整预算正在进行',multi_root,'multi')]:
        r=inspect(root,kind);results[kind]=r
        print_table(title+'：训练回报（不同实例，不作配对趋势）',r.get('training',[]))
        print_table(title+'：同实例配对验证终局HV',r.get('validation',[]))
        if r.get('selected'):print('选中 checkpoint：'+str(r['selected']['update']),flush=True)
    return results


def main():
    diag=Path(__file__).resolve().parent.parent/'transformer_ppo_runs/diagnostics'
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--multi-root',type=Path,default=diag/'ccp_icm_multiscale_joint_v1');a=p.parse_args()
    review(diag,a.multi_root.resolve())


if __name__=='__main__':main()
