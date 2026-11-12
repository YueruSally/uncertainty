#!/usr/bin/env python3
"""Read existing same-pool validation results; no training, inference or searches."""
from pathlib import Path
from collections import defaultdict
import argparse
import csv
import hashlib
import io
import json
import math
import random
import statistics

COMPONENTS=('I','C','M')


def summarize(rows,expected,bootstrap=20000):
    grouped=defaultdict(list);seen=set()
    for r in rows:
        if r['pool_source']!='validation':continue
        c=r['component'];name=r['instance'];key=(c,name)
        if c not in COMPONENTS or key in seen:raise ValueError('Wrong or duplicate validation identity: '+str(key))
        seen.add(key);n=int(r['candidate_pools']);switch=int(r['different_choices'])
        value=r['recorded_one_step_mean_hv_difference']
        if value is None or value=='':raise ValueError('Missing known-label gain for '+str(key))
        gain=float(value)
        if n<=0 or not 0<=switch<=n or not math.isfinite(gain):raise ValueError('Invalid count or gain: '+str(key))
        if switch==0 and gain!=0.:raise ValueError('Nonzero gain without changed choices: '+str(key))
        grouped[c].append({'component':c,'instance':name,'states':n,'changes':switch,'mean':gain,'sum':n*gain})
    if set(grouped)!=set(COMPONENTS):raise ValueError('Expected all three components')
    if any(len(grouped[c])!=expected[c] for c in COMPONENTS):raise ValueError('Incomplete validation instance coverage')
    names=[{r['instance'] for r in grouped[c]} for c in COMPONENTS]
    if not names[0]==names[1]==names[2]:raise ValueError('Components have different validation instance identities')
    summary=[];detail=[]
    for j,c in enumerate(COMPONENTS):
        part=sorted(grouped[c],key=lambda r:r['instance']);x=[r['mean'] for r in part]
        n=sum(r['states'] for r in part);changes=sum(r['changes'] for r in part);net=math.fsum(r['sum'] for r in part)
        rng=random.Random(997201+j)
        draws=sorted(statistics.fmean(rng.choices(x,k=len(x))) for _ in range(bootstrap))
        def quantile(p):
            index=(len(draws)-1)*p;lo=int(index);hi=min(lo+1,len(draws)-1)
            return draws[lo]+(draws[hi]-draws[lo])*(index-lo)
        summary.append({'组件':c,'验证实例数':len(part),'候选状态数':n,'改选次数':changes,'改选比例':changes/n,
            '实例等权平均单步HV差':statistics.fmean(x),'实例重采样95区间下限':quantile(.025),'实例重采样95区间上限':quantile(.975),
            '状态等权平均单步HV差':net/n,'仅改选状态的平均单步HV差':net/changes if changes else None,
            '实例均值为正':sum(v>0 for v in x),'实例均值为零':sum(v==0 for v in x),'实例均值为负':sum(v<0 for v in x)})
        for r in sorted(part,key=lambda r:(r['mean'],r['instance'])):
            detail.append({'组件':c,'验证实例':r['instance'],'候选状态数':r['states'],'改选次数':r['changes'],
                '全部状态平均单步HV差':r['mean'],'仅改选状态平均单步HV差':r['sum']/r['changes'] if r['changes'] else None})
    return summary,detail


def analyze(source):
    quick=json.loads((source/'QUICK_RESULT.json').read_text(encoding='utf-8'))
    if not quick['completed']:raise ValueError('The previous same-pool audit is incomplete')
    expected=quick['choice_replay']['validation_pool_instances_per_component']
    if any(expected.get(c)!=20 for c in COMPONENTS):raise ValueError('Expected 20 stored S0 validation instances for each component')
    path=source/'SAME_POOL_PER_TRAJECTORY.csv';content=path.read_bytes()
    rows=list(csv.DictReader(io.StringIO(content.decode('utf-8-sig'))))
    for c in COMPONENTS:
        recorded=[r for r in quick['choice_replay']['same_pool_summary'] if r['候选来源']=='validation' and r['组件']==c]
        own=[r for r in rows if r['pool_source']=='validation' and r['component']==c]
        if len(recorded)!=1 or (sum(int(r['candidate_pools']) for r in own),sum(int(r['different_choices']) for r in own))!=(recorded[0]['候选池数'],recorded[0]['不同选择次数']):
            raise ValueError('Candidate coverage differs from completed audit: '+c)
    summary,detail=summarize(rows,expected)
    return summary,detail,{'source':str(path),'source_sha256':hashlib.sha256(content).hexdigest(),
        'difference':'PPO-selected recorded feasible HV contribution minus shared-supervised-selected contribution on exactly the same candidate pool',
        'units':'Raw one-step HV difference under original label normalization; NOT percent and NOT final-search HV',
        'primary_aggregation':'Mean within each instance, then equal instance weights; components reported separately',
        'changed_states_mean':'sum(all-state mean times state count) / different-choice count; unchanged choices have exactly zero difference',
        'interval':'20,000 bootstrap draws of whole validation instances; within-state variation cannot be recovered from the aggregate CSV',
        'sign_counts':'Signs of instance means, not counts of beneficial/harmful candidate changes; tiny nonzero differences retained',
        'qualification':'These validation instances already guided model selection. This is diagnostic, not independent validation, not a gate for new experiments, and not evidence of a causal final-search effect.',
        'new_searches':0,'new_objective_evaluations':0,'new_training_steps':0,'new_model_inference_calls':0}


def print_csv(title,rows):
    buffer=io.StringIO(newline='');writer=csv.DictWriter(buffer,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print('\n'+title+'\n'+buffer.getvalue(),flush=True)


def main():
    default=Path(__file__).resolve().parent.parent/'transformer_ppo_runs/diagnostics/ccp_s0_joint_b400_v1_paired_audit_v1'
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,default=default);a=p.parse_args()
    summary,detail,metadata=analyze(a.source.resolve())
    print_csv('单步收益汇总：PPO − 共享监督，HV绝对差',summary)
    print_csv('各验证实例：按单步差值从低到高排列',detail)
    print(json.dumps(metadata,ensure_ascii=False,indent=2),flush=True)
    print('[S0-STEP-GAIN] COMPLETE; existing CSV only; source unchanged',flush=True)


if __name__=='__main__':main()
