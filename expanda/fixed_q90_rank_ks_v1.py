#!/usr/bin/env python3
"""Read all saved 6.2 plans: 20 q90 rank comparisons and first-replicate two-sample KS.

No optimization, new draws of transport scenarios, objective evaluation or training.
KS label permutations are statistical resampling of saved samples only.
"""
import argparse
import csv
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

sys.dont_write_bytecode=True
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name]='1'
EXP=Path(__file__).resolve().parent
LEGACY=EXP/'fixed_q90_s0_s11_v1'
sys.path.insert(0,str(LEGACY))
from common import CONFIGS,OBJECTIVES,DIAG,canonical,digest,read,write,freeze,csv_write,checked_plan,verify
spec=importlib.util.spec_from_file_location('saved_q90_statistics',LEGACY/'report.py')
legacy=importlib.util.module_from_spec(spec);spec.loader.exec_module(legacy)
import numpy as np

VERSION='fixed-q90-all-plan-rank-ks-v1'
PERMUTATIONS=999
ALPHA=.05
LABELS=dict(cost='成本',emission='排放',makespan='完成时间')


def rank_rows(reference,trials,scope):
    if trials.shape!=(20,*reference.shape) or reference.ndim!=2 or reference.shape[1]!=3:
        raise ValueError('Expected 20 replicates of an N by 3 quantile matrix')
    summary=[];detail=[]
    for j,obj in enumerate(OBJECTIVES):
        values=[]
        for k in range(20):
            rho=legacy.spearman(trials[k,:,j],reference[:,j])
            detail.append(dict(configuration=scope,objective=obj,replicate=k+1,plans=len(reference),spearman=rho))
            if rho is not None: values.append(rho)
        summary.append(dict(configuration=scope,objective=obj,plans=len(reference),replicates=20,
            defined_replicates=len(values),mean_spearman=float(np.mean(values)) if values else None,
            min_spearman=min(values) if values else None))
    return summary,detail


def ks_summary(rows,scope):
    summary=[]
    for obj in OBJECTIVES:
        group=[r for r in rows if r['objective']==obj]
        if not group: raise ValueError('Missing KS comparisons: '+scope+'/'+obj)
        not_rejected=sum(r['p_value']>ALPHA for r in group)
        summary.append(dict(configuration=scope,objective=obj,plans=len(group),
            p_gt_0_05_count=not_rejected,p_le_0_05_count=len(group)-not_rejected,
            p_gt_0_05_fraction=not_rejected/len(group),p_gt_0_05_percent=100*not_rejected/len(group),
            plans_with_pooled_sample_ties=sum(r['pooled_distinct_values']<1100 for r in group),
            alpha=ALPHA,permutations=PERMUTATIONS))
    return summary


def seed_for(cfg,plan_id,ordinal,objective):
    # The pre-existing representative-plan check used these seeds for decision 0.
    if ordinal==0: return 290000000+100*CONFIGS.index(cfg)+objective
    return int.from_bytes(hashlib.sha256(f'{VERSION}/{cfg}/{plan_id}/{objective}'.encode()).digest()[:8],'big')


def analyze_ks(cfg,instance,ids,small,big,heartbeat):
    if small.shape!=(len(ids),3,100) or big.shape!=(len(ids),3,1000):
        raise ValueError('Expected all plans with three objectives and 100/1000 samples')
    rows=[];last=0.
    for i,key in enumerate(ids):
        for j,obj in enumerate(OBJECTIVES):
            x,y=small[i,j],big[i,j]
            seed=seed_for(cfg,key,i,j)
            statistic,pvalue=legacy.ks_permutation(x,y,seed,permutations=PERMUTATIONS)
            rows.append(dict(configuration=cfg,instance=instance,exact_decision_id=key,objective=obj,
                sample100_replicate=1,n100=100,n1000=1000,ks_statistic=statistic,p_value=pvalue,
                p_gt_0_05=bool(pvalue>ALPHA),permutations=PERMUTATIONS,permutation_seed=seed,
                pooled_distinct_values=int(len(np.unique(np.r_[x,y])))))
        now=time.monotonic()
        if i==0 or i+1==len(ids) or now-last>=20:
            heartbeat(stage='ks_saved_samples',configuration=cfg,plans_done=i+1,plans_total=len(ids))
            last=now
    return rows


def run(args):
    source,out=args.source.resolve(),args.out.resolve()
    if source==out or source in out.parents or out in source.parents:
        raise ValueError('Use a separate output folder, outside the original 6.2 results')
    if (out/'PLAN.json').exists() and read(out/'PLAN.json').get('protocol')!=VERSION:
        raise ValueError('Output belongs to another experiment')
    out.mkdir(parents=True,exist_ok=True)
    with (out/'.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if hasattr(os,'nice'):
            try:
                os.nice(max(0,10-os.nice(0)))
            except OSError as exc:
                print('[Q90-RANK-KS] Cannot lower scheduling priority:',exc,flush=True)
        started=time.monotonic();files={}
        def pin(path,expected=None):
            path=Path(path).resolve();h=digest(path)
            if expected is not None and h!=expected: raise ValueError('Saved input changed: '+str(path))
            files[str(path)]=h
            return h
        def heartbeat(**kwargs):
            status=dict(completed=False,pid=os.getpid(),elapsed_seconds=round(time.monotonic()-started,1),
                        workers=1,threads=1,new_searches=0,new_objective_evaluations=0,
                        new_transport_realizations=0,training_updates=0,**kwargs)
            write(out/'STATUS.json',status);print('[Q90-RANK-KS]',json.dumps(status,ensure_ascii=False),flush=True)
        try:
            heartbeat(stage='checking_sources')
            pin(source/'STATUS.json');status=read(source/'STATUS.json')
            if not status.get('completed') or set(status['completed_configurations'])!=set(CONFIGS):
                raise ValueError('All twelve original 6.2 configurations must be complete')
            pin(source/'PLAN.json');original=checked_plan(source/'PLAN.json')
            if original['protocol']['version']!='fixed-q90-s0-s11-v1': raise ValueError('Unexpected source protocol')
            pin(Path(__file__))
            for path in (LEGACY/'report.py',LEGACY/'common.py'):
                expected=[h for n,h in original['files'].items() if n.endswith('/fixed_q90_s0_s11_v1/'+path.name)]
                if len(expected)!=1: raise ValueError('Missing original statistical implementation hash')
                pin(path,expected[0])
            protocol=dict(protocol=VERSION,source=str(source),source_plan_sha256=original['sha256'],
                source_status_sha256=files[str(source/'STATUS.json')],configurations=CONFIGS,replicates=20,
                ks='two-sided two-sample KS statistic; 999 random label permutations; p=(exceedances+1)/1000',
                ks_replicate=1,ks_alpha=ALPHA,code_sha256=digest(Path(__file__)),
                rank='P90 versus P90; average ranks for ties; ALL pools all configuration-plan identities, unweighted by configuration',
                new_searches=0,new_objective_evaluations=0,new_transport_realizations=0,training_updates=0)
            freeze(out/'PLAN.json',protocol)
            all_reference=[];all_trials=[];ranks=[];rank_detail=[];all_ks=[];percfg_ks=[];identity_rows=[]
            cache=out/'cache';cache.mkdir(exist_ok=True)
            for cfg in CONFIGS:
                folder=source/cfg
                pin(folder/'INPUTS.json');inputs=checked_plan(folder/'INPUTS.json')
                pin(folder/'RESULT.json');done=read(folder/'RESULT.json')
                ids=[r['exact_decision_id'] for r in inputs['decisions']]
                if (inputs['configuration']!=cfg or len(ids)!=len(set(ids)) or len(ids)!=inputs['unique_plans']
                    or not done['completed'] or done['inputs_sha256']!=inputs['sha256'] or done['unique_plans']!=len(ids)):
                    raise ValueError('Plan identities or original completion differ: '+cfg)
                seeds=inputs['new_seeds']
                if len(seeds)!=21 or len(set(seeds))!=21 or inputs['instance']['ccp100_seed'] in seeds:
                    raise ValueError('Expected independent reference and twenty fresh sets')
                packs={};pack_hashes={}
                for k in range(21):
                    tag='reference_1000' if k==0 else f'replicate_{k:02d}_100'
                    count=1000 if k==0 else 100
                    mp=folder/'packs'/(tag+'.json');path=folder/'packs'/(tag+'.npz')
                    pin(mp);meta=read(mp)
                    expected=dict(inputs_sha256=inputs['sha256'],audit_plan_sha256=original['sha256'],
                                  tag=tag,realizations=count,scenario_seed=seeds[k])
                    if not meta['completed'] or meta['identity']!=expected or meta['plans']!=len(ids):
                        raise ValueError('Pack metadata does not match fixed plans: '+str(mp))
                    pack_hashes[tag]=pin(path,meta['data_sha256'])
                    with np.load(path,allow_pickle=False) as f:
                        q=f['quantiles'].copy()
                        if q.shape!=(len(ids),3) or not np.isfinite(q).all(): raise ValueError('Invalid saved P90')
                        samples=f['samples'].copy() if k in (0,1) else None
                    if samples is not None:
                        if samples.shape!=(len(ids),3,count) or not np.isfinite(samples).all():
                            raise ValueError('Invalid saved realization array')
                        calculated=np.partition(samples,int(np.ceil(.9*count))-1,axis=-1)[...,int(np.ceil(.9*count))-1]
                        if not np.allclose(calculated,q,rtol=1e-10,atol=1e-8):
                            raise ValueError('Stored P90 differs from original ceil(0.9*N) order statistic')
                    packs[tag]=(q,samples)
                reference,big=packs['reference_1000']
                trials=np.stack([packs[f'replicate_{k:02d}_100'][0] for k in range(1,21)])
                small=packs['replicate_01_100'][1]
                rs,rd=rank_rows(reference,trials,cfg);ranks.extend(rs);rank_detail.extend(rd)
                # Reproduce the existing per-configuration correlation report if available.
                prior=folder/'SUMMARY.csv'
                if prior.exists():
                    pin(prior)
                    with prior.open(encoding='utf-8-sig',newline='') as f: old_summary={r['objective']:r for r in csv.DictReader(f)}
                    for r in rs:
                        for newkey,oldkey in [('mean_spearman','fresh100_mean_spearman'),('min_spearman','fresh100_min_spearman')]:
                            v=old_summary[r['objective']][oldkey]
                            if (r[newkey] is None)!=(v=='') or (v!='' and abs(r[newkey]-float(v))>1e-12):
                                raise ValueError('Existing per-configuration correlation not reproduced')
                cache_key=canonical(dict(inputs=inputs['sha256'],reference=pack_hashes['reference_1000'],
                    small=pack_hashes['replicate_01_100'],protocol=protocol,stats_sha=digest(LEGACY/'report.py')))
                cached=cache/(cfg+'.json')
                if cached.exists():
                    stored=read(cached)
                    if stored['key']!=cache_key: raise ValueError('KS cache belongs to different inputs')
                    ks=stored['rows']
                    expected={(key,obj) for key in ids for obj in OBJECTIVES}
                    if len(ks)!=len(expected) or {(r['exact_decision_id'],r['objective']) for r in ks}!=expected:
                        raise ValueError('Incomplete cached per-plan KS results')
                else:
                    ks=analyze_ks(cfg,inputs['instance']['instance_id'],ids,small,big,heartbeat)
                    write(cached,dict(key=cache_key,rows=ks))
                repfile=folder/'REPRESENTATIVE_KS.csv'
                if repfile.exists():
                    pin(repfile)
                    with repfile.open(encoding='utf-8-sig',newline='') as f: previous=list(csv.DictReader(f))
                    for r in previous:
                        one=next(x for x in ks if x['exact_decision_id']==r['representative_decision_id'] and x['objective']==r['objective'])
                        if abs(one['p_value']-float(r['permutation_p']))>1e-12 or abs(one['ks_statistic']-float(r['ks_statistic']))>1e-12:
                            raise ValueError('Original representative KS not reproduced')
                all_ks.extend(ks);percfg_ks.extend(ks_summary(ks,cfg))
                all_reference.append(reference);all_trials.append(trials)
                identity_rows.extend(dict(configuration=cfg,instance=inputs['instance']['instance_id'],exact_decision_id=key,
                                          reference1000_seed=seeds[0],fresh100_set1_seed=seeds[1]) for key in ids)
                csv_write(out/'SPEARMAN_BY_CONFIGURATION.csv',ranks)
                csv_write(out/'KS_BY_CONFIGURATION.csv',percfg_ks)
                heartbeat(stage='configuration_complete',configuration=cfg,completed_configurations=len(all_reference),total_configurations=12)
            n=sum(len(x) for x in all_reference)
            if n!=status['unique_plans'] or len(all_ks)!=3*n: raise ValueError('Missing fixed plans')
            pooled,pdetail=rank_rows(np.concatenate(all_reference),np.concatenate(all_trials,axis=1),'ALL')
            ksum=ks_summary(all_ks,'ALL')
            eq=[]
            for obj in OBJECTIVES:
                r=[v for v in ranks if v['objective']==obj];valid=[v['mean_spearman'] for v in r if v['mean_spearman'] is not None]
                kval=[v for v in percfg_ks if v['objective']==obj]
                eq.append(dict(objective=obj,configurations=12,defined_configurations=len(valid),
                    configuration_equal_mean_spearman=float(np.mean(valid)) if valid else None,
                    configuration_equal_p_gt_0_05_percent=float(np.mean([v['p_gt_0_05_percent'] for v in kval]))))
            csv_write(out/'SPEARMAN_ALL_PLANS.csv',pooled)
            csv_write(out/'SPEARMAN_REPLICATES.csv',pdetail+rank_detail)
            csv_write(out/'KS_ALL_PLANS.csv',ksum)
            csv_write(out/'KS_PER_PLAN.csv',all_ks)
            csv_write(out/'CONFIGURATION_EQUAL_SUPPLEMENT.csv',eq)
            csv_write(out/'PLAN_IDENTITIES.csv',identity_rows)
            fmt=lambda v:'未定义' if v is None else f'{v:.6f}'
            lines=['全部固定方案：P90秩相关及两样本K–S检验',f'12个配置，共{n}个配置内去重的固定方案；不筛选方案。',
                   '总体Spearman：每组先合并12配置的全部方案计算相关，再对20组报告平均/最小。',
                   '目标 | 总体平均ρ | 总体最低ρ | 配置等权平均ρ | K–S p>0.05方案数/总数 | 比例']
            for r,k,e in zip(pooled,ksum,eq):
                lines.append(f"{LABELS[r['objective']]} | {fmt(r['mean_spearman'])} | {fmt(r['min_spearman'])} | {fmt(e['configuration_equal_mean_spearman'])} | {k['p_gt_0_05_count']}/{n} | {k['p_gt_0_05_percent']:.3f}%")
            lines+=['','各配置内20组P90秩相关：平均值 / 最小值','配置 | 方案数 | 成本 | 排放 | 完成时间']
            for cfg in CONFIGS:
                rr=[r for r in ranks if r['configuration']==cfg]
                lines.append(f"{cfg} | {rr[0]['plans']} | "+' | '.join(f"{fmt(r['mean_spearman'])} / {fmt(r['min_spearman'])}" for r in rr))
            lines+=['','K–S使用第一组新100次和独立1000次的原始实现目标值，逐方案、逐目标检验；不是对两个P90数做检验。',
                    '双侧K–S统计量，999次标签置换，p=(至少同样极端的置换次数+1)/1000；保留重复值。',
                    '报告严格p>0.05比例，不做多重检验校正。它表示未检出分布差异，不证明等价或100次充分。',
                    '总体Spearman可能受配置间规模差异影响；应结合配置内结果。相关系数不等于Pareto支配关系保持率。',
                    '1000次是独立有限样本参考，不是真值；方案共用情景，检验结果之间有相关性。',
                    '源文件未修改；无新增搜索、方案评价、交通情景抽样或训练。']
            (out/'REPORT.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
            verify(files);write(out/'INPUT_HASHES.json',files)
            write(out/'STATUS.json',dict(completed=True,stage='all',configurations=12,fixed_plans=n,ks_tests=3*n,
                spearman_comparisons=13*3*20,elapsed_seconds=round(time.monotonic()-started,1),
                new_searches=0,new_objective_evaluations=0,new_transport_realizations=0,training_updates=0))
            print('\n'+'\n'.join(lines),flush=True);print('[Q90-RANK-KS] COMPLETE',out,flush=True)
        except Exception as exc:
            write(out/'STATUS.json',dict(completed=False,failed=True,error=str(exc),new_searches=0,new_objective_evaluations=0));raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=DIAG/'ccp_fixed_q90_s0_s11_v1')
    parser.add_argument('--out',type=Path,default=DIAG/'ccp_fixed_q90_rank_ks_v1')
    run(parser.parse_args())
