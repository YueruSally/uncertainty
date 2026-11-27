"""Complete P90 reports for fixed final fronts, with explicit reference methods."""
from collections import defaultdict
from statistics import mean, median, stdev
from addon_common import *

LABELS = {'BASE':'Adapted NSGA-II',METHOD:'NSGA-II (standard framework)',
          'I_RULE':'I (Directed)','I_LEARN':'I Transformer (SL)',
          'C_RULE':'C (Simple)','C_LEARN':'C Transformer (SL)',
          'M_RULE':'M (Balanced)','M_LEARN':'M Transformer (SL)',
          'ICM_HEURISTIC':'ICM (rules)','ICM_INDEPENDENT':'ICM Transformer (independent SL)',
          'ICM_SHARED_SUPERVISED':'ICM Transformer (shared SL)',
          'ICM_TERMINAL_PPO':'ICM Transformer (shared SL + PPO)',
          'ICM_RANDOM_RATE_CONTROLLED':'ICM (random replacement)',
          'ICM_SHARED_ZERO_THRESHOLD':'ICM Transformer (shared SL, threshold = 0)'}
TOLERANCES={'original':1e-12/1.331,'raw_0.01':.01/1.331,'reported_0.01':.01}
SHARED='ICM_SHARED_SUPERVISED'

def relative(a,b):return 100*(a-b)/b if b>0 else None
def equal_mean(rows,key):
    if not rows or any(r[key] is None for r in rows):return None
    groups=defaultdict(lambda:defaultdict(list))
    for r in rows:groups[r['configuration']][r['pair_instance']].append(r[key])
    return mean(mean(mean(v) for v in instances.values()) for instances in groups.values())
def wtl(delta,tolerance):return 'W' if delta>tolerance else 'L' if delta< -tolerance else 'T'
def bootstrap(np,rows,key,seed):
    """Fixed configurations; paired instance and seed resampling; no missing-pair dropping."""
    if not rows or any(r[key] is None for r in rows):return [None,None]
    rng=np.random.default_rng(seed);est=np.zeros(50000)
    configs=sorted({r['configuration'] for r in rows})
    for cfg in configs:
        part=[r for r in rows if r['configuration']==cfg]
        names=sorted({r['pair_instance'] for r in part})
        cells=[[r[key] for r in sorted(part,key=lambda r:r['seed']) if r['pair_instance']==name] for name in names]
        require(len({len(x) for x in cells})==1,'Unequal seed counts in bootstrap')
        values=np.asarray(cells);n,k=values.shape
        ii=rng.integers(0,n,size=(50000,n));ss=rng.integers(0,k,size=(50000,n,k))
        est+=values[ii[:,:,None],ss].mean(axis=(1,2))/len(configs)
    return np.quantile(est,[.025,.975]).tolist()

def weighted_median(np,values,weights):
    order=np.argsort(values,kind='stable');v=np.asarray(values)[order];w=np.asarray(weights)[order]
    cumulative=np.cumsum(w)/np.sum(w);i=int(np.searchsorted(cumulative,.5))
    return float((v[i]+v[i+1])/2) if abs(cumulative[i]-.5)<=1e-12 and i+1<len(v) else float(v[i])

def csv_read(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def key(r):return tuple(r[k] for k in ('table','pair_instance','network','method','seed','scenario_group'))
def summary(np,rows,seed):
    same=all(r['method']==r['reference_method'] and r['network']==r['reference_network'] for r in rows)
    ci=bootstrap(np,rows,'relative_percent',seed) if not same else [None,None]
    times=[r['search_seconds'] for r in rows]
    result=dict(runs=len(rows),instances=len({r['pair_instance'] for r in rows}),
        mean_hv=equal_mean(rows,'hv'),reference_mean_hv=equal_mean(rows,'reference_hv'),
        mean_relative_percent=equal_mean(rows,'relative_percent'),ci95_low=ci[0],ci95_high=ci[1],
        mean_hv_difference=equal_mean(rows,'hv_difference'),
        undefined_relative_pairs=sum(r['relative_percent'] is None for r in rows),
        runtime_mean_s=mean(times),runtime_sd_s=stdev(times) if len(times)>1 else None)
    for name in TOLERANCES:
        result['WTL_'+name]='/'.join(str(sum(r['wtl_'+name]==v for r in rows)) for v in 'WTL') if not same else '—'
    return result

def build_summaries(root,hvrows,np):
    index={key(r):r for r in hvrows};require(len(index)==len(hvrows),'Duplicate HV identity')
    pairs=[]
    for r in hvrows:
        if r['table']=='Table7':refs=[('small',SHARED)] if r['network']=='large' else [('small','BASE')]
        else:
            refs=[('expanded','BASE')]
            if (r['table'],r['pair_instance'],'expanded',METHOD,r['seed'],r['scenario_group']) in index:
                refs.append(('expanded',METHOD))
        for network,ref in refs:
            b=index[r['table'],r['pair_instance'],network,ref,r['seed'],r['scenario_group']]
            delta=r['hv']-b['hv']
            pairs.append(dict(**r,reference_network=network,reference_method=ref,reference_label=LABELS[ref],
                reference_hv=b['hv'],relative_percent=relative(r['hv'],b['hv']),hv_difference=delta,
                **{'wtl_'+name:wtl(delta,tol) for name,tol in TOLERANCES.items()}))
    summary_groups=defaultdict(list)
    for r in pairs:
        sk=tuple(r[k] for k in ('table','network','method','reference_network','reference_method','scenario_group'))
        summary_groups[sk].append(r)
    summaries=[]
    for i,(sk,rows) in enumerate(sorted(summary_groups.items())):
        if i%20==0:status(root,'summarising_HV',groups_done=i,groups_total=len(summary_groups),bootstrap_replicates=50000)
        for cfg in ['ALL']+sorted({r['configuration'] for r in rows},key=lambda x:int(x[1:]) if x.startswith('S') and x[1:].isdigit() else 99):
            part=rows if cfg=='ALL' else [r for r in rows if r['configuration']==cfg]
            summaries.append(dict(zip(('table','network','method','reference_network','reference_method','scenario_group'),sk),
                configuration=cfg,method_label=LABELS[sk[2]],reference_label=LABELS[sk[4]],
                evaluation_scenarios=int(sk[-1].rsplit('_',1)[1]),alpha=.9,
                **summary(np,part,490000000+len(summaries))))
    return pairs,summaries,index

def fmt(value,digits=2):return '—' if value is None else f'{value:.{digits}f}'
def write_display(dest,summaries):
    lines=['α=0.9；以下100/1000/5000均为独立复评场景数；每次搜索只使用同一最终解集。',
           'HV已除以1.331；时间为原完整搜索秒数，不是复评时间。BASE在新表中标为Adapted NSGA-II。',
           '95%区间：固定配置，配置内重抽实例、实例内重抽配对种子，50000次。',
           '原WTL容差=1e-12/1.331；另列原HV容差0.01及已除体积HV容差0.01，均不是相对提升1%。','']
    for table,network,ref in sorted({(r['table'],r['network'],r['reference_method']) for r in summaries}):
        lines += [f'{table} | {network} | Reference: {LABELS[ref]}',
            '| Method | Scenarios | Mean HV | Change (%) | 95% CI (%) | Search seconds | W/T/L original | W/T/L raw .01 | W/T/L reported .01 |',
            '| --- | --- | --- | --- | --- | --- | --- | --- | --- |']
        for r in summaries:
            if (r['table'],r['network'],r['reference_method'],r['configuration'])!=(table,network,ref,'ALL') or not r['scenario_group'].startswith('fresh_'):continue
            ci='—' if r['ci95_low'] is None else f"[{r['ci95_low']:+.2f}, {r['ci95_high']:+.2f}]"
            lines.append(f"| {r['method_label']} | {r['evaluation_scenarios']} | {r['mean_hv']:.6f} | {fmt(r['mean_relative_percent'])} | {ci} | {fmt(r['runtime_mean_s'])} ± {fmt(r['runtime_sd_s'])} | {r['WTL_original']} | {r['WTL_raw_0.01']} | {r['WTL_reported_0.01']} |")
        lines.append('')
    (dest/'REPORT.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    wide=[]
    groups=defaultdict(dict)
    for r in summaries:
        if r['configuration']=='ALL' and r['scenario_group'].startswith('fresh_'):
            groups[(r['table'],r['network'],r['method'],r['reference_method'])][r['evaluation_scenarios']]=r
    for sk,rs in sorted(groups.items()):
        require(set(rs)=={100,1000,5000},'Missing evaluation size')
        r=rs[5000];row=dict(table=sk[0],network=sk[1],method=r['method_label'],reference=r['reference_label'],runs=r['runs'])
        for n in (100,1000,5000):
            for col in ('mean_hv','mean_relative_percent','ci95_low','ci95_high','WTL_original','WTL_raw_0.01','WTL_reported_0.01'):
                row[f'{n}_{col}']=rs[n][col]
        row.update(search_runtime_mean_s=r['runtime_mean_s'],search_runtime_sd_s=r['runtime_sd_s']);wide.append(row)
    csv_write(dest/'COMPLETE_TABLES_100_1000_5000.csv',wide)

def publish_summaries(root,dest,hvrows,np):
    pairs,summaries,index=build_summaries(root,hvrows,np)
    csv_write(dest/'PER_RUN_HV.csv',hvrows);csv_write(dest/'PER_PAIR_HV.csv',pairs)
    csv_write(dest/'ALL_SCENARIO_SUMMARY.csv',summaries)
    outcomes=[]
    for sk in sorted({(r['table'],r['network'],r['method'],r['reference_method'],r['scenario_group']) for r in pairs}):
        rows=[r for r in pairs if (r['table'],r['network'],r['method'],r['reference_method'],r['scenario_group'])==sk]
        for tolerance in TOLERANCES:
            wins=[r['relative_percent'] for r in rows if r['wtl_'+tolerance]=='W']
            losses=[r['relative_percent'] for r in rows if r['wtl_'+tolerance]=='L']
            all_values=[r['relative_percent'] for r in rows]
            outcomes.append(dict(table=sk[0],network=sk[1],method=sk[2],reference_method=sk[3],scenario_group=sk[4],
                tolerance=tolerance,W=len(wins),T=sum(r['wtl_'+tolerance]=='T' for r in rows),L=len(losses),
                mean_win_percent=mean(wins) if wins and all(v is not None for v in wins) else None,
                mean_loss_decline_percent=-mean(losses) if losses and all(v is not None for v in losses) else None,
                median_all_paired_percent=median(all_values) if all(v is not None for v in all_values) else None))
    csv_write(dest/'WIN_LOSS_MAGNITUDES.csv',outcomes)
    for n in (100,1000,5000):csv_write(dest/f'RESULTS_{n}.csv',[r for r in summaries if r['scenario_group']==f'fresh_{n}'])
    csv_write(dest/'OPTIMIZATION_100.csv',[r for r in summaries if r['scenario_group']=='optimization_100'])
    diffs=[]
    for r in hvrows:
        if r['scenario_group']!='fresh_100':continue
        for n,k in ((100,1000),(100,5000),(1000,5000)):
            base=key(r)[:-1];a,b=index[base+(f'fresh_{n}',)]['hv'],index[base+(f'fresh_{k}',)]['hv']
            diffs.append({**{c:r[c] for c in ('table','configuration','instance','pair_instance','network','method','seed')},
                'comparison':f'{n}_vs_{k}','hv_a':a,'hv_reference':b,'relative_percent':relative(a,b)})
    csv_write(dest/'SCENARIO_DIFFERENCES_PER_RUN.csv',diffs)
    ds=[]
    for sk in sorted({(r['table'],r['network'],r['method'],r['comparison']) for r in diffs}):
        rows=[r for r in diffs if (r['table'],r['network'],r['method'],r['comparison'])==sk]
        ds.append(dict(table=sk[0],network=sk[1],method=sk[2],comparison=sk[3],runs=len(rows),
            mean_relative_percent=equal_mean(rows,'relative_percent')))
    csv_write(dest/'SCENARIO_DIFFERENCES.csv',ds)
    direct=[]
    for a,b in [(SHARED,'ICM_RANDOM_RATE_CONTROLLED'),(SHARED,'ICM_SHARED_ZERO_THRESHOLD'),
                ('ICM_RANDOM_RATE_CONTROLLED','ICM_HEURISTIC'),('ICM_TERMINAL_PPO',SHARED)]:
        for r in hvrows:
            if r['method']!=a or r['table']=='Table7':continue
            bk=(r['table'],r['pair_instance'],r['network'],b,r['seed'],r['scenario_group'])
            if bk not in index:continue
            ref=index[bk];base=index[r['table'],r['pair_instance'],r['network'],'BASE',r['seed'],r['scenario_group']]
            delta=r['hv']-ref['hv']
            direct.append(dict(**r,reference_method=b,reference_network=r['network'],reference_hv=ref['hv'],
                relative_percent=relative(r['hv'],ref['hv']),hv_difference=delta,
                delta_percentage_points_vs_adapted=100*delta/base['hv'] if base['hv']>0 else None,
                **{'wtl_'+name:wtl(delta,tol) for name,tol in TOLERANCES.items()}))
    if direct:
        csv_write(dest/'DIRECT_COMPARISONS_PER_PAIR.csv',direct);dr=[]
        for sk in sorted({(r['table'],r['method'],r['reference_method'],r['scenario_group']) for r in direct}):
            rows=[r for r in direct if (r['table'],r['method'],r['reference_method'],r['scenario_group'])==sk]
            ci=bootstrap(np,rows,'delta_percentage_points_vs_adapted',491000000+len(dr))
            dr.append(dict(table=sk[0],method=sk[1],reference_method=sk[2],scenario_group=sk[3],
                mean_difference_pp=equal_mean(rows,'delta_percentage_points_vs_adapted'),
                difference_pp_ci95_low=ci[0],difference_pp_ci95_high=ci[1],
                **summary(np,rows,492000000+len(dr))))
        csv_write(dest/'DIRECT_COMPARISONS.csv',dr)
    write_display(dest,summaries)
    return pairs,summaries

def export_existing(root,plan):
    """Available immediately: all three sizes, BEFORE the new union changes bounds."""
    marker=root/'EXISTING_RESULT_LOCATION.json'
    if marker.exists():q=read(marker);verify(q['files']);return Path(q['directory'])
    import numpy as np
    source=Path(read(Path(plan['parent_root'])/'RESULT_LOCATION.json')['directory'])
    hvrows=[]
    for row in csv_read(source/'PER_RUN_HV.csv'):
        require(float(row['alpha'])==.9,'Mixed alpha in original report')
        hvrows.append(dict(table=row['table'],configuration=row['configuration'],instance=row['instance'],
            pair_instance=row['pair_instance'],network=row['network'],method=row['method'],seed=int(row['seed']),
            scenario_group=row['scenario_group'],evaluation_scenarios=int(row['evaluation_scenarios']),alpha=.9,
            hv=float(row['hv']),final_solutions=int(row['final_solutions']),search_seconds=float(row['original_search_seconds']),
            actual_search_evaluations=int(row['actual_search_evaluations'])))
    require(len(hvrows)==810*4,'Original report incomplete')
    dest=root/('existing_P90_before_standard_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'));dest.mkdir()
    publish_summaries(root,dest,hvrows,np)
    freeze(dest/'REPORT.json',dict(completed=True,alpha=.9,scope='Original 810 P90 runs, BEFORE adding standard NSGA-II',
        source=str(source),new_searches=0,new_objective_evaluations=0,hv_recomputed=False,
        evaluation_scenarios=[100,1000,5000],hv_divisor=1.331,wtl_tolerances_reported_hv_units=TOLERANCES,
        bootstrap='50000, configurations fixed, paired instances then seeds',
        warning='These bounds apply to the original method panel only; final union report is separate'))
    freeze(marker,dict(directory=str(dest),files={str(p):digest(p) for p in dest.iterdir() if p.is_file()}))
    print('[STANDARD-NSGA2-P90] Existing three-size report:',dest,flush=True)
    return dest

def report(root,plan,merged,np,hv_fn,full):
    marker=root/'RESULT_LOCATION.json'
    if marker.exists():q=read(marker);verify(q['files']);return Path(q['directory'])
    old_source=Path(read(Path(plan['parent_root'])/'RESULT_LOCATION.json')['directory'])
    old_hvs={(r['table'],r['instance'],r['method'],int(r['seed']),r['scenario_group']):float(r['hv'])
             for r in csv_read(old_source/'PER_RUN_HV.csv')}
    old_bounds=read(old_source/'INSTANCE_BOUNDS.json')
    bounds={}
    for g,old,oldpacks,new,newpacks in merged:
        points=np.concatenate([packs[tag]['quantiles'] for packs in (oldpacks,newpacks) if packs is not None for tag in TAGS[1:]])
        require(len(points)>0,'No points for union bounds')
        lo,hi=points.min(axis=0),points.max(axis=0);unit=g['bounds_unit']
        if unit in bounds:
            lo=np.minimum(lo,bounds[unit]['minimum']);hi=np.maximum(hi,bounds[unit]['maximum'])
        bounds[unit]=dict(minimum=lo.tolist(),maximum=hi.tolist(),reference=[1.1]*3,divisor=1.331,
            scope='All P90 methods, search seeds, final decisions, fresh100/1000/5000; paired networks share bounds')
    hvrows,counts,shifts,plan_shifts,reproductions,coverage,beyond=[],[],[],[],[],[],[]
    for g,old,oldpacks,new,newpacks in merged:
        lo=np.asarray(bounds[g['bounds_unit']]['minimum']);hi=np.asarray(bounds[g['bounds_unit']]['maximum']);span=np.where(hi>lo,hi-lo,1.)
        prior=old_bounds[g['bounds_unit']];olo=np.asarray(prior['minimum']);ohi=np.asarray(prior['maximum']);ospan=np.where(ohi>olo,ohi-olo,1.)
        for packet,packs in ((old,oldpacks),(new,newpacks)):
            if packet is None:continue
            byid={r['exact_decision_id']:i for i,r in enumerate(packet['decisions'])}
            for r in packet['source_runs']:
                ids=[byid[m['exact_decision_id']] for m in packet['memberships'] if (m['method'],m['seed'])==(r['method'],r['seed'])]
                require(len(ids)==r['final_solutions'],'Front membership count differs')
                common={k:r[k] for k in ('table','configuration','instance','method','seed')}
                common.update(pair_instance=packet['instance'].get('pair_id',r['instance']),network=g['network'])
                seconds=read(r['run_path'])['optimisation_seconds']
                counts.append(dict(**common,final_solutions=len(ids),actual_search_evaluations=r['actual_evaluations'],
                    search_seconds=seconds,source_front=r['front_path']))
                for tag in TAGS:
                    points=packs[tag]['quantiles'][ids];z=(points-lo)/span
                    value=float(hv_fn(z.tolist(),[1.1]*3)/1.331) if len(ids) else 0.
                    hvrows.append(dict(**common,scenario_group=tag,evaluation_scenarios=int(tag.rsplit('_',1)[1]),alpha=.9,
                        hv=value,final_solutions=len(ids),search_seconds=seconds,actual_search_evaluations=r['actual_evaluations']))
                    beyond.append(dict(**common,scenario_group=tag,plans=len(ids),
                        outside_upper_bound=int(np.any(z>1,axis=1).sum()),
                        beyond_reference=int(np.any(z>1.1,axis=1).sum()),
                        beyond_reference_percent=float(np.any(z>1.1,axis=1).mean()*100) if len(ids) else None))
                    if packet is old:
                        previous=old_hvs[r['table'],r['instance'],r['method'],r['seed'],tag]
                        reproduced=float(hv_fn(((points-olo)/ospan).tolist(),[1.1]*3)/1.331) if len(ids) else 0.
                        require(abs(reproduced-previous)<=1e-9,'Old report HV not reproduced: '+str(common))
                        reproductions.append(dict(**common,scenario_group=tag,old_hv=previous,reproduced_old_hv=reproduced,
                            new_union_hv=value,change_due_to_bounds=value-previous))
                for j,obj in enumerate(('cost','emission','makespan')):
                    a=packs['optimization_100']['quantiles'][ids,j];b=packs['fresh_5000']['quantiles'][ids,j]
                    values=[relative(float(y),float(x)) for x,y in zip(a,b)]
                    valid=bool(values) and all(v is not None for v in values)
                    shifts.append(dict(**common,objective=obj,plans=len(ids),
                        mean_P90_relative_shift_percent=mean(values) if valid else None,
                        median_P90_relative_shift_percent=median(values) if valid else None))
                    cv=packs['fresh_5000']['coverage'][ids,j]
                    coverage.append(dict(**common,objective=obj,plans=len(ids),
                        mean_coverage=float(np.mean(cv)) if len(ids) else None,
                        median_coverage=float(np.median(cv)) if len(ids) else None,
                        fraction_below_085=float(np.mean(cv<.85)) if len(ids) else None))
                    for i,x,y,v in zip(ids,a,b,values):
                        plan_shifts.append(dict(**common,exact_decision_id=packet['decisions'][i]['exact_decision_id'],
                            objective=obj,optimization_P90=float(x),fresh5000_P90=float(y),
                            difference=float(y-x),relative_difference_percent=v))
    require(Counter(r['table'] for r in counts)=={'Table5':756,'Table6':108,'Table7':27},'Merged panel is incomplete')
    require(len(reproductions)==810*4 and len(hvrows)==891*4,'Missing old/new HV rows')
    dest=root/('results_union_P90_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'));dest.mkdir()
    freeze(dest/'INSTANCE_BOUNDS.json',bounds)
    csv_write(dest/'FINAL_FRONT_COUNTS.csv',counts);csv_write(dest/'P90_SHIFT_PER_RUN.csv',shifts)
    csv_write(dest/'P90_SHIFT_PER_SOLUTION.csv',plan_shifts)
    csv_write(dest/'ORIGINAL_QUANTILE_COVERAGE_PER_RUN.csv',coverage)
    csv_write(dest/'OUTSIDE_REFERENCE_PER_RUN.csv',beyond)
    sm=[]
    for sk in sorted({(r['table'],r['network'],r['method'],r['objective']) for r in shifts}):
        rows=[r for r in shifts if (r['table'],r['network'],r['method'],r['objective'])==sk]
        ss=[r for r in plan_shifts if (r['table'],r['network'],r['method'],r['objective'])==sk]
        configs={r['configuration'] for r in rows}
        instance_counts={cfg:len({r['pair_instance'] for r in rows if r['configuration']==cfg}) for cfg in configs}
        seed_counts={inst:len([r for r in rows if r['pair_instance']==inst]) for inst in {r['pair_instance'] for r in rows}}
        sizes={(r['pair_instance'],r['seed']):r['plans'] for r in rows}
        weights=[1/(len(configs)*instance_counts[r['configuration']]*seed_counts[r['pair_instance']]*sizes[r['pair_instance'],r['seed']]) for r in ss]
        vals=[r['relative_difference_percent'] for r in ss]
        valid=bool(vals) and all(v is not None for v in vals) and all(r['plans']>0 for r in rows)
        sm.append(dict(table=sk[0],network=sk[1],method=sk[2],objective=sk[3],runs=len(rows),
            mean_P90_relative_shift_percent=equal_mean(rows,'mean_P90_relative_shift_percent'),
            hierarchy_weighted_median_relative_percent=weighted_median(np,vals,weights) if valid else None,
            pooled_plan_median_relative_percent=median(vals) if valid else None,
            mean_of_run_medians_percent=equal_mean(rows,'median_P90_relative_shift_percent')))
    csv_write(dest/'P90_SHIFT_BY_METHOD.csv',sm)
    csv_write(dest/'OLD_NEW_HV_REPRODUCTION.csv',reproductions)
    _,summaries=publish_summaries(root,dest,hvrows,np)
    priorrows={(r['table'],r['network'],r['configuration'],r['method'],r['scenario_group']):r
               for r in csv_read(old_source/'ALL_SCENARIO_SUMMARY.csv')}
    changes=[]
    for r in summaries:
        if r['reference_method']==METHOD or r['method']==METHOD:continue
        q=priorrows[r['table'],r['network'],r['configuration'],r['method'],r['scenario_group']]
        changes.append(dict(table=r['table'],network=r['network'],configuration=r['configuration'],method=r['method'],
            scenario_group=r['scenario_group'],reference_method=r['reference_method'],
            old_mean_hv=float(q['mean_hv']),new_mean_hv=r['mean_hv'],
            old_mean_relative_percent=float(q['mean_paired_relative_percent']) if q['mean_paired_relative_percent'] else None,
            new_mean_relative_percent=r['mean_relative_percent']))
    csv_write(dest/'OLD_NEW_SUMMARY.csv',changes)
    freeze(dest/'REPORT.json',dict(completed=True,alpha=.9,original_runs_reused=810,new_runs=81,total_runs=891,
        new_training_updates=0,standard_protocol=plan['protocol'],commit=plan['commit'],parent_commit=plan['parent_commit'],
        evaluation_scenarios=[100,1000,5000],ranks=[90,900,4500],search_budget=5000,hv_divisor=1.331,
        parent=str(old_source),old_outputs_modified=False,old_report_hv_reproductions=len(reproductions),
        bootstrap='50000: configurations fixed; resample instances then paired seeds',
        mean='Seeds within instance; instances within configuration; configurations equally',
        wtl_tolerances_reported_hv_units=TOLERANCES,
        independent_scenarios='Exact saved P90 fresh100/1000/5000; independent nonnested seeds, shared by methods/search seeds',
        zero_reference_hv='Relative change undefined; no dropped pairs or division by epsilon',
        empty_front='HV0; no outcome-based retry',Table7='No new searches; same paired-network union bounds, same references',
        runtime='Original full search runtime; reevaluation and bootstrap excluded; hardware and concurrent load logged'))
    verify(read(root/'oos/INPUT_MANIFEST.json')['files']);plan_for(root)
    freeze(marker,dict(directory=str(dest),files={str(p):digest(p) for p in dest.iterdir() if p.is_file()}))
    print('[STANDARD-NSGA2-P90] COMPLETE:',dest,flush=True)
    return dest
