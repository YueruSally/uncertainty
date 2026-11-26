"""Read parent P80 packs, evaluate only new final plans, export new union HV."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
from statistics import mean, median, stdev
from addon_common import *

TAGS = ('optimization_100','fresh_100','fresh_1000','fresh_5000')
LABELS = {'BASE':'Adapted NSGA-II', METHOD:'NSGA-II (standard framework)',
          'I_RULE':'I (Directed)','I_LEARN':'I Transformer (SL)',
          'C_RULE':'C (Simple)','C_LEARN':'C Transformer (SL)',
          'M_RULE':'M (Balanced)','M_LEARN':'M Transformer (SL)',
          'ICM_HEURISTIC':'ICM (rules)',
          'ICM_INDEPENDENT':'ICM Transformer (independent SL)',
          'ICM_RANDOM_RATE_CONTROLLED':'ICM (random replacement)',
          'ICM_SHARED_ZERO_THRESHOLD':'ICM Transformer (shared SL, threshold = 0)',
          'ICM_SHARED_SUPERVISED':'ICM Transformer (shared SL)',
          'ICM_TERMINAL_PPO':'ICM Transformer (shared SL + PPO)'}


def equal_mean(rows, key):
    if not rows or any(r[key] is None for r in rows): return None
    groups = defaultdict(lambda:defaultdict(list))
    for r in rows: groups[r['configuration']][r['instance']].append(r[key])
    return mean(mean(mean(v) for v in instances.values()) for instances in groups.values())


def relative(x,y): return 100*(x-y)/y if y>0 else None


def bootstrap(np, rows, key, seed):
    """Configurations fixed; resample instances, then their paired seeds."""
    if any(r[key] is None for r in rows): return [None,None]
    rng=np.random.default_rng(seed); estimates=np.zeros(50000)
    configs=sorted({r['configuration'] for r in rows})
    for cfg in configs:
        part=[r for r in rows if r['configuration']==cfg]
        instances=sorted({r['instance'] for r in part})
        values=np.asarray([[r[key] for r in sorted(part,key=lambda x:x['seed']) if r['instance']==i] for i in instances])
        n,k=values.shape
        ii=rng.integers(0,n,size=(50000,n))
        ss=rng.integers(0,k,size=(50000,n,k))
        estimates+=values[ii[:,:,None],ss].mean(axis=(1,2))/len(configs)
    return np.quantile(estimates,[.025,.975]).tolist()


def assemble(root, plan, full, inv, np, ev):
    parent=Path(plan['parent_root'])/'oos'
    require(read(parent/'COMPLETE.json')['alpha']==.8,'Parent OOS alpha differs')
    manifest=read(parent/'PLAN.json'); require(manifest['alpha']==.8,'Parent OOS manifest differs')
    verify(manifest['files'])
    retained={str(parent/'PLAN.json'):digest(parent/'PLAN.json'),
              str(parent/'COMPLETE.json'):digest(parent/'COMPLETE.json')}
    groups=[g for g in manifest['groups'] if g['table'] in ('Table5','Table6')]
    require(len(groups)==27,'Expected 18+9 instances')
    merged=[]
    dest=root/'oos'; dest.mkdir(exist_ok=True)
    def progress(g,tag,done,total,count,physical,elapsed):
        status(root,'standard_final_front_reevaluation',instance=g['instance'],scenario_group=tag,
               done=done,total=total,evaluation_scenarios=count,elapsed_seconds=round(elapsed,1))
    for g in groups:
        old_folder=Path(g['folder']); old=read(old_folder/'INPUTS.json')
        old_packs={tag:full.load_pack(old_folder,old,tag) for tag in TAGS}
        for f in old_folder.rglob('*'):
            if f.is_file() and (f.name in ('INPUTS.json','instance.json','COMPLETE.json') or f.suffix in ('.npz','.json')):
                retained[str(f)]=digest(f)
        tasks=[t for t in plan['tasks'] if (t['table'],t['instance'])==(g['table'],g['instance'])]
        require(len(tasks)==3 and all(completed(t) for t in tasks),'Standard panel incomplete')
        solutions, members, runs = {},[],[]
        for t in tasks:
            folder=Path(t['folder']); r=read(folder/'RUN_COMPLETE.json'); cfg=read(folder/'configuration.json')
            require(r['scenario_digest']==old['source_config']['scenario_digest'] and
                    cfg['uncertainty']==old['source_config']['uncertainty'],'Standard/main scenarios differ')
            for row in read(folder/'final_feasible_nondominated.json'):
                key=inv.exact_id(row)
                if key in solutions:
                    require(np.allclose(inv.vector(row),inv.vector(solutions[key]),rtol=full.RTOL,atol=full.ATOL),
                            'Identical standard decision has inconsistent objective values')
                solutions[key]=dict(row,exact_decision_id=key)
                members.append(dict(method=METHOD,seed=t['seed'],exact_decision_id=key,source_solution_id=row['source_solution_id']))
            runs.append(dict(table=t['table'],configuration=t['entry']['configuration'],instance=t['instance'],
                method=METHOD,seed=t['seed'],network='expanded',run_path=str(folder/'RUN_COMPLETE.json'),
                front_path=str(folder/'final_feasible_nondominated.json'),final_solutions=r['final_front_size'],
                actual_evaluations=5000,scenario_digest=r['scenario_digest']))
            retained.update(read(folder/'JOB_COMPLETE.json')['files'])
        packet=dict(table=g['table'],instance=old['instance'],source_config=old['source_config'],source_runs=runs,
            decisions=[solutions[k] for k in sorted(solutions)],memberships=members,seeds=old['seeds'],
            bounds_unit=g['bounds_unit'],alpha=.8,workbook=old['workbook'],
            pilot_scenarios=str(old_folder/'scenarios'))
        packet['sha256']=canonical(packet)
        folder=dest/'instances'/g['table']/g['instance']
        freeze(folder/'INPUTS.json',packet); freeze(folder/'instance.json',packet['instance'])
        new_group={**g,'folder':str(folder),'unique_decisions':len(solutions),'inputs_sha256':packet['sha256']}
        packs={}
        if solutions:
            for tag in TAGS:
                full.evaluate_pack(np,ev,dest,new_group,packet,tag,[new_group],.8,progress)
                packs[tag]=full.load_pack(folder,packet,tag)
                require(read(folder/'evaluations'/tag/'COMPLETE.json')['scenario_digest']==
                        read(old_folder/'evaluations'/tag/'COMPLETE.json')['scenario_digest'],
                        'Independent draws not identical between methods')
        else:
            freeze(folder/'EMPTY_FINAL_FRONTS.json',dict(all_three_runs_empty=True,hv=0,
                definition='No feasible final plan; not replaced with a previous generation or a new search'))
            packs={tag:dict(quantiles=np.empty((0,3)),coverage=np.empty((0,3))) for tag in TAGS}
        merged.append((g,old,old_packs,packet,packs))
    verify(retained)
    freeze(dest/'INPUT_MANIFEST.json',dict(files=retained,parent_plan=manifest['source_plan_sha256'],
        standard_plan=plan['sha256'],alpha=.8,new_searches_this_stage=0,new_training_updates=0,
        scenario_groups='Exact saved parent draws; independent sizes, non-nested; shared by all methods/seeds'))
    return merged


def report(root, plan, merged, np, hv_fn):
    if (root/'RESULT_LOCATION.json').exists():
        q=read(root/'RESULT_LOCATION.json');verify(q['files']);return Path(q['directory'])
    dest=root/('results_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'));dest.mkdir()
    hvrows,counts,shifts,bounds=[],[],[],{}
    for g,old,old_packs,new,new_packs in merged:
        points=np.concatenate([packs[tag]['quantiles'] for packs in (old_packs,new_packs) for tag in TAGS[1:]])
        lo,hi=points.min(axis=0),points.max(axis=0); span=np.where(hi>lo,hi-lo,1.)
        bounds[g['bounds_unit']]=dict(minimum=lo.tolist(),maximum=hi.tolist(),reference=[1.1]*3,divisor=1.331,
            definition='All original P80 methods plus standard NSGA-II; all seeds and fresh100/1000/5000 final-plan P80 values')
        for packet,packs in ((old,old_packs),(new,new_packs)):
            ids_by_key={r['exact_decision_id']:i for i,r in enumerate(packet['decisions'])}
            for r in packet['source_runs']:
                ids=[ids_by_key[m['exact_decision_id']] for m in packet['memberships'] if
                     (m['method'],m['seed'])==(r['method'],r['seed'])]
                require(len(ids)==r['final_solutions'],'Final-front count differs')
                common={k:r[k] for k in ('table','configuration','instance','method','seed')}
                seconds=read(r['run_path'])['optimisation_seconds']
                counts.append(dict(**common,final_solutions=len(ids),actual_search_evaluations=r['actual_evaluations'],
                    search_seconds=seconds,source_front=r['front_path']))
                for tag in TAGS:
                    z=(packs[tag]['quantiles'][ids]-lo)/span
                    hvrows.append(dict(**common,method_label=LABELS.get(r['method'],r['method']),
                        scenario_group=tag,evaluation_scenarios=int(tag.rsplit('_',1)[1]),alpha=.8,
                        hv=float(hv_fn(z.tolist(),[1.1]*3)/1.331) if len(ids) else 0.,
                        final_solutions=len(ids),search_seconds=seconds,actual_search_evaluations=r['actual_evaluations']))
                for j,obj in enumerate(('cost','emission','makespan')):
                    a=packs['optimization_100']['quantiles'][ids,j];b=packs['fresh_5000']['quantiles'][ids,j]
                    values=[relative(float(y),float(x)) for x,y in zip(a,b)]
                    valid=bool(values) and all(v is not None for v in values)
                    shifts.append(dict(**common,objective=obj,plans=len(ids),
                        mean_P80_relative_shift_percent=mean(values) if valid else None,
                        median_P80_relative_shift_percent=median(values) if valid else None))
    freeze(dest/'INSTANCE_BOUNDS.json',bounds)
    csv_write(dest/'PER_RUN_HV.csv',hvrows);csv_write(dest/'FINAL_FRONT_COUNTS.csv',counts)
    csv_write(dest/'P80_SHIFT_PER_RUN.csv',shifts)
    index={(r['table'],r['instance'],r['seed'],r['method'],r['scenario_group']):r for r in hvrows}
    require(len(index)==len(hvrows),'Duplicate HV index')
    pairs=[]; summaries=[]
    for r in hvrows:
        for ref in ('BASE',METHOD):
            b=index[r['table'],r['instance'],r['seed'],ref,r['scenario_group']]
            delta=r['hv']-b['hv']
            pairs.append(dict(**r,reference_method=ref,reference_label=LABELS[ref],reference_hv=b['hv'],
                relative_percent=relative(r['hv'],b['hv']),hv_difference=delta,
                wtl='W' if delta>1e-12/1.331 else 'L' if delta< -1e-12/1.331 else 'T'))
    csv_write(dest/'PER_PAIR_HV.csv',pairs)
    keys=sorted({(r['table'],r['method'],r['reference_method'],r['scenario_group']) for r in pairs})
    for i,(table,method,ref,tag) in enumerate(keys):
        rows=[r for r in pairs if (r['table'],r['method'],r['reference_method'],r['scenario_group'])==(table,method,ref,tag)]
        for cfg in ['ALL']+sorted({r['configuration'] for r in rows}):
            part=rows if cfg=='ALL' else [r for r in rows if r['configuration']==cfg]
            interval=bootstrap(np,part,'relative_percent',480000000+i*20+len(summaries)) if method!=ref else [None,None]
            times=[r['search_seconds'] for r in part]
            summaries.append(dict(table=table,configuration=cfg,method=method,method_label=LABELS.get(method,method),
                reference_method=ref,reference_label=LABELS[ref],scenario_group=tag,
                evaluation_scenarios=int(tag.rsplit('_',1)[1]),runs=len(part),mean_hv=equal_mean(part,'hv'),
                reference_mean_hv=equal_mean(part,'reference_hv'),mean_relative_percent=equal_mean(part,'relative_percent'),
                ci95_low=interval[0],ci95_high=interval[1],undefined_relative_pairs=sum(r['relative_percent'] is None for r in part),
                W=sum(r['wtl']=='W' for r in part) if method!=ref else None,
                T=sum(r['wtl']=='T' for r in part) if method!=ref else None,
                L=sum(r['wtl']=='L' for r in part) if method!=ref else None,
                runtime_mean_s=mean(times),runtime_sd_s=stdev(times) if len(times)>1 else None))
    csv_write(dest/'ALL_SCENARIO_SUMMARY.csv',summaries)
    for n in (100,1000,5000):
        csv_write(dest/f'RESULTS_{n}.csv',[r for r in summaries if r['scenario_group']==f'fresh_{n}'])
    diffs=[]
    for r in hvrows:
        if r['scenario_group']!='fresh_100':continue
        for n,k in ((100,1000),(100,5000),(1000,5000)):
            key=(r['table'],r['instance'],r['seed'],r['method'])
            a,b=index[key+(f'fresh_{n}',)]['hv'],index[key+(f'fresh_{k}',)]['hv']
            diffs.append({**{c:r[c] for c in ('table','configuration','instance','method','seed')},
                'comparison':f'{n}_vs_{k}','hv_a':a,'hv_b':b,'relative_percent':relative(a,b)})
    csv_write(dest/'SCENARIO_DIFFERENCES_PER_RUN.csv',diffs)
    freeze(dest/'REPORT.json',dict(completed=True,alpha=.8,original_runs_reused=len(counts)-81,new_runs=81,
        total_runs=len(counts),standard_protocol=plan['protocol'],source_commit=plan['commit'],parent_commit=plan['parent_commit'],
        evaluation_scenarios=[100,1000,5000],ranks=[80,800,4000],search_budget=5000,hv_divisor=1.331,
        old_outputs_modified=False,bootstrap='50000: configurations fixed; resample instances then paired seeds',
        mean='Seeds within instance, instances within configuration, configurations equally',
        zero_reference_hv='Relative change undefined; no pairs dropped or division by epsilon',
        empty_final_front='HV=0, plan count=0; no outcome-based retry',
        labels='BASE is renamed Adapted NSGA-II in new displays only; raw method ID unchanged'))
    freeze(root/'RESULT_LOCATION.json',dict(directory=str(dest),files={str(f):digest(f) for f in dest.iterdir() if f.is_file()}))
    print('[STANDARD-NSGA2] New union report:',dest,flush=True)
    return dest


def main(root):
    plan=plan_for(root)
    sys.path.insert(0,str(EXP/'final_front_oos_hv_v1'))
    inv=load('inventory',EXP/'final_front_oos_hv_v1/inventory.py')
    load('run',EXP/'final_front_oos_hv_v1/run.py')
    full=load('standard_final_front_runtime',EXP/'final_front_oos_hv_v1/full.py')
    np,ev,hv=full.load_runtime()
    merged=assemble(root,plan,full,inv,np,ev)
    result=report(root,plan,merged,np,hv)
    verify(read(root/'oos/INPUT_MANIFEST.json')['files']);plan_for(root)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True,type=Path)
    main(p.parse_args().out.resolve())
