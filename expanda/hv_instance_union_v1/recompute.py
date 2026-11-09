#!/usr/bin/env python3
"""Post-hoc HV with shared per-instance bounds from all final feasible fronts."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

HERE = Path(__file__).resolve().parent
SUPPORT = HERE.parent/'formal_icm_s0_v1/src/support'
sys.path.insert(0,str(SUPPORT))
from runtime_common import read_json, write_json, digest, paired_summary
from transformer_ppo.archive import exact_hv_3d, nondominated

METHODS=('BASE','I_DIRECTED_BALANCED','I_K8_RESIDUAL','C_K8_SIMPLE',
         'C_K8_RESIDUAL','M_BALANCED','M_K8_TRANSFORMER','I_C_M_K8_TRANSFORMER')
RULES={'I_K8_RESIDUAL':'I_DIRECTED_BALANCED','C_K8_RESIDUAL':'C_K8_SIMPLE',
       'M_K8_TRANSFORMER':'M_BALANCED'}
OBJECTIVES=('cost','emission','makespan')
REFERENCE=(1.1,1.1,1.1)


def experiment_design(plan,plan_path):
    if plan.get('config',{}).get('protocol')=='frozen-old-components-b400-recheck-v1':
        config=plan['config'];methods=METHODS[:-1]
        identity=dict(plan);claimed=identity.pop('sha256')
        actual=hashlib.sha256(json.dumps(identity,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
        if actual!=claimed:raise ValueError('Expanded experiment plan digest mismatch')
        names=plan['instances'];seeds=config['algorithm_seeds']
        if (tuple(config['methods'])!=methods or len(names)!=10 or len(seeds)!=10 or plan['runs']!=700
            or config['population_size']!=20 or config['evaluation_budget']!=400):
            raise ValueError('Expected 10-instance x 10-seed x 7-method B400 expansion')
        return {'instances':names,'algorithm_seeds':seeds,'methods':methods,'plan_sha':claimed,
                'base_bootstrap_seed':962000,'rule_bootstrap_seed':963000,'protocol':config['protocol']}
    if plan.get('protocol')!='icm-common-seed-component-screen-v1' or tuple(plan['methods'])!=METHODS:
        raise ValueError('Expected original eight-method experiment or seven-method B400 expansion')
    if len(plan['instances'])!=3 or len(plan['algorithm_seeds'])!=10 or plan['runs']!=240:
        raise ValueError('Expected original 3-instance x 10-seed x 8-method experiment')
    return {'instances':plan['instances'],'algorithm_seeds':plan['algorithm_seeds'],'methods':METHODS,
            'plan_sha':digest(plan_path),'base_bootstrap_seed':931100,'rule_bootstrap_seed':931500,'protocol':plan['protocol']}


def bounds_of(fronts):
    # Union of per-run fronts, NOT the globally nondominated union front.
    points=[p for front in fronts for p in front]
    if not points:raise ValueError('No feasible final points for this instance; bounds undefined')
    lo=[min(p[j] for p in points) for j in range(3)]
    hi=[max(p[j] for p in points) for j in range(3)]
    return {'minimum':lo,'maximum':hi,'constant_axes':[j for j in range(3) if hi[j]==lo[j]],
            'front_point_count_with_duplicates':len(points),'unique_point_count':len(set(points)),
            'reference':list(REFERENCE)}


def normalize(points,bounds):
    lo,hi=bounds['minimum'],bounds['maximum']
    values=[tuple(0. if hi[j]==lo[j] else (p[j]-lo[j])/(hi[j]-lo[j]) for j in range(3)) for p in points]
    if any(v < -1e-12 or v > 1+1e-12 for p in values for v in p):
        raise ValueError('Point outside shared bounds; inputs changed')
    return values


def load_runs(source):
    plan_path=source/'PLAN.json';plan=read_json(plan_path)
    plan=experiment_design(plan,plan_path)
    plan_sha=plan['plan_sha'];inputs={str(plan_path.resolve()):digest(plan_path)};rows=[]
    if len(set(plan['instances']))!=len(plan['instances']) or len(set(plan['algorithm_seeds']))!=len(plan['algorithm_seeds']):
        raise ValueError('Duplicate instances or seeds in plan')
    for name in plan['instances']:
        for seed in plan['algorithm_seeds']:
            scenarios=set()
            for method in plan['methods']:
                folder=source/'runs'/name/f'seed_{seed}'/method
                paths=[folder/f for f in ('RUN_COMPLETE.json','configuration.json','final_feasible_nondominated.json')]
                for path in paths:inputs[str(path.resolve())]=digest(path)
                run,config,front=[read_json(p) for p in paths]
                if (not run['completed'] or run['instance']!=name or run['algorithm_seed']!=seed
                    or run['strategy']!=method or run['plan_sha256']!=plan_sha
                    or run['real_objective_evaluation_count']!=400 or run['scenario_digest']!=config['scenario_digest']):
                    raise ValueError(f'Invalid original run: {folder}')
                scenarios.add(run['scenario_digest'])
                points=[]
                for item in front:
                    if item.get('training_feasible') is not True:raise ValueError(f'Infeasible item in final feasible front: {folder}')
                    if item['source_run']!=name:raise ValueError('Front belongs to wrong instance')
                    p=tuple(float(item['optimisation_objectives'][k]) for k in OBJECTIVES)
                    if not all(math.isfinite(v) for v in p):raise ValueError('Nonfinite objective')
                    points.append(p)
                if set(nondominated(points))!=set(points):raise ValueError(f'Stored front contains dominated points: {folder}')
                old=config['normalization'];lo=old['minimum'];hi=old['maximum'];div=old['objective_divisor']
                if tuple(old['reference'])!=REFERENCE:raise ValueError('Original reference differs from 1.1; explicit review required')
                old_hv=exact_hv_3d([[(p[j]/div[j]-lo[j])/(hi[j]-lo[j]) for j in range(3)] for p in points],REFERENCE)
                if not math.isclose(old_hv,run['final_hypervolume'],rel_tol=1e-10,abs_tol=1e-12):
                    raise ValueError(f'Original HV cannot be reproduced: {folder}')
                rows.append({'instance':name,'algorithm_seed':seed,'method':method,'points':points,
                             'old_hv':run['final_hypervolume'],'replayed_old_hv':old_hv,'seconds':run['optimisation_seconds']})
            if len(scenarios)!=1:raise ValueError('Optimization scenarios not paired')
    return plan,rows,inputs


def write_csv(path,rows):
    with path.open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def compare(rows,method,reference,seed):
    index={(r['instance'],r['algorithm_seed'],r['method']):r for r in rows}
    pairs=[]
    for r in rows:
        if r['method']!=method:continue
        b=index[r['instance'],r['algorithm_seed'],reference]
        pairs.append({'instance':r['instance'],'absolute':r['hv']-b['hv'],
                      'percent':100*(r['hv']/b['hv']-1) if b['hv']>0 else None})
    absolute=paired_summary(pairs,'absolute',seed);percent=paired_summary(pairs,'percent',seed)
    if percent['undefined_pairs']:
        percent.update(mean=None,hierarchical_bootstrap_95ci_low=None,hierarchical_bootstrap_95ci_high=None)
    return {'reference':reference,'absolute':absolute,'percent':percent}


def execute(source,out):
    source=source.resolve();out=out.resolve()
    if source==out or source in out.parents or out in source.parents:
        raise ValueError('Use a separate sibling output; never overwrite original results')
    if out.exists():raise FileExistsError(f'Output already exists; preserve it: {out}')
    plan,rows,inputs=load_runs(source)
    bounds={name:bounds_of([r['points'] for r in rows if r['instance']==name]) for name in plan['instances']}
    for row in rows:row['hv']=exact_hv_3d(normalize(row['points'],bounds[row['instance']]),REFERENCE)
    keyed={(r['instance'],r['algorithm_seed'],r['method']):r for r in rows}
    table=[];methods={};per_instance=[];per_run=[]
    for i,method in enumerate(plan['methods']):
        actual=[r for r in rows if r['method']==method]
        vsbase=compare(rows,method,'BASE',plan['base_bootstrap_seed']+i)
        vsrule=compare(rows,method,RULES[method],plan['rule_bootstrap_seed']+i) if method in RULES else None
        oldrows=[dict(r,hv=r['old_hv']) for r in rows]
        oldsummary=compare(oldrows,method,'BASE',plan['base_bootstrap_seed']+i)
        methods[method]={'vs_base':vsbase,'vs_rule':vsrule,'old_vs_base':oldsummary}
        p=vsbase['percent']
        table.append({'method':method,'old_mean_percent_vs_base':oldsummary['percent']['mean'],
              'mean_percent_vs_base':p['mean'],'ci_low':p['hierarchical_bootstrap_95ci_low'],'ci_high':p['hierarchical_bootstrap_95ci_high'],
              'mean_absolute_vs_base':vsbase['absolute']['mean'],'rule':RULES.get(method),
              'mean_percent_vs_rule':vsrule['percent']['mean'] if vsrule else None,
              'rule_ci_low':vsrule['percent']['hierarchical_bootstrap_95ci_low'] if vsrule else None,
              'rule_ci_high':vsrule['percent']['hierarchical_bootstrap_95ci_high'] if vsrule else None,
              'wins':vsbase['absolute']['wins'],'ties':vsbase['absolute']['ties'],'losses':vsbase['absolute']['losses'],
              'mean_seconds':statistics.fmean(r['seconds'] for r in actual)})
        for name in plan['instances']:
            subset=[r for r in actual if r['instance']==name]
            entry=next((x for x in p['per_instance'] if x['instance']==name),None)
            per_instance.append({'instance':name,'method':method,'mean_hv':statistics.fmean(r['hv'] for r in subset),
                                 'mean_percent_vs_base':entry['mean'] if entry else None})
    for r in rows:
        b=keyed[r['instance'],r['algorithm_seed'],'BASE']
        per_run.append({k:r[k] for k in ('instance','algorithm_seed','method','old_hv','hv')} |
                       {'base_hv':b['hv'],'absolute_vs_base':r['hv']-b['hv'],
                        'percent_vs_base':100*(r['hv']/b['hv']-1) if b['hv']>0 else None,'front_points':len(r['points'])})
    inputs[str(Path(__file__).resolve())]=digest(Path(__file__))
    inputs[str((SUPPORT/'transformer_ppo/archive.py').resolve())]=digest(SUPPORT/'transformer_ppo/archive.py')
    inputs[str((SUPPORT/'runtime_common.py').resolve())]=digest(SUPPORT/'runtime_common.py')
    result={'completed':True,'runs':len(rows),'methods':methods,'reference':list(REFERENCE),
            'source_protocol':plan['protocol'],'instance_count':len(plan['instances']),
            'method_count':len(plan['methods']),'seeds_per_instance':len(plan['algorithm_seeds']),
            'normalization':'per-instance min/max over union of all methods and all seeds in this experiment final feasible nondominated fronts; no pooled-front nondominance filtering before bounds',
            'constant_axis_rule':'map to zero; recorded in bounds; no epsilon denominator',
            'original_hv_reproduced':True,'max_old_hv_replay_error':max(abs(r['old_hv']-r['replayed_old_hv']) for r in rows),
            'new_objective_evaluations':0,'new_training_steps':0,'new_search_runs':0,
            'qualification':'Post-hoc evaluation metric change only. Normalization must not be used to retrain or change gates. Intervals condition on the pooled bounds; not independent test evidence.'}
    out.mkdir(parents=True)
    write_json(out/'INSTANCE_BOUNDS.json',{'objective_order':OBJECTIVES,'instances':bounds})
    write_json(out/'INPUT_MANIFEST.json',{'source':str(source),'files':inputs})
    write_json(out/'QUICK_RESULT.json',result)
    write_csv(out/'RESULT_TABLE.csv',table);write_csv(out/'PER_INSTANCE.csv',per_instance);write_csv(out/'PER_RUN.csv',per_run)
    print((out/'RESULT_TABLE.csv').read_text(encoding='utf-8-sig'),flush=True)
    print(f'[HV-RECOMPUTE] COMPLETE {len(rows)} runs; zero new evaluations; {out}',flush=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();execute(a.source,a.out)
