"""Paired, configuration-equal descriptive selection; no screen-driven refitting."""
from common import *
import csv
from collections import defaultdict
import numpy as np

def comparison(rows,method,reference,bootstrap=10000):
    indexed={(r['configuration'],r['instance'],r['seed'],r['arm']):r for r in rows}
    groups=defaultdict(lambda:defaultdict(list));absolute=defaultdict(lambda:defaultdict(list));wins=ties=losses=zero=0
    for r in rows:
        if r['arm']!=method:continue
        c,i,s=r['configuration'],r['instance'],r['seed'];b=indexed[c,i,s,reference]
        a,h=float(r['final_hypervolume']),float(b['final_hypervolume']);d=a-h
        absolute[c][i].append(d);groups[c][i].append(100*d/h if h>0 else None)
        wins+=int(d>1e-12);losses+=int(d < -1e-12);ties+=int(abs(d)<=1e-12);zero+=int(h==0)
    def summarize(group):
        if any(x is None for v in group.values() for xs in v.values() for x in xs):return {'mean':None,'low':None,'high':None}
        rng=np.random.default_rng(953001);draws=[];means=[]
        for c,inst in sorted(group.items()):
            x=np.asarray([inst[i] for i in sorted(inst)],dtype=float)
            # Each configuration stays in every draw. Resample instances, then
            # seeds within each sampled instance, keeping method pairs intact.
            ids=rng.integers(len(x),size=(bootstrap,len(x)))
            seeds=rng.integers(x.shape[1],size=(bootstrap,len(x),x.shape[1]))
            draws.append(x[ids[:,:,None],seeds].mean(axis=(1,2)));means.append(x.mean())
        sample=np.mean(draws,axis=0)
        return {'mean':float(np.mean(means)),'low':float(np.quantile(sample,.025)),'high':float(np.quantile(sample,.975))}
    per=[]
    for c in sorted(groups):
        vals=[x for a in groups[c].values() for x in a]
        per.append({'configuration':c,'mean_percent':None if any(x is None for x in vals) else float(np.mean(vals)),
                    'mean_absolute_difference':float(np.mean([np.mean(x) for x in absolute[c].values()]))})
    return {'percent':summarize(groups),'absolute':summarize(absolute),'reference_zero_pairs':zero,
            'wins':wins,'ties':ties,'losses':losses,'per_configuration':per}

def csv_write(path,records):
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)

def union_hv(rows):
    """Postprocessing only. Do not feed evaluation-set bounds into training or policy."""
    from transformer_ppo.archive import exact_hv_3d
    bounds={};updated=[]
    for name in sorted({r['instance'] for r in rows}):
        group=[r for r in rows if r['instance']==name]
        points=np.asarray([p for r in group for p in r['final_front_raw']],dtype=float).reshape(-1,3)
        if len(points):
            lo=points.min(0);hi=points.max(0);width=np.where(hi>lo,hi-lo,1.)
        else:lo=np.zeros(3);hi=np.zeros(3);width=np.ones(3)
        bounds[name]={'minimum':lo.tolist(),'maximum':hi.tolist(),'zero_range_dimensions':(hi==lo).tolist(),'reference':[1.1]*3,'final_feasible_points':len(points)}
        for r in group:
            p=np.asarray(r['final_front_raw'],dtype=float).reshape(-1,3)
            hv=exact_hv_3d(((p-lo)/width).tolist(),[1.1]*3)
            updated.append({**r,'training_normalization_hv':r['final_hypervolume'],'final_hypervolume':hv})
    return updated,bounds


def check_pairs(rows,methods,settings):
    groups=defaultdict(list)
    for r in rows:groups[r['instance'],r['seed']].append(r)
    for group in groups.values():
        if len(group)!=len(methods) or {r['arm'] for r in group}!=set(methods):raise ValueError('Incomplete method pairing')
        if len({r['scenario_digest'] for r in group})!=1:raise ValueError('Paired scenarios differ')
        if any(r['evaluation_budget']!=settings['budget'] or not settings['budget']-1<=r['real_objective_evaluation_count']<=settings['budget'] or r['hidden_candidate_objective_evaluations']!=0 or r['ppo_updates']!=0 for r in group):raise ValueError('Budget/hidden-evaluation mismatch')


def select_rules(rows,settings,root,sha):
    methods=['BASE']+[c+'_'+v for c in 'ICM' for v in settings['rule_variants']]
    check_pairs(rows,methods,settings)
    rows,bounds=union_hv(rows);profiles={};scores={}
    for c in 'ICM':
        scores[c]={v:comparison(rows,c+'_'+v,'BASE') for v in settings['rule_variants']}
        absolute=any(scores[c][v]['reference_zero_pairs'] for v in settings['rule_variants'])
        metric='absolute' if absolute else 'percent'
        profiles[c]=max(settings['rule_variants'],key=lambda v:(scores[c][v][metric]['mean'],v=='GUARDED'))
    value={'plan_sha256':sha,'profiles':profiles,'comparisons':scores,'calibration_instance_bounds':bounds,
        'selection':'largest mean paired union-normalized final HV gain on calibration only; tie chooses GUARDED',
        'qualification':'Rule selection is development, not evidence of benefit; negative variants remain visible; screening is untouched.'}
    freeze(root/'RULE_SELECTION.json',value);return value


def report(raw_rows,settings,out):
    expected=settings['screen_per_configuration']*len(settings['algorithm_seeds'])*len(METHODS)
    if len(raw_rows)!=expected:raise ValueError('Incomplete S11 screening')
    check_pairs(raw_rows,METHODS,settings)
    rows,bounds=union_hv(raw_rows);result={};table=[];acceptance={}
    for m in METHODS:
        own=[r for r in rows if r['arm']==m];ref=m[0]+'_RULE' if m.endswith('LEARN') else 'BASE'
        b=comparison(rows,m,'BASE');h=comparison(rows,m,ref)
        seconds=float(np.mean([r['optimisation_seconds'] for r in own]))
        result[m]={'vs_base':b,'vs_rule':h,'rule_reference':ref,'mean_seconds':seconds,
            'training_normalization_vs_base':comparison(raw_rows,m,'BASE'),
            'evaluations':sum(r['real_objective_evaluation_count'] for r in own),
            'selection':{k:{v:sum(r['selection'][k][v] for r in own) for v in ('calls','fallbacks','switches','unsupported')} for k in 'ICM'}}
        table.append({'method':m,'mean_hv':float(np.mean([r['final_hypervolume'] for r in own])),
            'percent_vs_base':b['percent']['mean'],'ci_low':b['percent']['low'],'ci_high':b['percent']['high'],
            'absolute_hv_vs_base':b['absolute']['mean'],'rule_reference':ref,'percent_vs_rule':h['percent']['mean'],
            'rule_ci_low':h['percent']['low'],'rule_ci_high':h['percent']['high'],
            'wins_vs_base':b['wins'],'ties_vs_base':b['ties'],'losses_vs_base':b['losses'],
            'wins_vs_rule':h['wins'],'ties_vs_rule':h['ties'],'losses_vs_rule':h['losses'],'mean_seconds':seconds})
    for c in 'ICM':
        a=result[c+'_RULE']['vs_base'];b=result[c+'_LEARN']['vs_rule']
        # No epsilon denominators; use the explicitly reported absolute metric when required.
        metric='absolute' if a['reference_zero_pairs'] or b['reference_zero_pairs'] else 'percent'
        acceptance[c]={'metric':metric,'both_point_estimates_positive':a[metric]['mean']>0 and b[metric]['mean']>0,
            'both_95ci_lower_bounds_positive':a[metric]['low']>0 and b[metric]['low']>0,
            'gate':read_json(out/'models'/c/'BUNDLE.json')['gate']}
    per_run=[{k:r[k] for k in ('instance','seed','arm','final_hypervolume','training_normalization_hv','optimisation_seconds','real_objective_evaluation_count')} for r in rows]
    convergence=[]
    for m in METHODS:
        for n in settings['milestones']:
            vals=[p['hv'] for r in rows if r['arm']==m for p in r['milestone_hv'] if p['evaluation']==n]
            convergence.append({'method':m,'evaluation':n,'mean_hv_training_normalization':float(np.mean(vals)) if vals else None,'runs':len(vals)})
    csv_write(out/'RESULT_TABLE.csv',table);csv_write(out/'PER_RUN.csv',per_run);csv_write(out/'CONVERGENCE.csv',convergence)
    write_json(out/'INSTANCE_BOUNDS.json',bounds)
    quick={'completed':True,'reference_method':'BASE','design':settings,'methods':result,'acceptance':acceptance,
        'all_components_met_reliable_gain':all(x['both_95ci_lower_bounds_positive'] for x in acceptance.values()),
        'qualification':'S11 development panel only. Interval conditions on selected rules/models and these generated instances. No guarantee, final test or population-wide claim.',
        'next_action':'review these fixed results; no automatic retraining, PPO or full grid',
        'new_ppo_updates':0,'final_test_opened':False,'searches':len(rows),'search_evaluations':sum(r['real_objective_evaluation_count'] for r in rows)}
    write_json(out/'QUICK_RESULT.json',quick);write_json(out/'PAIRED_RUNS.json',rows)
    return quick
