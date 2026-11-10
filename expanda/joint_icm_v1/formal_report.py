"""Post-hoc per-instance union normalization for original eight arms plus shared ICM."""
from joint_common import *
from transformer_ppo.archive import exact_hv_3d,nondominated
import csv,statistics,math

def union_report(out,plan):
 source=Path(plan['source']);old=read_json(source/'PLAN.json');methods=old['methods']+[METHOD];rows=[]
 for n in plan['instances']:
  for seed in plan['seeds']:
   scenarios=set()
   for m in methods:
    folder=(out if m==METHOD else source)/'runs'/n/f'seed_{seed}'/m
    r=read_json(folder/'RUN_COMPLETE.json');cfg=read_json(folder/'configuration.json');front=read_json(folder/'final_feasible_nondominated.json')
    expected=plan['sha256'] if m==METHOD else plan['original_plan_sha']
    if (not r['completed'] or r['plan_sha256']!=expected or r['strategy']!=m or r['instance']!=n or r['algorithm_seed']!=seed or r['real_objective_evaluation_count']!=50000):raise ValueError('Incomplete formal grid')
    scenarios.add(cfg['scenario_digest']);points=[]
    for f in front:
     if f['training_feasible'] is not True or f['source_run']!=n:raise ValueError('Wrong front identity/feasibility')
     p=tuple(float(f['optimisation_objectives'][k]) for k in ('cost','emission','makespan'))
     if not all(math.isfinite(v) for v in p):raise ValueError('Nonfinite feasible front')
     points.append(p)
    if set(nondominated(points))!=set(points):raise ValueError('Front contains dominated solutions')
    norm=cfg['normalization'];lo=np.asarray(norm['minimum']);hi=np.asarray(norm['maximum']);div=np.asarray(norm['objective_divisor'])
    replay=exact_hv_3d([((np.asarray(p)/div-lo)/(hi-lo)).tolist() for p in points],norm['reference'])
    if not math.isclose(replay,r['final_hypervolume'],rel_tol=1e-9,abs_tol=1e-11):raise ValueError('Original HV not reproducible')
    rows.append({'instance':n,'seed':seed,'method':m,'points':points,'old_hv':r['final_hypervolume'],'seconds':r['optimisation_seconds']})
   if len(scenarios)!=1:raise ValueError('Scenario pairing failed')
 bounds={}
 for n in plan['instances']:
  group=[r for r in rows if r['instance']==n];points=[p for r in group for p in r['points']]
  if not points:raise ValueError('No feasible point in instance; relative normalization undefined')
  x=np.asarray(points);lo=x.min(0);hi=x.max(0);span=hi-lo
  bounds[n]={'minimum':lo.tolist(),'maximum':hi.tolist(),'reference':[1.1]*3}
  for r in group:
   z=[np.divide(np.asarray(p)-lo,span,out=np.zeros(3),where=span!=0).tolist() for p in r['points']]
   r['hv']=exact_hv_3d(z,(1.1,1.1,1.1))
 index={(r['instance'],r['seed'],r['method']):r for r in rows};table=[];comparisons={}
 refs={'I_K8_RESIDUAL':'I_DIRECTED_BALANCED','C_K8_RESIDUAL':'C_K8_SIMPLE','M_K8_TRANSFORMER':'M_BALANCED',METHOD:'I_C_M_K8_TRANSFORMER'}
 for j,m in enumerate(methods):
  for ref in ['BASE']+([refs[m]] if m in refs else []):
   rr=[r for r in rows if r['method']==m];pairs=[]
   for r in rr:
    b=index[r['instance'],r['seed'],ref]['hv'];pairs.append({'instance':r['instance'],'absolute':r['hv']-b,'percent':100*(r['hv']/b-1) if b>0 else None})
   absolute=paired_summary(pairs,'absolute',978000+j);percent=paired_summary(pairs,'percent',978100+j)
   if percent['undefined_pairs']:percent.update(mean=None,hierarchical_bootstrap_95ci_low=None,hierarchical_bootstrap_95ci_high=None)
   comparisons[m+'_vs_'+ref]={'absolute':absolute,'percent':percent}
   table.append({'method':m,'reference':ref,'mean_hv':statistics.fmean(r['hv'] for r in rr),'hv_sample_std':statistics.stdev(r['hv'] for r in rr),
    'mean_percent':percent['mean'],'ci_low':percent['hierarchical_bootstrap_95ci_low'],'ci_high':percent['hierarchical_bootstrap_95ci_high'],
    'mean_absolute_difference':absolute['mean'],'wins':absolute['wins'],'ties':absolute['ties'],'losses':absolute['losses'],
    'mean_seconds':statistics.fmean(r['seconds'] for r in rr)})
 target=out/'hv_union';target.mkdir(exist_ok=True)
 write_json(target/'INSTANCE_BOUNDS.json',bounds);write_json(target/'QUICK_RESULT.json',{'completed':True,'runs':len(rows),'comparisons':comparisons,
  'qualification':'Post-hoc shared bounds from all nine methods and seeds per instance. Not training inputs. No new evaluations. SD pools instances and seeds; paired CI clusters by instance.'})
 for file,values in [('RESULT_TABLE.csv',table),('PER_RUN.csv',[{k:r[k] for k in ('instance','seed','method','old_hv','hv','seconds')} for r in rows])]:
  with (target/file).open('w',newline='',encoding='utf-8-sig') as f:
   w=csv.DictWriter(f,fieldnames=list(values[0]));w.writeheader();w.writerows(values)
 print(f'[JOINT-REPORT] complete per-instance-union HV for all {len(rows)} searches; zero new evaluations',flush=True)
 return comparisons
