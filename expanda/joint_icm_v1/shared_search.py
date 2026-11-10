#!/usr/bin/env python3
"""Frozen shared-supervised search: same instances/seeds/engine as original formal grid."""
from joint_common import *
from runtime_common import _normalizer
from shared_policy import SharedPolicy,SharedController,Adapter
from mutation import ScaleSearchPolicy
from transformer_ppo.archive import RewardArchive,RewardConfig
from transformer_ppo.episode import run_episode
import argparse,fcntl,subprocess,time,signal,csv,math

def run_one(instance,seed,model,out,sha,budget=50000,population=250,generations=200,stochastic=False,record=False,reference_config=None):
 torch.set_num_threads(1);shared=SharedPolicy(model,stochastic,seed+81000000,record)
 info,_=read_instance(instance);policy=ScaleSearchPolicy(info['instance_id'],seed,'TRANSFORMER_K8_SIZE200',Adapter(shared,'M'))
 def factory(*args,**kwargs):
  norm=kwargs.pop('normalizer');return SharedController(*args,shared=shared,trainer=policy,
   archive=RewardArchive(norm),reward_config=RewardConfig(),generations=generations,pop_size=population,graph_observation={},**kwargs)
 result=run_episode(PACKAGE/'assets/data/data_expanded.xlsx',instance,out,None,_normalizer(PACKAGE/'assets/reward_normalization.json'),
  algorithm_seed=seed,evaluation_budget=budget,pop_size=population,generations=generations,evaluation_mode='ccp',component='BASE',
  scenario_count=100,ccp_confidence=.9,uncertainty_profile='transport_border_saa_v1',
  inference_controller_factories={'BASE':factory},method_override=METHOD)
 if result['real_objective_evaluation_count']!=budget:raise ValueError('Search did not use required budget; record incomplete attempt, do not silently change generation cap')
 if result['combined_hidden_candidate_objective_evaluations']!=0:raise ValueError('Hidden candidate evaluations')
 if record:shared.save_events(out/'TRAJECTORY.npz')
 cfg=read_json(out/'configuration.json')
 if reference_config is not None:
  reference=read_json(reference_config)
  for k in ('scenario_digest','instance_digest','normalization','seeds'):
   if reference[k]!=cfg[k]:raise ValueError('Formal pairing mismatch: '+k)
 result.update(instance=info['instance_id'],strategy=METHOD,algorithm_seed=seed,plan_sha256=sha,
    scenario_digest=cfg['scenario_digest'],model_bundle_sha256=digest(Path(model)/'BUNDLE.json'),new_training_steps=0,new_ppo_updates=0,stochastic=stochastic)
 write_json(out/'RUN_COMPLETE.json',result);return result

def prepare(source,model,out):
 old=read_json(source/'PLAN.json');cfg=old['settings'];bundle=read_json(model/'BUNDLE.json')
 if (cfg['population_size'],cfg['maximum_generations'],cfg['evaluation_budget'],old['runs'])!=(250,200,50000,2400):raise ValueError('Wrong original formal plan')
 if not (source/'SEARCH_COMPLETE').exists():raise ValueError('Complete original 2400 searches first; do not select shared checkpoints on these instances')
 if (str(torch.__version__),str(np.__version__))!=(old['torch_version'],old['numpy_version']):raise ValueError('Use same Torch/NumPy as original formal searches')
 verify(old['input_sha256']);verify({str(model/p):s for p,s in bundle['checkpoint_sha256'].items()})
 overlap=set(old['instances'])&set(sum(bundle['training_instances'].values(),[])+sum(bundle['validation_instances'].values(),[]))
 if overlap:raise ValueError('Training/validation overlap with formal screen')
 files={**sources(),str(source/'PLAN.json'):digest(source/'PLAN.json'),str(model/'BUNDLE.json'):digest(model/'BUNDLE.json')}
 files.update({str(model/p):s for p,s in bundle['checkpoint_sha256'].items()})
 for n in old['instances']:
  for seed in old['algorithm_seeds']:
   for method in old['methods']:
    folder=source/'runs'/n/f'seed_{seed}'/method
    for filename in ('RUN_COMPLETE.json','configuration.json','final_feasible_nondominated.json'):
     path=folder/filename;files[str(path)]=digest(path)
 plan={'protocol':'shared-icm-formal-extension-v1','source':str(source),'model':str(model),'settings':cfg,
       'instances':old['instances'],'seeds':old['algorithm_seeds'],'original_plan_sha':digest(source/'PLAN.json'),
       'files':files,'new_runs':300,'new_ppo_updates':0,'qualification':'S0 development formal-parameter comparison; frozen historical B400 features retained; not independent final test'}
 plan['sha256']=canonical_digest(plan);freeze(out/'PLAN.json',plan);return plan

def summarize(out,plan):
 source=Path(plan['source']);comparisons={};allrows=[]
 for n in plan['instances']:
  for seed in plan['seeds']:
   folder=out/'runs'/n/f'seed_{seed}'/METHOD;row=read_json(folder/'RUN_COMPLETE.json')
   if row['plan_sha256']!=plan['sha256'] or row['real_objective_evaluation_count']!=50000:raise ValueError('Invalid shared completion')
   for ref in ('BASE','I_C_M_K8_TRANSFORMER'):
    prev=read_json(source/'runs'/n/f'seed_{seed}'/ref/'RUN_COMPLETE.json')
    if prev['plan_sha256']!=plan['original_plan_sha'] or prev['scenario_digest']!=row['scenario_digest']:raise ValueError('Unpaired source result')
    b=prev['final_hypervolume'];h=row['final_hypervolume']
    allrows.append({'instance':n,'seed':seed,'reference':ref,'hv':h,'reference_hv':b,'absolute':h-b,'percent':100*(h/b-1) if b>0 else None,'seconds':row['optimisation_seconds']})
 for i,ref in enumerate(('BASE','I_C_M_K8_TRANSFORMER')):
  part=[r for r in allrows if r['reference']==ref];p=paired_summary(part,'percent',971000+i)
  if p['undefined_pairs']:p.update(mean=None,hierarchical_bootstrap_95ci_low=None,hierarchical_bootstrap_95ci_high=None)
  comparisons[ref]={'percent':p,'absolute':paired_summary(part,'absolute',971100+i)}
 result={'completed':True,'new_runs':300,'new_evaluations':15000000,'comparisons':comparisons,
   'normalization':'original frozen training bounds; do not mix with post-hoc per-instance union HV',
   'oos_completed':False,'new_ppo_updates':0,'mean_seconds':float(np.mean([r['seconds'] for r in allrows])),
   'qualification':plan['qualification']}
 write_json(out/'QUICK_RESULT.json',result)
 with (out/'PAIRED_RESULTS.csv').open('w',newline='',encoding='utf-8-sig') as f:
  w=csv.DictWriter(f,fieldnames=list(allrows[0]));w.writeheader();w.writerows(allrows)
 return result

def execute(out,plan):
 jobs=[(n,s) for n in plan['instances'] for s in plan['seeds']];active={};done=0;last=0;begin=time.monotonic()
 def stopped(*_):raise KeyboardInterrupt('Stopped; resume same command')
 previous={sig:signal.signal(sig,stopped) for sig in (signal.SIGTERM,signal.SIGINT)}
 try:
  while jobs or active:
   while jobs and len(active)<plan['settings']['workers']:
    n,s=jobs.pop(0);folder=out/'runs'/n/f'seed_{s}'/METHOD
    if (folder/'RUN_COMPLETE.json').exists():
     r=read_json(folder/'RUN_COMPLETE.json')
     if r['plan_sha256']!=plan['sha256'] or r['algorithm_seed']!=s or r['instance']!=n or r['real_objective_evaluation_count']!=50000:raise ValueError('Invalid resumed result')
     done+=1;continue
    if folder.exists():
     q=out/'interrupted'/f'{n}_{s}_{time.time_ns()}';q.parent.mkdir(exist_ok=True);folder.rename(q)
    log=out/'logs'/f'{n}_{s}_{time.time_ns()}.log';log.parent.mkdir(exist_ok=True);h=log.open('w')
    p=subprocess.Popen([sys.executable,'-u',str(Path(__file__)),'--worker','--out',str(out),'--instance',n,'--seed',str(s)],stdout=h,stderr=subprocess.STDOUT)
    active[p.pid]=(p,h,time.monotonic(),log)
    print(f'[JOINT-SEARCH] start {n}/{s} PID={p.pid}',flush=True)
   for pid,(p,h,t,log) in list(active.items()):
    if time.monotonic()-t>plan['settings']['per_run_timeout_seconds']:raise TimeoutError('Worker timeout: '+str(log))
    if p.poll() is None:continue
    h.close();del active[pid]
    if p.returncode:raise RuntimeError(str(log)+'\n'+log.read_text(errors='replace')[-4000:])
    done+=1;print(f'[JOINT-SEARCH] complete {done}/300',flush=True)
   if time.monotonic()-last>30:
    status={'completed':False,'finished':done,'total':300,'running':len(active),'seconds':time.monotonic()-begin};write_json(out/'STATUS.json',status);print('[JOINT-SEARCH] '+json.dumps(status),flush=True);last=time.monotonic()
   if active:time.sleep(.25)
  verify(plan['files']);summarize(out,plan)
  from formal_report import union_report
  union_report(out,plan)
  write_json(out/'STATUS.json',{'completed':True,'finished':300});print('[JOINT-SEARCH] COMPLETE; PPO not started',flush=True)
 except BaseException as exc:
  write_json(out/'STATUS.json',{'completed':False,'finished':done,'error':str(exc)})
  raise
 finally:
  for p,h,*_ in active.values():
   if p.poll() is None:p.terminate()
  for p,h,*_ in active.values():
   try:p.wait(timeout=10)
   except subprocess.TimeoutExpired:p.kill();p.wait()
   h.close()
  for sig,v in previous.items():signal.signal(sig,v)

def main():
 p=argparse.ArgumentParser();p.add_argument('--source',type=Path,default=PACKAGE/'formal_runs');p.add_argument('--model',type=Path);p.add_argument('--out',type=Path,required=True)
 p.add_argument('--worker',action='store_true');p.add_argument('--instance');p.add_argument('--seed',type=int);a=p.parse_args();out=a.out.resolve()
 if a.worker:
  plan=read_json(out/'PLAN.json');verify(plan['files'])
  if a.instance not in plan['instances'] or a.seed not in plan['seeds']:raise ValueError('Worker outside plan')
  row=run_one(PACKAGE/'assets/formal_instances'/f'{a.instance}.json',a.seed,Path(plan['model']),out/'runs'/a.instance/f'seed_{a.seed}'/METHOD,plan['sha256'],reference_config=Path(plan['source'])/'runs'/a.instance/f'seed_{a.seed}'/'BASE/configuration.json')
  reference=read_json(Path(plan['source'])/'runs'/a.instance/f'seed_{a.seed}'/'BASE/configuration.json');cfg=read_json(out/'runs'/a.instance/f'seed_{a.seed}'/METHOD/'configuration.json')
  for k in ('scenario_digest','instance_digest','normalization','seeds'):
   if reference[k]!=cfg[k]:raise ValueError('Formal pairing mismatch: '+k)
  return
 out.mkdir(parents=True,exist_ok=True)
 with (out/'.pipeline.lock').open('a+') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);plan=prepare(a.source.resolve(),a.model.resolve(),out);execute(out,plan)
if __name__=='__main__':main()
