#!/usr/bin/env python3
"""Prepared only. Explicit --execute is required; never called by the default pipeline.

Complete-episode PPO, gamma=1, terminal normalized HV only. No one-step shaping.
Shared supervised ensemble initializes actor and shared critic heads. Validation
includes update zero; a worse trained checkpoint is never forced into selection.
"""
from joint_common import *
from shared_policy import SharedPolicy
from shared_search import run_one
import argparse,fcntl,subprocess,time

UPDATES=(0,4,8)
CONFIG={'protocol':'shared-icm-terminal-ppo-v1','updates':8,'episodes_per_update':4,'gamma':1.,
 'reward':'final feasible-front HV under frozen training normalization; no intermediate reward or clipping',
 'epochs_per_update':2,'minibatch':64,'clip_ratio':.2,'learning_rate':.00003,
 'entropy_coefficient':.001,'value_coefficient':.5,'max_grad_norm':1.,'target_kl':.02,
 'validation_checkpoints':list(UPDATES),'validation_instances':20,'validation_seeds_per_instance':1,
 'population':250,'budget':50000,'max_generations':200,'scenario_count':100,
 'selection':'maximum paired mean final HV difference vs update zero on same validation instances and seed; ties choose earliest',
 'maximum_training_searches':32,'maximum_validation_searches':60,
 'automatic_execution':False}

def records(folder,reward):
 data=np.load(folder/'TRAJECTORY.npz',allow_pickle=False);out=[]
 for c in COMPONENTS:
  if c+'_action' not in data:continue
  for i,a in enumerate(data[c+'_action']):
   out.append({'component':c,'x':tuple(data[c+'_'+k][i] for k in ('batch','candidate','context')),
      'action':int(a),'logp':float(data[c+'_logp'][i]),'value':float(data[c+'_value'][i]),'return':reward})
 return out

def ppo_update(policy,episodes,settings=CONFIG):
 # Episode-equal weighting prevents long searches from receiving more weight merely for more decisions.
 rows=[]
 for events in episodes:
  if not events:raise ValueError('Episode has no recorded decisions')
  for e in events:rows.append({**e,'weight':1./len(events)/len(episodes)})
 optimizer=torch.optim.AdamW([p for m in policy.models for p in m.parameters()],lr=settings['learning_rate'],weight_decay=0.)
 groups={}
 for i,r in enumerate(rows):groups.setdefault((r['component'],r['x'][0].shape[1]),[]).append(i)
 def batch(ids):
  part=[rows[i] for i in ids];c=part[0]['component'];x=tuple(torch.tensor(np.stack([r['x'][j] for r in part]),dtype=torch.float32) for j in range(3))
  return c,x,part
 max_error=0.
 with torch.no_grad():
  for ids in groups.values():
   for start in range(0,len(ids),settings['minibatch']):
    c,x,part=batch(ids[start:start+settings['minibatch']]);dist,_=policy.distribution(c,x)
    actual=dist.log_prob(torch.tensor([r['action'] for r in part]));old=torch.tensor([r['logp'] for r in part])
    max_error=max(max_error,float((actual-old).abs().max()))
 if max_error>2e-5:raise ValueError(f'On-policy action probability mismatch: {max_error}')
 rng=np.random.default_rng(973000);steps=0;kl=0.;early=False
 for epoch in range(settings['epochs_per_update']):
  work=[]
  for ids in groups.values():
   perm=rng.permutation(ids);work.extend(perm[i:i+settings['minibatch']].tolist() for i in range(0,len(perm),settings['minibatch']))
  rng.shuffle(work)
  for ids in work:
   c,x,part=batch(ids);dist,value=policy.distribution(c,x);actions=torch.tensor([r['action'] for r in part])
   old=torch.tensor([r['logp'] for r in part]);ret=torch.tensor([r['return'] for r in part],dtype=torch.float32)
   advantage=ret-torch.tensor([r['value'] for r in part],dtype=torch.float32)
   weights=torch.tensor([r['weight'] for r in part],dtype=torch.float32)*len(rows)
   lp=dist.log_prob(actions);ratio=(lp-old).exp();kl=float(((ratio-1)-(lp-old)).mean().detach())
   if kl>settings['target_kl']:early=True;break
   actor=-torch.minimum(ratio*advantage,ratio.clamp(1-settings['clip_ratio'],1+settings['clip_ratio'])*advantage)
   loss=(weights*(actor+settings['value_coefficient']*(value-ret)**2-settings['entropy_coefficient']*dist.entropy())).mean()
   optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_([p for m in policy.models for p in m.parameters()],settings['max_grad_norm']);optimizer.step();steps+=1
  if early:break
 return {'optimizer_steps':steps,'approximate_kl':kl,'kl_early_stop':early,'on_policy_log_probability_max_error':max_error,
  'complete_episodes':len(episodes),'actions':len(rows),'terminal_returns':[e[0]['return'] for e in episodes]}

def snapshot(policy,folder,metrics=None):
 final=folder;folder=final.parent/(final.name+f'_building_{time.time_ns()}');folder.mkdir(parents=True);b=dict(policy.bundle);paths=[]
 for i,m in enumerate(policy.models):
  p=folder/f'member_{i}.pt';torch.save({'model':m.state_dict(),'dimensions':m.dimensions,'width':m.width},p);paths.append(p.name)
 b['checkpoints']=paths;b['checkpoint_sha256']={p:digest(folder/p) for p in paths};write_json(folder/'BUNDLE.json',b)
 if metrics is not None:write_json(folder/'UPDATE.json',metrics)
 folder.rename(final)

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--model',type=Path,required=True);p.add_argument('--instances',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
 p.add_argument('--execute',action='store_true');p.add_argument('--rollout-worker',action='store_true');p.add_argument('--instance',type=Path);p.add_argument('--seed',type=int);p.add_argument('--sample',action='store_true');a=p.parse_args()
 if a.rollout_worker:
  info,_=read_instance(a.instance)
  if info['development_role']!=('train' if a.sample else 'validation'):raise ValueError('Forbidden rollout split')
  run_one(a.instance,a.seed,a.model,a.out,'terminal-prepared-v1',stochastic=a.sample,record=a.sample);return
 if not a.execute:
  print(json.dumps({'prepared':True,'started':False,'config':CONFIG,'message':'No training or search started. Explicit execute and source training/validation catalog required.'},indent=2));return
 a.out=a.out.resolve();a.out.mkdir(parents=True,exist_ok=True)
 with (a.out/'.pipeline.lock').open('a+') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  b=read_json(a.model/'BUNDLE.json');train=sorted(set(b['training_instances']['I']));valid=sorted(set(b['validation_instances']['I']))
  if len(train)!=200 or len(valid)!=20 or set(train)&set(valid):raise ValueError('Expected frozen 200/20 split')
  inputfiles={str(a.model/'BUNDLE.json'):digest(a.model/'BUNDLE.json'),**sources()}
  for n in train+valid:
   f=a.instances/(n+'.json');info,_=read_instance(f)
   if info['configuration']!='S0' or info['development_role']!=('train' if n in train else 'validation'):raise ValueError('Wrong terminal split')
   inputfiles[str(f)]=digest(f)
  inputfiles.update({str(a.model/name):sha for name,sha in b['checkpoint_sha256'].items()})
  plan={'config':CONFIG,'files':inputfiles};freeze(a.out/'PLAN.json',plan)
  if (a.out/'QUICK_RESULT.json').exists():print('Already complete');return
  def rollout(n,seed,model,folder,sample):
   if (folder/'RUN_COMPLETE.json').exists():
    r=read_json(folder/'RUN_COMPLETE.json')
    if (r['instance'],r['algorithm_seed'],r['stochastic'],r.get('model_bundle_sha256'))!=(n,seed,sample,digest(model/'BUNDLE.json')):raise ValueError('Resume identity mismatch')
    return r
   if folder.exists():
    q=a.out/'interrupted'/f'{folder.name}_{time.time_ns()}';q.parent.mkdir(exist_ok=True);folder.rename(q)
   logfile=folder.with_suffix('.log');logfile.parent.mkdir(parents=True,exist_ok=True)
   cmd=[sys.executable,'-u',str(Path(__file__)),'--rollout-worker','--model',str(model),'--instances',str(a.instances),'--out',str(folder),'--instance',str(a.instances/(n+'.json')),'--seed',str(seed)]+(['--sample'] if sample else [])
   with logfile.open('w') as h:subprocess.run(cmd,stdout=h,stderr=subprocess.STDOUT,check=True,timeout=14400)
   return read_json(folder/'RUN_COMPLETE.json')
  first=a.out/'checkpoints/update_000'
  if not first.exists():snapshot(SharedPolicy(a.model),first)
  history=[];validation=[]
  for update in range(9):
   checkpoint=a.out/f'checkpoints/update_{update:03d}'
   if update:
    prev=a.out/f'checkpoints/update_{update-1:03d}';episode_events=[]
    for j in range(4):
     n=train[(update-1)*4+j];seed=974000+(update-1)*4+j;folder=a.out/'rollouts'/f'u{update:03d}_{j}_{n}'
     r=rollout(n,seed,prev,folder,True);episode_events.append(records(folder,r['final_hypervolume']))
    if not (checkpoint/'BUNDLE.json').exists():
     policy=SharedPolicy(prev);torch.manual_seed(975000+update);metrics=ppo_update(policy,episode_events);snapshot(policy,checkpoint,metrics)
    history.append({'update':update,**read_json(checkpoint/'UPDATE.json')})
   if update in UPDATES:
    hv=[]
    for j,n in enumerate(valid):
     r=rollout(n,976001,checkpoint,a.out/'validation'/f'u{update:03d}_{n}',False);hv.append(r['final_hypervolume'])
    baseline=validation[0]['hv'] if validation else hv
    validation.append({'update':update,'hv':hv,'mean_difference_vs_initial':float(np.mean(np.asarray(hv)-baseline))})
   write_json(a.out/'TRAINING_PROGRESS.json',{'completed_update':update,'training':history,'validation':validation})
  selected=max(validation,key=lambda r:(r['mean_difference_vs_initial'],-r['update']))
  verify(inputfiles)
  write_json(a.out/'QUICK_RESULT.json',{'completed':True,'selected':selected,'training':history,'validation':validation,
    'training_searches':32,'validation_searches':60,'search_evaluations':4600000,'final_test_opened':False,
    'next_action':'freeze selected checkpoint; no automatic formal evaluation'})
if __name__=='__main__':main()
