#!/usr/bin/env python3
"""Shared step-HV supervision from existing 200/20 S0 feedback; no searches."""
from joint_common import *
from shared_model import *
import argparse,math,time,fcntl
from collections import defaultdict

PROTOCOLS={'I':'i-k8-scale-data-v1','C':'c-k8-scale-data-v1','M':'m-prescreen-scale-data-v1'}
def discover(diag,c,explicit=None):
 if explicit:return explicit.resolve()
 matches=[]
 for path in diag.glob('*/PLAN.json'):
  if not (path.parent/'COLLECTION_COMPLETE').exists():continue
  if read_json(path).get('protocol')==PROTOCOLS[c]:
   if c=='M' and digest(path)!=read_json(PACKAGE/'assets/M/BUNDLE.json')['collection_plan_sha256']:continue
   matches.append(path.parent.resolve())
 if len(matches)!=1:raise ValueError(f'{c}: expected one completed collection, found {matches}; supply --{c.lower()}-collection')
 return matches[0]
def load_collections(folders):
 allrows={};files={};identities={};normalization_hash=digest(PACKAGE/'assets/reward_normalization.json')
 for c,folder in folders.items():
  plan=read_json(folder/'PLAN.json');quick=read_json(folder/'QUICK_RESULT.json')
  if (plan['protocol']!=PROTOCOLS[c] or plan['train_count']!=200 or plan['validation_count']!=20
      or not (folder/'COLLECTION_COMPLETE').exists()):raise ValueError('Incomplete historical collection: '+str(folder))
  if plan['input_sha256'][plan['normalization']]!=normalization_hash:raise ValueError('Label normalization differs from frozen search')
  records=plan['instances'];rows=[];identity={}
  for item in records:
   n=item['instance_id'];role=item['development_role']
   if role not in ('train','validation'):raise ValueError('Only training/validation labels allowed')
   file=folder/'runs'/n/'FEEDBACK.json';complete=read_json(file.parent/'RUN_COMPLETE.json');feedback=read_json(file)['rows']
   if complete['feedback_sha256']!=digest(file) or complete['feedback_state_count']!=len(feedback) or not feedback:
    raise ValueError('Feedback checksum/count mismatch: '+str(file))
   ipath=Path(plan['generated_instances'])/(n+'.json');info,_=read_instance(ipath)
   if digest(ipath)!=plan['input_sha256'][str(ipath)] or info['development_role']!=role or info['configuration']!='S0':raise ValueError('Instance split changed')
   identity[n]=(role,info['instance_digest']);files[str(ipath)]=digest(ipath)
   for r in feedback:
    if len(r['labels'])!=8:raise ValueError('Expected K8 labels')
    rows.append({**r,'instance':n,'development_role':role})
   for p in (file,file.parent/'RUN_COMPLETE.json'):files[str(p)]=digest(p)
  if len(rows)!=quick['state_count']:raise ValueError('Aggregate state count mismatch')
  if sum(v[0]=='train' for v in identity.values())!=200 or sum(v[0]=='validation' for v in identity.values())!=20:raise ValueError('Expected 200/20 instance split')
  identities[c]=identity;allrows[c]=rows
  for p in (folder/'PLAN.json',folder/'QUICK_RESULT.json'):files[str(p)]=digest(p)
 if any(x!=identities['I'] for x in identities.values()):raise ValueError('I/C/M instance identities or roles differ')
 return allrows,files

def evaluate(c,rows,pred,scale,settings):
 y=np.stack([targets(c,r) for r in rows]);choices=[]
 lambdas=settings['gate_lambdas'] if c!='M' else [0.]
 thresholds=settings['gate_thresholds'] if c!='M' else [0.]
 for lam in lambdas:
  for threshold in thresholds:
   scores=pred.mean(0)-lam*pred.std(0);scores[:,1:]-=threshold;scores[:,0]=0
   selected=scores.argmax(1);gain=y[np.arange(len(rows)),selected];by=defaultdict(list)
   for r,g in zip(rows,gain):by[r['instance']].append(float(g))
   mean=float(np.mean([np.mean(v) for v in by.values()]))
   choices.append({'lambda':lam,'threshold_scaled':threshold,'mean_raw_gain':mean,'normalized_gain':mean/scale,
     'switch_fraction':float(np.mean(selected!=0))})
 return min(choices,key=lambda r:(-r['normalized_gain'],r['switch_fraction'],r['lambda'],r['threshold_scaled']))

def train(rows,out,settings=SETTINGS,provenance=None):
 out.mkdir(parents=True,exist_ok=True);start=time.monotonic();torch.set_num_threads(1)
 train={c:[r for r in rows[c] if r['development_role']=='train'] for c in COMPONENTS}
 valid={c:[r for r in rows[c] if r['development_role']=='validation'] for c in COMPONENTS}
 names={k:sorted({r['instance'] for r in v}) for k,v in train.items()}
 if any(set(names[c])&{r['instance'] for r in valid[c]} for c in COMPONENTS):raise ValueError('Split leakage')
 scalers={c:fit_scaler(c,train[c]) for c in COMPONENTS}
 target={c:np.stack([targets(c,r) for r in train[c]]) for c in COMPONENTS}
 scales={c:max(float(np.sqrt(np.mean(target[c][:,1:]**2))),1e-10) for c in COMPONENTS}
 dims={c:[len(scalers[c][k]['mean']) for k in ('batch','candidate','context')] for c in COMPONENTS}
 plan={'settings':settings,'files':provenance or {},'environment':{'torch':str(torch.__version__),'numpy':str(np.__version__)},'training_instances':names,
       'validation_instances':{c:sorted({r['instance'] for r in valid[c]}) for c in COMPONENTS},'dimensions':dims,
       'scalers':scalers,'target_scales':scales}
 freeze(out/'PLAN.json',plan)
 if (out/'BUNDLE.json').exists():
  b=read_json(out/'BUNDLE.json');verify({str(out/p):sha for p,sha in b['checkpoint_sha256'].items()});return b
 packed={};groups={}
 for c in COMPONENTS:
  groups[c]=defaultdict(list)
  for i,r in enumerate(train[c]):groups[c][len(r['candidate_batch_tokens'][0])].append(i)
  packed[c]={n:pack(c,[train[c][i] for i in ids],scalers[c]) for n,ids in groups[c].items()}
 for seed in settings['seeds']:
  torch.manual_seed(seed);rng=np.random.default_rng(seed);model=SharedTransformer(dims,settings['width'])
  optimizer=torch.optim.AdamW(model.parameters(),lr=settings['learning_rate'],weight_decay=settings['weight_decay'])
  for epoch in range(1,settings['epochs']+1):
   work={}
   for c in COMPONENTS:
    work[c]=[]
    for n,ids in groups[c].items():
     order_=rng.permutation(len(ids))
     work[c].extend((n,order_[i:i+settings['batch_size']]) for i in range(0,len(ids),settings['batch_size']))
    rng.shuffle(work[c])
   losses=[]
   for step in range(max(map(len,work.values()))):
    optimizer.zero_grad();terms=[]
    for c in COMPONENTS:
     n,local=work[c][step%len(work[c])];ids=[groups[c][n][i] for i in local]
     x=tuple(t[local] for t in packed[c][n]);y=torch.tensor(target[c][ids]/scales[c],dtype=torch.float32)
     terms.append(rank_loss(model(c,*x)[0],y))
    loss=sum(terms)/3.;loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();losses.append(float(loss))
   if epoch in settings['checkpoint_epochs']:
    path=out/f'seed_{seed}_epoch_{epoch}.pt';torch.save({'model':model.state_dict(),'dimensions':dims,'width':settings['width']},path)
    print(f'[JOINT-SUP] seed={seed} epoch={epoch} loss={np.mean(losses):.6g}',flush=True)
 choices=[]
 for epoch in settings['checkpoint_epochs']:
  models=[]
  for seed in settings['seeds']:
   ck=load_checkpoint(out/f'seed_{seed}_epoch_{epoch}.pt');m=SharedTransformer(dims,settings['width']);m.load_state_dict(ck['model']);models.append(m)
  gates={c:evaluate(c,valid[c],predictions(models,c,valid[c],scalers[c]),scales[c],settings) for c in COMPONENTS}
  score=float(np.mean([g['normalized_gain'] for g in gates.values()]));choices.append({'epoch':epoch,'score':score,'gates':gates})
 best=min(choices,key=lambda x:(-x['score'],x['epoch']))
 checkpoints=[f'seed_{seed}_epoch_{best["epoch"]}.pt' for seed in settings['seeds']]
 bundle={**plan,'protocol':PROTOCOL,'checkpoints':checkpoints,'checkpoint_sha256':{p:digest(out/p) for p in checkpoints},
    'selected':best,'training_states':{c:len(train[c]) for c in COMPONENTS},'validation_states':{c:len(valid[c]) for c in COMPONENTS},
    'normalization_sha256':digest(PACKAGE/'assets/reward_normalization.json'),
    'new_objective_evaluations':0,'new_ppo_updates':0,'seconds':time.monotonic()-start}
 write_json(out/'VALIDATION.json',{'choices':choices,'selected':best});write_json(out/'BUNDLE.json',bundle)
 print('[JOINT-SUP] COMPLETE '+json.dumps({'selected':best,'states':bundle['training_states'],'new_objective_evaluations':0}),flush=True)
 return bundle

def main():
 p=argparse.ArgumentParser();p.add_argument('--diagnostics',type=Path,default=ROOT.parent/'transformer_ppo_runs/diagnostics')
 p.add_argument('--out',type=Path,required=True)
 for c in COMPONENTS:p.add_argument('--'+c.lower()+'-collection',type=Path)
 a=p.parse_args();a.out=a.out.resolve();a.out.mkdir(parents=True,exist_ok=True)
 with (a.out/'.train.lock').open('a+') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  folders={c:discover(a.diagnostics,c,getattr(a,c.lower()+'_collection')) for c in COMPONENTS}
  print('[JOINT-SUP] collections='+json.dumps({k:str(v) for k,v in folders.items()}),flush=True)
  rows,files=load_collections(folders);train(rows,a.out,provenance={**files,**sources()})
if __name__=='__main__':main()
