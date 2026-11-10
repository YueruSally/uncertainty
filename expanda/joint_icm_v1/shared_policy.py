"""One shared ensemble is called at I, C and M; no candidate outcome inputs."""
from joint_common import *
from shared_model import *
from combined import CombinedFrozenK8Controller,create_component_directories
from initialization import InitializationK8Controller
from crossover import CrossoverK8Controller
from mutation import ScaleSearchController

class SharedPolicy:
 def __init__(self,folder,stochastic=False,seed=0,record=False):
  self.folder=Path(folder);self.bundle=read_json(self.folder/'BUNDLE.json');b=self.bundle
  if b['protocol']!=PROTOCOL:raise ValueError('Wrong shared model protocol')
  if b['normalization_sha256']!=digest(PACKAGE/'assets/reward_normalization.json'):raise ValueError('Normalization changed')
  verify({str(self.folder/p):sha for p,sha in b['checkpoint_sha256'].items()});self.models=[]
  for path in b['checkpoints']:
   ck=load_checkpoint(self.folder/path);m=SharedTransformer(ck['dimensions'],ck['width']);m.load_state_dict(ck['model']);m.eval();self.models.append(m)
  self.stochastic=stochastic;self.record=record;self.rng=np.random.default_rng(seed);self.events=[]
  self.counts={c:{'calls':0,'rule_choices':0,'switches':0,'feature_values':0,'clipped_values':0} for c in COMPONENTS}
 def logits(self,c,x):
  out=[m(c,*x) for m in self.models];pred=torch.stack([v[0] for v in out]);g=self.bundle['selected']['gates'][c]
  score=pred.mean(0)-g['lambda']*pred.std(0,unbiased=False)
  score=torch.cat([torch.zeros_like(score[:,:1]),score[:,1:]-g['threshold_scaled']],1)
  return score,torch.stack([v[1] for v in out]).mean(0)
 def distribution(self,c,x):
  scores,value=self.logits(c,x)
  # Fixed 1% exploration has full support; argmax still matches gated supervision.
  probs=.99*torch.softmax(scores,-1)+.01/8
  return torch.distributions.Categorical(probs=probs),value
 def choose(self,c,row):
  raw=features(c,row);s=self.bundle['scalers'][c];stats=self.counts[c]
  for k,x in raw.items():
   z=np.abs((x-s[k]['mean'])/s[k]['std']);stats['feature_values']+=z.size;stats['clipped_values']+=int((z>s['clip']).sum())
  x=pack(c,[row],s)
  with torch.no_grad():
   dist,value=self.distribution(c,x)
   prob=dist.probs[0].numpy().astype(float);prob/=prob.sum()
   action=int(self.rng.choice(8,p=prob)) if self.stochastic else int(dist.probs.argmax(-1).item())
   logp=float(dist.log_prob(torch.tensor([action])).item())
  stats['calls']+=1;stats['rule_choices']+=int(action==0);stats['switches']+=int(action!=0)
  if self.record:self.events.append({'component':c,'x':tuple(v[0].numpy().copy() for v in x),'action':action,'logp':logp,'value':float(value.item())})
  return order(c,row)[action]
 def save_events(self,path):
  arrays={}
  for c in COMPONENTS:
   events=[r for r in self.events if r['component']==c]
   if not events:continue
   for i,k in enumerate(('batch','candidate','context')):arrays[c+'_'+k]=np.stack([r['x'][i] for r in events])
   for k in ('action','logp','value'):arrays[c+'_'+k]=np.array([r[k] for r in events])
  np.savez_compressed(path,**arrays)

class Adapter:
 def __init__(self,policy,c):self.policy=policy;self.c=c
 def select(self,row):return self.policy.choose(self.c,row)
 def predict(self,row):
  selected=self.select(row);scores=np.zeros(8);scores[selected]=1.;return scores

class SharedController(CombinedFrozenK8Controller):
 def __init__(self,*args,shared,reliable_options,arcs,**kwargs):
  self.shared=shared;batches=tuple(kwargs['batches']);norm=kwargs['archive'].normalizer
  ScaleSearchController.__init__(self,*args,**kwargs);root=create_component_directories(self.out)
  common=dict(method=self.method,instance=self.instance,seeds=self.seeds,normalizer=norm,
      reliable_options=reliable_options,batches=batches,arcs=arcs,pool_size=8)
  self.i_controller=InitializationK8Controller(root/'I',self.run_id,self.scenario_id,self.budget,
      selection_mode='transformer',selection_policy=Adapter(shared,'I'),**common)
  self.c_controller=CrossoverK8Controller(root/'C',self.run_id,self.scenario_id,self.budget,
      selection_mode='transformer',selection_policy=Adapter(shared,'C'),defer_selected_pair_mutation=True,**common)
  self._sync_children()
 def summary(self):return {**super().summary(),'shared_policy_counts':self.shared.counts,'shared_backbone':True}
