"""One Transformer backbone for I/C/M, with task input adapters and identity."""
from joint_common import *
from torch import nn
from collections import defaultdict
from i_k8_surrogate import state_context as ic_context
from crossover import model_order as c_order
from network import state_context as m_context

def order(component,row):
 return (1,0,2,3,4,5,6,7) if component=='I' else (c_order(row) if component=='C' else tuple(range(8)))
def features(component,row):
 o=order(component,row)
 return {'batch':np.asarray(row['candidate_batch_tokens'],dtype=float)[list(o)],
         'candidate':np.asarray(row['candidate_features'],dtype=float)[list(o)],
         'context':(m_context(row) if component=='M' else ic_context(row))}
def targets(component,row):
 o=order(component,row);q=np.array([r['hv_contribution'] for r in row['labels']],dtype=float)[list(o)]
 if not np.isfinite(q).all():raise ValueError('Nonfinite labels')
 return q-q[0]
def fit_scaler(c,rows):
 f=[features(c,r) for r in rows];result={'clip':10.}
 for k in ('batch','candidate','context'):
  x=np.concatenate([v[k].reshape(-1,v[k].shape[-1]) for v in f]);sd=x.std(0)
  result[k]={'mean':x.mean(0).tolist(),'std':np.where(sd>1e-8,sd,1.).tolist()}
 return result
def pack(c,rows,scaler):
 f=[features(c,r) for r in rows];values=[]
 for k in ('batch','candidate','context'):
  x=np.stack([v[k] for v in f]);x=(x-scaler[k]['mean'])/scaler[k]['std']
  if not np.isfinite(x).all():raise ValueError('Nonfinite model input')
  values.append(torch.tensor(np.clip(x,-scaler['clip'],scaler['clip']),dtype=torch.float32))
 return tuple(values)

class SharedTransformer(nn.Module):
 def __init__(self,dimensions,width=16):
  super().__init__();self.dimensions=dimensions;self.width=width
  self.batch_projection=nn.ModuleDict({c:nn.Linear(dimensions[c][0],width) for c in COMPONENTS})
  self.candidate_projection=nn.ModuleDict({c:nn.Linear(width+dimensions[c][1]+dimensions[c][2]+1,width) for c in COMPONENTS})
  self.task=nn.Embedding(3,width)
  def encoder():return nn.TransformerEncoder(nn.TransformerEncoderLayer(width,4,width*2,dropout=0.,activation='gelu',batch_first=True),1)
  self.batch_encoder=encoder();self.candidate_encoder=encoder();self.gain_head=nn.Linear(width,1)
  self.value_head=nn.Linear(width,1)
  nn.init.zeros_(self.value_head.weight);nn.init.zeros_(self.value_head.bias)
 def forward(self,c,batch,candidate,context):
  b,k,n,f=batch.shape
  h=self.batch_encoder(self.batch_projection[c](batch.reshape(b*k,n,f))).mean(1).reshape(b,k,-1)
  marker=torch.zeros(b,k,1,device=batch.device);marker[:,0]=1
  h=self.candidate_projection[c](torch.cat([h,candidate,context[:,None,:].expand(-1,k,-1),marker],-1))
  h=self.candidate_encoder(h+self.task.weight[COMPONENTS.index(c)])
  raw=self.gain_head(h).squeeze(-1)
  scores=torch.cat([torch.zeros_like(raw[:,:1]),raw[:,1:]],1)
  return scores,self.value_head(h.mean(1)).squeeze(-1)
def rank_loss(pred,target):
 delta=pred[:,:,None]-pred[:,None,:];truth=target[:,:,None]-target[:,None,:]
 mask=torch.triu(torch.ones((8,8),dtype=torch.bool,device=pred.device),diagonal=1)[None]&(truth.abs()>1e-6)
 if not mask.any():return pred.sum()*0
 w=truth[mask].abs().clamp(.1,5.)
 return (w*nn.functional.softplus(-truth[mask].sign()*delta[mask])).sum()/w.sum()
def predictions(models,c,rows,scaler):
 groups=defaultdict(list);pred=np.empty((len(models),len(rows),8))
 for i,r in enumerate(rows):groups[len(r['candidate_batch_tokens'][0])].append(i)
 with torch.no_grad():
  for ids in groups.values():
   for start in range(0,len(ids),64):
    chunk=ids[start:start+64];x=pack(c,[rows[i] for i in chunk],scaler)
    for j,m in enumerate(models):m.eval();pred[j,chunk]=m(c,*x)[0].numpy()
 return pred
