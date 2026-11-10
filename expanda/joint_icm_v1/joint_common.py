"""Isolated shared-ICM experiment; historical formal package is read-only."""
import os,sys,hashlib,json,inspect
from pathlib import Path
ROOT=Path(__file__).resolve().parent
PACKAGE=ROOT.parent/'formal_icm_s0_v1'
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
os.environ['MPLBACKEND']='Agg'
sys.path.insert(0,str(PACKAGE/'src/support'));sys.path.insert(0,str(PACKAGE/'src'))
from runtime_common import read_json,write_json,digest,_normalizer,paired_summary
from transformer_ppo.instance_catalog import canonical_digest,read_instance
import torch
import numpy as np
COMPONENTS=('I','C','M')
METHOD='ICM_SHARED_SUPERVISED'
PROTOCOL='icm-shared-supervised-v1'
SETTINGS={'protocol':PROTOCOL,'epochs':30,'checkpoint_epochs':[10,20,30],
 'seeds':[970001,970002,970003],'width':16,'batch_size':64,'learning_rate':.0003,'weight_decay':.0001,
 'loss':'equal-component mean of magnitude-weighted within-state pairwise logistic ranking',
 'gate_lambdas':[0.,.5,1.,2.],'gate_thresholds':[0.,.05,.10,.20],
 'terminal_training_enabled':False,'new_objective_evaluations':0,
 'feature_contract':'historical component features and progress definitions preserved; task-specific projections and scalers; shared encoders and gain head',
 'selection':'train/validation only; shared epoch maximizes equal-component validation residual divided by training RMS label scale; ties earliest epoch',
 'scope':'S0; existing 200 training and 20 validation instances; historical B400 labels; deploy at pop250/max200gens/B50000; no claim that labels cover late B50000 states'}
def freeze(path,data):
 if path.exists() and read_json(path)!=data:raise ValueError('Frozen inputs changed: '+str(path))
 write_json(path,data)
def load_checkpoint(path):
 kw={'weights_only':True} if 'weights_only' in inspect.signature(torch.load).parameters else {}
 return torch.load(path,map_location='cpu',**kw)
def verify(files):
 for p,sha in files.items():
  if digest(Path(p))!=sha:raise ValueError('Frozen file changed: '+p)
def sources():
 paths=list(ROOT.glob('*.py'))+list((PACKAGE/'src').rglob('*.py'))+[PACKAGE/'src/data/border_crossing_events.csv',PACKAGE/'assets/data/data_expanded.xlsx',PACKAGE/'assets/reward_normalization.json']
 return {str(p.resolve()):digest(p) for p in paths}
