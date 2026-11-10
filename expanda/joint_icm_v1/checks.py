"""Synthetic training and real B400 integration checks; never formal results."""
from joint_common import *
from shared_model import *
from train_shared import train,evaluate
from shared_policy import SharedPolicy,Adapter
from shared_search import run_one
from terminal_ppo import ppo_update,records
import tempfile,unittest,copy,contextlib,io

class JointChecks(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.tmp=tempfile.TemporaryDirectory();cls.root=Path(cls.tmp.name);cls.out=cls.root/'model';cls.rows={};rng=np.random.default_rng(55)
  for c in COMPONENTS:
   old=read_json(PACKAGE/'assets'/c/'BUNDLE.json');s=(old['models']['200'] if c=='M' else old)['scaler']
   d=[len(s[k]['mean']) for k in ('batch','candidate','context')];cls.rows[c]=[]
   for i in range(6):
    row={'instance':('train' if i<4 else 'valid')+str(i),'development_role':'train' if i<4 else 'validation',
      'candidate_batch_tokens':rng.normal(size=(8,3,d[0])).tolist(),'candidate_features':rng.normal(size=(8,d[1])).tolist(),
      'population_points':[[.1,.2,.3]],'slot_fraction':.2,'generation':11,'evaluation':220,
      'labels':[{'hv_contribution':float(j)/100} for j in range(8)]}
    cls.rows[c].append(row)
  settings={**SETTINGS,'epochs':1,'checkpoint_epochs':[1],'batch_size':2}
  train(cls.rows,cls.out,settings)
 @classmethod
 def tearDownClass(cls):cls.tmp.cleanup()
 def test_same_backbone_all_tasks_and_label_isolation(self):
  p=SharedPolicy(self.out);m=p.models[0]
  before=m.batch_encoder.layers[0].self_attn.in_proj_weight.detach().clone()
  optimizer=torch.optim.SGD(m.parameters(),lr=.01)
  for c in COMPONENTS:
   optimizer.zero_grad();x=pack(c,[self.rows[c][0]],p.bundle['scalers'][c]);pred,_=m(c,*x)
   self.assertEqual(pred.shape,(1,8));self.assertEqual(float(pred[0,0]),0.)
   rank_loss(pred,torch.tensor(targets(c,self.rows[c][0])[None],dtype=torch.float32)).backward()
   self.assertGreater(float(m.batch_encoder.layers[0].self_attn.in_proj_weight.grad.abs().sum()),0.);optimizer.step()
  self.assertFalse(torch.equal(before,m.batch_encoder.layers[0].self_attn.in_proj_weight))
  r=copy.deepcopy(self.rows['I'][0]);a=p.choose('I',r);r['labels']=[{'hv_contribution':999.}]*8
  self.assertEqual(a,p.choose('I',r))
 def test_ppo_probability_and_negative_return(self):
  p=SharedPolicy(self.out,True,7,True)
  for c in COMPONENTS:p.choose(c,self.rows[c][0])
  events=[{**e,'return':-.25} for e in p.events]
  stat=ppo_update(p,[events],{**__import__('terminal_ppo').CONFIG,'epochs_per_update':1})
  self.assertLess(stat['on_policy_log_probability_max_error'],2e-5);self.assertEqual(stat['terminal_returns'],[-.25])
  self.assertGreater(stat['optimizer_steps'],0)
  broken=SharedPolicy(self.out);bad=[{**e,'logp':e['logp']+.5} for e in events]
  with self.assertRaises(ValueError):ppo_update(broken,[bad])
 def test_real_icm_budget_and_all_task_calls(self):
  out=self.root/'real_run'
  with (self.root/'search.log').open('w') as f,contextlib.redirect_stdout(f):
   r=run_one(PACKAGE/'assets/search_instances/prescreen-search-S0-001.json',930001,self.out,out,'local-check',400,20,1000,True,True)
  self.assertEqual(r['real_objective_evaluation_count'],400)
  self.assertEqual(r['combined_hidden_candidate_objective_evaluations'],0)
  for c in COMPONENTS:self.assertGreater(r['shared_policy_counts'][c]['calls'],0)
  p=SharedPolicy(self.out);events=records(out,r['final_hypervolume'])
  stat=ppo_update(p,[events],{**__import__('terminal_ppo').CONFIG,'epochs_per_update':1})
  self.assertLess(stat['on_policy_log_probability_max_error'],2e-5)

 def test_shared_report_keeps_pairing_and_recomputes_all_methods(self):
  from formal_report import union_report
  source=Path('/Users/yueru/uncertainty/output/ICM_I_C_M_Complete_Reproduction_r3/raw_results')
  if not (source/'PLAN.json').exists():self.skipTest('Historical raw fixture not installed')
  old=read_json(source/'PLAN.json');staging=self.root/'report_fixture';new=self.root/'report_new';new.mkdir()
  write_json(staging/'PLAN.json',{'methods':old['methods']})
  for n in old['instances']:
   for seed in old['algorithm_seeds']:
    for m in old['methods']+[METHOD]:
     src=source/'runs'/n/f'seed_{seed}'/('BASE' if m==METHOD else m)
     dst=(new if m==METHOD else staging)/'runs'/n/f'seed_{seed}'/m;dst.mkdir(parents=True)
     for name in ('configuration.json','final_feasible_nondominated.json'):(dst/name).write_bytes((src/name).read_bytes())
     row=read_json(src/'RUN_COMPLETE.json');row.update(real_objective_evaluation_count=50000,strategy=m,plan_sha256='new' if m==METHOD else 'old')
     write_json(dst/'RUN_COMPLETE.json',row)
  result=union_report(new,{'source':str(staging),'instances':old['instances'],'seeds':old['algorithm_seeds'],'sha256':'new','original_plan_sha':'old'})
  self.assertEqual(result[METHOD+'_vs_BASE']['percent']['mean'],0.)
  self.assertEqual(result[METHOD+'_vs_BASE']['absolute']['ties'],30)

if __name__=='__main__':unittest.main()
