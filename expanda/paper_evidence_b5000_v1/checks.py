"""Synthetic evidence-export checks; no scientific runs or invented results."""
import copy
import csv
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('evidence_checks',HERE/'collect.py')
r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)


def row(seed,hv):
    return dict(instance='fixture',configuration='S0',seed=seed,instance_digest='instance',scenario_digest='scenario',
        evaluation_budget=5000,completed=True,optimisation_seconds=10.,final_hypervolume=hv,
        budget_fronts=[dict(evaluation=b,hv=hv,front_raw=[[hv,hv,hv]]) for b in r.s.BUDGETS])


class EvidenceChecks(unittest.TestCase):
    def test_full_panel_pairs_not_ratio_of_means_or_maximum(self):
        base=[row(1,.2),row(2,.5)];own=[row(1,.4),row(2,.6)]
        groups={m:copy.deepcopy(own) for m in r.NAMES};groups['BASE']=base
        with tempfile.TemporaryDirectory() as d:
            out=Path(d);table=r.summarize_development(groups,out,draws=20)
            self.assertEqual(len(table),33)
            value=next(v for v in table if v['方法标识']=='PPO_PILOT_U001' and v['评价次数']==5000)
            self.assertAlmostEqual(value['相对Base平均配对提升(%)'],60.)
            self.assertAlmostEqual(value['平均HV'],.5)
            self.assertEqual([value[k] for k in ('胜Base','平Base','负Base')],[2,0,0])
            with (out/'DEV_ENDPOINT_5000.csv').open(encoding='utf-8-sig') as f:
                self.assertEqual(len(list(csv.DictReader(f))),11)

    def test_actual_label_costs_shared_cm_trajectory_counted_once(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);out=root/'output';out.mkdir()
            entry=dict(instance_id='a',configuration='S0',development_role='training')
            r.write(root/'instances/catalog.json',dict(instances=[entry]))
            for arm,counts in [('COLLECT_I',dict(I=5,C=0,M=0)),('COLLECT_CM',dict(I=0,C=7,M=6))]:
                r.write(root/'collection/a'/arm/'RUN_COMPLETE.json',dict(completed=True,real_objective_evaluation_count=10,
                    audit_evaluations=counts,audit_states=dict(I=2,C=2,M=2),audit_seconds_by_component=dict(I=1,C=2,M=3),
                    audit_seconds=6,optimisation_seconds=20))
            r.write(root/'shared_model/BUNDLE.json',dict(training_seconds_this_invocation=1.2))
            r.export_supervision(root,root/'missing_independent',out)
            q=r.read(out/'SUPERVISION_COST_SUMMARY.json')
            self.assertEqual(q['completed_trajectories'],2)
            self.assertEqual(q['search_evaluations'],20)
            self.assertEqual(q['extra_label_evaluations'],18)
            self.assertEqual(q['cumulative_search_seconds'],40)
            with (out/'SUPERVISION_LABEL_COSTS.csv').open(encoding='utf-8-sig') as f:
                self.assertEqual(len(list(csv.DictReader(f))),3)

    def test_current_environment_is_not_historical_allocation(self):
        with patch.object(r,'command',return_value={'synthetic':True}):q=r.hardware()
        self.assertIsNone(q['historical_allocated_cores'])
        self.assertIsNone(q['historical_allocated_memory_bytes'])
        self.assertIn('captured_at_utc',q)

    def test_zero_base_and_pair_mismatch_are_not_replaced(self):
        q=r.p.paired([row(1,.4)],[row(1,0.)],draws=10)
        self.assertIsNone(q['mean_paired_percent'])
        with self.assertRaises(ValueError):r.p.paired([row(1,.4)],[dict(row(1,.2),scenario_digest='wrong')],draws=10)

    def test_old_and_new_ppo_versions_never_merge(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);old=root/'old';new=root/'pilot';out=root/'report';out.mkdir();cached=[];tuning=[]
            r.write(old/'PLAN.json',{'synthetic':True})
            for cfg in range(9):
                for inst in range(2):
                    name=f'valid-S{cfg}-{inst}'
                    if cfg in (0,4,8) and inst==0:tuning.append({'instance_id':name})
                    for seed in range(3):
                        v=dict(row(seed,.4),instance=name,configuration='S'+str(cfg),real_objective_evaluation_count=5000)
                        for u in (0,4,8):
                            path=old/f'terminal/validation/u{u:03d}'/name/f'seed_{seed}/RUN_COMPLETE.json'
                            r.write(path,dict(v,final_hypervolume=.4+u*.001))
                            if u==0:cached.append(str(path))
                        if cfg in (0,4,8) and inst==0:
                            for u in (1,2,3):r.write(new/f'runs/tuning/u{u:03d}/greedy'/name/f'seed_{seed}/RUN_COMPLETE.json',dict(v,final_hypervolume=.4-u*.001))
            r.write(new/'PLAN.json',dict(tuning=tuning,validation_baseline_files=cached))
            v=dict(row(1,.4),real_objective_evaluation_count=5000)
            for i in range(32):r.write(old/f'terminal/rollouts/u{i//4+1:03d}/instance-{i}/RUN_COMPLETE.json',v)
            for i in range(9):
                for phase in ('training','training_reference'):r.write(new/f'runs/{phase}/u{i//3+1:03d}/instance-{i}/seed_1/RUN_COMPLETE.json',v)
            for folder,prefix,count in ((old,'terminal/checkpoints',8),(new,'checkpoints',3)):
                for u in range(1,count+1):r.write(folder/prefix/f'update_{u:03d}/UPDATE.json',dict(optimizer_steps=4,update=u))
            r.write(old/'terminal/SELECTED.json',dict(update=0));r.write(new/'SELECTION.json',dict(best_updated_checkpoint=1,best_including_u0=0))
            original=r.p.paired
            with patch.object(r.p,'paired',side_effect=lambda *a,**kw:original(*a,**dict(kw,draws=10))):r.export_ppo(old,new,out)
            q=r.read(out/'PPO_PROVENANCE.json')
            self.assertEqual(q['new_table_training']['policy_searches'],9)
            self.assertEqual(q['new_table_training']['paired_reference_searches'],9)
            self.assertEqual(q['new_table_training']['tuning_updates'],[0,1,2,3])
            self.assertEqual(q['pilot_v2']['selection']['best_including_u0'],0)
            with (out/'PPO_VALIDATION.csv').open(encoding='utf-8-sig') as f:rows=list(csv.DictReader(f))
            self.assertEqual([int(x['runs']) for x in rows],[54,54,54,9,9,9,9])
            self.assertEqual([int(x['update']) for x in rows],[0,4,8,0,1,2,3])


if __name__=='__main__':unittest.main()
