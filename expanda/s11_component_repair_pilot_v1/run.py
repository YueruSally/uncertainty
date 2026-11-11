#!/usr/bin/env python3
"""Worker entry used by the bounded supervisor; no public experiment settings."""
from bootstrap import *
import argparse


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--out',type=Path,required=True);p.add_argument('--plan-sha',required=True)
    p.add_argument('--worker',action='store_true');p.add_argument('--fit-worker',action='store_true')
    p.add_argument('--name');p.add_argument('--seed',type=int);p.add_argument('--arm');p.add_argument('--run-out',type=Path)
    a=p.parse_args();root=a.out.resolve();plan=read_json(root/'PLAN.json')
    if canonical_digest({k:v for k,v in plan.items() if k!='sha256'})!=a.plan_sha:
        raise ValueError('Worker plan mismatch')
    s=plan['protocol']
    if a.fit_worker:
        from fit import train
        train(root,s,read_json(root/'instances/catalog.json'),a.plan_sha)
    elif a.worker:
        import worker
        from repair_controller import RepairController
        worker.ScreenController=RepairController
        if a.arm.startswith('COLLECT_'):
            s=dict(s,extra_label_budget_per_component_instance=s['label_caps'][a.arm[-1]])
        worker.work(root,s,a.name,a.seed,a.arm,a.run_out,a.plan_sha)
    else:
        raise ValueError('Use pipeline.py or run.sh')


if __name__=='__main__':main()
