#!/usr/bin/env python3
"""Small S11 panel; reuse immutable f6ff9f6 algorithms and worker implementation."""
from pathlib import Path
import sys
PILOT=Path(__file__).resolve().parent
sys.path.insert(0,str(PILOT.parent/'s11_targeted_v1'))
from common import *
from common import sources as parent_sources
from run import Paused,execute_jobs,collection_jobs,verify_frozen
import argparse,fcntl,signal,subprocess,time

def sources():
    return sorted(parent_sources()+list(PILOT.glob('*.py')))

def plan(root,settings):
    if digest(ENGINE/'baseline_uncertainty.py')!=settings['baseline_file_sha256']:raise ValueError('909315d transport baseline changed')
    files={str(p.relative_to(ROOT.parent)):digest(p) for p in sources()}
    data=ROOT.parent/'formal_icm_s0_v1/assets/data/data_expanded.xlsx'
    files[str(data.relative_to(ROOT.parent))]=digest(data)
    oldnorm=ROOT.parent/'formal_icm_s0_v1/assets/reward_normalization.json'
    files[str(oldnorm.relative_to(ROOT.parent))]=digest(oldnorm)
    import platform,numpy,torch
    value={'protocol':settings,'feature_version':FEATURE_VERSION,'files':files,
           'environment':{'python':platform.python_version(),'numpy':numpy.__version__,'torch':torch.__version__},
           'baseline_transport_equations':'unchanged formal_icm_s0_v1/src/baseline_uncertainty.py; all new behavior in S11 matched hooks',
           'checkpoint_policy':'new training only; old checkpoints prohibited'}
    value['sha256']=canonical_digest(value);freeze(root/'PLAN.json',value);return value['sha256']

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--stage',choices=['all','prepare','calibrate','collect','train','screen'],default='all')
    p.add_argument('--out',type=Path,default=ROOT.parent/'transformer_ppo_runs/diagnostics/ccp_s11_targeted_pilot_v1')
    p.add_argument('--source-catalog',type=Path);p.add_argument('--workers',type=int,default=2)
    a=p.parse_args();root=a.out.resolve();root.mkdir(parents=True,exist_ok=True);s=read_json(PILOT/'protocol.json')
    if a.workers<1:raise ValueError('workers must be positive')
    lock=(root/'pipeline.lock').open('a+')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise SystemExit('This output already has an active pipeline; do not start a duplicate')
    def interrupted(signum,frame):raise Paused(f'Received signal {signum}; preserving results')
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    deadline=time.monotonic()+s['invocation_seconds']
    try:
        sha=plan(root,s)
        from pilot_instances import prepare
        cat=prepare(root,s,a.source_catalog or Path(s['source_catalog_default']))
        print(f'[S11-PILOT] {s["configurations"]} / pop{s["population"]} / B{s["budget"]}: calibration, rule collection, fresh supervision, seven-arm screen; workers={a.workers}; no PPO',flush=True)
        if a.stage=='prepare':return
        if a.stage in ('all','calibrate'):
            pilot_methods=['BASE']+[c+'_'+v for c in 'ICM' for v in s['rule_variants']]
            pilot_jobs=[(r['instance_id'],r['algorithm_seed'],arm,root/'calibration'/r['instance_id']/arm) for r in cat['instances'] if r['development_role']=='calibration' for arm in pilot_methods]
            pilot_rows=execute_jobs(root,s,pilot_jobs,sha,deadline,a.workers)
            from report import select_rules
            selected=select_rules(pilot_rows,s,root,sha)
            print(f'[S11-PILOT] rule profiles frozen: {selected["profiles"]}',flush=True)
        if a.stage=='calibrate':return
        if not (root/'RULE_SELECTION.json').exists():raise ValueError('Run calibration first')
        if a.stage in ('all','collect'):
            execute_jobs(root,s,collection_jobs(root,cat),sha,deadline,a.workers)
        if a.stage in ('all','train'):
            # Separate process releases all training tensors before search workers start.
            proc=subprocess.Popen([sys.executable,'-u',str(ROOT/'run.py'),'--fit-worker','--out',str(root),'--plan-sha',sha])
            try:
                while proc.poll() is None:
                    if time.monotonic()>=deadline:raise Paused('Training invocation deadline reached; preserved checkpoints, restart same command')
                    try:proc.wait(timeout=min(30,max(1,deadline-time.monotonic())))
                    except subprocess.TimeoutExpired:print(f'[S11-PILOT] training heartbeat PID={proc.pid}',flush=True)
                if proc.returncode:raise RuntimeError('Training process failed; see preceding traceback')
            finally:
                if proc.poll() is None:
                    proc.terminate()
                    try:proc.wait(timeout=10)
                    except subprocess.TimeoutExpired:proc.kill();proc.wait()
        if a.stage in ('all','screen'):
            f=verify_frozen(root)
            if f['plan_sha256']!=sha:raise ValueError('Models belong to a different plan')
            jobs=[(r['instance_id'],seed,arm,root/'screen'/r['instance_id']/f'seed_{seed}'/arm) for r in cat['instances'] if r['development_role']=='screen' for seed in s['algorithm_seeds'] for arm in METHODS]
            rows=execute_jobs(root,s,jobs,sha,deadline,a.workers)
            from pilot_report import report
            q=report(rows,s,root);print(f'[S11-PILOT] COMPLETE pilot_direction={q["pilot_direction"]}; no automatic further experiments',flush=True)
        write_json(root/'STATUS.json',{'completed':True,'stage':a.stage,'final_test_opened':False})
    except Paused as e:
        write_json(root/'STATUS.json',{'completed':False,'paused':True,'reason':str(e),'resume':'Repeat the same command; complete runs are reused'});print(f'[S11-PILOT] PAUSED {e}',flush=True)
    except Exception as e:
        write_json(root/'STATUS.json',{'completed':False,'failed':True,'error':str(e)});raise
    finally:lock.close()
if __name__=='__main__':main()
