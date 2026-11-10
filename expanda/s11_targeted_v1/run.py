#!/usr/bin/env python3
"""Separate output tree, resumable stages and bounded process workers."""
from common import *
import argparse,fcntl,signal,subprocess,time,traceback

class Paused(Exception):pass

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

def verify_frozen(root):
    f=read_json(root/'models/FROZEN.json')
    for path,sha in f['files'].items():
        if digest(root/path)!=sha:raise ValueError(f'Frozen model/input changed: {path}')
    return f

def completed(folder,sha):
    p=folder/'RUN_COMPLETE.json'
    if not p.exists():return None
    r=read_json(p)
    if not r['completed'] or r['plan_sha256']!=sha:raise ValueError(f'Run belongs to different plan: {p}')
    if r.get('rule_selection_sha256'):
        roots=[a for a in folder.parents if (a/'RULE_SELECTION.json').exists()]
        if not roots or digest(roots[0]/'RULE_SELECTION.json')!=r['rule_selection_sha256']:raise ValueError('Rule profile changed after a completed run')
    if r.get('feedback_sha256') and digest(folder/'DATA.json')!=r['feedback_sha256']:raise ValueError('Stored labels changed')
    return r

def execute_jobs(root,settings,jobs,sha,deadline,workers):
    running={};results=[];pending=list(jobs);started=time.monotonic();heartbeat=0
    def stop():
        for proc,log,*_ in running.values():
            if proc.poll() is None:proc.terminate()
        for proc,log,*_ in running.values():
            try:proc.wait(timeout=10)
            except subprocess.TimeoutExpired:proc.kill();proc.wait()
            log.close()
    try:
        while pending or running:
            if time.monotonic()>=deadline:raise Paused('Invocation time limit; completed tasks reusable; interrupted tasks preserved')
            while pending and len(running)<workers:
                name,seed,arm,folder=pending.pop(0)
                prior=completed(folder,sha)
                if prior:results.append(prior);print(f'[S11-TARGET] reused {len(results)}/{len(jobs)} {name}/{seed}/{arm}',flush=True);continue
                if folder.exists():
                    archive=root/'interrupted';archive.mkdir(exist_ok=True)
                    destination=archive/f'{name}_{seed}_{arm}_{time.time_ns()}'
                    folder.rename(destination)
                    print(f'[S11-TARGET] preserved interrupted attempt: {destination}',flush=True)
                folder.mkdir(parents=True);logs=root/'logs';logs.mkdir(exist_ok=True)
                logfile=logs/f'{name}_{seed}_{arm}_{time.time_ns()}.log';log=logfile.open('w')
                cmd=[sys.executable,'-u',str(ROOT/'run.py'),'--worker','--out',str(root),'--name',name,'--seed',str(seed),'--arm',arm,'--run-out',str(folder),'--plan-sha',sha]
                proc=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT)
                running[str(folder)]=(proc,log,time.monotonic(),(name,seed,arm,folder),logfile)
                print(f'[S11-TARGET] start {name}/{seed}/{arm} PID={proc.pid} log={logfile}',flush=True)
            for key,(proc,log,t,job,logfile) in list(running.items()):
                if time.monotonic()-t>settings['task_timeout_seconds']:raise RuntimeError(f'Task exceeded fixed timeout; retained {logfile}')
                if proc.poll() is None:continue
                log.close();del running[key]
                if proc.returncode:raise RuntimeError(f'Worker failed ({proc.returncode}); see {logfile}')
                row=completed(job[3],sha)
                if row is None:raise RuntimeError(f'Worker missing completion: {job[3]}')
                results.append(row);print(f'[S11-TARGET] complete {len(results)}/{len(jobs)} {job[0]}/{job[2]} eval={row["real_objective_evaluation_count"]} seconds={row["optimisation_seconds"]:.1f}',flush=True)
            if time.monotonic()-heartbeat>=30:
                state={'phase':'running','completed_this_stage':len(results),'total_this_stage':len(jobs),'running':[{'instance':j[0],'seed':j[1],'arm':j[2],'seconds':round(time.monotonic()-t),'log':str(l)} for _,_,t,j,l in running.values()]}
                write_json(root/'STATUS.json',state);print(f'[S11-TARGET] heartbeat {len(results)}/{len(jobs)} running={len(running)} elapsed={time.monotonic()-started:.0f}s',flush=True);heartbeat=time.monotonic()
            if running:time.sleep(1)
        return results
    finally:stop()

def collection_jobs(root,cat):
    return [(r['instance_id'],r['algorithm_seed'],arm,root/'collection'/r['instance_id']/arm) for r in cat['instances'] if r['development_role'] in ('training','validation') for arm in ('COLLECT_I','COLLECT_C','COLLECT_M')]

def train(root,settings,cat,sha):
    from learning import fit_normalization,train_component
    jobs=collection_jobs(root,cat);data=[];cost=[]
    for name,seed,arm,folder in jobs:
        r=completed(folder,sha)
        if not r:raise ValueError(f'Collect first: {folder}')
        d=read_json(folder/'DATA.json');data.append((folder,d));cost.append(r)
        for component in (arm[-1],):
            if not all(d['phase_counts'][component]):raise ValueError(f'Missing a sampling phase: {name}/{component}; inspect this collection, no automatic extra sampling')
            if d['extra_evaluations'][component]>settings['extra_label_budget_per_component_instance']:raise ValueError('Label cap exceeded')
    report={'completed':True,'trajectories':len(jobs),'component_rule_full_trajectories':sum(j[2] in ('COLLECT_C','COLLECT_M') for j in jobs),'initialization_only_trajectories':sum(j[2]=='COLLECT_I' for j in jobs),
            'search_evaluations':sum(r['real_objective_evaluation_count'] for r in cost),'extra_label_evaluations':{c:sum(r['audit_evaluations'][c] for r in cost) for c in 'ICM'},
            'states':{c:sum(r['audit_states'][c] for r in cost) for c in 'ICM'},'label_evaluation_seconds':{c:sum(r['audit_seconds_by_component'][c] for r in cost) for c in 'ICM'},'elapsed_search_seconds':sum(r['optimisation_seconds'] for r in cost),
            'interrupted_attempts':len(list((root/'interrupted').glob('*'))) if (root/'interrupted').exists() else 0,
            'qualification':'Costs above cover completed runs. Interrupted-attempt work is retained separately and must not be reported as zero.'}
    write_json(root/'COLLECTION_SUMMARY.json',report)
    if (root/'models/FROZEN.json').exists():verify_frozen(root);print('[S11-TARGET-TRAIN] reuse frozen models',flush=True);return
    norm=fit_normalization([p for p,d in data if d['role']=='training']);freeze(root/'models/HV_NORMALIZATION.json',norm)
    for c in 'ICM':
        tr=[r for p,d in data if d['role']=='training' for r in d['rows'] if r['component']==c]
        va=[r for p,d in data if d['role']=='validation' for r in d['rows'] if r['component']==c]
        for role,rows in [('training',tr),('validation',va)]:
            if {r['instance'] for r in rows}!={r['instance_id'] for r in cat['instances'] if r['development_role']==role}:raise ValueError(f'Missing {role} instance in {c}')
        train_component(c,tr,va,norm,settings,root/'models'/c)
    assets=[root/'RULE_SELECTION.json',root/'models/HV_NORMALIZATION.json',root/'instances/catalog.json']+[p/'DATA.json' for p,d in data]
    assets+=list((root/'models').rglob('BUNDLE.json'))+list((root/'models').rglob('INPUT_SCALER.json'))+list((root/'models').rglob('VALIDATION.json'))+list((root/'models').rglob('*.pt'))
    freeze(root/'models/FROZEN.json',{'plan_sha256':sha,'files':{str(p.relative_to(root)):digest(p) for p in assets},'no_ppo':True,'screen_data_used_for_training':False})
    print('[S11-TARGET-TRAIN] all three supervised ensembles and gates frozen',flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--stage',choices=['all','prepare','calibrate','collect','train','screen'],default='all')
    p.add_argument('--out',type=Path,default=ROOT.parent/'transformer_ppo_runs/diagnostics/ccp_s11_targeted_supervised_v1')
    p.add_argument('--source-catalog',type=Path);p.add_argument('--workers',type=int,default=2)
    p.add_argument('--fit-worker',action='store_true',help=argparse.SUPPRESS)
    p.add_argument('--worker',action='store_true',help=argparse.SUPPRESS);p.add_argument('--name');p.add_argument('--seed',type=int);p.add_argument('--arm');p.add_argument('--run-out',type=Path);p.add_argument('--plan-sha')
    a=p.parse_args();root=a.out.resolve();root.mkdir(parents=True,exist_ok=True);s=read_json(ROOT/'protocol.json')
    if a.worker or a.fit_worker:
        from worker import work
        record=read_json(root/'PLAN.json')
        if canonical_digest({k:v for k,v in record.items() if k!='sha256'})!=a.plan_sha:raise ValueError('Worker plan mismatch')
        s=record['protocol']
        if a.fit_worker:train(root,s,read_json(root/'instances/catalog.json'),a.plan_sha)
        else:work(root,s,a.name,a.seed,a.arm,a.run_out,a.plan_sha)
        return
    if a.workers<1:raise ValueError('workers must be positive')
    lock=(root/'pipeline.lock').open('a+')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise SystemExit('This output already has an active pipeline; do not start a duplicate')
    def interrupted(signum,frame):raise Paused(f'Received signal {signum}; preserving results')
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    deadline=time.monotonic()+s['invocation_seconds']
    try:
        sha=plan(root,s)
        from instances import prepare
        cat=prepare(root,s,a.source_catalog or Path(s['source_catalog_default']))
        print(f'[S11-TARGET] {s["configurations"]} / pop{s["population"]} / B{s["budget"]}: calibration, rule collection, fresh supervision, seven-arm screen; workers={a.workers}; no PPO',flush=True)
        if a.stage=='prepare':return
        if a.stage in ('all','calibrate'):
            pilot_methods=['BASE']+[c+'_'+v for c in 'ICM' for v in s['rule_variants']]
            pilot_jobs=[(r['instance_id'],r['algorithm_seed'],arm,root/'calibration'/r['instance_id']/arm) for r in cat['instances'] if r['development_role']=='calibration' for arm in pilot_methods]
            pilot_rows=execute_jobs(root,s,pilot_jobs,sha,deadline,a.workers)
            from report import select_rules
            selected=select_rules(pilot_rows,s,root,sha)
            print(f'[S11-TARGET] rule profiles frozen: {selected["profiles"]}',flush=True)
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
                    except subprocess.TimeoutExpired:print(f'[S11-TARGET] training heartbeat PID={proc.pid}',flush=True)
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
            from report import report
            q=report(rows,s,root);print(f'[S11-TARGET] COMPLETE all_components_met_reliable_gain={q["all_components_met_reliable_gain"]}; no automatic further experiments',flush=True)
        write_json(root/'STATUS.json',{'completed':True,'stage':a.stage,'final_test_opened':False})
    except Paused as e:
        write_json(root/'STATUS.json',{'completed':False,'paused':True,'reason':str(e),'resume':'Repeat the same command; complete runs are reused'});print(f'[S11-TARGET] PAUSED {e}',flush=True)
    except Exception as e:
        write_json(root/'STATUS.json',{'completed':False,'failed':True,'error':str(e)});raise
    finally:lock.close()
if __name__=='__main__':main()
