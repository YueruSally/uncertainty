#!/usr/bin/env python3
"""Read-only follower of full training; run 12 additional early-budget searches."""
from bridge_fast import *
import argparse
import fcntl
import os
import platform
import signal
import time


def prepare(root,source,full_root,protocol):
    bundle,entries=verify_shared(source)
    old=read_json(full_root/'PLAN.json');verify(old['files'])
    if old['sha256']!=canonical_digest({k:v for k,v in old.items() if k!='sha256'}):raise ValueError('Full plan changed')
    if old['source']!=str(source) or old['normalization']!=bundle['normalization'] or old['entries']!=entries:raise ValueError('Wrong full-study source')
    environment={'python':platform.python_version(),'torch':str(torch.__version__),'numpy':str(np.__version__)}
    if old['environment']!=environment:raise ValueError('Use the full-study runtime environment')
    settings=copy.deepcopy(old['settings'])
    if settings['population']!=protocol['population'] or settings['progress_denominator']!=protocol['progress_denominator']:raise ValueError('Feature/population contract changed')
    if settings['budget']<protocol['budget']:raise ValueError('Fast check exceeds full budget')
    settings['budget']=protocol['budget'];settings['milestones']=[n for n in settings['milestones'] if n<=settings['budget']]
    if settings['budget'] not in settings['milestones']:settings['milestones'].append(settings['budget'])
    panel=[]
    for cfg in protocol['configurations']:
        group=sorted([r for r in entries if r['configuration']==cfg and r['development_role']=='validation'],key=lambda r:r['instance_id'])
        if not group:raise ValueError('No validation instance for '+cfg)
        panel.append(group[0]);freeze(root/'instances'/(group[0]['instance_id']+'.json'),group[0])
    provenance={**runtime_files_fast(),str(full_root/'PLAN.json'):digest(full_root/'PLAN.json'),str(source/'shared_model/BUNDLE.json'):digest(source/'shared_model/BUNDLE.json')}
    provenance.update({str(source/'shared_model'/p):s for p,s in bundle['checkpoint_sha256'].items()})
    provenance.update({str(root/'instances'/(r['instance_id']+'.json')):digest(root/'instances'/(r['instance_id']+'.json')) for r in panel})
    plan={'source':str(source),'full_source':str(full_root),'full_plan_sha256':old['sha256'],'settings':settings,'fast':protocol,
          'panel':panel,'seed':settings['ppo']['validation_seed'],'normalization':bundle['normalization'],'files':provenance,'environment':environment}
    plan['sha256']=canonical_digest(plan);freeze(root/'PLAN.json',plan);return plan


def model_folder(plan,arm):
    root=Path(plan['full_source'])
    path={'BASE':None,'ICM_INDEPENDENT':root/'independent_model','ICM_SHARED_SUPERVISED':Path(plan['source'])/'shared_model',
            'ICM_TERMINAL_PPO':root/'terminal/checkpoints'/f'update_{plan["fast"]["ppo_update"]:03d}'}[arm]
    return path.resolve() if path is not None else None


def binding(root,plan,arm):
    folder=model_folder(plan,arm)
    if folder is None:return
    checkpoint=folder/'BUNDLE.json';started=time.monotonic()
    while not checkpoint.exists():
        status_path=Path(plan['full_source'])/'STATUS.json'
        try:state=read_json(status_path) if status_path.exists() else {}
        except ValueError:state={}
        if state.get('failed') or state.get('paused') or state.get('completed'):
            raise full.Paused('Required model missing; full-study status='+str(state)+'; existing fast results preserved')
        if time.monotonic()-started>plan['fast']['model_wait_seconds']:raise full.Paused('Model wait time cap; repeat after the full trainer publishes update')
        write_json(root/'STATUS.json',{'completed':False,'stage':'waiting_for_model','model':str(folder),'waiting_seconds':int(time.monotonic()-started),'full_status':state})
        print(f'[FASTCHECK] waiting for {folder}; no new training started; full stage={state.get("stage")}',flush=True)
        time.sleep(30)
    for attempt in range(5):
        try:b=read_json(checkpoint);break
        except ValueError:
            if attempt==4:raise
            time.sleep(.2)
    verify({str(folder/p):s for p,s in b['checkpoint_sha256'].items()})
    shared=read_json(Path(plan['source'])/'shared_model/BUNDLE.json')
    for key in ('normalization','scalers','target_scales','training_instances','validation_instances'):
        if b[key]!=shared[key]:raise ValueError('Model is incompatible: '+arm+'/'+key)
    if arm=='ICM_TERMINAL_PPO' and (not b.get('terminal_checkpoint') or read_json(folder/'UPDATE.json')['update']!=plan['fast']['ppo_update']):raise ValueError('Wrong PPO update')
    freeze(root/'models'/(arm+'.json'),{'path':str(folder),'bundle_sha256':digest(checkpoint),'checkpoint_sha256':b['checkpoint_sha256']})


def make_task(root,plan,entry,arm):
    folder=model_folder(plan,arm)
    t=full.task(plan,entry['instance_id'],plan['seed'],arm,'validation',folder)
    return t,root/'runs'/entry['instance_id']/arm


def validate_worker(root,plan,t,folder):
    if t['phase']!='validation' or t['sample'] or t['arm'] not in plan['fast']['methods']:raise ValueError('Only frozen fast validation is allowed')
    for entry in plan['panel']:
        if entry['instance_id']!=t['instance']:continue
        expected,path=make_task(root,plan,entry,t['arm'])
        if expected!=t or path.resolve()!=folder.resolve():raise ValueError('Wrong seed/model/output')
        if t['model']:
            b=read_json(root/'models'/(t['arm']+'.json'))
            if b['path']!=t['model'] or b['bundle_sha256']!=t['bundle_sha256']:raise ValueError('Frozen fast model changed')
            verify({str(Path(t['model'])/p):s for p,s in b['checkpoint_sha256'].items()})
        return entry
    raise ValueError('Training, screen and test instances are forbidden here')


def execute(root,source,full_root,protocol,workers):
    from reporting_fast import report
    plan=prepare(root,source,full_root,protocol)
    # Read existing completed checkpoint/validation evidence before new work.
    spec=importlib.util.spec_from_file_location('completed_checkpoint_audit',FAST.parent/'audit_s0_joint_v1/checkpoints.py')
    audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)
    findings=audit.review(EXPANDA/'transformer_ppo_runs/diagnostics',full_root)
    write_json(root/'CHECKPOINT_AUDIT.json',findings)
    rows=[];done=[]
    # Produce Base/shared results while independent fitting or PPO may still run.
    for arm in ('BASE','ICM_SHARED_SUPERVISED','ICM_INDEPENDENT','ICM_TERMINAL_PPO'):
        binding(root,plan,arm)
        specs=[make_task(root,plan,r,arm) for r in plan['panel']]
        rows.extend(jobs(root,plan,specs,workers,'fast/'+arm));done.append(arm)
        ordered=[m for m in protocol['methods'] if m in done]
        result=report(rows,plan,root/'results',ordered)
        print(f'[FASTCHECK] available methods {len(done)}/4; results/00_当前小试对比.csv',flush=True)
    write_json(root/'STATUS.json',{'completed':True,'stage':'fastcheck_complete','full_study_untouched':True,'final_test_opened':False})
    print('[FASTCHECK] COMPLETE; full study continues independently',flush=True);return result


def main():
    diag=EXPANDA/'transformer_ppo_runs/diagnostics';p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=diag/'ccp_icm_multiscale_shared_v1')
    p.add_argument('--full-source',type=Path,default=diag/'ccp_icm_multiscale_joint_v1')
    p.add_argument('--out',type=Path,default=diag/'ccp_icm_multiscale_fastcheck_b5000_v1')
    p.add_argument('--workers',type=int,default=1)
    p.add_argument('--worker',action='store_true',help=argparse.SUPPRESS);p.add_argument('--task',type=Path);p.add_argument('--run-out',type=Path)
    a=p.parse_args();root=a.out.resolve();source=a.source.resolve();full_root=a.full_source.resolve();torch.set_num_threads(1)
    if any(root==r or root in r.parents or r in root.parents for r in (source,full_root)):raise ValueError('Use an isolated fast-check directory')
    root.mkdir(parents=True,exist_ok=True)
    if a.worker:
        plan=read_json(root/'PLAN.json');verify(plan['files'])
        if canonical_digest({k:v for k,v in plan.items() if k!='sha256'})!=plan['sha256']:raise ValueError('Fast plan changed')
        t=read_json(a.task);entry=validate_worker(root,plan,t,a.run_out)
        from search import run_search
        run_search(root/'instances'/(entry['instance_id']+'.json'),plan['settings'],t,a.run_out,plan['sha256']);return
    if a.workers<1:raise ValueError('workers must be positive')
    with (root/'.pipeline.lock').open('a+') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise SystemExit('Fast check already running; do not start another copy')
        def stop(signum,frame):raise full.Paused('Signal '+str(signum))
        signal.signal(signal.SIGINT,stop);signal.signal(signal.SIGTERM,stop)
        try:execute(root,source,full_root,read_json(FAST/'protocol.json'),a.workers)
        except full.Paused as e:
            write_json(root/'STATUS.json',{'completed':False,'paused':True,'reason':str(e),'resume':'same command; completed fast searches reused'})
            print('[FASTCHECK] PAUSED '+str(e),flush=True)
        except Exception as e:
            write_json(root/'STATUS.json',{'completed':False,'failed':True,'error':str(e)});raise


if __name__=='__main__':main()
