#!/usr/bin/env python3
"""Complete isolated alpha=.8 study: collect, SL, PPO, final search, fixed-front OOS."""
import os
import sys
sys.dont_write_bytecode=True
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name]='1'

from common08 import *
import argparse
import fcntl

STAGES=('collect','train','ppo','search','oos')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=DIAG/'ccp_alpha08_single_full_v1')
    parser.add_argument('--training-source',type=Path,default=DIAG/'ccp_icm_multiscale_shared_v1')
    parser.add_argument('--workers',type=int,choices=(1,2),default=1)
    parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--stage',choices=STAGES)
    parser.add_argument('--worker',type=Path)
    a=parser.parse_args();root=a.out.resolve();root.mkdir(parents=True,exist_ok=True)
    require(DIAG.resolve() in root.parents and root.name.startswith('ccp_alpha08_'),'Use a new ccp_alpha08_* diagnostics directory')
    try:os.nice(max(0,10-os.getpriority(os.PRIO_PROCESS,0)))
    except OSError:pass
    def stop(signum,frame):raise KeyboardInterrupt('Signal '+str(signum))
    # Let scheduler finally-blocks stop only this experiment's worker groups.
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    if a.worker:
        from worker08 import worker
        worker(root,read(a.worker));return
    if a.stage:
        plan=plan_for(root)
        if a.stage=='collect':
            from stages08 import collect_stage
            collect_stage(root,plan,a.workers)
        elif a.stage=='train':
            from stages08 import train_stage
            train_stage(root,plan)
        elif a.stage=='ppo':
            from experiment08 import ppo_stage
            ppo_stage(root,plan,a.workers)
        elif a.stage=='search':
            from experiment08 import evaluation_stage
            evaluation_stage(root,plan,a.workers)
        else:
            from reevaluate08 import execute
            execute(root,plan)
        return
    # Parent holds a separate lock throughout. Children never touch historical jobs.
    lockpath=root/'.pipeline.lock'
    with lockpath.open('a+') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise SystemExit('This alpha08 output already has an active process')
        proc=None
        try:
            from stages08 import prepare
            plan=prepare(root,a.training_source.resolve())
            if a.prepare_only:return
            for stage in STAGES:
                marker=root/'stages'/(stage+'.json')
                if marker.exists():
                    m=read(marker);require(m['plan_sha256']==plan['sha256'],'Stage plan differs');verify(m['files']);continue
                status(root,stage,workers=a.workers)
                proc=subprocess.Popen([sys.executable,'-B','-u',str(HERE/'run.py'),'--out',str(root),
                    '--stage',stage,'--workers',str(a.workers)],start_new_session=True)
                require(proc.wait()==0,'Stage stopped: '+stage+'; previous stages and all old results are preserved')
                proc=None
                names={'collect':['COLLECTION_SUMMARY.json'], 'train':['SL_SELECTION.json'],
                       'ppo':['ppo/SELECTION.json','ppo/CONFIRMATION.json'], 'search':['EVALUATION_TASKS.json','RANDOM_TARGETS.json'],
                       'oos':['oos/COMPLETE.json']}[stage]
                freeze(marker,dict(completed=True,plan_sha256=plan['sha256'],files={str(root/n):digest(root/n) for n in names}))
            verify(plan['source_files']);verify(plan['code_files'])
            status(root,'complete',completed_stage=True,all_stages_complete=True)
            state=read(root/'STATUS.json');state['completed']=True;write(root/'STATUS.json',state)
            print('[ALPHA08] COMPLETE '+str(root/'oos/RESULT_LOCATION.json'),flush=True)
        except BaseException as error:
            status(root,'stopped',error=str(error),resume='Repeat the same command; finished jobs are hash checked and reused')
            raise
        finally:
            if proc is not None and proc.poll() is None:
                os.killpg(proc.pid,signal.SIGTERM)
                try:proc.wait(timeout=20)
                except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()


if __name__=='__main__':main()
