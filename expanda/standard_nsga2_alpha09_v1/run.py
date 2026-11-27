#!/usr/bin/env python3
"""81 standard NSGA-II searches followed by shared-draw final-front P90 reports."""
import argparse
import fcntl
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

sys.dont_write_bytecode = True
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'

from addon_common import *


def schedule(root, plan):
    """One low-priority worker; old pipeline processes are never signalled."""
    tasks = plan['tasks']; durations = []; start = time.monotonic()
    for task in tasks:
        if completed(task): continue
        out = Path(task['folder'])
        if out.exists(): out.rename(out.with_name(out.name+'.incomplete-'+str(time.time_ns())))
        out.mkdir(parents=True)
        tp = root/'tasks'/(canonical(task)+'.json'); freeze(tp,task)
        log = root/'logs'/(task['instance']+f"_{task['seed']}_{time.time_ns()}.log")
        log.parent.mkdir(exist_ok=True)
        begin = time.monotonic()
        with log.open('x') as stream:
            process = subprocess.Popen([sys.executable,'-B','-u',str(HERE/'run.py'),
                '--out',str(root),'--worker',str(tp)],stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
            try:
                while True:
                    done = sum((Path(t['folder'])/'JOB_COMPLETE.json').exists() for t in tasks)
                    eta = ((len(tasks)-done)*sum(durations)/len(durations)/3600) if durations else None
                    status(root,'standard_searches',done=done,total=len(tasks),workers=1,
                        elapsed_seconds=round(time.monotonic()-start,1),estimated_remaining_search_hours=eta,
                        running=[dict(pid=process.pid,instance=task['instance'],seed=task['seed'],log=str(log))])
                    try: code = process.wait(timeout=20); break
                    except subprocess.TimeoutExpired: pass
                require(code==0 and completed(task),'Search failed; inspect '+str(log))
            finally:
                if process.poll() is None:
                    os.killpg(process.pid,signal.SIGTERM)
                    try: process.wait(timeout=10)
                    except subprocess.TimeoutExpired: os.killpg(process.pid,signal.SIGKILL); process.wait()
        durations.append(time.monotonic()-begin)
    require(all(completed(t) for t in tasks),'Search panel incomplete')
    freeze(root/'SEARCH_COMPLETE.json',dict(completed=True,new_searches=81,training_updates=0,
        plan_sha256=plan['sha256']))
    status(root,'searches_complete',done=81,total=81)


def main(args):
    root = args.out.resolve(); parent = args.parent.resolve()
    require(root!=parent and root not in parent.parents and parent not in root.parents,
            'Comparator output must be separate from the main experiment')
    root.mkdir(parents=True,exist_ok=True)
    if args.worker:
        from standard_worker import search
        # Full source hashes are checked by the scheduler at stage boundaries.
        plan = plan_for(root,verify_inputs=False); task = read(args.worker)
        search(root,plan,task); return
    with (root/'.lock').open('a+') as lock:
        try: fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError: raise ValueError('Another comparator process already owns this output')
        plan = prepare(root,parent)
        args._write_status = True
        # Immediate complete 100/1000/5000 export of existing P90 results.
        # It is explicitly labelled BEFORE adding the standard comparator.
        from report09 import export_existing
        export_existing(root,plan)
        if args.prepare_only: return
        if os.getpriority(os.PRIO_PROCESS,0)<10: os.nice(10-os.getpriority(os.PRIO_PROCESS,0))
        if not args.report_only: schedule(root,plan)
        if args.search_only: return
        require(all(completed(t) for t in plan['tasks']),'Finish the 81 searches first')
        # Source P90 reevaluation is already complete.
        # Separate process avoids frozen engines' generic module names colliding.
        status(root,'independent_P90_reevaluation')
        child = subprocess.Popen([sys.executable,'-B','-u',str(HERE/'reevaluate.py'),'--out',str(root)],start_new_session=True)
        try:
            code=child.wait()
            require(code==0,'Reevaluation/report failed; see launch log')
        finally:
            if child.poll() is None:
                os.killpg(child.pid,signal.SIGTERM)
                try:child.wait(timeout=10)
                except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
        plan_for(root)
        status(root,'complete',completed=True,new_searches=81,training_updates=0,
            report_location=read(root/'RESULT_LOCATION.json')['directory'])


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--parent',type=Path,default=DEFAULT_PARENT)
    parser.add_argument('--out',type=Path,default=DIAG/'ccp_standard_nsga2_alpha09_v1')
    parser.add_argument('--worker',type=Path)
    parser.add_argument('--prepare-only',action='store_true')
    modes=parser.add_mutually_exclusive_group()
    modes.add_argument('--search-only',action='store_true')
    modes.add_argument('--report-only',action='store_true')
    args=parser.parse_args()
    args._write_status = False
    def interrupted(*_): raise KeyboardInterrupt()
    signal.signal(signal.SIGTERM,interrupted)
    try: main(args)
    except (Exception,KeyboardInterrupt) as error:
        # A failed worker does not overwrite the parent scheduler's status.
        if not args.worker and args._write_status and args.out.exists():
            status(args.out,'stopped',error=type(error).__name__+': '+str(error))
        raise
