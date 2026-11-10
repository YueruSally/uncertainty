#!/usr/bin/env python3
"""Stop only the current user's exact large S11 job; retain all files."""
import argparse,fcntl,os,shlex,signal,subprocess,time
from pathlib import Path


def matching_processes(text,root,script,uid):
    matched={}
    for line in text.splitlines():
        pieces=line.strip().split(None,3)
        if len(pieces)!=4:continue
        try:
            pid,owner,ppid=map(int,pieces[:3]);args=shlex.split(pieces[3])
        except ValueError:continue
        if owner!=uid or str(script) not in args or '--out' not in args:continue
        pos=args.index('--out')
        if pos+1>=len(args) or args[pos+1]!=str(root):continue
        # The runner invokes Python with an absolute script and output path.
        # Do not match shells, grep commands, different roots, or other users.
        if not Path(args[0]).name.lower().startswith('python'):continue
        matched[pid]={'ppid':ppid,'worker':('--worker' in args or '--fit-worker' in args)}
    return matched


def snapshot(root,script):
    result=subprocess.run(['ps','-ww','-eo','pid=,uid=,ppid=,args='],check=True,text=True,capture_output=True)
    return matching_processes(result.stdout,root,script,os.getuid())


def ensure_unlocked(root):
    path=root/'pipeline.lock'
    if not path.exists():return
    with path.open('r') as handle:
        try:fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('Large output is still locked but no safely matched supervisor was found; inspect its PID before starting the pilot.')
        finally:fcntl.flock(handle,fcntl.LOCK_UN)


def pause(root,script,timeout=30):
    active=snapshot(root,script)
    if not active:
        ensure_unlocked(root)
        print('[S11-PAUSE] No matching large S11 process; existing files untouched.',flush=True);return
    parents=[p for p,v in active.items() if not v['worker']]
    targets=parents or list(active)  # Also handle workers orphaned by an earlier shell interruption.
    for pid in targets:
        # Recheck immediately before signaling, to avoid stale PID matches.
        if pid not in snapshot(root,script):continue
        try:os.kill(pid,signal.SIGTERM)
        except ProcessLookupError:continue
        print(f'[S11-PAUSE] Sent TERM to matching PID={pid}; preserving output.',flush=True)
    end=time.monotonic()+timeout
    while time.monotonic()<end:
        remaining=snapshot(root,script)
        if not remaining:
            ensure_unlocked(root)
            print('[S11-PAUSE] Large job stopped. Completed and interrupted files retained.',flush=True);return
        time.sleep(.5)
    raise RuntimeError(f'Large job still has matching processes {sorted(remaining)}; small pilot was not started. Inspect these PIDs first.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args()
    script=Path(__file__).resolve().parent.parent/'s11_targeted_v1/run.py'
    pause(args.root.resolve(),script.resolve())

if __name__=='__main__':main()
