"""Isolated alpha=.8 experiment. No historical files are written."""
from pathlib import Path
from datetime import datetime, timezone
import csv
import hashlib
import importlib.util
import json
import os
import signal
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
DIAG = EXP / 'transformer_ppo_runs/diagnostics'
ALPHA = .8
SHARED, PPO = 'ICM_SHARED_SUPERVISED', 'ICM_TERMINAL_PPO'
RANDOM, ZERO = 'ICM_RANDOM_RATE_CONTROLLED', 'ICM_SHARED_ZERO_THRESHOLD'
ARMS = ('BASE', 'I_RULE', 'I_LEARN', 'C_RULE', 'C_LEARN', 'M_RULE', 'M_LEARN',
        'ICM_HEURISTIC', 'ICM_INDEPENDENT', SHARED, PPO, RANDOM, ZERO)


def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def require(x, message):
    if not x: raise ValueError(message)
def canonical(x):
    return hashlib.sha256(json.dumps(x, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
def digest(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''): h.update(b)
    return h.hexdigest()
def write(p, x):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    t = p.with_name(p.name + '.writing')
    t.write_text(json.dumps(x, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    t.replace(p)
def freeze(p, x):
    if Path(p).exists(): require(read(p) == x, 'Frozen input changed: ' + str(p))
    else: write(p, x)
def verify(files):
    for p, h in files.items(): require(digest(p) == h, 'Pinned file changed: '+p)
def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); sys.modules[name] = m
    spec.loader.exec_module(m); return m
def csv_write(path, rows):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
def plan_for(root):
    p = read(root/'PLAN.json')
    require(canonical({k:v for k,v in p.items() if k != 'sha256'}) == p['sha256'], 'Plan digest changed')
    require(p['alpha'] == ALPHA and p['settings']['confidence'] == ALPHA, 'Wrong alpha')
    verify(p['code_files']); return p
def status(root, stage, **extra):
    value = dict(completed=False, stage=stage, pid=os.getpid(), alpha=ALPHA,
                 updated_utc=datetime.now(timezone.utc).isoformat(), **extra)
    write(root/'STATUS.json', value)
    print('[ALPHA08]', json.dumps(value, ensure_ascii=False), flush=True)
def job(root, kind, entry, seed, arm, folder, **kw):
    return dict(kind=kind, instance=entry['instance_id'], seed=int(seed), arm=arm,
                folder=str(folder), entry=entry, **kw)
def completed(task):
    p = Path(task['folder'])/'JOB_COMPLETE.json'
    if not p.exists(): return False
    q = read(p)
    require(q['task_sha256'] == canonical(task), 'Cached job belongs to another task')
    verify(q['files']); return True


def schedule(root, tasks, workers, stage):
    """Only subprocesses created here are stopped; resume never overwrites attempts."""
    from collections import deque
    plan = plan_for(root)
    pending = deque(t for t in tasks if not completed(t)); done = len(tasks)-len(pending)
    running = []; started = time.monotonic(); observed = []; last = 0.
    try:
        while pending or running:
            while pending and len(running) < workers:
                t = pending.popleft(); folder = Path(t['folder'])
                require(root in folder.parents, 'Output outside new experiment')
                if folder.exists(): folder.rename(folder.with_name(folder.name+'.incomplete-'+str(time.time_ns())))
                folder.mkdir(parents=True)
                tp = root/'tasks'/(canonical(t)+'.json'); freeze(tp, t)
                lp = root/'logs'/(stage.replace('/', '_')+'_'+canonical(t)+f'_{time.time_ns()}.log')
                lp.parent.mkdir(exist_ok=True); f = lp.open('w')
                proc = subprocess.Popen([sys.executable, '-B', '-u', str(HERE/'run.py'),
                    '--out', str(root), '--worker', str(tp)], stdout=f, stderr=subprocess.STDOUT,
                    start_new_session=True)
                running.append(dict(proc=proc, log=f, path=str(lp), task=t, start=time.monotonic()))
            for x in list(running):
                code = x['proc'].poll()
                require(time.monotonic()-x['start'] < 86400, 'Worker exceeded 24h: '+x['path'])
                if code is None: continue
                x['log'].close(); running.remove(x)
                require(code == 0 and completed(x['task']), 'Worker failed; inspect '+x['path'])
                observed.append(time.monotonic()-x['start']); done += 1
            now = time.monotonic()
            if now-last >= 20 or not (pending or running):
                avg = sum(observed)/len(observed) if observed else None
                eta = ((len(pending)*avg + sum(max(0,avg-(now-x['start'])) for x in running))/workers/3600
                       if avg is not None else None)
                status(root, stage, done=done, total=len(tasks), workers=workers,
                    elapsed_seconds=round(now-started,1), estimated_remaining_stage_hours=eta,
                    estimate_scope='This stage only; observed worker wall time, not a guaranteed finish time',
                    running=[dict(pid=x['proc'].pid, instance=x['task']['instance'], arm=x['task']['arm'],
                                  seconds=round(now-x['start']), log=x['path']) for x in running])
                last = now
            if running: time.sleep(1)
    finally:
        for x in running:
            p = x['proc']
            if p.poll() is None:
                os.killpg(p.pid, signal.SIGTERM)
                try: p.wait(timeout=10)
                except subprocess.TimeoutExpired: os.killpg(p.pid, signal.SIGKILL); p.wait()
            x['log'].close()
    verify(plan['code_files'])


def seal_job(task):
    folder = Path(task['folder'])
    files = {str(p):digest(p) for p in folder.rglob('*') if p.is_file() and p.name != 'JOB_COMPLETE.json'}
    require(files and (folder/'RUN_COMPLETE.json').exists(), 'Missing completed record')
    freeze(folder/'JOB_COMPLETE.json', dict(completed=True, task_sha256=canonical(task), files=files))
