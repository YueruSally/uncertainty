#!/usr/bin/env python3
"""Resume original formal experiment, train shared supervision and evaluate it. Never PPO."""
from joint_common import *
import argparse,fcntl,subprocess,signal,time

def command(cmd,log,status_root,stage,progress_root=None):
 log.parent.mkdir(parents=True,exist_ok=True)
 with log.open('a') as h:
  p=subprocess.Popen(cmd,stdout=h,stderr=subprocess.STDOUT,start_new_session=True)
  previous={}
  def stop(*_):raise KeyboardInterrupt('Stopped; completed search results retained')
  for sig in (signal.SIGINT,signal.SIGTERM):previous[sig]=signal.signal(sig,stop)
  try:
   while p.poll() is None:
    progress={}
    if progress_root:
     for name in ('STATUS.json','OOS_STATUS.json'):
      file=progress_root/name
      if file.exists():
       try:progress[name]=read_json(file)
       except (ValueError,OSError):pass
    row={'stage':stage,'pid':p.pid,'log':str(log),'progress':progress,'ppo_started':False}
    write_json(status_root/'STATUS.json',row);print('[JOINT-PIPELINE] '+json.dumps(row),flush=True)
    for _ in range(30):
     if p.poll() is not None:break
     time.sleep(1)
   if p.returncode:
    tail=log.read_text(errors='replace')[-6000:]
    write_json(status_root/'STATUS.json',{'completed':False,'stage':stage,'error':tail,'log':str(log),'ppo_started':False})
    raise RuntimeError(f'{stage} exited {p.returncode}; log={log}\n{tail}')
  finally:
   if p.poll() is None:
    os.killpg(p.pid,signal.SIGTERM)
    try:p.wait(timeout=15)
    except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
   for sig,handler in previous.items():signal.signal(sig,handler)

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,default=ROOT.parent/'transformer_ppo_runs/diagnostics/ccp_icm_formal_joint_v1')
 p.add_argument('--original-out',type=Path,default=PACKAGE/'formal_runs')
 p.add_argument('--diagnostics',type=Path,default=ROOT.parent/'transformer_ppo_runs/diagnostics')
 p.add_argument('--stage',choices=('all','supervised','original','shared-search'),default='all')
 for c in COMPONENTS:p.add_argument('--'+c.lower()+'-collection',type=Path)
 a=p.parse_args();out=a.out.resolve();old=a.original_out.resolve();out.mkdir(parents=True,exist_ok=True)
 with (out/'.pipeline.lock').open('a+') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  settings=read_json(PACKAGE/'experiment.json')
  freeze(out/'WORKFLOW.json',{'original_output':str(old),'original_config_sha256':digest(PACKAGE/'experiment.json'),
   'shared_settings':SETTINGS,'original_runs':2400,'shared_new_runs':300,'terminal_ppo_started':False})
  original=[sys.executable,'-u',str(PACKAGE/'run.py'),'--out',str(old)]
  if a.stage in ('all','original'):
   command(original+['--prepare'],out/'prepare.log',out,'prepare-original',old)
  if a.stage in ('all','supervised'):
   cmd=[sys.executable,'-u',str(ROOT/'train_shared.py'),'--diagnostics',str(a.diagnostics.resolve()),'--out',str(out/'shared_model')]
   for c in COMPONENTS:
    x=getattr(a,c.lower()+'_collection')
    if x:cmd.extend(['--'+c.lower()+'-collection',str(x.resolve())])
   command(cmd,out/'shared_training.log',out,'shared-supervised-training')
  if a.stage in ('all','original'):
   command(original+['--smoke'],out/'original_smoke.log',out,'original-eight-full-budget-checks',old)
   smoke=read_json(old/'SMOKE_REPORT.json');print('[ORIGINAL-COST] '+json.dumps({k:v for k,v in smoke.items() if k.startswith('estimated_') or k=='seconds_per_method'}),flush=True)
   # Preserve original 24-hour invocation limit. Resume only its explicit timeout, never arbitrary errors.
   while True:
    before=len(list((old/'runs').glob('*/*/*/RUN_COMPLETE.json')))+len(list((old/'oos').glob('*/*.json')))
    try:
     command(original+['--run'],out/'original_run.log',out,'original-2400-searches-and-oos',old);break
    except RuntimeError as exc:
     text=str(exc)
     if 'TimeoutError:' not in text or not any(s in text for s in ('invocation stopped','run stopped','OOS invocation stopped','OOS interrupted')):raise
     after=len(list((old/'runs').glob('*/*/*/RUN_COMPLETE.json')))+len(list((old/'oos').glob('*/*.json')))
     if after<=before:raise RuntimeError('Timeout without progress; inspect logs before resuming') from exc
     print('[JOINT-PIPELINE] original invocation limit reached; continuing unchanged completed-run resume',flush=True)
  if a.stage in ('all','shared-search'):
   command([sys.executable,'-u',str(ROOT/'shared_search.py'),'--source',str(old),'--model',str(out/'shared_model'),
      '--out',str(out/'shared_search')],out/'shared_search.log',out,'shared-supervised-300-searches',out/'shared_search')
  write_json(out/'STATUS.json',{'completed_requested_stage':a.stage,'ppo_started':False,
      'original_results':str(old),'shared_model':str(out/'shared_model'),'shared_results':str(out/'shared_search')})
  print('[JOINT-PIPELINE] requested stage COMPLETE; terminal PPO remains unstarted',flush=True)
if __name__=='__main__':main()
