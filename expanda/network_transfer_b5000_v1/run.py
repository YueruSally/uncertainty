#!/usr/bin/env python3
"""27 frozen single-policy searches after four explicit paired-input checks."""
import argparse
from collections import deque
import copy
from datetime import datetime, timezone
import fcntl
import importlib.util
import inspect
import math
import os
from pathlib import Path
import signal
import socket
from statistics import mean, stdev
import subprocess
import sys
import time

sys.dont_write_bytecode=True
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
HERE=Path(__file__).resolve().parent; EXP=HERE.parent
sys.path.insert(0,str(EXP))
from inputs import (read,write,freeze,digest,canonical,verify,copy_frozen,csv_write,
                    prepare_inputs,load_paths,load_scenarios,scenario_kwargs,SEEDS,BUDGETS,SHARED)


def engine():
    from icm_single_member_b5000 import load_module
    s=load_module(EXP)
    from transformer_ppo.episode import resolve_uncertainty_profile
    from search_b5000 import SharedPolicy,enabled_components
    s.SharedPolicy=SharedPolicy;s.enabled_components=enabled_components
    s.profile=resolve_uncertainty_profile('ccp','transport_border_saa_v1',100)
    s.p.torch.set_num_threads(1)
    return s


def checked_plan(path):
    q=read(path)
    if canonical({k:v for k,v in q.items() if k!='sha256'})!=q['sha256']:
        raise ValueError('Plan checksum mismatch: '+str(path))
    return q


def prepare(root,source,s):
    if (root/'PLAN.json').exists():
        plan=checked_plan(root/'PLAN.json')
        if plan['source']!=str(source):raise ValueError('Resume source changed')
        verify(plan['files']);return plan
    old=checked_plan(source/'PLAN.json')
    if not read(source/'STATUS.json').get('completed'):
        raise ValueError('Expected completed S0-S8 single-model source')
    st=copy.deepcopy(old['settings'])
    if (tuple(st[k] for k in ('population','budget','scenario_count','confidence','pool_size',
                              'progress_denominator','m_start_generation'))!=(250,5000,100,.9,8,50000,10)
            or st['milestones']!=BUDGETS or st['screen_seeds']!=SEEDS):
        raise ValueError('Source is not the agreed frozen single-model B5000 protocol')
    src=Path(old['shared_model']);bundle=read(src/'BUNDLE.json')
    if (bundle['checkpoints']!=['seed_991001_epoch_20.pt'] or bundle['normalization']!=old['normalization']
            or bundle['settings']['progress_denominator']!=50000):
        raise ValueError('Expected single shared member 991001, epoch 20, unchanged normalization')
    files={str(source/'PLAN.json'):digest(source/'PLAN.json')}
    model=root/'models/shared'
    for name in ['BUNDLE.json']+bundle['checkpoints']:
        if name!='BUNDLE.json' and digest(src/name)!=bundle['checkpoint_sha256'][name]:
            raise ValueError('Source model weight changed')
        if str(src/name) not in old['files'] or digest(src/name)!=old['files'][str(src/name)]:
            raise ValueError('Model is not the frozen source model: '+str(src/name))
        copy_frozen(src/name,model/name)
        files[str(src/name)]=digest(src/name);files[str(model/name)]=digest(model/name)
    s.SharedPolicy(model)  # Load/validate the actual one-member checkpoint before searches.
    for folder in ('formal_icm_s0_v1/src','component_screen_b20_v1','joint_icm_multiscale_v1',
                   'joint_icm_multiscale_terminal_v1','component_joint_b5000_v1',
                   'ppo_joint_b5000_pilot_v2','ppo_joint_b5000_screen27_v1',
                   'ppo_joint_b5000_base_report_v1','hv_joint_b5000_union_v1','network_transfer_b5000_v1'):
        for path in (EXP/folder).rglob('*'):
            if path.suffix not in ('.py','.csv','.json','.xlsx') or path.name=='checks.py':continue
            value=digest(path)
            if str(path) in old['files'] and old['files'][str(path)]!=value:
                raise ValueError('Historical runtime changed: '+str(path))
            files[str(path)]=value
    entries=prepare_inputs(root,s.p.base,s.p.np,s.profile)
    pre=read(root/'PREFLIGHT.json')
    if not pre['completed'] or not all(pre['checks'].values()):raise ValueError('Preflight not complete')
    for path in list((root/'inputs').rglob('*'))+list((root/'instances').glob('*.json')):
        if path.is_file():files[str(path)]=digest(path)
    for name in ('PREFLIGHT.json','PATH_LIBRARY.csv','PATH_OVERLAP.csv','BATCH_PATH_FEASIBILITY.csv','COMMON_SCENARIOS.csv'):
        files[str(root/name)]=digest(root/name)
    plan=dict(version=1,source=str(source),source_plan_sha256=old['sha256'],settings=st,
        uncertainty_profile=s.profile,normalization=old['normalization'],model=str(model),
        model_bundle_sha256=digest(model/'BUNDLE.json'),model_members=1,model_checkpoint=bundle['checkpoints'][0],
        gates=bundle['selected']['gates'],entries=entries,files=files,searches=27,training_updates=0,
        algorithm_seeds=SEEDS,preparation_seconds=pre['preparation_seconds'],
        reporting=dict(reference=[1.1,1.1,1.1],hv_divisor=1.331,
            bounds='For each batch set: union of both networks, all scheduled methods/seeds and 400/1000/5000 fronts',
            timing='Per-search time excludes the one-time cached path/scenario preparation; preparation reported separately',
            intervals='Paired seed bootstrap within each fixed batch set; 3 instances, nested demands; descriptive only'),
        purpose='Topology transfer: small Base/shared and large shared; no large Base; no model fitting or selection')
    plan['sha256']=canonical(plan);freeze(root/'PLAN.json',plan)
    print('[NETWORK] ALL FOUR CHECKS PASSED; 27 searches authorized; one frozen model',flush=True)
    return plan


def jobs(plan):
    return [dict(instance=e['instance_id'],pair_id=e['pair_id'],network=e['network'],seed=seed,arm=arm)
            for count in (10,20,30) for seed in SEEDS
            for e in plan['entries'] if e['K']==count
            for arm in (('BASE',SHARED) if e['network']=='small' else (SHARED,))]


def destination(root,job):
    return root/'runs'/job['instance']/f"seed_{job['seed']}"/job['arm']


def validate_run(r,plan,job):
    e=next(e for e in plan['entries'] if e['instance_id']==job['instance'])
    expected_model=plan['model_bundle_sha256'] if job['arm']==SHARED else None
    if (not r.get('completed') or r['plan_sha256']!=plan['sha256'] or r['task_sha256']!=canonical(job)
            or r['instance_digest']!=e['instance_digest'] or r['model_bundle_sha256']!=expected_model
            or r['path_library_sha256']!=e['path_library_sha256'] or r['cached_scenarios_sha256']!=e['scenarios_sha256']
            or r['real_objective_evaluation_count'] not in (4999,5000) or r['hidden_candidate_objective_evaluations']!=0
            or r['ppo_updates']!=0 or r['evaluation_budget']!=5000):
        raise ValueError('Run identity/budget validation failed: '+str(job))
    if sorted(x['evaluation'] for x in r['budget_fronts'])!=BUDGETS:
        raise ValueError('Missing budget snapshots')


def worker(root,plan,job,s):
    if job not in jobs(plan):raise ValueError('Unplanned search')
    verify(plan['files']);p=s.p;base=p.base;st=plan['settings']
    e=next(e for e in plan['entries'] if e['instance_id']==job['instance'])
    folder=Path(e['input_folder']);out=destination(root,job)
    if out.exists():
        if (out/'RUN_COMPLETE.json').exists():
            r=read(out/'RUN_COMPLETE.json');validate_run(r,plan,job);return
        # Keep interrupted artifacts for diagnosis; never delete an old attempt.
        out.rename(out.with_name(out.name+'.incomplete-'+str(time.time_ns())))
    policy=s.SharedPolicy(Path(plan['model'])) if job['arm']==SHARED else None
    scenarios=load_scenarios(base,p.np,folder);path_data=read(folder/'PATHS.json')
    expected_options=read(folder/'OPTION_PATH_IDS.json');counts=dict(paths=0,scenarios=0,options=0)
    originals={k:getattr(base,k) for k in ('build_path_library','build_scenario_set','build_reliable_path_options','clear_inherited_evaluation_cache')}
    signature=inspect.signature(originals['build_scenario_set'])
    def fixed_paths(nodes,regions,arcs,batches,timetable,lookup):
        counts['paths']+=1
        if [(b.batch_id,b.origin,b.destination,b.quantity,b.ET,b.LT) for b in batches] != [
                (b['batch_id'],b['origin'],b['destination'],b['quantity'],b['ET'],b['LT']) for b in e['batches']]:
            raise ValueError('Path request is not for the fixed batches')
        return load_paths(base,path_data,arcs)
    def fixed_scenarios(*args,**kwargs):
        q=signature.bind(*args,**kwargs);q.apply_defaults();v=q.arguments
        if not v['stochastic']:return originals['build_scenario_set'](*args,**kwargs)
        counts['scenarios']+=1
        if v['size']!=100 or v['seed']!=scenarios.seed or any(v[k]!=x for k,x in scenario_kwargs(plan['uncertainty_profile']).items()):
            raise ValueError('Search requested a different uncertainty profile')
        if {(a.from_node,a.to_node,a.mode) for a in v['arcs']}!=set(scenarios.travel_multiplier):
            raise ValueError('Scenario request is for another network')
        return scenarios
    def checked_options(*args,**kwargs):
        result=originals['build_reliable_path_options'](*args,**kwargs);counts['options']+=1
        ids={str(k[2]):[x.path.path_id for x in vv] for k,vv in result.items()}
        if ids!=expected_options:raise ValueError('Runtime path screening differs from preflight')
        return result
    holders=[]
    def factory(*args,**kwargs):
        ctl=p.BudgetController(*args,settings=st,arm=job['arm'],learned=policy,**kwargs);holders.append(ctl);return ctl
    base.build_path_library=fixed_paths;base.build_scenario_set=fixed_scenarios
    base.build_reliable_path_options=checked_options;base.clear_inherited_evaluation_cache=lambda *args:None
    started=datetime.now(timezone.utc).isoformat();wall=time.monotonic()
    try:
        result=p.run_episode(Path(e['workbook']),root/'instances'/(e['instance_id']+'.json'),out,None,
            p.legacy_learning.normalizer(plan['normalization']),algorithm_seed=job['seed'],
            evaluation_budget=5000,pop_size=250,generations=1000000,evaluation_mode='ccp',component='BASE',
            scenario_count=100,ccp_confidence=.9,uncertainty_profile='transport_border_saa_v1',
            inference_controller_factories={'BASE':factory},method_override=job['arm'])
    finally:
        for name,fn in originals.items():setattr(base,name,fn)
    if counts!={'paths':1,'scenarios':1,'options':1}:raise ValueError('Cached input hooks not used exactly once')
    ctl=holders[0];front=[list(p._optimisation_vector(x)) for x in read(out/'final_archive_lineage.json')]
    snaps=[x for x in ctl.budget_fronts if x['evaluation']!=5000]
    snaps.append(dict(evaluation=5000,actual_evaluations=result['real_objective_evaluation_count'],
        hv=result['final_hypervolume'],front_raw=front,controller_seconds=time.perf_counter()-ctl.start,
        definition='actual final feasible nondominated front'))
    result.update(instance=e['instance_id'],pair_id=e['pair_id'],configuration=e['configuration'],network=e['network'],
        seed=job['seed'],arm=job['arm'],phase='topology_transfer',stochastic=False,
        instance_digest=e['instance_digest'],batch_signature=e['batch_signature'],
        plan_sha256=plan['sha256'],task_sha256=canonical(job),
        model_bundle_sha256=plan['model_bundle_sha256'] if job['arm']==SHARED else None,
        path_library_sha256=e['path_library_sha256'],cached_scenarios_sha256=e['scenarios_sha256'],
        scenario_digest=read(out/'configuration.json')['scenario_digest'],
        budget_fronts=snaps,final_front_raw=front,cache_hook_calls=counts,
        hostname=socket.gethostname(),worker_started_utc=started,
        worker_finished_utc=datetime.now(timezone.utc).isoformat(),worker_wall_seconds=time.monotonic()-wall,
        torch_num_threads=p.torch.get_num_threads(),training_updates=0,
        final_solutions_sha256=digest(out/'final_feasible_nondominated.json'))
    validate_run(result,plan,job)
    for c in 'ICM':
        if c not in s.enabled_components(job['arm']) and result['selection'][c]['calls']:
            raise ValueError('Inactive component changed')
    write(out/'RUN_COMPLETE.json',result)


def report(root,plan,records,s,preview=False):
    from reports import create_report
    create_report(root,plan,records,s.p,preview)


def execute(a,s):
    root=a.out.resolve();source=a.source.resolve()
    if root==source or root in source.parents or source in root.parents:raise ValueError('Use a separate output directory')
    root.mkdir(parents=True,exist_ok=True)
    with (root/'.pipeline.lock').open('a+') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise SystemExit('This experiment is already running; inspect STATUS.json')
        processes=[];start=time.monotonic()
        def stop(signum,frame):raise KeyboardInterrupt('Signal '+str(signum))
        signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
        try:
            write(root/'STATUS.json',dict(completed=False,stage='four_preflight_checks',pid=os.getpid(),done=0,total=27))
            plan=prepare(root,source,s)
            if a.prepare_only:
                write(root/'STATUS.json',dict(completed=False,stage='preflight_complete',pid=os.getpid(),done=0,total=27))
                return
            pending=deque();done=[]
            for job in jobs(plan):
                path=destination(root,job)/'RUN_COMPLETE.json'
                if path.exists():
                    r=read(path);validate_run(r,plan,job);done.append(r)
                else:pending.append(job)
            last_print=0;preview_written=(root/'preview/RESULTS.csv').exists()
            while pending or processes:
                while pending and len(processes)<a.workers:
                    job=pending.popleft();log=root/'logs'/(canonical(job)+'.log');log.parent.mkdir(exist_ok=True)
                    f=log.open('a',encoding='utf-8')
                    proc=subprocess.Popen([sys.executable,'-B','-u',str(HERE/'run.py'),'--out',str(root),
                        '--worker','--job',str(jobs(plan).index(job))],stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
                    processes.append(dict(proc=proc,file=f,job=job,start=time.monotonic(),log=str(log)))
                    print('[NETWORK] start '+str(job)+' PID='+str(proc.pid),flush=True)
                for item in list(processes):
                    proc=item['proc']
                    if time.monotonic()-item['start']>max(7200,plan['settings'].get('task_timeout_seconds',7200)):
                        raise RuntimeError('Search timeout; see '+item['log'])
                    code=proc.poll()
                    if code is None:continue
                    item['file'].close();processes.remove(item)
                    if code:raise RuntimeError(f'Worker failed ({code}); see '+item['log'])
                    r=read(destination(root,item['job'])/'RUN_COMPLETE.json');validate_run(r,plan,item['job']);done.append(r)
                    print(f'[NETWORK] complete {len(done)}/27 {r["instance"]}/{r["arm"]}; {r["optimisation_seconds"]:.1f}s',flush=True)
                if not preview_written and len([r for r in done if r['configuration']=='B10'])==9:
                    report(root,plan,[r for r in done if r['configuration']=='B10'],s,preview=True);preview_written=True
                means={}
                for net,arm in [('small','BASE'),('small',SHARED),('large',SHARED)]:
                    ts=[r['optimisation_seconds'] for r in done if r['network']==net and r['arm']==arm]
                    means[net+'/'+arm]=mean(ts) if ts else (32. if arm=='BASE' else 138.)
                running=[dict(pid=x['proc'].pid,**x['job'],seconds=round(time.monotonic()-x['start']),log=x['log']) for x in processes]
                seconds=sum(means[j['network']+'/'+j['arm']] for j in pending)
                seconds+=sum(max(0,means[x['job']['network']+'/'+x['job']['arm']]-(time.monotonic()-x['start'])) for x in processes)
                write(root/'STATUS.json',dict(completed=False,stage='paired_searches',pid=os.getpid(),done=len(done),total=27,
                    workers=a.workers,running=running,elapsed_seconds=round(time.monotonic()-start,1),
                    remaining_estimate_hours=seconds/a.workers/3600,
                    estimate_note='Observed network/method means where available, otherwise historical initial estimates; excludes reporting',
                    training_updates=0))
                if time.monotonic()-last_print>=30:
                    print(f'[NETWORK] heartbeat {len(done)}/27; estimated remaining {seconds/a.workers/60:.1f} min',flush=True)
                    last_print=time.monotonic()
                if processes:time.sleep(2)
            if len(done)!=27:raise ValueError('Expected exactly 27 complete searches')
            verify(plan['files']);report(root,plan,done,s)
            write(root/'STATUS.json',dict(completed=True,stage='all',done=27,total=27,training_updates=0,
                elapsed_seconds=time.monotonic()-start,preparation_seconds=plan['preparation_seconds'],
                search_seconds_sum=sum(r['optimisation_seconds'] for r in done),hv_divisor=1.331))
            print('[NETWORK] COMPLETE '+str(root/'results'),flush=True)
        except BaseException as error:
            write(root/'STATUS.json',dict(completed=False,stage='paused' if isinstance(error,KeyboardInterrupt) else 'failed',
                error=str(error),resume='Repeat the same command; completed runs reused; interrupted attempts retained'))
            raise
        finally:
            for item in processes:
                proc=item['proc']
                if proc.poll() is None:
                    os.killpg(proc.pid,signal.SIGTERM)
                    try:proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
                item['file'].close()


def main():
    diag=EXP/'transformer_ppo_runs/diagnostics'
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=diag/'ccp_single_ppo_s0_s8_b5000_v1')
    parser.add_argument('--out',type=Path,default=diag/'ccp_network_transfer_single_b5000_v1')
    parser.add_argument('--workers',type=int,default=2)
    parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--worker',action='store_true');parser.add_argument('--job',type=int)
    a=parser.parse_args()
    if not 1<=a.workers<=4:parser.error('Use 1 to 4 worker processes')
    print('[NETWORK] loading unchanged search engine',flush=True);s=engine()
    if a.worker:
        plan=checked_plan(a.out.resolve()/'PLAN.json');worker(a.out.resolve(),plan,jobs(plan)[a.job],s)
    else:execute(a,s)


if __name__=='__main__':main()
