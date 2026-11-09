#!/usr/bin/env python3
"""Add only 30 combined-heuristic searches to the original 240-run experiment."""
import argparse
import fcntl
import importlib.util
from pathlib import Path
import signal
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'component_recheck_b400_icm_v1'))
spec=importlib.util.spec_from_file_location('icm_extension_runtime',HERE.parent/'component_recheck_b400_icm_v1/extend.py')
runtime=importlib.util.module_from_spec(spec);spec.loader.exec_module(runtime)
sys.path.insert(0,str(HERE))
import report
from runtime_common import read_json,write_json,digest,_normalizer
from transformer_ppo.instance_catalog import canonical_digest
from transformer_ppo.archive import RewardArchive,RewardConfig
from transformer_ppo.episode import run_episode
from mutation import ScaleSearchPolicy
import torch
import numpy

PACKAGE=runtime.legacy.PACKAGE
RULE=report.RULE
ADDED=(RULE,)
validate=runtime.validate
verify_files=runtime.verify_files


def prepare(source,out,workers):
    old=read_json(source/'PLAN.json')
    design,rows,originals=report.hv.load_runs(source)
    if design['protocol']!='icm-common-seed-component-screen-v1':
        raise ValueError('Use the original completed 240-run r3 experiment, not the 700/900-run expansion')
    expected={'scenario_count':100,'ccp_confidence':.9,'population_size':20,'evaluation_budget':400,'candidate_pool':8}
    if any(old.get(k)!=v for k,v in expected.items()):raise ValueError('Original experiment settings do not match')
    env={'torch':str(torch.__version__),'numpy':str(numpy.__version__)}
    if env!={'torch':old['torch_version'],'numpy':old['numpy_version']}:
        raise ValueError('Use original PyTorch/NumPy versions: '+str({'torch':old['torch_version'],'numpy':old['numpy_version']}))
    if workers!=old['workers']:raise ValueError('Use original parallelism for timing: '+str(old['workers']))
    core=[PACKAGE/'src/baseline_uncertainty.py',PACKAGE/'assets/data/data_expanded.xlsx',
          PACKAGE/'src/data/border_crossing_events.csv',PACKAGE/'assets/reward_normalization.json']
    core += [PACKAGE/'assets/search_instances'/f'{n}.json' for n in design['instances']]
    original_hashes=set(old['input_sha256'].values())
    for path in core:
        if digest(path) not in original_hashes:raise ValueError('Frozen engine/input differs from original experiment: '+str(path))
    files=core+list((PACKAGE/'src').rglob('*.py'))
    files += [Path(__file__),HERE/'report.py',HERE.parent/'component_recheck_b400_icm_v1/controller.py',
              HERE.parent/'component_recheck_b400_icm_v1/extend.py',HERE.parent/'component_recheck_b400_v1/run.py',
              HERE.parent/'hv_instance_union_v1/recompute.py']
    config={**expected,'algorithm_seeds':design['algorithm_seeds'],'methods':list(report.METHODS),
            'maximum_generations':1000,'protocol':report.PROTOCOL,'workers':workers}
    plan={'config':config,'instances':design['instances'],'runs':270,'new_runs':30,
          'source_root':str(source),'source_plan_sha256':design['plan_sha'],
          'runtime_files':{str(p.resolve()):digest(p) for p in files},
          'original_result_files':originals,'environment':env,
          'qualification':'Original 240 runs unchanged; 30 new combined heuristic runs. No training or PPO. Historical initialization and mutation activation retained.'}
    plan['sha256']=canonical_digest(plan)
    runtime.legacy.freeze(out/'PLAN.json',plan)
    return plan


def worker(out,n,s,m,folder,sha):
    plan=read_json(out/'PLAN.json');source=Path(plan['source_root'])
    if sha!=plan['sha256'] or n not in plan['instances'] or s not in plan['config']['algorithm_seeds'] or m!=RULE:
        raise ValueError('Worker not in frozen plan')
    verify_files(plan['runtime_files']);torch.set_num_threads(1)
    policy=ScaleSearchPolicy(n,s,'HEURISTIC_BALANCED',predictor=None)
    def factory(*args,**kwargs):
        norm=kwargs.pop('normalizer');reliable=kwargs.pop('reliable_options');batches=kwargs.pop('batches');arcs=kwargs.pop('arcs')
        return runtime.CombinedRuleController(*args,trainer=policy,archive=RewardArchive(norm),reward_config=RewardConfig(),
            batches=batches,generations=1000,pop_size=20,graph_observation={},reliable_options=reliable,arcs=arcs,**kwargs)
    result=run_episode(PACKAGE/'assets/data/data_expanded.xlsx',PACKAGE/'assets/search_instances'/f'{n}.json',folder,None,
        _normalizer(PACKAGE/'assets/reward_normalization.json'),algorithm_seed=s,evaluation_budget=400,pop_size=20,
        generations=1000,evaluation_mode='ccp',component='BASE',scenario_count=100,ccp_confidence=.9,
        uncertainty_profile='transport_border_saa_v1',inference_controller_factories={'BASE':factory},method_override=m)
    cfg=read_json(folder/'configuration.json')
    result.update(strategy=m,instance=n,algorithm_seed=s,plan_sha256=sha,scenario_digest=cfg['scenario_digest'],
                  new_training_steps=0,new_ppo_updates=0,learning_enabled=False)
    validate(result,m,s,sha)
    base=read_json(source/'runs'/n/f'seed_{s}'/'BASE/configuration.json')
    validate_pairing(cfg,base)
    write_json(folder/'RUN_COMPLETE.json',result)


def validate_pairing(cfg,base):
    for key in ('scenario_digest','instance_digest','normalization','seeds','population','evaluation_budget'):
        if cfg[key]!=base[key]:raise ValueError('Combined run differs from original Base: '+key)


def finish(out,plan,rows):
    verify_files(plan['runtime_files']);verify_files(plan['original_result_files'])
    target=out.parent/(out.name+'_hv_union_v1')
    if target.exists():
        manifest=read_json(target/'INPUT_MANIFEST.json')
        if Path(manifest['source']).resolve()!=out.resolve():raise ValueError('Existing report has another source')
        verify_files(manifest['files']);result=read_json(target/'QUICK_RESULT.json')
        if not result['completed'] or result['runs']!=270:raise ValueError('Incomplete report')
    else:
        staging=target.parent/(target.name+f'_building_{time.time_ns()}')
        result=report.execute(out,staging);staging.rename(target)
    for name in ('RESULT_TABLE.csv','STATISTICS.csv','PER_INSTANCE_STATISTICS.csv','00_组件对比及运行时间.csv','统计口径.txt'):
        (out/name).write_bytes((target/name).read_bytes())
    summary={'completed':True,'new_searches':len(rows),'reused_searches':240,'total_searches':270,
        'new_search_objective_evaluations':sum(r['real_objective_evaluation_count'] for r in rows),
        'hidden_candidate_objective_evaluations':0,'new_training_steps':0,'new_ppo_updates':0,
        'hv_recalculation_output':str(target),'methods':result['methods'],
        'interrupted_attempts':len(list((out/'interrupted').glob('*'))),
        'cost_qualification':'Completed-run cost; interrupted attempts may add evaluations.',
        'next_action':'Report nine arms with shared instance bounds; no automatic training or PPO.'}
    write_json(out/'QUICK_RESULT.json',summary)
    return summary


def execute_jobs(out,plan,workers):
    jobs=[(n,s,m) for n in plan['instances'] for s in plan['config']['algorithm_seeds'] for m in ADDED]
    active={};rows=[];started=time.monotonic();last=0
    def interrupt(*_):raise KeyboardInterrupt('Interrupted; same command resumes completed runs')
    handlers={sig:signal.signal(sig,interrupt) for sig in (signal.SIGINT,signal.SIGTERM)}
    try:
        while jobs or active:
            if time.monotonic()-started>7200:raise TimeoutError('Two-hour invocation limit; resume with same command')
            while jobs and len(active)<workers:
                n,s,m=jobs.pop(0);folder=out/'runs'/n/f'seed_{s}'/m
                if (folder/'RUN_COMPLETE.json').exists():
                    row=read_json(folder/'RUN_COMPLETE.json');validate(row,m,s,plan['sha256'])
                    if row['instance']!=n:raise ValueError('Wrong resumed instance')
                    rows.append(row);continue
                if folder.exists():
                    dest=out/'interrupted'/f'{n}_{s}_{m}_{time.time_ns()}';dest.parent.mkdir(exist_ok=True);folder.rename(dest)
                folder.mkdir(parents=True)
                log=out/'logs'/f'{n}_{s}_{m}_{time.time_ns()}.log';log.parent.mkdir(exist_ok=True);handle=log.open('w')
                proc=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),'--out',str(out),'--worker',
                    '--instance',n,'--seed',str(s),'--method',m,'--run-out',str(folder),'--sha',plan['sha256']],stdout=handle,stderr=subprocess.STDOUT)
                active[str(folder)]=(proc,handle,time.monotonic(),log)
                print(f'[ICM-270] start {n}/{s}/{m} PID={proc.pid}',flush=True)
            for folder,(proc,handle,began,log) in list(active.items()):
                if time.monotonic()-began>900:raise TimeoutError(f'Worker exceeded 15 minutes: {log}')
                if proc.poll() is None:continue
                handle.close();del active[folder]
                if proc.returncode:raise RuntimeError(f'Worker failed: {log}\n'+log.read_text(errors='replace')[-4000:])
                row=read_json(Path(folder)/'RUN_COMPLETE.json');rows.append(row)
                print(f'[ICM-270] complete {len(rows)}/{plan["new_runs"]} {row["strategy"]} HV={row["final_hypervolume"]:.7g}',flush=True)
            if time.monotonic()-last>=30:
                write_json(out/'STATUS.json',{'completed':False,'finished':len(rows),'total':plan['new_runs'],'running':len(active)})
                print(f'[ICM-270] heartbeat {len(rows)}/{plan["new_runs"]} running={len(active)}',flush=True);last=time.monotonic()
            if active:time.sleep(.2)
        print('[ICM-270] recomputing all nine methods with shared per-instance bounds',flush=True)
        result=finish(out,plan,rows)
        write_json(out/'STATUS.json',{'completed':True,'finished':len(rows),'total':plan['new_runs']})
        print('[ICM-270] COMPLETE; see RESULT_TABLE.csv and QUICK_RESULT.json',flush=True)
        return result
    except BaseException as exc:
        write_json(out/'STATUS.json',{'completed':False,'finished':len(rows),'error':str(exc)})
        raise
    finally:
        for proc,handle,*_ in active.values():
            if proc.poll() is None:proc.terminate()
        for proc,handle,*_ in active.values():
            try:proc.wait(timeout=10)
            except subprocess.TimeoutExpired:proc.kill();proc.wait()
            handle.close()
        for sig,handler in handlers.items():signal.signal(sig,handler)


def main():
    diag=HERE.parent/'transformer_ppo_runs/diagnostics'
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=diag/'ccp_icm_common_seed_v1_3inst_10seed_b400_r3')
    p.add_argument('--out',type=Path,default=diag/'ccp_icm_heuristic_270_v1')
    p.add_argument('--workers',type=int,choices=(3,),default=3)
    p.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    p.add_argument('--instance');p.add_argument('--seed',type=int);p.add_argument('--method',choices=ADDED)
    p.add_argument('--run-out',type=Path);p.add_argument('--sha')
    a=p.parse_args();source=a.source.resolve();out=a.out.resolve()
    if a.worker:worker(out,a.instance,a.seed,a.method,a.run_out,a.sha);return
    if source==out or source in out.parents or out in source.parents:raise ValueError('Use a separate sibling output')
    out.mkdir(parents=True,exist_ok=True)
    with (out/'.pipeline.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        plan=prepare(source,out,a.workers)
        print('[ICM-270] verified original 240 searches; add only 30 ICM heuristic searches; no training or PPO',flush=True)
        execute_jobs(out,plan,a.workers)

if __name__=='__main__':main()
