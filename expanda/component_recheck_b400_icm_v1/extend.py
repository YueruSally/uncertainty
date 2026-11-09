#!/usr/bin/env python3
"""Add two ICM arms to the frozen 700-run screen, then recompute nine-arm HV."""
import argparse
import fcntl
import importlib.util
from pathlib import Path
import signal
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('b400_legacy',HERE.parent/'component_recheck_b400_v1/run.py')
legacy=importlib.util.module_from_spec(spec);spec.loader.exec_module(legacy)
sys.path.insert(0,str(HERE.parent/'hv_instance_union_v1'))
import recompute as hv
from controller import CombinedRuleController
from combined import CombinedFrozenK8Controller
from mutation import ScaleSearchPolicy, FrozenScaleRanker
from transformer_ppo.archive import RewardArchive, RewardConfig
from transformer_ppo.episode import run_episode
from runtime_common import _normalizer
from transformer_ppo.instance_catalog import canonical_digest
from runtime_common import read_json,write_json,digest
import torch

RULE='I_C_M_K8_HEURISTIC';LEARN='I_C_M_K8_TRANSFORMER'
ADDED=(RULE,LEARN);ALL_METHODS=(*legacy.METHODS,*ADDED)
PROTOCOL='frozen-old-components-b400-icm-extension-v1'


def verify_files(files):
    for path,sha in files.items():
        if digest(Path(path))!=sha:raise ValueError(f'Frozen input changed: {path}')


def prepare(source,out):
    old=read_json(source/'PLAN.json')
    design=hv.experiment_design(old,source/'PLAN.json')
    if design['protocol']!='frozen-old-components-b400-recheck-v1':raise ValueError('Requires completed seven-method B400 expansion')
    return make_plan(source,out,old)


def make_plan(source,out,old):
    verify_files(old['files'])
    config=old['config']
    runtime=dict(old['files'])
    for path in [Path(__file__),HERE/'controller.py',HERE.parent/'hv_instance_union_v1/recompute.py',source/'PLAN.json']:
        runtime[str(path.resolve())]=digest(path)
    originals={}
    for n in old['instances']:
        for s in config['algorithm_seeds']:
            scenarios=set()
            for m in legacy.METHODS:
                folder=source/'runs'/n/f'seed_{s}'/m
                row=read_json(folder/'RUN_COMPLETE.json');legacy.validate(row,m,s,old['sha256'])
                if row['instance']!=n:raise ValueError('Mismatched original instance')
                scenarios.add(row['scenario_digest'])
                for filename in ('RUN_COMPLETE.json','configuration.json','final_feasible_nondominated.json'):
                    path=folder/filename;originals[str(path.resolve())]=digest(path)
            if len(scenarios)!=1:raise ValueError('Original scenarios are not paired')
    import platform,numpy
    env={'python':platform.python_version(),'numpy':str(numpy.__version__),'torch':str(torch.__version__)}
    if old.get('environment')!=env:raise ValueError('Use the same Python/NumPy/PyTorch versions as the original 700 runs')
    merged={**config,'protocol':PROTOCOL,'methods':list(ALL_METHODS)}
    plan={'config':merged,'instances':old['instances'],'runs':len(old['instances'])*len(config['algorithm_seeds'])*9,
          'new_runs':len(old['instances'])*len(config['algorithm_seeds'])*2,
          'source_root':str(source.resolve()),'source_plan_sha256':old['sha256'],
          'source_plan_file_sha256':digest(source/'PLAN.json'),
          'runtime_files':runtime,'original_result_files':originals,'environment':env,
          'qualification':'Reuse seven arms unchanged; both combined arms defer actual mutation until M stage. No training, PPO, hidden evaluation or final-test opening.'}
    plan['sha256']=canonical_digest(plan)
    legacy.freeze(out/'PLAN.json',plan)
    return plan


def validate(row,method,seed,sha):
    legacy.validate(row,method,seed,sha)
    if row.get('combined_hidden_candidate_objective_evaluations')!=0:raise ValueError('Hidden combined evaluation')
    parts=row['component_summaries']
    if parts['I']['initialization_k8_pool_events']!=20:raise ValueError('Incomplete combined initialization')
    expected_i='balanced' if method==RULE else 'transformer'
    expected_c='simple' if method==RULE else 'transformer'
    if parts['I']['initialization_k8_selection_mode']!=expected_i or parts['C']['crossover_k8_selection_mode']!=expected_c:
        raise ValueError('Combined selectors do not match assigned arm')
    if parts['M']['k8_scale_search']['hidden_candidate_objective_evaluations']!=0:raise ValueError('Hidden M evaluation')


def worker(out,n,s,m,folder,sha):
    plan=read_json(out/'PLAN.json');source=Path(plan['source_root']);cfg=plan['config']
    if sha!=plan['sha256'] or n not in plan['instances'] or s not in cfg['algorithm_seeds'] or m not in ADDED:
        raise ValueError('Worker not in frozen extension plan')
    verify_files(plan['runtime_files']);torch.set_num_threads(1)
    learned=m==LEARN
    predictor=FrozenScaleRanker(read_json(source/'models/M_BUNDLE.json'),digest) if learned else None
    policy=ScaleSearchPolicy(n,s,'TRANSFORMER_K8_SIZE200' if learned else 'HEURISTIC_BALANCED',predictor=predictor)
    def factory(*args,**kwargs):
        norm=kwargs.pop('normalizer');reliable=kwargs.pop('reliable_options');batches=kwargs.pop('batches');arcs=kwargs.pop('arcs')
        common=dict(trainer=policy,archive=RewardArchive(norm),reward_config=RewardConfig(),batches=batches,
                    generations=cfg['maximum_generations'],pop_size=cfg['population_size'],graph_observation={},
                    reliable_options=reliable,arcs=arcs,**kwargs)
        if learned:
            return CombinedFrozenK8Controller(*args,i_bundle=source/'models/I_BUNDLE.json',c_bundle=source/'models/C_BUNDLE.json',**common)
        return CombinedRuleController(*args,**common)
    result=run_episode(legacy.PACKAGE/'assets/data/data_expanded.xlsx',source/'instances'/f'{n}.json',folder,None,
            _normalizer(legacy.PACKAGE/'assets/reward_normalization.json'),algorithm_seed=s,
            evaluation_budget=400,pop_size=20,generations=1000,evaluation_mode='ccp',component='BASE',
            scenario_count=100,ccp_confidence=.9,uncertainty_profile='transport_border_saa_v1',
            inference_controller_factories={'BASE':factory},method_override=m)
    result.update(strategy=m,instance=n,algorithm_seed=s,plan_sha256=sha,
                  scenario_digest=read_json(folder/'configuration.json')['scenario_digest'],
                  new_training_steps=0,new_ppo_updates=0,learning_enabled=False,
                  role='frozen combined arms added to B400 development replication')
    validate(result,m,s,sha)
    reference=read_json(source/'runs'/n/f'seed_{s}'/'BASE/RUN_COMPLETE.json')
    if result['scenario_digest']!=reference['scenario_digest']:raise ValueError('Combined run scenarios differ from original Base')
    write_json(folder/'RUN_COMPLETE.json',result)


def finish(out,plan,rows):
    verify_files(plan['runtime_files']);verify_files(plan['original_result_files'])
    target=out.parent/(out.name+'_hv_union_v1')
    if target.exists():
        manifest=read_json(target/'INPUT_MANIFEST.json')
        if Path(manifest['source']).resolve()!=out.resolve():raise ValueError('Existing HV result has a different source')
        verify_files(manifest['files']);result=read_json(target/'QUICK_RESULT.json')
        if not result['completed'] or result['runs']!=plan['runs']:raise ValueError('Incomplete existing HV result')
    else:
        staging=target.parent/(target.name+f'_building_{time.time_ns()}')
        result=hv.execute(out,staging)
        staging.rename(target)
    summary={'completed':True,'new_searches':len(rows),'reused_searches':plan['runs']-len(rows),
             'new_search_objective_evaluations':sum(r['real_objective_evaluation_count'] for r in rows),
             'hidden_candidate_objective_evaluations':0,'new_training_steps':0,'new_ppo_updates':0,
             'hv_recalculation_output':str(target),'methods':result['methods'],
             'interrupted_attempts':len(list((out/'interrupted').glob('*'))),
             'cost_qualification':'Completed-run costs; partial interrupted attempts may add cost.',
             'next_action':'Review all nine methods under the same per-instance normalization; no automatic PPO.'}
    write_json(out/'QUICK_RESULT.json',summary)
    (out/'RESULT_TABLE.csv').write_bytes((target/'RESULT_TABLE.csv').read_bytes())
    return summary


def execute_jobs(out,plan,workers):
    jobs=[(n,s,m) for n in plan['instances'] for s in plan['config']['algorithm_seeds'] for m in ADDED]
    active={};rows=[];started=time.monotonic();last=0
    def interrupt(*_):raise KeyboardInterrupt('Interrupted; same command resumes completed runs')
    handlers={sig:signal.signal(sig,interrupt) for sig in (signal.SIGINT,signal.SIGTERM)}
    try:
        while jobs or active:
            if time.monotonic()-started>14400:raise TimeoutError('Four-hour invocation limit; resume with same command')
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
                print(f'[B400-ICM] start {n}/{s}/{m} PID={proc.pid}',flush=True)
            for folder,(proc,handle,began,log) in list(active.items()):
                if time.monotonic()-began>900:raise TimeoutError(f'Worker exceeded 15 minutes: {log}')
                if proc.poll() is None:continue
                handle.close();del active[folder]
                if proc.returncode:raise RuntimeError(f'Worker failed: {log}\n'+log.read_text(errors='replace')[-4000:])
                row=read_json(Path(folder)/'RUN_COMPLETE.json');rows.append(row)
                print(f'[B400-ICM] complete {len(rows)}/{plan["new_runs"]} {row["strategy"]} HV={row["final_hypervolume"]:.7g}',flush=True)
            if time.monotonic()-last>=30:
                write_json(out/'STATUS.json',{'completed':False,'finished':len(rows),'total':plan['new_runs'],'running':len(active)})
                print(f'[B400-ICM] heartbeat {len(rows)}/{plan["new_runs"]} running={len(active)}',flush=True);last=time.monotonic()
            if active:time.sleep(.2)
        print('[B400-ICM] recomputing all nine methods with shared per-instance bounds',flush=True)
        result=finish(out,plan,rows)
        write_json(out/'STATUS.json',{'completed':True,'finished':len(rows),'total':plan['new_runs']})
        print('[B400-ICM] COMPLETE; see RESULT_TABLE.csv and QUICK_RESULT.json',flush=True)
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
    p.add_argument('--source',type=Path,default=diag/'ccp_component_recheck_b400_v1')
    p.add_argument('--out',type=Path,default=diag/'ccp_component_recheck_b400_icm_v1')
    p.add_argument('--workers',type=int,choices=(1,2),default=1)
    p.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    p.add_argument('--instance');p.add_argument('--seed',type=int);p.add_argument('--method',choices=ADDED)
    p.add_argument('--run-out',type=Path);p.add_argument('--sha')
    a=p.parse_args();out=a.out.resolve();source=a.source.resolve()
    if a.worker:worker(out,a.instance,a.seed,a.method,a.run_out,a.sha);return
    if source==out or source in out.parents or out in source.parents:raise ValueError('Use a separate sibling output')
    out.mkdir(parents=True,exist_ok=True)
    with (out/'.pipeline.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        plan=prepare(source,out)
        if (out/'QUICK_RESULT.json').exists():
            print('[B400-ICM] already complete; existing RESULT_TABLE.csv retained',flush=True);return
        print(f'[B400-ICM] reuse 700 searches; add {plan["new_runs"]}; models frozen; no PPO',flush=True)
        execute_jobs(out,plan,a.workers)

if __name__=='__main__':main()
