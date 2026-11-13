#!/usr/bin/env python3
"""Finish four distinct frozen strategies; explicitly alias selected PPO update 0.

New scheduling/reporting only. Original worker code, PLAN, models and completed
RUN_COMPLETE records are unchanged. No synthetic PPO RUN_COMPLETE is created.
"""
from pathlib import Path
import argparse
import fcntl
import importlib.util
import json
import os
import signal
import sys
import time
import threading

HERE = Path(__file__).resolve().parent
EXPANDA = HERE.parent
LEGACY = EXPANDA/'joint_icm_multiscale_terminal_v1'
UNIQUE = ('BASE','ICM_HEURISTIC','ICM_INDEPENDENT','ICM_SHARED_SUPERVISED')
PPO = 'ICM_TERMINAL_PPO'
SHARED = 'ICM_SHARED_SUPERVISED'


def load_runtime():
    for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
        os.environ[key] = '1'
    sys.path.insert(0, str(LEGACY))
    spec = importlib.util.spec_from_file_location('frozen_multiscale_runner', LEGACY/'run.py')
    runner = importlib.util.module_from_spec(spec); spec.loader.exec_module(runner)
    import reporting
    import torch
    torch.set_num_threads(1)
    return runner, reporting


def identity(root, runner):
    from policy import TerminalPolicy
    from shared import SharedPolicy
    import torch
    plan = runner.read_json(root/'PLAN.json')
    if runner.canonical_digest({k:v for k,v in plan.items() if k != 'sha256'}) != plan['sha256']:
        raise ValueError('Invalid original frozen plan')
    runner.verify(plan['files'])
    selected = runner.read_json(root/'terminal/SELECTED.json')
    if selected['update'] != 0 or Path(selected['model']).resolve() != root/'terminal/checkpoints/update_000':
        raise ValueError('Only selected update 0 can be reused')
    if runner.digest(Path(selected['model'])/'BUNDLE.json') != selected['bundle_sha256']:
        raise ValueError('Selected model changed')
    shared = SharedPolicy(Path(plan['source'])/'shared_model')
    zero = TerminalPolicy(Path(selected['model']))
    for key in ('scalers','normalization','dimensions','target_scales','feature_version'):
        if shared.bundle[key] != zero.bundle[key]:
            raise ValueError('Different model input/output definition: '+key)
    if shared.bundle['selected']['gates'] != zero.bundle['selected']['gates']:
        raise ValueError('Different gates')
    if len(shared.models) != len(zero.models) or not all(
        torch.equal(v, b.state_dict()[k]) for a,b in zip(shared.models, zero.models)
        for k,v in a.state_dict().items()):
        raise ValueError('Shared and update 0 actor parameters differ')
    return plan, selected


def jobs(root, plan, selected, runner):
    model = {'BASE':None,'ICM_HEURISTIC':None,'ICM_INDEPENDENT':root/'independent_model',
             SHARED:Path(plan['source'])/'shared_model',PPO:Path(selected['model'])}
    return [(runner.task(plan,r['instance_id'],seed,m,'screen',model[m]),
             root/'screen'/r['instance_id']/f'seed_{seed}'/m)
            for r in plan['screen'] for seed in plan['settings']['screen_seeds'] for m in (*UNIQUE,PPO)]


def completed(all_jobs, plan, runner):
    found = []
    for t,folder in all_jobs:
        r = runner.valid_completion(folder,t,plan)
        if r is not None:
            found.append((r,folder/'RUN_COMPLETE.json'))
    return found


def check_reproductions(found):
    indexed = {(r['instance'],r['seed'],r['arm']):(r,p) for r,p in found}
    checks = []
    keys = ('final_hypervolume','final_front_raw','evaluation_order_digest',
            'real_objective_evaluation_count','selection','scenario_digest','instance_digest','milestone_hv')
    for (name,seed,arm),(r,p) in indexed.items():
        if arm != PPO or (name,seed,SHARED) not in indexed:
            continue
        ref,refp = indexed[name,seed,SHARED]
        if any(k not in r or k not in ref or r[k] != ref[k] for k in keys):
            raise ValueError(f'Observed PPO-u0 reproduction differs: {name}/{seed}')
        checks.append({'instance':name,'seed':seed,'all_checked_exact':True,
                       'ppo_run':str(p),'shared_run':str(refp)})
    if len(checks) < 2:
        raise ValueError('Require at least two completed exact paired reproduction checks')
    return checks


def alias_map(found, root, plan, runner):
    indexed = {(r['instance'],r['seed'],r['arm']):(r,p) for r,p in found}
    result = []
    for entry in plan['screen']:
        for seed in plan['settings']['screen_seeds']:
            name = entry['instance_id'];ref,refp = indexed[name,seed,SHARED]
            own = indexed.get((name,seed,PPO))
            result.append({'instance':name,'seed':seed,'requested_method':PPO,'alias_of':SHARED,
                           'metric_source':str(refp),'metric_source_sha256':runner.digest(refp),
                           'ppo_physically_executed':own is not None,
                           'ppo_run_file':str(own[1]) if own else '',
                           'ppo_run_sha256':runner.digest(own[1]) if own else '',
                           'qualification':'Identical selected u0 policy; reused metric is not a new measurement.'})
    return result


def write_report(found, root, out, plan, selected, runner, reporting):
    import csv
    import numpy as np
    checks = check_reproductions(found)
    unique = [r for r,p in found if r['arm'] in UNIQUE]
    expected = len(plan['screen'])*len(plan['settings']['screen_seeds'])*len(UNIQUE)
    if len(unique) != expected:
        raise ValueError('Four distinct strategy comparisons are incomplete')
    previous = reporting.METHODS
    try:
        reporting.METHODS = UNIQUE
        q = reporting.report(unique,plan,out,selected)
    finally:
        reporting.METHODS = previous
    aliases = alias_map(found,root,plan,runner)
    reporting.csv_write(out/'POLICY_REUSE.csv',aliases)
    reporting.csv_write(out/'UPDATE0_REPRODUCTION.csv',checks)
    actual_ppo = [r for r,p in found if r['arm'] == PPO]
    measured_keys = ('instance','configuration','seed','arm','final_hypervolume',
                     'optimisation_seconds','real_objective_evaluation_count')
    reporting.csv_write(out/'PPO_MEASURED_CHECKS.csv', [{k:r[k] for k in measured_keys} for r in actual_ppo])
    tablefile = out/'00_联合训练对比.csv'
    with tablefile.open(encoding='utf-8-sig',newline='') as f:
        table = list(csv.DictReader(f))
    pairs = len(aliases)
    for row in table:
        row.update(结果来源='实际运行',实测运行数=pairs,未执行而复用数=0)
    row = dict(table[-1])
    row.update(方法='ICM 终局PPO（选择u0；同策略复用）',结果来源='HV复用共享监督；不是独立实验组',
               实测运行数=len(actual_ppo),未执行而复用数=pairs-len(actual_ppo))
    row['平均搜索时间(秒)'] = ''  # Never pass shared runtime off as measured PPO runtime.
    table.append(row);reporting.csv_write(tablefile,table)
    physical = [r for r,p in found]
    q.update(original_planned_searches=pairs*5,distinct_policy_searches=len(unique),
             physically_completed_screen_searches=len(physical),
             skipped_duplicate_searches=pairs-len(actual_ppo),
             screen_searches=len(physical),
             screen_objective_evaluations=sum(r['real_objective_evaluation_count'] for r in physical),
             screen_optimization_seconds_sum=sum(r['optimisation_seconds'] for r in physical),
             aliases={PPO:{'alias_of':SHARED,'selected_update':0,'metric_rows':pairs,
                          'actual_reproduction_runs':len(actual_ppo),'unexecuted_reused_rows':pairs-len(actual_ppo),
                          'measured_reproduction_mean_seconds':float(np.mean([r['optimisation_seconds'] for r in actual_ppo]))}},
             new_training_steps=0,new_ppo_updates=0,
             source_experiment=str(root),continuation_output=str(out.parent))
    q['qualification'] += (' Four distinct policies were executed. The selected PPO-u0 row aliases shared supervision; '
                           'unexecuted aliases have no fabricated RUN_COMPLETE, runtime or objective evaluations. '
                           'PER_RUN and STATISTICS contain distinct measured strategies only; measured PPO checks are separate.')
    runner.write_json(out/'QUICK_RESULT.json',q)
    return q


def take_lock(file):
    try:
        fcntl.flock(file,fcntl.LOCK_EX|fcntl.LOCK_NB)
        return True
    except BlockingIOError:
        return False


def execute(a):
    from drain import handoff, recover, processes, matches, active
    root=a.source.resolve();out=a.out.resolve()
    if root == out or root in out.parents or out in root.parents:
        raise ValueError('Continuation output must be separate')
    out.mkdir(parents=True,exist_ok=True)
    def state(r):
        tmp=out/'STATUS.json.tmp'
        tmp.write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n')
        tmp.replace(out/'STATUS.json')
    with (out/'.continuation.lock').open('a+') as own:
        if not take_lock(own):
            raise RuntimeError('This continuation already has an active process')
        a.owns_output_lock = True
        if a.recover:
            recover(json.loads((out/'DRAIN.json').read_text()));return
        runner,reporting=load_runtime()
        if not (root/'PLAN.json').exists():
            raise ValueError('Source experiment missing')
        def stop(signum,frame):
            raise runner.Paused(f'Received signal {signum}')
        signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
        plan,selected=identity(root,runner)
        all_jobs=jobs(root,plan,selected,runner)
        check_reproductions(completed(all_jobs,plan,runner))
        amendment={'source':str(root),'original_plan_sha256':plan['sha256'],
                   'selected':selected,'unique_methods':list(UNIQUE),'alias_method':PPO,'alias_of':SHARED,
                   'reason':'Selected update0 actor/gates match shared supervision; exact paired reproduction checked.',
                   'original_files':plan['files'],
                   'continuation_code_sha256':{str(p):runner.digest(p) for p in HERE.glob('*.py')}}
        runner.freeze(out/'CONTINUATION_PLAN.json',amendment)
        with (root/'.pipeline.lock').open('a+') as lock:
            if not take_lock(lock):
                if not a.take_over:
                    raise RuntimeError('Original supervisor active; use --take-over to finish its workers before handoff')
                old=runner.read_json(root/'STATUS.json')
                if old.get('stage') != 'screen':
                    raise ValueError('Only the completed-training screen stage can be handed off')
                def drain_state(r):
                    # Keep a recovery identity across all subsequent handoff states.
                    if 'supervisor' in r:
                        r=dict(r,supervisor_pid=r['supervisor']['pid'],supervisor_start=r['supervisor']['start'])
                    runner.write_json(out/'DRAIN.json',r);state(r)
                    print('[MULTI-RESUME] '+json.dumps(r,ensure_ascii=False),flush=True)
                handoff(root,LEGACY/'run.py',drain_state,plan['settings']['task_timeout_seconds'])
                until=time.monotonic()+60
                while not take_lock(lock):
                    if time.monotonic()>until:
                        raise RuntimeError('Old supervisor did not release its lock; no new worker launched')
                    time.sleep(.5)
            # All source work from this point is serialized by the ORIGINAL lock.
            if Path('/proc/self/stat').exists():
                orphans=[r for r in processes() if active(r) and matches(r,LEGACY/'run.py',root,True)]
                if orphans:
                    raise RuntimeError('Existing source workers still active; no new worker launched: '+
                                       str([r['pid'] for r in orphans]))
            plan2,selected2=identity(root,runner)
            if plan2 != plan or selected2 != selected:
                raise ValueError('Frozen identity changed during handoff')
            before=completed(all_jobs,plan,runner);check_reproductions(before)
            before_paths={str(p):runner.digest(p) for r,p in before}
            runner.write_json(out/'PREEXISTING_RUNS.json',before_paths)
            wanted=[j for j in all_jobs if j[0]['arm'] in UNIQUE]
            have=sum(r['arm'] in UNIQUE for r,p in before)
            print(f'[MULTI-RESUME] distinct policies {have}/{len(wanted)} complete; '
                  f'{len(wanted)-have} searches remain; no new PPO training',flush=True)
            halt=threading.Event()
            def mirror():
                while not halt.wait(5):
                    try:
                        s=runner.read_json(root/'STATUS.json')
                        state(dict(s,continuation_output=str(out),original_plan_unchanged=True))
                    except (OSError,ValueError):
                        pass
            monitor=threading.Thread(target=mirror,daemon=True);monitor.start()
            try:
                runner.execute_jobs(root,plan,wanted,a.workers,'screen/four_distinct_policies')
            finally:
                halt.set();monitor.join()
            runner.verify(before_paths)
            found=completed(all_jobs,plan,runner)
            q=write_report(found,root,out/'results',plan,selected,runner,reporting)
            new=[r for r,p in found if str(p) not in before_paths]
            q.update(new_screen_searches_this_invocation=len(new),
                     new_objective_evaluations_this_invocation=sum(r['real_objective_evaluation_count'] for r in new))
            runner.write_json(out/'results/QUICK_RESULT.json',q)
            runner.write_json(root/'STATUS.json',{'completed':False,'stage':'continued_with_duplicate_policy_reuse',
                'requested_comparison_complete':True,'continuation_output':str(out),
                'physically_completed_screen_searches':len(found),
                'original_planned_searches':q['original_planned_searches'],
                'skipped_duplicate_searches':q['skipped_duplicate_searches'],
                'qualification':'Original five-arm grid not all executed; see separate continuation report.'})
            state({'completed':True,'stage':'all','result':str(out/'results/QUICK_RESULT.json'),
                   'physically_completed_screen_searches':len(found),
                   'skipped_duplicate_searches':q['skipped_duplicate_searches'],'final_test_opened':False})
            print('[MULTI-RESUME] COMPLETE '+json.dumps({'actual_screen_runs':len(found),
                  'skipped_duplicates':q['skipped_duplicate_searches'],'results':str(out/'results')},ensure_ascii=False),flush=True)


def main():
    diag=EXPANDA/'transformer_ppo_runs/diagnostics'
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=diag/'ccp_icm_multiscale_joint_v1')
    p.add_argument('--out',type=Path,default=diag/'ccp_icm_multiscale_joint_v1_unique_resume_v2')
    p.add_argument('--workers',type=int,default=2)
    p.add_argument('--take-over',action='store_true',help='Stop old dispatch, finish current workers, then resume unique comparisons')
    p.add_argument('--recover',action='store_true',help='Resume the recorded old supervisor after a killed handoff process')
    p.add_argument('--check-handoff',action='store_true',help='Check pidfd on this process only, without touching experiments')
    a=p.parse_args()
    if a.workers < 1:p.error('--workers must be positive')
    if a.check_handoff:
        from pidfd_compat import self_check
        print('[MULTI-RESUME] '+json.dumps(self_check()),flush=True)
        return
    try:
        execute(a)
    except Exception as e:
        # An interrupted handoff has already resumed the old dispatcher in finally.
        # During unique execution the original runner preserves finished records.
        if getattr(a,'owns_output_lock',False) and a.out.exists():
            p=a.out/'STATUS.json'
            p.write_text(json.dumps({'completed':False,'paused':type(e).__name__=='Paused',
                'failed':type(e).__name__!='Paused','reason':str(e),
                'resume':'Repeat the same command; verify whether the original supervisor resumed.'},
                ensure_ascii=False,indent=2)+'\n')
        raise


if __name__=='__main__':main()
