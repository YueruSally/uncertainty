#!/usr/bin/env python3
"""Frozen single-member evaluation on nine new S9--S11 instances; no fitting."""
import argparse
from collections import Counter, deque
import copy
from dataclasses import asdict
from datetime import datetime, timezone
import fcntl
import gzip
import importlib.util
import json
import os
from pathlib import Path
import shutil
import signal
from statistics import mean, stdev
import subprocess
import sys
import time

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
EXP = HERE.parent
DIAG = EXP/'transformer_ppo_runs/diagnostics'
spec = importlib.util.spec_from_file_location('large_frozen_engine', EXP/'final_generalization_b5000_v1/run.py')
engine = importlib.util.module_from_spec(spec)
spec.loader.exec_module(engine)
p, s = engine.p, engine.s
read, write, digest, canonical, freeze = engine.read, engine.write, engine.digest, engine.canonical, engine.freeze
ARMS = engine.ARMS
CONFIGS = ['S9', 'S10', 'S11']
BUDGETS = (400, 1000, 5000)
NAMES = dict(BASE='Base', C_RULE='Simple（仅交叉规则）',
             ICM_SHARED_SUPERVISED='ICM 共享监督 Transformer（单模型）',
             ICM_TERMINAL_PPO='ICM PPO u1（单模型）')


def checked(path):
    plan = read(path)
    if canonical({k:v for k,v in plan.items() if k != 'sha256'}) != plan['sha256']:
        raise ValueError('Plan checksum changed: '+str(path))
    return plan


def validate_protocol(q):
    if (q['configurations'] != CONFIGS or q['instances_per_configuration'] != 3 or
            tuple(q['methods']) != ARMS or q['algorithm_seeds'] != [264000001,264000002,264000003] or
            q['training_seed'] != 991001 or q['ppo_update'] != 1 or q['primary_endpoint'] != 5000 or
            q['primary_method'] != 'ICM_SHARED_SUPERVISED'):
        raise ValueError('Expected fixed 3 configurations x 3 new instances x 3 seeds x 4 methods')
    if [engine.SCENARIOS[c] for c in CONFIGS] != [(50,.5),(50,1.),(50,2.)]:
        raise ValueError('Large configuration definitions changed')


def check_single(folder, kind, norm):
    b = read(folder/'BUNDLE.json')
    if len(b['checkpoints']) != 1 or b['normalization'] != norm or b['settings']['progress_denominator'] != 50000:
        raise ValueError('Not the fixed single-member model: '+str(folder))
    name = b['checkpoints'][0]
    if Path(name).name != name or digest(folder/name) != b['checkpoint_sha256'][name]:
        raise ValueError('Model weight identity changed')
    if kind == 'shared' and name != f"seed_991001_epoch_{b['selected']['epoch']}.pt":
        raise ValueError('Shared member is not training seed 991001')
    if kind == 'ppo' and name != 'member_0.pt':
        raise ValueError('PPO member differs from the frozen single-member panel')
    return b


def generate(root, source_plan, protocol):
    st = source_plan['settings']
    catalog_path = Path(st['source_catalog_default'])
    catalog = read(catalog_path)
    if (catalog['catalog_digest'] != st['source_catalog_digest'] or
            canonical({k:v for k,v in catalog.items() if k != 'catalog_digest'}) != catalog['catalog_digest']):
        raise ValueError('Training OD catalog changed')
    workbook = EXP/'formal_icm_s0_v1/assets/data/data_expanded.xlsx'
    if digest(workbook) != catalog['source_data_sha256']:
        raise ValueError('Transport network changed')
    od = [tuple(v) for v in catalog['train_od_pool']]
    if not od or set(od) & {tuple(v) for v in catalog['test_od_pool']}:
        raise ValueError('Invalid OD pools')
    known, sources = engine.known_instances(root.parent, root)
    for e in source_plan['original_entries']+source_plan['entries']:
        known.add(engine.signature(e))
    initial_count = len(known)
    sources[str(catalog_path)] = digest(catalog_path)
    # Different batches and size; same network, OD pool and scalar sampling rules.
    sample = source_plan['original_entries'][0]
    release = sample['release_time_definition']
    base = p.base
    base.BORDER_EVENT_DEFINITIONS = base.load_border_event_definitions(base.DEFAULT_BORDER_EVENT_DATA_FILE)
    nodes, regions, _, _, _, arcs, timetables, raw, *_ = base.load_network_from_extended(str(workbook))
    tt, lookup = base.build_timetable_dict(timetables), base.build_arc_lookup(arcs)
    entries = []
    for ci,cfg in enumerate(CONFIGS):
        size, alpha = engine.SCENARIOS[cfg]
        for j in range(3):
            index = ci*3+j
            seed = protocol['instance_seed_base']+100*index
            batches = engine._batches(raw,od,size,alpha,seed,'train',release['maximum_h'],release['step_h'])
            values = [asdict(b) for b in batches]
            signature = canonical(values)
            if signature in known:
                raise ValueError('Generated batch set already exists; no outcome-based redraw')
            known.add(signature)
            e = {key:copy.deepcopy(sample[key]) for key in
                 ('schema_version','alpha_definition','release_time_definition','quantity_definition')}
            e['release_time_definition']['split_disjoint_slot_parity'] = False
            e['quantity_definition']['split_disjoint_centi_teu_parity'] = False
            e.update(instance_id=f'single-large-{cfg}-{j+1:03d}',configuration=cfg,K=size,
                deadline_window_alpha=alpha,split='test',development_role='final_test',instance_seed=seed,
                path_seed=engine._find_path_seed(seed+1000000,nodes,regions,arcs,batches,tt,lookup),
                algorithm_seed=protocol['algorithm_seeds'][0],ccp100_seed=protocol['scenario_seed_base']+index,
                oos5000_seed=None,od_pool='train',source_data_sha256=digest(workbook),
                source_catalog_digest=catalog['catalog_digest'],batch_signature=signature,batches=values,
                sampling_definition='New complete 50-batch sets; unchanged train OD/quantity/release distributions; no resampling by performance.')
            e['instance_digest'] = canonical(e)
            dest = root/'instances'/(e['instance_id']+'.json')
            freeze(dest,e)
            p.read_instance(dest)
            entries.append(e)
            print(f'[LARGE-SINGLE] prepared {len(entries)}/9 {e["instance_id"]}; no objective evaluations',flush=True)
    freeze(root/'EXCLUSION_AUDIT.json',dict(source_files=sources,historical_unique_batch_sets=initial_count,
        new_batch_sets=9,no_identical_batch_sets=True,outcome_based_resampling=False,
        qualification='S11 was inspected during prior development; these batch sets are new, not an assertion that S11 was never seen.'))
    freeze(root/'instances/catalog.json',dict(instances=entries,protocol=protocol))
    return entries


def prepare(root, source):
    if (root/'PLAN.json').exists():
        plan = checked(root/'PLAN.json')
        if plan['source'] != str(source):
            raise ValueError('Resume source differs')
        p.verify(plan['files'])
        return plan
    old = checked(source/'PLAN.json')
    p.verify(old['files'])
    if not read(source/'STATUS.json').get('completed') or old['training_seed'] != 991001 or old['ppo_update'] != 1:
        raise ValueError('Expected completed 54-pair single-member panel')
    if len(old['entries']) != 9 or len(old['original_entries']) != 9:
        raise ValueError('Missing completed second-instance panel')
    protocol = read(HERE/'protocol.json');validate_protocol(protocol)
    st = copy.deepcopy(old['settings'])
    if ((st['population'],st['budget'],st['scenario_count'],st['pool_size'],st['progress_denominator'],st['m_start_generation'])
            != (250,5000,100,8,50000,10) or st['confidence'] != .9 or st['milestones'] != list(BUDGETS)):
        raise ValueError('Fixed deployment settings differ')
    st['invocation_seconds'] = protocol['invocation_seconds']
    st['task_timeout_seconds'] = protocol['task_timeout_seconds']
    files = dict(old['files']); models = {}; metadata = {}
    for kind,arm in [('shared','ICM_SHARED_SUPERVISED'),('ppo','ICM_TERMINAL_PPO')]:
        src = Path(old['models'][kind]['path'])
        if digest(src/'BUNDLE.json') != old['models'][kind]['bundle_sha256']:
            raise ValueError('Source bundle changed')
        b = check_single(src,kind,old['normalization'])
        dst = root/'models'/kind;dst.mkdir(parents=True,exist_ok=True)
        for name in ['BUNDLE.json']+b['checkpoints']:
            expected = digest(src/name);target=dst/name
            if not target.exists():
                tmp=target.with_suffix('.copying');shutil.copyfile(src/name,tmp);tmp.replace(target)
            if digest(target) != expected:
                raise ValueError('Frozen copied model differs: '+str(target))
            files[str(src/name)] = expected;files[str(target)] = expected
        models[arm]=str(dst)
        metadata[arm]=dict(source=str(src),bundle_sha256=digest(dst/'BUNDLE.json'),members=1,
                           checkpoint=b['checkpoints'][0],gate=b['selected']['gates'],
                           historical_training_pilot=b.get('pilot'),deployment='deterministic greedy with frozen gates')
    # Verify runtime identity with the source experiment; new orchestration only.
    folders = ('component_joint_b5000_v1','component_screen_b20_v1','joint_icm_multiscale_v1',
        'joint_icm_multiscale_terminal_v1','ppo_joint_b5000_pilot_v2','ppo_joint_b5000_screen27_v1',
        'ppo_joint_b5000_base_report_v1','hv_joint_b5000_union_v1','paper_evidence_b5000_v1',
        'final_generalization_b5000_v1','single_s9_s11_b5000_v1')
    paths = [f for folder in folders for f in (EXP/folder).glob('*') if f.suffix in ('.py','.json','.sh')]
    paths += [f for f in (EXP/'formal_icm_s0_v1/src').rglob('*') if f.suffix in ('.py','.csv')]
    paths += [EXP/'formal_icm_s0_v1/assets/data/data_expanded.xlsx',source/'PLAN.json',source/'STATUS.json']
    for f in paths:
        if f.name=='checks.py':continue
        got=digest(f)
        if str(f) in old['files'] and old['files'][str(f)] != got:
            raise ValueError('Search runtime differs from the 54-pair experiment: '+str(f))
        files[str(f.resolve())]=got
    entries = generate(root,old,protocol)
    for f in [root/'EXCLUSION_AUDIT.json',root/'instances/catalog.json',*sorted((root/'instances').glob('single-large-*.json'))]:
        files[str(f)]=digest(f)
    files[str(Path(st['source_catalog_default']))]=digest(Path(st['source_catalog_default']))
    plan=dict(source=str(source),source_plan_sha256=old['sha256'],settings=st,protocol=protocol,
        models=models,model_metadata=metadata,selected_update=1,training_seed=991001,
        entries=entries,normalization=old['normalization'],files=files,maximum_searches=108,
        maximum_objective_evaluations=540000,primary_method='ICM_SHARED_SUPERVISED',
        normalization_reporting='Per-instance union across four methods, three seeds and all three budgets; reference (1.1,1.1,1.1), no /1.331; frozen training bounds reported separately.',
        purpose='Fixed single-member models from the existing 54-pair panel; S9-S11 new 50-batch sets only.',
        final_test=True,new_training_updates=0,checkpoint_selection_on_test=False,
        evaluation_configurations=CONFIGS,
        offline_review_recommendations_applied=False,
        uncertainty_evaluation='100 common optimization scenarios per instance; no independent 5000-scenario reevaluation')
    plan['sha256']=canonical(plan);freeze(root/'PLAN.json',plan)
    return plan


def jobs_for(root,plan):
    # Complete all four arms and three seeds for one instance before moving on.
    return [(engine.task(plan,e,seed,arm),root/'runs'/e['instance_id']/f'seed_{seed}'/arm)
            for e in plan['entries'] for seed in plan['protocol']['algorithm_seeds'] for arm in ARMS]


def statistics(own, reference, budget, arm, ref, preview=False):
    lookup={(r['instance'],r['seed']):r for r in reference};pairs=[]
    for r in own:
        b=lookup[r['instance'],r['seed']];p.check_pair(r,b)
        x=next(s['hv'] for s in r['budget_fronts'] if s['evaluation']==budget)
        y=next(s['hv'] for s in b['budget_fronts'] if s['evaluation']==budget)
        pairs.append(dict(configuration=r['configuration'],instance=r['instance'],seed=r['seed'],own=x,reference=y,
                          absolute=x-y,percent=100*(x-y)/y if y>0 else None))
    lo,hi=(None,None) if preview else p.hv.interval(pairs,'percent',3000)
    alo,ahi=(None,None) if preview else p.hv.interval(pairs,'absolute',3000)
    row=dict(评价次数=budget,方法=NAMES[arm],参照=NAMES[ref],配对数=len(pairs),平均HV=p.equal_mean(pairs,'own'),
        平均配对提升百分比=p.equal_mean(pairs,'percent') if all(r['percent'] is not None for r in pairs) else None,
        提升区间下限=lo,提升区间上限=hi,平均HV差=p.equal_mean(pairs,'absolute'),HV差区间下限=alo,HV差区间上限=ahi,
        胜='—' if arm==ref else sum(r['absolute']>1e-12 for r in pairs),
        平='—' if arm==ref else sum(abs(r['absolute'])<=1e-12 for r in pairs),
        负='—' if arm==ref else sum(r['absolute']< -1e-12 for r in pairs),
        搜索平均秒=None,搜索标准差秒=None,搜索总秒=None,推理平均秒=None,推理标准差秒=None)
    if budget==5000:
        times=[r['optimisation_seconds'] for r in own];inference=[r['inference_seconds'] for r in own]
        row.update(搜索平均秒=mean(times),搜索标准差秒=stdev(times) if len(times)>1 else None,搜索总秒=sum(times),
                   推理平均秒=mean(inference),推理标准差秒=stdev(inference) if len(inference)>1 else None)
    return row,[dict(method=arm,reference_method=ref,evaluation=budget,**v) for v in pairs]


def report(rows,root,plan,preview=False):
    if not preview:engine.validate_panel(rows,plan)
    groups={arm:[r for r in rows if r['arm']==arm] for arm in ARMS}
    expected={(e['instance'],e['seed']) for e in groups['BASE']}
    if not expected or any({(r['instance'],r['seed']) for r in rr}!=expected or len(rr)!=len(expected) for rr in groups.values()):
        raise ValueError('Incomplete preview pairing')
    for name in {r['instance'] for r in rows}:
        if {r['seed'] for r in rows if r['instance']==name} != set(plan['protocol']['algorithm_seeds']):
            raise ValueError('A preview needs all three seeds of each included instance')
        if len({r['scenario_digest'] for r in rows if r['instance']==name}) != 1:
            raise ValueError('Scenario pairing differs')
    norm,bounds=s.audit.union_normalize(groups)
    baseout=root/'preview' if preview else root
    for mode,panel in [('results_union',norm),('results_frozen',groups)]:
        dest=baseout/mode;dest.mkdir(parents=True,exist_ok=True)
        summary=[];direct=[];detail=[];configs=[]
        for budget in BUDGETS:
            for arm,ref in [(a,'BASE') for a in ARMS]+[('ICM_TERMINAL_PPO','ICM_SHARED_SUPERVISED')]:
                row,pp=statistics(panel[arm],panel[ref],budget,arm,ref,preview)
                (summary if ref=='BASE' else direct).append(row);detail.extend(pp)
                if ref=='BASE':
                    for cfg in CONFIGS:
                        own=[r for r in panel[arm] if r['configuration']==cfg]
                        reference=[r for r in panel[ref] if r['configuration']==cfg]
                        if own:
                            bycfg,_=statistics(own,reference,budget,arm,ref,preview)
                            configs.append(dict(配置=cfg,**bycfg))
        for name,rr in [('RESULTS.csv',summary),('PPO_vs_SHARED.csv',direct),('PER_RUN.csv',detail),('PER_CONFIGURATION.csv',configs)]:
            p.csv_write(dest/name,rr)
    write(baseout/'results_union/INSTANCE_BOUNDS.json',bounds)
    write(baseout/'REPORT.json',dict(completed=not preview,preview_only=preview,searches=len(rows),
        pairs_per_method=len(groups['BASE']),configurations=sorted({r['configuration'] for r in rows}),
        reference='BASE',new_training_updates=0,single_member=True,plan_sha256=plan['sha256'],
        qualification='Preview is descriptive; no early stopping or outcome-dependent selection.' if preview else 'Fixed S9-S11 new-instance evaluation; no refitting or selection.',
        normalization=plan['normalization_reporting'],
        statistics='Configuration-equal mean; paired hierarchical bootstrap within each fixed configuration (instances, then seeds), 3000 draws, 95% percentile interval; W/T/L absolute HV tolerance 1e-12.'))
    if not preview:
        dest=root/'fronts';dest.mkdir(exist_ok=True)
        with gzip.open(dest/'FINAL_SOLUTIONS.jsonl.gz','wt',encoding='utf-8') as f:
            for r in rows:
                path=root/'runs'/r['instance']/f'seed_{r["seed"]}'/r['arm']/'final_feasible_nondominated.json'
                if digest(path)!=r['final_solutions_sha256']:raise ValueError('Final solutions changed')
                f.write(json.dumps(dict(instance=r['instance'],configuration=r['configuration'],seed=r['seed'],method=r['arm'],
                    scenario_digest=r['scenario_digest'],raw_objectives=r['final_front_raw'],solutions=read(path)),ensure_ascii=False)+'\n')
        p.csv_write(root/'results_union/DEPLOYMENT_COSTS.csv',[dict(instance=r['instance'],configuration=r['configuration'],seed=r['seed'],
            method=r['arm'],search_seconds=r['optimisation_seconds'],inference_seconds=r['inference_seconds'],
            feature_seconds=r.get('feature_construction_seconds'),objective_seconds=r.get('objective_evaluation_seconds'),
            peak_memory_mib=r['peak_memory_mib'],hostname=r['hostname'],rule_selection=json.dumps(r.get('selection',{})),
            input_range_checks=json.dumps(r.get('input_range_checks',{}))) for r in rows])
    print('[LARGE-SINGLE] report ready:',baseout/'results_union/RESULTS.csv',flush=True)


def estimate(rows,pending,running,workers):
    by=defaultdict_list(rows)
    costs={arm:mean(r['optimisation_seconds'] for r in rows if r['arm']==arm)
           for arm in ARMS if any(r['arm']==arm for r in rows)}
    if len(costs)!=4:return dict(remaining_hours=None,qualification='Waiting for completed timings of all four methods')
    remaining=sum(by.get((t['instance'].split('-')[2],t['arm']),costs[t['arm']]) for t,_ in pending)
    remaining+=sum(max(0.,by.get((t['instance'].split('-')[2],t['arm']),costs[t['arm']])-(time.monotonic()-start))
                   for proc,stream,start,t,folder,log in running.values())
    return dict(remaining_hours=remaining/workers/3600,method_mean_seconds=costs,
        qualification='Extrapolation from completed runs; unseen configurations use method averages; excludes reporting/startup/contention, not a guaranteed finish time.')


def defaultdict_list(rows):
    groups={}
    for r in rows:groups.setdefault((r['configuration'],r['arm']),[]).append(r['optimisation_seconds'])
    return {k:mean(v) for k,v in groups.items()}


def execute_jobs(root,plan,workers):
    jobs=jobs_for(root,plan);pending=deque();rows=[];running={};start=time.monotonic();heartbeat=0;reported=set()
    for t,folder in jobs:
        r=p.scheduler.valid_completion(folder,t,plan)
        if r is None:pending.append((t,folder))
        else:engine.validate_run(r,plan);rows.append(r)
    try:
        while pending or running:
            if time.monotonic()-start>plan['settings']['invocation_seconds']:raise p.scheduler.Paused('Invocation time cap')
            while pending and len(running)<workers:
                t,folder=pending.popleft()
                if folder.exists():
                    archive=root/'interrupted';archive.mkdir(exist_ok=True)
                    folder.rename(archive/(t['instance']+'_'+folder.name+'_'+str(time.time_ns())))
                folder.mkdir(parents=True)
                taskfile=root/'tasks'/(canonical(t)+'.json');freeze(taskfile,t)
                logfile=root/'logs'/(canonical(t)+f'_{time.time_ns()}.log');logfile.parent.mkdir(exist_ok=True)
                stream=logfile.open('w')
                cmd=[sys.executable,'-B','-u',str(HERE/'run.py'),'--worker','--out',str(root),'--task',str(taskfile),'--run-out',str(folder)]
                proc=subprocess.Popen(cmd,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
                running[str(folder)]=(proc,stream,time.monotonic(),t,folder,logfile)
                print(f'[LARGE-SINGLE] start {t["instance"]}/{t["seed"]}/{t["arm"]} PID={proc.pid}',flush=True)
            for key,(proc,stream,began,t,folder,log) in list(running.items()):
                if time.monotonic()-began>plan['settings']['task_timeout_seconds']:raise p.scheduler.Paused('Worker timeout: '+str(log))
                if proc.poll() is None:continue
                stream.close();del running[key]
                if proc.returncode:raise RuntimeError(f'Worker exit {proc.returncode}; see {log}')
                r=p.scheduler.valid_completion(folder,t,plan)
                if r is None:raise ValueError('Missing completion record: '+str(folder))
                engine.validate_run(r,plan);rows.append(r)
                print(f'[LARGE-SINGLE] complete {len(rows)}/108 {t["instance"]}/{t["arm"]} seconds={r["optimisation_seconds"]:.1f}',flush=True)
            counts=Counter(r['instance'] for r in rows)
            full={name for name,n in counts.items() if n==12}
            if full and full!=reported:
                report([r for r in rows if r['instance'] in full],root,plan,True);reported=full
            if time.monotonic()-heartbeat>=30:
                eta=estimate(rows,pending,running,workers)
                state=dict(completed=False,stage='S9-S11_fixed_single',done=len(rows),total=108,
                    elapsed_seconds=time.monotonic()-start,remaining_estimate=eta,final_test_opened=True,
                    running=[dict(pid=proc.pid,instance=t['instance'],method=t['arm'],seed=t['seed'],
                                  seconds=int(time.monotonic()-began),log=str(log)) for proc,stream,began,t,folder,log in running.values()])
                write(root/'STATUS.json',state);write(root/'COST_ESTIMATE.json',eta)
                print(f'[LARGE-SINGLE] heartbeat {len(rows)}/108 running={len(running)} remaining_hours={eta.get("remaining_hours")}',flush=True)
                heartbeat=time.monotonic()
            if running:time.sleep(1)
        return rows
    finally:
        for proc,*_ in running.values():
            if proc.poll() is None:os.killpg(proc.pid,signal.SIGTERM)
        for proc,stream,*_ in running.values():
            try:proc.wait(timeout=10)
            except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
            stream.close()


def execute(a):
    root=a.out.resolve();source=a.source.resolve()
    if root==source or root in source.parents or source in root.parents:raise ValueError('Separate output required')
    root.mkdir(parents=True,exist_ok=True);p.torch.set_num_threads(1)
    with (root/'.pipeline.lock').open('a+') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise SystemExit('S9-S11 task is already running')
        def stop(signum,frame):raise p.scheduler.Paused(f'Received signal {signum}')
        signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
        try:
            plan=prepare(root,source)
            if a.prepare_only:
                print('[LARGE-SINGLE] 9 manifests / 108 tasks prepared; no searches',flush=True);return
            if not (root/'TEST_OPENED.json').exists():
                write(root/'TEST_OPENED.json',dict(opened=True,plan_sha256=plan['sha256'],opened_utc=datetime.now(timezone.utc).isoformat()))
            write(root/'ENVIRONMENT_LAUNCH.json',dict(environment=engine.ev.hardware(),requested_workers=a.workers,per_worker_threads=1))
            print('[LARGE-SINGLE] S9/S10/S11; 108 searches; models have exactly one member; no training/reselection',flush=True)
            started=time.monotonic();rows=execute_jobs(root,plan,a.workers);report(rows,root,plan)
            write(root/'STATUS.json',dict(completed=True,stage='all',done=108,total=108,new_training_updates=0,
                final_test_opened=True,automatic_further_experiments=False,wall_seconds_this_invocation=time.monotonic()-started))
            print('[LARGE-SINGLE] COMPLETE; results_union/RESULTS.csv',flush=True)
        except p.scheduler.Paused as e:
            write(root/'STATUS.json',dict(completed=False,paused=True,reason=str(e),resume='Repeat same command; completed searches reused'))
            print('[LARGE-SINGLE] PAUSED',str(e),flush=True)
        except Exception as e:
            write(root/'STATUS.json',dict(completed=False,failed=True,error=str(e)));raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=DIAG/'ccp_icm_single_member_second_instance_b5000_v1')
    parser.add_argument('--out',type=Path,default=DIAG/'ccp_single_s9_s11_b5000_v1')
    parser.add_argument('--workers',type=int,choices=(1,2),default=2)
    parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--worker',action='store_true');parser.add_argument('--task',type=Path);parser.add_argument('--run-out',type=Path)
    a=parser.parse_args()
    if a.worker:
        plan=s.checked_plan(a.out)
        for arm,kind in [('ICM_SHARED_SUPERVISED','shared'),('ICM_TERMINAL_PPO','ppo')]:check_single(Path(plan['models'][arm]),kind,plan['normalization'])
        engine.worker(a.out,plan,read(a.task),a.run_out)
    else:execute(a)


if __name__=='__main__':main()
