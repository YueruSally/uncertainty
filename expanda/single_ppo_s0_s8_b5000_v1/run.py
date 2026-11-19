#!/usr/bin/env python3
"""Single-member paired-terminal PPO: all S0--S8 in every update; u0 eligible."""
import argparse
from collections import Counter
import copy
import fcntl
import importlib.util
import json
from pathlib import Path
import shutil
import signal
from statistics import mean, stdev
import sys
import time

sys.dont_write_bytecode = True
HERE=Path(__file__).resolve().parent
EXP=HERE.parent
DIAG=EXP/'transformer_ppo_runs/diagnostics'
spec=importlib.util.spec_from_file_location('all9_source',EXP/'ppo_joint_b5000_screen27_v1/run.py')
s=importlib.util.module_from_spec(spec);spec.loader.exec_module(s)
p=s.p
read,write,digest,canonical,freeze=s.read,s.write,s.digest,s.canonical,s.freeze
CONFIGS=[f'S{i}' for i in range(9)]
BUDGETS=(400,1000,5000)
SHARED='ICM_SHARED_SUPERVISED'
PPO='ICM_TERMINAL_PPO'


def checked(path):
    q=read(path)
    if canonical({k:v for k,v in q.items() if k!='sha256'})!=q['sha256']:raise ValueError('Plan changed: '+str(path))
    return q


def validate_protocol(q):
    if (q['configurations']!=CONFIGS or q['updates']!=3 or q['episodes_per_update']!=9 or
        q['training_member']!=991001 or q['validation_seeds']!=[996001,996002,996003] or
        q['screen_seeds']!=[993001,993002,993003] or q['population']!=250 or q['budget']!=5000):
        raise ValueError('Expected fixed single-member 3 updates x 9 configurations protocol')


def single(folder,norm=None):
    b=read(folder/'BUNDLE.json')
    if len(b['checkpoints'])!=1:raise ValueError('Exactly one policy member is required')
    if norm is not None and b['normalization']!=norm:raise ValueError('Model normalization changed')
    p.verify({str(folder/n):h for n,h in b['checkpoint_sha256'].items()})
    return b


def source_records(source,old,files):
    entries={e['instance_id']:e for e in old['original_entries']+old['entries']}
    saved=[]
    for rec in old['reused']:
        path=Path(rec['path'])
        if digest(path)!=rec['sha256']:raise ValueError('Reused development run changed')
        saved.append(path)
    for e in old['entries']:
        for seed in old['settings']['screen_seeds']:
            for arm in p.hv.NAMES:
                saved.append(source/'runs'/e['instance_id']/f'seed_{seed}'/arm/'RUN_COMPLETE.json')
    groups={};seen=set()
    for path in saved:
        row=read(path);key=(row['instance'],row['seed'],row['arm'])
        if key in seen:raise ValueError('Duplicate development record')
        seen.add(key);s.audit.validate_run(row,entries[row['instance']],row['seed'],old['settings'])
        if row['stochastic']:raise ValueError('Development reference is not greedy')
        if row['arm']==SHARED and row['model_bundle_sha256']!=old['models']['shared']['bundle_sha256']:
            raise ValueError('Development shared model differs')
        files[str(path)]=digest(path);groups.setdefault(row['arm'],[]).append(str(path))
    if len(saved)!=594 or len(groups)!=11 or any(len(v)!=54 for v in groups.values()):
        raise ValueError('Expected 594 source searches / 54 per method')
    return groups


def prepare(root,source,training_source):
    if (root/'PLAN.json').exists():
        plan=checked(root/'PLAN.json')
        if plan['source']!=str(source) or plan['training_source']!=str(training_source):raise ValueError('Resume source differs')
        p.verify(plan['files']);return plan
    old=checked(source/'PLAN.json');training=checked(training_source/'PLAN.json')
    print('[SINGLE-PPO9] checking frozen single-model sources and old records',flush=True)
    p.verify(old['files']);p.verify(training['files'])
    if not read(source/'STATUS.json')['completed'] or not read(training_source/'STATUS.json')['completed']:
        raise ValueError('Sources must be completed')
    q=read(HERE/'protocol.json');validate_protocol(q)
    settings=copy.deepcopy(old['settings']);norm=old['normalization']
    if (old['training_seed']!=991001 or norm!=training['normalization'] or
        tuple(settings[k] for k in ('population','budget','scenario_count','confidence','progress_denominator','m_start_generation'))
        !=(250,5000,100,.9,50000,10) or settings['milestones']!=list(BUDGETS)):
        raise ValueError('Expected fixed pop250/B5000/CCP100 deployment')
    settings['task_timeout_seconds']=q['task_timeout_seconds'];settings['invocation_seconds']=q['invocation_seconds']
    settings['ppo']=copy.deepcopy(q)
    files={**old['files'],**training['files']}
    for f in [source/'PLAN.json',source/'STATUS.json',training_source/'PLAN.json',training_source/'STATUS.json',*HERE.glob('*.py'),HERE/'protocol.json',HERE/'run.sh']:
        files[str(f)]=digest(f)
    origin=Path(old['models']['shared']['path']);b=single(origin,norm)
    if digest(origin/'BUNDLE.json')!=old['models']['shared']['bundle_sha256'] or b['checkpoints']!=['seed_991001_epoch_20.pt']:
        raise ValueError('Wrong frozen single shared initialization')
    original_shared=Path(training['source'])/'shared_model';original_bundle=read(original_shared/'BUNDLE.json')
    for key in ('normalization','scalers','dimensions','selected','feature_version'):
        if b[key]!=original_bundle[key]:raise ValueError('Calibration feature/gate source differs: '+key)
    if b['checkpoint_sha256'][b['checkpoints'][0]]!=original_bundle['checkpoint_sha256'][b['checkpoints'][0]]:
        raise ValueError('Shared member provenance differs')
    files[str(original_shared/'BUNDLE.json')]=digest(original_shared/'BUNDLE.json')
    dest=root/'shared_model';dest.mkdir(parents=True,exist_ok=True)
    for name in ['BUNDLE.json']+b['checkpoints']:
        target=dest/name
        if not target.exists():shutil.copyfile(origin/name,target)
        if digest(target)!=digest(origin/name):raise ValueError('Copied shared model changed')
        files[str(origin/name)]=digest(origin/name);files[str(target)]=digest(target)
    train=[];tune=[];confirm=[]
    for cfg in CONFIGS:
        tt=sorted((e for e in training['entries'] if e['configuration']==cfg and e['development_role']=='training'),key=lambda e:e['instance_id'])
        vv=sorted((e for e in training['entries'] if e['configuration']==cfg and e['development_role']=='validation'),key=lambda e:e['instance_id'])
        if len(tt)!=6 or len(vv)!=2:raise ValueError('Expected 6 training and 2 validation instances per configuration')
        if {e['instance_id'] for e in tt} - set(b['training_instances']['I']) or {e['instance_id'] for e in vv} - set(b['validation_instances']['I']):
            raise ValueError('Model and source data roles differ')
        train.extend(tt[:3]);tune.append(vv[0]);confirm.append(vv[1])
    dev=old['original_entries']+old['entries'];entries=train+tune+confirm+dev
    if len(dev)!=18 or len({e['instance_id'] for e in entries})!=len(entries) or len({e['batch_signature'] for e in entries})!=len(entries):
        raise ValueError('Overlapping or incomplete data partitions')
    for e in entries:
        freeze(root/'instances'/(e['instance_id']+'.json'),e)
        files[str(root/'instances'/(e['instance_id']+'.json'))]=digest(root/'instances'/(e['instance_id']+'.json'))
    calibration=[]
    for cfg in CONFIGS:
        candidates=[]
        for path in sorted((training_source/'terminal/rollouts').glob('u*/*/RUN_COMPLETE.json')):
            r=read(path)
            if r['configuration']!=cfg:continue
            e=next(v for v in training['entries'] if v['instance_id']==r['instance'])
            if not r['completed'] or r['phase']!='training' or not r['stochastic'] or r['plan_sha256']!=training['sha256'] or e['development_role']!='training':
                raise ValueError('Invalid train-only temperature calibration source')
            trace=path.parent/'TRAJECTORY.npz'
            if digest(trace)!=r['trajectory_sha256']:raise ValueError('Calibration trajectory changed')
            files[str(path)]=digest(path);files[str(trace)]=digest(trace);candidates.append(str(trace));break
        if not candidates:raise ValueError('Missing existing training candidate states: '+cfg)
        calibration.extend(candidates)
        print('[SINGLE-PPO9] existing train-only calibration states:',cfg,flush=True)
    controls=source_records(source,old,files)
    plan=dict(source=str(source),training_source=str(training_source),source_plan_sha256=old['sha256'],
        training_source_plan_sha256=training['sha256'],settings=settings,pilot=q,normalization=norm,files=files,
        shared_model=str(dest),training=train,tuning=tune,confirmation=confirm,development=dev,entries=entries,
        calibration=calibration,source_records=controls,training_member=991001,model_members=1,
        training_configurations=CONFIGS,expected_training_trajectories=27,expected_reference_searches=27,
        maximum_new_searches=272,final_test_opened=False,offline_gate_proposals_applied=False,
        selection_uses_development_results=False,selection_includes_u0=True)
    plan['sha256']=canonical(plan);freeze(root/'PLAN.json',plan)
    print('[SINGLE-PPO9] frozen 27 training / 9 tuning / 9 confirmation / 18 development instances; old 594 runs reused',flush=True)
    return plan


def model_path(root,number):return root/'checkpoints'/f'update_{number:03d}'


def task(plan,e,seed,model,phase,number,sample=False):
    return p.make_task(plan,e,seed,model,phase,sample,sample,number)


def validation_jobs(root,plan,entries,number,phase):
    return [(task(plan,e,seed,model_path(root,number),phase,number),
        root/'runs'/phase/f'u{number:03d}'/'greedy'/e['instance_id']/f'seed_{seed}')
        for seed in plan['pilot']['validation_seeds'] for e in entries]


def training_jobs(root,plan,u):
    work=[]
    for i,cfg in enumerate(plan['pilot']['configurations']):
        e=sorted((e for e in plan['training'] if e['configuration']==cfg),key=lambda v:v['instance_id'])[u-1]
        seed=plan['pilot']['training_seed_base']+(u-1)*len(plan['pilot']['configurations'])+i
        for phase,number,sampled in [('training_reference',0,False),('training',u-1,True)]:
            work.append((task(plan,e,seed,model_path(root,number),phase,u,sampled),
                root/'runs'/phase/f'u{u:03d}'/e['instance_id']/f'seed_{seed}'))
    return work


def validate_task(root,plan,t,folder):
    e=next((e for e in plan['entries'] if e['instance_id']==t['instance']),None)
    if e is None or e['development_role']!=t['role']:raise ValueError('Wrong instance role')
    phase=t['phase'];u=t['update'];sample=False
    if phase in ('training','training_reference'):
        if not 1<=u<=plan['pilot']['updates']:raise ValueError('Invalid update')
        jobs=training_jobs(root,plan,u)
        if (t,folder) not in jobs:raise ValueError('Unscheduled training task')
        number=u-1 if phase=='training' else 0;sample=phase=='training'
    elif phase in ('tuning','confirmation'):
        if not 0<=u<=plan['pilot']['updates']:raise ValueError('Invalid checkpoint')
        group=plan['tuning'] if phase=='tuning' else plan['confirmation']
        if e not in group or t['seed'] not in plan['pilot']['validation_seeds']:raise ValueError('Wrong validation subset/seed')
        if phase=='confirmation' and u not in (0,read(root/'SELECTION.json')['selected_update']):raise ValueError('Confirmation cannot reselect')
        if (t,folder) not in validation_jobs(root,plan,group,u,phase):raise ValueError('Validation task changed')
        number=u
    elif phase in ('development','reproduction'):
        if e not in plan['development'] or t['seed'] not in plan['pilot']['screen_seeds']:raise ValueError('Wrong development pair')
        number=0 if phase=='reproduction' else read(root/'SELECTION.json')['selected_update']
        expected=root/'runs'/phase/f'u{number:03d}'/'greedy'/e['instance_id']/f'seed_{t["seed"]}'
        if u!=number or folder!=expected:raise ValueError('Wrong development checkpoint/path')
        if phase=='reproduction' and (e!=plan['development'][0] or t['seed'] not in plan['pilot']['screen_seeds'][:2]):
            raise ValueError('Only two predeclared u0 replays')
    else:raise ValueError('Unknown phase')
    if t!=task(plan,e,t['seed'],model_path(root,number),phase,u,sample):raise ValueError('Task differs from frozen schedule')
    b=single(model_path(root,number),plan['normalization']);original=single(Path(plan['shared_model']),plan['normalization'])
    if b['selected']!=original['selected'] or b['scalers']!=original['scalers']:raise ValueError('Gates/scalers changed')
    return e


def run_worker(root,plan,t,out):
    # Reuse the exact search/reward-recording engine; only task scheduling changes.
    previous=p.validate_task;p.validate_task=validate_task
    try:p.run_worker(root,plan,t,out)
    finally:p.validate_task=previous


def collect(root,plan,jobs,workers,stage):
    for t,folder in jobs:validate_task(root,plan,t,folder)
    p.scheduler.HERE2=HERE
    rows=p.scheduler.execute_jobs(root,plan,jobs,workers,stage)
    # Scheduler completion order is not guaranteed.
    by={(r['instance'],r['seed'],r['phase'],r['model_bundle_sha256']):r for r in rows}
    if len(by)!=len(jobs):raise ValueError('Duplicate/incomplete scheduled panel')
    for t,folder in jobs:
        r=by[t['instance'],t['seed'],t['phase'],t['bundle_sha256']]
        e=validate_task(root,plan,t,folder);s.audit.validate_run(r,e,t['seed'],plan['settings'])
        if r['task_sha256']!=canonical(t) or r['plan_sha256']!=plan['sha256'] or r['stochastic']!=t['sample'] or bool(r['trajectory_sha256'])!=t['record']:
            raise ValueError('Saved run provenance differs')
    return rows


def select_checkpoint(scores):
    if {x['update'] for x in scores}!=set(range(4)):raise ValueError('Need u0 through u3')
    if any(not p.np.isfinite(x['mean_difference_vs_shared']) for x in scores):raise ValueError('Nonfinite validation score')
    return max(scores,key=lambda v:(v['mean_difference_vs_shared'],-v['update']))['update']


def report(groups,dest,plan,union=False,pool=None,preview=False):
    # A separate main table always uses Base; no Base means validation-only direct comparison.
    if union:
        allgroups=copy.deepcopy(pool or {});allgroups.update(groups)
        allgroups,bounds=s.audit.union_normalize(allgroups)
        groups={k:allgroups[k] for k in groups};write(dest/'INSTANCE_BOUNDS.json',bounds)
    names={'BASE':'Base',SHARED:'ICM 共享监督 Transformer（单模型）',PPO:'ICM PPO（S0–S8单模型微调）'}
    selection=read(dest.parent/'SELECTION.json') if (dest.parent/'SELECTION.json').exists() else None
    if selection and selection['selected_update']==0:names[PPO]='ICM PPO验证选u0（复用单模型共享监督）'
    reference='BASE' if 'BASE' in groups else SHARED
    comparisons=[(k,reference) for k in groups]
    if reference=='BASE':comparisons.append((PPO,SHARED))
    summary=[];direct=[];per=[];bycfg=[]
    for b in BUDGETS:
        def at(rr):return [dict(r,metric=next(x['hv'] for x in r['budget_fronts'] if x['evaluation']==b)) for r in rr]
        for arm,ref in comparisons:
            pairs=p.paired(at(groups[arm]),at(groups[ref]),field='metric',draws=200 if preview else 3000)['per_run']
            def summarize(pp,rr):
                lo,hi=(None,None) if preview else p.hv.interval(pp,'percent',3000)
                alo,ahi=(None,None) if preview else p.hv.interval(pp,'absolute',3000)
                row=dict(评价次数=b,方法=names[arm],参照=names[ref],配对数=len(pp),平均HV=p.equal_mean(pp,'own'),
                    平均配对提升百分比=p.equal_mean(pp,'percent') if all(x['percent'] is not None for x in pp) else None,
                    提升区间下限=lo,提升区间上限=hi,平均HV差=p.equal_mean(pp,'absolute'),HV差区间下限=alo,HV差区间上限=ahi,
                    胜='—' if arm==ref else sum(x['absolute']>1e-12 for x in pp),平='—' if arm==ref else sum(abs(x['absolute'])<=1e-12 for x in pp),
                    负='—' if arm==ref else sum(x['absolute']< -1e-12 for x in pp),搜索平均秒=None,搜索标准差秒=None,推理平均秒=None)
                if b==5000:
                    times=[x['optimisation_seconds'] for x in rr];row.update(搜索平均秒=mean(times),搜索标准差秒=stdev(times) if len(times)>1 else None,推理平均秒=mean(x['inference_seconds'] for x in rr))
                return row
            row=summarize(pairs,groups[arm])
            (summary if ref==reference else direct).append(row)
            for cfg in CONFIGS:
                pp=[x for x in pairs if x['configuration']==cfg]
                if pp:bycfg.append(dict(配置=cfg,**summarize(pp,[x for x in groups[arm] if x['configuration']==cfg])))
            per.extend(dict(method=arm,reference_method=ref,evaluation=b,**x) for x in pairs)
    p.csv_write(dest/'RESULTS.csv',summary)
    if direct:p.csv_write(dest/'PPO_vs_SHARED.csv',direct)
    p.csv_write(dest/'PER_CONFIGURATION.csv',bycfg);p.csv_write(dest/'PER_PAIR.csv',per)
    write(dest/'REPORT.json',dict(plan_sha256=plan['sha256'],preview_only=preview,normalization='instance union, r=1.1, no /1.331' if union else 'frozen training bounds',
        primary_reference=reference,selection_basis=False,qualification='Development evidence; new searches are not created by reporting. u0 deployment aliases shared if selected.',
        selected_update=selection['selected_update'] if selection else None))
    print('[SINGLE-PPO9] report '+str(dest/'RESULTS.csv'),flush=True)


def source_group(plan,arm):return [read(Path(x)) for x in plan['source_records'][arm]]


def first_checkpoint(root,plan):
    first=model_path(root,0)
    if not first.exists():
        temp,cal=p.calibrate(Path(plan['shared_model']),[Path(x) for x in plan['calibration']],plan['pilot'])
        actor=p.PilotPolicy(Path(plan['shared_model']))
        if len(actor.models)!=1:raise ValueError('PPO initialization must be single-member')
        actor.bundle['pilot']={**plan['pilot'],'temperatures':temp}
        actor.bundle['fine_tuning_provenance']=dict(plan_sha256=plan['sha256'],training_configs=CONFIGS,initial_member=991001,model_members=1)
        p.snapshot(actor,first,metrics=dict(update=0,calibration=cal))
        write(root/'CALIBRATION.json',cal)
    b=single(first,plan['normalization'])
    if b['fine_tuning_provenance']['plan_sha256']!=plan['sha256'] or b['pilot']['configurations']!=CONFIGS:raise ValueError('u0 belongs to another protocol')
    initial=p.PilotPolicy(Path(plan['shared_model']));zero=p.PilotPolicy(first)
    if any(not p.torch.equal(v,zero.models[0].state_dict()[k]) for k,v in initial.models[0].state_dict().items()):raise ValueError('u0 does not equal shared initialization')
    return first


def optimize(root,plan,u,rows,jobs):
    indexed={(r['instance'],r['phase']):r for r in rows};episodes=[];rewards=[]
    folders={(t['instance'],t['phase']):folder for t,folder in jobs}
    for cfg in CONFIGS:
        rr=[r for r in rows if r['phase']=='training' and r['configuration']==cfg]
        if len(rr)!=1:raise ValueError('Each update needs exactly one trajectory from every S0-S8 configuration')
        r=rr[0];b=indexed[r['instance'],'training_reference'];p.check_pair(r,b)
        delta=r['final_hypervolume']-b['final_hypervolume'];episodes.append((folders[r['instance'],'training']/'TRAJECTORY.npz',delta))
        rewards.append(dict(configuration=cfg,instance=r['instance'],seed=r['seed'],policy_hv=r['final_hypervolume'],reference_hv=b['final_hypervolume'],paired_return=delta))
    checkpoint=model_path(root,u)
    if not checkpoint.exists():
        previous=model_path(root,u-1);policy=p.PilotPolicy(previous);reference=p.PilotPolicy(model_path(root,0))
        optimizer=p.optimizer_for(policy,plan['pilot'])
        if u>1:
            if digest(previous/'optimizer.pt')!=policy.bundle['optimizer_sha256']:raise ValueError('Optimizer state changed')
            optimizer.load_state_dict(p.load_checkpoint(previous/'optimizer.pt'))
        write(root/'STATUS.json',dict(completed=False,stage=f'optimizer/u{u:03d}',training_configurations=CONFIGS,model_members=1))
        begun=time.monotonic();metrics=p.update(policy,reference,episodes,plan['pilot'],optimizer,plan['pilot']['optimizer_seed_base']+u)
        metrics.update(update=u,reward_pairs=rewards,training_configurations=CONFIGS,policy_trajectories=9,model_members=1,optimizer_seconds=time.monotonic()-begun)
        p.snapshot(policy,checkpoint,optimizer,metrics)
    single(checkpoint,plan['normalization']);metrics=read(checkpoint/'UPDATE.json')
    if metrics['reward_pairs']!=rewards or metrics['training_configurations']!=CONFIGS or metrics['model_members']!=1:raise ValueError('Cached update provenance changed')
    print('[SINGLE-PPO9] update',u,'configurations=',','.join(CONFIGS),'trajectories=9 steps=',metrics['optimizer_steps'],flush=True)
    return metrics


def cost_report(root,plan):
    rows=[read(f) for f in root.glob('runs/**/RUN_COMPLETE.json')]
    rows=[r for r in rows if r.get('completed') and r.get('plan_sha256')==plan['sha256']]
    phase={}
    for r in rows:phase.setdefault(r['phase'],[]).append(r['optimisation_seconds'])
    write(root/'COSTS.json',dict(physical_searches=len(rows),objective_evaluations=sum(r['real_objective_evaluation_count'] for r in rows),
        phases={k:dict(searches=len(v),mean_seconds=mean(v),sum_seconds=sum(v)) for k,v in phase.items()},
        optimizer_seconds=sum(read(f)['optimizer_seconds'] for f in root.glob('checkpoints/update_*/UPDATE.json') if 'optimizer_seconds' in read(f)),
        reused_development_searches=594,qualification='Sum of search time is not wall time; all reused records retain original timing.'))


def execute(a):
    root=a.out.resolve();source=a.source.resolve();training_source=a.training_source.resolve()
    if any(root==x or x in root.parents or root in x.parents for x in (source,training_source)):raise ValueError('Separate output required')
    root.mkdir(parents=True,exist_ok=True);p.torch.set_num_threads(1)
    with (root/'.pipeline.lock').open('a+') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise SystemExit('This PPO task is already running')
        def stop(signum,frame):raise p.scheduler.Paused(f'Received signal {signum}')
        signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
        try:
            plan=prepare(root,source,training_source)
            if a.prepare_only:print('[SINGLE-PPO9] prepared, no new searches',flush=True);return
            first_checkpoint(root,plan);baseline=source_group(plan,SHARED);by={(r['instance'],r['seed']):r for r in baseline}
            e=plan['development'][0];replay=[]
            for seed in plan['pilot']['screen_seeds'][:2]:
                replay.append((task(plan,e,seed,model_path(root,0),'reproduction',0),root/'runs/reproduction/u000/greedy'/e['instance_id']/f'seed_{seed}'))
            for r in collect(root,plan,replay,a.workers,'single9/u0_reproduction'):
                b=by[r['instance'],r['seed']];p.check_pair(r,b)
                for key in ('final_front_raw','final_hypervolume','evaluation_order_digest','real_objective_evaluation_count','selection'):
                    if r[key]!=b[key]:raise ValueError('u0 differs from frozen shared: '+key)
                if [(x['evaluation'],x['front_raw']) for x in r['budget_fronts']]!=[(x['evaluation'],x['front_raw']) for x in b['budget_fronts']]:raise ValueError('u0 milestones differ')
            base_jobs=validation_jobs(root,plan,plan['tuning'],0,'tuning');n=len(plan['tuning'])
            first_baseline=collect(root,plan,base_jobs[:n],a.workers,'single9/tuning/u000_first_seed')
            scores=[dict(update=0,mean_difference_vs_shared=0.)];training_metrics=[]
            for u in range(1,4):
                jobs=training_jobs(root,plan,u);rows=collect(root,plan,jobs,a.workers,f'single9/train_pairs/u{u:03d}')
                training_metrics.append(optimize(root,plan,u,rows,jobs));cost_report(root,plan)
                tuning=validation_jobs(root,plan,plan['tuning'],u,'tuning')
                early=collect(root,plan,tuning[:n],a.workers,f'single9/preview/u{u:03d}')
                report({SHARED:first_baseline,PPO:early},root/'preview'/f'u{u:03d}',plan,preview=True)
                controls=collect(root,plan,base_jobs,a.workers,'single9/tuning/u000_all')
                current=collect(root,plan,tuning,a.workers,f'single9/tuning/u{u:03d}_all')
                stats=p.paired(current,controls)
                scores.append(dict(update=u,mean_difference_vs_shared=stats['mean_absolute_difference'],statistics=stats))
                report({SHARED:controls,PPO:current},root/'validation'/f'u{u:03d}',plan)
                write(root/'TRAINING_PROGRESS.json',dict(completed_update=u,validation=scores,training=training_metrics,model_members=1,training_configurations=CONFIGS))
            u=select_checkpoint(scores)
            selection=dict(selected_update=u,best_including_u0=u,best_updated_checkpoint=max(scores[1:],key=lambda v:(v['mean_difference_vs_shared'],-v['update']))['update'],
                model=str(model_path(root,u)),bundle_sha256=digest(model_path(root,u)/'BUNDLE.json'),validation=scores,rule=plan['pilot']['selection'],
                single_member=True,selected_u0_is_shared=(u==0),development_used_for_selection=False)
            freeze(root/'SELECTION.json',selection)
            confirm0=collect(root,plan,validation_jobs(root,plan,plan['confirmation'],0,'confirmation'),a.workers,'single9/confirmation/u000')
            confirm=confirm0 if u==0 else collect(root,plan,validation_jobs(root,plan,plan['confirmation'],u,'confirmation'),a.workers,f'single9/confirmation/u{u:03d}')
            report({SHARED:confirm0,PPO:confirm},root/'confirmation',plan)
            if u==0:
                deployed=copy.deepcopy(baseline)
                write(root/'REUSE_U0.json',dict(reason='Validation selected unchanged shared actor; exact reproduction checked on two fixed development seeds.',
                    physical_ppo_development_searches=0,reused_shared_pairs=54,source_files=plan['source_records'][SHARED]))
            else:
                work=[(task(plan,e,seed,model_path(root,u),'development',u),root/'runs/development'/f'u{u:03d}'/'greedy'/e['instance_id']/f'seed_{seed}')
                    for e in plan['development'] for seed in plan['pilot']['screen_seeds']]
                deployed=collect(root,plan,work,a.workers,f'single9/development/u{u:03d}')
            groups={'BASE':source_group(plan,'BASE'),SHARED:baseline,PPO:deployed}
            pool={'historical_'+k:source_group(plan,k) for k in plan['source_records']}
            report(groups,root/'results_frozen',plan)
            report(groups,root/'results_union',plan,union=True,pool=pool)
            cost_report(root,plan)
            write(root/'STATUS.json',dict(completed=True,stage='all',selected_update=u,training_configurations=CONFIGS,
                policy_training_trajectories=27,paired_reference_searches=27,model_members=1,development_pairs=54,
                final_test_opened=False,automatic_further_experiments=False,original_results_modified=False))
            print('[SINGLE-PPO9] COMPLETE',json.dumps(read(root/'STATUS.json')),flush=True)
        except p.scheduler.Paused as ex:
            write(root/'STATUS.json',dict(completed=False,paused=True,reason=str(ex),resume='Repeat same command; completed searches/checkpoints reused'))
            print('[SINGLE-PPO9] PAUSED',str(ex),flush=True)
        except Exception as ex:
            write(root/'STATUS.json',dict(completed=False,failed=True,error=str(ex)));raise


def main():
    q=argparse.ArgumentParser(description=__doc__)
    q.add_argument('--source',type=Path,default=DIAG/'ccp_icm_single_member_second_instance_b5000_v1')
    q.add_argument('--training-source',type=Path,default=DIAG/'ccp_component_joint_b5000_v1')
    q.add_argument('--out',type=Path,default=DIAG/'ccp_single_ppo_s0_s8_b5000_v1')
    q.add_argument('--workers',type=int,choices=(1,2),default=2);q.add_argument('--prepare-only',action='store_true')
    q.add_argument('--worker',action='store_true');q.add_argument('--task',type=Path);q.add_argument('--run-out',type=Path)
    a=q.parse_args()
    if a.worker:run_worker(a.out,checked(a.out/'PLAN.json'),read(a.task),a.run_out)
    else:execute(a)


if __name__=='__main__':main()
