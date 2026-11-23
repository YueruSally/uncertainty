#!/usr/bin/env python3
"""54 corrected random searches; reuse all controls, then union-HV/seed-bootstrap reports."""
import argparse
from collections import Counter
import copy
import csv
import fcntl
import importlib.util
import os
from pathlib import Path
import signal
import sys
import tarfile
import time

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
EXP = HERE.parent
DIAG = EXP/'transformer_ppo_runs/diagnostics'
sys.path.insert(0,str(HERE))
from policy_v2 import RateControlledRandom, TARGETS, policy_seed
from statistics_v2 import Panel, wide, DRAWS, BOOTSTRAP_SEED, VOLUME, BUDGETS, TOLERANCE
sys.path.insert(0,str(EXP/'shared_sl_ablations_b5000_v1'))
spec = importlib.util.spec_from_file_location('frozen_ablation_v1',EXP/'shared_sl_ablations_b5000_v1/run.py')
old = importlib.util.module_from_spec(spec); spec.loader.exec_module(old)
s,p = old.s,old.p
read,write,digest,canonical,freeze = old.read,old.write,old.digest,old.canonical,old.freeze
SHARED,RULE,ZERO,PPO = old.SHARED,old.RULE,old.ZERO,'ICM_TERMINAL_PPO'
RANDOM = 'ICM_RANDOM_RATE_CONTROLLED'
NAMES = dict(BASE='Base',I_RULE='I 启发式',I_LEARN='I Transformer',
             C_RULE='C 启发式',C_LEARN='C Transformer',M_RULE='M 启发式',M_LEARN='M Transformer',
             ICM_HEURISTIC='规则 ICM',ICM_INDEPENDENT='独立监督 ICM',
             ICM_SHARED_SUPERVISED='共享监督 ICM（单模型）',ICM_TERMINAL_PPO='共享监督 + PPO（S0–S8）',
             ICM_RANDOM_RATE_CONTROLLED='随机替换 ICM（修正后）',ICM_SHARED_ZERO_THRESHOLD='阈值为零 ICM')


def retain(path, files, expected=None):
    path = Path(path).resolve(); value = digest(path)
    if expected is not None and value != expected: raise ValueError('Frozen source changed: '+str(path))
    files[str(path)] = value
    return read(path) if path.suffix=='.json' else None


def snapshot_backup(root, files):
    """Archive exact original result files and dependency manifest before any search."""
    backup = root/'backup'; backup.mkdir(exist_ok=True)
    archive = backup/'OLD_RESULTS.tar.gz'
    if not archive.exists():
        temp = archive.with_suffix('.tmp')
        with tarfile.open(temp,'w:gz') as tar:
            for name in sorted(files):
                path = Path(name)
                # Captures reports, fronts in RUN_COMPLETE, selected weights and source code;
                # no new training data dump or search re-execution.
                if path.suffix.lower() in ('.json','.csv','.txt','.md','.py','.sh','.pt'):
                    tar.add(path,arcname='original/'+name.lstrip('/'),recursive=False)
        temp.replace(archive)
    manifest = dict(archive=str(archive),archive_sha256=digest(archive),files=files)
    freeze(backup/'MANIFEST.json',manifest)
    return manifest['archive_sha256']


def old_table_rows(ppo_root, ablation_root, source_plan, files):
    """Preserve old published values rather than rebuilding them under new bounds."""
    paths = dict(panel=Path(source_plan['source'])/'combined54/RESULTS.csv',
                 ppo=ppo_root/'results_union/RESULTS.csv',ablation=ablation_root/'results/RESULTS.csv')
    tables = {}
    for key,path in paths.items():
        retain(path,files)
        with path.open(encoding='utf-8-sig',newline='') as f: tables[key]=list(csv.DictReader(f))
    aliases = dict(BASE=['Base'],I_RULE=['I 启发式'],I_LEARN=['I Transformer'],
                   C_RULE=['C 启发式'],C_LEARN=['C Transformer'],M_RULE=['M 启发式'],M_LEARN=['M Transformer'],
                   ICM_HEURISTIC=['ICM 启发式','规则 ICM'],ICM_INDEPENDENT=['ICM 独立 Transformer'],
                   ICM_SHARED_SUPERVISED=['ICM 共享监督 Transformer（单模型）'],
                   ICM_TERMINAL_PPO=['ICM PPO（S0–S8单模型微调）','ICM PPO验证选u0（复用单模型共享监督）'],
                   ICM_RANDOM_RATE_CONTROLLED=[old.NAMES[old.RANDOM]],ICM_SHARED_ZERO_THRESHOLD=[old.NAMES[ZERO]])
    result=[]
    def field(r,keys):
        for key in keys:
            if key in r and r[key] != '': return float(r[key])
        raise ValueError('Missing old report field: '+str(keys))
    for method in NAMES:
        table='ppo' if method in (SHARED,PPO) else 'ablation' if method in (RANDOM,ZERO) else 'panel'
        for budget in BUDGETS:
            hits=[r for r in tables[table] if r['方法'] in aliases[method] and r['参照']=='Base' and int(r['评价次数'])==budget]
            if len(hits)!=1: raise ValueError(f'Cannot identify unique old value: {method}/{budget} in {paths[table]}')
            r=hits[0]
            if int(field(r,['配对数','运行对数']))!=54: raise ValueError('Old report does not contain 54 pairs')
            value=field(r,['平均HV']); divisor=1. if table=='ablation' else VOLUME
            result.append(dict(method=method,evaluation=budget,source=str(paths[table]),source_row=r,
                old_percent=field(r,['平均配对提升百分比','相对Base平均配对提升(%)']),
                old_low=field(r,['提升95区间下限','提升区间下限','提升95%下限(%)']),
                old_high=field(r,['提升95区间上限','提升区间上限','提升95%上限(%)']),
                old_hv_as_reported=value,old_hv_01=value/divisor,
                old_hv_conversion_divisor=divisor))
    return result


def prepare(root, ablation_root, ppo_root):
    if (root/'PLAN.json').exists():
        plan=old.checked(root/'PLAN.json')
        if plan['ablation_source']!=str(ablation_root) or plan['ppo_source']!=str(ppo_root):
            raise ValueError('Resume source differs')
        p.verify(plan['files'])
        if digest(root/'backup/OLD_RESULTS.tar.gz')!=plan['backup_sha256']: raise ValueError('Backup changed')
        return plan
    oldplan=old.checked(ablation_root/'PLAN.json'); source=old.checked(ppo_root/'PLAN.json')
    if oldplan['source']!=str(ppo_root) or source['model_members']!=1:
        raise ValueError('Expected existing single-model S0–S8 ablation sources')
    if not read(ablation_root/'STATUS.json').get('completed') or not read(ppo_root/'STATUS.json').get('completed'):
        raise ValueError('Old ablations and S0–S8 PPO must be completed; no missing controls will be rerun')
    p.verify(oldplan['files'])
    files=dict(oldplan['files'])
    for directory in (ablation_root,ppo_root,Path(source['source'])):
        for name in ('PLAN.json','STATUS.json','SELECTION.json','REUSE_U0.json'):
            path=directory/name
            if path.exists(): retain(path,files)
        for sub in ('results','results_union','combined54'):
            if (directory/sub).exists():
                for path in sorted((directory/sub).glob('*')):
                    if path.is_file() and path.suffix.lower() in ('.csv','.json','.txt','.md'): retain(path,files)
    for path in list(HERE.glob('*.py'))+[HERE/'run.sh',HERE/'README_CN.md']: retain(path,files)
    settings=copy.deepcopy(oldplan['settings']); entries=oldplan['entries']
    if (entries!=source['development'] or len(entries)!=18 or
        Counter(e['configuration'] for e in entries)!={f'S{i}':2 for i in range(9)} or
        tuple(settings[k] for k in ('population','budget','scenario_count','pool_size','progress_denominator','m_start_generation'))!=(250,5000,100,8,50000,10)
        or settings['confidence']!=.9 or settings['screen_seeds']!=old.SEEDS or settings['milestones']!=list(BUDGETS)):
        raise ValueError('Panel or search protocol differs from Section 6.4')
    for entry in entries:
        src=ppo_root/'instances'/(entry['instance_id']+'.json')
        if read(src)!=entry: raise ValueError('Instance changed')
        old.copy_frozen(src,root/'instances'/src.name,files)
    for name in ['BUNDLE.json','seed_991001_epoch_20.pt']:
        old.copy_frozen(Path(oldplan['shared_model'])/name,root/'shared_model'/name,files)
    bundle=read(root/'shared_model/BUNDLE.json')
    if bundle['checkpoints']!=['seed_991001_epoch_20.pt']:
        raise ValueError('Expected fixed single shared member 991001 at epoch 20')
    if any(bundle['selected']['gates'][c]['disabled'] or bundle['selected']['gates'][c]['threshold_scaled']!=t
           for c,t in zip('ICM',(.05,0.,0.))): raise ValueError('Shared gates changed')
    records=copy.deepcopy(source['source_records'])
    if set(records)!=set(NAMES)-{RANDOM,ZERO}: raise ValueError('Incomplete historical 11-method source')
    if records!=oldplan['source_records']: raise ValueError('Old controls do not match the PPO source plan')
    # Backup the historical PPO but exclude its fronts from the new normalization pool.
    for names in records.values():
        for name in names: retain(name,files,source['files'][name])
    selected=retain(ppo_root/'SELECTION.json',files); u=selected['selected_update']
    if u!=read(ppo_root/'STATUS.json')['selected_update'] or not selected['single_member']:
        raise ValueError('PPO checkpoint selection differs')
    if u==0:
        if not selected['selected_u0_is_shared'] or not (ppo_root/'REUSE_U0.json').exists():
            raise ValueError('Missing explicit u0 reuse provenance')
        records[PPO]=list(records[SHARED])
    else:
        model=ppo_root/'checkpoints'/f'update_{u:03d}'
        b=retain(model/'BUNDLE.json',files,selected['bundle_sha256'])
        if len(b['checkpoints'])!=1: raise ValueError('PPO is not single-member')
        for name,h in b['checkpoint_sha256'].items(): retain(model/name,files,h)
        records[PPO]=[]
        for entry in entries:
            for seed in old.SEEDS:
                path=ppo_root/'runs/development'/f'u{u:03d}'/'greedy'/entry['instance_id']/f'seed_{seed}'/'RUN_COMPLETE.json'
                r=retain(path,files)
                if (r['plan_sha256']!=source['sha256'] or r['model_bundle_sha256']!=selected['bundle_sha256']
                    or r['phase']!='development' or r['stochastic'] or r['arm']!=PPO):
                    raise ValueError('Incorrect current S0–S8 PPO run: '+str(path))
                records[PPO].append(str(path))
    records[ZERO]=[]; old_random=[]
    for entry in entries:
        for seed in old.SEEDS:
            for arm in (old.RANDOM,ZERO):
                path=ablation_root/'runs'/entry['instance_id']/f'seed_{seed}'/arm/'RUN_COMPLETE.json'
                r=retain(path,files)
                old.validate_new(r,old.make_task(oldplan,entry,seed,arm),oldplan)
                if arm==ZERO:
                    if any(g['disabled'] or g['threshold_scaled']!=0 for g in r['effective_gates'].values()):
                        raise ValueError('Existing zero-threshold control differs')
                    records[ZERO].append(str(path))
                else: old_random.append(r)
    groups={a:[read(Path(x)) for x in paths] for a,paths in records.items()}
    controls={(r['instance'],r['seed']):r for r in groups['BASE']}
    for arm,rows in groups.items():
        old.validate_panel(rows,entries,settings)
        for r in rows:
            p.check_pair(r,controls[r['instance'],r['seed']])
            if r['stochastic']: raise ValueError('Unexpected stochastic reused deployment')
            if arm not in (PPO,ZERO) and r['arm']!=arm: raise ValueError('Wrong historical method')
    if any(r['model_bundle_sha256']!=digest(root/'shared_model/BUNDLE.json') for r in groups[SHARED]):
        raise ValueError('Shared comparison model differs')
    target_audit=[]
    for c in 'ICM':
        n=sum(r['selection'][c]['calls'] for r in groups[SHARED]); k=sum(r['selection'][c]['switches'] for r in groups[SHARED])
        on=sum(r['random_selection'][c]['calls'] for r in old_random)
        eligible=on-sum(r['random_selection'][c]['no_legal_alternative'] for r in old_random)
        target_audit.append(dict(component=c,target_percent=100*TARGETS[c],shared_calls=n,shared_switches=k,
            shared_actual_percent=100*k/n,old_random_calls=on,old_random_eligible=eligible,
            old_eligible_fraction=eligible/on,
            old_path_fixed_probability_needed=TARGETS[c]*on/eligible if eligible else None,
            note='Old eligibility is diagnostic only; new policy controls counts online without HV.'))
    old_values=old_table_rows(ppo_root,ablation_root,source,files)
    freeze(root/'OLD_TABLE_VALUES.json',old_values); files[str(root/'OLD_TABLE_VALUES.json')]=digest(root/'OLD_TABLE_VALUES.json')
    freeze(root/'TARGET_AUDIT.json',target_audit); files[str(root/'TARGET_AUDIT.json')]=digest(root/'TARGET_AUDIT.json')
    print('[ABLATION-V2] Backing up original records and tables',flush=True)
    p.verify(files)
    backup_sha=snapshot_backup(root,files)
    p.verify(files)
    plan=dict(protocol='single-shared-random-rate-controlled-union-v2',ablation_source=str(ablation_root),ppo_source=str(ppo_root),
        entries=entries,settings=settings,normalization=oldplan['normalization'],shared_model=str(root/'shared_model'),
        files=files,source_records=records,selected_ppo_update=u,backup_sha256=backup_sha,
        targets=TARGETS,expected_scenarios={f"{r['instance']}/{r['seed']}":r['scenario_digest'] for r in groups['BASE']},
        methods=list(NAMES),new_searches=54,reused_method_rows=648,training_updates=0,
        old_random_total_search_seconds=sum(r['optimisation_seconds'] for r in old_random),
        hv_divisor=VOLUME,bootstrap_draws=DRAWS,bootstrap_seed=BOOTSTRAP_SEED,
        rate_rule='Only eligible decisions: q=clip(target*total_calls-switches_so_far,0,1); uniform legal non-rule choice; separate component random streams.',
        statistics='Fixed 18 instances; resample three paired seeds WITHIN each instance 50000 times; average seeds, instances within configuration, then nine configurations equally. No instance/configuration resampling.',
        qualification='Development diagnostic; adaptive intervention timing differs from learned gating. Actual rates, not guaranteed equality, are reported; no outcome-based retuning or seed replacement.')
    plan['sha256']=canonical(plan); freeze(root/'PLAN.json',plan)
    print('[ABLATION-V2] Prepared 54 random-only searches; 648 method rows reused; source backup complete',flush=True)
    return plan


def make_task(plan,entry,seed):
    task=p.scheduler.task(plan,entry['instance_id'],seed,SHARED,'screen',Path(plan['shared_model']))
    task.update(ablation=RANDOM,ablation_seed=policy_seed(entry['instance_id'],seed))
    return task


def jobs_for(root,plan):
    return [(make_task(plan,e,seed),root/'runs'/e['instance_id']/f'seed_{seed}'/RANDOM)
            for e in plan['entries'] for seed in old.SEEDS]


def validate_new(row,task,plan):
    entry=next(e for e in plan['entries'] if e['instance_id']==task['instance'])
    s.audit.validate_run(row,entry,task['seed'],plan['settings'],plan['expected_scenarios'][f"{task['instance']}/{task['seed']}"])
    if (row['plan_sha256']!=plan['sha256'] or row['task_sha256']!=canonical(task) or row['arm']!=RANDOM
        or row['ablation']!=RANDOM or row['model_bundle_sha256']!=task['bundle_sha256']
        or row['target_rates']!=plan['targets']): raise ValueError('New random run provenance differs')
    for c in 'ICM':
        a,b=row['random_selection'][c],row['selection'][c]
        if (a['calls']!=b['calls'] or a['switches']!=b['switches'] or a['fallbacks']!=b['fallbacks']
            or a['calls']!=a['eligible']+a['no_legal_alternative'] or a['calls']!=a['switches']+a['fallbacks']
            or a['switches']>a['eligible']): raise ValueError('Random/controller counters differ')


def worker(root,plan,task,folder):
    if (task,folder) not in jobs_for(root,plan): raise ValueError('Unscheduled random search')
    p.verify(plan['files'])
    import search_b5000 as search
    original,writer=search.SharedPolicy,search.write_json
    policies=[]
    def factory(model):
        policy=RateControlledRandom(plan['targets'],task['ablation_seed']); policies.append(policy); return policy
    def deferred(path,value):
        if Path(path).resolve()!=(folder/'RUN_COMPLETE.json').resolve(): return writer(path,value)
    search.SharedPolicy,search.write_json=factory,deferred
    try:
        row=search.run_search(root/'instances'/(task['instance']+'.json'),plan['settings'],task,folder,plan['sha256'])
    finally: search.SharedPolicy,search.write_json=original,writer
    row.update(arm=RANDOM,ablation=RANDOM,engine_arm=SHARED,stochastic=True,
               ablation_seed=task['ablation_seed'],target_rates=plan['targets'],random_selection=policies[0].counts,
               model_forward_calls=0,training_updates=0,selection_time_includes_model_forward=False)
    validate_new(row,task,plan); write(folder/'RUN_COMPLETE.json',row)


def replacement_report(groups,plan):
    rows=[]; per_run=[]
    for arm in (SHARED,RANDOM,ZERO,RULE):
        for c in 'ICM':
            calls=sum(r['selection'][c]['calls'] for r in groups[arm])
            switches=sum(r['selection'][c]['switches'] for r in groups[arm])
            shared_calls=sum(r['selection'][c]['calls'] for r in groups[SHARED])
            shared_switches=sum(r['selection'][c]['switches'] for r in groups[SHARED])
            actual=100*switches/calls if calls else None
            item=dict(method=arm,component=c,runs=54,calls=calls,switches=switches,fallbacks=calls-switches,
                      target_percent=100*plan['targets'][c],actual_percent=actual,
                      difference_from_target_pp=actual-100*plan['targets'][c] if actual is not None else None,
                      shared_actual_percent=100*shared_switches/shared_calls,
                      difference_from_shared_pp=actual-100*shared_switches/shared_calls if actual is not None else None,
                      eligible=None,no_legal_alternative=None,mean_conditional_probability=None,unserved_target_count=None,
                      eligibility_ceiling_percent=None)
            if arm==RANDOM:
                rr=[r['random_selection'][c] for r in groups[arm]]
                eligible=sum(x['eligible'] for x in rr)
                item.update(eligible=eligible,no_legal_alternative=calls-eligible,
                    mean_conditional_probability=sum(x['conditional_probability_sum'] for x in rr)/eligible if eligible else None,
                    unserved_target_count=plan['targets'][c]*calls-switches,
                    eligibility_ceiling_percent=100*eligible/calls if calls else None)
                for r in groups[arm]:
                    v=r['random_selection'][c]; n=v['calls']
                    per_run.append(dict(instance=r['instance'],configuration=r['configuration'],seed=r['seed'],component=c,
                        target_percent=100*plan['targets'][c],actual_percent=100*v['switches']/n if n else None,**v))
            rows.append(item)
    return rows,per_run


def report(root,plan,new):
    groups={a:[read(Path(x)) for x in names] for a,names in plan['source_records'].items()}
    groups[RANDOM]=new; groups={a:groups[a] for a in NAMES}
    controls={(r['instance'],r['seed']):r for r in groups['BASE']}
    for arm,rows in groups.items():
        old.validate_panel(rows,plan['entries'],plan['settings'])
        for r in rows: p.check_pair(r,controls[r['instance'],r['seed']])
    normalized,bounds=s.audit.union_normalize(groups)
    # New report directory is published only when complete. Never edit old tables.
    dest=root/'report_build'; dest.mkdir(exist_ok=True)
    panel=Panel(plan['entries'],old.SEEDS)
    summary=[]; direct=[]; details=[]
    for budget in BUDGETS:
        for arm,ref in [(a,'BASE') for a in NAMES]+[(SHARED,RANDOM),(RANDOM,RULE),(SHARED,ZERO)]:
            print(f'[ABLATION-V2] statistics B{budget} {arm}/{ref}; bootstrap={DRAWS}',flush=True)
            row,pairs=panel.compare(normalized[arm],normalized[ref],budget,arm,ref,NAMES)
            (summary if ref=='BASE' else direct).append(row); details.extend(pairs)
    rates,rate_details=replacement_report(groups,plan)
    before={(r['method'],r['evaluation']):r for r in read(root/'OLD_TABLE_VALUES.json')}
    changed=[]
    for row in summary:
        prior=before[row['method'],row['评价次数']]
        changed.append(dict(method=row['method'],方法=row['方法'],评价次数=row['评价次数'],
            旧提升百分比=prior['old_percent'],新提升百分比=row['平均配对提升百分比'],
            变化百分点=row['平均配对提升百分比']-prior['old_percent'],
            旧区间下限=prior['old_low'],旧区间上限=prior['old_high'],新区间下限=row['提升95区间下限'],新区间上限=row['提升95区间上限'],
            旧平均HV原样=prior['old_hv_as_reported'],旧平均HV换算到01=prior['old_hv_01'],新平均HV=row['平均HV'],
            旧胜=prior['source_row']['胜'],旧平=prior['source_row']['平'],旧负=prior['source_row']['负'],新胜=row['胜'],新平=row['平'],新负=row['负'],
            旧值来源=prior['source'],
            说明='随机策略、联合边界、bootstrap口径更新' if row['method']==RANDOM else '原搜索和时间复用；联合边界、bootstrap口径更新'))
    table=wide(summary,NAMES)
    outputs={'TABLE5.csv':table,'ALL_RESULTS.csv':summary,'ABLATIONS.csv':[r for r in table if r['method'] in (RANDOM,ZERO)],
             'ABLATIONS_LONG.csv':[r for r in summary if r['method'] in (RANDOM,ZERO)],
             'DIRECT_COMPARISONS.csv':direct,'PER_PAIR.csv':details,'OLD_NEW.csv':changed,
             'REPLACEMENT_RATES.csv':rates,'REPLACEMENT_PER_RUN.csv':rate_details}
    for name,rows in outputs.items(): p.csv_write(dest/name,rows)
    write(dest/'INSTANCE_BOUNDS.json',bounds)
    write(dest/'REPORT.json',dict(completed=True,plan_sha256=plan['sha256'],methods=list(NAMES),pairs_per_method=54,
        new_searches=54,reused_method_rows=648,selected_ppo_update=plan['selected_ppo_update'],
        hv_reference=[1.1]*3,hv_divisor=VOLUME,hv_already_divided=True,
        normalization='One union per instance: exactly the 13 displayed methods, all seeds and 400/1000/5000. Old random and old PPO fronts excluded.',
        aggregation='Mean paired percentages over seeds within instance, then instances within configuration, then configurations equally.',
        bootstrap=plan['statistics'],bootstrap_draws=DRAWS,bootstrap_seed=BOOTSTRAP_SEED,wtl_tolerance=TOLERANCE,
        primary_reference='BASE',time='Original optimisation_seconds; no subtraction, scaling, or replacement of historical timing.',
        rate_rule=plan['rate_rule'],qualification=plan['qualification'],old_results_modified=False))
    lines=['修正版随机组与统一HV报告','每种方法54次；仅随机组新增54次；其他方法和耗时复用。',
           'HV已经除以1.331。全部提升均为配对百分比，主表参照Base。',
           '95%区间：18个实例固定，在每实例内同步重抽配对种子50000次；不重抽实例。',
           '该区间描述固定开发实例上的种子波动，不能当作新实例泛化区间。',
           '组件 | 目标替换率 | 修正后实际替换率 | 差异（百分点） | 原共享监督实际替换率']
    for r in rates:
        if r['method']==RANDOM:
            lines.append(f"{r['component']} | {r['target_percent']:.3f}% | {r['actual_percent']:.6f}% | {r['difference_from_target_pp']:+.6f} | {r['shared_actual_percent']:.6f}%")
    lines += ['实际替换率未被改写。无合法候选或最后几次调用可能造成未消除的差额；不自动重抽种子或按HV选运行。',
              'TABLE5.csv / ABLATIONS.csv / DIRECT_COMPARISONS.csv / OLD_NEW.csv',
              '旧表可能使用不同归一化池及分层bootstrap；旧值按原报告保存，不冒充同口径统计。']
    (dest/'REPORT.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    p.verify(plan['files'])
    if digest(root/'backup/OLD_RESULTS.tar.gz')!=plan['backup_sha256']: raise ValueError('Backup changed')
    output=root/'results'
    if output.exists():
        output.rename(root/f'results_previous_{time.time_ns()}')
    dest.rename(output)
    print('\n'+'\n'.join(lines),flush=True)
    return output


def execute(args):
    root,ablation,ppo=args.out.resolve(),args.ablation_source.resolve(),args.ppo_source.resolve()
    if any(root==v or root in v.parents or v in root.parents for v in (ablation,ppo)):
        raise ValueError('A separate output directory is required')
    if (root/'PLAN.json').exists() and read(root/'PLAN.json').get('protocol')!='single-shared-random-rate-controlled-union-v2':
        raise ValueError('Refusing to write to an output directory owned by another experiment')
    root.mkdir(parents=True,exist_ok=True); p.torch.set_num_threads(1)
    with (root/'.pipeline.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        saved=(p.scheduler.HERE2,p.scheduler.valid_completion)
        def stop(sig,frame): raise p.scheduler.Paused(f'Signal {sig}')
        signal.signal(signal.SIGTERM,stop); signal.signal(signal.SIGINT,stop)
        try:
            plan=prepare(root,ablation,ppo); jobs=jobs_for(root,plan)
            original=p.scheduler.valid_completion
            def checked_completion(folder,task,frozen):
                r=original(folder,task,frozen)
                if r is not None: validate_new(r,task,frozen)
                return r
            p.scheduler.valid_completion=checked_completion; p.scheduler.HERE2=HERE
            done=sum(checked_completion(f,t,plan) is not None for t,f in jobs)
            print(f'[ABLATION-V2] random {done}/54; workers={args.workers}; targets={plan["targets"]}',flush=True)
            eta=plan['old_random_total_search_seconds']*(54-done)/54/args.workers/60
            print(f'[ABLATION-V2] Remaining search estimate from old random timings: {eta:.1f} minutes; excludes reporting/startup/contention.',flush=True)
            for value in read(root/'TARGET_AUDIT.json'):
                print('[ABLATION-V2] Old eligibility diagnostic:',value,flush=True)
            if args.prepare_only:
                write(root/'STATUS.json',dict(completed=False,stage='prepared',done=done,total=54,new_searches=0)); return
            if args.report_only:
                if done!=54: raise ValueError('Report-only requires all 54 corrected random runs')
                new=[checked_completion(f,t,plan) for t,f in jobs]
            else:
                new=p.scheduler.execute_jobs(root,plan,jobs,args.workers,'corrected_random/54')
            write(root/'STATUS.json',dict(completed=False,stage='union_hv_and_statistics',done=54,total=54,training_updates=0))
            result=report(root,plan,new)
            write(root/'STATUS.json',dict(completed=True,stage='all',done=54,total=54,new_searches=54,
                reused_method_rows=648,training_updates=0,original_results_modified=False,results=str(result),hv_already_divided=True))
            print('[ABLATION-V2] COMPLETE',result,flush=True)
        except p.scheduler.Paused as exc:
            write(root/'STATUS.json',dict(completed=False,paused=True,reason=str(exc),resume='Repeat same command')); raise SystemExit(str(exc))
        except Exception as exc:
            write(root/'STATUS.json',dict(completed=False,failed=True,error=str(exc))); raise
        finally: p.scheduler.HERE2,p.scheduler.valid_completion=saved


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ablation-source',type=Path,default=DIAG/'ccp_shared_sl_ablations_b5000_v1')
    parser.add_argument('--ppo-source',type=Path,default=DIAG/'ccp_single_ppo_s0_s8_b5000_v1')
    parser.add_argument('--out',type=Path,default=DIAG/'ccp_shared_sl_ablations_b5000_v2')
    parser.add_argument('--workers',type=int,choices=(1,2),default=2)
    parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--report-only',action='store_true')
    parser.add_argument('--worker',action='store_true')
    parser.add_argument('--task',type=Path); parser.add_argument('--run-out',type=Path)
    args=parser.parse_args()
    if args.worker: worker(args.out.resolve(),old.checked(args.out/'PLAN.json'),read(args.task),args.run_out.resolve())
    else: execute(args)


if __name__=='__main__': main()
