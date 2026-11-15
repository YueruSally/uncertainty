#!/usr/bin/env python3
"""Frozen four-method test on new S0-S11 batch sets. No model fitting or selection."""
import argparse
import copy
from dataclasses import asdict
from datetime import datetime, timezone
import fcntl
import gzip
import importlib.util
import json
from pathlib import Path
import resource
import signal
import socket
import sys
import time

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('final_evidence', HERE.parent/'paper_evidence_b5000_v1/collect.py')
ev = importlib.util.module_from_spec(spec); spec.loader.exec_module(ev)
s, p = ev.s, ev.p
read, write, digest, canonical, freeze = s.read, s.write, s.digest, s.canonical, s.freeze
scheduler = p.scheduler; scheduler.HERE2 = HERE
from search_b5000 import SharedPolicy, enabled_components
from transformer_ppo.instance_catalog import _batches, _find_path_seed, SCENARIOS

ARMS = ('BASE', 'C_RULE', 'ICM_SHARED_SUPERVISED', 'ICM_TERMINAL_PPO')
NAMES = dict(BASE='Base', C_RULE='Simple（仅交叉规则）',
             ICM_SHARED_SUPERVISED='ICM 共享监督 Transformer', ICM_TERMINAL_PPO='ICM PPO u1（最高分选择）')
BUDGETS = (400, 1000, 5000)


def validate_protocol(protocol):
    if (protocol['configurations'] != ['S'+str(i) for i in range(12)] or
            protocol['instances_per_configuration'] != 3 or tuple(protocol['methods']) != ARMS or
            len(protocol['algorithm_seeds']) != 3 or len(set(protocol['algorithm_seeds'])) != 3 or
            protocol['primary_method'] != 'ICM_SHARED_SUPERVISED' or protocol['primary_endpoint'] != 5000):
        raise ValueError('Expected the approved 12 x 3 x 3 x 4 test protocol')


def signature(entry):
    return canonical(entry['batches'])


def known_instances(diag, exclude):
    """Read instance manifests only; do not examine outcomes or choose favorable seeds."""
    known = set(); files = {}
    for path in sorted(diag.glob('*/instances/*.json')):
        if exclude in path.parents: continue
        data = read(path)
        entries = data.get('instances', []) if isinstance(data, dict) else []
        if isinstance(data, dict) and 'batches' in data: entries = [data]
        used = False
        for entry in entries:
            if isinstance(entry, dict) and 'batches' in entry:
                known.add(signature(entry)); used = True
        if used: files[str(path)] = digest(path)
    if not known: raise ValueError('No historical batch manifests available for overlap check')
    return known, files


def generate(root, old, protocol):
    sample = old['entries'][0]
    source_catalog_path = Path(old['settings']['source_catalog_default'])
    catalog = read(source_catalog_path)
    if (catalog['catalog_digest'] != old['settings']['source_catalog_digest'] or
            canonical({k:v for k,v in catalog.items() if k!='catalog_digest'}) != catalog['catalog_digest']):
        raise ValueError('Original OD catalog changed')
    workbook = p.EXPANDA/'formal_icm_s0_v1/assets/data/data_expanded.xlsx'
    if digest(workbook) != catalog['source_data_sha256']: raise ValueError('Workbook changed')
    # Keep the existing OD and quantity/release distribution. This is new-batch
    # and size generalization, not an extra unseen-OD or odd-grid experiment.
    od = [tuple(x) for x in catalog['train_od_pool']]
    known, sources = known_instances(root.parent, root)
    sources[str(source_catalog_path)] = digest(source_catalog_path)
    p.base.BORDER_EVENT_DEFINITIONS = p.base.load_border_event_definitions(p.base.DEFAULT_BORDER_EVENT_DATA_FILE)
    nodes, regions, _, _, _, arcs, timetables, raw, *_ = p.base.load_network_from_extended(str(workbook))
    tt = p.base.build_timetable_dict(timetables); lookup = p.base.build_arc_lookup(arcs)
    records = []; initial = len(known); release = sample['release_time_definition']
    for ci, cfg in enumerate(protocol['configurations']):
        k, alpha = SCENARIOS[cfg]
        for j in range(protocol['instances_per_configuration']):
            index = ci*protocol['instances_per_configuration']+j
            seed = protocol['instance_seed_base']+index*100
            batches = _batches(raw, od, k, alpha, seed, 'train', release['maximum_h'], release['step_h'])
            batch_data = [asdict(b) for b in batches]; sig = canonical(batch_data)
            if sig in known: raise ValueError('Batch-set collision; stop, never silently redraw')
            known.add(sig); name = f'final-b5000-{cfg}-{j+1:03d}'
            r = {key:copy.deepcopy(sample[key]) for key in ('schema_version','alpha_definition','release_time_definition','quantity_definition')}
            r['release_time_definition']['split_disjoint_slot_parity'] = False
            r['quantity_definition']['split_disjoint_centi_teu_parity'] = False
            r.update(instance_id=name, configuration=cfg, K=k, deadline_window_alpha=alpha,
                split='test', development_role='final_test', instance_seed=seed,
                path_seed=_find_path_seed(seed+1000000,nodes,regions,arcs,batches,tt,lookup),
                algorithm_seed=protocol['algorithm_seeds'][0], ccp100_seed=protocol['scenario_seed_base']+index,
                oos5000_seed=None, od_pool='train', source_data_sha256=digest(workbook),
                source_catalog_digest=catalog['catalog_digest'], batch_signature=sig, batches=batch_data,
                sampling_definition='Fresh joint batch sets from unchanged train OD, even centi-TEU and release-slot grids. Test membership is by new seed and distinct complete batch set, not by individual scalar values.')
            r['instance_digest'] = canonical(r); freeze(root/'instances'/(name+'.json'), r)
            p.read_instance(root/'instances'/(name+'.json')); records.append(r)
            print(f'[FINAL-PREPARE] {len(records)}/36 {name}; zero objective evaluations', flush=True)
    audit = dict(historical_unique_batch_sets=initial, source_files=sources, new_batch_sets=len(records),
        no_identical_batch_sets=True, outcome_based_resampling=False,
        qualification='Overlap checked against available historical instance manifests; previously used S11 configurations are not claimed never seen during method development.')
    freeze(root/'EXCLUSION_AUDIT.json', audit)
    freeze(root/'instances/catalog.json', dict(instances=records, protocol=protocol))
    return records


def prepare(root, source):
    if (root/'PLAN.json').exists():
        existing=s.checked_plan(root)
        if existing['source']!=str(source):raise ValueError('Resume source changed')
        return existing
    screen = s.checked_plan(source)
    if not read(source/'STATUS.json').get('completed'): raise ValueError('Development comparison incomplete')
    old = read(Path(screen['original'])/'PLAN.json'); pilot = read(Path(screen['source'])/'SELECTION.json')
    if screen['selected_update'] != 1 or pilot['best_updated_checkpoint'] != 1: raise ValueError('Expected frozen u1')
    protocol = read(HERE/'protocol.json'); validate_protocol(protocol); settings = copy.deepcopy(old['settings'])
    if (settings['population'], settings['budget'], settings['scenario_count'], settings['progress_denominator']) != (250,5000,100,50000):
        raise ValueError('Incompatible frozen search configuration')
    if tuple(settings['milestones']) != BUDGETS: raise ValueError('Milestone schedule differs')
    settings['invocation_seconds'] = protocol['invocation_seconds']
    settings['task_timeout_seconds'] = protocol['task_timeout_seconds']
    models = {'ICM_SHARED_SUPERVISED':str(Path(old['source'])/'shared_model'), 'ICM_TERMINAL_PPO':screen['model']}
    files = {name:sha for name,sha in screen['files'].items()
             if 'transformer_ppo_runs' not in Path(name).parts and Path(name).suffix in ('.py','.json','.csv','.xlsx','.yml')}
    for arm, model in models.items():
        folder = Path(model); b = read(folder/'BUNDLE.json')
        policy = SharedPolicy(folder) if arm=='ICM_SHARED_SUPERVISED' else p.PilotPolicy(folder)
        if b['normalization'] != old['normalization'] or b['settings']['progress_denominator'] != 50000:
            raise ValueError('Frozen model inputs changed')
        files[str(folder/'BUNDLE.json')] = digest(folder/'BUNDLE.json')
        files.update({str(folder/name):sha for name,sha in policy.bundle['checkpoint_sha256'].items()})
    records = generate(root, old, protocol)
    for path in [*HERE.glob('*.py'),HERE/'protocol.json',HERE/'run.sh',Path(ev.__file__),
                 source/'PLAN.json',Path(screen['source'])/'SELECTION.json',root/'EXCLUSION_AUDIT.json',
                 root/'instances/catalog.json',*sorted((root/'instances').glob('final-b5000-*.json'))]:
        files[str(path.resolve())] = digest(path)
    plan = dict(protocol=protocol, settings=settings, source=str(source), source_plan_sha256=screen['sha256'],
        models=models, selected_update=1, original_ppo_selection=pilot, entries=records,
        normalization=old['normalization'], files=files, maximum_searches=432, maximum_objective_evaluations=2160000,
        main_reference='BASE', primary_endpoint=5000, primary_method='ICM_SHARED_SUPERVISED',
        normalization_reporting='Predeclared same-instance union of all four methods, three seeds and 400/1000/5000 fronts; reference 1.1, no /1.331. Frozen training normalization also reported separately.',
        purpose='Fresh batches in S0-S8 and larger batch sets in S9-S11, reported separately. S11 was used during earlier development; these batch sets are new.',
        final_test=True, new_training_updates=0, checkpoint_selection_on_test=False,
        uncertainty_evaluation='100 fixed optimization scenarios per instance; no independent 5000-scenario reevaluation.')
    plan['sha256'] = canonical(plan); freeze(root/'PLAN.json',plan)
    return plan


def task(plan, entry, seed, arm):
    model = plan['models'].get(arm)
    return dict(instance=entry['instance_id'], seed=int(seed), arm=arm, phase='final_test', sample=False, record=False,
        model=model, bundle_sha256=digest(Path(model)/'BUNDLE.json') if model else None, normalization=plan['normalization'])


def validate_task(root, plan, t, folder):
    entry = next((e for e in plan['entries'] if e['instance_id']==t['instance']),None)
    if entry is None or entry['development_role']!='final_test' or entry['split']!='test': raise ValueError('Not a frozen final-test instance')
    if t['arm'] not in ARMS or t['seed'] not in plan['protocol']['algorithm_seeds']: raise ValueError('Unscheduled method/seed')
    if t != task(plan,entry,t['seed'],t['arm']): raise ValueError('Changed model or deployment rule')
    expected = root/'runs'/entry['instance_id']/f'seed_{t["seed"]}'/t['arm']
    if folder.resolve()!=expected.resolve(): raise ValueError('Unexpected worker output')
    return entry


def worker(root, plan, t, out):
    entry = validate_task(root,plan,t,out); p.verify(plan['files']); settings = plan['settings']
    started = datetime.now(timezone.utc).isoformat(); host = ev.hardware(); host['stage']='before_search'
    host['qualification']='Captured immediately before this actual search. Affinity/cgroup restrictions are observed limits, not proof of reserved cores or RAM; per-worker Torch threads are also recorded.'
    p.torch.set_num_threads(1)
    learned = None
    if t['arm']=='ICM_SHARED_SUPERVISED': learned = SharedPolicy(Path(t['model']))
    if t['arm']=='ICM_TERMINAL_PPO': learned = p.PilotPolicy(Path(t['model']),sample=False,record=False,seed=t['seed']+7000000)
    holders=[]
    def factory(*args,**kwargs):
        ctl=p.BudgetController(*args,settings=settings,arm=t['arm'],learned=learned,**kwargs);holders.append(ctl);return ctl
    normalizer=p.legacy_learning.normalizer(plan['normalization'])
    prior=p.base.clear_inherited_evaluation_cache;p.base.clear_inherited_evaluation_cache=lambda *args:None
    try:
        result=p.run_episode(p.EXPANDA/'formal_icm_s0_v1/assets/data/data_expanded.xlsx',root/'instances'/(entry['instance_id']+'.json'),
            out,None,normalizer,algorithm_seed=t['seed'],evaluation_budget=settings['budget'],pop_size=settings['population'],
            generations=1000000,evaluation_mode='ccp',component='BASE',scenario_count=settings['scenario_count'],
            ccp_confidence=settings['confidence'],uncertainty_profile='transport_border_saa_v1',
            inference_controller_factories={'BASE':factory},method_override=t['arm'])
    finally:p.base.clear_inherited_evaluation_cache=prior
    write(out/'ENVIRONMENT.json',host)
    ctl=holders[0];front=[list(p._optimisation_vector(r)) for r in read(out/'final_archive_lineage.json')]
    snaps=[r for r in ctl.budget_fronts if r['evaluation']!=settings['budget']]
    snaps.append(dict(evaluation=settings['budget'],actual_evaluations=result['real_objective_evaluation_count'],
        hv=result['final_hypervolume'],front_raw=front,controller_seconds=time.perf_counter()-ctl.start,
        definition='actual final feasible nondominated front'))
    result.update(instance=entry['instance_id'],configuration=entry['configuration'],instance_digest=entry['instance_digest'],
        seed=t['seed'],arm=t['arm'],phase='final_test',stochastic=False,plan_sha256=plan['sha256'],task_sha256=canonical(t),
        model_bundle_sha256=t['bundle_sha256'],trajectory_sha256=None,final_front_raw=front,budget_fronts=snaps,
        scenario_digest=read(out/'configuration.json')['scenario_digest'],hostname=socket.gethostname(),
        peak_memory_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(1024 if sys.platform!='darwin' else 1024**2),
        worker_started_utc=started,worker_finished_utc=datetime.now(timezone.utc).isoformat(),
        environment_sha256=digest(out/'ENVIRONMENT.json'),torch_num_threads=p.torch.get_num_threads(),
        final_solutions_sha256=digest(out/'final_feasible_nondominated.json'),new_ppo_updates=0)
    validate_run(result,plan)
    for c in 'ICM':
        if c not in enabled_components(t['arm']) and result['selection'][c]['calls']: raise ValueError('Inactive component changed')
    write(out/'RUN_COMPLETE.json',result)


def validate_run(row,plan):
    entry=next(e for e in plan['entries'] if e['instance_id']==row['instance'])
    if row['seed'] not in plan['protocol']['algorithm_seeds'] or row['arm'] not in ARMS: raise ValueError('Unexpected run')
    s.audit.validate_run(row,entry,row['seed'],plan['settings'])
    expected=task(plan,entry,row['seed'],row['arm'])
    if (row['plan_sha256']!=plan['sha256'] or row['task_sha256']!=canonical(expected)
            or row['model_bundle_sha256']!=expected['bundle_sha256'] or row['phase']!='final_test'
            or row['stochastic'] or row['trajectory_sha256'] or row['ppo_updates']): raise ValueError('Run provenance differs')


def validate_panel(rows,plan):
    expected={(e['instance_id'],seed,arm) for e in plan['entries'] for seed in plan['protocol']['algorithm_seeds'] for arm in ARMS}
    if len(rows)!=len(expected) or {(r['instance'],r['seed'],r['arm']) for r in rows}!=expected: raise ValueError('Incomplete/duplicate panel')
    scenarios={}
    for row in rows:
        validate_run(row,plan);scenarios.setdefault(row['instance'],set()).add(row['scenario_digest'])
    if any(len(v)!=1 for v in scenarios.values()): raise ValueError('Instance optimization scenarios differ across seeds or methods')


def report(rows,root,plan,draws=3000):
    validate_panel(rows,plan)
    groups={m:[r for r in rows if r['arm']==m] for m in ARMS}
    normalized,bounds=s.audit.union_normalize(groups);write(root/'results_union/INSTANCE_BOUNDS.json',bounds)
    scopes={'S0-S8_新批次':['S'+str(i) for i in range(9)],'S9-S11_规模泛化':['S'+str(i) for i in range(9,12)]}
    for mode,panel in [('results_union',normalized),('results_frozen',groups)]:
        summary=[];details=[];configs=[]
        for budget in plan['settings']['milestones']:
            at={m:[dict(r,metric=next(x['hv'] for x in r['budget_fronts'] if x['evaluation']==budget)) for r in v] for m,v in panel.items()}
            for scope,cfgs in scopes.items():
                part={m:[r for r in v if r['configuration'] in cfgs] for m,v in at.items()}
                if not part['BASE']:continue
                for m in ARMS:
                    q=p.paired(part[m],part['BASE'],field='metric',draws=draws)
                    lo,hi=p.hv.interval(q['per_run'],'percent',draws)
                    summary.append({'范围':scope,'评价次数':budget,'方法':NAMES[m],'参照':'Base','运行对数':q['pairs'],
                        '平均HV':q['mean_hv'],'相对Base平均配对提升(%)':q['mean_paired_percent'],'提升95%下限(%)':lo,'提升95%上限(%)':hi,
                        '平均HV绝对差':q['mean_absolute_difference'],'绝对差95%下限':q['absolute_ci_low'],'绝对差95%上限':q['absolute_ci_high'],
                        '胜':q['wins'],'平':q['ties'],'负':q['losses'],
                        '平均B5000秒':p.equal_mean(part[m],'optimisation_seconds') if budget==plan['settings']['budget'] else None,
                        '平均推理秒':ev.mean_optional(part[m],'inference_seconds') if budget==plan['settings']['budget'] else None})
                    details.extend(dict(scope=scope,method=m,evaluation=budget,**r) for r in q['per_run'])
                    for cfg in cfgs:
                        v=[r for r in q['per_run'] if r['configuration']==cfg]
                        if v:configs.append(dict(configuration=cfg,evaluation=budget,method=m,pairs=len(v),mean_hv=p.equal_mean(v,'own'),
                            mean_percent_vs_base=p.hv.equal_mean(v,'percent'),mean_absolute_vs_base=p.equal_mean(v,'absolute')))
        p.csv_write(root/mode/'00_新实例测试_相对Base.csv',summary)
        p.csv_write(root/mode/'01_终点结果.csv',[r for r in summary if r['评价次数']==plan['settings']['budget']])
        p.csv_write(root/mode/'PER_RUN.csv',details);p.csv_write(root/mode/'PER_CONFIGURATION.csv',configs)
    destination=root/'fronts';destination.mkdir(exist_ok=True)
    with gzip.open(destination/'FINAL_SOLUTIONS.jsonl.gz','wt',encoding='utf-8') as f:
        for r in rows:
            folder=root/'runs'/r['instance']/f'seed_{r["seed"]}'/r['arm']
            if digest(folder/'final_feasible_nondominated.json')!=r['final_solutions_sha256']:raise ValueError('Final solutions changed')
            f.write(json.dumps(dict(instance=r['instance'],configuration=r['configuration'],seed=r['seed'],method=r['arm'],
                scenario_digest=r['scenario_digest'],raw_objectives=r['final_front_raw'],solutions=read(folder/'final_feasible_nondominated.json')))+'\n')
    p.csv_write(root/'results_union/DEPLOYMENT_COSTS.csv',[dict(instance=r['instance'],configuration=r['configuration'],seed=r['seed'],method=r['arm'],
        search_seconds=r['optimisation_seconds'],inference_seconds=r.get('inference_seconds'),feature_seconds=r.get('feature_construction_seconds'),
        objective_seconds=r.get('objective_evaluation_seconds'),peak_memory_mib=r['peak_memory_mib'],hostname=r['hostname']) for r in rows])
    write(root/'REPORT.json',dict(completed=True,searches=len(rows),reference='BASE',primary_method=plan['primary_method'],
        checkpoint_selection_on_test=False,model_updates=0,independent_scenario_reevaluation=False,
        total_actual_evaluations=sum(r['real_objective_evaluation_count'] for r in rows),
        interval='Paired bootstrap of instances then seeds within each fixed configuration; configuration-equal aggregation; no max-run selection.',
        normalization=plan['normalization_reporting'],purpose=plan['purpose']))


def execute(a):
    root=a.out.resolve();source=a.source.resolve()
    if root==source or root in source.parents or source in root.parents:raise ValueError('Separate test directory required')
    root.mkdir(parents=True,exist_ok=True);p.torch.set_num_threads(1)
    with (root/'.pipeline.lock').open('a+') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise SystemExit('Test already running')
        def stop(signum,frame):raise scheduler.Paused(f'Received signal {signum}')
        signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
        try:
            plan=prepare(root,source)
            write(root/'ENVIRONMENT_LAUNCH.json',dict(environment=ev.hardware(),requested_workers=a.workers,
                per_worker_threads=1,launch_utc=datetime.now(timezone.utc).isoformat()))
            write(root/'STATUS.json',dict(completed=False,stage='prepared',final_test_opened=(root/'TEST_OPENED.json').exists(),total=432))
            if a.prepare_only:
                print('[FINAL] 36 instances / 432 fixed tasks prepared; no searches yet',flush=True);return
            if not (root/'TEST_OPENED.json').exists():
                write(root/'TEST_OPENED.json',dict(opened=True,plan_sha256=plan['sha256'],opened_utc=datetime.now(timezone.utc).isoformat()))
            jobs=[(task(plan,e,seed,m),root/'runs'/e['instance_id']/f'seed_{seed}'/m)
                  for e in plan['entries'] for seed in plan['protocol']['algorithm_seeds'] for m in ARMS]
            print('[FINAL] 432 searches, two separately reported scopes; no refitting or checkpoint reselection',flush=True)
            rows=scheduler.execute_jobs(root,plan,jobs,a.workers,'final_test')
            report(rows,root,plan)
            write(root/'CONCURRENCY.json',ev.concurrency(root))
            write(root/'STATUS.json',dict(completed=True,stage='all',done=len(rows),total=432,final_test_opened=True,
                new_training_updates=0,automatic_further_experiments=False))
            print('[FINAL] COMPLETE; results_union/01_终点结果.csv',flush=True)
        except scheduler.Paused as e:
            write(root/'STATUS.json',dict(completed=False,paused=True,reason=str(e),resume='same command; completed runs reused'))
            print('[FINAL] PAUSED '+str(e),flush=True)
        except Exception as e:
            write(root/'STATUS.json',dict(completed=False,failed=True,error=str(e)));raise


def main():
    diag=HERE.parent/'transformer_ppo_runs/diagnostics';parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=diag/'ccp_joint_ppo_b5000_screen27_v1')
    parser.add_argument('--out',type=Path,default=diag/'ccp_final_generalization_b5000_v1')
    parser.add_argument('--workers',type=int,default=2);parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--worker',action='store_true');parser.add_argument('--task',type=Path);parser.add_argument('--run-out',type=Path)
    a=parser.parse_args()
    if a.workers<1:parser.error('workers must be positive')
    if a.worker:worker(a.out,s.checked_plan(a.out),read(a.task),a.run_out)
    else:execute(a)


if __name__=='__main__':main()
