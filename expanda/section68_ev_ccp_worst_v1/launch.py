#!/usr/bin/env python3
"""Section 6.8: gated pilot and formal runs; never change historical outputs."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import fcntl
import json
import importlib.metadata
import os
from pathlib import Path
import platform
import shutil
import socket
import subprocess
import sys
import time
import traceback

sys.dont_write_bytecode = True
for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[k] = '1'
os.environ['MPLBACKEND'] = 'Agg'
from preflight import EXP, DIAG, HERE, read, save, sha, canonical, command
from engine import (Evaluator, base, csv_write, decision_id, objectives, select_plans,
                    save_scenarios, load_scenarios, scenario_digest, search, verify, OBJECTIVES)
import numpy as np

MODELS = ('EV', 'CCP', 'WORST')


def stamp():
    return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')


def status(root, **fields):
    r = dict(updated_utc=stamp(), pid=os.getpid(), **fields)
    path = root / 'STATUS.json'
    tmp = root / 'STATUS.writing'
    tmp.write_text(json.dumps(r, ensure_ascii=False, indent=2)+'\n')
    tmp.replace(path)
    with (root/'PROGRESS.jsonl').open('a') as f:
        f.write(json.dumps(r, ensure_ascii=False)+'\n')
    print('[SECTION68]', json.dumps(r, ensure_ascii=False), flush=True)


def environment_review(preflight_dir):
    old_path = DIAG/'ccp_paper_evidence_b5000_v1/ENVIRONMENT.json'
    old = read(old_path)
    now_cpu = command(['lscpu', '-J'])
    if now_cpu['returncode']:
        raise ValueError('Cannot read current CPU identity')
    cpu = {x['field'].rstrip(':'): x['data'] for x in json.loads(now_cpu['stdout'])['lscpu']}
    memory_kb = int(next(x for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemTotal:')).split()[1])
    got = dict(hostname=socket.gethostname(), python=platform.python_version(),
        cpu_model=cpu.get('Model name'), logical_cpus=os.cpu_count(), total_node_memory_bytes=memory_kb*1024)
    expected = {k: old.get(k) for k in got}
    # Exact machine and total memory check. Never call total cores/RAM a reservation.
    if got != expected:
        raise ValueError('Environment differs or original evidence lacks fields: '+json.dumps(dict(current=got, original=expected)))
    packages={k:importlib.metadata.version(k) for k in ('numpy','torch','pandas','openpyxl','matplotlib')}
    for k in ('numpy','torch'):
        if old.get(k) != packages[k]:
            raise ValueError(f'Package differs from original evidence: {k}, current={packages[k]}, original={old.get(k)}')
    return dict(passed=True, original_file=str(old_path), original_sha256=sha(old_path),
        current=got, original=expected, current_affinity=sorted(os.sched_getaffinity(0)),
        workers_limit=2, numerical_threads=1, packages=packages,
        historical_reserved_cores=old.get('historical_allocated_cores'),
        historical_reserved_memory_bytes=old.get('historical_allocated_memory_bytes'),
        qualification='Node hardware and Python match recorded evidence. Unknown historical resource reservations remain unknown.')


def inspect_existing_sources(audit, selected, out):
    """Inspect the full server inventory; refuse ambiguous 5000-scenario reuse."""
    inventory = read(audit/'EXISTING_SCENARIO_AND_ENVIRONMENT_FILES.json')
    names = {s['instance'] for s in selected}
    digests = {s['instance_digest'] for s in selected}
    candidates, ambiguities, reuse = [], [], {}
    for item in inventory:
        if item['kind'] not in ('candidate_scenario_file', 'experiment_manifest'):
            continue
        path = Path(item['path'])
        if path.suffix.lower() != '.json':
            # A matching raw pack without inspectable provenance cannot be silently replaced.
            sidecar=path.with_suffix('.json')
            side=read(sidecar) if sidecar.exists() else None
            known=isinstance(side,dict) and side.get('format')=='section68-network-scenarios-v1'
            if item.get('selected_instance_in_path') and not known:
                ambiguities.append(str(path))
            continue
        obj = read(path)
        raw = json.dumps(obj, ensure_ascii=False)
        matched = [s for s in selected if s['instance'] in raw or s['instance_digest'] in raw]
        record = dict(path=str(path), sha256=sha(path), matched_instances=[s['instance'] for s in matched])
        candidates.append(record)
        if not matched:
            record['disposition'] = 'different instances'
            continue
        if isinstance(obj,dict) and obj.get('format') == 'section68-network-scenarios-v1' and obj.get('size') == 5000:
            for s in matched:
                if obj['identity'].get('instance_digest') == s['instance_digest']:
                    if s['configuration'] in reuse:
                        ambiguities.append('Multiple packs: '+s['configuration'])
                    reuse[s['configuration']] = str(path.with_suffix(''))
            record['disposition'] = 'candidate reusable raw pack'
            continue
        # An experiment plan can contain unused oos5000_seed:null entries.
        # A non-null independent seed is an unresolved provenance record, not absence.
        def walk(x):
            if isinstance(x, dict):
                if x.get('instance_id', x.get('instance')) in names or x.get('instance_digest') in digests:
                    if x.get('oos5000_seed') is not None:
                        ambiguities.append(dict(path=str(path), record=x.get('instance_id', x.get('instance')),
                                                seed=x['oos5000_seed']))
                for v in x.values():
                    walk(v)
            elif isinstance(x, list):
                for v in x:
                    walk(v)
        walk(obj)
        if item['kind'] == 'candidate_scenario_file':
            # Never infer 'not 5000' from the directory name (s1000 etc.).
            ambiguities.append(dict(path=str(path), reason='matching instance; unsupported OOS schema'))
            record['disposition'] = 'requires provenance review'
        else:
            record['disposition'] = 'manifest inspected; no reusable pack identified'
    result = dict(scope='Full preflight inventory of diagnostics manifests and scenario candidates',
                  inspected=candidates, ambiguous=ambiguities, reusable=reuse)
    save(out/'EXISTING_SOURCE_REVIEW.json', result)
    if ambiguities:
        raise ValueError('Existing OOS provenance needs review; see EXISTING_SOURCE_REVIEW.json')
    return reuse


def prepare(args, root):
    audit = args.preflight.resolve()
    p = read(audit/'PREFLIGHT.json')
    if not p['completed'] or p['status'] != 'awaiting_source_review' or not all(c['passed'] for c in p['checks']):
        raise ValueError('Preflight not passed')
    verify(read(audit/'SOURCE_HASHES.json'))
    if socket.gethostname().split('.')[0] != 'lnx-grid-41':
        raise ValueError('Must run on lnx-grid-41')
    env = environment_review(audit)
    save(root/'ENVIRONMENT.json', env)
    print('[SECTION68] Original environment verified', flush=True)
    reuse = inspect_existing_sources(audit, p['selected'], root)
    t5 = read(Path(p['sources'][0])/'PLAN.json')
    files = read(audit/'SOURCE_HASHES.json')
    for name in ('PREFLIGHT.json', 'SOURCE_HASHES.json', 'EXISTING_SCENARIO_AND_ENVIRONMENT_FILES.json', 'ENVIRONMENT.json'):
        files[str(audit/name)] = sha(audit/name)
    files[env['original_file']] = env['original_sha256']
    for path in HERE.glob('*.py'):
        files[str(path)] = sha(path)
    (root/'instances').mkdir(); (root/'scenarios').mkdir(); (root/'runs').mkdir(); (root/'logs').mkdir()
    selected = []
    for s0 in p['selected']:
        s = dict(s0)
        cfg = s['configuration']
        instance = root/'instances'/(s['instance']+'.json')
        save(instance, read(s['instance_file']))
        files[str(instance)] = sha(instance)
        s['local_instance'] = str(instance)
        s['nominal_options_sha256'] = canonical(read(audit/(cfg+'_NOMINAL_PATH_OPTIONS.json')))
        ev = Evaluator(instance, p['uncertainty'])
        directory = root/'scenarios'/cfg; directory.mkdir()
        identity = dict(instance=s['instance'], instance_digest=s['instance_digest'],
                        network_sha256=sha(__import__('preflight').DATA), uncertainty=p['uncertainty'])
        opt = ev.make_scenarios(100, s['optimization_seed'])
        if scenario_digest(opt) != s['optimization_scenario_digest']:
            raise ValueError('Optimization scenario reproduction mismatch')
        save_scenarios(directory/'optimization_100', opt, identity)
        save_scenarios(directory/'expected_input', ev.ev_scenarios(), identity)
        if cfg in reuse:
            source_stem = Path(reuse[cfg])
            oos, meta = load_scenarios(source_stem)
            if meta['identity'] != identity or oos.seed == opt.seed or not oos.stochastic:
                raise ValueError('Existing independent pack has incompatible provenance')
            provenance = dict(action='reuse raw pack', source=str(source_stem))
            for suffix in ('.json', '.npz'):
                files[str(source_stem.with_suffix(suffix))] = sha(source_stem.with_suffix(suffix))
        else:
            if s.get('existing_oos_seed') is not None:
                raise ValueError('Independent seed exists but its pack was not located')
            # Separate fixed stream, chosen before any output is inspected.
            seed = 368500000 + int(cfg[1:])
            if seed in [x['optimization_seed'] for x in p['selected']]+s['search_seeds']:
                raise ValueError('Independent seed collides with optimization/search seeds')
            oos = ev.make_scenarios(5000, seed)
            provenance = dict(action='new independent stream', reason='No reusable matching pack identified in source inventory')
        if oos.size != 5000:
            raise ValueError('OOS realization count is not 5000')
        meta = save_scenarios(directory/'independent_5000', oos, identity)
        s['independent_seed'] = oos.seed
        s['independent_scenario_digest'] = meta['scenario_digest']
        s['independent_provenance'] = provenance
        for f in directory.iterdir():
            files[str(f)] = sha(f)
        for match in s['exact_table5_reproduction_sources']:
            files[match['front']] = sha(match['front'])
        selected.append(s)
        print('[SECTION68] Saved', cfg, 'optimization=100, independent=5000, seed=', oos.seed, flush=True)
    verify(files)
    tasks = [dict(configuration=s['configuration'], instance=s['instance'], seed=seed, model=model)
             for s in selected for seed in s['search_seeds'] for model in MODELS]
    plan = dict(protocol='section68-ev-ccp-worst-v1', output=str(root), preflight=str(audit),
        prepared_at_utc=stamp(), commit=command(['git','rev-parse','HEAD']),
        selected=selected, tasks=tasks, settings=t5['settings'], normalization=t5['normalization'],
        uncertainty=p['uncertainty'], files=files, workers_limit=2,
        independent_size=5000, quantile='ceil(alpha*N)-th smallest; no interpolation',
        timing='Original solver optimisation_seconds; re-evaluation measured separately',
        pilot_task_indices=[0,1,2], final_searches=18,
        counts='Plan evaluations, including one expected-input evaluation for EV; realization counts recorded separately',
        stop_on_nonexact_pilot_budget=True)
    plan['sha256'] = canonical(plan)
    save(root/'PLAN.json', plan)
    return plan


def task_dir(root, task):
    return root/'runs'/task['instance']/('seed_'+str(task['seed']))/task['model']


def completed_task(root, task, plan):
    directory = task_dir(root, task)
    finished = []
    for path in sorted(directory.glob('attempt_*/COMPLETE.json')):
        m = read(path)
        if m['plan_sha256'] != plan['sha256']:
            raise ValueError('Completed task belongs to another plan')
        verify(m['files'])
        finished.append(path.parent)
    if len(finished) > 1:
        raise ValueError('Multiple completed attempts; refusing to choose results')
    return finished[0] if finished else None


def summarize_values(ind, values):
    rows = {}
    for j, obj in enumerate(OBJECTIVES):
        rows[obj+'_mean'] = float(values[j].mean())
        rows[obj+'_p90'] = base.empirical_ccp_quantile(values[j], .9)
        rows[obj+'_max'] = float(values[j].max())
    rows['punctuality_percent'] = 100.*float(np.mean(list(ind.batch_on_time_prob.values())))
    return rows


def reevaluate(root, plan, task, folder, result):
    selected = next(s for s in plan['selected'] if s['instance'] == task['instance'])
    ev = Evaluator(selected['local_instance'], plan['uncertainty'])
    front = read(folder/'search/final_feasible_nondominated.json')
    choices, rule = select_plans(front)
    save(folder/'SELECTION.json', dict(indices=choices, scaling=rule,
         choices={k: dict(decision_id=decision_id(front[i]), own_objectives=objectives(front[i]).tolist()) for k,i in choices.items()}))
    save(folder/'FINAL_FRONT.json', [dict(r, evaluation_model=task['model'], solver='Base NSGA-II') for r in front])
    scene = root/'scenarios'/task['configuration']
    optimization, _ = load_scenarios(scene/'optimization_100')
    expected, _ = load_scenarios(scene/'expected_input')
    independent, _ = load_scenarios(scene/'independent_5000')
    cross, summaries, routes = [], [], []
    started = time.monotonic()
    # Reproduce every saved own-model objective before using any OOS values.
    ev.activate(expected if task['model']=='EV' else optimization, task['model'])
    worst_error = 0.
    for row in front:
        ind, _, _ = ev.evaluate(row)
        error = float(np.max(np.abs(np.asarray(ind.objectives)-objectives(row))))
        worst_error = max(worst_error, error)
        if not np.array_equal(np.asarray(ind.objectives), objectives(row)):
            save(folder/'OBJECTIVE_REPRODUCTION_FAILURE.json', dict(decision_id=decision_id(row),
                saved=objectives(row).tolist(), recomputed=list(ind.objectives), maximum_absolute_error=error))
            raise ValueError('Own-model final objective reproduction is not exact')
    for kind, i in choices.items():
        row = front[i]
        values = {}
        for model in MODELS:
            ev.activate(expected if model=='EV' else optimization, model)
            ind, _, _ = ev.evaluate(row)
            values[model] = list(ind.objectives)
            cross.append(dict(**task, plan_type=kind, evaluated_as=model, decision_id=decision_id(row),
                               **dict(zip(OBJECTIVES, ind.objectives))))
        if any(w < c for w, c in zip(values['WORST'], values['CCP'])):
            raise ValueError('Worst objective smaller than CCP on identical optimization scenarios')
    csv_write(folder/'CROSS_EVALUATION.csv', cross)
    ev.activate(independent)
    last = time.monotonic()
    for i, row in enumerate(front):
        types = [k for k, j in choices.items() if j==i]
        ind, values, arrivals = ev.evaluate(row, arrivals=bool(types))
        summary = dict(**task, decision_id=decision_id(row), front_index=i, **summarize_values(ind, values))
        summaries.append(summary)
        for kind in types:
            path = folder/(kind+'_SCENARIOS.npz')
            with path.open('xb') as f:
                np.savez_compressed(f, objectives=values, batch_arrival_h=arrivals,
                    batch_ids=np.asarray([b.batch_id for b in ev.batches]),
                    batch_deadline_h=np.asarray([b.LT for b in ev.batches]), scenario_id=np.arange(5000))
            # Plain CSV exports as well as exact binary arrays.
            csv_write(folder/(kind+'_SCENARIOS.csv'), [dict(scenario_id=j, **dict(zip(OBJECTIVES, values[:,j]))) for j in range(5000)])
            csv_write(folder/(kind+'_BATCH_ARRIVALS.csv'),
                (dict(scenario_id=j, batch_id=b.batch_id, arrival_h=float(arrivals[k,j]),
                      deadline_h=b.LT, on_time=bool(arrivals[k,j] <= b.LT))
                 for k,b in enumerate(ev.batches) for j in range(5000)))
            routes.extend(dict(**task, plan_type=kind, decision_id=decision_id(row), **r) for r in ev.route_metrics(row))
        now = time.monotonic()
        if i==0 or i+1==len(front) or now-last>=20:
            print('[SECTION68 OOS]', task, f'{i+1}/{len(front)}', flush=True)
            last=now
    csv_write(folder/'FRONT_OOS_SUMMARY.csv', summaries)
    chosen = [dict(**summaries[i], plan_type=kind,
                   search_seconds=result['optimisation_seconds'], actual_evaluations=result['real_objective_evaluation_count'])
              for kind,i in choices.items()]
    csv_write(folder/'SELECTED_OOS_SUMMARY.csv', chosen)
    csv_write(folder/'ROUTE_METRICS.csv', routes)
    checks = dict(all_front_objectives_reproduced=True, maximum_absolute_error=worst_error,
        all_front_plans_evaluated=len(front), independent_realizations=5000,
        worst_not_less_than_ccp=True, batch_arrivals_match_makespan=True,
        reevaluation_seconds=time.monotonic()-started, exact_table5_reproduction='not applicable')
    if task['model']=='CCP':
        sources = [s for s in selected['exact_table5_reproduction_sources'] if s['seed']==task['seed']]
        for source in sources:
            def signature(rows):
                return sorted((decision_id(r), tuple(objectives(r))) for r in rows)
            if signature(front) != signature(read(source['front'])):
                raise ValueError('CCP front differs from same-instance/seed Table 5 Base')
        checks['exact_table5_reproduction'] = 'passed' if sources else 'not applicable'
    save(folder/'CHECKS.json', checks)
    return checks


def worker(args):
    root=args.out.resolve(); plan=read(root/'PLAN.json')
    verify(plan['files'])
    if canonical({k:v for k,v in plan.items() if k!='sha256'})!=plan['sha256']:
        raise ValueError('Plan changed')
    if socket.gethostname().split('.')[0]!='lnx-grid-41':
        raise ValueError('Search worker must run on lnx-grid-41')
    task=plan['tasks'][args.task_index]
    folder=args.attempt.resolve()
    if folder.parent != task_dir(root,task) or folder.exists():
        raise ValueError('Worker output must be a new attempt under its own task directory')
    folder.mkdir(parents=True); (folder/'search').mkdir()
    os.environ['MPLCONFIGDIR']=str(folder/'matplotlib_cache')
    try:
        s=next(s for s in plan['selected'] if s['instance']==task['instance'])
        original_task=dict(instance=task['instance'], seed=task['seed'], arm='BASE', phase='validation',
                           sample=False, model=None, bundle_sha256=None, normalization=plan['normalization'])
        result,options=search(Path(s['local_instance']),plan['settings'],original_task,folder/'search',
                              plan['sha256'],task['model'],s['nominal_options_sha256'])
        cfg=read(folder/'search/configuration.json')
        if result['enabled_components'] or result['model_bundle_sha256'] is not None or any(result['selection'][c]['calls'] for c in 'ICM'):
            raise ValueError('Base-only requirement violated')
        expected_size=1 if task['model']=='EV' else 100
        if cfg['training_size']!=expected_size or cfg['alpha']!=(1. if task['model']=='WORST' else .9):
            raise ValueError('Incorrect objective aggregation settings')
        pack='expected_input' if task['model']=='EV' else 'optimization_100'
        scenario,meta=load_scenarios(root/'scenarios'/task['configuration']/pack)
        if cfg['scenario_digest']!=meta['scenario_digest']:
            raise ValueError('Search did not use the saved scenario pack')
        check=reevaluate(root,plan,task,folder,result)
        verify(plan['files'])
        all_files={str(p):sha(p) for p in folder.rglob('*') if p.is_file() and 'matplotlib_cache' not in p.parts}
        save(folder/'COMPLETE.json',dict(completed=True, plan_sha256=plan['sha256'], task=task,
             actual_evaluations=result['real_objective_evaluation_count'],
             search_seconds=result['optimisation_seconds'], checks=check, files=all_files))
    except BaseException as exc:
        save(folder/'FAILURE.json',dict(error=repr(exc), traceback=traceback.format_exc()))
        raise


def run_tasks(root,plan,indices,workers):
    pending=[i for i in indices if completed_task(root,plan['tasks'][i],plan) is None]
    running=[]; last=0.
    try:
        while pending or running:
            while pending and len(running)<workers:
                i=pending.pop(0); task=plan['tasks'][i]
                attempt=task_dir(root,task)/('attempt_'+stamp())
                log=root/'logs'/(f'{i:02d}_'+stamp()+'.log')
                f=log.open('x')
                p=subprocess.Popen([sys.executable,'-B','-u',str(HERE/'launch.py'),'--worker',
                    '--out',str(root),'--task-index',str(i),'--attempt',str(attempt)],stdout=f,stderr=subprocess.STDOUT,
                    cwd=EXP,start_new_session=True)
                running.append((i,p,f,log)); print('[SECTION68] Start',task,'PID=',p.pid,'log=',log,flush=True)
            for item in list(running):
                i,p,f,log=item
                if p.poll() is not None:
                    f.close(); running.remove(item)
                    if p.returncode or completed_task(root,plan['tasks'][i],plan) is None:
                        raise RuntimeError(f'Task {i} stopped; inspect {log}')
            if time.monotonic()-last>=15:
                done=sum(completed_task(root,t,plan) is not None for t in plan['tasks'])
                status(root,completed=False,stage='pilot' if len(indices)==3 else 'formal',done=done,total=18,
                       running=[dict(index=i,pid=p.pid,log=str(log)) for i,p,_,log in running])
                last=time.monotonic()
            if pending or running:
                time.sleep(1)
    finally:
        for _,p,f,_ in running:
            # Only processes created by this invocation; no broad kill/pkill.
            if p.poll() is None:
                p.terminate(); p.wait(timeout=30)
            f.close()


def export(root,plan,indices,label):
    import csv
    output=root/(label+'_'+stamp()); output.mkdir()
    selected=[]; cross=[]; routes=[]; counts=[]
    for i in indices:
        task=plan['tasks'][i]; folder=completed_task(root,task,plan)
        if folder is None:
            raise ValueError('Missing completed task at export')
        def rows(name):
            with (folder/name).open(encoding='utf-8-sig',newline='') as f:
                return list(csv.DictReader(f))
        selected.extend(rows('SELECTED_OOS_SUMMARY.csv'))
        cross.extend(rows('CROSS_EVALUATION.csv')); routes.extend(rows('ROUTE_METRICS.csv'))
        m=read(folder/'COMPLETE.json')
        counts.append(dict(**task,actual_evaluations=m['actual_evaluations'],search_seconds=m['search_seconds'],
                           reevaluation_seconds=m['checks']['reevaluation_seconds']))
    csv_write(output/'PER_SEED.csv',selected); csv_write(output/'CROSS_EVALUATION.csv',cross)
    csv_write(output/'ROUTE_METRICS.csv',routes); csv_write(output/'RUNTIME.csv',counts)
    numeric=[k for k in selected[0] if k.endswith(('_mean','_p90','_max'))]+['punctuality_percent','search_seconds','actual_evaluations']
    groups=defaultdict(list)
    for r in selected:
        groups[(r['configuration'],r['instance'],r['model'],r['plan_type'])].append(r)
    means=[]
    for key,rows in groups.items():
        means.append(dict(zip(('configuration','instance','model','plan_type'),key)) )
        means[-1].update(seeds=len(rows),**{k:float(np.mean([float(r[k]) for r in rows])) for k in numeric})
    csv_write(output/'SEED_MEANS.csv',means)
    lookup={(r['instance'],r['seed'],r['plan_type'],r['model']):r for r in selected}
    paired=[]
    for r in selected:
        if r['model']=='EV':
            continue
        b=lookup[(r['instance'],r['seed'],r['plan_type'],'EV')]
        paired.append({k:r[k] for k in ('configuration','instance','seed','model','plan_type')})
        paired[-1].update(reference='EV',
            cost_mean_change_percent=100*(float(r['cost_mean'])/float(b['cost_mean'])-1),
            makespan_p90_change_percent=100*(float(r['makespan_p90'])/float(b['makespan_p90'])-1),
            punctuality_change_percentage_points=float(r['punctuality_percent'])-float(b['punctuality_percent']))
    csv_write(output/'PAIRED_VS_EV.csv',paired)
    pg=defaultdict(list)
    for r in paired:
        pg[(r['configuration'],r['instance'],r['model'],r['plan_type'])].append(r)
    pm=[]
    for key,rows in pg.items():
        pm.append(dict(zip(('configuration','instance','model','plan_type'),key)))
        pm[-1].update(reference='EV',seeds=len(rows),**{k:float(np.mean([r[k] for r in rows])) for k in
            ('cost_mean_change_percent','makespan_p90_change_percent','punctuality_change_percentage_points')})
    csv_write(output/'PAIRED_SEED_MEANS.csv',pm)
    rg=defaultdict(list)
    for r in routes:
        rg[tuple(r[k] for k in ('configuration','instance','model','plan_type','metric','item'))].append(float(r['value']))
    csv_write(output/'ROUTE_SEED_MEANS.csv',
        [dict(zip(('configuration','instance','model','plan_type','metric','item'),key),
              seeds=len(values),mean=float(np.mean(values))) for key,values in rg.items()])
    lines=['6.8 '+('试跑' if len(indices)==3 else '正式实验')+'实际结果',
           '方案选择仅使用各模型自己的优化目标；复评使用每实例共用的5000个独立场景。',
           'EV为平均输入评价，不是100个场景输出的平均；CCP为第90小，最坏情况为最大值。',
           '配置 | 实例 | 模型 | 方案 | 种子数 | 均值成本 | P90成本 | P90排放 | P90完工时间(h) | 准时率(%)']
    for r in means:
        lines.append(' | '.join(str(r[k]) for k in ('configuration','instance','model','plan_type','seeds',
                 'cost_mean','cost_p90','emission_p90','makespan_p90','punctuality_percent')))
    lines.append('实际搜索评价次数：'+json.dumps(counts,ensure_ascii=False))
    (output/'REPORT_CN.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    save(output/'MANIFEST.json',dict(plan_sha256=plan['sha256'],tasks=indices,
        files={str(p):sha(p) for p in output.iterdir() if p.is_file()}))
    return output, counts


def main(args):
    if args.worker:
        worker(args); return
    if not 1<=args.workers<=2:
        raise ValueError('At most two searches in parallel')
    root=args.out.resolve()
    if root.exists() and not args.resume:
        raise ValueError('Output exists. Use --resume to verify and reuse completed tasks, or a new --out.')
    if not root.exists():
        root.mkdir(parents=True)
    lock=(root/'RUN.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    os.environ['MPLCONFIGDIR']=str(root/'matplotlib_cache')
    try:
        plan=read(root/'PLAN.json') if (root/'PLAN.json').exists() else prepare(args,root)
        if plan['output']!=str(root) or canonical({k:v for k,v in plan.items() if k!='sha256'})!=plan['sha256']:
            raise ValueError('Frozen plan changed')
        verify(plan['files'])
        # Recheck hardware for a resumed job too.
        environment_review(Path(plan['preflight']))
        if args.stage=='prepare':
            status(root,completed=False,stage='ready_for_pilot',done=0,total=18); return
        run_tasks(root,plan,[0,1,2],args.workers)
        output,counts=export(root,plan,[0,1,2],'PILOT_RESULTS')
        exact=all(r['actual_evaluations']==5000 for r in counts)
        report=dict(all_three_outputs_checked=True,actual_evaluation_counts=counts,
                    exact_5000=exact,ready_for_formal=exact,results=str(output))
        save(root/('PILOT_CHECK_'+stamp()+'.json'),report)
        if not exact:
            status(root,completed=False,stage='stopped_after_pilot',done=3,total=18,
                   reason='Pilot contains nonexact 5000 count; report actual counts before formal runs',pilot=report); return
        if args.stage=='pilot':
            status(root,completed=False,stage='pilot_passed',done=3,total=18,pilot=report); return
        run_tasks(root,plan,list(range(18)),args.workers)
        output,counts=export(root,plan,list(range(18)),'RESULTS')
        verify(plan['files'])
        status(root,completed=True,stage='all',done=18,total=18,results=str(output),
               actual_evaluation_counts=counts)
    except BaseException as exc:
        save(root/('FAILURE_'+stamp()+'.json'),dict(error=repr(exc),traceback=traceback.format_exc()))
        status(root,completed=False,stage='stopped',error=str(exc))
        raise
    finally:
        fcntl.flock(lock,fcntl.LOCK_UN); lock.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--preflight',type=Path,default=DIAG/'section68_preflight_20261124T155523046075Z')
    p.add_argument('--out',type=Path,default=DIAG/'section68_ev_ccp_worst_runs_v1')
    p.add_argument('--stage',choices=('prepare','pilot','all'),default='pilot')
    p.add_argument('--workers',type=int,default=2)
    p.add_argument('--resume',action='store_true')
    p.add_argument('--worker',action='store_true')
    p.add_argument('--task-index',type=int)
    p.add_argument('--attempt',type=Path)
    main(p.parse_args())
