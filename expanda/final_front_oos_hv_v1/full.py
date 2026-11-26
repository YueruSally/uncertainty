#!/usr/bin/env python3
"""Authorized full final-front reevaluation. Never runs search or training."""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

sys.dont_write_bytecode = True
for env in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[env] = '1'

from inventory import (EXP, HERE, DIAG, ROOT_NAMES, inventory, read, save, require,
                       digest, canonical, retain, verify, csv_write, vector)
from run import load_runtime, order_quantiles, normalized_hv, RTOL, ATOL, HV_ATOL
from full_io import (freeze_json, load_scenarios, save_scenarios, evaluator,
                     project_scenarios, old_reports, prepare_instances)
from full_reports import report

PROTOCOL = 'final-front-oos-hv-full-v1'
TAGS = ('optimization_100', 'fresh_100', 'fresh_1000', 'fresh_5000')
CHUNK = 64


def status(root, stage, **extra):
    value = dict(protocol=PROTOCOL, stage=stage, pid=os.getpid(), completed=False,
        updated_utc=datetime.now(timezone.utc).isoformat(), new_searches=0, training_updates=0)
    value.update(extra)
    tmp = root / 'STATUS.writing'
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    tmp.replace(root / 'STATUS.json')
    print('[FINAL-OOS-FULL]', json.dumps(value, ensure_ascii=False), flush=True)


def activate(ev, scenario):
    ev.base.ACTIVE_SCENARIO_SET = scenario
    ev.base._PATH_SCENARIO_CACHE = {}


def load_pack(folder, packet, tag):
    """Only a hash-checked, complete pack can be reused."""
    import numpy as np
    path = folder / 'evaluations' / tag / 'COMPLETE.json'
    meta = read(path)
    require(meta['completed'] and meta['inputs_sha256'] == packet['sha256'] and meta['tag'] == tag,
            '已完成评价包身份不符')
    for ext, expected in meta['scenario_files'].items():
        require(digest(folder/'scenarios'/(tag+ext)) == expected, '完成包对应的场景文件发生变化')
    parts = []
    for c in meta['chunks']:
        file = path.parent / c['name']
        require(digest(file) == c['sha256'], '评价块哈希不符: ' + str(file))
        with np.load(file, allow_pickle=False) as a:
            parts.append({k: a[k].copy() for k in a.files})
    values = {k: np.concatenate([a[k] for a in parts], axis=0) for k in parts[0]}
    require(values['decision_ids'].tolist() == [r['exact_decision_id'] for r in packet['decisions']],
            '完成包的决策顺序不同')
    require(values['quantiles'].shape == (len(packet['decisions']), 3)
            and np.isfinite(values['quantiles']).all(), '完成包维度或目标值不符')
    return values


def scenario_for(np, ev, root, group, packet, tag, groups, alpha):
    folder = Path(group['folder'])
    local = folder / 'scenarios' / tag
    count = int(tag.rsplit('_', 1)[1])
    seed = int(packet['instance']['ccp100_seed'] if tag == 'optimization_100' else packet['seeds'][str(count)])
    if local.with_suffix('.json').exists():
        scenario = load_scenarios(np, ev, local.with_suffix('.json'), local.with_suffix('.npz'))
    elif 'pilot_scenarios' in packet:
        prefix = Path(packet['pilot_scenarios']) / tag
        scenario = load_scenarios(np, ev, prefix.with_suffix('.json'), prefix.with_suffix('.npz'))
        save_scenarios(np, ev, local, scenario)
    elif tag == 'optimization_100' and group['table'] == 'Table7':
        src = Path(packet['instance']['input_folder'])
        scenario = load_scenarios(np, ev, src/'SCENARIOS.json', src/'SCENARIOS.npz', legacy=True)
        save_scenarios(np, ev, local, scenario)
    else:
        target = group
        if group['table'] == 'Table7':
            target = next(g for g in groups if g['bounds_unit'] == group['bounds_unit'] and g['network'] == 'large')
        master_prefix = Path(target['folder']) / 'scenarios' / tag
        if target != group and master_prefix.with_suffix('.json').exists():
            master = load_scenarios(np, ev, master_prefix.with_suffix('.json'), master_prefix.with_suffix('.npz'))
        else:
            tp = read(Path(target['folder'])/'INPUTS.json')
            er = evaluator(ev, Path(target['folder'])/'instance.json', tp['source_config'], tp['workbook'], alpha)
            er.scenarios(count, seed)
            master = ev.base.ACTIVE_SCENARIO_SET
        if target != group:
            er = evaluator(ev, folder/'instance.json', packet['source_config'], packet['workbook'], alpha)
            scenario = project_scenarios(np, ev, er, master)
        else:
            scenario = master
        save_scenarios(np, ev, local, scenario)
    require(scenario.size == count and scenario.seed == seed and scenario.stochastic, '场景数、种子或随机模式不符')
    if tag == 'optimization_100':
        require(ev.scenario_digest(scenario) == packet['source_runs'][0]['scenario_digest'], '原优化场景未精确复现')
    if group['table'] == 'Table7' and group['network'] == 'small' and tag != 'optimization_100':
        large = next(g for g in groups if g['bounds_unit'] == group['bounds_unit'] and g['network']=='large')
        prefix = Path(large['folder'])/'scenarios'/tag
        master = load_scenarios(np, ev, prefix.with_suffix('.json'), prefix.with_suffix('.npz'))
        require(all(np.array_equal(v,master.travel_multiplier[k]) for k,v in scenario.travel_multiplier.items())
                and all(np.array_equal(v,master.border_delay_h[k]) for k,v in scenario.border_delay_h.items()),
                '重新读取时大小网络共有场景不一致')
    return scenario


def evaluate_pack(np, ev, root, group, packet, tag, groups, alpha, progress):
    folder = Path(group['folder']); dest = folder/'evaluations'/tag
    complete = dest/'COMPLETE.json'
    if complete.exists():
        load_pack(folder, packet, tag)
        return read(complete)
    scenario = scenario_for(np, ev, root, group, packet, tag, groups, alpha)
    er = evaluator(ev, folder/'instance.json', packet['source_config'], packet['workbook'], alpha)
    activate(ev, scenario)
    count, seed = scenario.size, scenario.seed
    identity = dict(inputs_sha256=packet['sha256'], tag=tag, scenario_digest=ev.scenario_digest(scenario),
                    size=count, seed=seed, alpha=alpha)
    dest.mkdir(parents=True, exist_ok=True)
    chunks = []
    decisions = packet['decisions']
    start, last = time.monotonic(), 0.
    physical = 0
    for offset in range(0, len(decisions), CHUNK):
        rows = decisions[offset:offset+CHUNK]
        name = f'{offset:07d}'
        path = dest/(name+'.npz'); mp = dest/(name+'.json')
        spec = dict(**identity, offset=offset, ids=[r['exact_decision_id'] for r in rows])
        if mp.exists():
            meta = read(mp)
            require(meta['identity'] == spec and digest(path) == meta['data_sha256'], '已保存评价块不同')
        else:
            n = len(rows)
            q, avg, maximum, coverage = [np.empty((n,3)) for _ in range(4)]
            feasible, violation = np.empty(n,dtype=bool), np.empty(n)
            chunk_start = time.monotonic()
            for i, row in enumerate(rows):
                samples, reported, feasible[i], violation[i] = er.evaluate(row)
                require(samples.shape == (3, count), '逐场景目标维度不符')
                q[i] = order_quantiles(np, samples, alpha)
                avg[i], maximum[i] = samples.mean(axis=1), samples.max(axis=1)
                coverage[i] = (samples <= np.asarray(vector(row))[:,None]).mean(axis=1)
                with (root/'EVALUATION_LEDGER.jsonl').open('a',encoding='utf-8') as f:
                    f.write(json.dumps(dict(table=group['table'], instance=group['instance'], tag=tag,
                        decision_id=row['exact_decision_id'], realizations=count, seed=seed,
                        timestamp_utc=datetime.now(timezone.utc).isoformat()))+'\n')
                physical += 1
                require(np.allclose(q[i], reported, rtol=RTOL, atol=ATOL), '评价器分位数与精确顺序统计量不符')
                if tag == 'optimization_100':
                    if not feasible[i] or not np.allclose(q[i], vector(row), rtol=RTOL, atol=ATOL):
                        freeze_json(dest/'REPRODUCTION_FAILURE.json',dict(exact_decision_id=row['exact_decision_id'],
                            original=vector(row), recomputed=q[i].tolist(), feasible=bool(feasible[i])))
                        raise ValueError('原场景目标未复现，停止')
                now = time.monotonic()
                if now-last >= 20:
                    progress(group,tag,offset+i+1,len(decisions),count,physical,now-start)
                    last = now
            temp = path.with_suffix('.writing')
            with temp.open('wb') as f:
                np.savez_compressed(f, quantiles=q, means=avg, maxima=maximum, feasible=feasible,
                    normalized_violation=violation, coverage=coverage,
                    decision_ids=np.asarray([r['exact_decision_id'] for r in rows]))
            temp.replace(path)
            meta = dict(identity=spec, data_sha256=digest(path), evaluator_seconds=time.monotonic()-chunk_start,
                        fixed_plan_evaluations=n, plan_realizations=n*count)
            freeze_json(mp, meta)
        chunks.append(dict(name=path.name,sha256=meta['data_sha256']))
    result = dict(completed=True, **identity, chunks=chunks, unique_plans=len(decisions),
                  fixed_plan_evaluations=len(decisions), plan_realizations=len(decisions)*count,
                  invocation_seconds=time.monotonic()-start,
                  scenario_files={ext:digest(folder/'scenarios'/(tag+ext)) for ext in ('.json','.npz')})
    freeze_json(complete, result)
    load_pack(folder,packet,tag)
    progress(group,tag,len(decisions),len(decisions),count,physical,time.monotonic()-start)
    return result


def reproduce_hvs(np, hv_fn, root, groups):
    rows = []
    for g in groups:
        folder=Path(g['folder']); p=read(folder/'INPUTS.json')
        arrays = load_pack(folder,p,'optimization_100')
        byid = {r['exact_decision_id']:i for i,r in enumerate(p['decisions'])}
        for run in p['source_runs']:
            ids = [byid[m['exact_decision_id']] for m in p['memberships']
                   if (m['method'],m['seed']) == (run['method'],run['seed'])]
            hv = normalized_hv(np,hv_fn,arrays['quantiles'][ids],run['old_bounds'])
            row = dict(table=g['table'],instance=g['instance'],method=run['method'],seed=run['seed'],
                       old_hv=run['old_hv'],reproduced_hv=hv,absolute_error=abs(hv-run['old_hv']))
            rows.append(row)
            require(row['absolute_error'] <= HV_ATOL, '旧报告HV未复现: '+str(row))
    freeze_json(root/'ORIGINAL_HV_REPRODUCTION.json',dict(passed=True,runs=len(rows),rows=rows))
    if not (root/'ORIGINAL_HV_REPRODUCTION.csv').exists():
        csv_write(root/'ORIGINAL_HV_REPRODUCTION.csv',rows)


def execute(root, manifest):
    groups, alpha = manifest['groups'], manifest['alpha']
    verify(manifest['files'])
    for g in groups:
        p=read(Path(g['folder'])/'INPUTS.json')
        require(p['sha256'] == g['inputs_sha256'] == canonical({k:v for k,v in p.items() if k!='sha256'}), '全量实例输入变化')
        require(read(Path(g['folder'])/'instance.json') == p['instance'], '本地实例副本变化')
    np, ev, hv_fn = load_runtime()
    # Original scenarios and HV for every run must pass before any fresh pack evaluation.
    jobs = [(g,'optimization_100') for g in groups]
    jobs += [(g,tag) for g in groups for tag in TAGS[1:]]
    total_realizations = sum(g['unique_decisions']*int(tag.rsplit('_',1)[1]) for g,tag in jobs)
    completed_weight = 0
    launch = time.monotonic(); physical_weight = 0
    def heartbeat(g,tag,done,total,count,physical,elapsed):
        speed = (physical_weight+physical*count)/(time.monotonic()-launch) if time.monotonic()>launch else 0
        remaining = max(0,total_realizations-completed_weight-done*count)
        status(root,'reevaluating_saved_final_fronts',table=g['table'],instance=g['instance'],scenario_group=tag,
            plans_done=done,plans_total=total,completed_packs=sum((Path(x['folder'])/'evaluations'/t/'COMPLETE.json').exists() for x,t in jobs),
            total_packs=len(jobs),pack_seconds=round(elapsed,2),
            estimated_remaining_compute_hours=round(remaining/speed/3600,3) if speed else None,
            eta_note='Measured throughput; excludes final reporting, input verification and changing server load.')
    for ji,(g,tag) in enumerate(jobs):
        if ji == len(groups):
            reproduce_hvs(np,hv_fn,root,groups)
            verify(manifest['files'])
            status(root,'all_original_objectives_and_hvs_reproduced',runs=810)
        p=read(Path(g['folder'])/'INPUTS.json')
        before=time.monotonic()
        complete=Path(g['folder'])/'evaluations'/tag/'COMPLETE.json'
        reused=complete.exists()
        evaluate_pack(np,ev,root,g,p,tag,groups,alpha,heartbeat)
        weight=g['unique_decisions']*int(tag.rsplit('_',1)[1])
        completed_weight+=weight
        if not reused:
            physical_weight+=weight
        print('[FINAL-OOS-FULL] pack complete',g['table'],g['instance'],tag,
              'reused='+str(reused),'seconds='+str(round(time.monotonic()-before,2)),flush=True)
    verify(manifest['files'])
    status(root,'reporting',completed_packs=len(jobs),total_packs=len(jobs))
    if (root/'RESULT_LOCATION.json').exists():
        location=read(root/'RESULT_LOCATION.json')
        destination=Path(location['directory'])
        verify(location['files'])
    else:
        destination=report(np,hv_fn,root,groups,load_pack,alpha)
        verify(manifest['files'])
        freeze_json(root/'RESULT_LOCATION.json',dict(directory=str(destination),
            files={str(f):digest(f) for f in destination.iterdir() if f.is_file()}))
    status(root,'complete',completed=True,runs=810,main_evaluation_scenarios=5000,
        contrast_evaluation_scenarios=1000,source_files_unchanged=True,results=str(destination),
        unique_instance_decisions=sum(g['unique_decisions'] for g in groups),
        logical_plan_realizations=total_realizations,physical_evaluations_ledger=str(root/'EVALUATION_LEDGER.jsonl'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pilot',type=Path,help='Successful S0 pilot; exact frozen scenario arrays are reused for its instance.')
    parser.add_argument('--out',type=Path,help='New output directory only.')
    parser.add_argument('--resume',type=Path,help='Resume this full reevaluation only, reusing checked complete chunks.')
    parser.add_argument('--alpha',type=float,choices=(.8,.9),default=.9)
    for table,name in ROOT_NAMES.items():
        parser.add_argument('--'+table.lower(),type=Path,default=DIAG/name)
    args=parser.parse_args()
    require(not(args.resume and args.out),'--resume和--out不能并用')
    roots={t:getattr(args,t.lower()).resolve() for t in ROOT_NAMES}
    if args.resume:
        root=args.resume.resolve()
        require(root.is_dir() and (root/'PLAN.json').is_file(),'没有可续算的全量计划')
    else:
        require(args.pilot is not None,'必须指定已通过的S0试评目录')
        root=(args.out or DIAG/('ccp_final_front_oos_hv_full_v1_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))).resolve()
        source_paths=list(roots.values())+[args.pilot.resolve()]
        require(not root.exists() and not any(root==s or root in s.parents or s in root.parents for s in source_paths),
                '输出必须为不与来源重叠的新目录')
        root.mkdir(parents=True)
    print('[FINAL-OOS-FULL] output:',root,flush=True)
    lock=(root/'RUN.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    try:
        if args.resume:
            manifest=read(root/'PLAN.json')
            require(manifest['protocol']==PROTOCOL and manifest['alpha']==args.alpha
                    and canonical({k:v for k,v in manifest.items() if k!='sha256'})==manifest['sha256'],'续算计划身份不符')
            verify(manifest['entry_files'])
            if read(root/'STATUS.json').get('completed'):
                verify(manifest['files'])
                verify(read(root/'RESULT_LOCATION.json')['files'])
                print('该全量复评已经完成：',read(root/'RESULT_LOCATION.json')['directory']);return
        else:
            commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=EXP,text=True).strip()
            entry_files={str(p):digest(p) for p in sorted(HERE.glob('*.py'))}
            save(root/'CODE_VERSION.json',dict(commit=commit,entry_files=entry_files,python=sys.version,
                executable=sys.executable,tracked_worktree_status=subprocess.check_output(
                    ['git','status','--porcelain','--untracked-files=no'],cwd=EXP,text=True)))
            status(root,'inventory')
            inv,plans,entries,files=inventory(roots,root,args.alpha)
            old_reports(roots,plans,inv,entries,files)
            groups=prepare_instances(root,inv,entries,files,args.pilot.resolve(),args.alpha)
            # Large first guarantees its frozen draw arrays exist before small projection.
            groups.sort(key=lambda g:(g['table'],g['bounds_unit'],g['network']=='small',g['instance']))
            manifest=dict(protocol=PROTOCOL,alpha=args.alpha,groups=groups,files=files,entry_files=entry_files,
                commit=commit,pilot=str(args.pilot.resolve()),searches=0,training_updates=0,
                sampling='Separate fixed seeds for fresh100/1000/5000; nonnested. Shared by methods and search seeds; paired networks coupled.',
                expected_runs=810,main_evaluation_scenarios=5000,contrast_evaluation_scenarios=1000)
            manifest['sha256']=canonical(manifest)
            save(root/'PLAN.json',manifest)
            verify(files)
        execute(root,manifest)
    except BaseException as exc:
        status(root,'stopped',error=f'{type(exc).__name__}: {exc}',resume_command=f'python3 -B -u final_front_oos_hv_v1/full.py --resume {root}')
        raise
    finally:
        lock.close()


if __name__=='__main__':
    main()
