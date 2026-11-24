#!/usr/bin/env python3
"""Read saved Section 6.8 inputs and timings; no searches or plan evaluations."""
import argparse
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
import sys

sys.dont_write_bytecode = True
import launch as run
import numpy as np


def check_pack(stem, expected_size, stochastic):
    pack, meta = run.load_scenarios(stem)
    if pack.size != expected_size or pack.stochastic != stochastic:
        raise ValueError('Wrong saved scenario type: '+str(stem))
    for key, arr in pack.travel_multiplier.items():
        if arr.shape != (expected_size,) or (not stochastic and not np.all(arr == 1.)):
            raise ValueError('Unexpected transport inputs: '+str(key))
    for key, arr in pack.border_delay_h.items():
        if arr.shape != (expected_size,) or (not stochastic and not np.all(arr == pack.border_event_mean_h[key])):
            raise ValueError('Unexpected border inputs: '+str(key))
    return meta


def audit(root):
    root = root.resolve()
    plan = run.read(root/'PLAN.json')
    if run.canonical({k: v for k, v in plan.items() if k != 'sha256'}) != plan['sha256']:
        raise ValueError('Frozen plan changed')
    if plan['protocol'] != 'section68-ev-ccp-worst-v1' or len(plan['tasks']) != 18:
        raise ValueError('Expected the 18-run Section 6.8 experiment')
    run.verify(plan['files'])
    source_files = dict(plan['files'])
    packs, rows = {}, []
    for task in plan['tasks']:
        folder = run.completed_task(root, task, plan)
        if folder is None:
            raise ValueError('Incomplete task: '+str(task))
        metadata = run.read(folder/'COMPLETE.json')
        config = run.read(folder/'search/configuration.json')
        result = run.read(folder/'search/RUN_COMPLETE.json')
        source_files[str(folder/'COMPLETE.json')] = run.sha(folder/'COMPLETE.json')
        is_ev = task['model'] == 'EV'
        size = 1 if is_ev else 100
        mode = 'deterministic' if is_ev else 'ccp'
        tag = 'expected_input' if is_ev else 'optimization_100'
        stem = root/'scenarios'/task['configuration']/tag
        if str(stem) not in packs:
            packs[str(stem)] = check_pack(stem, size, not is_ev)
        pack = packs[str(stem)]
        selected = next(s for s in plan['selected'] if s['instance'] == task['instance'])
        expected_seed = 0 if is_ev else selected['optimization_seed']
        if (config['evaluation_mode'] != mode or config['training_size'] != size
                or config['uncertainty']['scenario_count'] != size
                or config['scenario_digest'] != pack['scenario_digest']
                or config['seeds']['training'] != expected_seed
                or pack['seed'] != expected_seed
                or config['seeds']['algorithm'] != task['seed']
                or config['instance_id'] != task['instance']
                or config['instance_digest'] != selected['instance_digest']
                or pack['identity']['instance_digest'] != selected['instance_digest']
                or config['alpha'] != (1. if task['model'] == 'WORST' else .9)):
            raise ValueError('Search configuration or scenario pack mismatch: '+str(task))
        n = result['real_objective_evaluation_count']
        if n != metadata['actual_evaluations'] or result['optimisation_seconds'] != metadata['search_seconds']:
            raise ValueError('Completion record differs from solver record')
        rows.append(dict(**task, evaluation_mode=mode, realizations_per_plan=size,
            stochastic=pack['stochastic'], actual_plan_evaluations=n,
            plan_realization_pairs=n*size,
            objective_evaluation_seconds=result['objective_evaluation_seconds'],
            other_optimisation_seconds=result['other_optimisation_seconds'],
            search_seconds=result['optimisation_seconds'],
            independent_reevaluation_seconds=metadata['checks']['reevaluation_seconds'],
            scenario_digest=config['scenario_digest']))
    groups = defaultdict(list)
    for row in rows:
        groups[(row['configuration'], row['model'])].append(row)
    if set(groups) != {(c, m) for c in ('S6','S8') for m in run.MODELS} or any(len(v) != 3 for v in groups.values()):
        raise ValueError('Expected S6/S8 x three models x three seeds')
    summary = []
    for (cfg, model), group in groups.items():
        row = dict(configuration=cfg, model=model, searches=3,
                   realizations_per_plan=group[0]['realizations_per_plan'])
        for key in ('objective_evaluation_seconds', 'other_optimisation_seconds',
                    'search_seconds', 'independent_reevaluation_seconds'):
            row[key+'_mean'] = mean(x[key] for x in group)
            row[key+'_sd'] = stdev(x[key] for x in group)
        summary.append(row)
    # Timestamped derived outputs only; all source records remain untouched.
    out = root/('EVALUATION_MODE_AUDIT_'+run.stamp())
    out.mkdir()
    run.csv_write(out/'PER_RUN.csv', rows)
    run.csv_write(out/'TIMING_SUMMARY.csv', summary)
    run.save(out/'AUDIT.json', dict(passed=True, verified_searches=len(rows),
        ev_searches=sum(r['model']=='EV' for r in rows),
        ev_definition='One constructed mean-input scenario, deterministic aggregation of element zero',
        mean_of_100_outputs_used_by_ev=False, source_plan_sha256=plan['sha256'],
        source_files=source_files, script_sha256=run.sha(__file__),
        scenario_packs=packs, new_searches=0, new_plan_evaluations=0,
        timing_qualification='Recorded timings only; no profiling or causal timing attribution. Other time is the existing residual.',
        pair_count_qualification='Plan-realization pairs describe workload, not uncached path simulation calls.'))
    print('EV六次：配置 | 种子 | 模式 | 每方案实现数 | 实际方案评价次数')
    for r in rows:
        if r['model']=='EV':
            print(f"{r['configuration']} | {r['seed']} | {r['evaluation_mode']} | {r['realizations_per_plan']} | {r['actual_plan_evaluations']}")
    print('\n配置 | 模型 | 每方案实现数 | 目标评价秒 | 其他搜索秒 | 总搜索秒（均值±样本标准差）')
    for r in summary:
        fields=[r['configuration'],r['model'],str(r['realizations_per_plan'])]
        fields += [f"{r[k+'_mean']:.2f} ± {r[k+'_sd']:.2f}" for k in
                   ('objective_evaluation_seconds','other_optimisation_seconds','search_seconds')]
        print(' | '.join(fields))
    print('PASS：EV为单次平均输入评价；CCP/最坏情况为100个优化场景。')
    print('新增搜索0，新增方案评价0；报告：',out)
    return out


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=run.DIAG/'section68_ev_ccp_worst_runs_v3')
    audit(p.parse_args().root)
