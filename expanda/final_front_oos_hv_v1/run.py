#!/usr/bin/env python3
"""Inventory all final fronts, reproduce S0, then stop after an independent-scenario pilot.

This entry point contains no search/training/full-study execution mode.
"""
import argparse
import csv
from datetime import datetime, timezone
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

sys.dont_write_bytecode = True
for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '1'

from inventory import (HERE, EXP, DIAG, ENGINE, DATA, OBJECTIVES, ROOT_NAMES, PPO,
                       canonical, digest, read, save, csv_write, retain, verify,
                       require, inventory, exact_id, vector)

PROTOCOL = 'final-front-oos-hv-pilot-v1'
SIZES = (100, 1000, 5000)
REF = (1.1, 1.1, 1.1)
DIVISOR = 1.331
RTOL, ATOL = 1e-10, 1e-8
HV_ATOL = 1e-10


def status(out, stage, **extra):
    value = dict(protocol=PROTOCOL, stage=stage, pid=os.getpid(),
                 updated_utc=datetime.now(timezone.utc).isoformat(), new_searches=0,
                 training_updates=0, **extra)
    tmp = out / 'STATUS.writing'
    with tmp.open('w', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')
    tmp.replace(out / 'STATUS.json')  # Only this newly-created output's status is updated.
    print('[FINAL-OOS]', json.dumps(value, ensure_ascii=False), flush=True)


def load_runtime():
    """Reuse the existing exact-decision evaluator; import no search controllers/policies."""
    import numpy as np
    sys.path.insert(0, str(EXP / 'fixed_q90_s0_s11_v1'))
    spec = importlib.util.spec_from_file_location('final_oos_existing_evaluator',
                                                 EXP / 'fixed_q90_s0_s11_v1/evaluate.py')
    ev = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ev)
    require(Path(ev.base.__file__).resolve() == ENGINE / 'baseline_uncertainty.py', '评价器路径不符')
    from transformer_ppo.archive import exact_hv_3d
    return np, ev, exact_hv_3d


def order_quantiles(np, samples, alpha):
    require(samples.ndim == 2 and samples.shape[0] == 3 and samples.shape[1] > 0
            and np.isfinite(samples).all(), '逐场景目标数组必须为有限的 3×n 数组')
    kth = math.ceil(alpha * samples.shape[1]) - 1
    return np.partition(samples, kth, axis=1)[:, kth]


def normalized_hv(np, hv_function, points, bounds):
    pts = np.asarray(points, dtype=float)
    lo, hi = np.asarray(bounds['minimum']), np.asarray(bounds['maximum'])
    require(pts.ndim == 2 and pts.shape[1] == 3 and len(pts) > 0 and np.isfinite(pts).all(), '无效目标点')
    span = np.where(hi > lo, hi - lo, 1.)
    scaled = (pts - lo) / span
    # Allow numerical roundoff only; never silently clip or remove a solution.
    require(np.all(scaled >= -1e-10) and np.all(scaled <= 1 + 1e-10), '目标点超出共同边界')
    return float(hv_function(scaled.tolist(), REF) / DIVISOR)


def save_scenario_pack(np, ev, folder, tag, identity):
    scenario = ev.base.ACTIVE_SCENARIO_SET
    travel_keys, border_keys = sorted(scenario.travel_multiplier), sorted(scenario.border_delay_h)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / (tag + '.npz')
    travel = np.asarray([scenario.travel_multiplier[k] for k in travel_keys]).reshape(-1, scenario.size)
    border = np.asarray([scenario.border_delay_h[k] for k in border_keys]).reshape(-1, scenario.size)
    with path.open('xb') as f:
        np.savez_compressed(f, travel=travel, border=border)
    meta = dict(format=PROTOCOL + '-network-draws', identity=identity, size=scenario.size,
                seed=int(scenario.seed), stochastic=scenario.stochastic, travel_keys=travel_keys,
                border_keys=border_keys, arc_border_event=[[k, v] for k, v in sorted(scenario.arc_border_event.items())],
                border_event_mean_h=[[k, v] for k, v in sorted(scenario.border_event_mean_h.items())],
                scenario_digest=ev.scenario_digest(scenario), data_sha256=digest(path))
    # Check the on-disk arrays, not just that the write returned.
    with np.load(path, allow_pickle=False) as pack:
        require(np.array_equal(travel, pack['travel']) and np.array_equal(border, pack['border']),
                '保存的网络场景数组不一致')
    save(folder / (tag + '.json'), meta)
    return meta


def pilot_inputs(out, roots, inv, plans, entries, files, alpha):
    s0 = sorted(e for e in entries['Table5'] if entries['Table5'][e]['configuration'] == 'S0')[0]
    entry = entries['Table5'][s0]
    seeds = sorted(plans['Table5']['settings']['screen_seeds'])
    chosen = [r for r in inv if r['table'] == 'Table5' and r['instance'] == s0
              and r['method'] in ('BASE', PPO)]
    require({(r['method'], r['seed']) for r in chosen} == {(m, s) for m in ('BASE', PPO) for s in seeds}
            and len(chosen) == 6, 'S0 试评必须为 Base/PPO 的三个种子，共六个前沿')
    configs = [read(Path(r['run_path']).parent / 'configuration.json') for r in chosen]
    config = configs[0]
    require(all(c['uncertainty'] == config['uncertainty'] for c in configs), 'S0 场景参数不同')
    require(entry['source_data_sha256'] == digest(DATA), 'S0 使用的网络与评价器不符')
    solutions, membership = {}, []
    for r in chosen:
        for row in read(r['front_path']):
            key = exact_id(row)
            if key in solutions:
                require(all(math.isclose(a, b, rel_tol=RTOL, abs_tol=ATOL)
                            for a, b in zip(vector(solutions[key]), vector(row))), '相同决策的原目标值不符')
            else:
                solutions[key] = dict(row, exact_decision_id=key)
            membership.append(dict(method=r['method'], seed=r['seed'], source_solution_id=row['source_solution_id'],
                                   exact_decision_id=key, source_front=r['front_path']))
    # Separate seeds/streams for each size; never take prefixes of a larger scenario set.
    new_seeds = {str(n): 381000000 + i for i, n in enumerate(SIZES)}
    known = {int(e['ccp100_seed']) for table_entries in entries.values() for e in table_entries.values()}
    require(not known.intersection(new_seeds.values()), '评估种子与优化种子重叠')
    require(len(set(new_seeds.values())) == 3, '三组评估种子必须不同')
    old_bounds_path = roots['Table5'] / 'results/INSTANCE_BOUNDS.json'
    old_pairs_path = roots['Table5'] / 'results/PER_PAIR.csv'
    old_report_path = roots['Table5'] / 'results/REPORT.json'
    for p in (old_bounds_path, old_pairs_path, old_report_path):
        retain(p, files)
    report = read(old_report_path)
    require(report['completed'] and report['plan_sha256'] == plans['Table5']['sha256']
            and report['hv_divisor'] == DIVISOR and report['hv_already_divided']
            and report['hv_reference'] == list(REF), 'Table5 报告的版本、HV单位或参考点不符')
    with old_pairs_path.open(encoding='utf-8-sig', newline='') as f:
        pairs = [r for r in csv.DictReader(f) if r['instance'] == s0 and r['evaluation'] == '5000'
                 and r['reference_method'] == 'BASE' and r['method'] in ('BASE', PPO)]
    require(len(pairs) == 6 and {(r['method'], int(r['seed'])) for r in pairs}
            == {(r['method'], r['seed']) for r in chosen}, '原 Table5 配对HV不齐全')
    expected = [dict(method=r['method'], seed=int(r['seed']), hv=float(r['own'])) for r in pairs]
    frozen = dict(protocol=PROTOCOL, alpha=alpha, instance=entry, source_config=config,
                  source_runs=chosen, decisions=[solutions[k] for k in sorted(solutions)],
                  memberships=membership, old_bounds=read(old_bounds_path)[s0], table5_expected_hv=expected,
                  new_seeds=new_seeds, optimization_seed=entry['ccp100_seed'], sizes=SIZES,
                  scenario_digest=chosen[0]['scenario_digest'], search_budget=5000,
                  pilot_normalization='Union of final Base/PPO fronts, all three seeds and fresh 100/1000/5000 groups only. Preliminary; full-study bounds will be recomputed after approval.',
                  reproduction_normalization='Unchanged original Table5 bounds and divisor; excluded from pilot fresh bounds.')
    frozen['sha256'] = canonical(frozen)
    save(out / 'PILOT_INPUTS.json', frozen)
    save(out / 'instance.json', entry)
    csv_write(out / 'SOLUTION_MEMBERSHIP.csv', membership)
    save(out / 'PILOT_SOURCE_FILES.json', files)
    return frozen


def evaluate_group(np, ev, evaluator, out, inputs, tag, count, seed):
    original = tag == 'optimization_100'
    verify(read(out / 'PILOT_SOURCE_FILES.json'))
    scenario_sha = evaluator.scenarios(count, seed)
    if original:
        require(scenario_sha == inputs['scenario_digest'], '原100个优化场景未精确复现，停止')
    meta = save_scenario_pack(np, ev, out / 'scenarios', tag,
                             dict(instance=inputs['instance']['instance_id'], role=tag,
                                  alpha=inputs['alpha'], inputs_sha256=inputs['sha256']))
    total = len(inputs['decisions'])
    quantiles = np.empty((total, 3), dtype=float)
    means, maxima = np.empty_like(quantiles), np.empty_like(quantiles)
    feasible = np.empty(total, dtype=bool)
    violation = np.empty(total, dtype=float)
    start, last = time.monotonic(), 0.
    for i, row in enumerate(inputs['decisions']):
        samples, reported, feasible[i], violation[i] = evaluator.evaluate(row)
        require(samples.shape == (3, count), '实际场景数与要求不符')
        quantiles[i] = order_quantiles(np, samples, inputs['alpha'])
        means[i], maxima[i] = samples.mean(axis=1), samples.max(axis=1)
        with (out / 'EVALUATION_LEDGER.jsonl').open('a', encoding='utf-8') as f:
            f.write(json.dumps(dict(tag=tag, decision=row['exact_decision_id'], realizations=count,
                                   scenario_seed=seed, completed_utc=datetime.now(timezone.utc).isoformat())) + '\n')
        if not np.allclose(reported, quantiles[i], rtol=RTOL, atol=ATOL):
            save(out / 'EVALUATOR_QUANTILE_MISMATCH.json', dict(tag=tag, decision=row['exact_decision_id'],
                 evaluator=reported.tolist(), order_statistic=quantiles[i].tolist()))
            raise ValueError('评价器输出与第 ceil(alpha*n) 小的值不一致')
        if original and (not feasible[i] or not np.allclose(quantiles[i], vector(row), rtol=RTOL, atol=ATOL)):
            save(out / 'ORIGINAL_OBJECTIVE_MISMATCH.json', dict(decision=row['exact_decision_id'],
                 original=vector(row), recomputed=quantiles[i].tolist(), feasible=bool(feasible[i])))
            raise ValueError('原优化场景目标值未复现，停止；未生成新场景')
        now = time.monotonic()
        if i == 0 or i + 1 == total or now - last >= 20:
            status(out, 'evaluating_fixed_final_solutions', completed=False, group=tag, plans_done=i+1,
                   plans_total=total, group_seconds=round(now-start, 2), search_budget=5000)
            last = now
    folder = out / 'evaluations'
    folder.mkdir(exist_ok=True)
    path = folder / (tag + '.npz')
    with path.open('xb') as f:
        np.savez_compressed(f, quantiles=quantiles, means=means, maxima=maxima,
                            feasible=feasible, normalized_violation=violation,
                            decision_ids=np.array([r['exact_decision_id'] for r in inputs['decisions']]))
    qrows = [dict(exact_decision_id=r['exact_decision_id'], evaluation_group=tag,
                  realizations=count, quantile_alpha=inputs['alpha'],
                  cost_quantile=float(quantiles[i, 0]), emission_quantile=float(quantiles[i, 1]),
                  makespan_quantile=float(quantiles[i, 2]), feasible=bool(feasible[i]))
             for i, r in enumerate(inputs['decisions'])]
    csv_write(folder / (tag + '.csv'), qrows)
    meta.update(plans=total, feasible_plans=int(feasible.sum()), evaluator_seconds=time.monotonic()-start,
                objective_pack_sha256=digest(path), fixed_plan_evaluations=total, plan_realizations=total*count,
                all_final_solutions_retained=True)
    if original:
        expected = np.array([vector(row) for row in inputs['decisions']])
        meta.update(reproduced=True, objective_rtol=RTOL, objective_atol=ATOL,
                    maximum_absolute_error_by_objective=np.abs(quantiles-expected).max(axis=0).tolist(),
                    maximum_relative_error_by_objective=(np.abs(quantiles-expected)/np.maximum(np.abs(expected), 1e-300)).max(axis=0).tolist())
    save(folder / (tag + '.json'), meta)
    verify(read(out / 'PILOT_SOURCE_FILES.json'))
    return quantiles, meta


def run_hvs(np, hv_fn, inputs, quantiles, bounds):
    by_id = {r['exact_decision_id']: i for i, r in enumerate(inputs['decisions'])}
    rows = []
    for source in sorted(inputs['source_runs'], key=lambda r: (r['method'], r['seed'])):
        ids = [by_id[r['exact_decision_id']] for r in inputs['memberships']
               if r['method'] == source['method'] and r['seed'] == source['seed']]
        require(len(ids) == source['final_solutions'], '重新评价的方案数与保存的前沿不符')
        rows.append(dict(configuration='S0', instance=source['instance'], method=source['method'],
                         seed=source['seed'], search_budget=5000, actual_search_evaluations=source['actual_evaluations'],
                         final_solutions=len(ids), hv=normalized_hv(np, hv_fn, quantiles[ids], bounds)))
    return rows


def pilot(out, roots, inv, plans, entries, files, alpha):
    inputs = pilot_inputs(out, roots, inv, plans, entries, files, alpha)
    np, ev, hv_fn = load_runtime()
    evaluator = ev.Evaluator(out / 'instance.json', inputs['source_config'])
    ev.base.CONFIDENCE_COST = ev.base.CONFIDENCE_EMISSION = ev.base.CONFIDENCE_TIME = alpha
    original, original_meta = evaluate_group(np, ev, evaluator, out, inputs, 'optimization_100', 100, inputs['optimization_seed'])
    hv_rows = run_hvs(np, hv_fn, inputs, original, inputs['old_bounds'])
    expected = {(r['method'], r['seed']): r['hv'] for r in inputs['table5_expected_hv']}
    for row in hv_rows:
        row.update(table5_hv=expected[row['method'], row['seed']])
        row['absolute_hv_error'] = abs(row['hv'] - row['table5_hv'])
        row['passed'] = row['absolute_hv_error'] <= HV_ATOL
    csv_write(out / 'ORIGINAL_HV_REPRODUCTION.csv', hv_rows)
    require(all(r['passed'] for r in hv_rows), '原100场景HV与Table5不一致，停止；未生成新场景')
    save(out / 'REPRODUCTION_PASS.json', dict(passed=True, instance=inputs['instance']['instance_id'],
         optimization_scenarios_exact_digest=True, original_objectives=original_meta,
         maximum_hv_error=max(r['absolute_hv_error'] for r in hv_rows), hv_absolute_tolerance=HV_ATOL,
         bounds='original Table5', divisor=DIVISOR))
    status(out, 'original_reproduction_passed', completed=False, source_runs=6,
           unique_plans=len(inputs['decisions']), new_searches_started=False)
    fresh, metas = {}, []
    for n in SIZES:
        fresh[n], meta = evaluate_group(np, ev, evaluator, out, inputs, f'fresh_{n}', n, inputs['new_seeds'][str(n)])
        metas.append(meta)
    all_points = np.concatenate([fresh[n] for n in SIZES])
    bounds = dict(minimum=all_points.min(axis=0).tolist(), maximum=all_points.max(axis=0).tolist(),
                  reference=list(REF), divisor=DIVISOR, methods=['BASE', PPO], search_budget=5000,
                  scenario_sizes=list(SIZES), scope=inputs['pilot_normalization'],
                  zero_span_rule='A zero objective range uses denominator 1, as in Table5.')
    save(out / 'PILOT_INSTANCE_BOUNDS.json', {inputs['instance']['instance_id']: bounds})
    per_run = []
    for n in SIZES:
        per_run.extend(dict(row, evaluation_scenarios=n, quantile_alpha=alpha)
                       for row in run_hvs(np, hv_fn, inputs, fresh[n], bounds))
    base = {(r['seed'], r['evaluation_scenarios']): r['hv'] for r in per_run if r['method'] == 'BASE'}
    for r in per_run:
        b = base[r['seed'], r['evaluation_scenarios']]
        require(b > 0, 'Base HV不为正，无法计算相对提升')
        r.update(base_hv=b, relative_base_percent=100*(r['hv']-b)/b)
    csv_write(out / 'PER_RUN_HV.csv', per_run)
    summaries, differences = [], []
    for method in ('BASE', PPO):
        lookup = {(r['seed'], r['evaluation_scenarios']): r['hv'] for r in per_run if r['method'] == method}
        for n in SIZES:
            part = [r for r in per_run if r['method'] == method and r['evaluation_scenarios'] == n]
            summaries.append(dict(configuration='S0', instance=inputs['instance']['instance_id'], method=method,
                 search_budget=5000, evaluation_scenarios=n, seeds=len(part),
                 mean_hv=float(np.mean([r['hv'] for r in part])),
                 mean_paired_relative_base_percent=float(np.mean([r['relative_base_percent'] for r in part]))))
        for n in (1000, 5000):
            for seed in sorted(plans['Table5']['settings']['screen_seeds']):
                a, b = lookup[seed, 100], lookup[seed, n]
                require(b > 0, '场景组参考HV不为正')
                differences.append(dict(method=method, seed=seed, comparison=f'100_vs_{n}',
                    hv_100=a, hv_reference=b, signed_relative_difference_percent=100*(a-b)/b))
    csv_write(out / 'PILOT_SUMMARY.csv', summaries)
    csv_write(out / 'SCENARIO_DIFFERENCES_PER_RUN.csv', differences)
    diff_summary = []
    for method in ('BASE', PPO):
        for comparison in ('100_vs_1000', '100_vs_5000'):
            part = [r for r in differences if r['method'] == method and r['comparison'] == comparison]
            a, b = np.mean([r['hv_100'] for r in part]), np.mean([r['hv_reference'] for r in part])
            diff_summary.append(dict(method=method, comparison=comparison, seeds=len(part),
                 mean_paired_signed_difference_percent=float(np.mean([r['signed_relative_difference_percent'] for r in part])),
                 mean_paired_absolute_difference_percent=float(np.mean([abs(r['signed_relative_difference_percent']) for r in part])),
                 difference_of_means_percent=float(100*(a-b)/b)))
    csv_write(out / 'SCENARIO_DIFFERENCES.csv', diff_summary)
    verify(files)
    text = [f"S0 实例：{inputs['instance']['instance_id']}；Base/PPO，各三个搜索种子。",
            '使用全部已保存的最终解；搜索上限5000，实际评价次数见PER_RUN_HV.csv。',
            '100/1000/5000列均指独立复评场景数，不是搜索检查点。HV已除以1.331。',
            '原优化100场景：场景摘要、目标值及Table5原边界下HV核对通过。',
            '本试评的新边界仅由S0的Base/PPO最终解、全部种子、三组新场景确定。',
            '全量阶段纳入其余方法后必须重新确定边界，不能将本试评HV拼入全量表。',
            '方法 | 复评场景数 | 平均HV | 相对Base平均配对变化(%)']
    text += [f"{r['method']} | {r['evaluation_scenarios']} | {r['mean_hv']:.9f} | {r['mean_paired_relative_base_percent']:+.6f}"
             for r in summaries]
    text += ['方法 | 场景比较 | 平均配对相对差(%) | 平均绝对相对差(%)',
             '有符号差定义：100*(HV_100-HV_n)/HV_n。']
    text += [f"{r['method']} | {r['comparison']} | {r['mean_paired_signed_difference_percent']:+.6f} | {r['mean_paired_absolute_difference_percent']:.6f}"
             for r in diff_summary]
    text += ['试评已停止，等待确认；没有运行任何新搜索，也未进行全量重新评价。']
    (out / 'REPORT.txt').write_text('\n'.join(text) + '\n', encoding='utf-8')
    print('\n'.join(text), flush=True)
    status(out, 'pilot_complete_awaiting_confirmation', completed=True, full_study_completed=False,
           instance=inputs['instance']['instance_id'], inventory_runs=len(inv),
           source_runs=6, unique_final_decisions=len(inputs['decisions']),
           fixed_plan_evaluations=len(inputs['decisions'])*4,
           plan_realizations=len(inputs['decisions'])*6200,
           evaluator_seconds=sum(m['evaluator_seconds'] for m in [original_meta]+metas),
           source_files_unchanged=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for table, name in ROOT_NAMES.items():
        parser.add_argument('--' + table.lower(), type=Path, default=DIAG / name)
    parser.add_argument('--out', type=Path, help='Must not already exist; default is a new timestamped directory.')
    parser.add_argument('--inventory-only', action='store_true')
    parser.add_argument('--alpha', type=float, choices=(.8, .9), default=.9,
                        help='Must match all saved searches. Does not relabel alpha=0.9 searches as alpha=0.8.')
    args = parser.parse_args()
    roots = {t: getattr(args, t.lower()).resolve() for t in ROOT_NAMES}
    out = (args.out or DIAG / ('ccp_final_front_oos_hv_pilot_v1_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))).resolve()
    require(not out.exists(), '输出目录已经存在；请用新目录，不覆盖或续写旧结果')
    require(not any(r == out or r in out.parents or out in r.parents for r in roots.values()), '输出目录不能与来源重叠')
    out.mkdir(parents=True)
    print('[FINAL-OOS] output:', out, flush=True)
    try:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=EXP, text=True).strip()
        tracked = subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=EXP, text=True)
        save(out / 'CODE_VERSION.json', dict(commit=commit, tracked_worktree_status=tracked,
             entry_files={str(p): digest(p) for p in sorted(HERE.glob('*.py'))},
             python=sys.version, executable=sys.executable, protocol=PROTOCOL))
        status(out, 'inventory', completed=False)
        inv, plans, entries, files = inventory(roots, out, alpha=args.alpha)
        if args.inventory_only:
            status(out, 'inventory_complete', completed=True, new_objective_evaluations=0)
            return
        pilot(out, roots, inv, plans, entries, files, alpha=args.alpha)
    except Exception as exc:
        status(out, 'stopped', completed=False, error=f'{type(exc).__name__}: {exc}')
        print('[FINAL-OOS] STOP:', exc, '\n报告目录:', out, file=sys.stderr, flush=True)
        raise SystemExit(2)


if __name__ == '__main__':
    main()
