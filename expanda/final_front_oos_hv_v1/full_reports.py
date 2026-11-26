"""Summaries of fixed final fronts. Evaluation-scenario count is not search budget."""
from collections import defaultdict
from pathlib import Path
from statistics import mean, median, stdev

from inventory import read, save, csv_write, require, OBJECTIVES, SHARED
from run import REF, DIVISOR

TOL = 1e-12 / DIVISOR


def equal_mean(rows, key):
    """Seeds, instances, then fixed configurations, with no silent missing-value exclusion."""
    if any(r[key] is None for r in rows):
        return None
    groups = defaultdict(lambda: defaultdict(list))
    for r in rows:
        groups[r['configuration']][r['pair_instance']].append(r[key])
    return mean(mean(mean(v) for v in instances.values()) for instances in groups.values())


def rel(a, b):
    return 100 * (a - b) / b if b > 0 else None


def weight_quantile(np, values, weights):
    order = np.argsort(values, kind='stable')
    v, w = np.asarray(values)[order], np.asarray(weights)[order]
    s = np.cumsum(w) / np.sum(w)
    i = int(np.searchsorted(s, .5))
    if abs(s[i] - .5) <= 1e-12 and i + 1 < len(v):
        return float((v[i] + v[i+1]) / 2)
    return float(v[i])


def report(np, hv_fn, root, groups, load_pack, alpha):
    """New report directory per invocation; completed evaluation packs stay immutable."""
    from datetime import datetime, timezone
    dest = root / ('reports_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    dest.mkdir()
    quantile_label = f'P{round(100*alpha):d}'
    packets = {g['folder']: read(Path(g['folder']) / 'INPUTS.json') for g in groups}
    bounds, packs = {}, {}
    for g in groups:
        folder = Path(g['folder']); p = packets[g['folder']]
        packs[g['folder']] = {tag: load_pack(folder, p, tag) for tag in
            ('optimization_100', 'fresh_100', 'fresh_1000', 'fresh_5000')}
        pts = np.concatenate([packs[g['folder']]['fresh_' + str(n)]['quantiles'] for n in (100, 1000, 5000)])
        lo, hi = pts.min(axis=0), pts.max(axis=0)
        unit = g['bounds_unit']
        if unit in bounds:
            lo = np.minimum(lo, bounds[unit]['minimum']); hi = np.maximum(hi, bounds[unit]['maximum'])
        bounds[unit] = dict(minimum=lo.tolist(), maximum=hi.tolist(), reference=list(REF), divisor=DIVISOR,
            scope='All methods and search seeds; all final decisions; fresh 100/1000/5000 only')
    save(dest / 'INSTANCE_BOUNDS.json', bounds)
    hvrows, shifts, run_shifts, counts, coverage = [], [], [], [], []
    for g in groups:
        p, pp = packets[g['folder']], packs[g['folder']]
        byid = {r['exact_decision_id']: i for i, r in enumerate(p['decisions'])}
        bound = bounds[g['bounds_unit']]
        lo, hi = np.asarray(bound['minimum']), np.asarray(bound['maximum'])
        span = np.where(hi > lo, hi-lo, 1.)
        for r in p['source_runs']:
            members = [v for v in p['memberships'] if v['method'] == r['method'] and v['seed'] == r['seed']]
            ids = [byid[v['exact_decision_id']] for v in members]
            require(len(ids) == r['final_solutions'], '最终解计数不符')
            common = dict(table=r['table'], configuration=r['configuration'], instance=r['instance'],
                pair_instance=p['instance'].get('pair_id', r['instance']), network=r['network'],
                method=r['method'], seed=r['seed'])
            record = read(r['run_path'])
            runtime = record.get('optimisation_seconds')
            counts.append(dict(**common, final_solutions=len(ids), unique_decisions=len(set(ids)),
                actual_search_evaluations=r['actual_evaluations'], source_front=r['front_path']))
            for tag, arrays in pp.items():
                z = (arrays['quantiles'][ids] - lo) / span
                outside = np.any(z > 1, axis=1)
                beyond_ref = np.any(z > np.asarray(REF), axis=1)
                zero_box = np.any(z >= np.asarray(REF), axis=1)
                hvrows.append(dict(**common, scenario_group=tag, evaluation_scenarios=int(tag.rsplit('_', 1)[1]),
                    alpha=alpha, final_solutions=len(ids), search_budget=5000,
                    actual_search_evaluations=r['actual_evaluations'],
                    hv=float(hv_fn(z.tolist(), REF) / DIVISOR), original_search_seconds=runtime,
                    outside_upper_bound=int(outside.sum()), outside_upper_bound_percent=float(outside.mean()*100),
                    beyond_reference=int(beyond_ref.sum()), beyond_reference_percent=float(beyond_ref.mean()*100),
                    zero_box_plans=int(zero_box.sum()), below_lower_bound=int(np.any(z < 0, axis=1).sum()),
                    reevaluated_infeasible=int((~arrays['feasible'][ids]).sum()), bounds_unit=g['bounds_unit']))
            for j, objective in enumerate(OBJECTIVES):
                a, b = pp['optimization_100']['quantiles'][ids, j], pp['fresh_5000']['quantiles'][ids, j]
                values = [rel(float(y), float(x)) for x, y in zip(a, b)]
                part = []
                for k, member in enumerate(members):
                    row = dict(**common, objective=objective, source_solution_id=member['source_solution_id'],
                        exact_decision_id=member['exact_decision_id'], original100_quantile=float(a[k]),
                        fresh5000_quantile=float(b[k]), difference=float(b[k]-a[k]),
                        signed_relative_difference_percent=values[k])
                    shifts.append(row); part.append(row)
                run_shifts.append(dict(**common, objective=objective, plans=len(ids),
                    zero_denominators=sum(x is None for x in values),
                    mean_difference=float(np.mean(b-a)), median_difference=float(np.median(b-a)),
                    mean_relative_percent=mean(values) if all(x is not None for x in values) else None,
                    median_relative_percent=median(values) if all(x is not None for x in values) else None))
                cv = pp['fresh_5000']['coverage'][ids, j]
                coverage.append(dict(**common, objective=objective, plans=len(ids),
                    mean_coverage=float(np.mean(cv)), median_coverage=float(np.median(cv)),
                    fraction_below_085=float(np.mean(cv < .85))))
    csv_write(dest / 'FINAL_FRONT_COUNTS.csv', counts)
    csv_write(dest / (quantile_label+'_SHIFT_PER_SOLUTION.csv'), shifts)
    csv_write(dest / (quantile_label+'_SHIFT_PER_RUN.csv'), run_shifts)
    csv_write(dest / 'ORIGINAL_QUANTILE_COVERAGE_PER_RUN.csv', coverage)
    # Pair using only an actual reference available in the experiment.
    index = {(r['table'], r['pair_instance'], r['network'], r['method'], r['seed'], r['scenario_group']): r for r in hvrows}
    require(len(index) == len(hvrows), 'HV索引重复')
    for r in hvrows:
        net = 'small' if r['table'] == 'Table7' else r['network']
        method = SHARED if r['table'] == 'Table7' and r['network'] == 'large' else 'BASE'
        reference = index[r['table'], r['pair_instance'], net, method, r['seed'], r['scenario_group']]
        delta = r['hv'] - reference['hv']
        r.update(reference_network=net, reference_method=method, reference_hv=reference['hv'], hv_difference=delta,
                 paired_relative_percent=rel(r['hv'], reference['hv']),
                 wtl='W' if delta > TOL else 'L' if delta < -TOL else 'T')
    csv_write(dest / 'PER_RUN_HV.csv', hvrows)
    summaries = []
    grouped = defaultdict(list)
    for r in hvrows:
        grouped[r['table'], r['network'], r['method'], r['scenario_group']].append(r)
    for (table, network, method, tag), rows in grouped.items():
        for cfg in ['ALL'] + sorted({r['configuration'] for r in rows}):
            part = rows if cfg == 'ALL' else [r for r in rows if r['configuration'] == cfg]
            times = [r['original_search_seconds'] for r in part]
            self_comparison = method == part[0]['reference_method'] and network == part[0]['reference_network']
            summaries.append(dict(table=table, configuration=cfg, network=network, method=method,
                scenario_group=tag, evaluation_scenarios=part[0]['evaluation_scenarios'],
                reference_network=part[0]['reference_network'], reference_method=part[0]['reference_method'],
                runs=len(part), instances=len({r['instance'] for r in part}), mean_hv=equal_mean(part, 'hv'),
                reference_mean_hv=equal_mean(part, 'reference_hv'), mean_hv_difference=equal_mean(part, 'hv_difference'),
                mean_paired_relative_percent=equal_mean(part, 'paired_relative_percent'),
                undefined_relative_pairs=sum(r['paired_relative_percent'] is None for r in part),
                W=None if self_comparison else sum(r['wtl'] == 'W' for r in part),
                T=None if self_comparison else sum(r['wtl'] == 'T' for r in part),
                L=None if self_comparison else sum(r['wtl'] == 'L' for r in part),
                original_runtime_mean_s=mean(times) if all(t is not None for t in times) else None,
                original_runtime_sd_s=stdev(times) if all(t is not None for t in times) and len(times)>1 else None))
    csv_write(dest / 'ALL_SCENARIO_SUMMARY.csv', summaries)
    csv_write(dest / 'MAIN_5000.csv', [r for r in summaries if r['scenario_group'] == 'fresh_5000'])
    csv_write(dest / 'CONTRAST_1000.csv', [r for r in summaries if r['scenario_group'] == 'fresh_1000'])
    diff = []
    for r in hvrows:
        if r['scenario_group'] != 'fresh_100':
            continue
        for n1, n2 in ((100, 1000), (100, 5000), (1000, 5000)):
            key = (r['table'], r['pair_instance'], r['network'], r['method'], r['seed'])
            a, b = index[key + (f'fresh_{n1}',)]['hv'], index[key + (f'fresh_{n2}',)]['hv']
            diff.append({**{k: r[k] for k in ('table','configuration','instance','pair_instance','network','method','seed')},
                'comparison': f'{n1}_vs_{n2}', 'hv_a': a, 'hv_b': b,
                'signed_relative_difference_percent': rel(a, b),
                'absolute_relative_difference_percent': abs(rel(a, b)) if b > 0 else None})
    csv_write(dest / 'SCENARIO_DIFFERENCES_PER_RUN.csv', diff)
    ds = []
    keys = {(r['table'], r['network'], r['method'], r['comparison']) for r in diff}
    for t, net, m, comp in sorted(keys):
        part = [r for r in diff if (r['table'],r['network'],r['method'],r['comparison']) == (t,net,m,comp)]
        ds.append(dict(table=t, network=net, method=m, comparison=comp, runs=len(part),
            mean_signed_relative_percent=equal_mean(part,'signed_relative_difference_percent'),
            mean_absolute_relative_percent=equal_mean(part,'absolute_relative_difference_percent')))
    csv_write(dest / 'SCENARIO_DIFFERENCES.csv', ds)
    aggregate = []
    keys = {(r['table'], r['network'], r['method'], r['objective']) for r in run_shifts}
    for t, net, m, obj in sorted(keys):
        rr = [r for r in run_shifts if (r['table'],r['network'],r['method'],r['objective']) == (t,net,m,obj)]
        ss = [r for r in shifts if (r['table'],r['network'],r['method'],r['objective']) == (t,net,m,obj)]
        config_n = len({r['configuration'] for r in rr})
        instances = {c: len({r['instance'] for r in rr if r['configuration']==c}) for c in {r['configuration'] for r in rr}}
        runs = {i: [r for r in rr if r['instance']==i] for i in {r['instance'] for r in rr}}
        sizes = {(r['instance'],r['seed']): r['plans'] for r in rr}
        weights = [1/(config_n * instances[r['configuration']] * len(runs[r['instance']]) * sizes[r['instance'],r['seed']]) for r in ss]
        vals = [r['signed_relative_difference_percent'] for r in ss]
        valid = all(v is not None for v in vals)
        aggregate.append(dict(table=t, network=net, method=m, objective=obj, runs=len(rr), plan_memberships=len(ss),
            zero_denominators=sum(v is None for v in vals),
            hierarchy_equal_mean_relative_percent=equal_mean(rr,'mean_relative_percent'),
            hierarchy_weighted_median_relative_percent=weight_quantile(np, vals, weights) if valid else None,
            hierarchy_mean_of_run_medians_percent=equal_mean(rr,'median_relative_percent'),
            pooled_plan_mean_relative_percent=mean(vals) if valid else None,
            pooled_plan_median_relative_percent=median(vals) if valid else None,
            hierarchy_mean_difference_in_objective_units=equal_mean(rr,'mean_difference')))
    csv_write(dest / (quantile_label+'_SHIFT_BY_METHOD.csv'), aggregate)
    save(dest / 'REPORT.json', dict(completed=True, alpha=alpha, runs=len(counts), hv_rows=len(hvrows),
        main_scenarios=5000, contrast_scenarios=1000, new_searches=0, training_updates=0,
        reference=REF, hv_divisor=DIVISOR, wtl_tolerance=TOL,
        summary='Seeds averaged within instance; instances within configuration; configurations equally.',
        shifts='100*(Qfresh5000-Qoptimization100)/Qoptimization100. Positive means a larger (worse) minimization objective.',
        medians='Hierarchy-weighted empirical median of all plan memberships; also report pooled median and mean of per-run medians.',
        zero_denominators='Undefined, not dropped: corresponding method relative mean/median left empty.',
        original100='Evaluated under fresh common bounds only for in-sample/out-of-sample comparison; old-bound reproduction is separate.',
        uncertainty_intervals='Not recomputed in this export; no old confidence intervals are reused.'))
    lines = ['全量最终解复评完成；没有新增搜索或训练。',
             '100/1000/5000均为独立复评场景数；所有列使用搜索结束时同一批最终解。',
             'HV已除以1.331。主结果为fresh_5000；fresh_1000为对照。',
             'Table7大网络参照小网络共享监督，没有大网络Base。',
             '表 | 网络 | 方法 | 复评场景 | 运行数 | 平均HV | 相对参照变化% | W/T/L']
    for r in summaries:
        if r['configuration'] == 'ALL' and r['scenario_group'] in ('fresh_1000', 'fresh_5000'):
            pct = r['mean_paired_relative_percent']
            wtl = '—' if r['W'] is None else f"{r['W']}/{r['T']}/{r['L']}"
            lines.append(f"{r['table']} | {r['network']} | {r['method']} | {r['evaluation_scenarios']} | {r['runs']} | {r['mean_hv']:.9f} | {pct if pct is not None else '未定义'} | {wtl}")
    (dest / 'REPORT.txt').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print('\n'.join(lines), flush=True)
    return dest
