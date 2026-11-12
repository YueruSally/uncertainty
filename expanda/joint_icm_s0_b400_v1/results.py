"""Report all old and new arms with one set of per-instance post-hoc HV bounds."""
from common import *
import math
import statistics

NAMES = {'BASE': 'Base', RULE: 'ICM 启发式', INDEPENDENT: 'ICM 独立 Transformer',
         SHARED: 'ICM 共享监督 Transformer', PPO: 'ICM 终局联合 PPO'}


def comparison_tables(rows, methods, hv):
    index = {(r['instance'], r['algorithm_seed'], r['method']): r for r in rows}
    comparisons = {}; table = []; concise = []
    for i, method in enumerate(methods):
        own = [r for r in rows if r['method'] == method]
        refs = ['BASE']
        if method in (INDEPENDENT, SHARED, PPO): refs += [RULE]
        if method in (SHARED, PPO): refs += [INDEPENDENT]
        if method == PPO: refs += [SHARED]
        comparisons[method] = {}
        for j, ref in enumerate(dict.fromkeys(refs)):
            c = hv.compare(rows, method, ref, 989000 + 10 * i + j)
            comparisons[method][ref] = c; p = c['percent']; a = c['absolute']
            percentages = [100 * (r['hv'] / index[r['instance'], r['algorithm_seed'], ref]['hv'] - 1)
                           if index[r['instance'], r['algorithm_seed'], ref]['hv'] > 0 else None for r in own]
            table.append({'方法': NAMES.get(method, method), '参照': NAMES.get(ref, ref),
                '运行对数': len(own), '平均HV': statistics.fmean(r['hv'] for r in own),
                'HV样本标准差': statistics.stdev(r['hv'] for r in own),
                '配对平均提升(%)': p['mean'],
                '配对提升标准差(百分点)': statistics.stdev(percentages) if all(x is not None for x in percentages) else None,
                '提升95%区间下限(%)': p['hierarchical_bootstrap_95ci_low'],
                '提升95%区间上限(%)': p['hierarchical_bootstrap_95ci_high'],
                '平均HV绝对差': a['mean'], '绝对差95%区间下限': a['hierarchical_bootstrap_95ci_low'],
                '绝对差95%区间上限': a['hierarchical_bootstrap_95ci_high'],
                '胜': a['wins'], '平': a['ties'], '负': a['losses'],
                '平均搜索时间(秒)': statistics.fmean(r['seconds'] for r in own)})
        if method in NAMES:
            b = comparisons[method]['BASE']['percent']
            r = comparisons[method].get(RULE, {}).get('percent', {})
            concise.append({'方法': NAMES[method], '相对Base提升(%)': b['mean'],
                'Base比较95%区间下限(%)': b['hierarchical_bootstrap_95ci_low'],
                'Base比较95%区间上限(%)': b['hierarchical_bootstrap_95ci_high'],
                '相对ICM启发式提升(%)': r.get('mean'),
                '平均搜索时间(秒)': statistics.fmean(v['seconds'] for v in own)})
    return comparisons, table, concise


def report(root, plan, added, out):
    design, rows, inputs, hv = history(Path(plan['source']))
    for name in plan['instances']:
        for seed in plan['seeds']:
            reference = read_json(Path(plan['original']) / 'runs' / name / f'seed_{seed}/BASE/configuration.json')
            for method in added:
                folder = root / 'screen' / name / f'seed_{seed}' / method
                r = read_json(folder / 'RUN_COMPLETE.json'); cfg = read_json(folder / 'configuration.json')
                if (not r['completed'] or r['plan_sha256'] != plan['sha256'] or r['strategy'] != method or
                    r['instance'] != name or r['algorithm_seed'] != seed or r['stochastic'] or
                    r['real_objective_evaluation_count'] != 400 or r['combined_hidden_candidate_objective_evaluations']):
                    raise ValueError('Invalid new screen record')
                for k in ('scenario_digest', 'instance_digest', 'normalization', 'seeds', 'population', 'evaluation_budget'):
                    if cfg[k] != reference[k]: raise ValueError('Pairing mismatch: ' + k)
                points = []
                for item in read_json(folder / 'final_feasible_nondominated.json'):
                    if item['training_feasible'] is not True or item['source_run'] != name:
                        raise ValueError('Invalid final feasible front')
                    point = tuple(float(item['optimisation_objectives'][k]) for k in hv.OBJECTIVES)
                    if not all(math.isfinite(v) for v in point): raise ValueError('Nonfinite objective')
                    points.append(point)
                if set(hv.nondominated(points)) != set(points): raise ValueError('Dominated stored final point')
                old = cfg['normalization']; lo = old['minimum']; hi = old['maximum']; div = old['objective_divisor']
                replay = hv.exact_hv_3d([[(p[j]/div[j]-lo[j])/(hi[j]-lo[j]) for j in range(3)] for p in points], hv.REFERENCE)
                if not math.isclose(replay, r['final_hypervolume'], rel_tol=1e-10, abs_tol=1e-12):
                    raise ValueError('Training-normalization HV replay differs')
                rows.append({'instance': name, 'algorithm_seed': seed, 'method': method, 'points': points,
                    'old_hv': r['final_hypervolume'], 'seconds': r['optimisation_seconds']})
                for filename in ('RUN_COMPLETE.json', 'configuration.json', 'final_feasible_nondominated.json'):
                    inputs[str(folder / filename)] = digest(folder / filename)
    bounds = {name: hv.bounds_of([r['points'] for r in rows if r['instance'] == name]) for name in plan['instances']}
    for r in rows: r['hv'] = hv.exact_hv_3d(hv.normalize(r['points'], bounds[r['instance']]), hv.REFERENCE)
    methods = list(design['methods']) + added
    comparisons, table, concise = comparison_tables(rows, methods, hv)
    out.mkdir(parents=True, exist_ok=True)
    hv.write_csv(out / 'STATISTICS.csv', table); hv.write_csv(out / '00_联合训练对比.csv', concise)
    hv.write_csv(out / 'PER_RUN.csv', [{k: r[k] for k in ('instance', 'algorithm_seed', 'method', 'old_hv', 'hv', 'seconds')} for r in rows])
    per_instance = []
    for name in plan['instances']:
        for method in methods:
            subset = [r for r in rows if r['instance'] == name]
            c = hv.compare(subset, method, 'BASE', 989900)['percent']
            per_instance.append({'instance': name, 'method': method, 'percent_vs_base': c['mean'],
                'mean_hv': statistics.fmean(r['hv'] for r in subset if r['method'] == method)})
    hv.write_csv(out / 'PER_INSTANCE.csv', per_instance)
    write_json(out / 'INSTANCE_BOUNDS.json', bounds)
    selected = read_json(root / 'terminal/SELECTED.json') if PPO in added else None
    result = {'completed': True, 'methods': comparisons, 'reused_searches': 270,
        'new_screen_searches': len(added) * 30, 'total_reported_searches': len(rows),
        'new_screen_objective_evaluations': len(added) * 12000,
        'ppo_training_searches': 32 if PPO in added else 0, 'ppo_validation_searches': 60 if PPO in added else 0,
        'completed_new_objective_evaluations': len(added) * 12000 + (36800 if PPO in added else 0),
        'hidden_screen_evaluations': 0, 'selected_ppo': selected,
        'supervised_model_reused': plan['source_model'] is not None,
        'interrupted_attempts': len(list((root / 'interrupted').glob('*'))),
        'qualification': 'Three previously inspected S0 development instances, not independent confirmation. Reporting min/max includes all methods here and may change historical percentages. Partial interrupted cost is additional. No guarantee of gain. If PPO selected update zero, it is the unchanged supervised policy.',
        'timing': SETTINGS_B400['timing'], 'final_test_opened': False}
    write_json(out / 'QUICK_RESULT.json', result)
    write_json(out / 'INPUT_MANIFEST.json', {'plan_sha256': plan['sha256'], 'files': inputs})
    print((out / '00_联合训练对比.csv').read_text(encoding='utf-8-sig'), flush=True)
    return result
