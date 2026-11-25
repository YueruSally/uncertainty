#!/usr/bin/env python3
"""Read existing Table 5 records only. No solver/evaluator/model imports."""
import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import sys

import numpy as np

sys.dont_write_bytecode = True
SHARED = 'ICM_SHARED_SUPERVISED'
RANDOM = 'ICM_RANDOM_RATE_CONTROLLED'
ZERO = 'ICM_SHARED_ZERO_THRESHOLD'
RULE = 'ICM_HEURISTIC'
NAMES = {SHARED: 'shared SL', RANDOM: '随机替换', ZERO: '全零阈值', RULE: '规则 ICM'}
PAIRS = ((SHARED, RANDOM), (SHARED, ZERO), (RANDOM, RULE))
BUDGETS = (400, 1000, 5000)
DRAWS = 50000
RNG_SEED = 20261125
DEFAULT = Path(__file__).resolve().parent / 'transformer_ppo_runs/diagnostics/ccp_shared_sl_ablations_b5000_v2'


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def close(a, b, label):
    if not np.isclose(float(a), float(b), rtol=1e-10, atol=1e-10):
        raise ValueError(f'{label}: {a} != {b}')


def signed_rank_exact(values):
    """Two-sided signed-rank, zero deletion, average ties, exact sign enumeration.

    Input units are instance means (18) or configuration means (9), not seeds.
    Round to 12 decimal places in percentage points only for rank/tie handling.
    Exact conditional random-sign null; no normal approximation or continuity correction.
    """
    x = np.round(np.asarray(values, dtype=float).ravel(), 12)
    x = x[x != 0]
    n = len(x)
    if n == 0:
        return dict(n_nonzero=0, statistic=0., p_value=1.)
    if n > 18:
        raise ValueError('Wilcoxon must not treat the 54 seed runs as independent units')
    order = np.argsort(np.abs(x), kind='stable')
    ranks2 = np.empty(n, dtype=int)
    start = 0
    while start < n:
        end = start + 1
        while end < n and abs(x[order[end]]) == abs(x[order[start]]):
            end += 1
        ranks2[order[start:end]] = start + 1 + end
        start = end
    observed = int(ranks2[x > 0].sum())
    total = int(ranks2.sum())
    sums = np.array([0], dtype=int)
    for rank in ranks2:
        sums = np.concatenate((sums, sums + rank))
    p = min(1., 2. * min(float(np.mean(sums <= observed)), float(np.mean(sums >= observed))))
    return dict(n_nonzero=n, statistic=min(observed, total - observed)/2., p_value=p)


def draw_indices():
    # S0-S8 are fixed strata. Resample two instances per configuration, then
    # three paired seeds independently within each selected instance copy.
    rng = np.random.default_rng(RNG_SEED)
    inst = rng.integers(0, 2, size=(DRAWS, 9, 2), dtype=np.uint8)
    seeds = rng.integers(0, 3, size=(DRAWS, 9, 2, 3), dtype=np.uint8)
    # Secondary, explicitly labelled interval under the original Table 5 v2
    # fixed-instance/seed-only protocol, for comparison with the existing paper.
    legacy = np.random.default_rng(20261123).integers(0, 3, size=(DRAWS, 18, 3), dtype=np.uint8)
    return inst, seeds, legacy


def intervals(x, indices):
    inst, seeds, legacy = indices
    sample = x[np.arange(9)[None, :, None, None], inst[..., None], seeds]
    hierarchical = sample.mean(axis=3).mean(axis=2).mean(axis=1)
    flat = x.reshape(18, 3)
    original = flat[np.arange(18)[None, :, None], legacy].mean(axis=2).reshape(DRAWS, 9, 2).mean(axis=2).mean(axis=1)
    return tuple(float(v) for v in np.percentile(hierarchical, [2.5, 97.5])), tuple(float(v) for v in np.percentile(original, [2.5, 97.5]))


def write_csv(path, rows):
    with path.open('x', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(root, output=None):
    root = root.resolve()
    dest = (output or root / ('ABLATION_READONLY_AUDIT_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))).resolve()
    if dest == root or dest in root.parents or dest == root/'results' or root/'results' in dest.parents:
        raise ValueError('Output must be separate from the existing reports')
    if dest.exists():
        raise ValueError('Output already exists; no files will be overwritten: ' + str(dest))
    hashes = {}

    def load(path, kind='json', expected=None):
        path = Path(path).resolve()
        raw = path.read_bytes()
        value = sha(raw)
        if expected is not None and value != expected:
            raise ValueError('Frozen source changed: ' + str(path))
        hashes[str(path)] = value
        text = raw.decode('utf-8-sig')
        if kind == 'csv':
            return list(csv.DictReader(io.StringIO(text)))
        return json.loads(text) if kind == 'json' else text

    status = load(root/'STATUS.json')
    plan = load(root/'PLAN.json')
    canon = json.dumps({k: v for k, v in plan.items() if k != 'sha256'}, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    if not status.get('completed') or sha(canon) != plan['sha256'] or plan['protocol'] != 'single-shared-random-rate-controlled-union-v2':
        raise ValueError('Requires the completed, unmodified Table 5 v2 plan')
    report = load(root/'results/REPORT.json')
    if (not report['completed'] or report['plan_sha256'] != plan['sha256'] or report['pairs_per_method'] != 54
            or not report['hv_already_divided'] or report['hv_divisor'] != 1.331 or report['hv_reference'] != [1.1]*3):
        raise ValueError('Unified HV report metadata differs')
    bounds = load(root/'results/INSTANCE_BOUNDS.json')
    table = {r['method']: r for r in load(root/'results/TABLE5.csv', 'csv')}
    details = load(root/'results/PER_PAIR.csv', 'csv')
    old_rates = {(r['method'], r['component']): r for r in load(root/'results/REPLACEMENT_RATES.csv', 'csv')}
    targets = load(root/'TARGET_AUDIT.json', expected=plan['files'][str(root/'TARGET_AUDIT.json')])
    source_code = {}
    for suffix in ('shared_sl_ablations_b5000_v2/policy_v2.py',
                   'shared_sl_ablations_b5000_v2/run.py',
                   'joint_icm_multiscale_terminal_v1/search.py',
                   'component_screen_b20_v1/controller.py'):
        matches = [(p, h) for p, h in plan['files'].items() if p.endswith('/'+suffix)]
        if len(matches) != 1:
            raise ValueError('Cannot identify pinned implementation: ' + suffix)
        path, digest = matches[0]
        load(path, kind='text', expected=digest)
        source_code[suffix] = dict(path=path, sha256=digest)

    configs = [f'S{i}' for i in range(9)]
    instances = {c: sorted(e['instance_id'] for e in plan['entries'] if e['configuration'] == c) for c in configs}
    seeds = sorted(plan['settings']['screen_seeds'])
    if any(len(v) != 2 for v in instances.values()) or seeds != [993001, 993002, 993003]:
        raise ValueError('Expected two instances per configuration and three common seeds')
    keys = [(n, s) for c in configs for n in instances[c] for s in seeds]
    expected_cfg = {n: c for c in configs for n in instances[c]}
    if len(set(keys)) != 54 or set(bounds) != set(expected_cfg):
        raise ValueError('Instance bounds or pairing differs')
    hv = {}
    for budget in BUDGETS:
        for method in ('BASE', SHARED, RANDOM, ZERO, RULE):
            rows = [r for r in details if r['method'] == method and r['reference_method'] == 'BASE' and int(r['evaluation']) == budget]
            by = {(r['instance'], int(r['seed'])): r for r in rows}
            if len(rows) != 54 or set(by) != set(keys):
                raise ValueError('Missing or duplicated paired HV rows')
            for (n, _), r in by.items():
                if r['configuration'] != expected_cfg[n]:
                    raise ValueError('Configuration mismatch')
            values = np.array([float(by[k]['own']) for k in keys]).reshape(9, 2, 3)
            base = np.array([float(by[k]['reference']) for k in keys]).reshape(9, 2, 3)
            if not np.isfinite(values).all() or not np.isfinite(base).all() or np.any(base <= 0):
                raise ValueError('Invalid HV')
            if method == 'BASE':
                if not np.array_equal(values, base):
                    raise ValueError('Base self-pair differs')
            elif not np.array_equal(base, hv[budget, 'BASE']):
                raise ValueError('Methods do not share exactly the same paired Base HV')
            gain = 100*(values-base)/base
            close(values.mean(), table[method][f'{budget}_平均HV'], 'Table 5 mean HV')
            close(gain.mean(), table[method][f'{budget}_平均配对提升百分比'], 'Table 5 gain')
            for k, g in zip(keys, gain.ravel()):
                close(g, by[k]['percent'], 'Saved per-pair percentage')
            hv[budget, method] = values

    print('已核对共同边界报告、54对完整面板及Table 5均值；不重算HV。', flush=True)
    rate_rows, missing, source_records = [], [], {}
    for method in (SHARED, RANDOM, ZERO):
        paths = ([root/'runs'/n/f'seed_{s}'/RANDOM/'RUN_COMPLETE.json' for n, s in keys]
                 if method == RANDOM else [Path(p) for p in plan['source_records'][method]])
        records = []
        for path in paths:
            r = load(path, expected=None if method == RANDOM else plan['files'][str(path)])
            if not r.get('completed') or r['arm'] != method or r['configuration'] != expected_cfg[r['instance']]:
                raise ValueError('Wrong completed search record')
            k = f"{r['instance']}/{r['seed']}"
            if r['scenario_digest'] != plan['expected_scenarios'][k]:
                raise ValueError('Optimization scenarios differ')
            if method == RANDOM and (r['plan_sha256'] != plan['sha256'] or r['target_rates'] != plan['targets']):
                raise ValueError('Random run does not belong to this v2 plan')
            if method == ZERO and (set(r['effective_gates']) != set('ICM') or
                    any(g['disabled'] or g['threshold_scaled'] != 0 for g in r['effective_gates'].values())):
                raise ValueError('Not an all-zero-threshold deployment')
            records.append(r)
        if len(records) != 54 or {(r['instance'], r['seed']) for r in records} != set(keys):
            raise ValueError('Source run panel differs')
        source_records[method] = [str(p) for p in paths]
        for c in 'ICM':
            calls = switches = eligible = known = 0
            for path, r in zip(paths, records):
                count = r['selection'][c]
                n, s = int(count['calls']), int(count['switches'])
                if n != s + count['fallbacks']:
                    raise ValueError('Invalid selection counters')
                calls += n
                switches += s
                source = r['random_selection'][c] if method == RANDOM else count
                e = source.get('eligible')
                if e is None and 'no_legal_alternative' in source:
                    e = n - source['no_legal_alternative']
                if method == RANDOM and (source['calls'] != n or source['switches'] != s or
                                         source['eligible'] + source['no_legal_alternative'] != n):
                    raise ValueError('Random and controller counts disagree')
                if e is None:
                    missing.append(dict(method=method, component=c, source=str(path),
                                        saved_selection_fields=','.join(sorted(count)),
                                        reason='No eligible-decision counter; calls/fallbacks/unsupported do not determine it'))
                else:
                    if not 0 <= s <= e <= n or int(e) != e:
                        raise ValueError('Invalid eligible-decision count')
                    eligible += int(e)
                    known += 1
            old = old_rates[method, c]
            if int(old['calls']) != calls or int(old['switches']) != switches:
                raise ValueError('Replacement counts do not match current Table 5 report')
            actual = 100*switches/calls if calls else None
            close(actual, old['actual_percent'], 'All-call replacement rate')
            if method == SHARED:
                target = next(x for x in targets if x['component'] == c)
                if target['shared_calls'] != calls or target['shared_switches'] != switches:
                    raise ValueError('Target rate provenance differs')
                close(100*plan['targets'][c], target['target_percent'], 'Configured target')
            rate_rows.append(dict(method=method, component=c, runs=54, calls=calls, switches=switches,
                                  legacy_rate_all_calls_percent=actual,
                                  runs_with_eligible_counter=known,
                                  eligible=eligible if known == 54 else None,
                                  requested_rate_eligible_percent=100*switches/eligible if known == 54 and eligible else None,
                                  status='verified' if known == 54 else 'not_identifiable_from_saved_counters'))

    indices = draw_indices()
    results, per_instance, per_run = [], [], []
    for budget in BUDGETS:
        for a, b in PAIRS:
            d = 100*(hv[budget, a] - hv[budget, b])/hv[budget, 'BASE']
            means = d.mean(axis=2)
            cfgmeans = means.mean(axis=1)
            interval, original_interval = intervals(d, indices)
            test = signed_rank_exact(means)
            cfgtest = signed_rank_exact(cfgmeans)
            close(cfgmeans.mean(), float(table[a][f'{budget}_平均配对提升百分比']) - float(table[b][f'{budget}_平均配对提升百分比']), 'Mean difference identity')
            results.append(dict(evaluation=budget, method_a=a, method_b=b, pairs=54, instances=18, configurations=9,
                                mean_difference_pp=float(cfgmeans.mean()),
                                hierarchical_ci95_low_pp=interval[0], hierarchical_ci95_high_pp=interval[1],
                                fixed_instance_seed_only_ci95_low_pp=original_interval[0],
                                fixed_instance_seed_only_ci95_high_pp=original_interval[1],
                                wilcoxon_instance_n_nonzero=test['n_nonzero'], wilcoxon_instance_statistic=test['statistic'],
                                wilcoxon_instance_p=test['p_value'], wilcoxon_configuration_n_nonzero=cfgtest['n_nonzero'],
                                wilcoxon_configuration_p=cfgtest['p_value']))
            for ci, c in enumerate(configs):
                for ni, n in enumerate(instances[c]):
                    per_instance.append(dict(evaluation=budget, method_a=a, method_b=b, configuration=c,
                                             instance=n, seed_mean_difference_pp=float(means[ci, ni])))
                    for si, seed in enumerate(seeds):
                        per_run.append(dict(evaluation=budget, method_a=a, method_b=b, configuration=c,
                                            instance=n, seed=seed, hv_a=float(hv[budget, a][ci, ni, si]),
                                            hv_b=float(hv[budget, b][ci, ni, si]), hv_base=float(hv[budget, 'BASE'][ci, ni, si]),
                                            difference_pp=float(d[ci, ni, si])))

    for path, value in hashes.items():
        if sha(Path(path).read_bytes()) != value:
            raise ValueError('Source changed during audit: ' + path)
    dest.mkdir(parents=True, exist_ok=False)
    for filename, rows in [('PAIRED_DIFFERENCES.csv', results), ('REPLACEMENT_RATES.csv', rate_rows),
                           ('INSTANCE_MEANS.csv', per_instance), ('PER_PAIR_DIFFERENCES.csv', per_run)]:
        write_csv(dest/filename, rows)
    if missing:
        write_csv(dest/'MISSING_ELIGIBLE_COUNTERS.csv', missing)
    protocol = dict(source=str(root), source_hashes=hashes, source_records=source_records,
                    source_implementation=source_code, targets=plan['targets'], rate_rule=plan['rate_rule'],
                    target_audit=targets, table5_current_rates_verified=True,
                    formula='100*(HV_A-HV_B)/HV_Base', bootstrap_draws=DRAWS, bootstrap_seed=RNG_SEED,
                    hierarchy='9 fixed configuration strata; resample 2 instances within each; resample 3 paired seeds within each selected instance copy; equal means seeds -> instances -> configurations',
                    secondary_interval='Original Table 5 v2: keep 18 instances fixed, resample 3 paired seeds per instance; seed=20261123',
                    wilcoxon='Two-sided exact sign-enumeration signed-rank on 18 seed-averaged instance differences; also 9 configuration means. Average tied absolute ranks; drop zeros after rounding percentage points to 12 decimals. Unadjusted p-values; symmetric sign-null, not a test of the arithmetic mean.',
                    eligible_limit='Legacy shared/zero controllers do not save legal-mask histories or eligible counts; do not infer these from fallbacks or from another method. No replay.',
                    new_searches=0, new_objective_evaluations=0, training_updates=0, new_hv_calculations=0,
                    script_sha256=sha(Path(__file__).read_bytes()))
    (dest/'AUDIT.json').write_text(json.dumps(protocol, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    lines = ['Table 5 v2：只读消融核对', '新增搜索/方案评价/训练更新/HV重算均为0。',
             '表注旧率已与本批54次运行逐项核对，但其分母是全部选择调用数。',
             '新定义的分母=至少有一个合法非规则候选的决策数；缺失时留空，不以calls代替。',
             '', '方法 | 组件 | 全部调用 | 替换数 | 旧口径% | 合法决策数 | 新口径%']
    for r in rate_rows:
        e = str(r['eligible']) if r['eligible'] is not None else '未保存'
        p = f"{r['requested_rate_eligible_percent']:.6f}" if r['requested_rate_eligible_percent'] is not None else '无法恢复'
        lines.append(f"{NAMES[r['method']]} | {r['component']} | {r['calls']} | {r['switches']} | {r['legacy_rate_all_calls_percent']:.6f} | {e} | {p}")
    lines += ['', 'd=100*(HV_A-HV_B)/HV_Base；单位：百分点。',
              '主要95%区间：配置固定，配置内重抽实例，再实例内重抽配对种子，50000次。',
              '同时输出原Table 5的固定实例、仅重抽种子区间；两种不确定性口径不可混称。',
              'Wilcoxon：种子先平均，以18实例为主；附9配置结果。双侧精确p，未做多重比较校正。',
              '评价次数 | 比较 | 平均差 | 层级95%区间 | 固定实例95%区间 | p(18实例) | p(9配置)']
    for r in results:
        lines.append(f"{r['evaluation']} | {NAMES[r['method_a']]} − {NAMES[r['method_b']]} | {r['mean_difference_pp']:+.6f} | "
                     f"[{r['hierarchical_ci95_low_pp']:+.6f}, {r['hierarchical_ci95_high_pp']:+.6f}] | "
                     f"[{r['fixed_instance_seed_only_ci95_low_pp']:+.6f}, {r['fixed_instance_seed_only_ci95_high_pp']:+.6f}] | "
                     f"{r['wilcoxon_instance_p']:.8g} | {r['wilcoxon_configuration_p']:.8g}")
    text = '\n'.join(lines)+'\n'
    (dest/'REPORT_CN.txt').write_text(text, encoding='utf-8')
    print(text)
    print('[READONLY-AUDIT] COMPLETE', dest)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=DEFAULT)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    main(args.root, args.out)
