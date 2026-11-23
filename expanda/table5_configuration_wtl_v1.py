#!/usr/bin/env python3
"""Split unified Table 5 paired outcomes by S0-S8; never run searches."""
import argparse
from collections import defaultdict
import csv
import hashlib
import io
import json
import math
from pathlib import Path

DEFAULT = Path(__file__).resolve().parent / 'transformer_ppo_runs/diagnostics/ccp_shared_sl_ablations_b5000_v2'
METHODS = (
    'BASE', 'I_RULE', 'I_LEARN', 'C_RULE', 'C_LEARN', 'M_RULE', 'M_LEARN',
    'ICM_HEURISTIC', 'ICM_INDEPENDENT', 'ICM_SHARED_SUPERVISED',
    'ICM_TERMINAL_PPO', 'ICM_RANDOM_RATE_CONTROLLED', 'ICM_SHARED_ZERO_THRESHOLD',
)
BUDGETS = (400, 1000, 5000)
CONFIGURATIONS = tuple(f'S{i}' for i in range(9))
OUTCOMES = ('胜', '平', '负')


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def outcome(difference, tolerance):
    return 0 if difference > tolerance else 2 if difference < -tolerance else 1


def close(actual, expected, label):
    if not math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError(f'{label}: {actual} != {expected}')


def main(root, output=None):
    root = root.resolve()
    source = root / 'results'
    out = (output or root / 'CONFIGURATION_WTL_v1').resolve()
    if out == root or out == source or source in out.parents or out in root.parents:
        raise ValueError('Output must be separate from the original report directory')
    hashes = {}

    def load(path, kind='json'):
        raw = path.read_bytes()
        hashes[str(path)] = digest(raw)
        text = raw.decode('utf-8-sig')
        return json.loads(text) if kind == 'json' else list(csv.DictReader(io.StringIO(text)))

    status = load(root / 'STATUS.json')
    plan = load(root / 'PLAN.json')
    meta = load(source / 'REPORT.json')
    payload = json.dumps({k: v for k, v in plan.items() if k != 'sha256'},
                         sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    if (not status.get('completed') or not meta.get('completed')
            or plan['protocol'] != 'single-shared-random-rate-controlled-union-v2'
            or digest(payload) != plan['sha256'] or meta['plan_sha256'] != plan['sha256']
            or meta['pairs_per_method'] != 54 or meta['primary_reference'] != 'BASE'
            or set(meta['methods']) != set(METHODS)
            or meta['hv_divisor'] != 1.331 or not meta['hv_already_divided']):
        raise ValueError('Requires the completed unified v2 report: 13 methods, 54 pairs')
    tolerance = float(meta['wtl_tolerance'])
    if not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError('Invalid source W/T/L tolerance')
    table_rows = load(source / 'TABLE5.csv', 'csv')
    table = {r['method']: r for r in table_rows}
    if len(table_rows) != 13 or set(table) != set(METHODS):
        raise ValueError('Missing or duplicate Table 5 methods')
    overall = {}
    for r in load(source / 'ALL_RESULTS.csv', 'csv'):
        if r['reference_method'] != 'BASE':
            continue
        key = (r['method'], int(r['评价次数']))
        if key in overall:
            raise ValueError(f'Duplicate aggregate: {key}')
        overall[key] = r
    expected_groups = {(m, b) for m in METHODS for b in BUDGETS}
    if set(overall) != expected_groups:
        raise ValueError('Incomplete ALL_RESULTS method/budget panel')
    groups = defaultdict(list)
    for r in load(source / 'PER_PAIR.csv', 'csv'):
        if r['reference_method'] == 'BASE':
            groups[r['method'], int(r['evaluation'])].append(r)
    if set(groups) != expected_groups:
        raise ValueError('Incomplete PER_PAIR method/budget panel')
    pairing = lambda r: (r['instance'], r['seed'])
    panel = None
    results, cells = [], {}
    for budget in BUDGETS:
        base_rows = groups['BASE', budget]
        base = {pairing(r): r for r in base_rows}
        if len(base_rows) != 54 or len(base) != 54:
            raise ValueError('Expected 54 unique Base pairs at each budget')
        cfg_instances, instance_seeds = defaultdict(set), defaultdict(set)
        for r in base.values():
            cfg_instances[r['configuration']].add(r['instance'])
            instance_seeds[r['instance']].add(r['seed'])
            own, ref = float(r['own']), float(r['reference'])
            if not math.isfinite(own) or own <= 0 or own != ref:
                raise ValueError('Invalid Base self-comparison')
        if (set(cfg_instances) != set(CONFIGURATIONS)
                or any(len(v) != 2 for v in cfg_instances.values())
                or len(instance_seeds) != 18 or any(len(v) != 3 for v in instance_seeds.values())
                or len({tuple(sorted(v)) for v in instance_seeds.values()}) != 1):
            raise ValueError('Expected S0-S8, two instances each, three common seeds')
        current_panel = {k: r['configuration'] for k, r in base.items()}
        if panel is not None and panel != current_panel:
            raise ValueError('Pair/configuration identities differ across budgets')
        panel = current_panel
        for method in METHODS:
            rows = groups[method, budget]
            if len(rows) != 54 or {pairing(r) for r in rows} != set(base):
                raise ValueError(f'Missing/duplicate paired runs: {method}/{budget}')
            counts = {c: [0, 0, 0] for c in CONFIGURATIONS}
            for r in rows:
                b = base[pairing(r)]
                own, ref = float(r['own']), float(r['reference'])
                if (not math.isfinite(own) or own < 0 or ref != float(b['own'])
                        or r['configuration'] != b['configuration']):
                    raise ValueError(f'Invalid pairing or HV: {method}/{budget}')
                difference = own - ref
                # Exact round-trip reproduction is necessary near the tie tolerance.
                if difference != float(r['absolute']):
                    raise ValueError('Saved HV difference does not reproduce exactly')
                close(100 * difference / ref, float(r['percent']), 'Saved paired percentage')
                counts[r['configuration']][outcome(difference, tolerance)] += 1
            total = [sum(v[i] for v in counts.values()) for i in range(3)]
            original = overall[method, budget]
            if int(original['配对数']) != 54 or int(table[method]['配对数']) != 54:
                raise ValueError('Wrong aggregate pair count')
            if method != 'BASE':
                if total != [int(original[k]) for k in OUTCOMES]:
                    raise ValueError(f'WTL mismatch with ALL_RESULTS: {method}/{budget}')
                if budget == 5000 and total != [int(table[method][k]) for k in OUTCOMES]:
                    raise ValueError(f'WTL mismatch with Table 5: {method}')
            elif total != [0, 54, 0]:
                raise ValueError('Base self-comparison must contain only ties')
            for cfg, values in list(counts.items()) + [('ALL', total)]:
                n = 54 if cfg == 'ALL' else 6
                if sum(values) != n:
                    raise ValueError(f'Wrong configuration pair count: {method}/{budget}/{cfg}')
                display = '—' if method == 'BASE' else '/'.join(map(str, values))
                cells[method, budget, cfg] = display
                results.append(dict(evaluation=budget, configuration=cfg, method=method,
                                    方法=table[method]['方法'], reference_method='BASE',
                                    pairs=n, wins=values[0], ties=values[1], losses=values[2],
                                    W_T_L=display))

    provenance = dict(protocol='table5-configuration-wtl-v1', source=str(source), files=hashes,
                      script_sha256=digest(Path(__file__).read_bytes()),
                      wtl_tolerance=tolerance, hv_already_divided=True,
                      configurations=list(CONFIGURATIONS), pairs_per_configuration=6,
                      all_results_verified=True, table5_verified=True,
                      new_searches=0, new_objective_evaluations=0, new_hv_calculations=0)
    if (out / 'SOURCE.json').exists() and json.loads((out / 'SOURCE.json').read_text()) != provenance:
        raise ValueError('Existing export uses a different source; choose a new --out')
    for name, expected in hashes.items():
        if digest(Path(name).read_bytes()) != expected:
            raise ValueError('Source changed during calculation: ' + name)
    out.mkdir(parents=True, exist_ok=True)

    def write(name, text):
        p = out / name
        temp = p.with_name(p.name + '.writing')
        temp.write_text(text, encoding='utf-8')
        temp.replace(p)

    def write_csv(name, rows):
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        write(name, '\ufeff' + buf.getvalue())

    lines = ['统一重算 v2：各方法相对同实例、同种子 Base 的胜/平/负。',
             'S0–S8 每配置2实例×3种子=6对；合计54对。Base自比较显示—。',
             f'沿用原报告HV差容差：{tolerance:.16g}；不重新计算HV。',
             '已核对三个节点的合计与ALL_RESULTS一致，5000次合计与TABLE5一致。']
    for budget in BUDGETS:
        wide = []
        for method in METHODS:
            wide.append({'方法': table[method]['方法'],
                         **{c: cells[method, budget, c] for c in CONFIGURATIONS},
                         '合计': cells[method, budget, 'ALL']})
        write_csv(f'WTL_{budget}.csv', wide)
        lines += ['', f'{budget:,}次评价：W/T/L', '',
                  '| ' + ' | '.join(wide[0]) + ' |',
                  '| ' + ' | '.join(['---'] * len(wide[0])) + ' |']
        lines.extend('| ' + ' | '.join(r.values()) + ' |' for r in wide)
    write_csv('WTL_LONG.csv', results)
    write('REPORT.txt', '\n'.join(lines) + '\n')
    write('SOURCE.json', json.dumps(provenance, ensure_ascii=False, indent=2) + '\n')
    print('\n'.join(lines))
    print('\n[CONFIGURATION-WTL] COMPLETE', out)
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=DEFAULT)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    main(args.root, args.out)
