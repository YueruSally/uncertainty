"""Recompute saved single-member fronts and export paired W/T/L by configuration."""
import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

import numpy as np
from hv_joint_b5000_union_v1.recompute import BUDGETS, NAMES, hv3


def write_csv(path, rows):
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def audit(root):
    plan = json.loads((root / 'PLAN.json').read_bytes())
    settings = plan['settings']
    assert settings['population'] == 250 and settings['budget'] == 5000
    assert settings['scenario_count'] == 100 and tuple(settings['milestones']) == BUDGETS
    assert settings['confidence'] == .9
    assert len(plan['entries']) == 9 and len(settings['screen_seeds']) == 3
    methods = list(NAMES)
    records = {}
    paths = [Path(p) for p in plan['reused']]
    paths += sorted((root / 'runs').glob('*/seed_*/*/RUN_COMPLETE.json'))
    for path in paths:
        content = path.read_bytes()
        if str(path) in plan['reused']:
            assert hashlib.sha256(content).hexdigest() == plan['reused'][str(path)], path
        row = json.loads(content)
        key = (row['instance'], int(row['seed']), row['arm'])
        assert key not in records, key
        assert row['completed'] and 4999 <= row['real_objective_evaluation_count'] <= 5000, key
        assert row['hidden_candidate_objective_evaluations'] == 0, key
        assert {p['evaluation'] for p in row['budget_fronts']} == set(BUDGETS), key
        records[key] = row
    expected = {(e['instance_id'], seed, m) for e in plan['entries']
                for seed in settings['screen_seeds'] for m in methods}
    assert set(records) == expected, 'Run panel differs from 11 methods x 27 runs'
    saved_bounds = json.loads((root / 'final/INSTANCE_BOUNDS.json').read_bytes())
    recomputed = {}
    front_checks = []
    for entry in plan['entries']:
        name = entry['instance_id']
        runs = [r for (instance, _, _), r in records.items() if instance == name]
        points = [p for r in runs for s in r['budget_fronts'] for p in s['front_raw']]
        data = np.asarray(points, dtype=float).reshape(-1, 3)
        assert np.isfinite(data).all()
        low = data.min(0) if len(data) else np.zeros(3)
        high = data.max(0) if len(data) else np.zeros(3)
        bounds = saved_bounds[name]
        assert np.allclose(low, bounds['minimum'], rtol=0, atol=1e-12), name
        assert np.allclose(high, bounds['maximum'], rtol=0, atol=1e-12), name
        assert bounds['reference'] == [1.1] * 3, name
        span = np.where(high > low, high - low, 1.)
        for row in runs:
            seed, method = int(row['seed']), row['arm']
            base = records[name, seed, 'BASE']
            assert row['instance_digest'] == entry['instance_digest']
            assert row['scenario_digest'] == plan['scenarios'][f'{name}/{seed}']
            assert row['scenario_digest'] == base['scenario_digest'], (name, seed, method)
            assert row['instance_digest'] == base['instance_digest'], (name, seed, method)
            snapshots = {s['evaluation']: s for s in row['budget_fronts']}
            for budget, snap in snapshots.items():
                assert snap['actual_evaluations'] == budget or (budget == 5000 and snap['actual_evaluations'] == 4999)
                raw = np.asarray(snap['front_raw'], dtype=float).reshape(-1, 3)
                normal = (raw - low) / span
                assert np.isfinite(normal).all() and ((normal >= 0) & (normal <= 1)).all()
                value = hv3(normal)
                assert -1e-12 <= value <= 1.331 + 1e-12
                recomputed[name, seed, method, budget] = value
            a = {tuple(p) for p in snapshots[1000]['front_raw']}
            b = {tuple(p) for p in snapshots[5000]['front_raw']}
            front_checks.append(dict(instance=name, seed=seed, method=method,
                                     objectives_same_1000_5000=a == b,
                                     points_1000=len(a), points_5000=len(b)))
    with (root / 'final/PER_PAIR.csv').open(encoding='utf-8-sig', newline='') as stream:
        pairs = [r for r in csv.DictReader(stream) if r['reference_method'] == 'BASE']
    assert len(pairs) == 891 and len({(r['instance'], r['seed'], r['method'], r['evaluation']) for r in pairs}) == 891
    max_error = 0.
    by = defaultdict(list)
    for row in pairs:
        name, seed, method, budget = row['instance'], int(row['seed']), row['method'], int(row['evaluation'])
        own = recomputed[name, seed, method, budget]
        base = recomputed[name, seed, 'BASE', budget]
        max_error = max(max_error, abs(own-float(row['own'])), abs(base-float(row['reference'])))
        assert row['configuration'] == records[name, seed, method]['configuration']
        by[method, budget].append(dict(configuration=row['configuration'], own=own, base=base, difference=own-base))
    assert max_error <= 1e-10, f'Recomputed HV differs from PER_PAIR.csv by {max_error}'
    out = root / 'record_audit'
    out.mkdir(exist_ok=True)
    summaries, wtl = [], []
    for budget in BUDGETS:
        for method in methods:
            rows = by[method, budget]
            summaries.append(dict(method=method, evaluation=budget, pairs=len(rows),
                                  mean_hv=mean(r['own'] for r in rows),
                                  base_mean_hv=mean(r['base'] for r in rows),
                                  mean_difference=mean(r['difference'] for r in rows)))
        for cfg in [f'S{i}' for i in range(9)]:
            line = dict(configuration=cfg, evaluation=budget)
            for method in methods:
                values = [r['difference'] for r in by[method, budget] if r['configuration'] == cfg]
                assert len(values) == 3
                wins = sum(v > 1e-12 for v in values)
                ties = sum(abs(v) <= 1e-12 for v in values)
                losses = sum(v < -1e-12 for v in values)
                line[method] = '—' if method == 'BASE' else f'{wins}/{ties}/{losses}'
            wtl.append(line)
    write_csv(out / 'MEAN_HV.csv', summaries)
    write_csv(out / 'WTL_BY_CONFIGURATION.csv', wtl)
    write_csv(out / 'FRONT_COMPARISON.csv', front_checks)
    print('All 297 saved runs checked. Maximum HV recomputation error:', max_error, flush=True)
    print('I_RULE: evaluation, Base mean HV, I mean HV, mean difference')
    for row in summaries:
        if row['method'] == 'I_RULE':
            print(row['evaluation'], row['base_mean_hv'], row['mean_hv'], row['mean_difference'])
    for row in wtl:
        if row['evaluation'] == 5000:
            print(row)
    (out / 'TABLE_NOTE.txt').write_text(
        'W/T/L compares each method with Base on the same instance, algorithm seed and evaluation checkpoint. '
        'Win: HV difference > 1e-12. Tie: absolute HV difference <= 1e-12. Loss: difference < -1e-12. '
        'Each configuration contains one instance and three algorithm seeds. Base is displayed as an em dash. '
        'HV uses per-instance bounds pooled over all 11 methods, three seeds and three checkpoints, '
        'with reference (1.1, 1.1, 1.1). Intermediate records contain objective points; '
        'chromosomes and per-scenario objectives cannot be reconstructed from these records.\n', encoding='utf-8')
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print('Plot requires matplotlib. All numerical CSV reports are ready:', out)
        return
    fig, ax = plt.subplots(figsize=(10, 6))
    for method in methods:
        ax.plot(range(3), [mean(r['own'] for r in by[method, b]) for b in BUDGETS],
                marker='o', label=method, linewidth=1.5)
    ax.set_xticks(range(3), ['400', '1,000', '5,000'])
    ax.set_xlabel('Solution evaluations (equally spaced checkpoints)')
    ax.set_ylabel('Mean hypervolume')
    ax.grid(alpha=.2)
    ax.legend(fontsize=8, loc='upper left', bbox_to_anchor=(1, 1))
    fig.tight_layout()
    for suffix in ('png', 'pdf'):
        fig.savefig(out / f'MEAN_HV.{suffix}', dpi=220)
    plt.close(fig)
    print('Reports and plot:', out)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parent /
                        'transformer_ppo_runs/diagnostics/ccp_icm_single_member_panel_b5000_v1')
    audit(parser.parse_args().root)
