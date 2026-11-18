"""Read-only diagnosis of saved paired HV results; no search or model changes."""
import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean, median


def main(root):
    source = root / 'final/PER_PAIR.csv'
    with source.open(encoding='utf-8-sig', newline='') as f:
        rows = [r for r in csv.DictReader(f) if r['reference_method'] == 'BASE']
    data = {}
    configs = {}
    for r in rows:
        key = (r['instance'], int(r['seed']), r['method'], int(r['evaluation']))
        if key in data:
            raise ValueError(f'Duplicate pair: {key}')
        data[key] = (float(r['own']), float(r['reference']))
        configs[r['instance']] = r['configuration']
    if len(data) != 891:
        raise ValueError(f'Expected 891 checkpoint rows, found {len(data)}')
    comparisons = [(m, 'BASE') for m in sorted({k[2] for k in data}) if m != 'BASE']
    comparisons += [('ICM_TERMINAL_PPO', 'ICM_SHARED_SUPERVISED')]
    comparisons += [(f'{c}_LEARN', f'{c}_RULE') for c in 'ICM']
    details = []
    for own, ref in comparisons:
        for instance, seed, method, budget in sorted(data):
            if method != own:
                continue
            a, base = data[instance, seed, own, budget]
            b, base_ref = data[instance, seed, ref, budget]
            if abs(base - base_ref) > 1e-12:
                raise ValueError('Mismatched Base reference')
            delta = a-b
            details.append(dict(configuration=configs[instance], instance=instance,
                seed=seed, evaluation=budget, method=own, reference=ref,
                own_hv=a, reference_hv=b, difference=delta,
                percent=100*delta/b if b > 0 else '',
                W_T_L='W' if delta > 1e-12 else 'L' if delta < -1e-12 else 'T'))
    out = root / 'loss_audit'
    out.mkdir(exist_ok=True)
    def save(name, values):
        with (out/name).open('w', encoding='utf-8-sig', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(values[0]))
            w.writeheader(); w.writerows(values)
    save('ALL_PAIRED_DETAILS.csv', details)
    losses = sorted([r for r in details if r['evaluation'] == 5000 and r['W_T_L'] == 'L'],
                    key=lambda r: r['difference'])
    if losses:
        save('LOSSES_5000.csv', losses)
    grouped = defaultdict(list)
    for r in details:
        grouped[r['configuration'], r['method'], r['reference'], r['evaluation']].append(r)
    summary = []
    for (cfg, method, ref, budget), rr in sorted(grouped.items()):
        dd = [r['difference'] for r in rr]
        gg = [r['percent'] for r in rr if r['percent'] != '']
        summary.append(dict(configuration=cfg, method=method, reference=ref, evaluation=budget,
            pairs=len(rr), W_T_L='/'.join(str(sum(r['W_T_L'] == c for r in rr)) for c in 'WTL'),
            mean_difference=mean(dd), median_difference=median(dd),
            mean_percent=mean(gg) if len(gg)==len(rr) else '',
            min_difference=min(dd), max_difference=max(dd)))
    save('BY_CONFIGURATION.csv', summary)
    report = ['# Saved-run loss audit', '',
        f'Source: `{source}`', '',
        'No new searches. HV values are read from the final paired report. ',
        'W/T/L: difference > 1e-12 / absolute difference <= 1e-12 / difference < -1e-12.',
        'Each configuration has one instance and three seeds. These are descriptive checks, not causal conclusions.', '',
        '## All 5000-evaluation comparisons', '',
        '| Method | Reference | W/T/L | Mean gain (%) | Median gain (%) | Mean gain in wins (%) | Mean gain in losses (%) |',
        '|---|---|---|---:|---:|---:|---:|']
    def fmt(values):
        return f'{mean(values):+.4f}' if values else '—'
    for own, ref in comparisons:
        rr = [r for r in details if r['method']==own and r['reference']==ref and r['evaluation']==5000]
        gains = [r['percent'] for r in rr if r['percent'] != '']
        counts = '/'.join(str(sum(r['W_T_L']==c for r in rr)) for c in 'WTL')
        win = [r['percent'] for r in rr if r['W_T_L']=='W' and r['percent']!='']
        lose = [r['percent'] for r in rr if r['W_T_L']=='L' and r['percent']!='']
        avg = fmt(gains) if len(gains)==len(rr) else 'undefined'
        med = f'{median(gains):+.4f}' if len(gains)==len(rr) else 'undefined'
        report.append(f'| {own} | {ref} | {counts} | {avg} | {med} | {fmt(win)} | {fmt(lose)} |')
    report += ['', '## Priority checks: S0 combinations, S3 PPO, S5 crossover', '',
        '| Config | Seed | Method | Reference | Gain at 400 (%) | Gain at 1000 (%) | Gain at 5000 (%) |',
        '|---|---|---|---|---:|---:|---:|']
    for own, ref in comparisons:
        for instance, seed in sorted({(k[0], k[1]) for k in data}):
            cfg = configs[instance]
            focus = (cfg=='S0' and own.startswith('ICM_') and ref=='BASE') or (cfg=='S3' and own=='ICM_TERMINAL_PPO') or (cfg=='S5' and own.startswith('C_'))
            if not focus:
                continue
            cells = []
            for budget in (400,1000,5000):
                a = data[instance,seed,own,budget][0]
                b = data[instance,seed,ref,budget][0]
                cells.append(f'{100*(a-b)/b:+.4f}' if b > 0 else 'undefined')
            report.append(f'| {cfg} | {seed} | {own} | {ref} | ' + ' | '.join(cells) + ' |')
    report += ['', '## Limits', '',
        'Inspect original front points and execution logs before attributing losses to feasibility, diversity, ranking, or generalization. ',
        'Final HV differences alone do not identify the faulty decision. Missing candidate traces require targeted replay, not invented explanations.']
    (out/'REPORT.md').write_text('\n'.join(report)+'\n', encoding='utf-8')
    print('Read', len(data), 'checkpoint rows; zero new evaluations.')
    print('Open in VS Code:', out/'REPORT.md')
    print('CSV details:', out)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path(__file__).resolve().parent /
                   'transformer_ppo_runs/diagnostics/ccp_icm_single_member_panel_b5000_v1')
    main(p.parse_args().root)
