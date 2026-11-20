#!/usr/bin/env python3
"""Show reference, existing-model and independently calibrated CVs together.

Replots saved histogram counts only: no new draws, fitting, searches or changes
to any model. Each CV represents a Monte Carlo batch, not an observed site.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path

os.environ['MPLBACKEND'] = 'Agg'

GROUPS = {'travel_rail': 'Rail travel', 'travel_water': 'Sea travel',
          'border_rail': 'Rail border'}
ARMS = ('reference', 'current', 'calibrated')


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    a = parser.parse_args()
    root, out = a.root.resolve(), a.out.resolve()
    if root == out or root in out.parents or out in root.parents:
        raise ValueError('Use a separate output directory.')
    if out.exists() and any(out.iterdir()):
        raise ValueError('Output is not empty; choose a fresh directory.')
    source_files = [root / name for name in (
        'STATUS.json', 'PLAN.json', 'HISTOGRAM_COUNTS.csv', 'CASE_RESULTS.csv')]
    hashes = {str(p): digest(p) for p in source_files}
    if not json.loads((root/'STATUS.json').read_text())['completed']:
        raise ValueError('The calibration experiment has not completed.')
    plan = json.loads((root/'PLAN.json').read_text())
    counts = read_csv(root/'HISTOGRAM_COUNTS.csv')
    cases = read_csv(root/'CASE_RESULTS.csv')
    out.mkdir(parents=True)
    os.environ['MPLCONFIGDIR'] = str(out/'.matplotlib')
    import numpy as np
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'pdf.fonttype': 42, 'ps.fonttype': 42,
                         'svg.fonttype': 'path'})
    batch_size = plan['validation_batch_size']
    details = {}
    for group, title in GROUPS.items():
        panels = {}
        for arm in ARMS:
            rows = sorted((r for r in counts if r['source_group'] == group
                           and r['arm'] == arm), key=lambda r: int(r['bin_index']))
            if len(rows) != 21:
                raise ValueError(f'{group}/{arm}: expected 21 saved bins.')
            values = np.array([int(r['count']) for r in rows])
            if (values < 0).any() or values.sum() <= 0:
                raise ValueError('Invalid saved histogram counts.')
            bounds = [(float(r['lower']), float(r['upper'])) for r in rows]
            probs = values / values.sum()
            if not np.allclose(probs, [float(r['probability']) for r in rows],
                               rtol=0, atol=1e-12):
                raise ValueError('Saved probabilities and counts disagree.')
            panels[arm] = dict(bounds=bounds, counts=values, probabilities=probs)
        if any(panels[k]['bounds'] != panels['reference']['bounds'] for k in ARMS):
            raise ValueError('The three arms must use identical saved bins.')
        upper = panels['reference']['bounds'][-1][0]
        width = upper / 20
        if not math.isinf(panels['reference']['bounds'][-1][1]):
            raise ValueError('Expected an explicit overflow bin.')
        x = (np.arange(21) + .5)*width
        ymax = max(panels[k]['probabilities'].max() for k in ARMS)*1.15
        fig, axes = plt.subplots(1, 3, figsize=(12.8, 5.2), sharex=True, sharey=True)
        fig.subplots_adjust(left=.07, right=.985, bottom=.29, top=.78, wspace=.13)
        titles = ['Reference reconstruction', 'Existing experiment model',
                  'Independent calibrated model']
        colors = ['#5aa6d1', '#8b939d', '#d99a4a']
        details[group] = {}
        for ax, arm, subtitle, color in zip(axes, ARMS, titles, colors):
            ax.bar(x, panels[arm]['probabilities'], width=.96*width,
                   color=color, edgecolor='#222222', linewidth=.6)
            subset = [r for r in cases if r['source_group'] == group and r['arm'] == arm]
            popcvs = sorted(set(float(r['model_population_cv']) for r in subset))
            cvtext = ', '.join(f'{v:.3f}' for v in popcvs)
            ax.set_title(subtitle+'\nPopulation CV: '+cvtext, fontsize=10.5, pad=10)
            ax.set(xlim=(0, upper+width*1.15), ylim=(0, ymax),
                   xlabel=f'Sample CV ({batch_size} realizations / batch)')
            ticks = np.linspace(0, upper, 5)
            ax.set_xticks(ticks)
            ax.set_xticklabels([f'{v:.2f}' for v in ticks])
            ax.tick_params(direction='in', top=True, right=True, labelsize=9)
            ax.text(.98, .97, f'Last bin: > {upper:.2f}', transform=ax.transAxes,
                    ha='right', va='top', fontsize=8, color='#555555')
            details[group][arm] = dict(population_cv=popcvs,
                histogram_counts=panels[arm]['counts'].tolist(),
                saved_batches=int(panels[arm]['counts'].sum()))
        axes[0].set_ylabel('Probability')
        fig.suptitle(title+' — parameter alignment audit', y=.97, fontsize=15)
        note = ('Identical saved batches and bins as the calibration report; no values shifted or resampled.\n'
                'These are sampling distributions of estimated CVs, not empirical distributions across routes or ports.\n'
                'The calibrated model is a separate experiment; it was not used to produce existing search results.')
        if group == 'travel_water':
            note += '\nReference cases use the two reported range endpoints; they do not establish two real-world modes.'
        if group == 'border_rail':
            note += ('\nReference CVs are lognormal-implied from port/direction summaries. Existing input CV = 0.50;'
                     '\nits capped-law population CV is 0.494. The independent calibration removes that cap.')
        fig.text(.07, .035, note, fontsize=8.5, color='#444444', linespacing=1.4)
        for ext in ('png', 'pdf', 'svg'):
            fig.savefig(out/(group.upper()+'_ALIGNMENT.'+ext), dpi=300,
                        bbox_inches='tight', facecolor='white')
        plt.close(fig)
        print('[CV-ALIGNMENT]', group, 'saved', flush=True)
    if any(digest(Path(p)) != h for p, h in hashes.items()):
        raise RuntimeError('A source report changed during plotting.')
    report = dict(completed=True, input_sha256=hashes,
                  new_draws=0, calibration_updates=0, new_searches=0,
                  experiment_parameters_changed=False, details=details)
    (out/'ALIGNMENT.json').write_text(json.dumps(report, indent=2)+'\n')
    (out/'README_CN.txt').write_text(
        '本报告只重画既有校准实验的三组统计数据，未修改任何样本、CV参数或搜索结果。\n'
        '三列依次为文献重建参考、原实验参数的采样器、独立校准采样器。\n'
        '图中的样本CV来自Monte Carlo批次，并非逐路段或逐口岸实测CV。\n'
        '海运两个峰来自人为选定的文献范围端点，不能解释为真实海运有两个峰。\n'
        '铁路口岸三个参考案例不可通过混合抽样直接替代同一个口岸的分布；'
        '若用于新实验，须先映射具体口岸、方向与时间过程，核对封顶设置。\n'
        '原实验的CV与文献重建CV不一致，说明尚未完成参数对齐；不能用第三列的匹配证明第二列正确。\n',
        encoding='utf-8')
    print('[CV-ALIGNMENT] COMPLETE', out)


if __name__ == '__main__':
    main()
