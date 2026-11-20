#!/usr/bin/env python3
"""Independent parametric CV calibration experiment; not an empirical validation.

Published summaries define reconstructed lognormal reference cases. Fit a new,
uncapped positive sampler to each case on fixed calibration draws, then check it
on independent draws. Histograms show sampling distributions of estimated CVs,
not distributions of observed road/port CVs. Existing search inputs stay intact.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from pathlib import Path
from statistics import NormalDist

os.environ['MPLBACKEND'] = 'Agg'
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '1'

import numpy as np
from cv_evidence_audit_v1 import distribution
from plot_cv_literature_v2 import SOURCES, evidence, sha, write_csv

GROUPS = ['travel_road', 'travel_rail', 'travel_water',
          'border_road', 'border_rail', 'border_water']
TITLES = ['Road travel', 'Rail travel', 'Sea travel',
          'Road border', 'Rail border', 'Sea border']


def cases_from_audit(audit):
    cases, excluded = [], []
    for r in evidence(audit):
        if r['source'] == 'TTI2011':
            excluded.append(dict(source_group=r['source_group'], label=r['label'],
                                 reason='Kept in prior evidence report; not pooled with the selected China-Kazakhstan border cases.'))
            continue
        if r['evidence_kind'] == 'published_range':
            # Two stated endpoint benchmarks, NOT a uniform or fitted CV law.
            for side in ('low', 'high'):
                case = dict(r)
                case.update(label=f'Asia-Europe {side} endpoint',
                            literature_cv=r[f'literature_cv_{side}'],
                            evidence_kind='range_endpoint_design_case',
                            note='Two equal-weight endpoint benchmarks only; no empirical CV distribution or interpolated CV population is assumed.')
                cases.append(case)
        elif r['literature_cv'] is None:
            excluded.append(dict(source_group=r['source_group'], label=r['label'], reason=r['note']))
        else:
            cases.append(dict(r))
    for r in cases:
        r['benchmark_mean'] = r['mean'] if r['mean'] is not None else 1.
        r['benchmark_unit'] = r['time_unit'] or 'mean-normalized multiplier'
        r['case_id'] = r['source_group']+' / '+r['label']
        r['calibrated_family'] = 'uncapped lognormal (new isolated model)'
    return cases, excluded


def rng_for(seed, case_id, phase):
    token = int(hashlib.sha256(case_id.encode()).hexdigest()[:8], 16)
    return np.random.default_rng(np.random.SeedSequence([seed, token, phase]))


def fit_sampler(z, cv):
    """Fit sample CV on calibration z only; keep its sample mean exactly one.

    Positive lognormal reparameterization avoids negative times from linear
    spread expansion. Finite calibration error is exposed on independent draws.
    """
    def normalized(s):
        # Shift prevents overflow and does not change CV.
        a = s*z
        shift = float(a.max())
        x = np.exp(a-shift)
        return x, shift
    lo, hi = 0., 1.
    while True:
        x, _ = normalized(hi)
        if x.std(ddof=1)/x.mean() >= cv:
            break
        hi *= 2
        if hi > 32:
            raise ValueError('Target CV cannot be attained on the fixed calibration sample.')
    for _ in range(70):
        mid = (lo+hi)/2
        x, _ = normalized(mid)
        if x.std(ddof=1)/x.mean() < cv:
            lo = mid
        else:
            hi = mid
    sigma = (lo+hi)/2
    x, shift = normalized(sigma)
    mu = -shift-math.log(float(x.mean()))
    sample = np.exp(mu+sigma*z)
    return dict(mu=mu, sigma=sigma,
                calibration_sample_mean=float(sample.mean()),
                calibration_sample_cv=float(sample.std(ddof=1)/sample.mean()),
                population_mean=math.exp(mu+sigma*sigma/2),
                population_cv=math.sqrt(math.expm1(sigma*sigma)),
                population_p90=math.exp(mu+NormalDist().inv_cdf(.9)*sigma))


def generate_stats(rng, parameters, replicates, batch_size, cap=None):
    x = np.exp(parameters['mu']+parameters['sigma'] *
               rng.standard_normal((replicates, batch_size)))
    if cap is not None:
        x = np.minimum(x, cap)
    return dict(cv=x.std(axis=1, ddof=1)/x.mean(axis=1),
                p90=np.quantile(x, .9, axis=1, method='linear'),
                mean=x.mean(axis=1))


def kl_counts(p, q, alpha):
    p, q = p.astype(float)+alpha, q.astype(float)+alpha
    p, q = p/p.sum(), q/q.sum()
    mask = p > 0
    if np.any(q[mask] == 0):
        return math.inf
    return float(np.sum(p[mask]*np.log(p[mask]/q[mask])))


def make_plot(out, group, title, summary, counts, upper, batch_size):
    import matplotlib.pyplot as plt
    width = upper/20
    x = np.arange(21)*width + width/2
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 5.7), sharex=True, sharey=True)
    fig.subplots_adjust(left=.095, right=.975, top=.79, bottom=.27, wspace=.16)
    ymax = max(float(counts[k].max()/counts[k].sum()) for k in ('reference', 'calibrated'))*1.17
    for ax, arm, label in zip(axes, ('reference', 'calibrated'),
                              ('Reconstructed target CV', 'Calibrated simulation CV')):
        ax.bar(x, counts[arm]/counts[arm].sum(), width=width*.96,
               color='#5aa6d1', edgecolor='#222222', linewidth=.65)
        ax.set_title(label, fontsize=12, pad=10)
        ax.set(xlabel=f'Sample CV ({batch_size} realizations / batch)',
               ylabel='Probability', ylim=(0, ymax), xlim=(0, upper+width*1.15))
        ax.tick_params(direction='in', top=True, right=True, labelsize=9)
        ticks = np.linspace(0, upper, 5)
        ax.set_xticks(ticks)
        ax.set_xticklabels([f'{t:g}' for t in ticks])
        ax.text(.98, .97, f'Overflow bin: > {upper:g}', transform=ax.transAxes,
                ha='right', va='top', fontsize=8, color='#555555')
    axes[1].set_ylabel('')
    fig.suptitle(title+' — independent calibration check', fontsize=15, y=.98)
    fig.text(.5, .885, f"Smoothed KL(target || calibrated) = {summary['kl_target_to_calibrated_smoothed']:.4f} nats",
             ha='center', fontsize=11)
    n = summary['benchmark_cases']
    note = (f"{n} literature-derived case(s); {summary['batches_per_case']} independent batches per case and per side.\n"
            'Sampling distributions from reconstructed models; not observed across-road / across-port CV distributions.\n'
            'Same bins, axes and batch size; KL uses a fixed pseudocount of 0.5 per bin.\n'
            'Calibration and validation draws are independent. Small KL measures model matching, not empirical validity.')
    if group == 'travel_water':
        note += '\nSea travel uses only the two reported CV endpoints as benchmark cases.'
    if group == 'border_water':
        note += '\nNew customs-time case only; the sea-border source in the existing search model remains inactive.'
    fig.text(.095, .045, note, fontsize=8, color='#555555', linespacing=1.45)
    for ext in ('png', 'pdf', 'svg'):
        fig.savefig(out/(group.upper()+'_CV_MATCH.'+ext), dpi=300,
                    facecolor='white', bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parent /
                        'transformer_ppo_runs/diagnostics/ccp_cv_evidence_audit_v1')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=20261120)
    parser.add_argument('--calibration-size', type=int, default=20000)
    parser.add_argument('--replicates', type=int, default=1000)
    parser.add_argument('--batch-size', type=int, default=100)
    a = parser.parse_args()
    if a.calibration_size < 100 or a.replicates < 20 or a.batch_size < 10:
        raise ValueError('Use calibration-size >= 100, replicates >= 20, batch-size >= 10.')
    start = time.monotonic()
    root, out = a.root.resolve(), a.out.resolve()
    if root == out or root in out.parents or out in root.parents:
        raise ValueError('Output must be separate from the original audit.')
    if out.exists() and any(out.iterdir()):
        raise ValueError('Output already contains files; select a fresh --out directory.')
    source = root/'AUDIT.json'
    audit = json.loads(source.read_text(encoding='utf-8-sig'))
    if not audit.get('completed'):
        raise ValueError('Source audit incomplete.')
    cases, excluded = cases_from_audit(audit)
    hashes = {str(p): sha(p) for p in (source, Path(__file__),
              Path(__file__).with_name('plot_cv_literature_v2.py'),
              Path(__file__).with_name('cv_evidence_audit_v1.py'))}
    plan = dict(seed=a.seed, calibration_size=a.calibration_size,
                validation_batches_per_case=a.replicates, validation_batch_size=a.batch_size,
                cv_sd_ddof=1, quantile_method='linear', kl_direction='target || model',
                kl_log_base='natural', kl_pseudocount_per_bin=.5,
                histogram_bins='20 equal bins to predetermined upper bound plus an explicit overflow bin',
                equal_weight='literature benchmark cases within source; not traffic or population weights',
                reference='reconstructed ordinary lognormal, not raw empirical data',
                calibrated='new uncapped lognormal; fitted on calibration draws only',
                current='unchanged parameters read from original audit; capped border law retained',
                histogram_interpretation='distribution of sample CV estimates across Monte Carlo batches, not heterogeneous empirical entity CVs',
                cap_change='new calibrated sampler removes original border cap; no old search or model inputs modified',
                new_searches=0, new_transport_objective_evaluations=0, training_updates=0,
                sensitivity_sweep=False, original_parameters_changed=False,
                input_sha256=hashes, cases=cases, excluded=excluded)
    out.mkdir(parents=True)
    os.environ['MPLCONFIGDIR'] = str(out/'.matplotlib')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'path'})
    (out/'PLAN.json').write_text(json.dumps(plan, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    write_csv(out/'EXCLUDED_CASES.csv', excluded)
    (out/'SOURCES.json').write_text(json.dumps({**audit['sources'], **SOURCES}, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    case_rows, parameters, stats = [], {}, {}
    for case in cases:
        key, cv = case['case_id'], case['literature_cv']
        reference = distribution(cv)
        calibrated = fit_sampler(rng_for(a.seed, key, 0).standard_normal(a.calibration_size), cv)
        parameters[key] = calibrated
        current = distribution(case['configured_model_cv'], case['model_cap_factor'])
        arms = {'reference': reference, 'calibrated': calibrated}
        if case['model_active']:
            arms['current'] = current
        stats[key] = {}
        for phase, (arm, model) in enumerate(arms.items(), 1):
            sample = generate_stats(rng_for(a.seed, key, phase), model, a.replicates,
                                    a.batch_size, case['model_cap_factor'] if arm == 'current' else None)
            stats[key][arm] = sample
            popmean = (1. if arm in ('reference', 'current') else model['population_mean'])
            popcv = (model['actual_multiplier_cv'] if arm in ('reference', 'current') else model['population_cv'])
            popq = model['p90_multiplier'] if arm in ('reference', 'current') else model['population_p90']
            case_rows.append(dict(source_group=case['source_group'], case=case['label'], arm=arm,
                evidence_kind=case['evidence_kind'], unit=case['benchmark_unit'],
                reference_mean=case['benchmark_mean'], target_population_cv=cv,
                model_population_cv=popcv, model_population_mean=popmean*case['benchmark_mean'],
                population_p90=popq*case['benchmark_mean'],
                population_p90_difference_percent=100*(popq/reference['p90_multiplier']-1),
                validation_mean_sample_cv=float(sample['cv'].mean()),
                validation_sd_sample_cv=float(sample['cv'].std(ddof=1)),
                validation_mean_sample_p90=float(sample['p90'].mean())*case['benchmark_mean'],
                validation_mean_abs_p90_error_percent=float(np.mean(abs(sample['p90']/reference['p90_multiplier']-1)))*100,
                calibration_mean=calibrated['calibration_sample_mean'] if arm=='calibrated' else '',
                calibration_cv=calibrated['calibration_sample_cv'] if arm=='calibrated' else '',
                n_batches=a.replicates, batch_size=a.batch_size))
        print('[CV-CALIBRATION]', key, f'target CV={cv:.4f}; calibrated population CV={calibrated["population_cv"]:.4f}', flush=True)
    write_csv(out/'CASE_RESULTS.csv', case_rows)
    (out/'CALIBRATED_PARAMETERS.json').write_text(json.dumps(parameters, indent=2)+'\n')
    arrays = {f'case{i}_{arm}_{metric}': values for i, case in enumerate(cases)
              for arm, sample in stats[case['case_id']].items()
              for metric, values in sample.items()}
    np.savez_compressed(out/'VALIDATION_BATCH_STATISTICS.npz', **arrays)
    summaries, histograms = [], []
    for group, title in zip(GROUPS, TITLES):
        part = [r for r in cases if r['source_group'] == group]
        upper = max(.4, 2*max(max(r['literature_cv'], r['configured_model_cv']) for r in part))
        edges = np.r_[np.linspace(0., upper, 21), np.inf]
        counts = {}
        for arm in ('reference', 'calibrated', 'current'):
            available = [stats[c['case_id']][arm]['cv'] for c in part if arm in stats[c['case_id']]]
            if available:
                counts[arm] = np.histogram(np.concatenate(available), bins=edges)[0]
                for i, count in enumerate(counts[arm]):
                    histograms.append(dict(source_group=group, arm=arm, bin_index=i,
                                           lower=edges[i], upper=edges[i+1], count=int(count),
                                           probability=float(count/counts[arm].sum())))
        row = dict(source_group=group, benchmark_cases=len(part), batches_per_case=a.replicates,
                   total_cv_values_per_side=int(counts['reference'].sum()),
                   kl_target_to_calibrated_smoothed=kl_counts(counts['reference'], counts['calibrated'], .5),
                   kl_target_to_calibrated_raw=kl_counts(counts['reference'], counts['calibrated'], 0),
                   kl_target_to_current_smoothed=kl_counts(counts['reference'], counts['current'], .5) if 'current' in counts else '',
                   kl_target_to_current_raw=kl_counts(counts['reference'], counts['current'], 0) if 'current' in counts else '',
                   reference_overflow_count=int(counts['reference'][-1]),
                   calibrated_overflow_count=int(counts['calibrated'][-1]),
                   calibrated_mean_abs_population_p90_difference_percent=float(np.mean([
                       abs(r['population_p90_difference_percent']) for r in case_rows
                       if r['source_group']==group and r['arm']=='calibrated'])))
        summaries.append(row)
        make_plot(out, group, title, row, counts, upper, a.batch_size)
    write_csv(out/'SOURCE_RESULTS.csv', summaries)
    write_csv(out/'HISTOGRAM_COUNTS.csv', histograms)
    notes = '''本轮是独立的参数化校准实验，不是实测分布验证，也不更新已有搜索结果。
左图：由文献均值/标准差或在对数正态假设下由均值/中位数建立参考模型，再独立采样得到批次CV。
右图：新建未封顶的正值对数正态采样器，在固定校准样本上匹配目标CV和均值，再使用独立随机数生成验证批次。
默认每个CV对应一批100次实现，标准差用ddof=1；默认每个文献案例每侧有1000批，实际参数见PLAN.json。重复批次是Monte Carlo重复，不是新增实测路段或口岸。
不同参考案例等权，只是本次设计权重，不是线路交通量权重。有限案例数没有因重复采样而增加。
六类来源分别绘图，不能将六个不同随机来源混成一条总体CV分布。海运运输只采用报告范围的两个端点，未假设区间内均匀分布。铁路阿拉山口无法按该方式重建，已列入排除表。
公路与铁路口岸原模型仍保留3.5倍均值的封顶，用于current对照；校准版取消封顶，属于新的独立假设。非负变量若上限是均值的3.5倍，CV至多sqrt(2.5)≈1.581；原封顶不能覆盖所有文献案例。
海运口岸只建立独立的Callao海关时间案例；现有搜索中该来源均值仍为0，未激活，因此没有current CV对照。海关时间不等于全港停留时间。
CV直方图是估计量的抽样分布，不等同于学长论文中各条边真实异质CV的分布。图片采用相同的左右面板、概率纵轴、公共分箱和坐标；溢出值单独成箱，不丢弃。
KL使用D_KL(target||model)、自然对数、每箱固定伪计数0.5；原始未平滑KL也保存在SOURCE_RESULTS.csv。相同统计层级才能比较KL，本轮数值不能直接与论文0.058判定优劣。
小KL说明新采样器能复现所设参考模型的CV行为；同一分布族按目标CV校准后接近是预期结果，不能据此证明对数正态适合真实数据或当前运输网络已完成实证标定。
CASE_RESULTS.csv另报目标/模型总体CV、总体P90差和验证批次P90绝对误差。P90参考是文献重建模型的分位数，不是实测分位数。CV匹配不保证真实尾部或相关结构匹配。
若后续把新校准参数接入优化，必须另存新版本并重新评价受影响的实验，不能把旧HV改称新参数下的结果。本轮不运行NSGA-II、训练或敏感性扫描。
'''
    (out/'README_CN.txt').write_text(notes, encoding='utf-8')
    if any(sha(Path(p)) != digest for p, digest in hashes.items()):
        raise RuntimeError('An input file changed while the calibration experiment ran.')
    status = dict(completed=True, elapsed_seconds=time.monotonic()-start, benchmark_cases=len(cases),
                  calibration_random_draws=len(cases)*a.calibration_size,
                  validation_random_draws=sum(len(v) for v in stats.values())*a.replicates*a.batch_size,
                  new_searches=0, new_transport_objective_evaluations=0, training_updates=0,
                  original_parameters_changed=False, simulated_reference=True, input_sha256=hashes)
    (out/'STATUS.json').write_text(json.dumps(status, indent=2)+'\n')
    print('[CV-CALIBRATION] COMPLETE', out, flush=True)
    for r in summaries:
        print(r['source_group'], 'current KL=', r['kl_target_to_current_smoothed'],
              'calibrated KL=', r['kl_target_to_calibrated_smoothed'], flush=True)


if __name__ == '__main__':
    main()
