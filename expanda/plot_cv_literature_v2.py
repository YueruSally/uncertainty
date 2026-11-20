#!/usr/bin/env python3
"""Plot sparse literature CV evidence against the frozen audited settings.

No invented observations, histogram fitting, transport evaluations or changes
to model parameters. Requires the completed cv_evidence_audit_v1 AUDIT.json.
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
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '1'

from cv_evidence_audit_v1 import distribution, ln_kl

SOURCES = {
    'Kwon1995': {
        'url': 'https://onlinepubs.trb.org/Onlinepubs/trr/1995/1489/1489-001.pdf',
        'location': 'Table 8, printed p. 7; 1990-91 data',
        'scope': 'North American double-stack ramp-to-ramp long-haul service; not China-Europe calibration. CV calculated from reported mean and SD. Right skew does not establish lognormality.',
    },
    'CAREC2023': {
        'url': 'https://cpmm.carecprogram.org/uploads/CPMM-Annual-Report-2023_4th.pdf',
        'location': 'Tables 4.1/4.2, printed p.12; Tables 4.5/4.6, printed p.14',
        'scope': '2023 BCP data, published 2025. Selected most time-consuming BCP tables, not a representative sample of all BCPs. CV inferred under a two-parameter lognormal assumption from mean and median.',
    },
    'SeaIntelligence2026': {
        'url': 'https://www.sea-intelligence.com/images/press_docs/ss757/20260324_-_Sea-Intelligence_Sunday_Spotlight_757_Press_Release.pdf',
        'location': 'Sunday Spotlight 757 press release, Mar 25, 2026',
        'scope': 'Asia-Europe network transit-time CV range after the Cape routing shift; includes network scheduling effects. Not a probability distribution, confidence interval or direct single-arc calibration.',
    },
    'SongWang2021': {
        'url': 'https://doi.org/10.1111/rssa.12627',
        'scope': 'Erlang results concern liner arrival and handling time deviations, not a direct fit of absolute sailing time. No CV observations extracted here.',
    },
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def evidence(audit):
    rows = []

    def add(group, label, source, kind, mean=None, sd=None, median=None,
            cv=None, low=None, high=None, unit='', note=''):
        if kind == 'sd_over_mean':
            cv = sd / mean
        elif kind == 'lognormal_implied':
            if mean <= median:
                cv = None
                note = ('Mean below median: cannot exactly match both summaries '
                        'with an ordinary two-parameter lognormal. This alone '
                        'does not prove left skew or reject a population family.')
            else:
                cv = math.sqrt((mean / median)**2 - 1)
        family, mode = group.split('_')
        profile = audit['profile']
        cfg = profile['mode_time_cv' if family == 'travel' else 'border_delay_cv'][mode]
        cap = (profile['mode_time_max_factor'][mode] if family == 'travel'
               else profile['border_delay_max_factor'])
        active = family == 'travel' or audit['border_nodes'][mode]['positive_nodes'] > 0
        current = distribution(cfg, cap)
        rows.append(dict(source_group=group, label=label, source=source,
                         evidence_kind=kind, mean=mean, sd=sd, median=median,
                         time_unit=unit, literature_cv=cv,
                         literature_cv_low=low, literature_cv_high=high,
                         configured_model_cv=cfg,
                         effective_model_cv=current['actual_multiplier_cv'] if active else None,
                         model_cap_factor=cap, model_active=active, note=note))

    for i, cv in enumerate((.151, .141, .195, .139, .167, .114), 1):
        add('travel_road', f'Distance group {i}', 'Zhao2011', 'published_group_cv', cv=cv,
            note='Six published distance-group summaries, not raw route-level observations.')
    add('travel_rail', 'Eastbound long-haul', 'Kwon1995', 'sd_over_mean',
        mean=70.8, sd=10.9, unit='h')
    add('travel_rail', 'Westbound long-haul', 'Kwon1995', 'sd_over_mean',
        mean=67.4, sd=10.7, unit='h')
    add('travel_water', 'Asia-Europe range', 'SeaIntelligence2026', 'published_range',
        low=.25, high=.35, note='Reported range, not a confidence interval; no midpoint or pseudo-samples generated.')
    for label, mean, median in (
            ('Alashankou OUT', 37.2, 27.8), ('Khorgos OUT', 51.4, 31.7),
            ('Dostyk IN', 22.2, 12.8), ('Nur Zholy IN', 16.5, 15.9)):
        add('border_road', label, 'CAREC2023', 'lognormal_implied',
            mean=mean, median=median, unit='h')
    add('border_road', 'BOTA (TTI)', 'TTI2011', 'sd_over_mean',
        mean=49.5, sd=28.9, unit='min', note='Different border and time period; reported data exclude times above 120 min. P90 below is reconstructed, not empirical.')
    for label, mean, median in (
            ('Alashankou OUT', 17.5, 19.5), ('Khorgos OUT', 12.5, 6.5),
            ('Altynkol IN', 76.9, 66.0), ('Dostyk IN', 74.3, 73.1)):
        add('border_rail', label, 'CAREC2023', 'lognormal_implied',
            mean=mean, median=median, unit='h')
    sea = audit['sea_reference']
    add('border_water', 'Callao customs', 'Carballo2016', 'lognormal_implied',
        mean=sea['mean_days'], median=sea['median_days'], unit='day',
        note='Customs delay, not total port dwell; current sea-border means are zero, so the model CV is inactive.')
    return rows


def comparisons(rows):
    result = []
    for r in rows:
        base = {k: r[k] for k in ('source_group', 'label', 'source', 'evidence_kind', 'time_unit')}
        base.update(reference_mean=r['mean'], reference_cv=r['literature_cv'],
                    model_configured_cv=r['configured_model_cv'],
                    model_effective_cv=r['effective_model_cv'],
                    reference_p90_lognormal=None, model_p90_same_mean=None,
                    signed_p90_error_percent=None, absolute_p90_error_percent=None,
                    auxiliary_uncapped_KL_model_to_reference=None,
                    auxiliary_uncapped_KL_reference_to_model=None,
                    status='', qualification='')
        if not r['model_active']:
            base['status'] = 'not_comparable_inactive_model'
        elif r['literature_cv'] is None:
            base['status'] = ('reported_range_only' if r['evidence_kind'] == 'published_range'
                              else 'no_exact_lognormal_reconstruction')
        elif r['mean'] is None:
            base['status'] = 'cv_only_no_time_moments'
        else:
            mean = r['mean']
            ref = distribution(r['literature_cv'])
            model = distribution(r['configured_model_cv'], r['model_cap_factor'])
            proxy = distribution(r['configured_model_cv'])
            qr = mean * ref['p90_multiplier']
            qm = mean * model['p90_multiplier']
            base.update(reference_p90_lognormal=qr, model_p90_same_mean=qm,
                        signed_p90_error_percent=100*(qm/qr-1),
                        absolute_p90_error_percent=100*abs(qm/qr-1),
                        auxiliary_uncapped_KL_model_to_reference=ln_kl(proxy, ref),
                        auxiliary_uncapped_KL_reference_to_model=ln_kl(ref, proxy),
                        status='conditional_shape_comparison',
                        qualification='Reference assumes ordinary lognormal; P90 not observed. Both distributions use the literature mean, not the deployed route mean. KL uses uncapped mean-matched LN proxies in nats, not CV-histogram KL. For a capped model versus an uncapped continuous reference, strict KL is infinite in both directions.')
        result.append(base)
    return result


def write_csv(path, rows):
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def plots(out, rows, comp):
    os.environ['MPLCONFIGDIR'] = str(out/'.matplotlib')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'path'})
    blue, orange, grey = '#4166ff', '#e88817', '#6c7480'

    def save(fig, name):
        for ext in ('png', 'pdf', 'svg'):
            fig.savefig(out/(name+'.'+ext), dpi=300, facecolor='white', bbox_inches='tight')
        plt.close(fig)
        print('[CV-LITERATURE] saved', name, flush=True)

    groups = ['travel_road', 'travel_rail', 'travel_water', 'border_road', 'border_rail', 'border_water']
    titles = ['Road travel', 'Rail travel', 'Sea travel', 'Road border', 'Rail border', 'Sea border']
    notes = ['Zhao (2011): distance-group summaries', 'Kwon et al. (1995): North American freight',
             'Sea-Intelligence (2026): network transit', 'CAREC (2023); TTI (2011) shown separately',
             'CAREC (2023): LN-implied values', 'Carballo (2016): customs, not full port dwell']
    xmax = [.25, .25, .42, 1.7, 1.95, 2.85]
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 8.8))
    fig.subplots_adjust(left=.13, right=.97, bottom=.20, top=.83, wspace=.78, hspace=.86)
    for panel, (ax, group, title, note, limit) in enumerate(zip(axes.flat, groups, titles, notes, xmax)):
        part = [r for r in rows if r['source_group'] == group]
        for i, r in enumerate(part):
            cv = r['literature_cv']
            if r['evidence_kind'] == 'published_range':
                lo, hi = r['literature_cv_low'], r['literature_cv_high']
                ax.plot([lo, hi], [i, i], color=orange, lw=3, solid_capstyle='butt')
                ax.plot([lo, hi], [i, i], '|', color=orange, ms=11)
                ax.text((lo+hi)/2, i-.3, f'{lo:.2f}–{hi:.2f}', ha='center', color=orange, fontsize=9)
            elif cv is None:
                ax.text(.03*limit, i, 'Not reconstructed: mean < median', va='center', fontsize=8, color=grey,
                        bbox=dict(facecolor='white', edgecolor='none', alpha=.9, pad=1))
            else:
                marker = 'D' if r['evidence_kind'] == 'lognormal_implied' else 'o'
                ax.plot(cv, i, marker=marker, color=orange, ms=6)
                ax.text(cv+.035*limit, i, f'{cv:.3f}', va='center', fontsize=9, color='#985b15')
        r = part[0]
        effective = r['effective_model_cv']
        active = r['model_active']
        v = effective if active else r['configured_model_cv']
        ax.axvline(v, color=blue if active else grey, ls='--', lw=1.5)
        label = f'Model CV = {v:.3f}' if active else f'Configured CV = {v:.2f}; INACTIVE'
        if r['model_cap_factor'] is not None and active:
            label += f" (input {r['configured_model_cv']:.2f})"
        ax.text(0, -.27, label, transform=ax.transAxes, color=blue if active else grey, fontsize=9)
        ax.set(yticks=list(range(len(part))), yticklabels=[r['label'] for r in part],
               xlim=(0, limit), ylim=(len(part)-.5, -.6), xlabel='Coefficient of variation (CV)')
        ax.tick_params(axis='y', length=0, labelsize=9)
        ax.grid(axis='x', alpha=.16)
        ax.set_axisbelow(True)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_title(f'({chr(97+panel)}) {title}', loc='left', pad=35, fontsize=13)
        ax.text(0, 1.07, note, transform=ax.transAxes, fontsize=8, color=grey)
    handles = [Line2D([], [], color=orange, marker='o', ls='', label='Reported CV or SD / mean'),
               Line2D([], [], color=orange, marker='D', ls='', label='Lognormal-implied CV'),
               Line2D([], [], color=orange, lw=3, label='Published range (not a CI)'),
               Line2D([], [], color=blue, ls='--', label='Current model CV')]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.53, .935), ncol=4, frameon=False, fontsize=9)
    fig.suptitle('Literature CV evidence and current model settings', y=.995, fontsize=17)
    fig.text(.06, .055, 'Points represent published summaries, not raw observations; horizontal scales differ by panel.\n'
             'Lognormal-implied CV = sqrt[(mean / median)^2 - 1]. No CV histogram is inferred from these sparse summaries.\n'
             'Border multipliers are capped at 3.5 and mean-calibrated. Sea-border means are currently zero.\n'
             'External route / port statistics provide contextual evidence, not direct calibration of every modeled arc.',
             fontsize=9, color=grey, linespacing=1.5)
    save(fig, 'CV_LITERATURE_MODEL_COMPARISON')

    valid = [r for r in comp if r['status'] == 'conditional_shape_comparison']
    fig, axes = plt.subplots(1, 2, figsize=(13, 7.6), sharey=True)
    fig.subplots_adjust(left=.25, right=.97, bottom=.22, top=.82, wspace=.24)
    for i, r in enumerate(valid):
        val = r['signed_p90_error_percent']
        axes[0].barh(i, val, color=blue, height=.58)
        axes[0].text(val+(1 if val >= 0 else -1), i, f'{val:+.2f}%',
                     ha='left' if val >= 0 else 'right', va='center', fontsize=9)
        axes[1].plot(r['auxiliary_uncapped_KL_model_to_reference'], i-.1, 'o', color=blue)
        axes[1].plot(r['auxiliary_uncapped_KL_reference_to_model'], i+.1, 'D', color=orange)
    labels = [('Rail travel: ' if r['source_group'] == 'travel_rail' else
               'Road border: ' if r['source_group'] == 'border_road' else 'Rail border: ')+r['label'] for r in valid]
    axes[0].set(yticks=list(range(len(valid))), yticklabels=labels,
                ylim=(len(valid)-.5, -.5), xlabel='Signed P90 difference (%)')
    vals = [r['signed_p90_error_percent'] for r in valid]
    axes[0].set_xlim(min(min(vals)-12, -12), max(max(vals)+12, 12))
    axes[1].set_xlabel('Auxiliary uncapped lognormal KL (nats)')
    axes[0].set_title('(a) Same mean; current code shape vs LN reference', fontsize=11, pad=15)
    axes[1].set_title('(b) Direction matters; uncapped proxies only', fontsize=11, pad=15)
    fig.legend(handles=[Line2D([], [], color=blue, marker='o', ls='', label='Model to reference'),
                        Line2D([], [], color=orange, marker='D', ls='', label='Reference to model')],
               loc='upper center', bbox_to_anchor=(.81, .91), ncol=2, fontsize=9, frameon=False)
    for ax in axes:
        ax.axvline(0, color=grey, lw=.8)
        ax.grid(axis='x', alpha=.15)
        ax.set_axisbelow(True)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(axis='y', length=0, labelsize=9)
    fig.suptitle('Conditional P90 and KL comparisons from literature summaries', y=.96, fontsize=15)
    fig.text(.05, .055, 'Reference P90 values are lognormal reconstructions, not empirical quantiles. Comparisons hold the literature mean fixed.\n'
             'For border times, P90 uses the actual capped model form; finite KL uses UNCAPPED mean-matched proxies.\n'
             'These are time-distribution comparisons, not KL between distributions of observed CV values. No pass threshold is applied.\n'
             'Alashankou rail is not reconstructed; sea-border model is inactive; sea-travel evidence supplies only a range.',
             fontsize=9, color=grey, linespacing=1.5)
    save(fig, 'CV_CONDITIONAL_P90_KL')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path(__file__).resolve().parent /
                   'transformer_ppo_runs/diagnostics/ccp_cv_evidence_audit_v1')
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    root, out = a.root.resolve(), a.out.resolve()
    if root == out or root in out.parents or out in root.parents:
        raise ValueError('Use a separate output directory outside the original audit.')
    source = root/'AUDIT.json'
    data = json.loads(source.read_text(encoding='utf-8-sig'))
    if not data.get('completed'):
        raise ValueError('The source audit is incomplete.')
    inputs = dict(source=str(source), source_sha256=sha(source), script_sha256=sha(Path(__file__)),
                  analytic_helper_sha256=sha(Path(__file__).with_name('cv_evidence_audit_v1.py')),
                  new_searches=0, new_objective_evaluations=0, training_updates=0,
                  sensitivity_experiments=0, parameters_changed=False)
    out.mkdir(parents=True, exist_ok=True)
    lock = out/'PLOT_INPUTS.json'
    if lock.exists() and json.loads(lock.read_text()) != inputs:
        raise ValueError('Inputs or script changed; select a fresh --out directory.')
    if not lock.exists() and any(out.iterdir()):
        raise ValueError('Output contains unrelated files; select a fresh --out directory.')
    lock.write_text(json.dumps(inputs, indent=2)+'\n')
    rows = evidence(data)
    comp = comparisons(rows)
    write_csv(out/'LITERATURE_CV.csv', rows)
    write_csv(out/'CONDITIONAL_P90_KL.csv', comp)
    (out/'SOURCES.json').write_text(json.dumps({**data['sources'], **SOURCES}, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    (out/'CAPTIONS_CN.txt').write_text(
        '图1：六类来源的文献CV证据与当前代码参数。圆点为文献CV或标准差/均值；菱形为假设普通对数正态后由均值/中位数反推；线段仅为文献范围，不是置信区间。各面板横轴范围不同。海运口岸均值为零，当前没有有效CV。\n'
        '图2：统一为各文献均值后的条件P90及辅助KL比较。参考P90均为对数正态重建值，不是实测分位数；公路/铁路口岸模型P90采用现有封顶、均值校准形式。有限KL仅为两个未封顶、同均值的辅助对数正态之间的KL，不是CV分布KL，也不是实际封顶模型与连续参考之间的KL。\n'
        '阿拉山口铁路均值小于中位数，无法精确匹配普通对数正态的这两个汇总统计量；仅据此不能断言总体左偏。海运CV区间没有生成中点或虚拟观测。\n'
        'CAREC各表挑选的是耗时较长的口岸，不代表所有口岸；进出境方向、年份和时间定义均应保留。Kwon为北美集装箱长途运输；Sea-Intelligence为2026年绕航背景下的亚欧网络时间，均不是当前每条运输弧的直接标定。\n'
        'Lessan的log-logistic结论来自客运高铁；Song和Wang的Erlang结论针对到达/作业时间偏差，不能直接替代绝对海运时间分布。\n'
        '以上图表补充参数依据，不等于证明当前所有CV准确。没有生成经验CV直方图、修改CV、执行敏感性实验或重跑搜索。\n', encoding='utf-8')
    plots(out, rows, comp)
    if sha(source) != inputs['source_sha256']:
        raise RuntimeError('Original audit changed during plotting.')
    (out/'STATUS.json').write_text(json.dumps(dict(completed=True, **inputs), indent=2)+'\n')
    print('[CV-LITERATURE] COMPLETE', out, flush=True)
    for r in comp:
        if r['status'] == 'conditional_shape_comparison':
            print(f"{r['source_group']} / {r['label']}: refCV={r['reference_cv']:.4f}, "
                  f"P90 error={r['signed_p90_error_percent']:+.2f}%, "
                  f"auxKL(model/ref)={r['auxiliary_uncapped_KL_model_to_reference']:.5f}")


if __name__ == '__main__':
    main()
