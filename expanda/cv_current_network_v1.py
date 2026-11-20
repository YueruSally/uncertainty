#!/usr/bin/env python3
"""Current-network CV audit: actual frozen sampler, 100 draws/entity, no fitting."""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time

os.environ['MPLBACKEND'] = 'Agg'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '1'
sys.dont_write_bytecode = True

import numpy as np
from cv_evidence_audit_v1 import SOURCES as OLD_SOURCES, read_profile, distribution
from plot_cv_literature_v2 import SOURCES, evidence, write_csv

GROUPS = {
    'travel_road': 'Road transport', 'travel_rail': 'Rail transport',
    'travel_water': 'Sea transport', 'border_road': 'Road border',
    'border_rail': 'Rail border', 'border_water': 'Sea border',
}
EPSILON = 0.001
REALIZATIONS = 100


def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def save_json(p, data):
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def load_engine(path):
    spec = importlib.util.spec_from_file_location('cv_frozen_main_engine', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def literature(profile, groups):
    """Keep original evidence, adding verified carrier rows without double counting."""
    audit = dict(profile=profile, sea_reference=dict(mean_days=5.2, median_days=2),
                 border_nodes={mode: dict(positive_nodes=len(groups['border_'+mode]))
                               for mode in ('road', 'rail', 'water')})
    rows = evidence(audit)
    for r in rows:
        r['histogram_included'] = r['literature_cv'] is not None
        r['location'] = {**OLD_SOURCES, **SOURCES}[r['source']].get('location', '')
        if r['source'] == 'Zhao2011':
            r['location'] = 'Dissertation Table 6.2, printed p.79'
        if r['source'] == 'TTI2011':
            r['location'] = 'Table V-1, printed p.52'
        if r['source'] == 'Carballo2016':
            r['location'] = 'Table 1, 2013 Customs Delay'
        if r['source_group'] == 'travel_rail':
            r['histogram_included'] = False
            r['note'] += ' Context only: Table 8 aggregate, not pooled with Table 9 carrier rows.'
    template = next(r for r in rows if r['source_group'] == 'travel_rail')
    for carrier, direction, mean_h, sd_h in (
            ('K', 'E', 66.0, 4.9), ('K', 'W', 71.4, 16.4),
            ('L', 'E', 39.4, 7.6), ('L', 'W', 38.0, 7.7),
            ('M', 'E', 99.4, 25.1), ('M', 'W', 82.7, 22.4)):
        r = dict(template)
        r.update(label=f'Carrier {carrier}, {direction}bound', mean=mean_h, sd=sd_h,
                 literature_cv=sd_h/mean_h, histogram_included=True,
                 location='Table 9, printed p.8',
                 note='SD/mean of published carrier-direction summary; equal weight per row. '
                      'North American double-stack service, not a local single-arc calibration.')
        rows.append(r)
    return rows


def histogram_metrics(reference, model, width, minimum_upper=0):
    top = max([width, minimum_upper] + list(reference) + list(model))
    edges = np.arange(math.ceil(top/width)+2, dtype=float)*width
    p, q = (np.histogram(values, bins=edges)[0] for values in (reference, model))
    result = dict(bin_width=width, bins=len(p), reference_count=len(reference),
                  model_count=len(model), smoothing_mass=EPSILON,
                  kl_reference_to_model_raw='', kl_reference_to_model_smoothed='',
                  kl_model_to_reference_raw='', kl_model_to_reference_smoothed='')
    if len(reference) and len(model):
        p0, q0 = p/p.sum(), q/q.sum()
        ps, qs = ((1-EPSILON)*v+EPSILON/len(p) for v in (p0, q0))
        def kl(a, b):
            mask = a > 0
            return float('inf') if np.any(b[mask] == 0) else float(np.sum(a[mask]*np.log(a[mask]/b[mask])))
        for label, a, b, aa, bb in (
                ('reference_to_model', p0, q0, ps, qs), ('model_to_reference', q0, p0, qs, ps)):
            raw = kl(a, b)
            result[f'kl_{label}_raw'] = raw if math.isfinite(raw) else 'inf'
            result[f'kl_{label}_smoothed'] = kl(aa, bb)
    return result, edges, p, q


def draw_pair(axes, group, summary, hist, compact=False):
    edges, pc, qc = hist
    blue = '#62a5d2'
    model_values = summary['model_entities']
    for ax, counts, title in zip(axes, (pc, qc), ('Literature reference CV', 'Current-model sample CV')):
        if counts.sum():
            ax.bar(edges[:-1], counts/counts.sum(), width=np.diff(edges), align='edge',
                   color=blue, edgecolor='#263744', linewidth=.65)
        ax.set(xlabel='CV', ylabel='Probability per bin', xlim=(0, edges[-1]))
        ax.set_title(title, fontsize=10 if compact else 12, pad=10)
        ax.tick_params(direction='in', top=True, right=True)
        ax.grid(False)
    ymax = max([.1]+[float(x.max()/x.sum()) for x in (pc, qc) if x.sum()])*1.42
    for ax in axes:
        ax.set_ylim(0, ymax)
    if group == 'travel_water':
        axes[0].axvspan(.25, .35, color=blue, alpha=.22)
        axes[0].text(.5, .58, 'Reported interval: 0.25–0.35\nNo case frequencies available',
                     transform=axes[0].transAxes, ha='center', fontsize=9)
    elif not pc.sum():
        axes[0].text(.5, .5, 'No eligible reference CV', transform=axes[0].transAxes, ha='center')
    if not model_values:
        axes[1].text(.5, .54, 'Inactive in the main experiment\nNo positive-mean sea-border delay',
                     transform=axes[1].transAxes, ha='center', fontsize=9)
    else:
        axes[1].axvline(summary['configured_cv'], color='#9e3d20', ls='--', lw=1.2)
        axes[1].text(.96, .94, f'n = {model_values}; 100 draws/entity\nInput CV = {summary["configured_cv"]:.2f}',
                     transform=axes[1].transAxes, ha='right', va='top', fontsize=8 if compact else 10)
    if pc.sum():
        axes[0].text(.96, .94, f'n = {int(pc.sum())} published summaries', transform=axes[0].transAxes,
                     ha='right', va='top', fontsize=8 if compact else 10)


def plot_all(out, summaries, histograms):
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'DejaVu Serif', 'font.size': 10,
                         'pdf.fonttype': 42, 'svg.fonttype': 'none', 'axes.linewidth': .8})
    for summary in summaries:
        group = summary['source_group']
        fig, axes = plt.subplots(1, 2, figsize=(8.6, 4.8), sharey=True)
        draw_pair(axes, group, summary, histograms[group])
        value = summary['kl_reference_to_model_smoothed']
        suffix = (f'KLε (reference || model) = {value:.3f} nats' if value != ''
                  else 'KL = N/A: '+('reference is an interval' if group == 'travel_water' else 'source inactive'))
        fig.suptitle(GROUPS[group]+'\n'+suffix, fontsize=13, y=.99)
        notes = {
            'travel_road': 'Six published distance-group mean CVs; not individual measured road CVs.',
            'travel_rail': 'Six carrier × direction cases (Kwon, Table 9); Table 8 aggregates are not pooled.',
            'travel_water': 'Published corridor range only; endpoints are not treated as observed cases.',
            'border_road': 'Four lognormal-implied CAREC CVs plus one TTI SD/mean; heterogeneous external cases.',
            'border_rail': 'Three lognormal-implied CAREC CVs; Alashankou is excluded (mean < median).',
            'border_water': 'Callao customs CV is lognormal-implied; the current model has no active sea-border sampling.',
        }[group]
        fig.text(.5, .025, notes+'\nHistograms show unsmoothed proportions. KL uses ε = 0.001 uniform mixing; sparse-reference diagnostic.',
                 ha='center', va='bottom', fontsize=8)
        fig.subplots_adjust(left=.085, right=.98, bottom=.22, top=.77, wspace=.20)
        for ext in ('png', 'pdf', 'svg'):
            fig.savefig(out/(group.upper()+'_CV_CURRENT.'+ext), dpi=240)
        plt.close(fig)
    fig, axes = plt.subplots(6, 2, figsize=(11, 20))
    for row, summary in enumerate(summaries):
        group = summary['source_group']
        draw_pair(axes[row], group, summary, histograms[group], compact=True)
        value = summary['kl_reference_to_model_smoothed']
        kl = f'KLε = {value:.3f}' if value != '' else 'KL = N/A'
        axes[row, 0].set_title(GROUPS[group]+' — literature ('+kl+')', fontsize=11)
        axes[row, 1].set_title(GROUPS[group]+' — existing model', fontsize=11)
    fig.suptitle('Literature CV evidence vs unchanged main-experiment sampler', y=.995, fontsize=15)
    fig.tight_layout(rect=(0, .025, 1, .985), h_pad=2.0)
    fig.text(.5, .009, '100 draws per actual network entity; no fitting. Sparse and heterogeneous literature summaries; sea transport provides a range only.\n'
             'Common bins within each pair; natural-log KL with ε = 0.001 uniform mixing. Histograms themselves are not smoothed.',
             ha='center', fontsize=9)
    fig.savefig(out/'CV_CURRENT_OVERVIEW.png', dpi=180)
    fig.savefig(out/'CV_CURRENT_OVERVIEW.pdf')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--seed', type=int, default=20261120)
    args = parser.parse_args()
    exp = Path(__file__).resolve().parent
    out = (args.out or exp/'transformer_ppo_runs/diagnostics/ccp_cv_current_network_v1').resolve()
    engine = exp/'formal_icm_s0_v1/src/baseline_uncertainty.py'
    episode = engine.parent/'support/transformer_ppo/episode.py'
    data = exp/'formal_icm_s0_v1/assets/data/data_expanded.xlsx'
    events = engine.parent/'data/border_crossing_events.csv'
    paths = [engine, episode, data, events, Path(__file__),
             exp/'cv_evidence_audit_v1.py', exp/'plot_cv_literature_v2.py']
    hashes = {str(p.relative_to(exp)): digest(p) for p in paths}
    protocol = dict(seed=args.seed, realizations_per_entity=REALIZATIONS, cv_ddof=1,
                    kl_uniform_mixing=EPSILON, travel_bin_width=.02, border_bin_width=.10,
                    input_sha256=hashes)
    if out.exists() and any(out.iterdir()):
        if (out/'STATUS.json').exists() and (out/'PLAN.json').exists():
            status = json.loads((out/'STATUS.json').read_text())
            old = json.loads((out/'PLAN.json').read_text())
            if status.get('completed') and old.get('protocol') == protocol:
                for filename, checksum in status['artifacts_sha256'].items():
                    if not (out/filename).is_file() or digest(out/filename) != checksum:
                        raise ValueError('Previously completed artifact changed: '+filename)
                print('[CV-CURRENT] Existing completed report reused:', out)
                return
        raise SystemExit('Output already contains another or incomplete run; use a new --out: '+str(out))
    out.mkdir(parents=True, exist_ok=True)
    os.environ['MPLCONFIGDIR'] = str(out/'.mplconfig')
    started = time.monotonic()
    profile = read_profile(episode)
    if (profile['mode_time_cv'] != dict(road=.15, rail=.10, water=.20)
            or profile['mode_time_max_factor'] != dict(road=None, rail=None, water=None)
            or profile['border_delay_cv'] != dict(road=.5, rail=.5, water=.5)
            or profile['border_delay_max_factor'] != 3.5
            or profile['border_mean_source'] != 'node_border' or profile['border_include_customs']):
        raise ValueError('Main profile differs from requested frozen settings; no overrides applied')
    base = load_engine(engine)
    with (out/'NETWORK_LOAD.log').open('w', encoding='utf-8') as log, contextlib.redirect_stdout(log):
        base.BORDER_EVENT_DEFINITIONS = base.load_border_event_definitions(base.DEFAULT_BORDER_EVENT_DATA_FILE)
        network = base.load_network_from_extended(str(data))
        arcs = network[5]
        border = base.load_border_delay_map_from_workbook(data, include_customs=profile['border_include_customs'])
        scenario = base.build_scenario_set(arcs, border, REALIZATIONS, args.seed, stochastic=True,
            border_event_definitions=base.BORDER_EVENT_DEFINITIONS,
            mode_time_cv=profile['mode_time_cv'], mode_time_max_factor=profile['mode_time_max_factor'],
            border_delay_cv=profile['border_delay_cv'], border_delay_max_factor=profile['border_delay_max_factor'],
            border_mean_source=profile['border_mean_source'])
    groups = {g: [] for g in GROUPS}
    entities, arrays = [], {}
    travel_means = {}
    for a in arcs:
        key = (a.from_node, a.to_node, a.mode)
        nominal = base.nominal_arc_travel_time(a, {})
        if key in travel_means and not math.isclose(travel_means[key], nominal, abs_tol=1e-9):
            raise ValueError('Ambiguous nominal arc time: '+str(key))
        travel_means[key] = nominal
    for family, samples, means in (
            ('travel', scenario.travel_multiplier, travel_means),
            ('border', scenario.border_delay_h, scenario.border_event_mean_h)):
        for key, values in sorted(samples.items()):
            nominal = means[key]
            hours = np.asarray(values, dtype=float)*(nominal if family == 'travel' else 1)
            if len(hours) != REALIZATIONS or not np.isfinite(hours).all() or np.any(hours < 0):
                raise ValueError('Invalid samples: '+str(key))
            group = family+'_'+key[2]
            sample_mean = float(hours.mean())
            cv = float(hours.std(ddof=1)/sample_mean) if sample_mean > 0 else None
            cap = profile['mode_time_max_factor'][key[2]] if family == 'travel' else profile['border_delay_max_factor']
            if cap is not None and np.any(hours > cap*nominal+1e-9):
                raise ValueError('Cap was not preserved')
            array_key = f'entity_{len(entities):04d}'
            arrays[array_key] = hours
            row = dict(array_key=array_key, source_group=group, from_node=key[0], to_node=key[1],
                unit='hours', realizations=REALIZATIONS, nominal_mean_h=nominal,
                sample_mean_h=sample_mean, sample_sd_h=float(hours.std(ddof=1)), sample_cv=cv,
                sample_p90_h=float(np.quantile(hours, .9)), cap_multiplier=cap,
                cap_hit_fraction=float(np.mean(hours >= cap*nominal-1e-12)) if cap and nominal > 0 else 0,
                entity_type='directed arc' if family == 'travel' else 'directional border event')
            entities.append(row)
            if cv is not None:
                groups[group].append(cv)
    refs = literature(profile, groups)
    save_json(out/'PLAN.json', dict(protocol=protocol, profile=profile,
        sampler='frozen baseline_uncertainty.build_scenario_set -> sample_capped_mean_one_lognormal',
        model_parameters_changed=False, calibrated_model=False, cv_unit='sample SD (ddof=1) / sample mean',
        reference_weighting='equal per eligible published case; not weighted by trip counts',
        model_weighting='one CV per unique sampling key used by the engine; equal weight',
        normalization='histogram counts divided by their own total; common bins within each source',
        kl='natural logarithm; p_e=(1-e)*p+e/K, q_e=(1-e)*q+e/K; e=0.001; raw KL also reported',
        source_scope='network-wide marginal sampling audit; not a new S0-S11 search'))
    write_csv(out/'PER_ENTITY_CV.csv', entities)
    write_csv(out/'REFERENCE_CASES.csv', refs)
    np.savez_compressed(out/'NETWORK_SAMPLES_100.npz', **arrays)
    save_json(out/'SOURCES.json', {**OLD_SOURCES, **SOURCES})
    summaries, histograms, binrows = [], {}, []
    for group in GROUPS:
        family, mode = group.split('_')
        rcv = [r['literature_cv'] for r in refs if r['source_group'] == group and r['histogram_included']]
        model = groups[group]
        # Sea's range controls axis extent only, never reference or model counts.
        metric, edges, pc, qc = histogram_metrics(rcv, model, .02 if family == 'travel' else .10,
                                                 minimum_upper=.35 if group == 'travel_water' else 0)
        cv = profile['mode_time_cv' if family == 'travel' else 'border_delay_cv'][mode]
        cap = profile['mode_time_max_factor'][mode] if family == 'travel' else profile['border_delay_max_factor']
        analytical = distribution(cv, cap)
        summary = dict(source_group=group, configured_cv=cv, model_entities=len(model),
            model_mean_sample_cv=float(np.mean(model)) if model else '',
            model_min_sample_cv=float(min(model)) if model else '',
            model_max_sample_cv=float(max(model)) if model else '',
            model_population_cv=analytical['actual_multiplier_cv'] if model else '', **metric)
        summaries.append(summary)
        histograms[group] = edges, pc, qc
        for i in range(len(pc)):
            binrows.append(dict(source_group=group, left=float(edges[i]), right=float(edges[i+1]),
                reference_count=int(pc[i]), model_count=int(qc[i]),
                reference_probability=float(pc[i]/pc.sum()) if pc.sum() else '',
                model_probability=float(qc[i]/qc.sum()) if qc.sum() else ''))
        print('[CV-CURRENT]', group, 'reference cases:',len(rcv), 'model entities:',len(model),
              'KL_eps:',summary['kl_reference_to_model_smoothed'], flush=True)
    write_csv(out/'SOURCE_RESULTS.csv', summaries)
    write_csv(out/'HISTOGRAM_COUNTS.csv', binrows)
    plot_all(out, summaries, histograms)
    if hashes != {str(p.relative_to(exp)): digest(p) for p in paths}:
        raise RuntimeError('Inputs changed during audit')
    notes = '''现有参数 CV 对照：没有校准或调参，没有新搜索、方案评价或训练。

右栏直接调用主实验冻结引擎 build_scenario_set，使用 transport_border_saa_v1。
运输 CV 为 0.15/0.10/0.20、无封顶；公路/铁路口岸 CV 输入 0.50，封顶3.5倍并保持总体期望均值。
这项“均值校正”是原采样函数已有规则，不是本次新增的文献校准。
每个唯一有向弧、方向性口岸事件恰好100次实现；CV = 样本标准差(ddof=1)/样本均值。
主程序口岸事件均值按两端节点取值合并；实体不是独立物理口岸，也没有按人为指定均值重建。
不复制实体、不以大量重复批次充当新增弧或口岸。原始100个小时值保存在NPZ，并用PER_ENTITY_CV.csv索引。

左栏按文献案例等权；公路为6个距离组的平均CV，不是逐路段实测CV。
铁路采用Kwon表9六个承运人×方向案例；表8两条长途汇总仅保留为背景，不重复纳入直方图。
口岸含由均值/中位数在对数正态假设下反推的CV；TTI为SD/mean。
铁路Alashankou均值小于中位数，无法这样反推，保留在REFERENCE_CASES.csv中并排除出图。
海运运输只有0.25–0.35范围，没有概率权重；画区间、KL留空。海运口岸未启用，模型CV和KL不适用。

两侧共用分箱（运输0.02，口岸0.10），纵轴是每箱概率，画出的柱子不做平滑。
原始离散KL遇零支持可以为inf；为便于描述，另报固定0.001均匀混合平滑后的双向KL，单位nats。
标题采用参考||模型方向。平滑量、分箱和稀疏参考影响数值；不得与其他分箱/样本定义下的0.058作通过阈值比较。
没有事先假定公路KL一定小，也没有根据KL更改参数、抽样种子或参考案例。
这是不同地区、时代及统计单位的外部证据一致性检查，不是同一网络的实测校准证明。
不能因为模型参数落在几个参考值中间，就声称已匹配其分布；也不能由CV匹配推断P90或相关结构已验证。

SOURCE_RESULTS.csv为汇总；REFERENCE_CASES.csv记录文献数值及是否入图；SOURCES.json给出处。
各来源PNG/PDF/SVG以*_CV_CURRENT命名；CV_CURRENT_OVERVIEW为总览。图内英文避免中文字体乱码。
'''
    (out/'README_CN.txt').write_text(notes, encoding='utf-8')
    artifacts = {p.name:digest(p) for p in out.iterdir() if p.is_file() and p.name != 'STATUS.json'}
    save_json(out/'STATUS.json', dict(completed=True, seconds=time.monotonic()-started,
        new_searches=0, new_objective_evaluations=0, training_updates=0,
        parameters_changed=False, total_sampled_entities=len(entities),
        total_scalar_realizations=len(entities)*REALIZATIONS, artifacts_sha256=artifacts))
    print('[CV-CURRENT] COMPLETE', out, flush=True)


if __name__ == '__main__':
    main()
