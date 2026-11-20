#!/usr/bin/env python3
"""Five time-distribution panels using the unchanged main-experiment sampler."""
import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
from statistics import NormalDist
import sys
import time

os.environ['MPLBACKEND'] = 'Agg'
for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'
sys.dont_write_bytecode = True
import numpy as np
from cv_evidence_audit_v1 import read_profile, distribution

GROUPS = (
    ('travel_road', 'Road transport', '公路运输', .05, .25),
    ('travel_rail', 'Rail transport', '铁路运输', .05, .20),
    ('travel_water', 'Sea transport', '海运运输', .10, .30),
    ('border_road', 'Road border', '公路口岸', .30, .70),
    ('border_rail', 'Rail border', '铁路口岸', .30, .70),
)
N = 10000
NORMAL = NormalDist()


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False,
                              indent=2)+'\n', encoding='utf-8')


def load_engine(path):
    spec = importlib.util.spec_from_file_location('cv_shapes_frozen_engine', path)
    engine = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = engine
    spec.loader.exec_module(engine)
    return engine


def reference_cases(rows, sources):
    """Preserve reported summaries; reconstruct only under an explicit LN assumption."""
    result = []
    for row in rows:
        if row['source_group'] not in {g[0] for g in GROUPS}:
            continue
        kind = row['evidence_kind']
        mean = float(row['mean']) if row['mean'] else None
        sd = float(row['sd']) if row['sd'] else None
        median = float(row['median']) if row['median'] else None
        note = row['note'].replace('equal weight per row. ', '')
        case = dict(source_group=row['source_group'], case=row['label'],
                    source=row['source'], location=row['location'],
                    url=sources[row['source']]['url'], evidence_kind=kind,
                    reported_mean=mean, reported_sd=sd, reported_median=median,
                    reported_time_unit=row['time_unit'], cv=None, plotted=True,
                    interpretation='', note=note)
        if row['source']=='CAREC2023':
            case['location'] = ('Tables 4.1/4.2, printed p.12' if row['source_group']=='border_road'
                                else 'Tables 4.5/4.6, printed p.14')
        if kind == 'published_range':
            for side, key in (('lower endpoint', 'literature_cv_low'),
                              ('upper endpoint', 'literature_cv_high')):
                result.append(dict(case, case=case['case']+' — '+side,
                    cv=float(row[key]),
                    interpretation='Mean-one lognormal illustration at a reported range endpoint; not an observed case'))
            continue
        if kind == 'sd_over_mean':
            cv = sd/mean
            interpretation = 'Lognormal reconstructed from reported mean and SD'
        elif kind == 'lognormal_implied':
            if mean < median:
                result.append(dict(case, plotted=False,
                    interpretation='Not reconstructed: mean < median, incompatible with matching both using a two-parameter lognormal'))
                continue
            cv = math.sqrt((mean/median)**2-1)
            interpretation = 'Lognormal reconstructed from reported mean and median'
        elif kind == 'published_group_cv':
            cv = float(row['literature_cv'])
            interpretation = 'Mean-one lognormal illustration from a distance-group mean CV; no original time distribution'
            i = int(row['label'].split()[-1])-1
            case['case'] = f'Distance {10*i}–{10*(i+1)} mi'
        else:
            raise ValueError('Unsupported reference evidence: '+kind)
        if not math.isclose(cv, float(row['literature_cv']), rel_tol=1e-10):
            raise ValueError('Reference CV does not reproduce: '+row['label'])
        result.append(dict(case, cv=cv, interpretation=interpretation))
    order = {g[0]:i for i,g in enumerate(GROUPS)}
    return sorted(result,key=lambda r:order[r['source_group']])


def lognormal_pdf(x, cv):
    sigma = math.sqrt(math.log1p(cv*cv))
    mu = -.5*sigma*sigma
    return np.exp(-.5*((np.log(x)-mu)/sigma)**2)/(x*sigma*math.sqrt(2*math.pi))


def lognormal_quantile(probability, cv):
    sigma = math.sqrt(math.log1p(cv*cv))
    return math.exp(-.5*sigma*sigma+sigma*NORMAL.inv_cdf(probability))


def tail_probability(x, cv):
    sigma = math.sqrt(math.log1p(cv*cv))
    return 1-NORMAL.cdf((math.log(x)+.5*sigma*sigma)/sigma)


def density_estimator(samples, cap):
    """Positive-support log-KDE; hard-cap atom is retained separately, never smeared."""
    continuous = samples if cap is None else samples[samples < cap]
    logs = np.log(continuous)
    bandwidth = 1.06*logs.std(ddof=1)*len(logs)**(-.2)
    log_cap = None if cap is None else math.log(cap)

    def evaluate(x):
        values = np.zeros_like(x, dtype=float)
        indices = np.flatnonzero((x > 0) & ((x < cap) if cap is not None else True))
        for start in range(0, len(indices), 100):
            ix = indices[start:start+100]
            z = np.log(x[ix])[:, None]
            kernel = np.exp(-.5*((z-logs)/bandwidth)**2).sum(axis=1)
            if log_cap is not None:
                # Reflect the continuous log-density at the upper boundary.
                kernel += np.exp(-.5*((z+logs-2*log_cap)/bandwidth)**2).sum(axis=1)
            values[ix] = kernel/(len(samples)*bandwidth*math.sqrt(2*math.pi)*x[ix])
        return values

    return evaluate, bandwidth, len(samples)-len(continuous)


def captions(cases):
    names = []
    for group, en, cn, _, _ in GROUPS:
        names.append(cn+'：'+'；'.join(r['source']+' '+r['case'] for r in cases
                                      if r['source_group']==group and r['plotted']))
    cn = (
        '五类不确定性来源的时间分布形状比较。横轴为实际时间除以该来源的总体期望时间 T/E[T]；'
        '不除以抽样均值，也不对样本进行事后校准。每个来源按主实验采样函数分别抽取10000个基线样本，'
        '并在用户指定的敏感性下界、上界各抽10000次。红色粗线为基线的核密度估计，灰色虚线为两端敏感性分布，'
        '浅蓝细线为对数正态假设下的文献参考密度。各子图使用独立横轴及纵轴。'
        '红线和灰线采用正值支持的对数域核密度估计；口岸连续部分在封顶边界反射校正并在边界处终止，'
        '封顶点的离散概率不平滑为密度，其概率另列于MODEL_SAMPLE_STATS.csv。因此口岸连续曲线面积为1减封顶概率。'
        '基线保持主实验参数及既有封顶均值校准；敏感性仅改变本绘图脚本传给采样函数的CV实参，未修改配置文件、搜索或训练。'
        '同一来源的三种CV使用同一随机种子；不同来源种子不同。'
        '文献曲线不是原始数据拟合：铁路运输及TTI按均值/标准差重建，CAREC按均值/中位数反推；'
        '公路运输只有分距离组的平均CV，蓝线只是由该CV构造的均值为1的对数正态形状参照；'
        '海运两条蓝线只是文献区间端点的对数正态示意，不是两个实测案例。'
        '参考分布不套用主模型的封顶。所有参考曲线各自归一化，不合并为混合分布。'
        '铁路Alashankou的均值小于中位数，无法匹配两参数对数正态，保留在表中但不画曲线；海运口岸未启用，省略。'
        '为看清主体形状，横轴至少覆盖每条参考密度的99%分位和模拟样本的99.5%分位，'
        '超出绘图范围的概率另存PLOT_TAILS.csv；曲线不按可见范围重新归一化。'
        '图形相似不能单独证明分布拟合正确或参数已校准，不计算KL。\n\n案例对应：\n'+'\n'.join(names)+'\n')
    en = (
        'Time-distribution shapes for the five active uncertainty sources. '
        'The horizontal coordinate is time divided by its population mean, T/E[T]. '
        'Thick red curves show log-domain kernel density estimates from 10,000 draws using the unchanged main-experiment sampler; '
        'grey dashed curves use the prescribed lower and upper CV values for a sampling-only sensitivity illustration. '
        'Thin blue curves are mean-normalised lognormal references reconstructed from published mean/SD or mean/median summaries. '
        'Where only CVs are available (road distance groups and sea-range endpoints), blue curves are lognormal shape illustrations, not empirical fits. '
        'Hard-cap point masses are reported separately; continuous border densities stop at the cap and integrate to one minus the cap mass. '
        'Panels have separate axes. Display limits omit only the tails recorded in PLOT_TAILS.csv, without renormalising visible densities. '
        'The main search/training settings are unchanged. Case identities and CVs are listed in the accompanying table.\n')
    return cn, en


def main():
    exp = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=exp/'transformer_ppo_runs/diagnostics/ccp_cv_current_network_v1')
    parser.add_argument('--out', type=Path, default=exp/'transformer_ppo_runs/diagnostics/ccp_cv_distribution_shapes_v1')
    parser.add_argument('--seed', type=int, default=20261121)
    args = parser.parse_args()
    src, out = args.source.resolve(), args.out.resolve()
    engine_path = exp/'formal_icm_s0_v1/src/baseline_uncertainty.py'
    profile_path = engine_path.parent/'support/transformer_ppo/episode.py'
    input_paths = [engine_path, profile_path, exp/'cv_evidence_audit_v1.py',
                   src/'PLAN.json', src/'STATUS.json', src/'REFERENCE_CASES.csv', src/'SOURCES.json']
    if not all(p.is_file() for p in input_paths):
        raise SystemExit('Missing frozen engine/profile or completed CV evidence audit: '+str(src))
    hashes = {str(p):sha(p) for p in input_paths}
    source_status = json.loads((src/'STATUS.json').read_text())
    source_plan = json.loads((src/'PLAN.json').read_text())
    if not source_status.get('completed'):
        raise ValueError('The source evidence audit must be complete')
    for name in ('PLAN.json', 'REFERENCE_CASES.csv', 'SOURCES.json'):
        if sha(src/name) != source_status['artifacts_sha256'][name]:
            raise ValueError('Source artifact changed: '+name)
    for p in (engine_path, profile_path):
        if sha(p) != source_plan['protocol']['input_sha256'][str(p.relative_to(exp))]:
            raise ValueError('Main experiment code changed since source audit: '+str(p))
    protocol = dict(seed=args.seed, draws_per_source_and_curve=N, input_sha256=hashes,
        script_sha256=sha(Path(__file__)), shared_random_numbers_within_source=True,
        normalization='time / population mean; mean-one sampler; no empirical rescaling',
        sampler='baseline_uncertainty.sample_capped_mean_one_lognormal',
        new_searches=0, new_objective_evaluations=0, training_updates=0)
    if out.exists() and any(out.iterdir()):
        p = out/'STATUS.json'
        old = json.loads(p.read_text()) if p.exists() else {}
        if old.get('completed') and old.get('protocol') == protocol:
            if any(sha(out/n)!=h for n,h in old['artifacts_sha256'].items()):
                raise ValueError('Previously completed artifact was modified')
            print('[CV-SHAPES] Reusing completed figure:', out)
            return
        raise SystemExit('Output exists; use another --out to preserve it: '+str(out))
    out.mkdir(parents=True, exist_ok=True)
    os.environ['MPLCONFIGDIR'] = str(out/'.mplconfig')
    started = time.monotonic()
    profile = read_profile(profile_path)
    if (profile['mode_time_cv'] != dict(road=.15, rail=.10, water=.20)
            or profile['mode_time_max_factor'] != dict(road=None, rail=None, water=None)
            or profile['border_delay_cv'] != dict(road=.5, rail=.5, water=.5)
            or profile['border_delay_max_factor'] != 3.5):
        raise ValueError('Baseline differs from specified main experiment; no override applied')
    engine = load_engine(engine_path)
    sources = json.loads((src/'SOURCES.json').read_text())
    cases = reference_cases(read_csv(src/'REFERENCE_CASES.csv'), sources)
    stats, sample_arrays, estimators, mass_errors = [], {}, {}, []
    for i, (group,en,cn,lower,upper) in enumerate(GROUPS):
        family, mode = group.split('_')
        baseline = profile['mode_time_cv' if family=='travel' else 'border_delay_cv'][mode]
        cap = profile['mode_time_max_factor'][mode] if family=='travel' else profile['border_delay_max_factor']
        for role,cv in (('baseline',baseline),('lower',lower),('upper',upper)):
            key = group+'_'+role
            samples = engine.sample_capped_mean_one_lognormal(
                size=N, cv=cv, cap_factor=cap, rng=np.random.default_rng(args.seed+i))
            if (samples.shape!=(N,) or not np.isfinite(samples).all()
                    or np.any(samples<=0) or (cap is not None and np.any(samples>cap))):
                raise ValueError('Invalid primary-sampler draws')
            if abs(samples.mean()-1)>6*samples.std(ddof=1)/math.sqrt(N):
                raise ValueError('Unexpected departure from the main sampler population mean')
            sample_arrays[key] = samples
            density,h,hits = density_estimator(samples,cap)
            estimators[key] = density
            continuous = samples if cap is None else samples[samples<cap]
            lo = continuous.min()*math.exp(-7*h)
            hi = continuous.max()*math.exp(7*h) if cap is None else np.nextafter(float(cap),0.)
            check_x = np.geomspace(lo,hi,900)
            integrate = getattr(np,'trapezoid',None) or np.trapz
            area = float(integrate(density(check_x),check_x))
            error = abs(area+hits/N-1)
            if error>.004:
                raise ValueError('Density plus hard-cap mass does not integrate to one')
            mass_errors.append(error)
            expected = distribution(cv,cap)
            stats.append(dict(source_group=group,source=en,role=role,input_cv=cv,
                sample_size=N,seed=args.seed+i,cap_factor=cap,
                population_mean=1.0,sample_mean=float(samples.mean()),
                sample_sd=float(samples.std(ddof=1)),sample_cv=float(samples.std(ddof=1)/samples.mean()),
                population_cv_after_cap=expected['actual_multiplier_cv'],
                sample_p90=float(np.quantile(samples,.9)),
                cap_hits=hits,cap_hit_fraction=hits/N,theoretical_cap_mass=expected['cap_mass'],
                kde_log_bandwidth=h,continuous_density_area=area))
        print('[CV-SHAPES] sampled and checked:',en,flush=True)
    np.savez_compressed(out/'SAMPLES_10000.npz',**sample_arrays)
    write_csv(out/'REFERENCE_CASES.csv',cases)
    write_csv(out/'MODEL_SAMPLE_STATS.csv',stats)
    (out/'SOURCES.json').write_bytes((src/'SOURCES.json').read_bytes())
    cn_caption,en_caption = captions(cases)
    (out/'CAPTION_CN.txt').write_text(cn_caption,encoding='utf-8')
    (out/'CAPTION_EN.txt').write_text(en_caption,encoding='utf-8')
    lines = ['# CV 参数与文献案例','',
        '数值仅描述本图。敏感性曲线只用于抽样形状比较，没有更改主实验参数，也没有重新搜索或训练。',
        '文献分布为对数正态重建或形状示意，不是原始数据拟合。','',
        '| 来源 | 基线输入 CV | 敏感性下界 | 敏感性上界 | 封顶倍数 |',
        '|---|---:|---:|---:|---:|']
    for group,en,cn,low,high in GROUPS:
        r = next(r for r in stats if r['source_group']==group and r['role']=='baseline')
        lines.append(f'| {cn} | {r["input_cv"]:.2f} | {low:.2f} | {high:.2f} | {r["cap_factor"] or "无"} |')
    lines += ['', '| 来源 | 文献案例 | CV | 依据 | 原文位置 |','|---|---|---:|---|---|']
    group_cn = {g[0]:g[2] for g in GROUPS}
    for r in cases:
        value = f'{r["cv"]:.4f}' if r['cv'] is not None else '无法反推；不绘制'
        evidence_cn = {
            'sd_over_mean':'报告标准差÷均值；假设对数正态',
            'lognormal_implied':'由均值和中位数反推；假设对数正态',
            'published_group_cv':'报告的距离组平均CV；构造形状参照',
            'published_range':'报告区间的端点；构造形状参照',
        }[r['evidence_kind']]
        if not r['plotted']:
            evidence_cn = '均值小于中位数，不能同时匹配两参数对数正态'
        lines.append(f'| {group_cn[r["source_group"]]} | [{r["source"]} · {r["case"]}]({r["url"]}) | {value} | {evidence_cn} | {r["location"]} |')
    lines += ['','注：铁路的长途汇总与承运人方向案例分别展示，不将这些曲线汇成混合分布；海运两项为同一区间的端点，不是两个实测案例。',
        '口岸按主程序封顶后有离散概率质量。连续密度曲线不包含该点质量，详见 MODEL_SAMPLE_STATS.csv。']
    (out/'CV_CASES.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    plot(out,cases,stats,sample_arrays,estimators)
    if hashes != {str(p):sha(p) for p in input_paths}:
        raise RuntimeError('An input changed during the run')
    artifacts = {p.name:sha(p) for p in out.iterdir() if p.is_file()}
    save_json(out/'STATUS.json',dict(completed=True,protocol=protocol,
        model_config_files_modified=False,kl_computed=False,
        sampler_values_generated=5*3*N,maximum_density_mass_error=max(mass_errors),
        elapsed_seconds=time.monotonic()-started,artifacts_sha256=artifacts))
    print('[CV-SHAPES] COMPLETE',out,flush=True)


def plot(out,cases,stats,samples,estimators):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plt.rcParams.update({'font.family':'DejaVu Serif','font.size':11,
                         'axes.linewidth':.8,'pdf.fonttype':42,'svg.fonttype':'none'})
    red,gray,blue = '#c93639','#777777','#86bada'
    fig = plt.figure(figsize=(13.8,8.6))
    gs = fig.add_gridspec(2,6,left=.065,right=.985,bottom=.20,top=.98,
                         hspace=.60,wspace=.90)
    slots = [gs[0,0:2],gs[0,2:4],gs[0,4:6],gs[1,1:3],gs[1,3:5]]
    tails,curve_rows = [],[]
    for i,((group,en,cn,_,_),slot) in enumerate(zip(GROUPS,slots)):
        ax = fig.add_subplot(slot)
        refs = [r for r in cases if r['source_group']==group and r['plotted']]
        bound = max([lognormal_quantile(.99,r['cv']) for r in refs]+
                    [float(np.quantile(samples[group+'_'+r],.995)) for r in ('baseline','lower','upper')])
        if group.startswith('border_'):
            bound = max(bound,3.5)
        xmax = math.ceil(bound*1.04*5)/5
        x = np.linspace(.0001,xmax,1600)
        for r in refs:
            y = lognormal_pdf(x,r['cv'])
            ax.plot(x,y,color=blue,lw=1.1,alpha=.85,zorder=2)
            tails.append(dict(source_group=group,curve=r['source']+' / '+r['case'],
                              x_max=xmax,probability_above_x_max=tail_probability(xmax,r['cv'])))
            curve_rows.extend(dict(source_group=group,curve=r['source']+' / '+r['case'],
                                   x=float(a),density=float(b)) for a,b in zip(x,y))
        for role in ('lower','upper','baseline'):
            key = group+'_'+role
            stat = next(r for r in stats if r['source_group']==group and r['role']==role)
            # The point mass at the cap is in the table; continuous curves stop there.
            grid = x if stat['cap_factor'] is None else x[x<stat['cap_factor']]
            y = estimators[key](grid)
            ax.plot(grid,y,color=red if role=='baseline' else gray,
                    lw=2.5 if role=='baseline' else 1.5,
                    ls='-' if role=='baseline' else '--',zorder=4 if role=='baseline' else 3)
            tails.append(dict(source_group=group,curve='Model / '+role,x_max=xmax,
                              probability_above_x_max=float(np.mean(samples[key]>xmax))))
            curve_rows.extend(dict(source_group=group,curve='Model / '+role,x=float(a),density=float(b))
                              for a,b in zip(grid,y))
        ax.set(xlim=(0,xmax),ylim=(0,None),xlabel='Time / mean',ylabel='Density')
        ax.tick_params(direction='in',top=True,right=True,labelsize=9.5)
        ax.text(.5,-.29,f'({chr(97+i)}) {en}',transform=ax.transAxes,ha='center',va='top',fontsize=12)
        ax.grid(False)
    fig.legend(handles=[
        Line2D([],[],color=red,lw=2.5,label='Current model'),
        Line2D([],[],color=gray,lw=1.5,ls='--',label='Sensitivity bounds'),
        Line2D([],[],color=blue,lw=1.1,label='Literature-based lognormal references')],
        loc='lower center',bbox_to_anchor=(.5,.017),ncol=3,frameon=False,fontsize=11)
    for ext in ('png','pdf','svg'):
        fig.savefig(out/('CV_DISTRIBUTION_SHAPES.'+ext),dpi=260)
    plt.close(fig)
    write_csv(out/'PLOT_TAILS.csv',tails)
    write_csv(out/'PLOT_CURVES.csv',curve_rows)


if __name__=='__main__':
    main()
