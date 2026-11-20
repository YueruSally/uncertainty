#!/usr/bin/env python3
"""CV evidence range plot; reuse saved main-sampler draws, no KL or calibration."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ['MPLBACKEND'] = 'Agg'
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
sys.dont_write_bytecode = True
import numpy as np

GROUPS = (
    ('travel_road', 'Road transport'), ('travel_rail', 'Rail transport'),
    ('travel_water', 'Sea transport'), ('border_road', 'Road border'),
    ('border_rail', 'Rail border'),
)


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def read_csv(p):
    with p.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def write_csv(p, rows):
    with p.open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def short_label(row):
    if row['source'] == 'Zhao2011':
        i = int(row['label'].split()[-1])-1
        return f'Zhao {10*i}–{10*(i+1)} mi'
    if row['source'] == 'Kwon1995':
        if row['label'].startswith('Carrier '):
            return 'Kwon '+row['label'][8]+('-E' if 'Ebound' in row['label'] else '-W')
        return 'Kwon long-'+('E' if row['label'].startswith('East') else 'W')
    if row['source'] == 'TTI2011':
        return 'TTI (BOTA)'
    return row['label']


def main():
    exp = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=exp/'transformer_ppo_runs/diagnostics/ccp_cv_current_network_v1')
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    src = args.source.resolve()
    out = (args.out or src/'range_plot_v1').resolve()
    names = ('PLAN.json', 'STATUS.json', 'PER_ENTITY_CV.csv', 'REFERENCE_CASES.csv',
             'NETWORK_SAMPLES_100.npz', 'SOURCES.json')
    if not all((src/n).is_file() for n in names):
        raise SystemExit('Saved 100-realization audit files are missing: '+str(src))
    hashes = {n:sha(src/n) for n in names}
    plan = json.loads((src/'PLAN.json').read_text())
    status = json.loads((src/'STATUS.json').read_text())
    if not status.get('completed') or plan['protocol']['realizations_per_entity'] != 100:
        raise ValueError('A completed 100-realization audit is required')
    for n in names:
        if n != 'STATUS.json' and hashes[n] != status['artifacts_sha256'][n]:
            raise ValueError('Saved source artifact changed: '+n)
    # Do not accept samples from a different engine, network or uncertainty profile.
    for n, checksum in plan['protocol']['input_sha256'].items():
        if n.startswith('formal_icm_s0_v1/') and sha(exp/n) != checksum:
            raise ValueError('Frozen main-experiment input differs: '+n)
    protocol = dict(source=str(src), input_sha256=hashes, script_sha256=sha(Path(__file__)),
                    percentile_method='linear', percentiles=[5,95], cv_sd_ddof=1)
    if out.exists() and any(out.iterdir()):
        p = out/'STATUS.json'
        old = json.loads(p.read_text()) if p.exists() else {}
        if old.get('completed') and old.get('protocol') == protocol:
            for n, checksum in old['artifacts_sha256'].items():
                if sha(out/n) != checksum:
                    raise ValueError('Completed plot artifact changed: '+n)
            print('[CV-RANGE] Reusing completed plot:', out)
            return
        raise SystemExit('Choose a new --out; the existing folder is not overwritten: '+str(out))
    out.mkdir(parents=True, exist_ok=True)
    os.environ['MPLCONFIGDIR'] = str(out/'.mplconfig')
    rows = read_csv(src/'PER_ENTITY_CV.csv')
    reference = read_csv(src/'REFERENCE_CASES.csv')
    summaries = []
    with np.load(src/'NETWORK_SAMPLES_100.npz', allow_pickle=False) as pack:
        for group, label in GROUPS:
            part = [r for r in rows if r['source_group'] == group]
            if len(part) != len({(r['from_node'],r['to_node']) for r in part}):
                raise ValueError('Duplicate sampling entities')
            cvs = []
            for r in part:
                x = pack[r['array_key']]
                if x.shape != (100,) or not np.isfinite(x).all() or x.mean() <= 0:
                    raise ValueError('Invalid saved realizations')
                cv = float(x.std(ddof=1)/x.mean())
                if not np.isclose(cv, float(r['sample_cv']), rtol=1e-12, atol=1e-14):
                    raise ValueError('Saved CV does not reproduce')
                cvs.append(cv)
            lo, hi = np.percentile(cvs, [5,95])
            family, mode = group.split('_')
            summaries.append(dict(source_group=group, source_label=label, entities=len(cvs),
                realizations_per_entity=100,
                input_cv=plan['profile']['mode_time_cv' if family=='travel' else 'border_delay_cv'][mode],
                mean_sample_cv=float(np.mean(cvs)), sample_cv_p05=float(lo), sample_cv_p95=float(hi),
                interpretation='5th–95th percentiles across entity CV estimates; not a confidence interval'))
    valid = []
    for r in reference:
        if r['source_group'] not in dict(GROUPS):
            continue
        # Display all valid evidence points, including distinct Table 8 aggregates.
        # They are not pooled into a literature average or a distribution statistic.
        valid.append(dict(source_group=r['source_group'], label=short_label(r), source=r['source'],
            evidence_kind=r['evidence_kind'], cv=r['literature_cv'],
            cv_low=r['literature_cv_low'], cv_high=r['literature_cv_high'],
            displayed=bool(r['literature_cv'] or r['literature_cv_low']),
            location=r['location'], note=r['note'].split(' Context only:')[0]))
    write_csv(out/'MODEL_CV_RANGE.csv', summaries)
    write_csv(out/'LITERATURE_POINTS.csv', valid)
    (out/'SOURCES.json').write_bytes((src/'SOURCES.json').read_bytes())
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':10,
                         'pdf.fonttype':42, 'svg.fonttype':'none'})
    fig, ax = plt.subplots(figsize=(14, 8.6))
    blue, teal, red = '#315e83', '#347a79', '#bf3434'
    ys = np.arange(4,-1,-1)*1.2
    for i, (summary,y) in enumerate(zip(summaries, ys)):
        group = summary['source_group']
        if i % 2 == 0:
            ax.axhspan(y-.57, y+.57, color='#f6f8fa', zorder=0)
        points = sorted([r for r in valid if r['source_group']==group and r['cv']], key=lambda r:float(r['cv']))
        # Short callouts spread densely packed transport labels, preserving actual x positions.
        label_xs = np.linspace(.015, .76 if group=='travel_rail' else .65, len(points)) if group.startswith('travel_') else [float(r['cv']) for r in points]
        for j, (r,tx) in enumerate(zip(points, label_xs)):
            x = float(r['cv'])
            inferred = r['evidence_kind']=='lognormal_implied'
            py = y+.08 + (j%2)*.05 if group.startswith('travel_') else y+.10
            ax.plot(x, py, 'D' if inferred else 'o', ms=5, color=teal if inferred else blue, zorder=4)
            tag = r['label'].replace('Zhao ', 'Zhao\n').replace('Kwon ', 'Kwon\n')
            label = tag+'\n'+f'{x:.3f}'
            ax.text(tx,y+.40,label,fontsize=8.2,
                ha='center' if not group.startswith('travel_') else 'left', va='center',
                color=teal if inferred else blue,zorder=4)
            # Keep leaders below the label block, including neighbouring labels.
            anchor=(tx+.022,y+.23) if group.startswith('travel_') else (tx,y+.27)
            ax.annotate('',xy=(x,py),xytext=anchor,
                arrowprops=dict(arrowstyle='-',color='#82919d',lw=.65,shrinkA=0,shrinkB=4),zorder=3)
        for r in [r for r in valid if r['source_group']==group and r['cv_low']]:
            lo,hi = float(r['cv_low']),float(r['cv_high'])
            ax.fill_between([lo,hi], y-.43, y+.30, color=blue, alpha=.15, lw=0)
            ax.plot([lo,hi],[y+.10,y+.10], color=blue, lw=2)
            ax.text((lo+hi)/2, y+.40, 'Sea-Intelligence (2026)\n0.25–0.35', ha='center', va='center', fontsize=9, color=blue)
        m,lo,hi = (summary[k] for k in ('mean_sample_cv','sample_cv_p05','sample_cv_p95'))
        ax.errorbar(m,y-.20,xerr=np.array([[m-lo],[hi-m]]),fmt='o',color=red,
                    ecolor=red,ms=6,capsize=4,lw=2,zorder=5)
        ax.text(hi+.025,y-.20,f'{m:.3f}  [{lo:.3f}, {hi:.3f}]',va='center',fontsize=9,color=red)
    ax.set_yticks(ys, [s['source_label']+f'\n(n = {s["entities"]})' for s in summaries])
    ax.tick_params(axis='y',length=0,pad=14)
    ax.set(xlim=(-.04,1.82), ylim=(-.6,5.44), xlabel='Coefficient of variation (CV)')
    ax.set_xticks(np.arange(0,1.81,.2))
    ax.grid(axis='x',color='#dfe4e8',lw=.6,zorder=0)
    for side in ('top','right','left'):
        ax.spines[side].set_visible(False)
    handles = [Line2D([],[],marker='o',ls='',color=blue,label='Reported CV / SD ÷ mean'),
               Line2D([],[],marker='D',ls='',color=teal,label='Lognormal-implied CV'),
               Patch(facecolor=blue,alpha=.15,label='Published CV interval'),
               Line2D([],[],marker='o',color=red,lw=2,label='Model: mean and 5th–95th percentiles')]
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.55,.025),ncol=2,frameon=False,fontsize=9.5)
    fig.subplots_adjust(left=.14,right=.98,bottom=.15,top=.98)
    for ext in ('png','pdf','svg'):
        fig.savefig(out/('CV_RANGE_COMPARISON.'+ext),dpi=260)
    plt.close(fig)
    (out/'CAPTION_CN.txt').write_text(
        '五类启用来源的文献CV证据与现有模型样本CV。每个主实验有向弧/方向性口岸事件使用已保存的100次实现；'
        '红点为逐实体样本CV的算术平均，红色误差线为实体之间样本CV的第5和第95百分位，不是均值的置信区间。'
        '圆点表示文献报告CV或标准差/均值，菱形表示在对数正态假设下由均值/中位数反推CV，阴影表示文献区间。'
        '密集点仅在纵向轻微错开，横坐标保持原值；引线连接来源标签和实际数值。'
        '公路运输为六个距离组的平均CV；铁路同时展示六个承运人方向案例和两条长途汇总，未将这些值合并计算分布指标。'
        '铁路Alashankou均值小于中位数，不能如此反推，未绘制；海运口岸未启用，未绘制。'
        '文献统计单位、地区、年代不同。落在文献点的最小—最大范围内仅说明数量级相容，不代表已完成实测校准。'
        '图中不计算KL、不进行模型校准，不改变主实验参数或结果。\n',encoding='utf-8')
    if hashes != {n:sha(src/n) for n in names}:
        raise RuntimeError('Source files changed')
    artifacts={p.name:sha(p) for p in out.iterdir() if p.is_file()}
    (out/'STATUS.json').write_text(json.dumps(dict(completed=True,protocol=protocol,
        new_searches=0,new_training_updates=0,new_random_draws=0,kl_computed=False,
        artifacts_sha256=artifacts),ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    for s in summaries:
        print(f'{s["source_label"]}: {s["mean_sample_cv"]:.4f} '
              f'[{s["sample_cv_p05"]:.4f}, {s["sample_cv_p95"]:.4f}], n={s["entities"]}')
    print('[CV-RANGE] COMPLETE',out)


if __name__=='__main__':
    main()
