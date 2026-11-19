#!/usr/bin/env python3
"""Plot completed CV evidence audit only; no sensitivity runs or evaluations."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path

os.environ['MPLBACKEND'] = 'Agg'
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '1'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path(__file__).resolve().parent /
                   'transformer_ppo_runs/diagnostics/ccp_cv_evidence_audit_v1')
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    root, out = a.root.resolve(), a.out.resolve()
    if root == out or root in out.parents or out in root.parents:
        raise ValueError('Use a separate plot directory, outside the source audit')
    source = root/'AUDIT.json'
    source_sha = sha(source)
    data = json.loads(source.read_text(encoding='utf-8-sig'))
    if not data.get('completed'):
        raise ValueError('Source audit is incomplete')
    road = next(r for r in data['road_border'] if r['方案'] == '当前代码')
    sea = data['sea_reference']
    out.mkdir(parents=True, exist_ok=True)
    provenance = dict(source=str(source), source_sha256=source_sha,
                      script_sha256=sha(Path(__file__)), new_searches=0,
                      new_objective_evaluations=0, training_updates=0,
                      sensitivity_experiments=0, current_setting_only=True)
    plan = out/'PLOT_INPUTS.json'
    if plan.exists() and json.loads(plan.read_text()) != provenance:
        raise ValueError('Plot inputs changed; choose a fresh --out directory')
    plan.write_text(json.dumps(provenance, indent=2)+'\n')
    os.environ['MPLCONFIGDIR'] = str(out/'.matplotlib')
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':10,
                         'pdf.fonttype':42, 'ps.fonttype':42, 'svg.fonttype':'path'})
    blue, orange, grey = '#4166ff', '#ff9800', '#707985'

    def save(fig, name):
        for ext in ('png', 'pdf', 'svg'):
            fig.savefig(out/(name+'.'+ext), dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        print('[CV-PLOT] saved', out/(name+'.png'), flush=True)

    def clean(ax):
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(direction='out')

    # Input CV and effective CV are different for capped border multipliers.
    rows = data['six_sources']
    expected = ['travel_road','travel_rail','travel_water','border_road','border_rail','border_water']
    if [r['来源'] for r in rows] != expected:
        raise ValueError('Unexpected source order')
    labels = ['Road travel','Rail travel','Sea travel','Road border','Rail border','Sea border']
    notes = [
        'Lognormal; external CV support\nChen (2018); Zhao (2011); transfer limitation',
        'Lognormal simplification; assumed CV\nLessan (2018): log-logistic for passenger HSR',
        'Lognormal evidence in a narrow channel\nWu (2020); CV = 0.20 remains an assumption',
        'Capped lognormal; external moment benchmark\nTTI (2011): CV = 28.9 / 49.5 = 0.58384',
        'Capped lognormal; assumed CV\nNo direct CV calibration data identified',
        'INACTIVE: all current border means are zero\nConfigured CV has no effect; effective CV undefined',
    ]
    fig, (ax, note) = plt.subplots(1,2,figsize=(13.5,6),gridspec_kw={'width_ratios':[1.05,1.6]})
    fig.subplots_adjust(left=.12,right=.98,top=.80,bottom=.17,wspace=.12)
    for i,r in enumerate(rows):
        color = grey if i==5 else blue
        ax.hlines(i,0,r['配置CV'],color='#dce3ef',lw=3)
        ax.plot(r['配置CV'],i,'o',mfc='white',mec=color,ms=9,mew=1.6)
        if isinstance(r['正均值乘子实际CV'], (int,float)):
            ax.plot(r['正均值乘子实际CV'],i,'x',color=blue,ms=7,mew=1.5)
        ax.text(r['配置CV']+.018,i-.15,f"{r['配置CV']:.2f}",color=color,fontsize=9)
        note.text(0,i,notes[i],va='center',fontsize=10,color=grey if i==5 else '#202830',linespacing=1.5)
    ax.plot(28.9/49.5,3,'D',color=orange,ms=7)
    ax.text(.598,3.18,'0.58384',color='#a35c00',fontsize=9)
    ax.set(yticks=np.arange(6),yticklabels=labels,xlim=(0,.72),ylim=(5.55,-.6),xlabel='Coefficient of variation (CV)')
    ax.grid(axis='x',alpha=.15); clean(ax)
    note.set(xlim=(0,1),ylim=(5.55,-.6)); note.axis('off')
    handles=[Line2D([],[],marker='o',mfc='white',mec=blue,ls='',label='Configured CV'),
             Line2D([],[],marker='x',color=blue,ls='',label='Effective CV (positive mean)'),
             Line2D([],[],marker='D',color=orange,ls='',label='External moment benchmark')]
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.52,.92),ncol=3,frameon=False,fontsize=9)
    fig.suptitle('Six uncertainty sources: current settings and evidence',y=.985,fontsize=15)
    fig.text(.12,.05,'Border cap = 3.5 x mean. Configured CV 0.50 gives effective CV 0.49384 after capping.\nExternal evidence is not route-specific calibration. No sensitivity experiments are included.',fontsize=9,color=grey)
    save(fig,'CV_01_SOURCES_AND_EVIDENCE')

    fig, axes = plt.subplots(1,3,figsize=(13.4,4.9))
    fig.subplots_adjust(left=.065,right=.98,top=.80,bottom=.28,wspace=.43)
    for ax in axes: clean(ax); ax.grid(axis='y',alpha=.15); ax.set_axisbelow(True)
    values = [road['代码形式P90_分钟'],road['参考P90_分钟']]
    axes[0].bar([0,1],values,color=[blue,orange],width=.55)
    axes[0].set(xticks=[0,1],xticklabels=['Current code','Moment reference'],ylim=(0,105),ylabel='P90 (minutes)')
    axes[0].tick_params(axis='x',labelsize=9)
    for i,v in enumerate(values): axes[0].text(i,v+1.8,f'{v:.2f}',ha='center',fontsize=11)
    axes[0].set_title('(a) Road border',pad=27)
    axes[0].text(.5,1.015,f"Signed P90 error: {road['P90有符号相对误差%']:+.2f}%",transform=axes[0].transAxes,ha='center',fontsize=10)
    x=np.arange(2); w=.32
    reconstructed=[sea['reconstructed_p90_days'],sea['reconstructed_p95_days']]
    reference=[sea['reference_p90_days'],sea['reference_p95_days']]
    for shift,v,color,label in [(-w/2,reconstructed,blue,'Reconstructed LN'),(w/2,reference,orange,'Published quantile')]:
        axes[1].bar(x+shift,v,w,color=color,label=label)
        for xx,vv in zip(x+shift,v): axes[1].text(xx,vv+.45,f'{vv:.2f}',ha='center',fontsize=9)
    axes[1].set(xticks=x,xticklabels=[f"P90\n{sea['p90_signed_error_percent']:+.2f}%",f"P95\n{sea['p95_signed_error_percent']:+.2f}%"],ylim=(0,25),ylabel='Time (days)')
    axes[1].set_title('(b) Sea customs: reference reconstruction',pad=27,fontsize=10)
    axes[1].legend(loc='upper left',fontsize=8,frameon=False)
    kl=[road['未封顶辅助KL_model_to_ref'],road['未封顶辅助KL_ref_to_model']]
    axes[2].bar([0,1],kl,color=[blue,orange],width=.55)
    for i,v in enumerate(kl): axes[2].text(i,v+.0008,f'{v:.5f}',ha='center')
    axes[2].set(xticks=[0,1],xticklabels=['Model to reference','Reference to model'],ylim=(0,.032),ylabel='KL divergence (nats)')
    axes[2].tick_params(axis='x',labelsize=8)
    axes[2].set_title('(c) Road border: auxiliary KL',pad=27,fontsize=11)
    axes[2].text(.5,1.015,'UNCAPPED lognormal proxies only',transform=axes[2].transAxes,ha='center',fontsize=9)
    fig.suptitle('Quantile agreement and auxiliary distribution distance',fontsize=15,y=.985)
    fig.text(.065,.085,'(a) Both distributions use mean 49.5 min; reference P90 is reconstructed from TTI moments, not an observed P90.\n(b) Carballo (2016), Table 1, 2013 customs delay. The reconstruction is not the current sea-border model.\n(c) Finite KL values do not describe the capped code distribution; no universal pass threshold is applied.',fontsize=9,color=grey,linespacing=1.55)
    save(fig,'CV_02_P90_AND_KL')

    # Analytic densities, not fabricated empirical observations or KDE fits.
    cv=road['输入CV']; cap=data['profile']['border_delay_max_factor']; mean=49.5
    from statistics import NormalDist
    normal=NormalDist(); sigma=math.sqrt(math.log1p(cv*cv))
    def moment(mu):
        z=(math.log(cap)-mu)/sigma
        return math.exp(mu+.5*sigma*sigma)*normal.cdf(z-sigma)+cap*normal.cdf(-z)
    lo,hi=-50.,math.log(cap)+50.
    for _ in range(120):
        mid=(lo+hi)/2
        if moment(mid)<1:lo=mid
        else:hi=mid
    mu=(lo+hi)/2
    refsigma=math.sqrt(math.log1p((28.9/49.5)**2)); refmu=-.5*refsigma*refsigma
    grid=np.linspace(.01,mean*cap+30,2000)
    def pdf(x,m,s):
        return np.exp(-.5*((np.log(x/mean)-m)/s)**2)/(x*s*np.sqrt(2*np.pi))
    fig, ax=plt.subplots(figsize=(7.8,4.9)); fig.subplots_adjust(bottom=.25,top=.85,left=.12,right=.96)
    mask=grid<mean*cap
    ax.plot(grid,pdf(grid,refmu,refsigma),color=orange,lw=1.7,label='Moment-matched reference')
    ax.fill_between(grid,pdf(grid,refmu,refsigma),color=orange,alpha=.18)
    ax.plot(grid[mask],pdf(grid[mask],mu,sigma),color=blue,lw=1.7,label='Current code: continuous part')
    ax.fill_between(grid[mask],pdf(grid[mask],mu,sigma),color=blue,alpha=.18)
    for val,color in zip(values,[blue,orange]):ax.axvline(val,color=color,ls='--',lw=1)
    mass=normal.cdf((mu-math.log(cap))/sigma)
    ax.axvline(mean*cap,color=grey,ls=':',lw=1,ymax=.40)
    ax.text(.98,.48,f'Cap: {mean*cap:.2f} min\nPoint mass: {mass*100:.3f}%\n(not a density height)',transform=ax.transAxes,ha='right',fontsize=9,color=grey)
    ax.set(xlabel='Road border time (minutes)',ylabel='Density (1/min)',xlim=(0,grid[-1]),ylim=(0,None))
    ax.legend(fontsize=9,frameon=False); clean(ax)
    fig.suptitle('Road border: analytic distribution comparison',fontsize=13)
    fig.text(.12,.08,f'Dashed lines: P90 = {values[0]:.2f} min (code), {values[1]:.2f} min (reference).\nAnalytic curves, not empirical fits. The capped distribution also has an atom at its cap.',fontsize=9,color=grey)
    save(fig,'CV_03_ROAD_BORDER_DENSITY')
    (out/'CAPTIONS_CN.txt').write_text(
        '图1：六类来源的当前CV与证据。铁路、海运运输及铁路口岸的CV保留假设标记；海运口岸当前均值为0，未激活。\n'
        '图2：当前设定的P90核对与辅助KL。公路参照由文献统计量重建；海关分位数是文献对照。KL仅对应未封顶辅助LN。\n'
        '图3：公路口岸的解析密度曲线，非实测拟合；封顶位置的点质量另行标注。\n'
        '本次只绘图，不包含敏感性实验、重新搜索或训练。数据和原结果不变。\n',encoding='utf-8')
    (out/'SOURCES.json').write_text(json.dumps(data['sources'],ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    if sha(source)!=source_sha:raise RuntimeError('Source changed during plotting')
    (out/'STATUS.json').write_text(json.dumps(dict(completed=True,**provenance),indent=2)+'\n')
    print('[CV-PLOT] COMPLETE',out,flush=True)


if __name__=='__main__':
    main()
