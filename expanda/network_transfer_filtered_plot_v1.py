#!/usr/bin/env python3
"""Display-only time filtering of a saved small-network Pareto front."""
import argparse
import importlib.util
import math
import os
from pathlib import Path
import sys

sys.dont_write_bytecode = True
EXP = Path(__file__).resolve().parent
sys.path.insert(0, str(EXP/'network_transfer_b5000_v1'))
from inputs import read, write, freeze, digest, verify, csv_write, SHARED, SEEDS
import numpy as np


def masks_for(fronts, cutoff):
    if not math.isfinite(cutoff) or cutoff < 0:
        raise ValueError('Time cutoff must be a finite nonnegative number')
    return {'small':fronts['small'][:,2] >= cutoff,
            'large':np.ones(len(fronts['large']),dtype=bool)}


def plot(out, fronts, masks, cutoff, configuration, seed, rule):
    os.environ['MPLCONFIGDIR'] = str(out/'.matplotlib')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,
                         'pdf.fonttype':42,'svg.fonttype':'none'})
    colors={'small':'#2878b5','large':'#e07b39'}
    scale=np.array([1e6,1e9,1.])
    labels=('Cost P90 (million USD)',r'Emissions P90 (billion g CO$_2$)','Completion time P90 (h)')
    fig,axes=plt.subplots(1,3,figsize=(13.,4.3))
    for ax,(i,j) in zip(axes,((0,1),(0,2),(1,2))):
        for net in ('small','large'):
            pts=fronts[net][masks[net]]/scale
            ax.scatter(pts[:,i],pts[:,j],s=24,color=colors[net],alpha=.62,linewidths=.4)
        ax.set_xlabel(labels[i]);ax.set_ylabel(labels[j])
        ax.ticklabel_format(style='plain',useOffset=False);ax.grid(alpha=.12)
        if j==2:
            ax.axhline(cutoff,color='#888888',ls=':',lw=1.,zorder=0)
    shown=int(masks['small'].sum());total=len(masks['small'])
    label='large-network minimum' if rule=='large_minimum' else 'fixed cutoff'
    note=('Display filter only: small-network time P90 >= '+format(cutoff,'.6g')+' h ('+label+').\n'
          +'Small network: '+str(shown)+'/'+str(total)+' points shown; '+str(total-shown)+' hidden. '
          +'Large network: all '+str(len(fronts['large']))+' points shown.')
    fig.legend(handles=[Line2D([],[],marker='o',ls='',color=colors['small'],label='Small network (filtered)'),
                        Line2D([],[],marker='o',ls='',color=colors['large'],label='Large network (all points)')],
               loc='upper center',bbox_to_anchor=(.5,1.),ncol=2,frameon=False)
    fig.text(.5,.025,note,ha='center',va='bottom',fontsize=9)
    fig.tight_layout(rect=(0,.17,1,.90))
    stem=configuration+'_FILTERED_PARETO_seed'+str(seed)
    for ext in ('png','pdf','svg'):
        fig.savefig(out/(stem+'.'+ext),dpi=300,bbox_inches='tight')
    plt.close(fig)
    return stem


def execute(args):
    root=args.root.resolve()
    status=read(root/'STATUS.json')
    if not status.get('completed') or status.get('done')!=27:
        raise ValueError('Expected the completed 27-run network experiment')
    spec=importlib.util.spec_from_file_location('network_filtered_runtime',EXP/'network_transfer_b5000_v1/run.py')
    runtime=importlib.util.module_from_spec(spec);spec.loader.exec_module(runtime)
    plan=runtime.checked_plan(root/'PLAN.json');verify(plan['files'])
    files={str(root/n):digest(root/n) for n in ('PLAN.json','STATUS.json')}
    files[str(Path(__file__).resolve())]=digest(Path(__file__).resolve())
    entries={e['network']:e for e in plan['entries'] if e['configuration']==args.configuration}
    fronts={};solutions={}
    for net in ('small','large'):
        e=entries[net]
        job=next(j for j in runtime.jobs(plan) if (j['instance'],j['arm'],j['seed'])==(e['instance_id'],SHARED,args.seed))
        folder=runtime.destination(root,job);path=folder/'RUN_COMPLETE.json';rr=read(path)
        runtime.validate_run(rr,plan,job);files[str(path)]=digest(path)
        path=folder/'final_feasible_nondominated.json';files[str(path)]=digest(path)
        if files[str(path)]!=rr['final_solutions_sha256']:raise ValueError('Final solution archive changed')
        solutions[net]=read(path)
        pts=[[s['optimisation_objectives'][k] for k in ('cost','emission','makespan')] for s in solutions[net]]
        if not pts or pts!=rr['final_front_raw']:raise ValueError('Final front and saved solutions disagree')
        fronts[net]=np.asarray(pts,dtype=float)
        if fronts[net].shape[1]!=3 or not np.isfinite(fronts[net]).all():raise ValueError('Invalid objective values')
    large_min=float(fronts['large'][:,2].min())
    cutoff=large_min if args.use_large_min else (555. if args.cutoff is None else args.cutoff)
    rule='large_minimum' if args.use_large_min else 'fixed_cutoff'
    masks=masks_for(fronts,cutoff)
    tag='large_min' if args.use_large_min else format(cutoff,'.12g').replace('.','p')+'h'
    out=args.out.resolve() if args.out else root.with_name(root.name+'_filtered_'+tag+'_v1')
    if root==out or root in out.parents or out in root.parents:raise ValueError('Use a separate output directory')
    out.mkdir(parents=True,exist_ok=True)
    protocol=dict(source=str(root),configuration=args.configuration,seed=args.seed,files=files,
        filter_rule=rule,small_time_lower_bound_h=cutoff,large_minimum_time_h=large_min,
        small_original_points=len(fronts['small']),small_displayed_points=int(masks['small'].sum()),
        small_hidden_points=int((~masks['small']).sum()),large_points=len(fronts['large']),
        large_all_points_displayed=True,source_statistics_unchanged=True,
        new_searches=0,new_objective_evaluations=0,training_updates=0)
    freeze(out/'FILTER_MANIFEST.json',protocol)
    rows=[]
    for net in ('small','large'):
        for index,p in enumerate(fronts[net]):
            rows.append(dict(network=net,source_solution_id=solutions[net][index]['source_solution_id'],
                cost_P90=float(p[0]),emission_P90=float(p[1]),time_P90_h=float(p[2]),
                displayed=bool(masks[net][index]),reason='shown' if masks[net][index] else 'small_time_below_display_cutoff'))
    csv_write(out/'POINTS_AND_DISPLAY_MASK.csv',rows)
    stem=plot(out,fronts,masks,cutoff,args.configuration,args.seed,rule)
    (out/'CAPTION_CN.txt').write_text(
        f'{args.configuration}，种子{args.seed}，冻结单模型共享监督，原搜索预算5000次。\n'
        f'仅为筛选展示：隐藏小网络完成时间P90严格小于{cutoff:.12g}小时的点；大网络全部点保留。\n'
        f'阈值规则：{rule}。大网络实际最短完成时间P90为{large_min:.12g}小时，不默认等同于555小时。\n'
        f'小网络显示{int(masks["small"].sum())}/{len(fronts["small"])}个点，隐藏{int((~masks["small"]).sum())}个点。\n'
        '这不是小网络完整前沿；不能用筛选后的图证明大网络在全部权衡上支配小网络。\n'
        '原始点、HV、胜平负、运行时间均未改动；没有重新选折中点，图中不画折中星号。\n',encoding='utf-8')
    verify(files);verify(plan['files'])
    write(out/'STATUS.json',dict(completed=True,figure=stem+'.png',new_searches=0,new_objective_evaluations=0,training_updates=0))
    print('[NETWORK-FILTER] large minimum time P90 (h): '+format(large_min,'.12g'),flush=True)
    print('[NETWORK-FILTER] small-network display cutoff (h): '+format(cutoff,'.12g'),flush=True)
    print('[NETWORK-FILTER] small points shown/total: '+str(int(masks['small'].sum()))+'/'+str(len(fronts['small'])),flush=True)
    print('[NETWORK-FILTER] COMPLETE '+str(out),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=EXP/'transformer_ppo_runs/diagnostics/ccp_network_transfer_single_b5000_v1')
    parser.add_argument('--out',type=Path)
    parser.add_argument('--configuration',choices=('B10','B20','B30'),default='B30')
    parser.add_argument('--seed',type=int,choices=SEEDS,default=SEEDS[0])
    group=parser.add_mutually_exclusive_group()
    group.add_argument('--cutoff',type=float,help='Hide small-network points with time P90 strictly below this value (default 555 h)')
    group.add_argument('--use-large-min',action='store_true',help='Use the actual large-front minimum time instead of 555 h')
    execute(parser.parse_args())
