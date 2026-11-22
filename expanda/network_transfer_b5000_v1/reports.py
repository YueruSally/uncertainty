"""Paired topology reports; empirical HV/1.331; never replaces historical tables."""
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
import math

from inputs import read,write,csv_write,BUDGETS,SHARED,SEEDS

LABELS={('small','BASE'):'小网络 Base',('small',SHARED):'小网络 共享监督（单模型）',
        ('large',SHARED):'大网络 共享监督（单模型）'}


def create_report(root,plan,records,p,preview=False):
    np=p.np; destination=root/('preview' if preview else 'results')
    destination.mkdir(exist_ok=True)
    configurations=sorted({r['configuration'] for r in records},key=lambda x:int(x[1:]))
    if len(records)!=9*len(configurations):raise ValueError('Report requires complete network/method/seed blocks')
    keys={(r['configuration'],r['network'],r['arm'],r['seed']) for r in records}
    if len(keys)!=len(records):raise ValueError('Duplicate run')
    for cfg in configurations:
        for net,arm in LABELS:
            if {r['seed'] for r in records if (r['configuration'],r['network'],r['arm'])==(cfg,net,arm)}!=set(SEEDS):
                raise ValueError('Missing paired seed')
        if len({r['batch_signature'] for r in records if r['configuration']==cfg})!=1:
            raise ValueError('Unpaired demand across networks')
    bounds={}; per_run=[]; lookup={}
    for cfg in configurations:
        rr=[r for r in records if r['configuration']==cfg]
        points=[pt for r in rr for snap in r['budget_fronts'] for pt in snap['front_raw']]
        if points:
            array=np.asarray(points,dtype=float)
            if array.ndim!=2 or array.shape[1]!=3 or not np.isfinite(array).all():raise ValueError('Invalid front objectives')
            lo=array.min(axis=0);hi=array.max(axis=0)
        else:lo=np.zeros(3);hi=np.ones(3)
        span=np.where(hi>lo,hi-lo,1.)
        bounds[cfg]=dict(minimum=lo.tolist(),maximum=hi.tolist(),reference=[1.1]*3,hv_divisor=1.331,
                        all_fronts_empty=not bool(points),scope=plan['reporting']['bounds'])
        for r in rr:
            for snap in sorted(r['budget_fronts'],key=lambda x:x['evaluation']):
                pts=[((np.asarray(pt)-lo)/span).tolist() for pt in snap['front_raw']]
                hv=p.hv.hv3(pts)/1.331 if pts else 0.
                if not math.isfinite(hv) or hv < -1e-12 or hv>1+1e-12:raise ValueError('HV outside normalized reference box')
                row=dict(configuration=cfg,instance=r['pair_id'],network=r['network'],method=r['arm'],seed=r['seed'],
                    evaluation=snap['evaluation'],hv=hv,front_size=len(pts),
                    checkpoint_seconds=snap.get('controller_seconds'),
                    search_seconds=r['optimisation_seconds'],inference_seconds=r['inference_seconds'],
                    path_library_sha256=r['path_library_sha256'],scenarios_sha256=r['cached_scenarios_sha256'])
                per_run.append(row);lookup[cfg,r['network'],r['arm'],r['seed'],snap['evaluation']]=row
    write(destination/'INSTANCE_BOUNDS.json',bounds);csv_write(destination/'PER_RUN.csv',per_run)
    main=[];per_cfg=[];paired_rows=[];comparisons={}
    for own,ref,label in [(('small',SHARED),('small','BASE'),'small_shared_vs_base'),
                          (('large',SHARED),('small',SHARED),'large_vs_small_shared')]:
        summary=[]
        for budget in BUDGETS:
            pairs=[]
            for cfg in configurations:
                for seed in SEEDS:
                    a=lookup[(cfg,)+own+(seed,budget)];b=lookup[(cfg,)+ref+(seed,budget)]
                    delta=a['hv']-b['hv']
                    pair=dict(configuration=cfg,instance=a['instance'],seed=seed,evaluation=budget,
                        own=a['hv'],reference=b['hv'],absolute=delta,
                        percent=100*delta/b['hv'] if b['hv']>0 else None,
                        checkpoint_seconds=a['checkpoint_seconds'],search_seconds=a['search_seconds'])
                    pairs.append(pair);paired_rows.append(dict(comparison=label,**pair))
            for scope,values in [('ALL',pairs)]+[(c,[v for v in pairs if v['configuration']==c]) for c in configurations]:
                plo,phi=p.hv.interval(values,'percent',3000);alo,ahi=p.hv.interval(values,'absolute',3000)
                times=[v['search_seconds'] if budget==5000 else v['checkpoint_seconds'] for v in values]
                tm=mean(times) if all(x is not None for x in times) else None
                ts=stdev(times) if tm is not None and len(times)>1 else None
                tol=1e-12/1.331
                r=dict(范围=scope,评价次数=budget,方法=LABELS[own],参照=LABELS[ref],配对数=len(values),
                    平均HV=p.hv.equal_mean(values,'own'),参照平均HV=p.hv.equal_mean(values,'reference'),
                    平均配对提升百分比=p.hv.equal_mean(values,'percent'),提升95区间下限=plo,提升95区间上限=phi,
                    平均HV差=p.hv.equal_mean(values,'absolute'),HV差95区间下限=alo,HV差95区间上限=ahi,
                    胜=sum(v['absolute']>tol for v in values),平=sum(abs(v['absolute'])<=tol for v in values),
                    负=sum(v['absolute']< -tol for v in values),平均运行秒=tm,运行秒标准差=ts)
                (summary if scope=='ALL' else per_cfg).append(r)
            comparisons[label+'/'+str(budget)]={'per_run':pairs}
        csv_write(destination/('01_小网络共享监督相对Base.csv' if label=='small_shared_vs_base' else '02_大网络相对小网络_共享监督.csv'),summary)
    for budget in BUDGETS:
        for arm,label in LABELS.items():
            rows=[r for r in per_run if r['evaluation']==budget and (r['network'],r['method'])==arm]
            times=[r['search_seconds'] if budget==5000 else r['checkpoint_seconds'] for r in rows]
            main.append(dict(评价次数=budget,方法=label,运行数=len(rows),平均HV=mean(r['hv'] for r in rows),
                平均运行秒=mean(times) if all(t is not None for t in times) else None,
                运行秒标准差=stdev(times) if all(t is not None for t in times) and len(times)>1 else None,
                平均终点推理秒=mean(r['inference_seconds'] for r in rows) if budget==5000 else None))
    csv_write(destination/'RESULTS.csv',main);csv_write(destination/'PER_CONFIGURATION.csv',per_cfg)
    csv_write(destination/'PER_PAIR.csv',paired_rows)
    # Predeclared representative: each search's final point nearest the origin
    # in the shared normalized three-objective space. No preference fitted to results.
    compromise=[]
    for r in records:
        pts=r['final_front_raw'];bb=bounds[r['configuration']]
        if not pts:continue
        lo=np.asarray(bb['minimum']);span=np.asarray(bb['maximum'])-lo;span=np.where(span>0,span,1.)
        idx=min(range(len(pts)),key=lambda i:(float(np.sum(((np.asarray(pts[i])-lo)/span)**2)),tuple(pts[i])))
        compromise.append(dict(configuration=r['configuration'],network=r['network'],method=r['arm'],seed=r['seed'],
            cost_P90=pts[idx][0],emission_P90=pts[idx][1],makespan_P90_h=pts[idx][2],
            selection='Minimum Euclidean distance to shared normalized origin; equal objective weights'))
    if compromise:csv_write(destination/'COMPROMISE_P90.csv',compromise)
    write(destination/'QUICK_RESULT.json',dict(completed=not preview,preview=preview,comparisons=comparisons,
        reporting=plan['reporting'],normalization_bounds=bounds,model_bundle_sha256=plan['model_bundle_sha256'],
        training_updates=0,preparation_seconds=plan['preparation_seconds'],
        total_search_seconds=sum(r['optimisation_seconds'] for r in records),runs=len(records),
        interpretation='Network contrast measures the frozen policy under the fixed budget, not a theoretical network optimum.'))
    plot(destination,per_run,records,configurations)
    print('[NETWORK] report ready: '+str(destination),flush=True)


def plot(destination,rows,records,configs):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors=['#666666','#2678b2','#cf6b24']
    labels=['Small: Base','Small: shared SL','Large: shared SL']
    fig,axes=plt.subplots(1,len(configs),figsize=(4.3*len(configs),3.6),squeeze=False)
    for ax,cfg in zip(axes[0],configs):
        for (key,_),color,label in zip(LABELS.items(),colors,labels):
            y=[mean(r['hv'] for r in rows if r['configuration']==cfg and
                    (r['network'],r['method'])==key and r['evaluation']==b) for b in BUDGETS]
            ax.plot(BUDGETS,y,'o-',color=color,label=label,lw=1.6)
        ax.set_title(cfg[1:]+' batches');ax.set_xlabel('Objective evaluations');ax.set_ylabel('HV / 1.331')
        ax.set_xticks(BUDGETS);ax.set_xticklabels(['400','1,000','5,000']);ax.grid(alpha=.18);ax.legend(fontsize=8)
    fig.tight_layout()
    for ext in ('png','pdf'):fig.savefig(destination/f'HV_PROGRESS.{ext}',dpi=220,bbox_inches='tight')
    plt.close(fig)
    fig,axes=plt.subplots(len(configs),3,figsize=(12,3.1*len(configs)),squeeze=False)
    objective_labels=['Cost P90','Emission P90 (g CO2)','Completion time P90 (h)']
    for axs,cfg in zip(axes,configs):
        for ax,(i,j) in zip(axs,[(0,1),(0,2),(1,2)]):
            for key,color,label in zip(LABELS,colors,labels):
                r=next(r for r in records if r['configuration']==cfg and r['seed']==SEEDS[0] and (r['network'],r['arm'])==key)
                pts=r['final_front_raw']
                ax.scatter([x[i] for x in pts],[x[j] for x in pts],s=10,alpha=.45,color=color,label=label)
            ax.set_xlabel(objective_labels[i]);ax.set_ylabel(objective_labels[j]);ax.set_title(cfg[1:]+' batches')
            ax.ticklabel_format(style='sci',axis='both',scilimits=(-3,4));ax.legend(fontsize=7)
    fig.tight_layout()
    for ext in ('png','pdf'):fig.savefig(destination/f'FRONT_PROJECTIONS_seed{SEEDS[0]}.{ext}',dpi=220,bbox_inches='tight')
    plt.close(fig)
