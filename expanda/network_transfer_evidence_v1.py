#!/usr/bin/env python3
"""Read-only evidence addendum for the completed 27-run network experiment."""
import argparse
import csv
import importlib.util
import math
import os
from pathlib import Path
from statistics import mean
import sys

sys.dont_write_bytecode = True
EXP = Path(__file__).resolve().parent
sys.path.insert(0, str(EXP/'network_transfer_b5000_v1'))
from inputs import read, write, freeze, digest, verify, csv_write, SHARED, SEEDS


def csv_read(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def vector(solution):
    return [solution['optimisation_objectives'][k] for k in ('cost', 'emission', 'makespan')]


def cargo_usage(solution, entry, small_arcs, large_arcs):
    """Count each allocated TEU once per node/mode/any-new category, not per leg."""
    small_nodes = {v for k in small_arcs for v in k[:2]}
    nodes = {v for k in large_arcs for v in k[:2]}
    metrics = {('node', n): 0. for n in sorted(nodes)}
    metrics.update({('mode', m): 0. for m in ('road', 'rail', 'water')})
    metrics.update({('network', k): 0. for k in ('any_new_arc', 'any_new_node')})
    batches = {b['batch_id']: b for b in entry['batches']}
    if len(solution['decision']) != len(batches):
        raise ValueError('Incomplete allocation in final solution')
    seen = set()
    for decision in solution['decision']:
        batch = batches[decision['batch_id']]
        if batch['batch_id'] in seen or (decision['origin'], decision['destination']) != (batch['origin'], batch['destination']):
            raise ValueError('Duplicate or mismatched batch')
        seen.add(batch['batch_id'])
        allocations = decision['allocations']
        if not math.isclose(sum(a['share'] for a in allocations), 1., abs_tol=1e-8):
            raise ValueError('Allocation shares do not sum to one')
        for a in allocations:
            if a['share'] < 0 or len(a['nodes']) != len(a['modes'])+1:
                raise ValueError('Invalid saved route')
            arcs = set(zip(a['nodes'][:-1], a['nodes'][1:], a['modes']))
            if not arcs <= large_arcs or (entry['network']=='small' and not arcs <= small_arcs):
                raise ValueError('Saved route contains an unavailable arc')
            amount = batch['quantity']*a['share']
            for n in set(a['nodes']): metrics['node', n] += amount
            for m in set(a['modes']): metrics['mode', m] += amount
            if arcs-small_arcs: metrics['network', 'any_new_arc'] += amount
            if set(a['nodes'])-small_nodes: metrics['network', 'any_new_node'] += amount
    total = sum(b['quantity'] for b in batches.values())
    return [dict(kind=kind, item=item, cargo_TEU=amount, total_cargo_TEU=total,
                 cargo_percent=100*amount/total, new_node=(item not in small_nodes) if kind=='node' else '')
            for (kind, item), amount in metrics.items()]


def plots(out, records, selected, usage, configuration, seeds):
    os.environ['MPLCONFIGDIR'] = str(out/'.matplotlib')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':10, 'pdf.fonttype':42, 'svg.fonttype':'none'})
    seed = seeds[0]  # Fixed in the original protocol; never chosen by improvement.
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.8))
    labels = ('Cost P90 (USD)', r'Emissions P90 (g CO$_2$)', 'Completion time P90 (h)')
    for ax, (i, j) in zip(axes, ((0,1), (0,2), (1,2))):
        for net, color in (('small','#2678b2'), ('large','#cf6b24')):
            rr = next(r for r in records if (r['configuration'],r['network'],r['seed'],r['arm']) == (configuration,net,seed,SHARED))
            pts = np.asarray(rr['final_front_raw'])
            ax.scatter(pts[:,i], pts[:,j], s=14, alpha=.45, color=color, label=net.capitalize()+' network')
            point = selected[configuration, net, seed]['objectives']
            ax.scatter(point[i], point[j], marker='*', s=105, color=color, edgecolors='black', linewidths=.5)
        ax.set_xlabel(labels[i]); ax.set_ylabel(labels[j]); ax.legend(fontsize=8)
        ax.ticklabel_format(style='sci', axis='both', scilimits=(-3,4), useOffset=False)
    fig.tight_layout()
    for ext in ('png','pdf','svg'):
        fig.savefig(out/f'{configuration}_FRONT_PROJECTIONS_seed{seed}.{ext}', dpi=300, bbox_inches='tight')
    plt.close(fig)
    rows = [r for r in usage if r['configuration']==configuration and r['network']=='large'
            and (r['kind']=='network' or (r['kind']=='node' and r['new_node']))]
    items = sorted({(r['kind'],r['item']) for r in rows})
    display = {'any_new_arc':'Any added arc', 'any_new_node':'Any added node'}
    fig, ax = plt.subplots(figsize=(7.2, max(4., .32*len(items)+1.)))
    for pos, (kind,item) in enumerate(items):
        vv = [r['cargo_percent'] for r in rows if (r['kind'],r['item'])==(kind,item)]
        ax.barh(pos, mean(vv), color='#b9d9ec', height=.65)
        for index, value in enumerate(vv):
            ax.plot(value, pos+(index-(len(vv)-1)/2)*.13, 'o', ms=3.6, color='#2678b2')
    ax.set_yticks(range(len(items)))
    ax.set_yticklabels([display.get(item,item) for _,item in items])
    ax.set_xlabel('Allocated cargo using the node or an added arc (%)'); ax.set_xlim(0,100)
    ax.invert_yaxis(); fig.tight_layout()
    for ext in ('png','pdf','svg'):
        fig.savefig(out/f'{configuration}_ADDED_ROUTE_USAGE.{ext}', dpi=300, bbox_inches='tight')
    plt.close(fig)


def execute(args):
    root = args.root.resolve()
    out = args.out.resolve() if args.out else root.with_name(root.name+'_evidence_v1')
    if root==out or root in out.parents or out in root.parents:
        raise ValueError('Use a separate addendum directory')
    status = read(root/'STATUS.json')
    if not status.get('completed') or status['done']!=27:
        raise ValueError('Wait for the completed 27-run experiment')
    spec = importlib.util.spec_from_file_location('net_evidence_runtime', EXP/'network_transfer_b5000_v1/run.py')
    runtime = importlib.util.module_from_spec(spec); spec.loader.exec_module(runtime)
    plan = runtime.checked_plan(root/'PLAN.json'); verify(plan['files'])
    engine = runtime.engine(); np = engine.p.np
    configs = ['B10','B20','B30']
    records = []; files = {str(root/n):digest(root/n) for n in ('PLAN.json','STATUS.json')}
    files[str(Path(__file__))] = digest(Path(__file__))
    for job in runtime.jobs(plan):
        path = runtime.destination(root,job)/'RUN_COMPLETE.json'
        rr = read(path); runtime.validate_run(rr,plan,job)
        records.append(rr); files[str(path)] = digest(path)
    result_dir = root/'results'
    for name in ('INSTANCE_BOUNDS.json','PER_RUN.csv','COMPROMISE_P90.csv',
                 '01_小网络共享监督相对Base.csv','02_大网络相对小网络_共享监督.csv','PER_CONFIGURATION.csv'):
        files[str(result_dir/name)] = digest(result_dir/name)
    bounds = read(result_dir/'INSTANCE_BOUNDS.json')
    per_run = csv_read(result_dir/'PER_RUN.csv')
    stored = {(r['configuration'],r['network'],r['method'],int(r['seed']),int(r['evaluation'])):float(r['hv']) for r in per_run}
    if len(stored)!=81 or len(per_run)!=81: raise ValueError('Expected 81 unique saved HV checkpoints')
    audit = []; recomputed = {}
    for cfg in configs:
        rr = [r for r in records if r['configuration']==cfg]
        points = [pt for r in rr for sn in r['budget_fronts'] for pt in sn['front_raw']]
        if not points: raise ValueError('Empty front union: '+cfg)
        lo = np.min(points,axis=0); hi = np.max(points,axis=0)
        bb = bounds[cfg]
        if (not np.array_equal(lo,bb['minimum']) or not np.array_equal(hi,bb['maximum'])
                or bb['reference']!=[1.1]*3 or bb['hv_divisor']!=1.331):
            raise ValueError('Saved boundaries do not match the cross-network union: '+cfg)
        span = np.where(hi>lo,hi-lo,1.); errors = []
        for r in rr:
            for sn in r['budget_fronts']:
                pts = [((np.asarray(pt)-lo)/span).tolist() for pt in sn['front_raw']]
                value = engine.p.hv.hv3(pts)/1.331 if pts else 0.
                key = (cfg,r['network'],r['arm'],r['seed'],sn['evaluation'])
                errors.append(abs(value-stored[key])); recomputed[key] = value
        if max(errors)>1e-10: raise ValueError('Saved HV does not reproduce: '+cfg)
        audit.append(dict(configuration=cfg,runs=len(rr),checkpoints=len(errors),
                          joint_bounds_match=True,max_HV_recomputation_error=max(errors),
                          minimum=lo.tolist(),maximum=hi.tolist(),reference=[1.1]*3,divisor=1.331))
    wtl=[]
    for filename, own, ref in (
        ('01_小网络共享监督相对Base.csv',('small',SHARED),('small','BASE')),
        ('02_大网络相对小网络_共享监督.csv',('large',SHARED),('small',SHARED))):
        for row in csv_read(result_dir/filename):
            b=int(row['评价次数']); ds=[]; gains=[]
            for cfg in configs:
                for seed in SEEDS:
                    a=recomputed[(cfg,)+own+(seed,b)]; v=recomputed[(cfg,)+ref+(seed,b)]
                    ds.append(a-v); gains.append(100*(a-v)/v if v>0 else None)
            tol=1e-12/1.331
            counts=[sum(d>tol for d in ds),sum(abs(d)<=tol for d in ds),sum(d< -tol for d in ds)]
            if counts!=[int(row[k]) for k in ('胜','平','负')]:raise ValueError('W/T/L does not reproduce')
            if all(v is not None for v in gains) and abs(mean(gains)-float(row['平均配对提升百分比']))>1e-9:
                raise ValueError('Paired percentage does not reproduce')
            wtl.append(row)
    selected={}; objectives=[]; usage=[]; route_summary=[]
    existing_compromise={(r['configuration'],r['network'],r['method'],int(r['seed'])):r for r in csv_read(result_dir/'COMPROMISE_P90.csv')}
    for cfg in configs:
        entries={e['network']:e for e in plan['entries'] if e['configuration']==cfg}
        arc_sets={net:{tuple(k) for k in read(Path(e['input_folder'])/'SCENARIOS.json')['travel_keys']} for net,e in entries.items()}
        if not arc_sets['small']<=arc_sets['large']:raise ValueError('Small graph arcs are not a subset')
        bb=bounds[cfg]; lo=np.asarray(bb['minimum']); hi=np.asarray(bb['maximum']); span=np.where(hi>lo,hi-lo,1.)
        for net in ('small','large'):
            for seed in SEEDS:
                rr=next(r for r in records if (r['configuration'],r['network'],r['seed'],r['arm'])==(cfg,net,seed,SHARED))
                path=root/'runs'/rr['instance']/f'seed_{seed}'/SHARED/'final_feasible_nondominated.json'
                if digest(path)!=rr['final_solutions_sha256']:raise ValueError('Final allocation file changed')
                files[str(path)]=digest(path); solutions=read(path)
                if not solutions or [vector(x) for x in solutions]!=rr['final_front_raw']:
                    raise ValueError('Final objective/allocation mapping mismatch')
                idx=min(range(len(solutions)),key=lambda i:(float(np.sum(((np.asarray(vector(solutions[i]))-lo)/span)**2)),tuple(vector(solutions[i]))))
                sol=solutions[idx]; point=vector(sol)
                old=existing_compromise[cfg,net,SHARED,seed]
                if any(abs(point[j]-float(old[k]))>1e-7 for j,k in enumerate(('cost_P90','emission_P90','makespan_P90_h'))):
                    raise ValueError('Compromise selection differs from the original report')
                selected[cfg,net,seed]=dict(source_solution_id=sol['source_solution_id'],objectives=point)
                for item in cargo_usage(sol,entries[net],arc_sets['small'],arc_sets['large']):
                    usage.append(dict(configuration=cfg,network=net,seed=seed,source_solution_id=sol['source_solution_id'],**item))
        for seed in SEEDS:
            a=selected[cfg,'small',seed]['objectives']; b=selected[cfg,'large',seed]['objectives']
            for j,obj in enumerate(('cost','emission','makespan')):
                objectives.append(dict(configuration=cfg,seed=seed,objective=obj,small_P90=a[j],large_P90=b[j],
                    reduction_percent=100*(a[j]-b[j])/a[j] if a[j]>0 else None))
    objective_summary=[]
    for cfg in configs:
        for obj in ('cost','emission','makespan'):
            rows=[r for r in objectives if r['configuration']==cfg and r['objective']==obj]
            objective_summary.append(dict(configuration=cfg,objective=obj,seeds=len(rows),
                small_mean_P90=mean(r['small_P90'] for r in rows),large_mean_P90=mean(r['large_P90'] for r in rows),
                mean_paired_reduction_percent=mean(r['reduction_percent'] for r in rows) if all(r['reduction_percent'] is not None for r in rows) else None))
        for kind,item in sorted({(r['kind'],r['item']) for r in usage if r['configuration']==cfg}):
            for net in ('small','large'):
                rows=[r for r in usage if (r['configuration'],r['network'],r['kind'],r['item'])==(cfg,net,kind,item)]
                route_summary.append(dict(configuration=cfg,network=net,kind=kind,item=item,new_node=rows[0]['new_node'],
                    mean_cargo_percent=mean(r['cargo_percent'] for r in rows),min_cargo_percent=min(r['cargo_percent'] for r in rows),
                    max_cargo_percent=max(r['cargo_percent'] for r in rows),seeds=len(rows)))
    out.mkdir(parents=True,exist_ok=True)
    freeze(out/'AUDIT_INPUTS.json',dict(source=str(root),files=files,new_searches=0,new_objective_evaluations=0,
        training_updates=0,representative_seed=SEEDS[0],compromise='Original shared-bounds equal-weight normalized-distance rule'))
    write(out/'NORMALIZATION_AUDIT.json',audit)
    for name,rows in [('WTL_ALL_NODES.csv',wtl),('OBJECTIVE_P90_BY_SEED.csv',objectives),
        ('OBJECTIVE_P90_SUMMARY.csv',objective_summary),('ROUTE_USAGE_BY_SEED.csv',usage),('ROUTE_USAGE_SUMMARY.csv',route_summary)]:
        csv_write(out/name,rows)
    plots(out,records,selected,usage,'B30',SEEDS)
    (out/'CAPTION_CN.txt').write_text(
        '30批实例，5000次方案评价；前沿图固定使用原计划首个种子993001，只比较大小网络共享监督。\n'
        '坐标为原单位的三个目标P90；星号按原报告的共同归一化空间到原点的等权距离选取折中点。\n'
        '目标降幅正值表示大网络较小网络降低，逐种子配对后平均；这是折中点比较，不是HV贡献分解。\n'
        '路线使用图的柱为3个种子各自折中方案的平均货量占比，散点为每个种子。\n'
        '分流按批次货量乘以路径分配比例计；经过同一节点或方式的货量每条分流只计一次。\n'
        '不同节点和运输方式的货量占比可以重叠，不能相加当作100%。\n'
        '新增节点/弧以对齐后小网络实际弧集为参照；上海、宁波已属于小网络可使用节点。\n'
        '图表是既有搜索结果的描述，不能单凭路线使用比例判定改善的因果来源。\n',encoding='utf-8')
    verify(files)
    verify(plan['files'])
    write(out/'STATUS.json',dict(completed=True,source_runs=27,max_HV_error=max(r['max_HV_recomputation_error'] for r in audit),
        new_searches=0,new_objective_evaluations=0,training_updates=0))
    print('[NETWORK-EVIDENCE] All cross-network bounds, 81 HV checkpoints and W/T/L verified.',flush=True)
    print('[NETWORK-EVIDENCE] B30 compromise-point objective reductions:',flush=True)
    for row in objective_summary:
        if row['configuration']=='B30':print(row,flush=True)
    print('[NETWORK-EVIDENCE] COMPLETE '+str(out),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=EXP/'transformer_ppo_runs/diagnostics/ccp_network_transfer_single_b5000_v1')
    parser.add_argument('--out',type=Path)
    execute(parser.parse_args())
