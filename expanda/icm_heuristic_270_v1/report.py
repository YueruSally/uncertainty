"""Read-only 240+30 front union and descriptive statistics. No new evaluations."""
from pathlib import Path
import csv
import hashlib
import json
import statistics
import sys

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'hv_instance_union_v1'))
import recompute as hv

PROTOCOL='original-240-plus-icm-heuristic-v1'
RULE='I_C_M_K8_HEURISTIC'
METHODS=(*hv.METHODS,RULE)
original_design=hv.experiment_design


def design(plan,path):
    if plan.get('config',{}).get('protocol')!=PROTOCOL:return original_design(plan,path)
    identity=dict(plan);sha=identity.pop('sha256')
    actual=hashlib.sha256(json.dumps(identity,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
    if actual!=sha:raise ValueError('Extension plan changed')
    source=Path(plan['source_root']);old_path=source/'PLAN.json'
    old=original_design(hv.read_json(old_path),old_path)
    if (old['protocol']!='icm-common-seed-component-screen-v1' or old['plan_sha']!=plan['source_plan_sha256']
        or plan['instances']!=old['instances'] or plan['config']['algorithm_seeds']!=old['algorithm_seeds']
        or tuple(plan['config']['methods'])!=METHODS or plan['runs']!=270 or plan['new_runs']!=30):
        raise ValueError('Expected original 240-run grid plus only 30 heuristic runs')
    for filename,expected in plan['original_result_files'].items():
        if hv.digest(Path(filename))!=expected:raise ValueError(f'Original result changed: {filename}')
    return dict(instances=old['instances'],algorithm_seeds=old['algorithm_seeds'],methods=METHODS,
        plan_sha=sha,base_bootstrap_seed=931100,rule_bootstrap_seed=931500,protocol=PROTOCOL,
        reused_source=str(source),reused_plan_sha=old['plan_sha'],reused_methods=list(old['methods']))


hv.experiment_design=design
LABELS={'BASE':'Base','I_DIRECTED_BALANCED':'I 启发式','I_K8_RESIDUAL':'I Transformer',
        'C_K8_SIMPLE':'C 启发式','C_K8_RESIDUAL':'C Transformer','M_BALANCED':'M 启发式',
        'M_K8_TRANSFORMER':'M Transformer',RULE:'ICM 启发式','I_C_M_K8_TRANSFORMER':'ICM Transformer'}


def statistics_tables(out,rows,summary):
    index={(r['instance'],r['algorithm_seed'],r['method']):r for r in rows}
    table=[];per_instance=[]
    for m in METHODS:
        chosen=[r for r in rows if r['method']==m]
        for ref in ('BASE',hv.RULES[m]) if m in hv.RULES else ('BASE',):
            pairs=[]
            for r in chosen:
                b=index[r['instance'],r['algorithm_seed'],ref]
                pairs.append(dict(instance=r['instance'],hv=r['hv'],absolute=r['hv']-b['hv'],
                    percent=100*(r['hv']/b['hv']-1) if b['hv']>0 else None,seconds=r['seconds']))
            comparison=summary['methods'][m]['vs_base' if ref=='BASE' else 'vs_rule']
            p=comparison['percent'];a=comparison['absolute']
            values=[v['percent'] for v in pairs if v['percent'] is not None]
            table.append({'方法':LABELS[m],'参照方法':LABELS[ref],'运行对数':len(pairs),
                '平均HV':statistics.fmean(v['hv'] for v in pairs),'HV样本标准差_跨30次':statistics.stdev(v['hv'] for v in pairs),
                '平均配对提升(%)':p['mean'],'配对提升样本标准差_跨30对(百分点)':statistics.stdev(values) if len(values)==len(pairs) else None,
                '提升95%区间下限(%)':p['hierarchical_bootstrap_95ci_low'],'提升95%区间上限(%)':p['hierarchical_bootstrap_95ci_high'],
                '平均HV绝对差':a['mean'],'绝对差95%区间下限':a['hierarchical_bootstrap_95ci_low'],
                '绝对差95%区间上限':a['hierarchical_bootstrap_95ci_high'],
                '胜':a['wins'],'平':a['ties'],'负':a['losses'],'平均单次运行时间(秒)':statistics.fmean(v['seconds'] for v in pairs)})
            for n in sorted({v['instance'] for v in pairs}):
                v=[x for x in pairs if x['instance']==n];percent=[x['percent'] for x in v if x['percent'] is not None]
                per_instance.append({'实例':n,'方法':LABELS[m],'参照方法':LABELS[ref],'种子数':len(v),
                    '平均HV':statistics.fmean(x['hv'] for x in v),'种子间HV样本标准差':statistics.stdev(x['hv'] for x in v),
                    '平均配对提升(%)':statistics.fmean(percent) if len(percent)==len(v) else None,
                    '种子间提升样本标准差(百分点)':statistics.stdev(percent) if len(percent)==len(v) else None,
                    '胜':sum(x['absolute']>1e-12 for x in v),'平':sum(abs(x['absolute'])<=1e-12 for x in v),
                    '负':sum(x['absolute']< -1e-12 for x in v)})
    hv.write_csv(out/'STATISTICS.csv',table);hv.write_csv(out/'PER_INSTANCE_STATISTICS.csv',per_instance)
    groups=[('I','I_DIRECTED_BALANCED','I_K8_RESIDUAL'),('C','C_K8_SIMPLE','C_K8_RESIDUAL'),
            ('M','M_BALANCED','M_K8_TRANSFORMER'),('ICM',RULE,'I_C_M_K8_TRANSFORMER')]
    times={m:statistics.fmean(r['seconds'] for r in rows if r['method']==m) for m in METHODS}
    display=[{'组件':'Base','启发式(%)':'0.00','Transformer(%)':'0.00','Time(秒；启发式 / Transformer)':f'{times["BASE"]:.2f}'}]
    def fmt(v):return '' if v is None else f'{v:.2f}'
    for label,r,l in groups:
        display.append({'组件':label,'启发式(%)':fmt(summary['methods'][r]['vs_base']['percent']['mean']),
            'Transformer(%)':fmt(summary['methods'][l]['vs_base']['percent']['mean']),
            'Time(秒；启发式 / Transformer)':f'{times[r]:.2f} / {times[l]:.2f}'})
    hv.write_csv(out/'00_组件对比及运行时间.csv',display)
    (out/'统计口径.txt').write_text('本表为旧3实例×10种子×9方法，共270次；仅ICM启发式新增30次。\n'
        '九组共同拟合每实例min/max，原八组也全部重算，不混用900次扩大实验。\n'
        '标准差采用n-1分母；跨30次/对的标准差混合实例与种子差异，只作描述。分实例文件单列各实例10种子的标准差。\n'
        '置信区间保留实例与配对种子的分层bootstrap，先实例内汇总，再实例等权；仅有3个开发实例。\n'
        '胜平负按HV绝对差与1e-12容差判断。相对Base与相对启发式是两种独立配对比较。\n'
        'Base或参照HV为零时总体百分比留空，报告绝对差，不加小常数。未另外新增显著性检验。\n'
        'Time是平均单次搜索秒数；保留原八组记录，新组采用原计划的3个并发进程，但运行时段和机器负载可能不同。\n',encoding='utf-8')


def execute(source,out):
    result=hv.execute(source,out)
    _,rows,_=hv.load_runs(source)
    bounds=hv.read_json(out/'INSTANCE_BOUNDS.json')['instances']
    for r in rows:r['hv']=hv.exact_hv_3d(hv.normalize(r['points'],bounds[r['instance']]),hv.REFERENCE)
    statistics_tables(out,rows,result)
    manifest=hv.read_json(out/'INPUT_MANIFEST.json');manifest['files'][str(Path(__file__).resolve())]=hv.digest(Path(__file__))
    hv.write_json(out/'INPUT_MANIFEST.json',manifest)
    return result
