"""Descriptive paired early-budget results, separate from full-study selection."""
from bridge_fast import *
from reporting import csv_write,union_hv

LABELS={'BASE':'Base','ICM_INDEPENDENT':'ICM 独立 Transformer','ICM_SHARED_SUPERVISED':'ICM 共享监督 Transformer','ICM_TERMINAL_PPO':'ICM 终局PPO（固定更新2）'}


def table(rows,methods):
    index={(r['instance'],r['seed'],r['arm']):r for r in rows};result=[]
    for method in methods:
        own=[r for r in rows if r['arm']==method]
        row={'方法':LABELS[method],'运行数':len(own),'平均HV':float(np.mean([r['final_hypervolume'] for r in own]))}
        for ref,label in [('BASE','Base'),('ICM_INDEPENDENT','独立训练'),('ICM_SHARED_SUPERVISED','共享监督')]:
            if ref not in methods:continue
            delta=[r['final_hypervolume']-index[r['instance'],r['seed'],ref]['final_hypervolume'] for r in own]
            base=[index[r['instance'],r['seed'],ref]['final_hypervolume'] for r in own]
            row['相对'+label+'平均HV差']=float(np.mean(delta))
            row['相对'+label+'平均提升(%)']=float(np.mean([100*d/b for d,b in zip(delta,base)])) if all(b>0 for b in base) else None
            row['胜'+label]=sum(d>1e-12 for d in delta);row['平'+label]=sum(abs(d)<=1e-12 for d in delta);row['负'+label]=sum(d< -1e-12 for d in delta)
        row['平均搜索时间(秒)']=float(np.mean([r['optimisation_seconds'] for r in own]));result.append(row)
    return result


def report(rows,plan,out,methods):
    expected={(r['instance_id'],plan['seed'],m) for r in plan['panel'] for m in methods}
    if len(rows)!=len(expected) or {(r['instance'],r['seed'],r['arm']) for r in rows}!=expected:raise ValueError('Missing or duplicate fast-check runs')
    for r in rows:
        if r['phase']!='validation' or r['stochastic'] or r['hidden_candidate_objective_evaluations'] or r['ppo_updates']:raise ValueError('Fast check must be frozen validation')
        if r['evaluation_budget']!=plan['settings']['budget'] or not plan['settings']['budget']-1<=r['real_objective_evaluation_count']<=plan['settings']['budget']:raise ValueError('Fast check budget mismatch')
    for n in {r['instance'] for r in rows}:
        if len({(r['scenario_digest'],r['instance_digest']) for r in rows if r['instance']==n})!=1:raise ValueError('Broken pairing')
    main=table(rows,methods);out.mkdir(parents=True,exist_ok=True)
    csv_write(out/'00_当前小试对比.csv',main)
    per=[]
    for cfg in plan['fast']['configurations']:
        per.extend({'配置':cfg,**r} for r in table([r for r in rows if r['configuration']==cfg],methods))
    csv_write(out/'PER_CONFIGURATION.csv',per)
    csv_write(out/'PER_RUN.csv',[{k:r[k] for k in ('configuration','instance','seed','arm','final_hypervolume','optimisation_seconds','real_objective_evaluation_count','peak_memory_mib')} for r in rows])
    diagnostics=[]
    for r in rows:
        for c in COMPONENTS:diagnostics.append({'instance':r['instance'],'method':r['arm'],'component':c,**r['selection'][c]})
    csv_write(out/'EXECUTION_DIAGNOSTICS.csv',diagnostics)
    final=methods==plan['fast']['methods']
    if final:
        normalized,bounds=union_hv(rows);csv_write(out/'01_统一界限敏感性检查.csv',table(normalized,methods));write_json(out/'INSTANCE_BOUNDS.json',bounds)
    result={'completed':final,'available_methods':methods,'fixed_ppo_update':plan['fast']['ppo_update'],
        'population':plan['settings']['population'],'budget':plan['settings']['budget'],'progress_denominator':plan['settings']['progress_denominator'],
        'completed_searches':len(rows),'objective_evaluations':sum(r['real_objective_evaluation_count'] for r in rows),
        'new_training_steps':0,'new_ppo_updates':0,'hidden_candidate_objective_evaluations':0,'final_test_opened':False,
        'qualification':'Three already-used validation instances, one seed each, B5000. Descriptive early-budget evidence only; no confidence interval or checkpoint selection. Full-budget study continues independently and must not be retuned from this panel.'}
    write_json(out/'QUICK_RESULT.json',result)
    print('\n'+(out/'00_当前小试对比.csv').read_text(encoding='utf-8-sig'),flush=True)
    return result
