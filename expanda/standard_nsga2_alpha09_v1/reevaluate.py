"""Read parent P90 packs, evaluate only new final plans, export new union HV."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
from statistics import mean, median, stdev
from addon_common import *

def assemble(root, plan, full, inv, np, ev):
    parent=Path(plan['parent_root'])
    require(read(parent/'STATUS.json')['completed'],'Parent full reevaluation incomplete')
    manifest=read(parent/'PLAN.json'); require(manifest['alpha']==.9,'Parent OOS manifest differs')
    verify(manifest['files'])
    retained={str(parent/'PLAN.json'):digest(parent/'PLAN.json'),
              str(parent/'STATUS.json'):digest(parent/'STATUS.json')}
    groups=[g for g in manifest['groups'] if g['table'] in ('Table5','Table6')]
    require(len(groups)==27,'Expected 18+9 instances')
    merged=[]
    dest=root/'oos'; dest.mkdir(exist_ok=True)
    def progress(g,tag,done,total,count,physical,elapsed):
        status(root,'standard_final_front_reevaluation',instance=g['instance'],scenario_group=tag,
               done=done,total=total,evaluation_scenarios=count,elapsed_seconds=round(elapsed,1))
    for g in groups:
        old_folder=Path(g['folder']); old=read(old_folder/'INPUTS.json')
        old_packs={tag:full.load_pack(old_folder,old,tag) for tag in TAGS}
        for f in old_folder.rglob('*'):
            if f.is_file() and (f.name in ('INPUTS.json','instance.json','COMPLETE.json') or f.suffix in ('.npz','.json')):
                retained[str(f)]=digest(f)
        tasks=[t for t in plan['tasks'] if (t['table'],t['instance'])==(g['table'],g['instance'])]
        require(len(tasks)==3 and all(completed(t) for t in tasks),'Standard panel incomplete')
        solutions, members, runs = {},[],[]
        for t in tasks:
            folder=Path(t['folder']); r=read(folder/'RUN_COMPLETE.json'); cfg=read(folder/'configuration.json')
            require(r['scenario_digest']==old['source_config']['scenario_digest'] and
                    cfg['uncertainty']==old['source_config']['uncertainty'],'Standard/main scenarios differ')
            for row in read(folder/'final_feasible_nondominated.json'):
                key=inv.exact_id(row)
                if key in solutions:
                    require(np.allclose(inv.vector(row),inv.vector(solutions[key]),rtol=full.RTOL,atol=full.ATOL),
                            'Identical standard decision has inconsistent objective values')
                solutions[key]=dict(row,exact_decision_id=key)
                members.append(dict(method=METHOD,seed=t['seed'],exact_decision_id=key,source_solution_id=row['source_solution_id']))
            runs.append(dict(table=t['table'],configuration=t['entry']['configuration'],instance=t['instance'],
                method=METHOD,seed=t['seed'],network='expanded',run_path=str(folder/'RUN_COMPLETE.json'),
                front_path=str(folder/'final_feasible_nondominated.json'),final_solutions=r['final_front_size'],
                actual_evaluations=5000,scenario_digest=r['scenario_digest']))
            retained.update(read(folder/'JOB_COMPLETE.json')['files'])
        packet=dict(table=g['table'],instance=old['instance'],source_config=old['source_config'],source_runs=runs,
            decisions=[solutions[k] for k in sorted(solutions)],memberships=members,seeds=old['seeds'],
            bounds_unit=g['bounds_unit'],alpha=.9,workbook=old['workbook'],
            pilot_scenarios=str(old_folder/'scenarios'))
        packet['sha256']=canonical(packet)
        folder=dest/'instances'/g['table']/g['instance']
        freeze(folder/'INPUTS.json',packet); freeze(folder/'instance.json',packet['instance'])
        new_group={**g,'folder':str(folder),'unique_decisions':len(solutions),'inputs_sha256':packet['sha256']}
        packs={}
        if solutions:
            for tag in TAGS:
                full.evaluate_pack(np,ev,dest,new_group,packet,tag,[new_group],.9,progress)
                packs[tag]=full.load_pack(folder,packet,tag)
                require(read(folder/'evaluations'/tag/'COMPLETE.json')['scenario_digest']==
                        read(old_folder/'evaluations'/tag/'COMPLETE.json')['scenario_digest'],
                        'Independent draws not identical between methods')
        else:
            freeze(folder/'EMPTY_FINAL_FRONTS.json',dict(all_three_runs_empty=True,hv=0,
                definition='No feasible final plan; not replaced with a previous generation or a new search'))
            packs={tag:dict(quantiles=np.empty((0,3)),coverage=np.empty((0,3))) for tag in TAGS}
        merged.append((g,old,old_packs,packet,packs))
    verify(retained)
    freeze(dest/'INPUT_MANIFEST.json',dict(files=retained,parent_plan=manifest['sha256'],
        standard_plan=plan['sha256'],alpha=.9,new_searches_this_stage=0,new_training_updates=0,
        scenario_groups='Exact saved parent draws; independent sizes, non-nested; shared by all methods/seeds'))
    for g in manifest['groups']:
        if g['table']!='Table7':continue
        old_folder=Path(g['folder']);old=read(old_folder/'INPUTS.json')
        packs={tag:full.load_pack(old_folder,old,tag) for tag in TAGS}
        merged.append((g,old,packs,None,None))
    return merged


def main(root):
    plan=plan_for(root)
    sys.path.insert(0,str(EXP/'final_front_oos_hv_v1'))
    inv=load('inventory',EXP/'final_front_oos_hv_v1/inventory.py')
    load('run',EXP/'final_front_oos_hv_v1/run.py')
    full=load('standard_final_front_runtime',EXP/'final_front_oos_hv_v1/full.py')
    np,ev,hv=full.load_runtime()
    merged=assemble(root,plan,full,inv,np,ev)
    from report09 import report
    result=report(root,plan,merged,np,hv,full)
    verify(read(root/'oos/INPUT_MANIFEST.json')['files']);plan_for(root)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True,type=Path)
    main(p.parse_args().out.resolve())
