"""Use unchanged final-front evaluator on the new P80 fronts, never checkpoints."""
from common08 import *
import math


def execute(root,plan):
    sys.path.insert(0,str(EXP/'final_front_oos_hv_v1'))
    inv=load('inventory',EXP/'final_front_oos_hv_v1/inventory.py')
    load('run',EXP/'final_front_oos_hv_v1/run.py')
    full=load('alpha08_final_front_runtime',EXP/'final_front_oos_hv_v1/full.py')
    tasks=read(root/'EVALUATION_TASKS.json')['tasks']
    require(len(tasks)==810 and all(completed(t) for t in tasks),'Final evaluation panel incomplete')
    dest=root/'oos';dest.mkdir(exist_ok=True)
    files=dict(plan['code_files']);source=[]
    for t in tasks:
        folder=Path(t['folder']);e=t['entry'];r=read(folder/'RUN_COMPLETE.json')
        for name in ('RUN_COMPLETE.json','configuration.json','final_feasible_nondominated.json','JOB_COMPLETE.json'):
            files[str(folder/name)]=digest(folder/name)
        source.append(inv.inspect_run(t['table'],root,dict(files=files),e,t['arm'],t['seed'],folder/'RUN_COMPLETE.json',files,ALPHA))
    groups=[];units={};seeds_used=[]
    for table,entries in plan['tables'].items():
        for e in sorted(entries,key=lambda e:e['instance_id']):
            name=e['instance_id'];unit=table+'/'+e.get('pair_id',name)
            if unit not in units:
                units[unit]={str(n):381000000+10*len(units)+j for j,n in enumerate((100,1000,5000))}
                seeds_used.extend(units[unit].values())
            rs=[r for r in source if r['table']==table and r['instance']==name]
            configs=[read(Path(r['run_path']).parent/'configuration.json') for r in rs]
            require(len({r['scenario_digest'] for r in rs})==1 and all(c['uncertainty']==configs[0]['uncertainty'] for c in configs),
                    'Methods/seeds did not share optimization scenarios')
            solutions={};members=[]
            for r in rs:
                for row in read(r['front_path']):
                    key=inv.exact_id(row)
                    if key in solutions:
                        require(all(math.isclose(a,b,rel_tol=full.RTOL,abs_tol=full.ATOL) for a,b in zip(inv.vector(row),inv.vector(solutions[key]))),
                                'Same decision has inconsistent optimization objectives')
                    else:solutions[key]=dict(row,exact_decision_id=key)
                    members.append(dict(method=r['method'],seed=r['seed'],exact_decision_id=key,source_solution_id=row['source_solution_id']))
            folder=dest/'instances'/table/name
            packet=dict(table=table,instance=e,source_config=configs[0],source_runs=rs,
                decisions=[solutions[k] for k in sorted(solutions)],memberships=members,seeds=units[unit],
                bounds_unit=unit,alpha=ALPHA,workbook=str(e.get('workbook',inv.DATA)))
            packet['sha256']=canonical(packet)
            freeze(folder/'INPUTS.json',packet);freeze(folder/'instance.json',e)
            groups.append(dict(table=table,instance=name,network=e.get('network','expanded'),folder=str(folder),
                bounds_unit=unit,inputs_sha256=packet['sha256'],unique_decisions=len(solutions)))
    opt={e['ccp100_seed'] for es in plan['tables'].values() for e in es}
    require(len(seeds_used)==len(set(seeds_used)) and not opt.intersection(seeds_used),'OOS seeds overlap')
    manifest=dict(alpha=ALPHA,commit=plan['commit'],files=files,groups=groups,source_plan_sha256=plan['sha256'],
        quantile='ceil(alpha*n)-th order statistic, no interpolation',sizes=[100,1000,5000],ranks=[80,800,4000],
        groups_independent=True,nested=False,shared_across_methods_and_search_seeds=True)
    freeze(dest/'PLAN.json',manifest);verify(files)
    np,ev,hv=full.load_runtime()
    order=sorted(groups,key=lambda g:(g['bounds_unit'],g['network']=='small'))
    def progress(g,tag,done,total,count,physical,elapsed):
        status(root,'P80_final_front_reevaluation',instance=g['instance'],scenario_group=tag,
               done=done,total=total,scenario_count=count,elapsed_seconds=round(elapsed,1),new_searches_this_stage=0)
    # Every original P80 vector must reproduce before any independent reevaluation.
    for g in order:
        packet=read(Path(g['folder'])/'INPUTS.json')
        full.evaluate_pack(np,ev,dest,g,packet,'optimization_100',groups,ALPHA,progress)
    freeze(dest/'OPTIMIZATION_P80_REPRODUCTION.json',dict(passed=True,runs=810,alpha=ALPHA,
        check='All saved final decision P80 vectors match reevaluation on identical original 100 scenarios'))
    for g in order:
        packet=read(Path(g['folder'])/'INPUTS.json')
        for n in (100,1000,5000):full.evaluate_pack(np,ev,dest,g,packet,'fresh_'+str(n),groups,ALPHA,progress)
    verify(files)
    # The shared reporting engine takes alpha explicitly; result names are P80.
    if (dest/'RESULT_LOCATION.json').exists():
        location=read(dest/'RESULT_LOCATION.json');verify(location['files']);result=Path(location['directory'])
    else:
        result=full.report(np,hv,dest,groups,full.load_pack,ALPHA)
        freeze(dest/'RESULT_LOCATION.json',dict(directory=str(result),
            files={str(f):digest(f) for f in result.iterdir() if f.is_file()}))
    coverage=[]
    for g in groups:
        folder=Path(g['folder']);p=read(folder/'INPUTS.json');pack=full.load_pack(folder,p,'fresh_5000')
        by={r['exact_decision_id']:i for i,r in enumerate(p['decisions'])}
        for r in p['source_runs']:
            ids=[by[m['exact_decision_id']] for m in p['memberships'] if (m['method'],m['seed'])==(r['method'],r['seed'])]
            for j,obj in enumerate(inv.OBJECTIVES):
                x=pack['coverage'][ids,j]
                coverage.append(dict(table=g['table'],instance=g['instance'],method=r['method'],seed=r['seed'],objective=obj,
                    mean_coverage=float(np.mean(x)),median_coverage=float(np.median(x)),fraction_below_075=float(np.mean(x<.75))))
    if not (dest/'P80_COVERAGE_PER_RUN.csv').exists():csv_write(dest/'P80_COVERAGE_PER_RUN.csv',coverage)
    freeze(dest/'COMPLETE.json',dict(completed=True,alpha=ALPHA,runs=810,groups=len(groups),
        report_location=str(dest/'RESULT_LOCATION.json'),new_searches_this_stage=0,original_results_modified=False))
