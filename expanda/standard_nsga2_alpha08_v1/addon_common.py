"""Isolated comparator; unique module name avoids frozen engine imports."""
from pathlib import Path
import sys
from collections import Counter

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
sys.path.insert(0, str(EXP / 'alpha08_main_v1'))
import common08 as c
from common08 import read, write, freeze, verify, require, digest, canonical, load
from common08 import csv_write, seal_job, completed

METHOD = 'NSGA2_STANDARD'
PROTOCOL = dict(name='standard-nsga2-transport-encoding-v1', population=250, budget=5000,
    confidence=.8, scenario_count=100, crossover_probability=.90, mutation_probability=.15,
    max_initial_paths=3, initialization='Random paths and normalised random shares; no heuristic seeds',
    crossover='Uniform crossover of complete batch allocations',
    mutation='Per offspring 0.15; uniform batch; equal-probability add/delete/replace path or share perturbation; no objective guidance; no path-count cap',
    selection='Deb fast nondominated sorting, crowded binary tournament, parent+offspring elitism',
    constraint_handling='Feasible dominates infeasible; among infeasible smaller dimensionless aggregate violation wins',
    repair='Representation validity only; no capacity optimisation or feasibility boost',
    reference='Deb et al. (2002), IEEE TEC 6(2):182-197, doi:10.1109/4235.996017',
    qualification='Standard NSGA-II framework; transport encoding/operators are explicit problem adaptations, not claimed to be supplied by Deb et al.')


def status(root, stage, **kw):
    import os
    from datetime import datetime, timezone
    v = dict(completed=False, stage=stage, pid=os.getpid(), alpha=.8,
             updated_utc=datetime.now(timezone.utc).isoformat())
    v.update(kw)
    write(root/'STATUS.json', v)
    print('[STANDARD-NSGA2]', v, flush=True)


def plan_for(root):
    p = read(root/'PLAN.json')
    require(canonical({k:v for k,v in p.items() if k!='sha256'}) == p['sha256'], 'Plan changed')
    require(p['protocol'] == PROTOCOL, 'Comparator definition changed')
    verify(p['files'])
    parent = c.plan_for(Path(p['parent_root']))
    require(parent['sha256'] == p['parent_plan_sha256'], 'Parent P80 plan changed')
    return p


def prepare(root, parent_root):
    import subprocess
    parent = c.plan_for(parent_root)
    if (root/'PLAN.json').exists():
        p = plan_for(root)
        require(p['parent_root'] == str(parent_root), 'Resume parent differs')
        return p
    require(not [x for x in root.iterdir() if x.name!='.lock'], 'Use a new output directory')
    require((parent['alpha'], parent['settings']['population'], parent['settings']['budget'],
             parent['settings']['scenario_count']) == (.8,250,5000,100), 'Unexpected P80 parent settings')
    tasks = []
    files = {str(parent_root/'PLAN.json'): digest(parent_root/'PLAN.json')}
    for t, expected in [('Table5', {f'S{i}':2 for i in range(9)}),
                        ('Table6', {f'S{i}':3 for i in range(9,12)})]:
        require(Counter(e['configuration'] for e in parent['tables'][t]) == expected, 'Instance panel differs: '+t)
        require(len(parent['seeds'][t]) == len(set(parent['seeds'][t])) == 3, 'Three common seeds required')
        for e in parent['tables'][t]:
            src = parent_root/'instances'/(e['instance_id']+'.json')
            require(read(src) == e, 'Parent instance differs')
            files[str(src)] = digest(src)
            freeze(root/'instances'/src.name, e)
            for seed in parent['seeds'][t]:
                tasks.append(dict(table=t, entry=e, instance=e['instance_id'], seed=seed, arm=METHOD,
                    folder=str(root/'searches'/t/e['instance_id']/f'seed_{seed}')))
    require(len(tasks)==len({(t['table'],t['instance'],t['seed']) for t in tasks})==81,
            'Expected 54+27 unique new searches')
    for f in HERE.iterdir():
        if f.is_file() and f.suffix in ('.py','.sh','.md'):
            files[str(f)] = digest(f)
    p = dict(protocol=PROTOCOL, parent_root=str(parent_root), parent_plan_sha256=parent['sha256'],
        parent_commit=parent['commit'], commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=EXP,text=True).strip(),
        files=files, tasks=tasks, training_updates=0, scope='Table5 S0-S8 + Table6 S9-S11; no Table7',
        search_evaluations=5000, reevaluation_scenarios=[100,1000,5000], quantile_ranks=[80,800,4000],
        reports='New union bounds including standard NSGA-II; both adapted and standard references explicitly labelled',
        old_method_label='BASE is Adapted NSGA-II; historical identifiers/files are unchanged')
    p['sha256'] = canonical(p); freeze(root/'PLAN.json', p)
    status(root,'preflight_complete',new_searches_planned=81,parent_commit=p['parent_commit'])
    return p
