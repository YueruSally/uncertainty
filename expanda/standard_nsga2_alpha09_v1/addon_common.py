"""Separate P90 comparator. Existing P80/P90 programs and results stay read-only."""
from pathlib import Path
from collections import Counter
from datetime import datetime, timezone
import csv
import hashlib
import importlib.util
import json
import os
import subprocess
import sys

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
DIAG = EXP / 'transformer_ppo_runs/diagnostics'
DEFAULT_PARENT = DIAG / 'ccp_final_front_oos_hv_full_v1_20261126T133631290126Z'
METHOD = 'NSGA2_STANDARD'
ARMS = ('BASE','I_RULE','I_LEARN','C_RULE','C_LEARN','M_RULE','M_LEARN',
        'ICM_HEURISTIC','ICM_INDEPENDENT','ICM_SHARED_SUPERVISED',
        'ICM_TERMINAL_PPO','ICM_RANDOM_RATE_CONTROLLED','ICM_SHARED_ZERO_THRESHOLD')
TAGS = ('optimization_100','fresh_100','fresh_1000','fresh_5000')

def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def require(x, message):
    if not x: raise ValueError(message)
def canonical(x):
    return hashlib.sha256(json.dumps(x, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
def digest(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''): h.update(b)
    return h.hexdigest()
def write(p, value):
    p=Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    temp=p.with_name(p.name+'.writing')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    temp.replace(p)
def freeze(p, value):
    if Path(p).exists(): require(read(p)==value, 'Frozen file differs: '+str(p))
    else: write(p,value)
def verify(files):
    for name,h in files.items(): require(digest(name)==h,'Pinned input differs: '+name)
def load(name, path):
    spec=importlib.util.spec_from_file_location(name,path)
    mod=importlib.util.module_from_spec(spec); sys.modules[name]=mod
    spec.loader.exec_module(mod); return mod
def csv_write(path, rows):
    require(bool(rows),'Refusing empty CSV: '+str(path))
    with Path(path).open('x',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
def completed(task):
    p=Path(task['folder'])/'JOB_COMPLETE.json'
    if not p.exists(): return False
    q=read(p)
    require(q['completed'] and q['task_sha256']==canonical(task),'Cached search belongs to another task')
    verify(q['files']); return True
def seal_job(task):
    folder=Path(task['folder'])
    require((folder/'RUN_COMPLETE.json').exists(),'Missing search completion record')
    freeze(folder/'JOB_COMPLETE.json',dict(completed=True,task_sha256=canonical(task),
        files={str(p):digest(p) for p in folder.rglob('*') if p.is_file() and p.name!='JOB_COMPLETE.json'}))

PROTOCOL = dict(name='standard-nsga2-transport-encoding-v1',population=250,budget=5000,
    confidence=.9,scenario_count=100,crossover_probability=.90,mutation_probability=.15,
    max_initial_paths=3,initialization='Random paths and normalised random shares; no heuristic seeds',
    crossover='Uniform crossover of complete batch allocations',
    mutation='Per offspring 0.15; uniform batch; equal-probability add/delete/replace path or share perturbation; no objective guidance; no path-count cap',
    selection='Deb fast nondominated sorting, crowded binary tournament, parent+offspring elitism',
    constraint_handling='Feasible dominates infeasible; among infeasible smaller dimensionless aggregate violation wins',
    repair='Representation validity only; no capacity optimisation or feasibility boost',
    reference='Deb et al. (2002), IEEE TEC 6(2):182-197, doi:10.1109/4235.996017',
    qualification='Standard NSGA-II framework with explicit transport encoding/operators, not author source code')

def status(root,stage,**extra):
    value=dict(completed=False,stage=stage,pid=os.getpid(),alpha=.9,
               updated_utc=datetime.now(timezone.utc).isoformat())
    value.update(extra);write(root/'STATUS.json',value)
    print('[STANDARD-NSGA2-P90]',json.dumps(value,ensure_ascii=False),flush=True)

def parent_manifest(parent):
    m=read(parent/'PLAN.json')
    require(m['alpha']==.9 and m['protocol']=='final-front-oos-hv-full-v1','Requires the completed alpha=.9 full reevaluation')
    require(canonical({k:v for k,v in m.items() if k!='sha256'})==m['sha256'],'Parent manifest hash differs')
    require(read(parent/'STATUS.json').get('completed') is True,'Parent P90 reevaluation incomplete')
    verify(m['files']); verify(m['entry_files'])
    location=read(parent/'RESULT_LOCATION.json');verify(location['files'])
    require(read(Path(location['directory'])/'REPORT.json')['alpha']==.9,'Parent report alpha differs')
    return m

def plan_for(root,verify_inputs=True):
    p=read(root/'PLAN.json')
    require(canonical({k:v for k,v in p.items() if k!='sha256'})==p['sha256'],'Plan hash differs')
    require(p['protocol']==PROTOCOL and p['settings']['confidence']==.9,'Wrong comparator definition or alpha')
    if verify_inputs:verify(p['files'])
    verify(p['code_files'])
    require(read(Path(p['parent_root'])/'PLAN.json')['sha256']==p['parent_plan_sha256'],'Parent changed')
    return p

def prepare(root,parent):
    if (root/'PLAN.json').exists():
        p=plan_for(root);require(p['parent_root']==str(parent),'Resume source differs');return p
    require(not [p for p in root.iterdir() if p.name!='.lock'],'Use a new output directory')
    status(root,'checking_frozen_P90_inputs')
    m=parent_manifest(parent)
    files=dict(m['files']); files.update(m['entry_files'])
    for name in ('PLAN.json','STATUS.json','RESULT_LOCATION.json'):
        files[str(parent/name)]=digest(parent/name)
    files.update(read(parent/'RESULT_LOCATION.json')['files'])
    # Read actual historical settings, rather than guessing from current defaults.
    settings_plans=[Path(p) for p in m['files'] if Path(p).name=='PLAN.json'
                    and read(p).get('protocol')=='single-shared-random-rate-controlled-union-v2']
    require(len(settings_plans)==1,'Cannot uniquely locate frozen Table 5 settings')
    settings=read(settings_plans[0])['settings']
    require(tuple(settings[k] for k in ('confidence','population','budget','scenario_count'))==(.9,250,5000,100),
            'Historical P90 search settings differ')
    tasks=[]; groups=m['groups']; total=Counter()
    for table,expected,nmethods in [('Table5',{f'S{i}':2 for i in range(9)},13),
                                   ('Table6',{f'S{i}':3 for i in range(9,12)},3)]:
        packets=[read(Path(g['folder'])/'INPUTS.json') for g in groups if g['table']==table]
        require(Counter(p['instance']['configuration'] for p in packets)==expected,'Instance panel differs: '+table)
        common_seeds=None
        for packet in packets:
            require(packet['alpha']==.9,'Source packet alpha differs')
            require(canonical({k:v for k,v in packet.items() if k!='sha256'})==packet['sha256'],'Source packet hash differs')
            e=packet['instance'];cfg=packet['source_config']
            require((cfg['alpha'],cfg['population'],cfg['evaluation_budget'],cfg['training_size'])==(.9,250,5000,100),
                    'Source configuration differs')
            runs=packet['source_runs'];base=[r for r in runs if r['method']=='BASE']
            seeds=sorted(r['seed'] for r in base)
            require(len(runs)==3*nmethods and len(seeds)==len(set(seeds))==3,'Incomplete source run panel')
            require({r['method'] for r in runs}==set(ARMS if table=='Table5' else ('BASE','ICM_SHARED_SUPERVISED','ICM_TERMINAL_PPO')),
                    'Unexpected methods')
            require(common_seeds is None or common_seeds==seeds,'Search seeds differ across instances')
            common_seeds=seeds
            freeze(root/'instances'/(e['instance_id']+'.json'),e)
            for seed in seeds:
                tasks.append(dict(table=table,entry=e,instance=e['instance_id'],seed=seed,arm=METHOD,
                    expected_scenario_digest=cfg['scenario_digest'],source_uncertainty=cfg['uncertainty'],
                    workbook=packet['workbook'],folder=str(root/'searches'/table/e['instance_id']/f'seed_{seed}')))
        total[table]=sum(len(p['source_runs']) for p in packets)
    for g in groups:
        folder=Path(g['folder'])
        for p in folder.rglob('*'):
            if p.is_file() and p.suffix in ('.json','.npz'):files[str(p)]=digest(p)
    require(len(tasks)==81 and total=={'Table5':702,'Table6':81},'Expected 81 new runs and 783 reused main/generalization runs')
    require(sum(len(read(Path(g['folder'])/'INPUTS.json')['source_runs']) for g in groups if g['table']=='Table7')==27,
            'Missing Table 7 runs')
    # Pin imported computation modules too. These are not edited by this addon.
    code={str(p):digest(p) for p in HERE.iterdir() if p.is_file() and p.suffix in ('.py','.sh','.md')}
    for directory in ('formal_icm_s0_v1/src','ppo_joint_b5000_pilot_v2','component_joint_b5000_v1',
                      'joint_icm_multiscale_terminal_v1','joint_icm_multiscale_v1','final_front_oos_hv_v1'):
        code.update({str(p):digest(p) for p in (EXP/directory).rglob('*.py')})
    kernel=EXP/'standard_nsga2_alpha08_v1/kernel.py';code[str(kernel)]=digest(kernel)
    norm=EXP/'formal_icm_s0_v1/assets/reward_normalization.json';files[str(norm)]=digest(norm)
    for t in tasks:files[t['workbook']]=digest(t['workbook'])
    p=dict(protocol=PROTOCOL,settings=settings,parent_root=str(parent),parent_plan_sha256=m['sha256'],
        parent_commit=m['commit'],commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=EXP,text=True).strip(),
        tracked_changes=subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=EXP,text=True),
        files=files,code_files=code,tasks=tasks,training_updates=0,new_searches=81,old_runs_reused=810,
        scope='Standard P90: Table5 S0-S8 54 + Table6 S9-S11 27; Table7 preserved and reported',
        reevaluation_scenarios=[100,1000,5000],quantile_ranks=[90,900,4500],search_evaluations=5000,
        sampling='Exact existing P90 draws reused. Fresh sizes independent, nonnested; all methods/seeds share draws.',
        reports='New per-instance union bounds across all methods/seeds and fresh100/1000/5000. HV /1.331.',
        old_method_label='BASE is Adapted NSGA-II; raw identifiers preserved',
        wtl='Primary raw tolerance 1e-12; separate sensitivity at raw 1e-2 and reported HV 1e-2')
    p['sha256']=canonical(p);freeze(root/'PLAN.json',p)
    status(root,'preflight_complete',new_searches_planned=81,old_runs_reused=810)
    return p
