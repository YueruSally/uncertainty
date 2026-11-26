"""Process-local adapters around the frozen collection/search implementations."""
from common08 import *
import inspect
import resource
import socket
import platform


def runtime():
    sys.path.insert(0,str(EXP/'ppo_joint_b5000_pilot_v2'))
    return load('alpha08_pilot_engine',EXP/'ppo_joint_b5000_pilot_v2/run.py')


class TrackedPolicy:
    def __init__(self,policy):
        self.policy=policy
        self.eligibility={c:dict(calls=0,eligible=0,switches=0) for c in 'ICM'}
    def __getattr__(self,name): return getattr(self.policy,name)
    def select(self,row):
        q=self.eligibility[row['component']];q['calls']+=1
        q['eligible']+=int(any(row['legal_mask'][1:]))
        choice=int(self.policy.select(row));q['switches']+=int(choice!=0)
        require(row['legal_mask'][choice], 'Policy selected an illegal candidate')
        return choice


def network_hooks(p,entry,profile):
    """Same saved nominal path library and coupled scenarios as original Table 7."""
    m=load('alpha08_network_inputs',EXP/'network_transfer_b5000_v1/inputs.py')
    b=p.base; folder=Path(entry['input_folder'])
    scenarios=m.load_scenarios(b,p.np,folder)
    pathdata=read(folder/'PATHS.json'); options=read(folder/'OPTION_PATH_IDS.json')
    originals={k:getattr(b,k) for k in ('build_path_library','build_scenario_set','build_reliable_path_options')}
    signature=inspect.signature(originals['build_scenario_set']);counts=dict(paths=0,scenarios=0,options=0)
    def paths(nodes,regions,arcs,batches,timetable,lookup):
        require([(x.batch_id,x.origin,x.destination,x.quantity,x.ET,x.LT) for x in batches] ==
                [(x['batch_id'],x['origin'],x['destination'],x['quantity'],x['ET'],x['LT']) for x in entry['batches']],
                'Network-transfer batches differ')
        counts['paths']+=1;return m.load_paths(b,pathdata,arcs)
    def draws(*args,**kwargs):
        bound=signature.bind(*args,**kwargs);bound.apply_defaults();v=bound.arguments
        if not v['stochastic']:return originals['build_scenario_set'](*args,**kwargs)
        require(v['size']==100 and v['seed']==scenarios.seed and
                all(v[k]==x for k,x in m.scenario_kwargs(profile).items()),'Network sampler settings differ')
        require({(a.from_node,a.to_node,a.mode) for a in v['arcs']}==set(scenarios.travel_multiplier),'Network arcs differ')
        counts['scenarios']+=1;return scenarios
    def screened(*args,**kwargs):
        result=originals['build_reliable_path_options'](*args,**kwargs)
        require({str(k[2]):[x.path.path_id for x in vs] for k,vs in result.items()}==options,'Nominal feasible paths differ')
        counts['options']+=1;return result
    b.build_path_library,b.build_scenario_set,b.build_reliable_path_options=paths,draws,screened
    return originals,counts


def search(root,plan,task):
    p=runtime();p.torch.set_num_threads(1)
    s=plan['settings'];e=task['entry'];out=Path(task['folder']);arm=task['arm']
    require(read(root/'instances'/(e['instance_id']+'.json'))==e,'Instance copy changed')
    if task.get('table'):
        require(e in plan['tables'][task['table']] and task['seed'] in plan['seeds'][task['table']], 'Unscheduled evaluation pair')
    else:
        require(e in plan['training_entries'],'Training/validation instance outside split')
    model=Path(task['model']) if task.get('model') else None
    if model:
        require(digest(model/'BUNDLE.json')==task['bundle_sha256'],'Model bundle changed')
        b=read(model/'BUNDLE.json')
        require(b['settings']['confidence']==ALPHA and len(b['checkpoints'])==1,'Wrong policy alpha/members')
    engine_arm=SHARED if arm in (RANDOM,ZERO) else arm
    policy=None
    if arm==RANDOM:
        r=load('alpha08_random',EXP/'shared_sl_ablations_b5000_v2/policy_v2.py')
        policy=r.RateControlledRandom(task['targets'],r.policy_seed(e['instance_id'],task['seed']))
    elif arm==PPO or task.get('record'):
        policy=p.PilotPolicy(model,sample=task.get('sample',False),seed=task['seed']+7000000,record=task.get('record',False))
    elif arm==ZERO:
        policy=p.SharedPolicy(model)
        for c in 'ICM':policy.bundle['selected']['gates'][c]['threshold_scaled']=0.
    elif arm==SHARED:policy=p.SharedPolicy(model)
    elif arm.endswith('_LEARN') or arm=='ICM_INDEPENDENT':policy=p.IndependentPolicy(model)
    if policy is not None:policy=TrackedPolicy(policy)
    controllers=[]
    def factory(*args,**kwargs):
        c=p.BudgetController(*args,settings=s,arm=engine_arm,learned=policy,**kwargs)
        controllers.append(c);return c
    priors={};counts=None
    if task.get('table')=='Table7':priors,counts=network_hooks(p,e,plan['network_profile'])
    priors['clear_inherited_evaluation_cache']=p.base.clear_inherited_evaluation_cache
    p.base.clear_inherited_evaluation_cache=lambda *args:None
    started=time.monotonic()
    try:
        r=p.run_episode(Path(e.get('workbook',EXP/'formal_icm_s0_v1/assets/data/data_expanded.xlsx')),
            root/'instances'/(e['instance_id']+'.json'),out,None,p.legacy_learning.normalizer(task['normalization']),
            algorithm_seed=task['seed'],evaluation_budget=s['budget'],pop_size=s['population'],generations=1000000,
            evaluation_mode='ccp',component='BASE',scenario_count=100,ccp_confidence=ALPHA,
            uncertainty_profile='transport_border_saa_v1',inference_controller_factories={'BASE':factory},method_override=engine_arm)
    finally:
        for k,v in priors.items():setattr(p.base,k,v)
    require(r['real_objective_evaluation_count'] in (4999,5000) and r['hidden_candidate_objective_evaluations']==0,'Budget/hidden evaluation mismatch')
    require(counts is None or counts==dict(paths=1,scenarios=1,options=1),'Network cache hooks not used exactly once')
    cfg=read(out/'configuration.json')
    require(cfg['alpha']==ALPHA and cfg['training_size']==100 and cfg['seeds']['training']==e['ccp100_seed'],'Not P80 / common optimization draws')
    old=[v for v in plan['source_runs'] if v['instance']==e['instance_id']]
    if old:require(cfg['scenario_digest']==old[0]['scenario_digest'],'Optimization scenarios differ from original panel')
    front=read(out/'final_feasible_nondominated.json')
    sys.path.insert(0,str(EXP/'final_front_oos_hv_v1'))
    inv=load('alpha08_front_check',EXP/'final_front_oos_hv_v1/inventory.py')
    require(bool(front),'Empty final feasible front')
    for row in front:inv.validate_decision(row,e)
    points=[inv.vector(row) for row in front]
    ctl=controllers[0]
    trajectory=None
    if task.get('record'):
        policy.save_events(out/'TRAJECTORY.npz');trajectory=digest(out/'TRAJECTORY.npz')
    active=[] if arm=='BASE' else ([arm[0]] if arm[0] in 'ICM' and arm in ('I_RULE','I_LEARN','C_RULE','C_LEARN','M_RULE','M_LEARN') else list('ICM'))
    for c in 'ICM':
        if c not in active:require(r['selection'][c]['calls']==0,'Inactive component made choices')
    r.update(instance=e['instance_id'],configuration=e['configuration'],instance_digest=e['instance_digest'],seed=task['seed'],
        arm=arm,phase=task.get('phase','evaluation'),stochastic=bool(task.get('sample') or arm==RANDOM),
        plan_sha256=plan['sha256'],task_sha256=canonical(task),model_bundle_sha256=task.get('bundle_sha256'),
        trajectory_sha256=trajectory,scenario_digest=cfg['scenario_digest'],final_front_raw=points,
        final_solutions_sha256=digest(out/'final_feasible_nondominated.json'),enabled_components=active,
        budget_fronts=[dict(evaluation=5000,actual_evaluations=r['real_objective_evaluation_count'],
            hv=r['final_hypervolume'],front_raw=points,controller_seconds=time.perf_counter()-ctl.start,
            definition='Only the final feasible nondominated decision front is used for reporting')],
        hostname=socket.gethostname(),worker_wall_seconds=time.monotonic()-started,
        environment=dict(python=platform.python_version(),numpy=p.np.__version__,torch=str(p.torch.__version__),
            cpu_count=os.cpu_count(),load_average=list(os.getloadavg()),nice=os.getpriority(os.PRIO_PROCESS,0),
            threads={k:os.environ.get(k) for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS')}),
        alpha=ALPHA,commit=plan['commit'],network=e.get('network','expanded'),
        eligibility=policy.eligibility if policy else None,
        random_selection=policy.counts if arm==RANDOM else None,
        zero_disabled_components=[c for c in 'ICM' if policy.bundle['selected']['gates'][c]['disabled']] if arm==ZERO else None,
        peak_memory_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(1024 if sys.platform!='darwin' else 1024**2))
    write(out/'RUN_COMPLETE.json',r)


def worker(root,task):
    plan=plan_for(root)
    require(root in Path(task['folder']).parents,'Output outside isolated root')
    if task['kind']=='collection':
        require(task['entry'] in plan['training_entries'] and task['arm'] in ('COLLECT_I','COLLECT_CM')
                and task['seed']==task['entry']['algorithm_seed'],'Unscheduled label collection')
        sys.path.insert(0,str(EXP/'joint_icm_multiscale_v1'))
        from collection import work
        work(root,plan['protocol'],task['instance'],task['seed'],task['arm'],Path(task['folder']),plan['sha256'])
        cfg=read(Path(task['folder'])/'configuration.json')
        require(cfg['alpha']==ALPHA and cfg['training_size']==100,'Collection did not use P80')
    elif task['kind']=='search':search(root,plan,task)
    else:raise ValueError('Unknown task kind')
    seal_job(task)
