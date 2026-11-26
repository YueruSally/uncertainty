"""Preparation, new P80 label collection and single-member supervised training."""
from common08 import *
from collections import Counter
import copy


def prepare(root, training_source):
    if (root/'PLAN.json').exists():
        p = plan_for(root)
        require(p['training_source'] == str(training_source), 'Resume source changed')
        verify(p['source_files']); return p
    existing=[f for f in root.iterdir() if f.name!='.pipeline.lock']
    require(not existing, 'New experiment directory must be empty; preserve failed preflight output and choose a new --out')
    sys.path.insert(0, str(EXP/'final_front_oos_hv_v1'))
    inv = load('inventory', EXP/'final_front_oos_hv_v1/inventory.py')
    audit = root/'source_inventory'; audit.mkdir()
    roots = {t:DIAG/n for t,n in inv.ROOT_NAMES.items()}
    runs, old, entries, files = inv.inventory(roots, audit, .9)
    source_plan = read(training_source/'PLAN.json')
    require(source_plan['protocol']['confidence'] == .9, 'Unexpected original supervision alpha')
    require(canonical({k:v for k,v in source_plan.items() if k!='sha256'}) == source_plan['sha256'], 'Supervision source plan changed')
    verify(source_plan['files'])
    catpath = training_source/'instances/catalog.json'
    trainval = read(catpath)['instances']
    require(Counter((e['configuration'],e['development_role']) for e in trainval) ==
            {(f'S{i}',r):n for i in range(9) for r,n in [('training',6),('validation',2)]}, 'Expected 54 training / 18 validation')
    allids = [e['instance_id'] for e in trainval]+[e['instance_id'] for es in entries.values() for e in es.values()]
    require(len(set(allids)) == len(allids), 'Training/validation/evaluation instance identity overlaps')
    train_signatures = {e['batch_signature'] for e in trainval}
    require(not train_signatures.intersection(e['batch_signature'] for es in entries.values() for e in es.values()),
            'Training/validation cargo overlaps evaluation instances')
    for e in trainval:
        p = training_source/'instances'/(e['instance_id']+'.json')
        require(read(p) == e, 'Training instance changed')
        files[str(p)] = digest(p)
    for p in [catpath, training_source/'PLAN.json']: files[str(p)] = digest(p)
    sup = copy.deepcopy(source_plan['protocol'])
    require((sup['population'],sup['budget'],sup['scenario_count'],sup['states'],sup['training_seeds']) ==
            (250,50000,100,dict(I=48,C=24,M=48),[991001,991002,991003]), 'Historical supervision protocol differs')
    sup.update(confidence=ALPHA, training_seeds=[991001], protocol='alpha08-single-supervision-v1')
    sup['gate_lambda'] = [0.]  # Single member: the ensemble-spread term is identically zero.
    settings = copy.deepcopy(old['Table5']['settings'])
    settings.update(confidence=ALPHA, milestones=[5000])
    require((settings['population'],settings['budget'],settings['scenario_count'],settings['progress_denominator'],settings['m_start_generation']) ==
            (250,5000,100,50000,10), 'Unexpected search settings')
    q = read(EXP/'single_ppo_s0_s8_b5000_v1/protocol.json')
    q.update(protocol='alpha08-single-ppo-all9-v1', milestones=[5000])
    q.update(development='New P80 final searches; no P90 models, labels or search results reused.',
        new_searches='198 if u0 selected, otherwise 225, excluding final evaluation panels',
        new_objective_evaluations_maximum=1125000,
        first_preview='Complete tuning panel after update 1; no outcome-based early stopping.')
    settings['ppo'] = q
    # Pin only code/data used by this entry; never write or monkey-patch files on disk.
    code = {}
    directories = ['alpha08_main_v1','formal_icm_s0_v1/src','component_screen_b20_v1',
        'joint_icm_multiscale_v1','joint_icm_multiscale_terminal_v1','component_joint_b5000_v1',
        'ppo_joint_b5000_pilot_v2','single_ppo_s0_s8_b5000_v1','ppo_joint_b5000_screen27_v1',
        'ppo_joint_b5000_base_report_v1','hv_joint_b5000_union_v1','shared_sl_ablations_b5000_v2',
        'network_transfer_b5000_v1','fixed_q90_s0_s11_v1','final_front_oos_hv_v1']
    for d in directories:
        for f in (EXP/d).rglob('*'):
            if f.is_file() and f.suffix in ('.py','.json','.csv','.xlsx','.sh'): code[str(f)] = digest(f)
    for f in [EXP/'formal_icm_s0_v1/assets/data/data_expanded.xlsx',EXP/'formal_icm_s0_v1/assets/reward_normalization.json']:
        code[str(f)] = digest(f)
    for table, es in entries.items():
        for e in es.values():
            freeze(root/'instances'/(e['instance_id']+'.json'),e)
            if table == 'Table7':
                for n in ('PATHS.json','OPTION_PATH_IDS.json','SCENARIOS.json','SCENARIOS.npz'):
                    f=Path(e['input_folder'])/n; files[str(f)]=digest(f)
                files[e['workbook']]=digest(e['workbook'])
    for e in trainval: freeze(root/'instances'/(e['instance_id']+'.json'),e)
    freeze(root/'instances/catalog.json',dict(instances=trainval, copied_instance_definitions_only=True))
    optseeds={e['ccp100_seed'] for es in entries.values() for e in es.values()}
    require(not optseeds.intersection(381000000+10*i+j for i in range(30) for j in range(3)), 'OOS seed overlaps optimization')
    version=subprocess.check_output(['git','rev-parse','HEAD'],cwd=EXP,text=True).strip()
    p=dict(experiment_protocol='alpha08-main-final-oos-v1',alpha=ALPHA,commit=version,
        training_source=str(training_source),protocol=sup,settings=settings,pilot=q,training_entries=trainval,
        tables={t:list(es.values()) for t,es in entries.items()},
        seeds={'Table5':[993001,993002,993003], 'Table6':old['Table6']['protocol']['algorithm_seeds'],
               'Table7':[993001,993002,993003]},
        network_profile=old['Table7']['uncertainty_profile'],
        source_runs=runs,source_files=files,code_files=code,
        policy_members=1,training_member=991001,report_search_budget=5000,evaluation_sizes=[100,1000,5000],
        quantile_ranks=[80,800,4000],reference=[1.1,1.1,1.1],hv_divisor=1.331,
        reuse='Instance definitions, sampler, code and network path libraries only; no alpha=.9 labels, weights or fronts',
        random_rate_rule='Observed alpha=.8 shared deployment switches / all calls, component-wise; no outcome/HV used',
        zero_threshold_rule='Set enabled components thresholds to zero; retain validation-disabled components and legal masks',
        oos_seeds='381000000+10*bound_unit_index+group_index; independent groups; same fixed draws as alpha=.9 protocol when unit ordering agrees',
        calibration='Nine new greedy P80 shared-policy training searches; one per configuration; no validation labels',
        final_evaluation_runs=810,collection_base_searches=72,collection_initialization_only=72,
        ppo_training_searches=27,ppo_paired_reference_searches=27)
    p['sha256']=canonical(p); freeze(root/'PLAN.json',p)
    status(root,'preflight_complete',planned_final_evaluation_runs=810,training_instances=54,validation_instances=18)
    return p


def collect_stage(root, p, workers):
    tasks=[job(root,'collection',e,e['algorithm_seed'],a,root/'collection'/e['instance_id']/a)
           for e in p['training_entries'] for a in ('COLLECT_I','COLLECT_CM')]
    schedule(root,tasks,workers,'P80_collection')
    sys.path.insert(0,str(EXP/'joint_icm_multiscale_v1'))
    import collection
    collection.completed_dataset(root,p['protocol'],p['training_entries'],p['sha256'])


def train_stage(root,p):
    sys.path.insert(0,str(EXP/'joint_icm_multiscale_terminal_v1'))
    import bridge
    from data import load_dataset
    import training
    folders=[root/'collection'/e['instance_id']/a for e in p['training_entries'] for a in ('COLLECT_I','COLLECT_CM')]
    rows,norm=load_dataset(folders,p['protocol']['progress_denominator'])
    provenance={str(root/'PLAN.json'):digest(root/'PLAN.json'),**p['code_files']}
    for f in folders:
        for n in ('DATA.json','RUN_COMPLETE.json'): provenance[str(f/n)]=digest(f/n)
    status(root,'P80_SL_training')
    coverage=dict(full_budget_ready=True,trajectory_budget=50000,confidence=ALPHA,
                  progress_denominator=50000,fixed_initial_seeds_preserved=True,old_weights_used=False)
    shared=training.fit(rows,norm,p['protocol'],root/'shared_model',provenance,coverage)
    independent=bridge.fit_independent(root,root/'independent_model')
    for b in (shared,independent):
        require(b['settings']['confidence']==ALPHA and b['settings']['training_seeds']==[991001]
                and len(b['checkpoints'])==1,'Not a P80 single-member model')
    freeze(root/'SL_SELECTION.json',dict(shared=shared['selected'],independent=independent['selected'],
        shared_bundle_sha256=digest(root/'shared_model/BUNDLE.json'),
        independent_bundle_sha256=digest(root/'independent_model/BUNDLE.json')))
