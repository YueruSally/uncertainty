from common import *
import random,time
import numpy as np
import torch
import baseline_uncertainty as base
from transformer_ppo.archive import ObjectiveNormalizer
from transformer_ppo.episode import run_episode,_optimisation_vector
from controller import ScreenController
from learning import FrozenPolicy,normalizer

def work(root,settings,name,seed,arm,out,plan_sha):
    torch.set_num_threads(1);instance_path=root/'instances'/f'{name}.json';info,_=read_instance(instance_path)
    collect=arm[-1] if arm.startswith('COLLECT_') else None
    calibration=info['development_role']=='calibration'
    effective_arm=(collect+'_RULE') if collect else arm
    settings=dict(settings)
    rules_sha=None
    if calibration:
        if arm!='BASE':
            component,variant=arm.split('_',1)
            if variant not in settings['rule_variants']:raise ValueError('Unknown calibration rule')
            settings['rule_profile']=variant;effective_arm=component+'_RULE'
    else:
        selected=read_json(root/'RULE_SELECTION.json');rules_sha=digest(root/'RULE_SELECTION.json')
        if selected['plan_sha256']!=plan_sha:raise ValueError('Rule selection belongs to another experiment')
        settings['rule_profile']=selected['profiles'].get(effective_arm[0],'LEGACY')
    normpath=root/'models/HV_NORMALIZATION.json'
    if collect or calibration:
        norm=read_json(ROOT.parent/'formal_icm_s0_v1/assets/reward_normalization.json')
        if collect and info['development_role'] not in ('training','validation'):raise ValueError('Screen instances may not supply labels')
    else:
        norm=read_json(normpath)
        if info['development_role']!='screen':raise ValueError('Screen run is not in screening split')
    learned=FrozenPolicy(root/'models'/arm[0]) if arm.endswith('LEARN') else None
    holders=[]
    def factory(*args,**kwargs):
        ctl=ScreenController(*args,settings=settings,arm=effective_arm,collect=collect,learned=learned,**kwargs);holders.append(ctl);return ctl
    data=ROOT.parent/'formal_icm_s0_v1/assets/data/data_expanded.xlsx'
    # Retain valid inherited evaluations. The controller verifies the exact
    # scenario + chromosome key; a changed offspring is always evaluated.
    original_clear=base.clear_inherited_evaluation_cache
    original_evaluate=base._evaluate_individual_impl;observed_calls=[0]
    def counted_evaluate(*args,**kwargs):
        observed_calls[0]+=1
        return original_evaluate(*args,**kwargs)
    base._evaluate_individual_impl=counted_evaluate
    base.clear_inherited_evaluation_cache=lambda *individuals: None
    try:
        result=run_episode(data,instance_path,out,None,normalizer(norm),algorithm_seed=seed,
        evaluation_budget=settings['budget'],pop_size=settings['population'],generations=1000000,
        evaluation_mode='ccp',component='BASE',scenario_count=settings['scenario_count'],ccp_confidence=settings['confidence'],
        uncertainty_profile='transport_border_saa_v1',inference_controller_factories={'BASE':factory},method_override=arm)
    finally:
        base.clear_inherited_evaluation_cache=original_clear
        base._evaluate_individual_impl=original_evaluate
    ctl=holders[0]
    expected_calls=ctl.evaluations+sum(ctl.audit_counts.values())
    if observed_calls[0]!=expected_calls:
        raise ValueError(f'Unaccounted objective evaluations: observed={observed_calls[0]}, charged={expected_calls}')
    if collect=='I':
        if len(ctl.population)!=settings['population']:raise ValueError('Incomplete initialization collection')
    elif not settings['budget']-1<=ctl.evaluations<=settings['budget']:
        raise ValueError(f'Incomplete budget: {ctl.evaluations}/{settings["budget"]}')
    result['milestone_hv']=[p for p in result['milestone_hv'] if p['evaluation']!=settings['budget']]
    if collect!='I':
        result['milestone_hv'].append({'evaluation':settings['budget'],'actual_evaluations':ctl.evaluations,'hv':result['final_hypervolume'],'definition':'final population under evaluation cap; a two-child atomic step may leave one evaluation unused'})
    if collect:
        needed=collect
        if any(ctl.audit_states[k]==0 for k in needed):raise ValueError('No usable candidate states; stop without automatic data expansion')
        write_json(out/'DATA.json',{'instance':name,'configuration':info['configuration'],'role':info['development_role'],
            'divisor':divisor(info),'feasible_raw':ctl.raw_points,'rows':ctl.feedback,'extra_evaluations':ctl.audit_counts,
            'actual_search_evaluations':ctl.evaluations,'phase_counts':ctl.phase_counts})
    result.update(observed_objective_evaluations=observed_calls[0],rule_selection_sha256=rules_sha,final_front_raw=[list(_optimisation_vector(x)) for x in read_json(out/'final_archive_lineage.json')],instance=name,configuration=info['configuration'],seed=seed,arm=arm,plan_sha256=plan_sha,
        feature_version=FEATURE_VERSION,scenario_digest=read_json(out/'configuration.json')['scenario_digest'],
        feedback_sha256=digest(out/'DATA.json') if collect else None,
        budget_definition='actual complete-solution evaluations; unchanged valid inherited evaluations reused; audit cache isolated',
        rss_units='KiB on Linux, bytes on macOS')
    write_json(out/'RUN_COMPLETE.json',result)
