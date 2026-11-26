"""All-nine-configuration PPO and evaluation panels, using new P80 models only."""
from common08 import *
from collections import defaultdict
from statistics import mean


def verify_supervised(root):
    selected=read(root/'SL_SELECTION.json')
    for kind in ('shared','independent'):
        require(digest(root/(kind+'_model')/'BUNDLE.json')==selected[kind+'_bundle_sha256'],
                'Selected P80 supervised model changed: '+kind)


def model_job(root,plan,e,seed,arm,folder,model=None,**extra):
    norm=read(root/'shared_model/BUNDLE.json')['normalization']
    return job(root,'search',e,seed,arm,folder,model=str(model) if model else None,
        bundle_sha256=digest(model/'BUNDLE.json') if model else None,normalization=norm,**extra)


def records(tasks):return [read(Path(t['folder'])/'RUN_COMPLETE.json') for t in tasks]


def validation_score(current,reference):
    controls={(r['instance'],r['seed']):r for r in reference};parts=defaultdict(lambda:defaultdict(list))
    for r in current:
        b=controls[r['instance'],r['seed']]
        require(r['scenario_digest']==b['scenario_digest'] and r['alpha']==b['alpha']==ALPHA,'PPO validation not paired')
        parts[r['configuration']][r['instance']].append(r['final_hypervolume']-b['final_hypervolume'])
    require(set(parts)=={f'S{i}' for i in range(9)},'PPO validation missing a configuration')
    return mean(mean(mean(v) for v in instances.values()) for instances in parts.values())


def ppo_stage(root,plan,workers):
    verify_supervised(root)
    from worker08 import runtime
    p=runtime();p.torch.set_num_threads(1);q=plan['pilot']
    folder=root/'ppo';folder.mkdir(exist_ok=True)
    trains={c:sorted([e for e in plan['training_entries'] if e['development_role']=='training' and e['configuration']==c],key=lambda x:x['instance_id']) for c in q['configurations']}
    vals={c:sorted([e for e in plan['training_entries'] if e['development_role']=='validation' and e['configuration']==c],key=lambda x:x['instance_id']) for c in q['configurations']}
    shared=root/'shared_model'
    calibration=[model_job(root,plan,trains[c][0],q['training_seed_base']+i,SHARED,
        folder/'calibration'/trains[c][0]['instance_id'],shared,phase='calibration',sample=False,record=True)
        for i,c in enumerate(q['configurations'])]
    schedule(root,calibration,workers,'P80_PPO/calibration')
    def checkpoint(u):return folder/'checkpoints'/f'update_{u:03d}'
    zero=checkpoint(0)
    if not zero.exists():
        temps,info=p.calibrate(shared,[Path(t['folder'])/'TRAJECTORY.npz' for t in calibration],q)
        actor=p.PilotPolicy(shared);require(len(actor.models)==1,'PPO must be single-member')
        actor.bundle['pilot']={**q,'temperatures':temps}
        actor.bundle['fine_tuning_provenance']=dict(alpha=ALPHA,plan_sha256=plan['sha256'],training_configurations=q['configurations'],member=991001)
        p.snapshot(actor,zero,metrics=dict(update=0,calibration=info));freeze(folder/'CALIBRATION.json',info)
    freeze(folder/'CALIBRATION.json',read(zero/'UPDATE.json')['calibration'])
    before=p.PilotPolicy(shared);initial=p.PilotPolicy(zero)
    require(all(p.torch.equal(v,initial.models[0].state_dict()[k]) for k,v in before.models[0].state_dict().items()),'u0 not unchanged P80 shared model')
    def validation_jobs(u,which):
        return [model_job(root,plan,vals[c][which],seed,PPO,
            folder/('tuning' if which==0 else 'confirmation')/f'u{u:03d}'/vals[c][which]['instance_id']/f'seed_{seed}',
            checkpoint(u),phase='validation',sample=False,record=False)
            for c in q['configurations'] for seed in q['validation_seeds']]
    base_jobs=validation_jobs(0,0);schedule(root,base_jobs,workers,'P80_PPO/tuning_u0')
    controls=records(base_jobs);scores=[dict(update=0,mean_difference_vs_shared=0.)];updates=[]
    for u in range(1,q['updates']+1):
        tasks=[]
        for i,c in enumerate(q['configurations']):
            e=trains[c][u-1];seed=q['training_seed_base']+(u-1)*9+i
            for phase,number,sampled in [('training_reference',0,False),('training',u-1,True)]:
                tasks.append(model_job(root,plan,e,seed,PPO,folder/phase/f'u{u:03d}'/e['instance_id']/f'seed_{seed}',
                    checkpoint(number),phase=phase,sample=sampled,record=sampled))
        schedule(root,tasks,workers,f'P80_PPO/training_pairs_u{u}')
        rows=records(tasks);index={(r['configuration'],r['phase']):r for r in rows}
        episodes=[];rewards=[]
        for c in q['configurations']:
            r,b=index[c,'training'],index[c,'training_reference'];p.check_pair(r,b)
            reward=r['final_hypervolume']-b['final_hypervolume']
            t=next(t for t in tasks if t['entry']['configuration']==c and t['phase']=='training')
            episodes.append((Path(t['folder'])/'TRAJECTORY.npz',reward))
            rewards.append(dict(configuration=c,instance=r['instance'],seed=r['seed'],paired_return=reward))
        if not checkpoint(u).exists():
            policy,reference=p.PilotPolicy(checkpoint(u-1)),p.PilotPolicy(zero)
            optimizer=p.optimizer_for(policy,q)
            if u>1:
                require(digest(checkpoint(u-1)/'optimizer.pt')==policy.bundle['optimizer_sha256'],'Optimizer state changed')
                optimizer.load_state_dict(p.load_checkpoint(checkpoint(u-1)/'optimizer.pt'))
            status(root,f'P80_PPO/update_{u}')
            start=time.monotonic()
            metrics=p.update(policy,reference,episodes,q,optimizer,q['optimizer_seed_base']+u)
            metrics.update(update=u,policy_trajectories=9,reward_pairs=rewards,optimizer_seconds=time.monotonic()-start,alpha=ALPHA)
            p.snapshot(policy,checkpoint(u),optimizer,metrics)
        metrics=read(checkpoint(u)/'UPDATE.json');require(metrics['reward_pairs']==rewards,'PPO cached rewards differ');updates.append(metrics)
        jobs=validation_jobs(u,0);schedule(root,jobs,workers,f'P80_PPO/tuning_u{u}')
        scores.append(dict(update=u,mean_difference_vs_shared=validation_score(records(jobs),controls)))
        write(folder/'TRAINING_PROGRESS.json',dict(validation=scores,training=updates))
    selected=max(scores,key=lambda x:(x['mean_difference_vs_shared'],-x['update']))['update']
    result=dict(selected_update=selected,model=str(checkpoint(selected)),bundle_sha256=digest(checkpoint(selected)/'BUNDLE.json'),
        candidates=scores,selection_includes_u0=True,selection_uses_evaluation_instances=False,alpha=ALPHA,
        policy_training_searches=27,paired_reference_searches=27,calibration_searches=9)
    freeze(folder/'SELECTION.json',result)
    confirm0=validation_jobs(0,1);schedule(root,confirm0,workers,'P80_PPO/confirmation_u0')
    confirm=confirm0 if selected==0 else validation_jobs(selected,1)
    if selected:schedule(root,confirm,workers,'P80_PPO/confirmation_selected')
    freeze(folder/'CONFIRMATION.json',dict(mean_difference_vs_shared=validation_score(records(confirm),records(confirm0)),
                                         reselection=False,selected_update=selected))


def evaluation_stage(root,plan,workers):
    verify_supervised(root)
    selected=read(root/'ppo/SELECTION.json'); shared=root/'shared_model'; independent=root/'independent_model'
    ppo=Path(selected['model']);require(digest(ppo/'BUNDLE.json')==selected['bundle_sha256'],'Selected PPO changed')
    def tasks_for(table,arms,targets=None):
        result=[]
        for e in plan['tables'][table]:
            for seed in plan['seeds'][table]:
                for arm in arms:
                    if table=='Table7' and e['network']=='large' and arm=='BASE':continue
                    m=independent if arm.endswith('_LEARN') or arm=='ICM_INDEPENDENT' else (
                        shared if arm in (SHARED,ZERO) else ppo if arm==PPO else None)
                    extra=dict(table=table,phase='evaluation',sample=False,record=False)
                    if arm==RANDOM:extra['targets']=targets
                    result.append(model_job(root,plan,e,seed,arm,root/'evaluation'/table/'runs'/e['instance_id']/f'seed_{seed}'/arm,m,**extra))
        return result
    controls=tasks_for('Table5',['BASE',SHARED]);schedule(root,controls,workers,'P80_Table5/Base_shared')
    shared_rows=[r for r in records(controls) if r['arm']==SHARED]
    targets={c:sum(r['selection'][c]['switches'] for r in shared_rows)/sum(r['selection'][c]['calls'] for r in shared_rows) for c in 'ICM'}
    freeze(root/'RANDOM_TARGETS.json',dict(alpha=ALPHA,source='New P80 shared SL: 54 deployment runs',
        denominator='all selection calls; same definition as Table5 v2',targets=targets,
        source_files={str(Path(t['folder'])/'RUN_COMPLETE.json'):digest(Path(t['folder'])/'RUN_COMPLETE.json') for t in controls if t['arm']==SHARED}))
    rest=tasks_for('Table5',[a for a in ARMS if a not in ('BASE',SHARED)],targets)
    schedule(root,rest,workers,'P80_Table5/remaining')
    large=tasks_for('Table6',['BASE',SHARED,PPO]);schedule(root,large,workers,'P80_Table6/S9_S11')
    network=tasks_for('Table7',['BASE',SHARED]);schedule(root,network,workers,'P80_Table7/networks')
    all_tasks=controls+rest+large+network
    require(len(all_tasks)==810 and all(completed(t) for t in all_tasks),'Incomplete P80 final search panel')
    rates=[]
    for arm in (SHARED,RANDOM,ZERO):
        rs=records([t for t in all_tasks if t.get('table')=='Table5' and t['arm']==arm])
        for c in 'ICM':
            calls=sum(r['eligibility'][c]['calls'] for r in rs);eligible=sum(r['eligibility'][c]['eligible'] for r in rs)
            switches=sum(r['eligibility'][c]['switches'] for r in rs)
            rates.append(dict(method=arm,component=c,calls=calls,eligible=eligible,switches=switches,
                percent_all_calls=100*switches/calls if calls else None,
                percent_eligible=100*switches/eligible if eligible else None,
                random_target_percent=100*targets[c] if arm==RANDOM else None,
                random_target_error_pp=100*(switches/calls-targets[c]) if arm==RANDOM and calls else None))
    if not (root/'REPLACEMENT_RATES.csv').exists():csv_write(root/'REPLACEMENT_RATES.csv',rates)
    freeze(root/'EVALUATION_TASKS.json',dict(tasks=all_tasks,alpha=ALPHA,runs=810))
