#!/usr/bin/env python3
"""Complete terminal feedback for the already frozen single-intervention task.

Counterfactual SEARCHES are training cost. Deployment never evaluates branches.
All branches change one concrete choice, then execute the unchanged Balanced
policy to B400. A hindsight best branch is not a deployable performance claim.
"""
import argparse
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

SOURCE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE))

import numpy as np
import torch
from m_candidate_information import CandidatePolicy, CandidateController
from m_operator_offline import read_json, write_json, digest, transform
from m_terminal_hv_ppo import (decision_input_digest, initialize, checkpoint_model,
    save_checkpoint, optimizer_for, update_ppo, METHODS)
from multi_instance_policy import Decision
from run_learning_components_deterministic import _normalizer
from run_m_operator_closed_loop_v1 import load_run
from run_m_terminal_hv_ppo_v1 import verify_pair, report_search
from transformer_ppo.episode import run_episode
from transformer_ppo.instance_catalog import read_instance

VALID_INSTANCES = ('train-S0-014', 'train-S0-016')
EPOCHS = 40
PPO_SEEDS = (901001, 901002, 901003, 901004)
PPO_VALIDATION_UPDATES = (0, 2, 4)


def log(message):
    print(f'[M-TERMINAL-FEEDBACK] {message}', flush=True)


class FixedIntervention(CandidatePolicy):
    """Training-data executor, not a learned policy or an online oracle."""
    def __init__(self, instance, seed, action, bundle):
        super().__init__(instance, seed, 'HEURISTIC_BALANCED')
        self.forced_action = action
        self.bundle = read_json(bundle)
        self.first_supported_input_sha256 = None
        self.fixed_choices = 0
        self.transition = None

    def decide(self, summary, specific, mask):
        if not self.fixed_choices and mask[4] and mask[:4].any():
            if not mask[self.forced_action]:
                raise ValueError('predeclared branch is illegal at the shared decision point')
            self.fixed_choices = 1
            self.first_supported_input_sha256 = decision_input_digest(summary, specific, mask)
            x = transform(np.concatenate([summary, specific], axis=1)[None], self.bundle['scaler'])
            self.transition = {'tokens': x[0].tolist(), 'eligible': mask.tolist(), 'action': self.forced_action}
            self.switches += int(self.forced_action != 4)
            self.action_counts[str(self.forced_action)] += 1
            probabilities = [0.]*5
            probabilities[self.forced_action] = 1.
            return Decision(self.forced_action, probabilities, [0.]*5,
                {'inference_seconds': 0., 'switched': self.forced_action != 4,
                 'decision_kind': 'fixed_terminal_training_branch'})
        return super().decide(summary, specific, mask)


def assert_terminal_pair(baseline, branch, first_hash):
    if (baseline['pairing'] != branch['pairing']
            or branch['first_supported_input_sha256'] != first_hash
            or branch['real_objective_evaluation_count'] != 400
            or branch.get('ppo_updates', 0) or branch.get('counterfactual_evaluation_count', 0)
            or branch.get('training_feedback_evaluations', 0)):
        raise ValueError('terminal branch changed state, scenarios, budget, or executor')
    if branch['fixed_choices'] != 1 or branch['closed_loop_switches'] > 1:
        raise ValueError('terminal branch must contain exactly one fixed choice')


def branch_worker(a):
    torch.set_num_threads(1)
    info, _ = read_instance(a.instance)
    policy = FixedIntervention(info['instance_id'], a.seed, a.action, a.bundle)
    result = run_episode(a.data, a.instance, a.out, policy, _normalizer(a.normalization), a.seed,
        evaluation_budget=400, pop_size=20, generations=1000, evaluation_mode='ccp', component='M',
        scenario_count=100, ccp_confidence=.9, uncertainty_profile='transport_border_saa_v1',
        inference_controller_factories={'M': CandidateController})
    if policy.fixed_choices != 1 or policy.model_evaluations:
        raise ValueError('fixed branch did not execute exactly once without a learned model')
    result.update(strategy='TRAINING_BRANCH', algorithm_seed=a.seed, plan_sha256=a.plan_sha,
        fixed_action=a.action, fixed_choices=policy.fixed_choices, source_bundle_sha256=digest(a.bundle),
        first_supported_input_sha256=policy.first_supported_input_sha256,
        role='counterfactual full search for training only; never included as deployed policy performance')
    write_json(a.out/'INPUT.json', policy.transition)
    result['input_sha256'] = digest(a.out/'INPUT.json')
    write_json(a.out/'RUN_COMPLETE.json', result)


def load_contexts(source):
    """Reuse authenticated receipts, including zero-return and Balanced samples."""
    plan = read_json(source/'PLAN.json')
    result = read_json(source/'QUICK_RESULT.json')
    if (plan.get('protocol') != 'single-intervention-ppo-v2' or not result.get('completed')
            or not (source/'PIPELINE_COMPLETE').exists()):
        raise ValueError('requires the completed single-intervention v2 experiment')
    for path, sha in plan['input_sha256'].items():
        if digest(path) != sha:
            raise ValueError(f'frozen source input changed: {path}')
    sha = digest(source/'PLAN.json')
    contexts, files = [], [source/'PLAN.json', source/'QUICK_RESULT.json']
    pairs = [(n, s) for s in plan['train_seeds'] for n in plan['train_instances']]
    for i, (name, seed) in enumerate(pairs):
        index = i//plan['episodes_per_update']
        hfolder = source/'training_baselines'/name/f'seed_{seed}'/'HEURISTIC_BALANCED'
        rfolder = source/'rollouts'/f'update_{index:03d}'/name/f'seed_{seed}'/'TRANSFORMER_PPO'
        baseline, rollout = (load_run(p, name, sha) for p in (hfolder, rfolder))
        verify_pair([baseline, rollout])
        trajectory_path = rfolder/'TRAJECTORY.json'
        if digest(trajectory_path) != rollout['trajectory_sha256']:
            raise ValueError('source trajectory changed')
        trajectory = read_json(trajectory_path)
        if len(trajectory['transitions']) != 1 or trajectory['policy_sha256'] != rollout['policy_sha256']:
            raise ValueError('source is not a single frozen choice')
        transition = trajectory['transitions'][0]
        action = transition['action']
        if not transition['eligible'][4] or not transition['eligible'][action]:
            raise ValueError('source action mask is invalid')
        if not np.isfinite(rollout['final_hypervolume']) or baseline['final_hypervolume'] <= 0:
            raise ValueError('invalid source terminal HV')
        hvs = [None]*5
        hvs[4] = baseline['final_hypervolume']
        if action == 4 and not np.isclose(rollout['final_hypervolume'], hvs[4], rtol=0, atol=1e-12):
            raise ValueError('sampled Balanced source changed the baseline result')
        hvs[action] = rollout['final_hypervolume']
        contexts.append({'instance': name, 'seed': seed, 'tokens': transition['tokens'],
            'eligible': transition['eligible'], 'first_input_sha256': rollout['first_supported_input_sha256'],
            'baseline': baseline, 'terminal_hvs': hvs})
        files.extend([trajectory_path, hfolder/'RUN_COMPLETE.json', hfolder/'configuration.json',
            rfolder/'RUN_COMPLETE.json', rfolder/'configuration.json'])
    if len(contexts) != plan['training_episode_count']:
        raise ValueError('incomplete source context grid')
    for name in plan['test_instances']:
        for seed in plan['test_seeds']:
            folder=source/'development'/name/f'seed_{seed}'/'HEURISTIC_BALANCED'
            files.extend([folder/'RUN_COMPLETE.json',folder/'configuration.json'])
    bundles = [Path(p) for p in plan['input_sha256'] if Path(p).name == 'BUNDLE.json']
    if len(bundles) != 1:
        raise ValueError('ambiguous source bundle')
    return plan, bundles[0], contexts, files


def expected_score(model, rows):
    x = torch.tensor([r['tokens'] for r in rows], dtype=torch.float32)
    mask = torch.tensor([r['eligible'] for r in rows], dtype=torch.bool)
    gains = torch.tensor([[0. if q is None else q/r['terminal_hvs'][4]-1
                           for q in r['terminal_hvs']] for r in rows], dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        probs = model(x, mask)[0].softmax(-1)
        expected = (probs*gains).sum(-1).numpy()
    per_instance = {name: float(np.mean([v for r,v in zip(rows,expected) if r['instance']==name]))
                    for name in sorted({r['instance'] for r in rows})}
    return {'mean_expected_terminal_gain': float(np.mean(list(per_instance.values()))),
        'per_instance_expected_gain': per_instance,
        'mean_switch_probability': float((1-probs[:,4]).mean()),
        'definition': 'exact expectation over the five recorded terminal branches, conditional on these contexts; not new-instance deployed performance'}


def fit_complete_feedback(bundle, rows, out, plan_sha):
    started=time.perf_counter()
    train = [r for r in rows if r['instance'] not in VALID_INSTANCES]
    valid = [r for r in rows if r['instance'] in VALID_INSTANCES]
    if not train or not valid:
        raise ValueError('requires separate training and validation instances')
    info = read_json(bundle)
    if set(VALID_INSTANCES) & set(info['training_instances']):
        raise ValueError('full-feedback validation overlaps the original pretraining data')
    if any((q is None) == legal for r in rows for q,legal in zip(r['terminal_hvs'],r['eligible'])):
        raise ValueError('all legal terminal outcomes are required; no missing-action masking')
    x = torch.tensor([r['tokens'] for r in train], dtype=torch.float32)
    mask = torch.tensor([r['eligible'] for r in train], dtype=torch.bool)
    q = torch.tensor([[0. if v is None else v/r['terminal_hvs'][4]-1 for v in r['terminal_hvs']]
                      for r in train], dtype=torch.float32)
    scale = max(float(q[mask].std(unbiased=False)), 1e-4)
    model = initialize(bundle)
    optimizer = torch.optim.Adam(model.actors.parameters(), lr=1e-4)
    selected, history = None, []
    for epoch in range(EPOCHS+1):
        if epoch:
            model.eval()
            logits, _ = model(x, mask)
            # Full-information objective, NOT an off-policy PPO update.
            loss = -(logits.softmax(-1)*q/scale).sum(-1).mean()
            optimizer.zero_grad()
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.actors.parameters(), 1.)
            if not torch.isfinite(loss) or not torch.isfinite(norm):
                raise FloatingPointError('non-finite complete-feedback policy gradient')
            optimizer.step()
        if epoch % 5 == 0:
            metrics = {'epoch': epoch, 'training': expected_score(model,train), 'validation': expected_score(model,valid)}
            history.append(metrics)
            if selected is None or metrics['validation']['mean_expected_terminal_gain'] > selected['validation']['mean_expected_terminal_gain']+1e-12:
                selected = deepcopy(metrics)
                save_checkpoint(out/'pretrained.pt', {'protocol':'terminal-hv-ppo-v1',
                    'execution_mode':'single-intervention','bundle_sha256':digest(bundle),'plan_sha256':plan_sha,
                    'update':0,'feature_count':len(info['scaler']['mean']),'ensemble_count':len(model.actors),
                    'model':model.state_dict(),'optimizer':optimizer_for(model).state_dict(),
                    'complete_feedback_epoch':epoch,'training_kind':'full-information terminal policy objective; PPO updates=0'})
            log(f'full-feedback epoch={epoch}/{EPOCHS} expected_validation={100*metrics["validation"]["mean_expected_terminal_gain"]:+.4f}%')
    report = {'completed':True,'training_contexts':len(train),'validation_contexts':len(valid),
        'validation_instances':list(VALID_INSTANCES),'training_only_reward_scale':scale,
        'selected':selected,'history':history,'new_ppo_updates':0,
        'optimizer_steps':EPOCHS,'elapsed_seconds':time.perf_counter()-started,
        'checkpoint':str(out/'pretrained.pt'),'sha256':digest(out/'pretrained.pt'),
        'dataset_sha256':digest(out/'DATASET.json')}
    write_json(out/'PRETRAINING.json',report)
    return report


def collect_and_fit(a, deadline):
    torch.set_num_threads(1)
    source_plan, bundle, contexts, files = load_contexts(a.source.resolve())
    root = a.out.resolve()
    for p in (a.data.resolve(), a.normalization.resolve()):
        if source_plan['input_sha256'].get(str(p)) != digest(p):
            raise ValueError('workbook/normalization differ from frozen source')
    missing = [(r['instance'],r['seed'],i) for r in contexts for i,v in enumerate(r['terminal_hvs']) if r['eligible'][i] and v is None]
    files += [Path(__file__), Path(__file__).with_name('run_m_terminal_feedback_v3.sh'), bundle]+[Path(p) for p in source_plan['input_sha256']]
    for r in contexts:
        p=(a.instances/(r['instance']+'.json')).resolve()
        if source_plan['input_sha256'].get(str(p)) != digest(p):
            raise ValueError('instance catalog differs from frozen source')
    plan={'protocol':'complete-terminal-feedback-v3','single_intervention':True,'source':str(a.source.resolve()),
        'data':str(a.data.resolve()),'normalization':str(a.normalization.resolve()),
        'instances':str(a.instances.resolve()),'bundle':str(bundle),
        'source_plan_sha256':digest(a.source/'PLAN.json'),'contexts':len(contexts),
        'train_instances':[n for n in source_plan['train_instances'] if n not in VALID_INSTANCES],
        'validation_instances':list(VALID_INSTANCES),'validation_role':'held-out instances for model selection only',
        'ccp_scenarios':100,'population':20,'budget':400,'confidence':.9,
        'new_training_branch_searches':len(missing),'new_training_branch_evaluations':400*len(missing),
        'old_source_training_evaluations':source_plan['training_episode_count']*800,
        'ppo_seeds':list(PPO_SEEDS),'ppo_validation_updates':list(PPO_VALIDATION_UPDATES),
        'test_instances':source_plan['test_instances'],'test_seeds':source_plan['test_seeds'],
        'maximum_additional_searches':len(missing)+2*len(PPO_SEEDS)*(len(source_plan['train_instances'])-len(VALID_INSTANCES))
            +2*len(source_plan['test_instances'])*len(source_plan['test_seeds']),
        'development_heuristic_reused':True,
        'time_note':'heuristic development timings reused from source; this is not a simultaneous hardware-matched speed comparison',
        'final_rule':'both development instances positive versus Balanced and full-feedback-pretrained; paired-seed interval lower bound >0 against both; selected PPO update >0',
        'epochs':EPOCHS,'learning_rate':1e-4,'checkpoint_candidates':list(range(0,EPOCHS+1,5)),
        'workers':a.workers,'threads_per_worker':1,'timeout_per_search':300,'timeout_per_invocation':2700,
        'training_objective':'maximize sum_a pi(a|pre-action state) * (final_HV_a / final_HV_Balanced - 1); train-side constant scale',
        'input_sha256':{str(p.resolve()):digest(p) for p in files}}
    if (root/'PLAN.json').exists() and read_json(root/'PLAN.json') != plan:
        raise ValueError('frozen terminal-feedback plan changed')
    write_json(root/'PLAN.json',plan)
    sha=digest(root/'PLAN.json')
    if (root/'PRETRAINING.json').exists():
        existing=read_json(root/'PRETRAINING.json')
        if digest(existing['checkpoint']) != existing['sha256'] or digest(root/'DATASET.json') != existing['dataset_sha256']:
            raise ValueError('saved pretrained checkpoint changed')
        log('complete feedback and pretraining already committed')
        return
    stop=threading.Event()
    env={**os.environ,'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'}
    indexed={(r['instance'],r['seed']):r for r in contexts}
    (root/'logs').mkdir(exist_ok=True)

    def branch(job):
        name,seed,action=job
        folder=root/'branches'/name/f'seed_{seed}'/f'action_{action}'
        if not (folder/'RUN_COMPLETE.json').exists():
            if stop.is_set() or time.monotonic() >= deadline:
                raise TimeoutError('invocation cap reached; completed branches retained')
            if folder.exists() and any(folder.iterdir()):
                old=root/'interrupted_runs'/f'{name}_{seed}_{action}_{time.time_ns()}'
                old.parent.mkdir(exist_ok=True)
                folder.rename(old)
            cmd=[sys.executable,'-u',str(Path(__file__).resolve()),'--worker','--instance',str(a.instances/(name+'.json')),
                '--data',str(a.data),'--normalization',str(a.normalization),'--bundle',str(bundle),'--out',str(folder),
                '--seed',str(seed),'--action',str(action),'--plan-sha',sha]
            logfile=root/'logs'/f'{name}_{seed}_{action}_{time.time_ns()}.log'
            with logfile.open('w') as handle, subprocess.Popen(cmd,cwd=SOURCE,env=env,stdout=handle,stderr=subprocess.STDOUT) as proc:
                started=last=time.monotonic()
                try:
                    while proc.poll() is None:
                        now=time.monotonic()
                        if stop.is_set() or now>=deadline or now-started>300:
                            raise TimeoutError(f'terminal branch timeout: {logfile}')
                        if now-last>15:
                            log(f'running {name}/{seed}/action={action} pid={proc.pid} elapsed={now-started:.0f}s')
                            last=now
                        time.sleep(.2)
                finally:
                    if proc.poll() is None:
                        proc.kill(); proc.wait()
                if proc.returncode:
                    raise RuntimeError(f'branch failed: {logfile}\n'+ '\n'.join(logfile.read_text(errors='replace').splitlines()[-20:]))
        row=load_run(folder,name,sha)
        source=indexed[name,seed]
        assert_terminal_pair(source['baseline'],row,source['first_input_sha256'])
        if row['fixed_action']!=action or row['input_sha256']!=digest(folder/'INPUT.json'):
            raise ValueError('branch identity/input changed')
        recorded=read_json(folder/'INPUT.json')
        np.testing.assert_allclose(recorded['tokens'],source['tokens'],rtol=0,atol=0)
        if recorded['eligible'] != source['eligible']:
            raise ValueError('branch legality mask changed')
        return job,row['final_hypervolume']

    log(f'contexts={len(contexts)} missing_terminal_branches={len(missing)} all_costs_counted; no new PPO')
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        futures=[pool.submit(branch,j) for j in missing]
        try:
            for i,f in enumerate(as_completed(futures),1):
                (name,seed,action),hv=f.result()
                indexed[name,seed]['terminal_hvs'][action]=hv
                write_json(root/'STATUS.json',{'stage':'complete terminal feedback','completed':i,'total':len(missing)})
                log(f'branches {i}/{len(missing)} {name}/{seed}/action={action}')
        except BaseException:
            stop.set()
            for f in futures:
                f.cancel()
            raise
    rows=[{k:v for k,v in r.items() if k!='baseline'} for r in contexts]
    write_json(root/'DATASET.json',{'rows':rows,'plan_sha256':sha,
        'new_training_searches':len(missing),'new_objective_evaluations':400*len(missing),
        'definition':'full B400 terminal returns after one fixed action; remaining search Balanced'})
    fit_complete_feedback(bundle,rows,root,sha)
    write_json(root/'STATUS.json',{'stage':'pretraining complete; bounded PPO and deployed comparison still required',
        'new_training_searches':len(missing),'new_objective_evaluations':400*len(missing)})
    log('complete feedback + full-information pretraining COMPLETE; not a deployed-performance claim')


def finish_ppo_and_search(a, deadline):
    root=a.out.resolve()
    plan=read_json(root/'PLAN.json')
    sha=digest(root/'PLAN.json')
    if (root/'PIPELINE_COMPLETE').exists():
        log('complete result retained; no retraining or model selection')
        return
    rows=read_json(root/'DATASET.json')['rows']
    valid=[r for r in rows if r['instance'] in plan['validation_instances']]
    stop=threading.Event()
    env={**os.environ,'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'}

    def job(stage,name,seed,method,checkpoint=None,sample=False):
        folder=root/stage/name/f'seed_{seed}'/method
        model_sha=digest(checkpoint) if checkpoint else None
        if not (folder/'RUN_COMPLETE.json').exists():
            if stop.is_set() or time.monotonic() >= deadline:
                raise TimeoutError('bounded invocation ended; completed work is reusable')
            if folder.exists() and any(folder.iterdir()):
                old=root/'interrupted_runs'/f'{stage.replace("/","_")}_{name}_{seed}_{method}_{time.time_ns()}'
                old.parent.mkdir(exist_ok=True)
                folder.rename(old)
            command=[sys.executable,'-u',str(SOURCE/'run_m_terminal_hv_ppo_v1.py'),'--worker',
                '--single-intervention','--instance',str(Path(plan['instances'])/(name+'.json')),
                '--normalization',plan['normalization'],'--data',plan['data'],'--bundle',plan['bundle'],
                '--out',str(folder),'--seed',str(seed),'--method',method,'--plan-sha',sha]
            if checkpoint:
                command += ['--checkpoint',str(checkpoint),'--checkpoint-sha',model_sha]
            if sample:
                command += ['--sample']
            logfile=root/'logs'/f'{stage.replace("/","_")}_{name}_{seed}_{method}_{time.time_ns()}.log'
            with logfile.open('w') as handle, subprocess.Popen(command,cwd=SOURCE,env=env,stdout=handle,stderr=subprocess.STDOUT) as proc:
                started=last=time.monotonic()
                try:
                    while proc.poll() is None:
                        now=time.monotonic()
                        if stop.is_set() or now>=deadline or now-started>300:
                            raise TimeoutError(f'search cap reached: {logfile}')
                        if now-last>=15:
                            log(f'running {stage}/{name}/{method} pid={proc.pid} elapsed={now-started:.0f}s')
                            last=now
                        time.sleep(.2)
                finally:
                    if proc.poll() is None:
                        proc.kill(); proc.wait()
                if proc.returncode:
                    raise RuntimeError(f'search failed: {logfile}\n'+'\n'.join(logfile.read_text(errors='replace').splitlines()[-20:]))
        result=load_run(folder,name,sha)
        if (result['policy_sha256']!=model_sha or result['collect_for_training']!=sample
                or result['sampling']!=(method!=METHODS[0]) or result['strategy']!=method
                or result['algorithm_seed']!=seed):
            raise ValueError('saved policy/sample identity mismatch')
        if sample:
            path=folder/'TRAJECTORY.json'
            if digest(path)!=result['trajectory_sha256']:
                raise ValueError('on-policy trajectory changed')
            t=read_json(path)
            if t['policy_sha256']!=model_sha or len(t['transitions'])!=1:
                raise ValueError('expected one on-policy decision from the current actor')
            result['transitions']=t['transitions']
        return result

    def parallel(jobs,stage):
        values=[]
        with ThreadPoolExecutor(max_workers=a.workers) as pool:
            futures=[pool.submit(job,*j) for j in jobs]
            try:
                for f in as_completed(futures):
                    values.append(f.result())
                    write_json(root/'STATUS.json',{'stage':stage,'completed':len(values),'total':len(jobs)})
                    log(f'{stage} {len(values)}/{len(jobs)} complete')
            except BaseException:
                stop.set()
                for f in futures:
                    f.cancel()
                raise
        return values

    current=root/'pretrained.pt'
    selection,validation,updates=None,[],[]
    for index in range(len(plan['ppo_seeds'])+1):
        model,saved=checkpoint_model(current,digest(current))
        if saved['plan_sha256']!=sha or saved['update']!=index:
            raise ValueError('checkpoint from another frozen update')
        if index:
            updates.append(saved['metrics'])
        if index in plan['ppo_validation_updates']:
            score=expected_score(model,valid)
            record={'update':index,'checkpoint':str(current),'sha256':digest(current),**score}
            validation.append(record)
            if selection is None or record['mean_expected_terminal_gain']>selection['mean_expected_terminal_gain']+1e-12:
                selection=record
            log(f'PPO validation update={index} expected_gain={100*score["mean_expected_terminal_gain"]:+.4f}% selected={selection["update"]}')
        write_json(root/'TRAINING_PROGRESS.json',{'completed_ppo_updates':index,'validation':validation,'selected':selection,'ppo':updates})
        if index==len(plan['ppo_seeds']):
            break
        next_checkpoint=root/'ppo_checkpoints'/f'update_{index+1:03d}.pt'
        if not next_checkpoint.exists():
            seed=plan['ppo_seeds'][index]
            batch=[]
            for name in plan['train_instances']:
                batch += [('ppo_baselines',name,seed,METHODS[0]),
                    (f'ppo_rollouts/update_{index:03d}',name,seed,METHODS[2],current,True)]
            results=parallel(batch,f'on-policy block {index+1}')
            grouped={}
            for r in results:
                grouped.setdefault(r['instance'],{})[r['strategy']]=r
            episodes=[]
            for name in plan['train_instances']:
                h,t=grouped[name][METHODS[0]],grouped[name][METHODS[2]]
                verify_pair([h,t])
                episodes.append({'balanced_hv':h['final_hypervolume'],'final_hv':t['final_hypervolume'],'transitions':t['transitions']})
            optimizer=optimizer_for(model)
            optimizer.load_state_dict(saved['optimizer'])
            optimizer_started=time.perf_counter()
            metrics=update_ppo(model,optimizer,episodes,index+1)
            metrics['optimizer_seconds']=time.perf_counter()-optimizer_started
            save_checkpoint(next_checkpoint,{**saved,'update':index+1,'model':model.state_dict(),
                'optimizer':optimizer.state_dict(),'metrics':metrics,'training_kind':'on-policy terminal PPO after full-information pretraining'})
            log(f'PPO committed update={index+1} actions={metrics["actions"]} reward={metrics["mean_terminal_reward"]:+.6f}')
        current=next_checkpoint
    if (root/'SELECTED_POLICY.json').exists() and read_json(root/'SELECTED_POLICY.json')!=selection:
        raise ValueError('frozen model selection changed')
    write_json(root/'SELECTED_POLICY.json',selection)
    # Fixed development grid; use every method/seed, never pick outcomes.
    jobs=[]
    baseline=[]
    source=Path(plan['source'])
    for name in plan['test_instances']:
        for seed in plan['test_seeds']:
            baseline.append(load_run(source/'development'/name/f'seed_{seed}'/METHODS[0],name,plan['source_plan_sha256']))
            jobs.extend([('development',name,seed,METHODS[1],root/'pretrained.pt'),
                ('development',name,seed,METHODS[2],Path(selection['checkpoint']))])
    tested=parallel(jobs,'frozen B400 development')+baseline
    report=report_search(tested,{**plan,'single_intervention':True})
    receipts=[read_json(p) for p in root.glob('**/RUN_COMPLETE.json') if 'interrupted_runs' not in p.parts]
    nbranches=plan['new_training_branch_searches']
    npairs=len(plan['train_instances'])*len(plan['ppo_seeds'])
    expected=nbranches+2*npairs+2*len(plan['test_instances'])*len(plan['test_seeds'])
    if len(receipts)!=expected or any(r['real_objective_evaluation_count']!=400 for r in receipts):
        raise ValueError('completed search cost/grid differs from predeclared plan')
    passed=report['development_gain_established'] and selection['update']>0
    pretraining=read_json(root/'PRETRAINING.json')
    source_receipts=[read_json(p) for p in source.glob('**/RUN_COMPLETE.json') if 'interrupted_runs' not in p.parts]
    complete={'completed':True,'protocol':plan['protocol'],'single_intervention':True,
        **report,'selected':selection,'development_pass':passed,
        'ppo_checkpoint_selected':selection['update']>0,
        'training_summary':{'completed_ppo_updates':len(updates),'ppo_learning_decisions':sum(m['actions'] for m in updates),
            'full_information_optimizer_steps':pretraining['optimizer_steps'],
            'full_information_training_seconds':pretraining['elapsed_seconds'],
            'ppo_optimizer_seconds':sum(m['optimizer_seconds'] for m in updates),
            'new_completed_search_optimization_seconds':sum(r['optimisation_seconds'] for r in receipts),
            'new_full_branch_training_evaluations':nbranches*400,'new_ppo_and_baseline_evaluations':2*npairs*400,
            'completed_run_objective_evaluations':400*len(receipts),'completed_search_runs':len(receipts),
            'source_experiment_cost':read_json(source/'QUICK_RESULT.json')['training_summary'],
            'source_completed_search_optimization_seconds':sum(r['optimisation_seconds'] for r in source_receipts),
            'interrupted_runs':len(list((root/'interrupted_runs').glob('*'))),
            'uncommitted_interruption_cost_excluded':bool(list((root/'interrupted_runs').glob('*')))},
        'equal_test_parameters':True,'scenario_exact':True,'time_note':plan['time_note'],
        'conclusion':'PROMISING_REQUIRES_NEW_INSTANCES_AND_OOS' if passed else 'FULL_TERMINAL_FEEDBACK_NO_RELIABLE_PPO_GAIN',
        'qualification':'Validation integrates frozen full-search branch returns; deployment samples one action and evaluates no extra branch. Development results are not independent confirmation.'}
    write_json(root/'COMPLETE.json',{**complete,'runs':tested,'validation':validation,'ppo_updates':updates})
    write_json(root/'QUICK_RESULT.json',complete)
    with (root/'RESULT_TABLE.csv').open('w') as handle:
        writer=csv.writer(handle)
        writer.writerow(['instance','method','mean_hv','mean_seconds','paired_percent_vs_balanced'])
        for r in report['per_instance']:
            for method in METHODS:
                effect=0. if method==METHODS[0] else r['pretrained_vs_balanced' if method==METHODS[1] else 'ppo_vs_balanced']['mean']
                writer.writerow([r['instance'],method,r['methods'][method]['mean_hv'],r['methods'][method]['mean_seconds'],effect])
    (root/'PIPELINE_COMPLETE').write_text('complete; inspect development_pass\n')
    write_json(root/'STATUS.json',{'stage':'complete','development_pass':passed})
    log(f'COMPLETE development_pass={passed}; selected PPO update={selection["update"]}')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path)
    p.add_argument('--instances',type=Path)
    p.add_argument('--data',type=Path,default=SOURCE/'data/data_expanded.xlsx')
    p.add_argument('--normalization',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--workers',type=int,choices=(1,2),default=2)
    p.add_argument('--worker',action='store_true')
    p.add_argument('--instance',type=Path)
    p.add_argument('--bundle',type=Path)
    p.add_argument('--seed',type=int)
    p.add_argument('--action',type=int,choices=range(5))
    p.add_argument('--plan-sha')
    a=p.parse_args()
    if a.worker:
        branch_worker(a)
        return
    if a.source is None or a.instances is None:
        p.error('coordinator requires --source and --instances')
    a.out.mkdir(parents=True,exist_ok=True)
    with (a.out/'.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            deadline=time.monotonic()+2700
            collect_and_fit(a,deadline)
            finish_ppo_and_search(a,deadline)
        except Exception as error:
            write_json(a.out/'STOPPED.json',{'reason':str(error),'completed_branches_retained':True})
            raise


if __name__=='__main__':
    main()
