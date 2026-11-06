#!/usr/bin/env python3
"""Instance-internal temporal learnability screen using saved K8 feedback."""
import argparse
import json
from pathlib import Path
import random
import time

import numpy as np
import torch

from m_operator_offline import digest, read_json, write_json
from m_prescreen_surrogate import (CandidateGainTransformer, fit_scaler,
    gain_loss, gain_targets, pack_rows)
from run_learning_components_deterministic import _normalizer
from train_m_prescreen_ranker_v2 import predict, score
from train_m_prescreen_surrogate import load_rows

SOURCE=Path(__file__).resolve().parent
PROTOCOL=SOURCE/'experiments/m_prescreen_protocol_v1.json'
SEEDS=(914001,914002,914003)
EPOCHS=80
BATCH_SIZE=16


def log(message):
    print(f'[M-K8-WITHIN] {message}',flush=True)


def temporal_split(rows,instance):
    selected=[row for row in rows if row['instance']==instance]
    by_trajectory={}
    for row in selected:
        by_trajectory.setdefault(row['trajectory_seed'],[]).append(row)
    early,late=[],[]
    details=[]
    for seed,trajectory in sorted(by_trajectory.items()):
        trajectory=sorted(trajectory,key=lambda row:(
            row['generation'],row['evaluation'],row['event']))
        cut=len(trajectory)//2
        if cut<4 or len(trajectory)-cut<4:
            raise ValueError('trajectory is too short for temporal screen')
        early.extend(trajectory[:cut]); late.extend(trajectory[cut:])
        details.append({'trajectory_seed':seed,'states':len(trajectory),
            'early_states':cut,'late_states':len(trajectory)-cut,
            'last_early_generation':trajectory[cut-1]['generation'],
            'first_late_generation':trajectory[cut]['generation']})
    if not early or not late:
        raise ValueError(f'missing temporal split: {instance}')
    return early,late,details


def shuffled_batches(length,rng):
    order=np.arange(length); rng.shuffle(order)
    return [order[start:start+BATCH_SIZE]
        for start in range(0,length,BATCH_SIZE)]


def fit(seed,rows,scaler,normalizer,target_scale,out,instance):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.set_num_threads(1)
    packed=pack_rows(rows[:1],scaler,normalizer)
    model=CandidateGainTransformer(packed[0].shape[-1],
        packed[1].shape[-1],packed[2].shape[-1])
    optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4)
    rng=np.random.default_rng(seed); history=[]; started=time.monotonic()
    for epoch in range(1,EPOCHS+1):
        model.train(); total=count=0
        for indices in shuffled_batches(len(rows),rng):
            part=[rows[int(index)] for index in indices]
            values=pack_rows(part,scaler,normalizer)
            loss=gain_loss(model(*values[:4]),gain_targets(part,target_scale))
            optimizer.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            optimizer.step(); total+=loss.item()*len(part); count+=len(part)
        if epoch in (20,40,60,80):
            history.append({'epoch':epoch,'train_loss':total/count})
            log(f'{instance} seed={seed} epoch={epoch}/{EPOCHS} train={total/count:.6f}')
    checkpoint=out/f'{instance}_seed_{seed}.pt'
    torch.save({'protocol':'m-prescreen-within-instance-v1',
        'instance':instance,'seed':seed,'epoch':EPOCHS,
        'model':model.state_dict(),'batch_feature_count':packed[0].shape[-1],
        'candidate_feature_count':packed[1].shape[-1],
        'context_feature_count':packed[2].shape[-1]},checkpoint)
    return model,{'seed':seed,'epoch':EPOCHS,'history':history,
        'checkpoint':str(checkpoint.resolve()),
        'checkpoint_sha256':digest(checkpoint),
        'elapsed_seconds':time.monotonic()-started}


def execute(args):
    protocol=read_json(PROTOCOL); design=protocol['within_instance_screen']
    if (design['training_seeds']!=list(SEEDS) or design['epochs']!=EPOCHS or
            design['new_objective_evaluations']!=0):
        raise ValueError('implementation differs from frozen temporal design')
    rows,sources=load_rows(args.collection.resolve(),protocol)
    normalizer=_normalizer(args.normalization)
    out=args.out.resolve(); out.mkdir(parents=True,exist_ok=True)
    files=(Path(__file__).resolve(),SOURCE/'m_prescreen_surrogate.py',
           SOURCE/'train_m_prescreen_ranker_v2.py',PROTOCOL)
    frozen={'protocol':'m-prescreen-within-instance-v1',
        'collection':str(args.collection.resolve()),
        'collection_plan_sha256':digest(args.collection/'PLAN.json'),
        'normalization':str(args.normalization.resolve()),
        'normalization_sha256':digest(args.normalization),
        'instances':design['source_instances'],'seeds':list(SEEDS),
        'epochs':EPOCHS,'batch_size':BATCH_SIZE,'learning_rate':3e-4,
        'weight_decay':1e-4,'split':design['split'],
        'source_sha256':{**sources,**{str(path):digest(path) for path in files}}}
    if (out/'PLAN.json').exists() and read_json(out/'PLAN.json')!=frozen:
        raise ValueError('temporal screen inputs changed; choose a new output')
    write_json(out/'PLAN.json',frozen)
    reports=[]
    for instance in design['source_instances']:
        early,late,split=temporal_split(rows,instance)
        scaler=fit_scaler(early,normalizer)
        alternatives=np.asarray([float(label['hv_contribution'])-
            float(row['labels'][0]['hv_contribution'])
            for row in early for label in row['labels'][1:]])
        target_scale=max(float(np.sqrt(np.mean(alternatives**2))),1e-10)
        models=[]; training=[]
        for seed in SEEDS:
            model,record=fit(seed,early,scaler,normalizer,target_scale,out,instance)
            models.append(model); training.append(record)
        early_prediction=np.mean([predict(model,early,scaler,normalizer,target_scale)
                                  for model in models],axis=0)
        late_prediction=np.mean([predict(model,late,scaler,normalizer,target_scale)
                                 for model in models],axis=0)
        reports.append({'instance':instance,'early_states':len(early),
            'late_states':len(late),'target_scale':target_scale,
            'split':split,'training':training,
            'early_fit':score(early,early_prediction),
            'late_test':score(late,late_prediction)})
    passed=all(row['late_test']['mean_net_gain']>0 and
               row['late_test']['switches']>0 for row in reports)
    bundle={'protocol':'m-prescreen-within-instance-v1','plan':frozen,
        'models':{row['instance']:{'target_scale':row['target_scale'],
            'checkpoints':[model['checkpoint'] for model in row['training']],
            'checkpoint_sha256':{model['checkpoint']:model['checkpoint_sha256']
                for model in row['training']}}
            for row in reports},'temporal_screen_pass':passed}
    write_json(out/'BUNDLE.json',bundle)
    result={'completed':True,'new_objective_evaluations':0,
        'new_searches':0,'new_ppo_updates':0,'instances':reports,
        'temporal_screen_pass':passed,
        'next_action':('implement pretrained plus instance-online update in paired B400 search'
            if passed else 'end K8 prescreen branch; B400 trajectories do not support instance-internal temporal learning')}
    write_json(out/'QUICK_RESULT.json',result)
    (out/'SCREEN_COMPLETE').write_text(digest(out/'BUNDLE.json')+'\n')
    log(json.dumps({'temporal_screen_pass':passed,
        'instances':[{'instance':row['instance'],
            'early_states':row['early_states'],'late_states':row['late_states'],
            'early_mean_gain':row['early_fit']['mean_net_gain'],
            'late_mean_gain':row['late_test']['mean_net_gain'],
            'late_switches':row['late_test']['switches'],
            'late_oracle':[part['mean_oracle_gain']
                for part in row['late_test']['per_instance']][0]}
            for row in reports],'next_action':result['next_action']},indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--collection',type=Path,required=True)
    parser.add_argument('--normalization',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    execute(parser.parse_args())


if __name__=='__main__':
    main()
