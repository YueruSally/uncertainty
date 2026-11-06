#!/usr/bin/env python3
"""Train the precommitted direct-gain K8 ranker on a fresh validation split."""
import argparse
from collections import Counter
import json
from pathlib import Path
import random
import time

import numpy as np
import torch

from m_operator_offline import digest, read_json, write_json
from m_prescreen_surrogate import (CandidateGainTransformer, choose_by_gain,
    fit_scaler, gain_loss, gain_targets, pack_rows)
from run_learning_components_deterministic import _normalizer
from train_m_prescreen_surrogate import load_rows

SOURCE = Path(__file__).resolve().parent
PROTOCOL = SOURCE/'experiments/m_prescreen_protocol_v1.json'
EPOCHS = 80
CHECKPOINT_EPOCHS = (10,20,40,60,80)
BATCH_SIZE = 16


def log(message):
    print(f'[M-K8-RANK] {message}', flush=True)


def load_validation(collection, design):
    plan = read_json(collection/'PLAN.json')
    if (plan.get('protocol') != 'm-prescreen-k8-ranker-validation-v2' or
            plan.get('instances') != design['validation_instances'] or
            plan.get('seeds') != design['validation_collection_seeds'] or
            not (collection/'COLLECTION_COMPLETE').exists()):
        raise ValueError('requires the precommitted ranker-v2 validation collection')
    rows, hashes = [], {}
    for instance in design['validation_instances']:
        for seed in design['validation_collection_seeds']:
            folder = collection/'runs'/instance/f'seed_{seed}'
            complete = read_json(folder/'RUN_COMPLETE.json')
            feedback = folder/'FEEDBACK.json'
            if (complete.get('feedback_sha256') != digest(feedback) or
                    complete.get('instance_id') != instance or
                    complete.get('algorithm_seed') != seed):
                raise ValueError(f'validation provenance mismatch: {folder}')
            value = read_json(feedback)
            for row in value['rows']:
                if len(row['labels']) != 8 or row['candidate_sources'][0] != 'balanced':
                    raise ValueError('ranker validation state is not exact K8')
                rows.append({**row, 'instance': instance,
                             'trajectory_seed': seed})
            hashes[str(feedback.resolve())] = digest(feedback)
    if len(rows) != read_json(collection/'QUICK_RESULT.json')['state_count']:
        raise ValueError('ranker validation state count changed')
    return rows, hashes


def batches(indices, rng):
    order = np.asarray(indices).copy()
    rng.shuffle(order)
    return [order[start:start+BATCH_SIZE]
            for start in range(0,len(order),BATCH_SIZE)]


def evaluate_loss(model, rows, scaler, normalizer, target_scale):
    model.eval()
    total = 0.
    with torch.inference_mode():
        for start in range(0,len(rows),BATCH_SIZE):
            part = rows[start:start+BATCH_SIZE]
            packed = pack_rows(part, scaler, normalizer)
            loss = gain_loss(model(*packed[:4]), gain_targets(part,target_scale))
            total += loss.item()*len(part)
    return total/len(rows)


def fit_one(seed, train_rows, validation_rows, scaler, normalizer,
            target_scale, out):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.set_num_threads(1)
    shape = pack_rows(train_rows[:1], scaler, normalizer)
    model = CandidateGainTransformer(shape[0].shape[-1],
        shape[1].shape[-1],shape[2].shape[-1])
    optimizer = torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4)
    rng = np.random.default_rng(seed)
    history, best = [], None
    started = time.monotonic()
    for epoch in range(1,EPOCHS+1):
        model.train(); total=count=0
        for indices in batches(np.arange(len(train_rows)),rng):
            part=[train_rows[int(i)] for i in indices]
            packed=pack_rows(part,scaler,normalizer)
            loss=gain_loss(model(*packed[:4]),gain_targets(part,target_scale))
            optimizer.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            optimizer.step(); total += loss.item()*len(part); count += len(part)
        if epoch in CHECKPOINT_EPOCHS:
            valid=evaluate_loss(model,validation_rows,scaler,normalizer,target_scale)
            record={'epoch':epoch,'train_loss':total/count,'validation_loss':valid}
            history.append(record)
            log(f'seed={seed} epoch={epoch}/{EPOCHS} train={total/count:.6f} valid={valid:.6f}')
            if best is None or (valid,epoch)<(best['validation_loss'],best['epoch']):
                best={**record,'state':{name:value.detach().cpu().clone()
                    for name,value in model.state_dict().items()}}
    model.load_state_dict(best.pop('state'))
    checkpoint=out/f'seed_{seed}.pt'
    torch.save({'protocol':'m-prescreen-direct-gain-ranker-v2','seed':seed,
        'epoch':best['epoch'],'model':model.state_dict(),
        'batch_feature_count':shape[0].shape[-1],
        'candidate_feature_count':shape[1].shape[-1],
        'context_feature_count':shape[2].shape[-1]},checkpoint)
    return model,{'seed':seed,'selected':best,'history':history,
        'checkpoint':str(checkpoint.resolve()),
        'checkpoint_sha256':digest(checkpoint),
        'elapsed_seconds':time.monotonic()-started}


def predict(model, rows, scaler, normalizer, scale):
    values=[]
    model.eval()
    with torch.inference_mode():
        for start in range(0,len(rows),BATCH_SIZE):
            packed=pack_rows(rows[start:start+BATCH_SIZE],scaler,normalizer)
            values.append(model(*packed[:4]).numpy()*scale)
    return np.concatenate(values)


def score(rows,prediction):
    selected=choose_by_gain(prediction)
    records=[]
    for row,action in zip(rows,selected):
        labels=row['labels']; baseline=float(labels[0]['hv_contribution'])
        gain=float(labels[int(action)]['hv_contribution'])-baseline
        oracle=max(float(label['hv_contribution'])-baseline for label in labels)
        records.append({'instance':row['instance'],'trajectory_seed':row['trajectory_seed'],
            'generation':row['generation'],'evaluation':row['evaluation'],
            'selected':int(action),'net_gain':gain,'oracle_gain':oracle})
    per=[]
    for instance in sorted({row['instance'] for row in records}):
        part=[row for row in records if row['instance']==instance]
        gain=np.asarray([row['net_gain'] for row in part])
        per.append({'instance':instance,'states':len(part),
            'mean_net_gain':float(gain.mean()),
            'switches':sum(row['selected']!=0 for row in part),
            'switch_fraction':float(np.mean([row['selected']!=0 for row in part])),
            'positive_gain_total':float(gain[gain>0].sum()),
            'negative_gain_total':float(gain[gain<0].sum()),
            'gain_p10':float(np.quantile(gain,.1)),
            'mean_oracle_gain':float(np.mean([row['oracle_gain'] for row in part]))})
    gain=np.asarray([row['net_gain'] for row in records])
    return {'states':len(records),'mean_net_gain':float(gain.mean()),
        'switches':sum(row['selected']!=0 for row in records),
        'switch_fraction':float(np.mean([row['selected']!=0 for row in records])),
        'positive_gain_total':float(gain[gain>0].sum()),
        'negative_gain_total':float(gain[gain<0].sum()),
        'gain_p10':float(np.quantile(gain,.1)),
        'action_counts':dict(Counter(str(row['selected']) for row in records)),
        'per_instance':per,'decisions':records}


def execute(args):
    protocol=read_json(PROTOCOL); design=protocol['ranker_v2']
    original_rows,original_hashes=load_rows(args.training_collection.resolve(),protocol)
    train_names=set(design['training_instances'])
    train_rows=[row for row in original_rows if row['instance'] in train_names]
    validation_rows,validation_hashes=load_validation(
        args.validation_collection.resolve(),design)
    if ({row['instance'] for row in train_rows}!=train_names or
            {row['instance'] for row in validation_rows}!=set(design['validation_instances'])):
        raise ValueError('ranker instance split changed')
    normalizer=_normalizer(args.normalization)
    scaler=fit_scaler(train_rows,normalizer)
    alternative=np.asarray([float(label['hv_contribution'])-
        float(row['labels'][0]['hv_contribution'])
        for row in train_rows for label in row['labels'][1:]])
    target_scale=max(float(np.sqrt(np.mean(alternative**2))),1e-10)
    seeds=tuple(design['training_seeds'])
    if (design['epochs']!=EPOCHS or design['checkpoint_epochs']!=list(CHECKPOINT_EPOCHS)
            or design['batch_size']!=BATCH_SIZE or design['learning_rate']!=3e-4
            or design['weight_decay']!=1e-4):
        raise ValueError('ranker implementation differs from frozen design')
    out=args.out.resolve(); out.mkdir(parents=True,exist_ok=True)
    files=(Path(__file__).resolve(),SOURCE/'m_prescreen_surrogate.py',PROTOCOL)
    frozen={'protocol':'m-prescreen-direct-gain-ranker-v2',
        'training_collection':str(args.training_collection.resolve()),
        'validation_collection':str(args.validation_collection.resolve()),
        'normalization':str(args.normalization.resolve()),
        'normalization_sha256':digest(args.normalization),
        'train_instances':sorted(train_names),
        'validation_instances':design['validation_instances'],
        'train_states':len(train_rows),'validation_states':len(validation_rows),
        'training_seeds':list(seeds),'epochs':EPOCHS,
        'checkpoint_epochs':list(CHECKPOINT_EPOCHS),'batch_size':BATCH_SIZE,
        'target_scale':target_scale,'scaler':scaler,
        'source_sha256':{**original_hashes,**validation_hashes,
            **{str(path):digest(path) for path in files}}}
    if (out/'PLAN.json').exists() and read_json(out/'PLAN.json')!=frozen:
        raise ValueError('ranker frozen inputs changed; choose a new output')
    write_json(out/'PLAN.json',frozen)
    models=[]; training=[]
    for seed in seeds:
        model,report=fit_one(seed,train_rows,validation_rows,scaler,
            normalizer,target_scale,out)
        models.append(model); training.append(report)
    prediction=np.mean([predict(model,validation_rows,scaler,normalizer,
        target_scale) for model in models],axis=0)
    validation=score(validation_rows,prediction)
    passed=all(row['mean_net_gain']>0 and row['switches']>0
               for row in validation['per_instance'])
    bundle={**frozen,'checkpoints':[row['checkpoint'] for row in training],
        'checkpoint_sha256':{row['checkpoint']:row['checkpoint_sha256']
                             for row in training},
        'selected_epochs':[row['selected']['epoch'] for row in training],
        'offline_gate_pass':passed}
    write_json(out/'BUNDLE.json',bundle)
    write_json(out/'TRAINING.json',{'completed':True,'models':training})
    result={'completed':True,'new_objective_evaluations':0,'new_searches':0,
        'new_ppo_updates':0,'training_states':len(train_rows),
        'validation_states':len(validation_rows),
        'selected_epochs':bundle['selected_epochs'],
        'target_scale':target_scale,'offline_validation':validation,
        'offline_gate_pass':passed,
        'next_action':('run bounded paired B400 search without PPO' if passed else
            'stop direct-gain ranker before search; it did not beat Balanced on both fresh validation instances')}
    write_json(out/'QUICK_RESULT.json',result)
    (out/'TRAINING_COMPLETE').write_text(digest(out/'BUNDLE.json')+'\n')
    log(json.dumps({key:result[key] for key in ('training_states',
        'validation_states','selected_epochs','target_scale',
        'offline_gate_pass','next_action')},indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training-collection',type=Path,required=True)
    parser.add_argument('--validation-collection',type=Path,required=True)
    parser.add_argument('--normalization',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    execute(parser.parse_args())


if __name__=='__main__':
    main()
