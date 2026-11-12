#!/usr/bin/env python3
"""Read-only paired HV and identical-pool replay audit of completed S0 joint runs."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import math
import sys
from collections import defaultdict
import numpy as np

HERE=Path(__file__).resolve().parent
INDEPENDENT='I_C_M_K8_TRANSFORMER'
SHARED='ICM_SHARED_SUPERVISED'
PPO='ICM_SHARED_TERMINAL_PPO'
PAIRS=[('共享监督−独立训练',SHARED,INDEPENDENT),('终局PPO−共享监督',PPO,SHARED)]


def read(path):return json.loads(path.read_text(encoding='utf-8'))
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()
def write(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def csv_write(path,rows):
    if not rows:return
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def read_hv(root,plan):
    with (root/'results/PER_RUN.csv').open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
    indexed={}
    for r in rows:
        if r['method'] not in (INDEPENDENT,SHARED,PPO):continue
        key=(r['instance'],int(r['algorithm_seed']),r['method'])
        if key in indexed:raise ValueError('Duplicate run: '+str(key))
        values={k:float(r[k]) for k in ('hv','old_hv')}
        if not all(math.isfinite(x) and x>=0 for x in values.values()):raise ValueError('Invalid HV')
        indexed[key]=values
    expected={(n,s,m) for n in plan['instances'] for s in plan['seeds'] for m in (INDEPENDENT,SHARED,PPO)}
    if set(indexed)!=expected:raise ValueError('Missing or unexpected paired runs; no partial-data statistics')
    return indexed


def stats(values,draws):
    x=np.asarray(values,dtype=float)
    return {'平均HV差':float(x.mean()),'95%区间下限':float(np.quantile(draws,.025)),
            '95%区间上限':float(np.quantile(draws,.975)),'差值标准差':float(x.std(ddof=1)) if x.size>1 else None,
            '差值中位数':float(np.median(x)),'最小差值':float(x.min()),'最大差值':float(x.max()),
            '正差合计':float(x[x>0].sum()),'负差合计':float(x[x<0].sum()),
            '胜':int((x>1e-12).sum()),'平':int((np.abs(x)<=1e-12).sum()),'负':int((x< -1e-12).sum())}


def paired_hv(indexed,plan,bootstrap=20000):
    summary=[];per=[];detail=[];rng=np.random.default_rng(997101)
    names=sorted(plan['instances']);seeds=sorted(plan['seeds'])
    for column,normalization in [('hv','全部方法统一界限'),('old_hv','训练归一化')]:
        for label,method,reference in PAIRS:
            matrix=np.array([[indexed[n,s,method][column]-indexed[n,s,reference][column] for s in seeds] for n in names])
            means=matrix.mean(1)
            for i,n in enumerate(names):
                sample=matrix[i,rng.integers(len(seeds),size=(bootstrap,len(seeds)))].mean(1)
                rest=np.delete(means,i)
                per.append({'口径':normalization,'比较':label,'实例':n,'配对种子数':len(seeds),
                            **stats(matrix[i],sample),'去掉本实例后其余实例平均差':float(rest.mean()) if len(rest) else None})
                for j,s in enumerate(seeds):
                    detail.append({'口径':normalization,'比较':label,'实例':n,'算法种子':s,
                        '方法HV':indexed[n,s,method][column],'参照HV':indexed[n,s,reference][column],'HV差':float(matrix[i,j])})
            ids=rng.integers(len(names),size=(bootstrap,len(names)))
            pick=rng.integers(len(seeds),size=(bootstrap,len(names),len(seeds)))
            sample=matrix[ids[:,:,None],pick].mean((1,2))
            summary.append({'口径':normalization,'比较':label,'实例数':len(names),'运行对数':matrix.size,
                **stats(matrix,sample),'均值为正的实例数':int((means>1e-12).sum()),
                '均值为负的实例数':int((means< -1e-12).sum()),'均值持平的实例数':int((np.abs(means)<=1e-12).sum())})
    return summary,per,detail


def gate_audit(shared,ppo):
    import torch
    a,b=shared.bundle,ppo.bundle
    for key in ('scalers','normalization_sha256'):
        if a[key]!=b[key]:raise ValueError('Cannot replay saved normalized tensors under different '+key)
    gates=[]
    for c in ('I','C','M'):
        x,y=a['selected']['gates'][c],b['selected']['gates'][c]
        gates.append({'component':c,'shared_lambda':x['lambda'],'ppo_lambda':y['lambda'],
            'shared_threshold_scaled':x['threshold_scaled'],'ppo_threshold_scaled':y['threshold_scaled'],
            'same_gate':x['lambda']==y['lambda'] and x['threshold_scaled']==y['threshold_scaled'],
            'deployment':'deterministic argmax of 0.99*softmax(gated_score)+0.01/8; reference ordered index 0; first index wins ties',
            'candidate_order':'I: Directed first; C: same Simple-first permutation; M: Balanced first',
            'mask':'historical S0 policy scores all 8 proposals; no additional legal-mask tensor in this policy'})
    if len(shared.models)!=len(ppo.models):raise ValueError('Different ensembles')
    diff=total=0.;changed=0;maximum=0.
    for a,b in zip(shared.models,ppo.models):
        for name,x in a.state_dict().items():
            if name.startswith('value_head.'):continue
            y=b.state_dict()[name];d=(y-x).double()
            diff+=float((d*d).sum());total+=float((x.double()**2).sum())
            changed+=int(not torch.equal(x,y));maximum=max(maximum,float(d.abs().max()))
    return gates,{'changed_actor_tensors':changed,'actor_parameter_l2_change':math.sqrt(diff),
                  'relative_actor_parameter_l2_change':math.sqrt(diff/total) if total else None,'actor_max_absolute_change':maximum}


def replay_arrays(arrays,shared,ppo,owner,stochastic,metadata):
    import torch
    output=[]
    for c in ('I','C','M'):
        if c+'_batch' not in arrays:continue
        n=len(arrays[c+'_batch']);different=shared_rule=ppo_rule=encoding_ties=manual_shared=manual_ppo=owner_argmax_errors=0
        tv=0.;error=0.;label_gain=0.
        for start in range(0,n,64):
            end=min(start+64,n)
            x=tuple(torch.tensor(arrays[c+'_'+k][start:end],dtype=torch.float32) for k in ('batch','candidate','context'))
            with torch.no_grad():
                sd,_=shared.distribution(c,x);pd,_=ppo.distribution(c,x)
                od=owner.distribution(c,x)[0] if owner is not None else None
                sc,_=shared.logits(c,x);pc,_=ppo.logits(c,x)
            sa=sd.probs.argmax(-1).numpy();pa=pd.probs.argmax(-1).numpy()
            if od is not None:
                oa=od.probs.argmax(-1).numpy();actual=np.asarray(arrays[c+'_action'][start:end],dtype=int)
                lp=od.log_prob(torch.tensor(actual,dtype=torch.long)).numpy()
                error=max(error,float(np.max(np.abs(lp-arrays[c+'_logp'][start:end]))))
                if not stochastic:owner_argmax_errors+=int(np.sum(oa!=actual))
            if c+'_label_hv' in arrays:
                labels=arrays[c+'_label_hv'][start:end];label_gain+=float(np.sum(labels[np.arange(len(sa)),pa]-labels[np.arange(len(sa)),sa]))
            different+=int(np.sum(sa!=pa));shared_rule+=int(np.sum(sa==0));ppo_rule+=int(np.sum(pa==0))
            manual_shared+=int(np.sum(sa!=sc.argmax(-1).numpy()));manual_ppo+=int(np.sum(pa!=pc.argmax(-1).numpy()))
            tv+=float((.5*(sd.probs-pd.probs).abs().sum(-1)).sum())
            for j in range(len(sa)):
                if sa[j]!=pa[j] and np.array_equal(x[0][j,sa[j]].numpy(),x[0][j,pa[j]].numpy()) and np.array_equal(x[1][j,sa[j]].numpy(),x[1][j,pa[j]].numpy()):encoding_ties+=1
        if error>2e-5 or owner_argmax_errors:raise ValueError(f'Saved trajectory does not reproduce owning policy: {metadata} {c} logp={error}, action_mismatches={owner_argmax_errors}')
        output.append({**metadata,'component':c,'candidate_pools':n,'different_choices':different,
            'different_choice_fraction':different/n if n else None,'shared_rule_choices':shared_rule,'ppo_rule_choices':ppo_rule,
            'mean_probability_total_variation':tv/n if n else None,'different_index_but_identical_encoded_candidate':encoding_ties,
            'shared_score_vs_probability_argmax_mismatches':manual_shared,'ppo_score_vs_probability_argmax_mismatches':manual_ppo,
            'owner_logp_max_error':error if owner is not None else None,
            'recorded_one_step_mean_hv_difference':label_gain/n if n and c+'_label_hv' in arrays else None})
    return output


def replay(root,plan,out,manifest):
    for name,expected in plan['runtime_files'].items():
        if sha(Path(name))!=expected:raise ValueError('Historical inference code/input changed: '+name)
        manifest[name]=expected
    sys.path.insert(0,str(HERE.parent/'joint_icm_v1'))
    from shared_policy import SharedPolicy
    import torch
    torch.set_num_threads(1)
    if plan['environment']!={'torch':str(torch.__version__),'numpy':str(np.__version__)}:
        raise ValueError('Use the original Torch/NumPy environment: '+str(plan['environment']))
    selected=read(root/'terminal/SELECTED.json');model=Path(selected['model'])
    if sha(model/'BUNDLE.json')!=selected['bundle_sha256']:raise ValueError('Selected PPO bundle changed')
    ready=read(root/'shared_model/READY.json')
    if ready['plan_sha256']!=plan['sha256'] or ready['bundle_sha256']!=sha(root/'shared_model/BUNDLE.json'):
        raise ValueError('Frozen shared model changed')
    manifest[str(root/'shared_model/READY.json')]=sha(root/'shared_model/READY.json')
    shared=SharedPolicy(root/'shared_model');ppo=SharedPolicy(model)
    gates,weights=gate_audit(shared,ppo);csv_write(out/'GATES.csv',gates)
    owners={};bundles=[root/'shared_model']+sorted((root/'terminal/checkpoints').glob('update_[0-9][0-9][0-9]'))
    for folder in bundles:
        b=read(folder/'BUNDLE.json');owners[sha(folder/'BUNDLE.json')]=folder
        for p in ['BUNDLE.json']+b['checkpoints']:manifest[str(folder/p)]=sha(folder/p)
    inventory=[];replayed=[];complete_screen=True;cache={}
    # Original validation candidate pools retain raw features and all labels.
    # Replay them without refitting gates or using the labels as model inputs.
    from shared_model import pack,order
    protocols={'i-k8-scale-data-v1':'I','c-k8-scale-data-v1':'C','m-prescreen-scale-data-v1':'M'}
    feedback_seen=defaultdict(set)
    for name,expected in shared.bundle.get('files',{}).items():
        path=Path(name)
        if path.name!='FEEDBACK.json' or path.parent.name not in set(sum(shared.bundle['validation_instances'].values(),[])):continue
        collection=path.parents[2];cp=collection/'PLAN.json'
        if not cp.exists():continue
        if sha(cp)!=shared.bundle['files'].get(str(cp)):raise ValueError('Validation collection plan changed: '+str(cp))
        manifest[str(cp)]=sha(cp)
        c=protocols.get(read(cp).get('protocol'))
        if c is None or path.parent.name not in shared.bundle['validation_instances'][c]:continue
        if sha(path)!=expected:raise ValueError('Stored validation feedback changed: '+str(path))
        if path.parent.name in feedback_seen[c]:raise ValueError('Duplicated validation feedback')
        feedback_seen[c].add(path.parent.name);raw=read(path)['rows'];x=pack(c,raw,shared.bundle['scalers'][c])
        arrays={c+'_'+k:t.numpy() for k,t in zip(('batch','candidate','context'),x)}
        arrays[c+'_label_hv']=np.asarray([[r['labels'][j]['hv_contribution'] for j in order(c,r)] for r in raw],dtype=float)
        replayed+=replay_arrays(arrays,shared,ppo,None,False,{'pool_source':'validation','instance':path.parent.name,'seed':None,
            'trajectory':collection.name+'/'+path.parent.name,'source_model':'stored labeled validation pools'})
        manifest[str(path)]=expected
    print('[S0-AUDIT] stored validation pools: '+json.dumps({c:len(feedback_seen[c]) for c in ('I','C','M')}),flush=True)
    targets=[]
    for name in plan['instances']:
        for seed in plan['seeds']:
            for method in (SHARED,PPO):
                folder=root/'screen'/name/f'seed_{seed}'/method;r=read(folder/'RUN_COMPLETE.json')
                if (r['instance'],r['algorithm_seed'],r['strategy'],r['stochastic'],r['plan_sha256'])!=(name,seed,method,False,plan['sha256']):raise ValueError('Wrong development run')
                expected_model=shared if method==SHARED else ppo
                if not r['completed'] or r['model_bundle_sha256']!=sha(expected_model.folder/'BUNDLE.json'):raise ValueError('Wrong deployed checkpoint')
                manifest[str(folder/'RUN_COMPLETE.json')]=sha(folder/'RUN_COMPLETE.json')
                exists=(folder/'TRAJECTORY.npz').exists();complete_screen &= exists
                inventory.append({'instance':name,'seed':seed,'method':method,'full_model_input_saved':exists,
                    'I_calls':r['shared_policy_counts']['I']['calls'],'C_calls':r['shared_policy_counts']['C']['calls'],'M_calls':r['shared_policy_counts']['M']['calls']})
                if exists:targets.append(('development',folder,r))
    for folder in sorted((root/'terminal/rollouts').glob('*')):
        if not (folder/'RUN_COMPLETE.json').exists():continue
        r=read(folder/'RUN_COMPLETE.json')
        if not r.get('completed') or not r.get('stochastic') or r['plan_sha256']!=plan['sha256'] or r['instance'] not in plan['train_instances']:raise ValueError('Wrong training replay source')
        if (folder/'TRAJECTORY.npz').exists():targets.append(('training',folder,r))
    for i,(role,folder,r) in enumerate(targets):
        owner_folder=owners.get(r['model_bundle_sha256'])
        if owner_folder is None:raise ValueError('Missing owning checkpoint for '+str(folder))
        if str(owner_folder) not in cache:cache[str(owner_folder)]=SharedPolicy(owner_folder)
        owner=cache[str(owner_folder)]
        if owner.bundle['scalers']!=shared.bundle['scalers']:raise ValueError('Saved inputs have different scaling')
        with np.load(folder/'TRAJECTORY.npz',allow_pickle=False) as data:arrays={k:data[k] for k in data.files}
        for c in ('I','C','M'):
            if len(arrays.get(c+'_action',[]))!=r['shared_policy_counts'][c]['calls']:raise ValueError('Incomplete recorded model calls')
        metadata={'pool_source':role,'instance':r['instance'],'seed':r['algorithm_seed'],'trajectory':folder.name,'source_model':owner_folder.name}
        replayed+=replay_arrays(arrays,shared,ppo,owner,r['stochastic'],metadata)
        manifest[str(folder/'TRAJECTORY.npz')]=sha(folder/'TRAJECTORY.npz');manifest[str(folder/'RUN_COMPLETE.json')]=sha(folder/'RUN_COMPLETE.json')
        print(f'[S0-AUDIT] replay {i+1}/{len(targets)} {role}/{folder.name}; zero evaluations',flush=True)
    summary=[]
    for role in sorted({r['pool_source'] for r in replayed}):
        for c in ('I','C','M'):
            part=[r for r in replayed if r['pool_source']==role and r['component']==c];n=sum(r['candidate_pools'] for r in part)
            if not n:continue
            different=sum(r['different_choices'] for r in part)
            summary.append({'候选来源':role,'组件':c,'轨迹数':len(part),'候选池数':n,'不同选择次数':different,'不同选择比例':different/n,
                            '共享监督选择规则次数':sum(r['shared_rule_choices'] for r in part),'PPO选择规则次数':sum(r['ppo_rule_choices'] for r in part),
                            '平均概率分布差异TV':sum(r['mean_probability_total_variation']*r['candidate_pools'] for r in part)/n})
    csv_write(out/'CANDIDATE_STORAGE.csv',inventory);csv_write(out/'SAME_POOL_PER_TRAJECTORY.csv',replayed);csv_write(out/'02_同候选选择差异.csv',summary)
    # Deployment action counts are descriptive only: policies visit different pools.
    deploy=[]
    for r in inventory:
        folder=root/'screen'/r['instance']/f'seed_{r["seed"]}'/r['method'];value=read(folder/'RUN_COMPLETE.json')
        for c in ('I','C','M'):deploy.append({'instance':r['instance'],'seed':r['seed'],'method':r['method'],'component':c,**value['shared_policy_counts'][c]})
    csv_write(out/'DEPLOYMENT_COUNTS.csv',deploy)
    return {'selected_update':selected['update'],'actor_weight_change':weights,'same_pool_summary':summary,
            'validation_pool_instances_per_component':{c:len(feedback_seen[c]) for c in ('I','C','M')},
            'complete_development_candidate_replay_available':bool(complete_screen),
            'development_runs_with_full_inputs':sum(r['full_model_input_saved'] for r in inventory),
            'development_runs_expected':len(inventory),
            'qualification':'Only saved complete model inputs are replayed. Validation pools were already used for model selection; any known one-step label difference is not final-search gain. Training pools diagnose decision changes but do not explain held-out development performance. Different indices can encode duplicate solutions. Different-run rule-switch counts are not same-pool disagreement. No final causal gain is inferred.'}


def execute(root,out,hv_only=False):
    if out.exists():raise ValueError('Use a new diagnostic output directory: '+str(out))
    if out==root or out in root.parents:raise ValueError('Do not overwrite source')
    plan=read(root/'PLAN.json');quick=read(root/'results/QUICK_RESULT.json')
    if not quick['completed'] or not read(root/'STATUS.json')['completed']:raise ValueError('Source experiment incomplete')
    if plan['settings']['population']!=20 or plan['settings']['budget']!=400:raise ValueError('This audit is for the completed S0 B400 experiment')
    inputs=read(root/'results/INPUT_MANIFEST.json')
    if inputs['plan_sha256']!=plan['sha256']:raise ValueError('Report has wrong plan')
    for name,expected in inputs['files'].items():
        if sha(Path(name))!=expected:raise ValueError('Source report input changed: '+name)
    indexed=read_hv(root,plan);summary,per,detail=paired_hv(indexed,plan)
    out.mkdir(parents=True)
    csv_write(out/'00_配对HV差汇总.csv',summary);csv_write(out/'01_各实例配对HV差.csv',per);csv_write(out/'PER_PAIR_HV.csv',detail)
    manifest={str(root/p):sha(root/p) for p in ('PLAN.json','results/PER_RUN.csv','results/QUICK_RESULT.json','results/INPUT_MANIFEST.json','terminal/SELECTED.json')}
    result={'completed':False,'source':str(root),'paired_hv':summary,'new_searches':0,'new_objective_evaluations':0,'new_training_steps':0,
            'hv_qualification':'Absolute paired differences, instance means weighted equally. Per-instance CI resamples paired seeds; pooled CI resamples instances then paired seeds. Three previously inspected development instances only.'}
    write(out/'QUICK_RESULT.json',result)
    for filename in ('00_配对HV差汇总.csv','01_各实例配对HV差.csv'):
        print('\n'+filename+'\n'+(out/filename).read_text(encoding='utf-8-sig'),flush=True)
    if not hv_only:result['choice_replay']=replay(root,plan,out,manifest)
    result['completed']=True;write(out/'QUICK_RESULT.json',result);write(out/'INPUT_MANIFEST.json',{'files':manifest,'audit_code_sha256':sha(Path(__file__))})
    filename='02_同候选选择差异.csv'
    if (out/filename).exists():print('\n'+filename+'\n'+(out/filename).read_text(encoding='utf-8-sig'),flush=True)
    if not hv_only:print(json.dumps(result['choice_replay'],ensure_ascii=False,indent=2),flush=True)
    print('[S0-AUDIT] COMPLETE; zero searches, zero objective evaluations: '+str(out),flush=True)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);diag=HERE.parent/'transformer_ppo_runs/diagnostics'
    p.add_argument('--source',type=Path,default=diag/'ccp_s0_joint_b400_v1');p.add_argument('--out',type=Path,default=diag/'ccp_s0_joint_b400_v1_paired_audit_v1')
    p.add_argument('--hv-only',action='store_true',help='Only paired final-HV audit; no model loading')
    a=p.parse_args();execute(a.source.resolve(),a.out.resolve(),a.hv_only)


if __name__=='__main__':main()
