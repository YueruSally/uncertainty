#!/usr/bin/env python3
"""One bounded S11 repair pilot; original models and results stay untouched."""
from bootstrap import *
import argparse,fcntl,signal,subprocess,time


def attach_source(root,source,settings):
    source=source.resolve()
    if root==source or source in root.parents or root in source.parents:
        raise ValueError('Source and new outputs must be separate trees')
    frozen=legacy_run.verify_frozen(source);oldplan=read_json(source/'PLAN.json')
    if frozen['plan_sha256']!=oldplan['sha256']:
        raise ValueError('Source plan mismatch')
    for name,sha in oldplan['files'].items():
        if digest(NEW.parent/name)!=sha:raise ValueError('Original source code changed: '+name)
    old=oldplan['protocol']
    for key in ('population','budget','scenario_count','confidence','progress_denominator','pool_size','configurations'):
        if old[key]!=settings[key]:raise ValueError('Frozen C incompatible with new settings: '+key)
    if read_json(source/'RULE_SELECTION.json')['profiles']['C']!=settings['rule_profiles']['C']:
        raise ValueError('C rule must remain unchanged')
    bundle=read_json(source/'models/C/BUNDLE.json')
    # No old I/M weights or old screening outcomes are inputs to the new fit.
    names=['BUNDLE.json','INPUT_SCALER.json','VALIDATION.json']+bundle['checkpoints']
    for name in names:
        src=source/'models/C'/name;dest=root/'models/C'/name;dest.parent.mkdir(parents=True,exist_ok=True)
        if dest.exists() and digest(dest)!=digest(src):raise ValueError('Frozen C copy changed')
        if not dest.exists():dest.write_bytes(src.read_bytes())
    record={'source':str(source),'plan_sha256':oldplan['sha256'],'frozen_manifest_sha256':digest(source/'models/FROZEN.json'),
            'C_assets':{name:digest(root/'models/C'/name) for name in names},
            'previous_instance_signatures':[r['batch_signature'] for r in read_json(source/'instances/catalog.json')['instances']],
            'previous_version':'c2f3110','old_results_overwritten':False}
    freeze(root/'SOURCE.json',record)


def plan(root,settings):
    from common import sources
    files=sources()+list(NEW.glob('*.py'))+[NEW/'protocol.json',
        NEW.parent/'formal_icm_s0_v1/assets/data/data_expanded.xlsx',
        NEW.parent/'formal_icm_s0_v1/assets/reward_normalization.json']
    if digest(ENGINE/'baseline_uncertainty.py')!=settings['baseline_file_sha256']:
        raise ValueError('Transport equations changed')
    import platform,numpy,torch
    value={'protocol':settings,'feature_version':FEATURE_VERSION,
           'files':{str(p.relative_to(NEW.parent)):digest(p) for p in sorted(set(files))},
           'source_sha256':digest(root/'SOURCE.json'),
           'environment':{'python':platform.python_version(),'numpy':numpy.__version__,'torch':torch.__version__}}
    value['sha256']=canonical_digest(value);freeze(root/'PLAN.json',value);return value['sha256']


def jobs(root,cat,component):
    return [(r['instance_id'],r['algorithm_seed'],'COLLECT_'+component,
             root/'collection'/r['instance_id']/('COLLECT_'+component))
            for r in cat['instances'] if r['development_role'] in ('training','validation')]


def fit_process(root,sha,deadline):
    proc=subprocess.Popen([sys.executable,'-u',str(NEW/'run.py'),'--fit-worker','--out',str(root),'--plan-sha',sha])
    try:
        while proc.poll() is None:
            if time.monotonic()>=deadline:raise legacy_run.Paused('Training deadline')
            try:proc.wait(timeout=30)
            except subprocess.TimeoutExpired:print('[S11-REPAIR] training heartbeat',flush=True)
        if proc.returncode:raise RuntimeError('Training failed; inspect preceding traceback')
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:proc.wait(timeout=10)
            except subprocess.TimeoutExpired:proc.kill();proc.wait()


def final_report(root,s,cat,rows):
    from report import report
    q=report(rows,s,root)
    q['exploratory_interval_condition']=q.pop('all_components_met_reliable_gain')
    collected=[read_json(folder/'RUN_COMPLETE.json') for c in 'IM' for _,_,_,folder in jobs(root,cat,c)]
    q.update(pilot_only=True,automatic_expansion_started=False,C_retrained=False,
             changes=s['declared_changes'],source=read_json(root/'SOURCE.json'),
             collection_full_searches=sum(r['arm']=='COLLECT_M' for r in collected),
             collection_initialization_only=sum(r['arm']=='COLLECT_I' for r in collected),
             total_completed_search_evaluations=sum(r['real_objective_evaluation_count'] for r in rows+collected),
             additional_label_evaluations=sum(sum(r['audit_evaluations'].values()) for r in collected),
             interrupted_attempts=len(list((root/'interrupted').glob('*'))) if (root/'interrupted').exists() else 0,
             qualification='S11 development only: two new instances, one paired seed. I/M updated candidate/rule/training protocols; C frozen. All seven methods use the same instances, scenarios and real-evaluation cap. Outcomes do not establish stable superiority.',
             next_action='Stop and review all outcomes; no automatic extra training, search, PPO or independent-scenario reevaluation.')
    write_json(root/'QUICK_RESULT.json',q)
    print('[S11-REPAIR] COMPLETE',flush=True)
    print((root/'RESULT_TABLE.csv').read_text(),flush=True)
    print('[S11-REPAIR] final gates '+str({c:read_json(root/'models'/c/'BUNDLE.json')['gate'] for c in 'ICM'}),flush=True)
    return q


def execute(root,s,source,source_catalog,workers,stage='all'):
    root=root.resolve();root.mkdir(parents=True,exist_ok=True)
    if workers<1:raise ValueError('workers must be positive')
    lock=(root/'pipeline.lock').open('a+')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise RuntimeError('This output already has an active pipeline')
    deadline=time.monotonic()+s['invocation_seconds']
    try:
        attach_source(root,source,s);sha=plan(root,s)
        from new_instances import prepare
        cat=prepare(root,s,source_catalog)
        freeze(root/'RULE_SELECTION.json',{'plan_sha256':sha,'profiles':s['rule_profiles'],
               'selection':'Prespecified new I capacity-coordinated Directed and M GUARDED; C copied unchanged; no outcome-driven rule calibration'})
        if stage=='prepare':return
        print('[S11-REPAIR] I-label preflight first; C frozen; 6 initialization-only + 6 M collections + 14 screening searches at production settings',flush=True)
        execute_jobs(root,s,jobs(root,cat,'I'),sha,deadline,workers)
        from fit import signal_check
        if not signal_check(root,s,cat):
            stop={'completed':False,'stopped':True,'reason':'I candidates still lack positive training opportunities across independent instances',
                  'screen_started':False,'next_action':'Review I_SIGNAL_CHECK.json; no automatic collection expansion or full search'}
            write_json(root/'STATUS.json',stop)
            write_json(root/'QUICK_RESULT.json',{**stop,'I_signal_check':read_json(root/'I_SIGNAL_CHECK.json')})
            print('[S11-REPAIR] STOP '+str(stop),flush=True);return
        if stage=='preflight':
            write_json(root/'STATUS.json',{'completed':True,'stage':'preflight','screen_started':False});return
        execute_jobs(root,s,jobs(root,cat,'M'),sha,deadline,workers)
        fit_process(root,sha,deadline)
        if legacy_run.verify_frozen(root)['plan_sha256']!=sha:raise ValueError('Frozen model plan mismatch')
        screen=[(r['instance_id'],seed,arm,root/'screen'/r['instance_id']/f'seed_{seed}'/arm)
                for r in cat['instances'] if r['development_role']=='screen'
                for seed in s['algorithm_seeds'] for arm in METHODS]
        rows=execute_jobs(root,s,screen,sha,deadline,workers)
        final_report(root,s,cat,rows)
        write_json(root/'STATUS.json',{'completed':True,'stage':'all','final_test_opened':False})
    except legacy_run.Paused as e:
        write_json(root/'STATUS.json',{'completed':False,'paused':True,'reason':str(e),'resume':'Repeat the same command; completed runs reused'})
        print('[S11-REPAIR] PAUSED '+str(e),flush=True)
    except Exception as e:
        write_json(root/'STATUS.json',{'completed':False,'failed':True,'error':str(e)});raise
    finally:lock.close()


def main():
    s=read_json(NEW/'protocol.json');p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,default=NEW.parent/'transformer_ppo_runs/diagnostics/ccp_s11_component_repair_pilot_v1')
    p.add_argument('--source',type=Path,default=Path(s['source_pilot_default']))
    p.add_argument('--source-catalog',type=Path,default=Path(s['source_catalog_default']))
    p.add_argument('--workers',type=int,default=2);p.add_argument('--stage',choices=['all','prepare','preflight'],default='all')
    a=p.parse_args()
    def stop(signum,frame):raise legacy_run.Paused('Received signal '+str(signum))
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    execute(a.out,s,a.source,a.source_catalog,a.workers,a.stage)


if __name__=='__main__':main()
