"""Compare frozen single/ensemble C selectors on the same saved validation pools."""
import argparse
import csv
import json
import sys
from pathlib import Path
from collections import defaultdict

EXP = Path(__file__).resolve().parent
sys.path.insert(0, str(EXP / 'joint_icm_multiscale_terminal_v1'))
from bridge import IndependentPolicy, predictions, legacy_learning, progress_row, digest, torch, np


def read(p):
    return json.loads(p.read_bytes())


def config_mean(rows, field):
    groups = defaultdict(lambda: defaultdict(list))
    for r in rows:
        groups[r['configuration']][r['instance']].append(r[field])
    return float(np.mean([np.mean([np.mean(v) for v in instances.values()])
                          for instances in groups.values()]))


def main(root):
    torch.set_num_threads(1)
    plan = read(root / 'PLAN.json')
    single_dir = Path(plan['models']['independent'])
    old = read(root.parent / 'ccp_component_joint_b5000_v1/PLAN.json')
    ensemble_dir = Path(old['independent_model'])
    sb, eb = read(single_dir/'BUNDLE.json'), read(ensemble_dir/'BUNDLE.json')
    if digest(single_dir/'BUNDLE.json') != plan['hashes']['independent']:
        raise ValueError('Single bundle no longer matches experiment plan')
    if len(sb['checkpoints']) != 1 or len(eb['checkpoints']) != 3:
        raise ValueError('Expected one member versus three members')
    member = sb['checkpoints'][0]
    if sb['checkpoint_sha256'][member] != eb['checkpoint_sha256'][member]:
        raise ValueError('Single member differs from ensemble member')
    for key in ('selected', 'normalization', 'scalers', 'target_scales', 'validation_instances', 'settings'):
        if sb[key] != eb[key]:
            raise ValueError('Single and ensemble metadata differ: '+key)
    provenance = eb.get('files') or read(ensemble_dir/'PLAN.json')['files']
    expected = set(eb['validation_instances']['C'])
    rows, data_files = [], {}
    seen = set()
    for filename, sha in sorted(provenance.items()):
        p = Path(filename)
        if p.name != 'DATA.json' or p.parent.name != 'COLLECT_CM' or p.parent.parent.name not in expected:
            continue
        if digest(p) != sha:
            raise ValueError('Saved validation data changed: '+str(p))
        d = read(p)
        if d['role'] != 'validation' or d['instance'] not in expected or d['instance'] in seen:
            raise ValueError('Invalid validation identity: '+str(p))
        seen.add(d['instance']); data_files[str(p)] = sha
        for index, raw in enumerate(d['rows']):
            if raw['component'] != 'C':
                continue
            if raw['instance'] != d['instance'] or raw['configuration'] != d['configuration']:
                raise ValueError('State identity mismatch')
            r = progress_row(raw, eb['settings']['progress_denominator'])
            r['_source_index'] = index
            if len(r['legal_mask']) != 8 or not r['legal_mask'][0] or len(r['labels']) != 8 or any(len(x)!=2 for x in r['labels']):
                raise ValueError('Invalid C candidate pool')
            rows.append(r)
    if seen != expected or len(rows) != eb['validation_states']['C']:
        raise ValueError(f'Missing validation data: {len(seen)}/{len(expected)} instances, {len(rows)} states')
    if {r['configuration'] for r in rows} != {f'S{i}' for i in range(9)}:
        raise ValueError('Expected S0-S8 validation coverage')
    print(f'[C-AUDIT] {len(rows)} saved validation pools; no new search evaluations', flush=True)
    policies = {'single': IndependentPolicy(single_dir), 'ensemble': IndependentPolicy(ensemble_dir)}
    preds = {name: predictions(p.models, 'C', rows, eb['scalers']['C'], eb['normalization'])
             for name, p in policies.items()}
    gate = eb['selected']['gates']['C']
    result = []
    for i, r in enumerate(rows):
        label = legacy_learning.labels(r, eb['normalization'])
        if not np.isfinite(label).all() or any(not np.isfinite(p[:,i]).all() for p in preds.values()):
            raise ValueError('Nonfinite labels or predictions')
        line = dict(configuration=r['configuration'], instance=r['instance'], source_row=r['_source_index'],
                    evaluation=r['evaluation'], legal_positions=int(sum(r['legal_mask'])))
        for name, policy in policies.items():
            choice = legacy_learning.decide(preds[name][:,i], r['legal_mask'], gate['lambda'], gate['threshold_scaled'], gate['disabled'])
            deployed = policy.select(r)
            line[name+'_choice'] = choice
            line[name+'_deployment_choice'] = deployed
            line[name+'_batch_deploy_match'] = choice == deployed
            line[name+'_switch'] = deployed != 0
            line[name+'_raw_gain'] = float(label[deployed])
            line[name+'_scaled_gain'] = float(label[deployed]/eb['target_scales']['C'])
            line[name+'_chosen_pred_mean'] = float(preds[name][:,i,deployed].mean())
        line['different_choice'] = line['single_deployment_choice'] != line['ensemble_deployment_choice']
        line['single_minus_ensemble_raw_gain'] = line['single_raw_gain']-line['ensemble_raw_gain']
        result.append(line)
        if (i+1)%48==0:
            print(f'[C-AUDIT] labeled and checked deployment {i+1}/{len(rows)}', flush=True)
    summary=[]
    for cfg in ['ALL']+[f'S{i}' for i in range(9)]:
        rr = result if cfg=='ALL' else [r for r in result if r['configuration']==cfg]
        for name in policies:
            changed = [r for r in rr if r[name+'_switch']]
            summary.append(dict(configuration=cfg, policy=name, states=len(rr),
                instances=len({r['instance'] for r in rr}),
                switch_percent=100*float(np.mean([r[name+'_switch'] for r in rr])),
                different_choice_percent=100*float(np.mean([r['different_choice'] for r in rr])),
                mean_raw_gain=config_mean(rr,name+'_raw_gain'),
                mean_scaled_gain=config_mean(rr,name+'_scaled_gain'),
                switches_better_than_rule=sum(r[name+'_raw_gain']>1e-12 for r in changed),
                switches_tied_with_rule=sum(abs(r[name+'_raw_gain'])<=1e-12 for r in changed),
                switches_worse_than_rule=sum(r[name+'_raw_gain']< -1e-12 for r in changed),
                batch_deploy_mismatches=sum(not r[name+'_batch_deploy_match'] for r in rr)))
    historical = legacy_learning.validation_score(rows,preds['ensemble'],None,gate['lambda'],gate['threshold_scaled'],
        gate['disabled'],targets=np.array([legacy_learning.labels(r,eb['normalization']) for r in rows])/eb['target_scales']['C'])
    out = root/'c_same_pool_audit'
    out.mkdir(exist_ok=True)
    for filename, values in [('SUMMARY.csv',summary),('PER_STATE.csv',result)]:
        with (out/filename).open('w',encoding='utf-8-sig',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(values[0]));w.writeheader();w.writerows(values)
    record=dict(single_bundle=str(single_dir),ensemble_bundle=str(ensemble_dir),single_sha=digest(single_dir/'BUNDLE.json'),
        ensemble_sha=digest(ensemble_dir/'BUNDLE.json'),validation_files=data_files,gate=gate,
        original_validation_score=historical,
        recorded_switch_fraction=gate['switch_fraction'],
        switch_fraction_difference=historical['switch_fraction']-gate['switch_fraction'],
        scaled_gain_difference=historical['mean_gain']-gate['mean_gain'],
        clipping={n:p.range_checks['C'] for n,p in policies.items()},
        new_searches=0,new_objective_evaluations=0,model_updates=0)
    (out/'AUDIT.json').write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
    report=['# C same-pool validation audit','',
        'Frozen C validation pools only. Gains are immediate HV differences against rule candidate 0, NOT final-search improvements or percentages.',
        'Gates/scalers/normalization are unchanged. Mean gains first average states within instances, then instances within configurations, then configurations equally.',
        'Switch percentages count states. Better/tie/worse uses raw gain tolerance 1e-12 and counts switched states only.', '',
        '| Config | Policy | States | Switch % | Different choice % | Mean raw gain | Better/tie/worse among switches | Batch/deploy mismatches |',
        '|---|---|---:|---:|---:|---:|---|---:|']
    for s in summary:
        report.append(f"| {s['configuration']} | {s['policy']} | {s['states']} | {s['switch_percent']:.2f} | {s['different_choice_percent']:.2f} | {s['mean_raw_gain']:+.8g} | {s['switches_better_than_rule']}/{s['switches_tied_with_rule']}/{s['switches_worse_than_rule']} | {s['batch_deploy_mismatches']} |")
    report += ['', '## Historical ensemble validation reproduction', '',
        f"Recorded switch fraction: {gate['switch_fraction']:.12g}; recomputed: {historical['switch_fraction']:.12g}.",
        f"Recorded scaled gain: {gate['mean_gain']:.12g}; recomputed: {historical['mean_gain']:.12g}.", '',
        'Deployment masks/gates are checked on identical saved inputs. Search-state distribution and crossover execution are not tested by this offline report.',
        'Validation was used for historical model selection; this audit is diagnostic, not an independent effectiveness test.']
    (out/'REPORT.md').write_text('\n'.join(report)+'\n',encoding='utf-8')
    print('\n'.join(report),flush=True)
    print('Saved:',out)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=EXP/'transformer_ppo_runs/diagnostics/ccp_icm_single_member_panel_b5000_v1')
    main(parser.parse_args().root)
