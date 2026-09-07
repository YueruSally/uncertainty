#!/usr/bin/env python3
"""Fixed-decision marginal q90 stability; no optimisation or Pareto filtering."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
import numpy as np
import baseline_uncertainty as base
from run_ccp_candidate_pool import decision_signature
from run_ev_ccp_oos_pilot import scenario_digest
from run_formal_ev_vs_ccp_s30_30runs import restore_individual
from run_ccp30_q90_oos_validation import load_ccp30_rows, audit_training_semantics

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'formal_ev_vs_ccp_s30_30runs/run_01/CCP30/final_feasible_nondominated.json'
DEFAULT_OUT = ROOT / 'formal_ev_vs_ccp_s30_30runs/run1_sample_stability'
OBJECTIVES = ('cost', 'emission', 'makespan')


def atomic_json(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def write_csv(path, rows):
    tmp = path.with_suffix('.tmp')
    with tmp.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    tmp.replace(path)


def measure(sample, sorted_oos):
    sample = np.asarray(sample, dtype=float)
    sorted_oos = np.asarray(sorted_oos, dtype=float)
    if not sample.size or not sorted_oos.size or not np.isfinite(sample).all() or not np.isfinite(sorted_oos).all():
        raise ValueError('Empty or non-finite objective outcomes')
    q = base.empirical_ccp_quantile(sample, .9)
    oq = base.empirical_ccp_quantile(sorted_oos, .9)
    coverage = float(np.searchsorted(sorted_oos, q, side='right') / len(sorted_oos))
    return dict(q90_reestimate=q, q90_oos=oq, signed_error=q-oq,
                signed_error_pct=100*(q-oq)/oq if oq != 0 else None,
                coverage=coverage, exceedance=1-coverage,
                coverage_gap_pp=100*(coverage-.9),
                absolute_coverage_gap_pp=100*abs(coverage-.9))


def records(arrays, oos, sources, seed, sizes, role):
    rows = []
    for i, source in enumerate(sources):
        for size in sizes:
            for j, obj in enumerate(OBJECTIVES):
                rows.append(dict(solution_id=source['source_solution_id'],
                    decision_fingerprint=source['decision_fingerprint'],
                    role=role, seed=seed, S=size, objective=obj,
                    **measure(arrays[i, j, :size], oos[i, j])))
    return rows


def summarize(rows):
    groups = {}
    for row in rows:
        groups.setdefault((row['role'], row['seed'], row['S'], row['objective']), []).append(row)
    result = []
    for (role, seed, size, obj), group in groups.items():
        c = np.array([r['coverage'] for r in group])
        e = [r['signed_error_pct'] for r in group if r['signed_error_pct'] is not None]
        result.append(dict(role=role, seed=seed, S=size, objective=obj,
            solution_count=len(group), mean_coverage=float(c.mean()),
            median_coverage=float(np.median(c)), p05_coverage=float(np.quantile(c,.05)),
            p95_coverage=float(np.quantile(c,.95)), min_coverage=float(c.min()),
            max_coverage=float(c.max()), mean_abs_gap_pp=float(100*np.abs(c-.9).mean()),
            mean_signed_error_pct=float(np.mean(e)) if e else None,
            below_90_count=int((c<.9).sum()), above_90_count=int((c>.9).sum()),
            equal_90_count=int((c==.9).sum())))
    return result


def report(out, rows, count):
    summaries = summarize(rows)
    write_csv(out/'per_seed_summary.csv', summaries)
    independent = [r for r in summaries if r['role']=='independent_reestimate']
    aggregate = []
    for size in sorted({r['S'] for r in independent}):
        for obj in OBJECTIVES:
            g = [r for r in independent if r['S']==size and r['objective']==obj]
            c = np.array([r['mean_coverage'] for r in g])
            aggregate.append(dict(S=size, objective=obj, seeds=len(g),
                mean_coverage=float(c.mean()), sd_seed_mean_coverage=float(c.std(ddof=1)) if len(c)>1 else None,
                min_seed_mean_coverage=float(c.min()), max_seed_mean_coverage=float(c.max()),
                mean_absolute_gap_pp=float(np.mean([r['mean_abs_gap_pp'] for r in g]))))
    write_csv(out/'by_sample_size.csv', aggregate)
    stability=[]
    groups={}
    for r in rows:
        if r['role']=='independent_reestimate':
            groups.setdefault((r['solution_id'],r['S'],r['objective']),[]).append(r)
    for (sid,size,obj),g in groups.items():
        c=np.array([r['coverage'] for r in g]);q=np.array([r['q90_reestimate'] for r in g])
        stability.append(dict(solution_id=sid,S=size,objective=obj,seeds=len(g),
            mean_coverage=float(c.mean()),sd_coverage=float(c.std(ddof=1)) if len(c)>1 else None,
            q90_mean=float(q.mean()),q90_sd=float(q.std(ddof=1)) if len(q)>1 else None))
    write_csv(out/'per_solution_stability.csv',stability)
    text = ['# Fixed CCP30 decision q90 stability', '',
        f'{count} fixed decisions. All original/independent thresholds share the same OOS set.',
        'This is conditional on the existing Run-1 decisions, not a repeated-optimisation experiment.',
        'The original training sample selected these decisions; independent samples only re-estimate them.',
        'Coverage concerns separate objective thresholds, not on-time or joint feasibility.',
        'Seeds use nested prefixes of a master sample; changing the maximum S changes the random stream.',
        'Solution-level observations share scenarios and are correlated. Seed SD below is descriptive, not a confidence interval.',
        '', '| S | Objective | Seeds | Mean coverage | SD of seed means | Mean absolute gap (pp) |',
        '|---|---|---|---|---|---|']
    for r in aggregate:
        sd=r['sd_seed_mean_coverage']
        text.append(f"| {r['S']} | {r['objective']} | {r['seeds']} | {r['mean_coverage']:.2%} | {format(sd,'.2%') if sd is not None else 'NA'} | {r['mean_absolute_gap_pp']:.3f} |")
    text += ['', '90% is the nominal target. ceil(0.9*S)/(S+1) is only an expectation reference for an independently fixed decision with continuous iid outcomes; it is not a guarantee or a universal correction. Timetable outcomes can have ties.',
             'Neither closeness to 90% nor improved estimates demonstrates that independently re-optimised CCP models improve.']
    (out/'REPORT.md').write_text('\n'.join(text)+'\n')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(14,4),sharey=True)
    for ax,obj in zip(axes,OBJECTIVES):
        for seed in sorted({r['seed'] for r in independent}):
            g=sorted([r for r in independent if r['seed']==seed and r['objective']==obj],key=lambda r:r['S'])
            ax.plot([r['S'] for r in g],[100*r['mean_coverage'] for r in g],marker='o',alpha=.45)
        ax.axhline(90,color='black',linestyle='--',label='Nominal 90%')
        original=next(r for r in summaries if r['role']=='original_training' and r['objective']==obj)
        ax.scatter([30],[100*original['mean_coverage']],marker='*',s=150,color='red',label='Original training S=30',zorder=5)
        ax.set_title(obj);ax.set_xlabel('Re-estimation sample size S');ax.grid(alpha=.2)
    axes[0].set_ylabel('Mean OOS coverage across fixed solutions (%)');axes[0].legend(fontsize=8)
    fig.tight_layout();fig.savefig(out/'coverage_by_sample_size.png',dpi=180);plt.close(fig)


def run(args):
    sizes=sorted(args.sizes);seeds=args.seeds
    if not sizes or min(sizes)<1 or len(set(sizes))!=len(sizes):
        raise ValueError('Sample sizes must be unique positive integers')
    if not seeds or len(set(seeds))!=len(seeds) or any(s<0 for s in seeds):
        raise ValueError('Seeds must be unique nonnegative integers')
    if args.oos_size<1 or args.oos_seed<0 or args.oos_seed==920001 or set(seeds)&{920001,args.oos_seed}:
        raise ValueError('Original, re-estimation and OOS seeds must be distinct')
    if args.limit is not None and args.limit<1:
        raise ValueError('Limit must be positive')
    sources=load_ccp30_rows();audit_training_semantics(sources)
    config=json.loads(SOURCE.with_name('configuration.json').read_text())
    for key,value in base.waiting_emission_configuration().items():
        if config.get(key)!=value: raise RuntimeError(f'Source Scheme B differs: {key}')
    if args.limit: sources=sources[:args.limit]
    input_paths=[SOURCE,SOURCE.with_name('configuration.json'),ROOT/'data/data_expanded.xlsx',
        Path(base.DEFAULT_BORDER_EVENT_DATA_FILE)] + [ROOT/p for p in (
        'baseline_uncertainty.py','run_ccp_candidate_pool.py','run_ev_ccp_oos_pilot.py',
        'run_formal_ev_vs_ccp_s30_30runs.py','run_ccp30_q90_oos_validation.py','run_ccp30_sample_stability.py')]
    manifest=dict(sizes=sizes,seeds=seeds,master_size=max(sizes),oos_size=args.oos_size,
        oos_seed=args.oos_seed,limit=args.limit,solution_count=len(sources),original_seed=920001,
        original_size=30,alpha=.9,numpy_version=np.__version__,python_version=sys.version,
        input_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in input_paths},
        waiting=base.waiting_emission_configuration(),post_validation_pareto_filtering=False,
        new_optimisation=False,smoke_test=args.limit is not None or args.oos_size!=5000)
    out=args.out;out.mkdir(parents=True,exist_ok=True)
    if (out/'manifest.json').exists():
        if json.loads((out/'manifest.json').read_text())!=manifest:
            raise RuntimeError('Output configuration/input changed. Use a new --out directory.')
    elif any(out.iterdir()):
        raise RuntimeError('Nonempty output directory without manifest; use a new directory')
    else: atomic_json(out/'manifest.json',manifest)
    base.BORDER_EVENT_DEFINITIONS=base.load_border_event_definitions(base.DEFAULT_BORDER_EVENT_DATA_FILE)
    net=base.load_network_from_extended(ROOT/'data/data_expanded.xlsx')
    (_,_,hold,proc,transcost,arcs,timetables,batches,waiting,_,carbon,_,_,trans,border,theta,_)=net
    for b in batches: b.penalty_per_teu_h=base.DEFAULT_LATE_PENALTY_USD_PER_TEU_H
    tt=base.build_timetable_dict(timetables);lookup=base.build_arc_lookup(arcs)
    # Rebuild persisted paths exactly; generating a random path library is unnecessary.
    individuals=[restore_individual(s,{},tt,lookup) for s in sources]
    base.RISK_METRIC='ccp'
    base.CONFIDENCE_COST=base.CONFIDENCE_EMISSION=base.CONFIDENCE_TIME=.9
    def evaluate(size,seed,label):
        ss=base.build_scenario_set(arcs,border,size,seed,stochastic=True,
            border_event_definitions=base.BORDER_EVENT_DEFINITIONS)
        base.ACTIVE_SCENARIO_SET=ss;base._PATH_SCENARIO_CACHE={}
        arrays=np.empty((len(sources),3,size))
        for i,(ind,source) in enumerate(zip(individuals,sources)):
            if decision_signature(ind)!=source['decision_fingerprint']: raise RuntimeError('Decision changed before evaluation')
            base.evaluate_individual(ind,batches,arcs,tt,waiting,base.WAIT_EMISSION_gCO2_per_TEU_H_DEFAULT,
                node_hold_cost=hold,node_proc_cost=proc,carbon_tax_map=carbon,trans_map=trans,
                border_delay_map=border,theta_rm=theta,node_trans_cost=transcost)
            if decision_signature(ind)!=source['decision_fingerprint']: raise RuntimeError('Decision changed during evaluation')
            arrays[i]=np.array([ind.cost_s,ind.emission_s,ind.makespan_s])
            if (i+1)%25==0 or i==len(sources)-1: print(f'{label}: {i+1}/{len(sources)}',flush=True)
        if not np.isfinite(arrays).all(): raise RuntimeError('Non-finite simulator output')
        return arrays,scenario_digest(ss)
    cache=out/'oos_sorted.npz'
    if cache.exists():
        with np.load(cache,allow_pickle=False) as z: oos=z['outcomes'];od=str(z['digest'])
        if oos.shape!=(len(sources),3,args.oos_size) or not np.isfinite(oos).all() or np.any(np.diff(oos,axis=2)<0):
            raise RuntimeError('Invalid OOS cache')
    else:
        oos,od=evaluate(args.oos_size,args.oos_seed,'Common OOS');oos.sort(axis=2)
        with (out/'oos_sorted.tmp').open('wb') as f: np.savez_compressed(f,outcomes=oos,digest=od)
        (out/'oos_sorted.tmp').replace(cache)
    original,td=evaluate(30,920001,'Original training replay')
    audit=[]
    for i,s in enumerate(sources):
        for j,obj in enumerate(OBJECTIVES):
            q=base.empirical_ccp_quantile(original[i,j],.9);stored=float(s['optimisation_objectives'][obj])
            audit.append(dict(solution_id=s['source_solution_id'],objective=obj,stored=stored,replayed=q,
                matches=bool(np.isclose(q,stored,rtol=1e-10,atol=1e-7))))
    write_csv(out/'training_replay_audit.csv',audit)
    if not all(r['matches'] for r in audit): raise RuntimeError('Stored training q90 was not reproduced; see audit CSV')
    if td!=config['scenario_digest']: raise RuntimeError('Original training scenario digest differs')
    rows=records(original,oos,sources,920001,[30],'original_training')
    for seed in seeds:
        checkpoint=out/f'seed_{seed}.json'
        if checkpoint.exists():
            saved=json.loads(checkpoint.read_text());sr=saved['rows']
            if len(sr)!=len(sources)*len(sizes)*3: raise RuntimeError('Incomplete seed checkpoint')
            print(f'Resumed seed {seed}',flush=True)
        else:
            master,md=evaluate(max(sizes),seed,f'Re-estimation seed {seed}')
            sr=records(master,oos,sources,seed,sizes,'independent_reestimate')
            atomic_json(checkpoint,dict(master_digest=md,rows=sr))
        rows.extend(sr)
    write_csv(out/'per_solution_seed_size.csv',rows)
    report(out,rows,len(sources))
    atomic_json(out/'COMPLETE.json',dict(solution_count=len(sources),rows=len(rows),
        oos_digest=od,training_digest=td,training_replay_passed=True,
        decision_fingerprints_unchanged=True,smoke_test=manifest['smoke_test']))
    print(f'Completed: {out}',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sizes',type=int,nargs='+',default=[30,100,300])
    p.add_argument('--seeds',type=int,nargs='+',default=list(range(940001,940011)))
    p.add_argument('--oos-size',type=int,default=5000)
    p.add_argument('--oos-seed',type=int,default=930001)
    p.add_argument('--limit',type=int,help='Smoke test only: first N original solutions')
    p.add_argument('--out',type=Path,default=DEFAULT_OUT)
    run(p.parse_args())

if __name__=='__main__': main()
