#!/usr/bin/env python3
"""Post-hoc diagnostics only: no search, training, source writes or new selection.

Recompute saved final fronts independently; preserve the paired-mean estimand.
Best-of-seeds is a separate, equally budgeted multistart descriptive statistic.
"""
from pathlib import Path
from collections import defaultdict
from statistics import mean, median
import argparse
import csv
import hashlib
import json
import math

METHODS = ('BASE', 'ICM_HEURISTIC', 'ICM_INDEPENDENT', 'ICM_SHARED_SUPERVISED')


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def csv_write(path, rows):
    if not rows:
        return
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def points_checked(points):
    if any(len(p) != 3 or not all(math.isfinite(x) for x in p) for p in points):
        raise ValueError('Invalid raw objective points')
    return sorted(set(tuple(map(float, p)) for p in points))


def hv(points, ref=(1.1, 1.1, 1.1)):
    """Independent z-slab / x-slab union of rectangles, all objectives minimize."""
    pts = [p for p in points_checked(points) if all(x < y for x, y in zip(p, ref))]
    zs = sorted({p[2] for p in pts}) + [ref[2]]
    volume = 0.
    for left, right in zip(zs, zs[1:]):
        active = sorted((x, y) for x, y, z in pts if z <= left)
        area = 0.; best_y = ref[1]
        for i, (x, y) in enumerate(active):
            best_y = min(best_y, y)
            next_x = active[i+1][0] if i+1 < len(active) else ref[0]
            area += (next_x-x) * (ref[1]-best_y)
        volume += (right-left)*area
    return volume


def transform(points, lo, hi, div=(1., 1., 1.)):
    return [[(p[i]/div[i]-lo[i])/(hi[i]-lo[i] if hi[i] > lo[i] else 1.)
             for i in range(3)] for p in points]


def equal_config(rows, field):
    groups = defaultdict(lambda: defaultdict(list))
    for r in rows:
        groups[r['configuration']][r['instance']].append(r[field])
    if any(x is None for g in groups.values() for xs in g.values() for x in xs):
        return None
    return mean(mean(mean(xs) for xs in g.values()) for g in groups.values())


def percent(value, reference):
    return 100*(value-reference)/reference if reference > 0 else None


def comparisons(rows, field):
    by = {(r['instance'], r['seed'], r['arm']): r for r in rows}
    result = []
    for method in METHODS:
        for reference in ('BASE', 'ICM_HEURISTIC', 'ICM_INDEPENDENT'):
            pairs = []
            for r in rows:
                if r['arm'] != method:
                    continue
                b = by[r['instance'], r['seed'], reference][field]
                pairs.append(dict(instance=r['instance'], configuration=r['configuration'],
                                  delta=r[field]-b, percent=percent(r[field], b)))
            result.append(dict(method=method, reference=reference, normalization=field, pairs=len(pairs),
                               mean_paired_percent=equal_config(pairs, 'percent'),
                               mean_absolute_delta=equal_config(pairs, 'delta'),
                               wins=sum(r['delta'] > 1e-12 for r in pairs),
                               ties=sum(abs(r['delta']) <= 1e-12 for r in pairs),
                               losses=sum(r['delta'] < -1e-12 for r in pairs)))
    return result


def best_of_seeds(rows, budget):
    output = []
    for instance in sorted({r['instance'] for r in rows}):
        group = [r for r in rows if r['instance'] == instance]
        subsets = {m: [r for r in group if r['arm'] == m] for m in METHODS}
        seed_sets = [set(r['seed'] for r in rs) for rs in subsets.values()]
        if not seed_sets[0] or any(s != seed_sets[0] for s in seed_sets):
            raise ValueError('Best-of-seeds requires identical seed sets and restart counts')
        for m, rs in subsets.items():
            best = max(rs, key=lambda r: (r['union_hv'], -r['seed']))
            base_best = max(r['union_hv'] for r in subsets['BASE'])
            rule_best = max(r['union_hv'] for r in subsets['ICM_HEURISTIC'])
            output.append(dict(instance=instance, configuration=rs[0]['configuration'], method=m,
                restarts=len(rs), total_evaluation_cap=len(rs)*budget,
                actual_evaluations=sum(r['real_objective_evaluation_count'] for r in rs),
                total_seconds=sum(r['optimisation_seconds'] for r in rs),
                mean_hv=mean(r['union_hv'] for r in rs), median_hv=median(r['union_hv'] for r in rs),
                minimum_hv=min(r['union_hv'] for r in rs), best_hv=best['union_hv'], best_seed=best['seed'],
                base_best_hv=base_best, rule_best_hv=rule_best,
                best_vs_base_percent=percent(best['union_hv'], base_best),
                best_vs_rule_percent=percent(best['union_hv'], rule_best)))
    return output


def execute(root, report, out):
    root, report, out = root.resolve(), report.resolve(), out.resolve()
    if root == out or root in out.parents or report == out or report in out.parents:
        raise ValueError('Use a separate sibling diagnostic output, not the source tree')
    out.mkdir(parents=True, exist_ok=False)
    inputs = {}
    def load(path):
        data = read(path); inputs[str(path)] = digest(path); return data
    plan = load(root/'PLAN.json'); settings = plan['settings']
    if canonical({k:v for k,v in plan.items() if k != 'sha256'}) != plan['sha256']:
        raise ValueError('Source PLAN digest invalid')
    quick = load(report/'QUICK_RESULT.json')
    if not quick['completed']:
        raise ValueError('Final report incomplete')
    stored = load(report/'PAIRED_RUNS.json')
    index = {(r['instance'],r['seed'],r['arm']): r for r in stored}
    expected = {(i['instance_id'],s,m) for i in plan['screen']
                for s in settings['screen_seeds'] for m in METHODS}
    if len(index) != len(stored) or set(index) != expected:
        raise ValueError('Reported runs missing/duplicated')
    rows = []; max_error = 0.; bounds_out = {}; norm = plan['normalization']
    for entry in plan['screen']:
        name = entry['instance_id']; group = []
        instance = load(root/'instances'/(name+'.json'))
        quantities = sum(b['quantity'] for b in instance['batches'])
        horizon = max(b['LT'] for b in instance['batches'])-min(b['ET'] for b in instance['batches'])
        div = (max(1., quantities), max(1., quantities), max(1., horizon))
        for seed in settings['screen_seeds']:
            paired = []
            for method in METHODS:
                r = load(root/'screen'/name/f'seed_{seed}'/method/'RUN_COMPLETE.json')
                if (not r['completed'] or r['plan_sha256'] != plan['sha256'] or
                    (r['instance'],r['seed'],r['arm'],r['configuration']) != (name,seed,method,entry['configuration']) or
                    r['instance_digest'] != instance['instance_digest'] or r['phase'] != 'screen' or r['stochastic'] or
                    r['hidden_candidate_objective_evaluations'] != 0 or r['ppo_updates'] != 0 or
                    r['evaluation_budget'] != settings['budget'] or
                    not settings['budget']-1 <= r['real_objective_evaluation_count'] <= settings['budget']):
                    raise ValueError('Invalid run identity/budget: '+name+'/'+method)
                points_checked(r['final_front_raw']); group.append(r); paired.append(r)
                lineage_path = root/'screen'/name/f'seed_{seed}'/method/'final_archive_lineage.json'
                lineage = load(lineage_path)
                lineage_points = [[p['optimisation_objectives'][axis] for axis in ('cost','emission','makespan')]
                                  for p in lineage]
                if points_checked(lineage_points) != points_checked(r['final_front_raw']):
                    raise ValueError('Final objective order/front differs from saved lineage')
            if len({r['scenario_digest'] for r in paired}) != 1:
                raise ValueError('Scenario pairing mismatch: '+name)
        points = [p for r in group for p in r['final_front_raw']]
        lo = [min(p[i] for p in points) for i in range(3)] if points else [0.]*3
        hi = [max(p[i] for p in points) for i in range(3)] if points else [0.]*3
        bounds_out[name] = dict(minimum=lo, maximum=hi, reference=[1.1]*3)
        for r in group:
            u = hv(transform(r['final_front_raw'], lo, hi))
            normalized = transform(r['final_front_raw'], norm['minimum'], norm['maximum'], div)
            t = hv(normalized, norm['reference'])
            old = index[r['instance'],r['seed'],r['arm']]
            error = max(abs(u-old['final_hypervolume']), abs(t-r['final_hypervolume']))
            max_error = max(max_error, error)
            if error > 1e-9:
                raise ValueError(f'HV recomputation disagrees: {name}/{r["seed"]}/{r["arm"]}: {error}')
            rows.append(dict(r, union_hv=u, recomputed_training_hv=t,
                             training_reference_excluded_points=sum(any(p[i] >= norm['reference'][i] for i in range(3)) for p in normalized)))
        print('[SCREEN-AUDIT] independently checked '+name, flush=True)
    paired_stats = comparisons(rows, 'union_hv') + comparisons(rows, 'recomputed_training_hv')
    for r in paired_stats:
        if r['normalization'] != 'union_hv':
            continue
        reported = quick['methods'][r['method']]['comparisons'][r['reference']]['percent']['mean']
        actual = r['mean_paired_percent']
        if (actual is None) != (reported is None) or (actual is not None and abs(actual-reported) > 1e-8):
            raise ValueError('Reported paired mean disagrees')
    best = best_of_seeds(rows, settings['budget'])
    best_summary = [dict(method=m, statistic='post-hoc best of equal seed count within each instance',
        restarts_per_instance=len(settings['screen_seeds']), evaluation_cap_per_instance=len(settings['screen_seeds'])*settings['budget'],
        best_vs_base_percent=equal_config([r for r in best if r['method']==m], 'best_vs_base_percent'),
        best_vs_rule_percent=equal_config([r for r in best if r['method']==m], 'best_vs_rule_percent')) for m in METHODS]
    execution = []
    for m in METHODS:
        own = [r for r in rows if r['arm']==m]
        for c in 'ICM':
            counts = {k:sum(r['selection'][c][k] for r in own) for k in ('calls','fallbacks','switches','unsupported')}
            z = [r.get('input_range_checks',{}).get(c,{}) for r in own]
            values = sum(r.get('values',0) for r in z); clipped = sum(r.get('clipped',0) for r in z)
            execution.append(dict(method=m, component=c, **counts,
                switch_fraction=counts['switches']/counts['calls'] if counts['calls'] else None,
                fallback_fraction=counts['fallbacks']/counts['calls'] if counts['calls'] and m not in ('BASE','ICM_HEURISTIC') else None,
                input_values=values, input_clipped=clipped, clip_fraction=clipped/values if values else None))
    milestones = []
    for evaluation in settings['milestones']:
        sample = []
        for r in rows:
            points = [p for p in r['milestone_hv'] if p['evaluation']==evaluation]
            if len(points) != 1:
                raise ValueError('Missing/duplicate milestone')
            sample.append(dict(r, milestone_hv_value=points[0]['hv']))
        milestones.extend(dict(r, evaluation=evaluation, normalization='frozen_training_normalization')
                          for r in comparisons(sample, 'milestone_hv_value'))
    bundles = {}
    for name, folder in [('shared',Path(plan['source'])/'shared_model'),('independent',root/'independent_model')]:
        b = load(folder/'BUNDLE.json')
        for rel, expected_hash in b['checkpoint_sha256'].items():
            path = folder/rel; actual_hash = digest(path); inputs[str(path)] = actual_hash
            if actual_hash != expected_hash:
                raise ValueError('Changed model checkpoint')
        if b['normalization'] != norm:
            raise ValueError('Model/run normalization mismatch')
        screen_names = {r['instance_id'] for r in plan['screen']}
        for c in 'ICM':
            tr, va = set(b['training_instances'][c]), set(b['validation_instances'][c])
            if tr & va or (tr|va) & screen_names:
                raise ValueError('Train/validation/screen leakage')
        bundles[name] = {k:b[k] for k in ('settings','coverage','training_states','validation_states','selected','target_scales','loss','feature_version')}
    files = {'PAIRED_MEAN_CHECK.csv':paired_stats, 'BEST_OF_SEEDS_PER_INSTANCE.csv':best,
             'BEST_OF_SEEDS_SUMMARY.csv':best_summary, 'EXECUTION.csv':execution,
             'CONVERGENCE_PAIRED.csv':milestones}
    run_fields = ('instance','configuration','seed','arm','union_hv','recomputed_training_hv',
                  'training_reference_excluded_points','real_objective_evaluation_count',
                  'generations_completed','final_feasible_fraction','cache_hits','fixed_initial_slots_preserved',
                  'deferred_c_pairs','feature_construction_seconds','inference_seconds','objective_evaluation_seconds','optimisation_seconds','stop_reason')
    files['RUN_CHECK.csv'] = [{k:r.get(k) for k in run_fields} for r in rows]
    for filename, records in files.items(): csv_write(out/filename, records)
    write(out/'MODEL_SUMMARY.json', bundles); write(out/'INSTANCE_BOUNDS.json', bounds_out)
    if any(digest(Path(p)) != sha for p,sha in inputs.items()):
        raise ValueError('Source changed during read-only audit')
    write(out/'INPUT_HASHES.json', inputs)
    summary = dict(completed=True, checked_runs=len(rows), max_hv_recompute_error=max_error,
        paired_means_reproduced=True, shared_model=bundles['shared']['selected'],
        new_searches=0, new_training_steps=0, new_objective_evaluations=0,
        best_of_seeds_post_hoc=True, main_estimator_unchanged=True,
        notes=['Intervals in original report are conditional on one instance per configuration.',
               'Best-of-seeds compares every method best with every reference best, never best vs mean.',
               'Three B50000 restarts consume up to B150000; do not label this a B50000 single-run result.',
               'No observation here alone establishes overfitting or the causal cause of search degradation.'])
    write(out/'SUMMARY.json', summary)
    print('[SCREEN-AUDIT] COMPLETE', json.dumps({k:summary[k] for k in ('checked_runs','max_hv_recompute_error','new_objective_evaluations')}))
    for filename in ('PAIRED_MEAN_CHECK.csv','BEST_OF_SEEDS_SUMMARY.csv','EXECUTION.csv'):
        print('\n'+filename+'\n'+(out/filename).read_text(encoding='utf-8-sig'))
    print('\nCONVERGENCE vs BASE (frozen training normalization; percent)')
    for r in milestones:
        if r['method'] != 'BASE' and r['reference'] == 'BASE':
            print(r['method'], r['evaluation'], r['mean_paired_percent'])
    print('\nMODEL GATES AND LABEL COUNTS')
    for name,b in bundles.items():
        print(json.dumps(dict(model=name,training_states=b['training_states'],validation_states=b['validation_states'],
                              selected=b['selected']),ensure_ascii=False))
    return summary


def main():
    diag = Path(__file__).resolve().parents[1]/'transformer_ppo_runs/diagnostics'
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, default=diag/'ccp_icm_multiscale_joint_v1')
    p.add_argument('--report', type=Path, default=diag/'ccp_icm_multiscale_joint_v1_unique_resume_v2/results')
    p.add_argument('--out', type=Path, default=diag/'ccp_icm_multiscale_joint_v1_screen_audit_v1')
    a = p.parse_args(); execute(a.source,a.report,a.out)


if __name__ == '__main__': main()
