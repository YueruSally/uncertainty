"""Quantile accuracy diagnostics: objectives ranked separately, never pooled."""
import math
import json
import numpy as np
from common import OBJECTIVES, CONFIGS, csv_write, read, write


def ranks(values):
    _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    return (np.cumsum(counts)-.5*counts+.5)[inverse]


def spearman(x, y):
    x, y = ranks(x), ranks(y)
    x -= x.mean(); y -= y.mean()
    denominator = np.linalg.norm(x)*np.linalg.norm(y)
    return float(np.dot(x, y)/denominator) if denominator > 0 else None


def reference_quantile_interval(samples, probability=.9):
    """Distribution-free order-statistic interval; ties make it conservative."""
    n = samples.shape[-1]
    k = np.arange(n+1)
    logs = np.asarray([math.lgamma(n+1)-math.lgamma(int(i)+1)-math.lgamma(n-int(i)+1)
                      for i in k]) + k*math.log(probability)+(n-k)*math.log(1-probability)
    weights = np.exp(logs-logs.max()); cdf = np.cumsum(weights/weights.sum())
    lower = int(np.searchsorted(cdf, .025))
    upper = int(np.searchsorted(cdf, .975))+1
    if lower < 1 or upper > n:
        raise ValueError('Finite order statistic interval unavailable')
    sorted_values = np.sort(samples, axis=-1)
    return sorted_values[..., lower-1], sorted_values[..., upper-1], (lower, upper)


def ks_permutation(x, y, seed, permutations=999):
    """Two-sided KS statistic with a permutation p-value, retaining ties."""
    n, m = len(x), len(y)
    z = np.concatenate([x, y]); order = np.argsort(z, kind='stable')
    sorted_z = z[order]
    ends = np.r_[np.flatnonzero(np.diff(sorted_z)), n+m-1]
    labels = np.r_[np.ones(n), np.zeros(m)][order]
    def statistic(lab):
        a = np.cumsum(lab)[ends]
        return float(np.max(np.abs(a/n - (ends+1-a)/m)))
    observed = statistic(labels)
    rng = np.random.default_rng(seed)
    exceeds = sum(statistic(rng.permutation(labels)) >= observed-1e-15 for _ in range(permutations))
    return observed, (exceeds+1)/(permutations+1)


def number(value):
    return None if value is None or not np.isfinite(value) else float(value)


def stats(errors):
    finite = errors[np.isfinite(errors)]
    if not finite.size:
        return dict(mean=None, p95=None, maximum=None)
    return dict(mean=float(finite.mean()), p95=float(np.quantile(finite, .95)), maximum=float(finite.max()))


def configuration_report(folder, inputs, metas):
    cfg = inputs['configuration']
    ids = [d['exact_decision_id'] for d in inputs['decisions']]
    with np.load(folder/'packs/reference_1000.npz') as f:
        reference = f['quantiles'].copy(); reference_samples = f['samples'].copy()
    ci_low, ci_high, ranks_ci = reference_quantile_interval(reference_samples)
    with np.load(folder/'packs/original_100.npz') as f:
        original = f['quantiles'].copy()
    trials = []
    for i in range(1, 21):
        with np.load(folder/'packs'/f'replicate_{i:02d}_100.npz') as f:
            trials.append(f['quantiles'].copy())
    trials = np.stack(trials)
    denominator = np.where(np.abs(reference) > 1e-15, np.abs(reference), np.nan)
    errors = 100*np.abs(trials-reference[None])/denominator[None]
    original_errors = 100*np.abs(original-reference)/denominator
    signed = 100*(trials-reference[None])/denominator[None]
    summary, repeated, plan_rows = [], [], []
    for j, obj in enumerate(OBJECTIVES):
        rho = [spearman(q[:, j], reference[:, j]) for q in trials]
        valid_rho = [r for r in rho if r is not None]
        a, b = stats(errors[:, :, j]), stats(original_errors[:, j])
        sd = np.std(trials[:, :, j], axis=0, ddof=1)
        summary.append(dict(configuration=cfg, instance=inputs['instance']['instance_id'], objective=obj,
            unique_plans=len(ids), reference_size=1000, replicate_size=100, replicates=20,
            original100_mean_abs_error_percent=b['mean'], original100_max_abs_error_percent=b['maximum'],
            fresh100_mean_abs_error_percent=a['mean'], fresh100_p95_abs_error_percent=a['p95'],
            fresh100_max_abs_error_percent=a['maximum'],
            fresh100_mean_signed_error_percent=stats(signed[:, :, j])['mean'],
            original100_spearman=spearman(original[:, j], reference[:, j]),
            fresh100_mean_spearman=number(np.mean(valid_rho)) if valid_rho else None,
            fresh100_min_spearman=min(valid_rho) if valid_rho else None,
            defined_spearman_replicates=len(valid_rho), zero_reference_plans=int(np.isnan(denominator[:, j]).sum()),
            mean_q90_sd=float(sd.mean()), mean_q90_sd_percent=stats(100*sd/denominator[:, j])['mean'],
            reference95_mean_relative_width_percent=stats(100*(ci_high[:, j]-ci_low[:, j])/denominator[:, j])['mean']))
        for i, q in enumerate(trials):
            e = stats(errors[i, :, j])
            repeated.append(dict(configuration=cfg, objective=obj, replicate=i+1,
                scenario_seed=inputs['new_seeds'][i+1], plans=len(ids), mean_abs_error_percent=e['mean'],
                p95_abs_error_percent=e['p95'], max_abs_error_percent=e['maximum'], spearman=rho[i]))
        for p, key in enumerate(ids):
            plan_rows.append(dict(configuration=cfg, exact_decision_id=key, objective=obj,
                original100_q90=float(original[p, j]), reference1000_q90=float(reference[p, j]),
                reference95_low=float(ci_low[p, j]), reference95_high=float(ci_high[p, j]),
                fresh100_mean_q90=float(trials[:, p, j].mean()), fresh100_sd_q90=float(sd[p]),
                fresh100_min_q90=float(trials[:, p, j].min()), fresh100_max_q90=float(trials[:, p, j].max()),
                fresh100_mean_abs_error_percent=stats(errors[:, p, j])['mean'],
                **{f'replicate_{i+1:02d}_q90':float(trials[i,p,j]) for i in range(20)}))
    csv_write(folder/'SUMMARY.csv', summary)
    csv_write(folder/'REPLICATE_METRICS.csv', repeated)
    csv_write(folder/'PER_PLAN_QUANTILES.csv', plan_rows)
    csv_write(folder/'SOURCE_MEMBERSHIP.csv', inputs['member_rows'])
    csv_write(folder/'FEASIBILITY.csv', [dict(pack=m['identity']['tag'], plans=m['plans'],
        feasible=m['feasible_plans'], infeasible=m['plans']-m['feasible_plans']) for m in metas])
    # The first exact decision hash is fixed before new realizations are drawn.
    with np.load(folder/'packs/replicate_01_100.npz') as f:
        small = f['samples'][0].copy()
    big = reference_samples[0]
    distributions = []
    for j, obj in enumerate(OBJECTIVES):
        d, p = ks_permutation(small[j], big[j], 290000000+100*CONFIGS.index(cfg)+j)
        distributions.append(dict(configuration=cfg, representative_decision_id=ids[0], objective=obj,
            sample100_seed=inputs['new_seeds'][1], reference1000_seed=inputs['new_seeds'][0],
            ks_statistic=d, permutation_p=p, permutations=999,
            qualification='Auxiliary, not proof of equivalent q90; representative preselected by hash'))
    csv_write(folder/'REPRESENTATIVE_KS.csv', distributions)
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(12, 6), constrained_layout=True)
    for j, obj in enumerate(OBJECTIVES):
        bins = np.histogram_bin_edges(np.r_[small[j], big[j]], bins=25)
        axes[0, j].hist(big[j], bins=bins, density=True, alpha=.45, label='Independent 1,000')
        axes[0, j].hist(small[j], bins=bins, density=True, histtype='step', linewidth=1.5, label='Fresh 100: set 1')
        axes[0, j].set(title=obj.capitalize(), ylabel='Density')
        for x, label in [(big[j], 'Independent 1,000'), (small[j], 'Fresh 100: set 1')]:
            axes[1, j].step(np.sort(x), np.arange(1, len(x)+1)/len(x), where='post', label=label)
        axes[1, j].set(xlabel=obj.capitalize()+' (original units)', ylabel='Empirical CDF')
    axes[0, 0].legend(fontsize=8); fig.suptitle(cfg+' | fixed representative transport plan')
    fig.savefig(folder/'REPRESENTATIVE_DISTRIBUTIONS.png', dpi=180)
    fig.savefig(folder/'REPRESENTATIVE_DISTRIBUTIONS.pdf'); plt.close(fig)
    result = dict(completed=True, configuration=cfg, instance=inputs['instance']['instance_id'],
        inputs_sha256=inputs['sha256'], unique_plans=len(ids), source_runs=6,
        original_reproduction=metas[0], summary=summary,
        original_reproduction_evaluations=metas[0]['plans'],
        fresh_fixed_plan_evaluations=sum(m['fixed_plan_evaluations'] for m in metas[1:]),
        fresh_plan_realizations=sum(m['plan_realizations'] for m in metas[1:]),
        original_plan_realizations=100*metas[0]['plans'],
        evaluator_seconds=sum(m['seconds'] for m in metas),
        reference95_order_statistic_ranks=list(ranks_ci), new_searches=0, training_updates=0,
        note='All fixed plans retained, including new infeasible plans; no reoptimization, model selection, or automatic sample-size change.')
    ledger = [json.loads(line) for line in (folder/'EVALUATION_LEDGER.jsonl').read_text().splitlines() if line.strip()]
    result['physical_fixed_plan_evaluations_including_retries'] = len(ledger)
    result['physical_plan_realizations_including_retries'] = sum(r['realizations'] for r in ledger)
    write(folder/'RESULT.json', result)
    return result


def aggregate(root, results):
    rows = [row for result in results for row in result['summary']]
    csv_write(root/'SUMMARY.csv', rows)
    all_rows = []
    for obj in OBJECTIVES:
        group = [r for r in rows if r['objective'] == obj]
        row = dict(objective=obj, completed_configurations=len(group))
        for key in ('fresh100_mean_abs_error_percent', 'fresh100_mean_spearman', 'mean_q90_sd_percent'):
            vals = [r[key] for r in group if r[key] is not None]
            row['configuration_equal_'+key] = float(np.mean(vals)) if vals else None
        maxima = [r['fresh100_max_abs_error_percent'] for r in group if r['fresh100_max_abs_error_percent'] is not None]
        row['worst_abs_error_percent'] = max(maxima) if maxima else None
        all_rows.append(row)
    csv_write(root/'CONFIGURATION_EQUAL_SUMMARY.csv', all_rows)
    lines = ['# Fixed-decision q90 precision audit', '',
        'No optimization or training. Existing searches/models/results are unchanged.',
        'One predetermined instance per configuration; Base and single Shared SL final B5000 fronts, three seeds each.',
        'Plans are merged by exact decision identity. New random realizations are shared across plans within each instance.',
        'Independent 1,000-realization estimates are finite-sample references, not exact population quantiles.',
        'Fresh-set errors pool 20 sets and all unique plans within each instance; configuration summary weights configurations equally.',
        'Original 100 includes selection bias from optimization and is reported separately. Spearman is objective-wise; it does not establish Pareto ranking stability.',
        'Order-statistic reference intervals concern individual q90 values, not simultaneous intervals over all plans.',
        'KS permutation results and distribution figures are auxiliary. A nonsignificant KS result does not prove equivalence.',
        'This is a fixed-solution evaluation check, not a rerun at a larger search budget.', '',
        '| Config | Objective | Plans | Fresh100 mean error % | Worst error % | Mean Spearman |',
        '|---|---|---:|---:|---:|---:|']
    def fmt(x): return 'undefined' if x is None else f'{x:.5g}'
    for r in rows:
        lines.append(f"| {r['configuration']} | {r['objective']} | {r['unique_plans']} | {fmt(r['fresh100_mean_abs_error_percent'])} | {fmt(r['fresh100_max_abs_error_percent'])} | {fmt(r['fresh100_mean_spearman'])} |")
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    return dict(completed_configurations=[r['configuration'] for r in results],
        unique_plans=sum(r['unique_plans'] for r in results),
        original_reproduction_evaluations=sum(r['original_reproduction_evaluations'] for r in results),
        fresh_fixed_plan_evaluations=sum(r['fresh_fixed_plan_evaluations'] for r in results),
        fresh_plan_realizations=sum(r['fresh_plan_realizations'] for r in results),
        original_plan_realizations=sum(r['original_plan_realizations'] for r in results),
        physical_fixed_plan_evaluations_including_retries=sum(r['physical_fixed_plan_evaluations_including_retries'] for r in results),
        physical_plan_realizations_including_retries=sum(r['physical_plan_realizations_including_retries'] for r in results),
        evaluator_seconds=sum(r['evaluator_seconds'] for r in results))
