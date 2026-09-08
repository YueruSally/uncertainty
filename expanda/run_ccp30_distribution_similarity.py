#!/usr/bin/env python3
"""Compare fixed CCP30 outcome distributions with a common 5,000-scenario OOS reference.

This is a fixed-decision diagnostic.  It does not optimise new CCP solutions and it
does not apply Pareto filtering.  For each of the existing Run-1 CCP30 decisions,
the script evaluates nested samples of sizes 30, 100 and 300 under ten independent
seeds and compares each empirical objective distribution with the same OOS sample.
"""
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
import run_ccp30_sample_stability as stability


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "formal_ev_vs_ccp_s30_30runs/run_01/CCP30/final_feasible_nondominated.json"
DEFAULT_OUT = ROOT / "formal_ev_vs_ccp_s30_30runs/run1_distribution_similarity_v2"
DEFAULT_OOS_CACHE = ROOT / "formal_ev_vs_ccp_s30_30runs/run1_sample_stability/oos_sorted.npz"
OBJECTIVES = ("cost", "emission", "makespan")
PRIMARY_METRICS = ("js_distance", "wasserstein_normalized", "ks_statistic")


def atomic_json(path: Path, value) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def _validate_vector(values, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float).reshape(-1)
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError(f"{name} must be non-empty and finite")
    return array


def reference_histogram_edges(reference: np.ndarray, bins: int) -> np.ndarray:
    """Return OOS-defined edges, with overflow bins for training observations."""
    if bins < 2:
        raise ValueError("histogram bins must be at least 2")
    reference = _validate_vector(reference, "reference")
    lo = float(reference.min())
    hi = float(reference.max())
    if lo == hi:
        width = max(abs(lo), 1.0) * 1e-9
        interior = np.linspace(lo - width, hi + width, bins + 1)[1:-1]
    else:
        interior = np.linspace(lo, hi, bins + 1)[1:-1]
    return np.concatenate(([-np.inf], interior, [np.inf]))


def smoothed_histogram_probabilities(
    values: np.ndarray, edges: np.ndarray, smoothing: float
) -> np.ndarray:
    if smoothing <= 0:
        raise ValueError("smoothing must be positive")
    counts = np.histogram(values, bins=edges)[0].astype(float)
    counts += smoothing
    return counts / counts.sum()


def empirical_cdf_distances(sample: np.ndarray, reference: np.ndarray) -> tuple[float, float]:
    """Return exact one-dimensional Wasserstein-1 distance and KS statistic."""
    x = np.sort(_validate_vector(sample, "sample"))
    y = _validate_vector(reference, "reference")
    if np.any(np.diff(y) < 0):
        y = np.sort(y)
    grid = np.sort(np.concatenate((x, y)))
    cdf_x = np.searchsorted(x, grid, side="right") / x.size
    cdf_y = np.searchsorted(y, grid, side="right") / y.size
    ks = float(np.max(np.abs(cdf_x - cdf_y)))
    if grid.size < 2:
        return 0.0, ks
    deltas = np.diff(grid)
    wasserstein = float(np.sum(np.abs(cdf_x[:-1] - cdf_y[:-1]) * deltas))
    return wasserstein, ks


def ks_two_sample_pvalue(ks_statistic: float, n_sample: int, n_reference: int) -> float:
    """Asymptotic two-sided two-sample KS p-value.

    This avoids adding SciPy as a runtime dependency. The statistic remains the
    primary KS result; the p-value is supplementary because non-rejection is not
    evidence that two distributions are equivalent.
    """
    if n_sample < 1 or n_reference < 1:
        raise ValueError("KS sample sizes must be positive")
    d = float(ks_statistic)
    if not 0.0 <= d <= 1.0:
        raise ValueError("KS statistic must lie in [0, 1]")
    if d == 0.0:
        return 1.0
    effective_n = math.sqrt(n_sample * n_reference / (n_sample + n_reference))
    lam = (effective_n + 0.12 + 0.11 / effective_n) * d
    if lam < 1.18:
        terms = [
            math.exp(-((2 * k - 1) ** 2) * math.pi**2 / (8.0 * lam**2))
            for k in range(1, 1000)
        ]
        cdf = math.sqrt(2.0 * math.pi) * sum(terms) / lam
        return float(min(1.0, max(0.0, 1.0 - cdf)))
    survival = 2.0 * sum(
        (-1.0) ** (k - 1) * math.exp(-2.0 * k * k * lam * lam)
        for k in range(1, 1000)
    )
    return float(min(1.0, max(0.0, survival)))


def distribution_metrics(
    sample, reference, *, bins: int = 20, smoothing: float = 0.5
) -> dict:
    """Compute distribution distances and retain the existing q90 diagnostics.

    Histogram edges are defined only by the OOS reference and are therefore fixed
    across S and seeds for a given solution/objective.  Jeffreys smoothing keeps KL
    finite when a small training sample leaves bins empty.  JS is reported as a
    distance (square root of Jensen-Shannon divergence, natural-log scale).
    """
    sample = _validate_vector(sample, "sample")
    reference = _validate_vector(reference, "reference")
    reference_sorted = reference if not np.any(np.diff(reference) < 0) else np.sort(reference)
    edges = reference_histogram_edges(reference, bins)
    p = smoothed_histogram_probabilities(sample, edges, smoothing)
    q = smoothed_histogram_probabilities(reference, edges, smoothing)
    midpoint = 0.5 * (p + q)
    kl_sample_to_oos = float(np.sum(p * np.log(p / q)))
    kl_oos_to_sample = float(np.sum(q * np.log(q / p)))
    js_divergence = float(
        0.5 * np.sum(p * np.log(p / midpoint))
        + 0.5 * np.sum(q * np.log(q / midpoint))
    )
    js_distance = math.sqrt(max(js_divergence, 0.0))
    wasserstein, ks = empirical_cdf_distances(sample, reference)
    ks_pvalue = ks_two_sample_pvalue(ks, sample.size, reference.size)
    q25, q75 = np.quantile(reference, [0.25, 0.75])
    scale = float(q75 - q25)
    scale_basis = "oos_iqr"
    if scale <= 0:
        scale = float(np.std(reference))
        scale_basis = "oos_sd"
    if scale <= 0:
        scale = max(abs(float(np.mean(reference))), 1.0)
        scale_basis = "oos_mean_or_one"
    q90 = stability.measure(sample, reference_sorted)
    sample_mean = float(np.mean(sample))
    oos_mean = float(np.mean(reference))
    return {
        "js_distance": js_distance,
        "js_divergence": js_divergence,
        "wasserstein": wasserstein,
        "wasserstein_normalized": wasserstein / scale,
        "wasserstein_scale": scale,
        "wasserstein_scale_basis": scale_basis,
        "ks_statistic": ks,
        "ks_pvalue": ks_pvalue,
        "kl_sample_to_oos": kl_sample_to_oos,
        "kl_oos_to_sample": kl_oos_to_sample,
        "symmetric_kl": 0.5 * (kl_sample_to_oos + kl_oos_to_sample),
        "sample_mean": sample_mean,
        "oos_mean": oos_mean,
        "mean_signed_error_pct": (
            100.0 * (sample_mean - oos_mean) / oos_mean if oos_mean != 0 else None
        ),
        "mean_absolute_error_pct": (
            100.0 * abs(sample_mean - oos_mean) / abs(oos_mean) if oos_mean != 0 else None
        ),
        **q90,
    }


def build_records(
    training: np.ndarray,
    oos: np.ndarray,
    sources: list[dict],
    seed: int,
    sizes: list[int],
    bins: int,
    smoothing: float,
) -> list[dict]:
    rows = []
    for index, source in enumerate(sources):
        for size in sizes:
            for objective_index, objective in enumerate(OBJECTIVES):
                metrics = distribution_metrics(
                    training[index, objective_index, :size],
                    oos[index, objective_index],
                    bins=bins,
                    smoothing=smoothing,
                )
                rows.append(
                    {
                        "solution_id": source["source_solution_id"],
                        "decision_fingerprint": source["decision_fingerprint"],
                        "seed": seed,
                        "S": size,
                        "objective": objective,
                        "histogram_bins": bins,
                        "histogram_smoothing": smoothing,
                        **metrics,
                    }
                )
    return rows


def build_oos_baseline_records(
    comparison_oos: np.ndarray,
    reference_oos: np.ndarray,
    sources: list[dict],
    comparison_seed: int,
    reference_seed: int,
    oos_size: int,
    bins: int,
    smoothing: float,
) -> list[dict]:
    """Compare two independent OOS samples to estimate the distance noise floor."""
    rows = []
    for index, source in enumerate(sources):
        for objective_index, objective in enumerate(OBJECTIVES):
            metrics = distribution_metrics(
                comparison_oos[index, objective_index],
                reference_oos[index, objective_index],
                bins=bins,
                smoothing=smoothing,
            )
            rows.append(
                {
                    "solution_id": source["source_solution_id"],
                    "decision_fingerprint": source["decision_fingerprint"],
                    "seed": comparison_seed,
                    "S": oos_size,
                    "objective": objective,
                    "comparison": "independent_oos_baseline",
                    "reference_oos_seed": reference_seed,
                    "comparison_oos_seed": comparison_seed,
                    "histogram_bins": bins,
                    "histogram_smoothing": smoothing,
                    **metrics,
                }
            )
    return rows


def _metric_summary(values: np.ndarray, prefix: str) -> dict:
    numeric = np.array(
        [float(value) for value in values if value is not None and np.isfinite(float(value))],
        dtype=float,
    )
    if numeric.size == 0:
        return {
            f"{prefix}_mean": None,
            f"{prefix}_median": None,
            f"{prefix}_p25": None,
            f"{prefix}_p75": None,
            f"{prefix}_p90": None,
            f"{prefix}_max": None,
        }
    return {
        f"{prefix}_mean": float(np.mean(numeric)),
        f"{prefix}_median": float(np.median(numeric)),
        f"{prefix}_p25": float(np.quantile(numeric, 0.25)),
        f"{prefix}_p75": float(np.quantile(numeric, 0.75)),
        f"{prefix}_p90": float(np.quantile(numeric, 0.90)),
        f"{prefix}_max": float(np.max(numeric)),
    }


def summarize_records(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    metric_names = (
        "js_distance",
        "wasserstein_normalized",
        "ks_statistic",
        "ks_pvalue",
        "mean_signed_error_pct",
        "mean_absolute_error_pct",
        "kl_sample_to_oos",
        "kl_oos_to_sample",
        "symmetric_kl",
        "absolute_coverage_gap_pp",
        "coverage",
    )
    per_seed = []
    groups = {}
    for row in rows:
        groups.setdefault((row["seed"], row["S"], row["objective"]), []).append(row)
    for (seed, size, objective), group in sorted(groups.items()):
        summary = {
            "seed": seed,
            "S": size,
            "objective": objective,
            "solution_count": len(group),
        }
        for metric in metric_names:
            summary.update(_metric_summary(np.array([r[metric] for r in group]), metric))
        summary["ks_pvalue_gt_0_05_fraction"] = float(
            np.mean([r["ks_pvalue"] > 0.05 for r in group])
        )
        per_seed.append(summary)

    aggregate = []
    groups = {}
    for row in rows:
        groups.setdefault((row["S"], row["objective"]), []).append(row)
    for (size, objective), group in sorted(groups.items()):
        summary = {
            "S": size,
            "objective": objective,
            "seeds": len({r["seed"] for r in group}),
            "solutions": len({r["solution_id"] for r in group}),
            "observations": len(group),
        }
        for metric in metric_names:
            summary.update(_metric_summary(np.array([r[metric] for r in group]), metric))
        summary["ks_pvalue_gt_0_05_fraction"] = float(
            np.mean([r["ks_pvalue"] > 0.05 for r in group])
        )
        aggregate.append(summary)
    return per_seed, aggregate


def build_size_comparison(aggregate: list[dict]) -> list[dict]:
    lookup = {(r["S"], r["objective"]): r for r in aggregate}
    if not all((size, objective) in lookup for size in (30, 100, 300) for objective in OBJECTIVES):
        return []
    rows = []
    for objective in OBJECTIVES:
        for metric in PRIMARY_METRICS:
            values = [lookup[(size, objective)][f"{metric}_median"] for size in (30, 100, 300)]
            total_reduction = values[0] - values[2]
            captured = (values[0] - values[1]) / total_reduction if total_reduction > 0 else None
            rows.append(
                {
                    "objective": objective,
                    "metric": metric,
                    "S30_median": values[0],
                    "S100_median": values[1],
                    "S300_median": values[2],
                    "reduction_30_to_100": values[0] - values[1],
                    "reduction_100_to_300": values[1] - values[2],
                    "share_of_30_to_300_reduction_captured_by_S100": captured,
                }
            )
    return rows


def choose_representative(rows: list[dict], first_size: int) -> tuple[str, int]:
    grouped = {}
    for row in rows:
        if row["S"] == first_size:
            grouped.setdefault((row["solution_id"], row["seed"]), []).append(row["js_distance"])
    scores = [(key, float(np.mean(values))) for key, values in grouped.items()]
    median = float(np.median([score for _, score in scores]))
    return min(scores, key=lambda item: abs(item[1] - median))[0]


def plot_distance_boxplots(out: Path, rows: list[dict], sizes: list[int]) -> None:
    import matplotlib.pyplot as plt

    labels = {
        "js_distance": "Jensen-Shannon distance",
        "wasserstein_normalized": "Normalised Wasserstein distance",
        "ks_statistic": "KS statistic",
    }
    fig, axes = plt.subplots(3, 3, figsize=(14, 11))
    for row_index, objective in enumerate(OBJECTIVES):
        for column_index, metric in enumerate(PRIMARY_METRICS):
            ax = axes[row_index, column_index]
            values = [
                [r[metric] for r in rows if r["objective"] == objective and r["S"] == size]
                for size in sizes
            ]
            ax.boxplot(values, tick_labels=[str(size) for size in sizes], showfliers=False)
            ax.set_title(f"{objective.title()}: {labels[metric]}")
            ax.set_xlabel("Training sample size S")
            ax.set_ylabel(labels[metric])
            ax.grid(axis="y", alpha=0.25)
    fig.suptitle(
        "Fixed-decision objective-distribution distance to the common OOS-5000 reference",
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out / "distribution_distance_boxplots.png", dpi=180)
    plt.close(fig)


def _plot_ecdf(ax, values: np.ndarray, label: str, **kwargs) -> None:
    ordered = np.sort(values)
    probabilities = np.arange(1, ordered.size + 1) / ordered.size
    ax.step(ordered, probabilities, where="post", label=label, **kwargs)


def plot_representative_ecdfs(
    out: Path,
    rows: list[dict],
    sources: list[dict],
    arrays_by_seed: dict[int, np.ndarray],
    oos: np.ndarray,
    sizes: list[int],
) -> dict:
    import matplotlib.pyplot as plt

    solution_id, seed = choose_representative(rows, min(sizes))
    source_index = next(
        index for index, source in enumerate(sources) if source["source_solution_id"] == solution_id
    )
    training = arrays_by_seed[seed]
    colors = ["#377eb8", "#ff7f00", "#4daf4a", "#984ea3"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    for objective_index, objective in enumerate(OBJECTIVES):
        ax = axes[objective_index]
        for color, size in zip(colors, sizes):
            _plot_ecdf(
                ax,
                training[source_index, objective_index, :size],
                f"S={size}",
                color=color,
                linewidth=1.7,
            )
        _plot_ecdf(
            ax,
            oos[source_index, objective_index],
            "OOS=5,000",
            color="black",
            linewidth=2.2,
        )
        ax.set_title(objective.title())
        ax.set_xlabel("Objective outcome")
        ax.set_ylabel("Empirical cumulative probability")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    fig.suptitle(f"Representative fixed solution {solution_id}; re-estimation seed {seed}")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out / "representative_objective_ecdfs.png", dpi=180)
    plt.close(fig)
    selected = {"solution_id": solution_id, "seed": seed, "selection_basis": "median S-min composite JS distance"}
    atomic_json(out / "representative_selection.json", selected)
    return selected


def make_recommendation(
    aggregate: list[dict],
    comparison: list[dict],
    coverage_tolerance_pp: float,
    minimum_gain_captured: float,
) -> dict:
    lookup = {(r["S"], r["objective"]): r for r in aggregate}
    if not comparison:
        return {
            "candidate": "undetermined",
            "reason": "Automatic S=30/S=100 assessment requires sample sizes 30, 100 and 300.",
        }
    gaps = {
        size: {
            objective: lookup[(size, objective)]["absolute_coverage_gap_pp_mean"]
            for objective in OBJECTIVES
        }
        for size in (30, 100)
    }
    plateau_passes = sum(
        row["share_of_30_to_300_reduction_captured_by_S100"] is not None
        and row["share_of_30_to_300_reduction_captured_by_S100"] >= minimum_gain_captured
        for row in comparison
    )
    plateau_required = math.ceil(2 * len(comparison) / 3)
    s30_coverage_ok = all(value <= coverage_tolerance_pp for value in gaps[30].values())
    s100_coverage_ok = all(value <= coverage_tolerance_pp for value in gaps[100].values())
    if s30_coverage_ok:
        candidate = "S=30"
        reason = "S=30 meets the declared coverage-gap tolerance for all three objectives."
    elif s100_coverage_ok and plateau_passes >= plateau_required:
        candidate = "S=100"
        reason = (
            "S=100 meets the declared coverage-gap tolerance and captures the required share "
            "of the distribution-distance improvement from S=30 to S=300."
        )
    else:
        candidate = "Neither S=30 nor S=100 is confirmed by the declared heuristic"
        reason = "Review S=300 or revise the pre-declared practical tolerances with substantive justification."
    return {
        "candidate": candidate,
        "reason": reason,
        "coverage_tolerance_pp": coverage_tolerance_pp,
        "minimum_gain_captured": minimum_gain_captured,
        "plateau_comparisons_passed": plateau_passes,
        "plateau_comparisons_required": plateau_required,
        "mean_absolute_coverage_gap_pp": gaps,
        "scope": "fixed-decision diagnostic; requires re-optimisation before a final CCP training-size choice",
    }


def write_report(
    out: Path,
    aggregate: list[dict],
    baseline_aggregate: list[dict],
    recommendation: dict,
) -> None:
    lookup = {(r["S"], r["objective"]): r for r in aggregate}
    baseline_lookup = {r["objective"]: r for r in baseline_aggregate}
    sizes = sorted({r["S"] for r in aggregate})
    lines = [
        "# CCP30 fixed-decision distribution similarity",
        "",
        "This experiment compares nested training-sample objective distributions with one common OOS-5000 reference. It does not re-optimise decisions or filter solutions.",
        "",
        "Primary metrics are Jensen-Shannon distance, normalised Wasserstein distance and the KS statistic. KL and the asymptotic two-sided KS p-value are supplementary. A p-value above 0.05 means that this test did not detect a difference; it does not prove equivalence.",
        "",
        "| S | Objective | Median JS | Median normalised W1 | Median KS | Mean absolute mean difference (%) | KS p>0.05 | Mean absolute coverage gap (pp) |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for size in sizes:
        for objective in OBJECTIVES:
            row = lookup[(size, objective)]
            lines.append(
                f"| {size} | {objective} | {row['js_distance_median']:.6f} | "
                f"{row['wasserstein_normalized_median']:.6f} | {row['ks_statistic_median']:.6f} | "
                f"{row['mean_absolute_error_pct_mean']:.3f} | "
                f"{100.0 * row['ks_pvalue_gt_0_05_fraction']:.1f}% | "
                f"{row['absolute_coverage_gap_pp_mean']:.3f} |"
            )
    lines += [
        "",
        "## Independent OOS-5000 reference baseline",
        "",
        "The table below compares a second independent OOS sample with the common OOS reference. It estimates the non-zero distance expected from Monte Carlo sampling even when both samples use the same uncertainty model.",
        "",
        "| Objective | Median JS | Median normalised W1 | Median KS | Mean absolute mean difference (%) | KS p>0.05 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for objective in OBJECTIVES:
        row = baseline_lookup[objective]
        lines.append(
            f"| {objective} | {row['js_distance_median']:.6f} | "
            f"{row['wasserstein_normalized_median']:.6f} | {row['ks_statistic_median']:.6f} | "
            f"{row['mean_absolute_error_pct_mean']:.3f} | "
            f"{100.0 * row['ks_pvalue_gt_0_05_fraction']:.1f}% |"
        )
    lines += [
        "",
        "## Pre-declared practical assessment",
        "",
        f"Candidate: **{recommendation['candidate']}**",
        "",
        recommendation["reason"],
        "",
        "This assessment is descriptive and conditional on the existing 263 Run-1 decisions. Distribution similarity alone does not prove that a separately re-optimised CCP model has equivalent solution quality.",
    ]
    (out / "REPORT.md").write_text("\n".join(lines) + "\n")


def configure_problem(sources: list[dict]):
    config = json.loads(SOURCE.with_name("configuration.json").read_text())
    for key, value in base.waiting_emission_configuration().items():
        if config.get(key) != value:
            raise RuntimeError(f"Source Scheme B differs: {key}")
    base.BORDER_EVENT_DEFINITIONS = base.load_border_event_definitions(base.DEFAULT_BORDER_EVENT_DATA_FILE)
    network = base.load_network_from_extended(ROOT / "data/data_expanded.xlsx")
    (_, _, hold, proc, transcost, arcs, timetables, batches, waiting, _, carbon, _, _, trans, border, theta, _) = network
    for batch in batches:
        batch.penalty_per_teu_h = base.DEFAULT_LATE_PENALTY_USD_PER_TEU_H
    timetable_dict = base.build_timetable_dict(timetables)
    arc_lookup = base.build_arc_lookup(arcs)
    individuals = [restore_individual(source, {}, timetable_dict, arc_lookup) for source in sources]
    base.RISK_METRIC = "ccp"
    base.CONFIDENCE_COST = base.CONFIDENCE_EMISSION = base.CONFIDENCE_TIME = 0.9
    return config, individuals, (hold, proc, transcost, arcs, batches, waiting, carbon, trans, border, theta, timetable_dict)


def evaluate_scenarios(
    size: int,
    seed: int,
    label: str,
    sources: list[dict],
    individuals,
    problem,
) -> tuple[np.ndarray, str]:
    hold, proc, transcost, arcs, batches, waiting, carbon, trans, border, theta, timetable_dict = problem
    scenarios = base.build_scenario_set(
        arcs,
        border,
        size,
        seed,
        stochastic=True,
        border_event_definitions=base.BORDER_EVENT_DEFINITIONS,
    )
    base.ACTIVE_SCENARIO_SET = scenarios
    base._PATH_SCENARIO_CACHE = {}
    arrays = np.empty((len(sources), 3, size))
    for index, (individual, source) in enumerate(zip(individuals, sources)):
        if decision_signature(individual) != source["decision_fingerprint"]:
            raise RuntimeError("Decision changed before evaluation")
        base.evaluate_individual(
            individual,
            batches,
            arcs,
            timetable_dict,
            waiting,
            base.WAIT_EMISSION_gCO2_per_TEU_H_DEFAULT,
            node_hold_cost=hold,
            node_proc_cost=proc,
            carbon_tax_map=carbon,
            trans_map=trans,
            border_delay_map=border,
            theta_rm=theta,
            node_trans_cost=transcost,
        )
        if decision_signature(individual) != source["decision_fingerprint"]:
            raise RuntimeError("Decision changed during evaluation")
        arrays[index] = np.array([individual.cost_s, individual.emission_s, individual.makespan_s])
        if (index + 1) % 25 == 0 or index == len(sources) - 1:
            print(f"{label}: {index + 1}/{len(sources)}", flush=True)
    if not np.isfinite(arrays).all():
        raise RuntimeError("Non-finite simulator output")
    return arrays, scenario_digest(scenarios)


def load_array_cache(path: Path, expected_shape: tuple[int, ...]) -> tuple[np.ndarray, str]:
    with np.load(path, allow_pickle=False) as archive:
        arrays = archive["outcomes"]
        digest = str(archive["digest"])
    if arrays.shape != expected_shape or not np.isfinite(arrays).all():
        raise RuntimeError(f"Invalid array cache: {path}")
    return arrays, digest


def save_array_cache(path: Path, arrays: np.ndarray, digest: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as handle:
        np.savez_compressed(handle, outcomes=arrays, digest=digest)
    tmp.replace(path)


def verify_cached_scenario_digest(
    cached_digest: str, size: int, seed: int, problem, cache_path: Path
) -> None:
    arcs = problem[3]
    border = problem[8]
    scenarios = base.build_scenario_set(
        arcs,
        border,
        size,
        seed,
        stochastic=True,
        border_event_definitions=base.BORDER_EVENT_DEFINITIONS,
    )
    expected = scenario_digest(scenarios)
    if cached_digest != expected:
        raise RuntimeError(f"Scenario digest mismatch for cache: {cache_path}")


def run(args) -> None:
    sizes = sorted(args.sizes)
    seeds = list(args.seeds)
    if not sizes or min(sizes) < 2 or len(sizes) != len(set(sizes)):
        raise ValueError("Sample sizes must be unique integers of at least 2")
    if not seeds or len(seeds) != len(set(seeds)) or any(seed < 0 for seed in seeds):
        raise ValueError("Seeds must be unique nonnegative integers")
    if args.oos_size < 2 or args.oos_seed < 0 or args.oos_seed in seeds:
        raise ValueError("OOS seed must be distinct from the re-estimation seeds")
    if (
        args.oos_baseline_seed < 0
        or args.oos_baseline_seed == args.oos_seed
        or args.oos_baseline_seed in seeds
    ):
        raise ValueError("Second OOS seed must be distinct from the first OOS and training seeds")
    if args.limit is not None and args.limit < 1:
        raise ValueError("Limit must be positive")
    if args.coverage_tolerance_pp <= 0 or not 0 < args.minimum_gain_captured <= 1:
        raise ValueError("Invalid practical assessment threshold")

    sources = stability.load_ccp30_rows()
    stability.audit_training_semantics(sources)
    if args.limit:
        sources = sources[: args.limit]
    config, individuals, problem = configure_problem(sources)
    args.out.mkdir(parents=True, exist_ok=True)
    input_paths = [
        SOURCE,
        SOURCE.with_name("configuration.json"),
        ROOT / "data/data_expanded.xlsx",
        Path(base.DEFAULT_BORDER_EVENT_DATA_FILE),
        ROOT / "baseline_uncertainty.py",
        ROOT / "run_ccp30_sample_stability.py",
        Path(__file__).resolve(),
    ]
    manifest = {
        "purpose": "fixed-decision objective-distribution similarity",
        "sizes": sizes,
        "seeds": seeds,
        "master_size": max(sizes),
        "oos_size": args.oos_size,
        "oos_seed": args.oos_seed,
        "oos_baseline_seed": args.oos_baseline_seed,
        "solution_count": len(sources),
        "histogram_bins": args.histogram_bins,
        "histogram_smoothing": args.histogram_smoothing,
        "coverage_tolerance_pp": args.coverage_tolerance_pp,
        "minimum_gain_captured": args.minimum_gain_captured,
        "limit": args.limit,
        "numpy_version": np.__version__,
        "python_version": sys.version,
        "input_sha256": {
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in input_paths
        },
        "waiting": base.waiting_emission_configuration(),
        "post_validation_pareto_filtering": False,
        "new_optimisation": False,
        "common_oos_reference": True,
        "independent_oos_reference_baseline": True,
        "ks_pvalue_method": "asymptotic_two_sided",
        "nested_training_prefixes": True,
        "smoke_test": args.limit is not None or args.oos_size != 5000,
    }
    manifest_path = args.out / "manifest.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != manifest:
            raise RuntimeError("Output configuration or input changed; use a new --out directory")
    elif any(args.out.iterdir()):
        raise RuntimeError("Non-empty output directory without manifest; use a new directory")
    else:
        atomic_json(manifest_path, manifest)

    standard_oos = args.oos_cache or DEFAULT_OOS_CACHE
    expected_oos_shape = (len(sources), 3, args.oos_size)
    if standard_oos.exists() and args.limit is None and args.oos_size == 5000:
        oos, oos_digest_value = load_array_cache(standard_oos, expected_oos_shape)
        verify_cached_scenario_digest(
            oos_digest_value, args.oos_size, args.oos_seed, problem, standard_oos
        )
        if np.any(np.diff(oos, axis=2) < 0):
            raise RuntimeError(f"Existing OOS cache is not sorted: {standard_oos}")
        print(f"Reused common OOS cache: {standard_oos}", flush=True)
    else:
        local_oos = args.out / "cache/oos_sorted.npz"
        if local_oos.exists():
            oos, oos_digest_value = load_array_cache(local_oos, expected_oos_shape)
            verify_cached_scenario_digest(
                oos_digest_value, args.oos_size, args.oos_seed, problem, local_oos
            )
        else:
            oos, oos_digest_value = evaluate_scenarios(
                args.oos_size, args.oos_seed, "Common OOS", sources, individuals, problem
            )
            oos.sort(axis=2)
            save_array_cache(local_oos, oos, oos_digest_value)

    baseline_oos_cache = args.oos_baseline_cache
    baseline_expected_shape = (len(sources), 3, args.oos_size)
    if baseline_oos_cache is not None and baseline_oos_cache.exists():
        oos_baseline, oos_baseline_digest = load_array_cache(
            baseline_oos_cache, baseline_expected_shape
        )
        verify_cached_scenario_digest(
            oos_baseline_digest,
            args.oos_size,
            args.oos_baseline_seed,
            problem,
            baseline_oos_cache,
        )
        if np.any(np.diff(oos_baseline, axis=2) < 0):
            raise RuntimeError(f"Second OOS cache is not sorted: {baseline_oos_cache}")
        print(f"Reused second OOS cache: {baseline_oos_cache}", flush=True)
    else:
        local_oos_baseline = args.out / "cache/oos_baseline_sorted.npz"
        if local_oos_baseline.exists():
            oos_baseline, oos_baseline_digest = load_array_cache(
                local_oos_baseline, baseline_expected_shape
            )
            verify_cached_scenario_digest(
                oos_baseline_digest,
                args.oos_size,
                args.oos_baseline_seed,
                problem,
                local_oos_baseline,
            )
        else:
            oos_baseline, oos_baseline_digest = evaluate_scenarios(
                args.oos_size,
                args.oos_baseline_seed,
                "Independent OOS baseline",
                sources,
                individuals,
                problem,
            )
            oos_baseline.sort(axis=2)
            save_array_cache(local_oos_baseline, oos_baseline, oos_baseline_digest)

    baseline_rows = build_oos_baseline_records(
        oos_baseline,
        oos,
        sources,
        args.oos_baseline_seed,
        args.oos_seed,
        args.oos_size,
        args.histogram_bins,
        args.histogram_smoothing,
    )
    write_csv(args.out / "oos_reference_baseline_per_solution.csv", baseline_rows)
    _, baseline_aggregate = summarize_records(baseline_rows)
    write_csv(args.out / "oos_reference_baseline_summary.csv", baseline_aggregate)

    all_rows = []
    arrays_by_seed = {}
    cache_dir = args.training_cache_dir or (args.out / "cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    for seed in seeds:
        cache_path = cache_dir / f"training_seed_{seed}_S{max(sizes)}.npz"
        expected_shape = (len(sources), 3, max(sizes))
        if cache_path.exists():
            training, training_digest = load_array_cache(cache_path, expected_shape)
            verify_cached_scenario_digest(
                training_digest, max(sizes), seed, problem, cache_path
            )
            print(f"Reused training cache for seed {seed}", flush=True)
        else:
            training, training_digest = evaluate_scenarios(
                max(sizes), seed, f"Training seed {seed}", sources, individuals, problem
            )
            save_array_cache(cache_path, training, training_digest)
        arrays_by_seed[seed] = training
        seed_rows = build_records(
            training,
            oos,
            sources,
            seed,
            sizes,
            args.histogram_bins,
            args.histogram_smoothing,
        )
        all_rows.extend(seed_rows)
        print(f"Distribution metrics completed for seed {seed}", flush=True)

    write_csv(args.out / "per_solution_seed_size_distribution.csv", all_rows)
    per_seed, aggregate = summarize_records(all_rows)
    write_csv(args.out / "per_seed_distribution_summary.csv", per_seed)
    write_csv(args.out / "by_sample_size_distribution.csv", aggregate)
    comparison = build_size_comparison(aggregate)
    if comparison:
        write_csv(args.out / "sample_size_comparison.csv", comparison)
    recommendation = make_recommendation(
        aggregate,
        comparison,
        args.coverage_tolerance_pp,
        args.minimum_gain_captured,
    )
    atomic_json(args.out / "sample_size_recommendation.json", recommendation)
    plot_distance_boxplots(args.out, all_rows, sizes)
    representative = plot_representative_ecdfs(
        args.out, all_rows, sources, arrays_by_seed, oos, sizes
    )
    write_report(args.out, aggregate, baseline_aggregate, recommendation)
    atomic_json(
        args.out / "COMPLETE.json",
        {
            "solution_count": len(sources),
            "seeds": len(seeds),
            "sizes": sizes,
            "objectives": list(OBJECTIVES),
            "records": len(all_rows),
            "oos_digest": oos_digest_value,
            "oos_baseline_digest": oos_baseline_digest,
            "oos_seed": args.oos_seed,
            "oos_baseline_seed": args.oos_baseline_seed,
            "representative": representative,
            "candidate": recommendation["candidate"],
            "decision_fingerprints_unchanged": True,
            "new_optimisation": False,
            "smoke_test": manifest["smoke_test"],
        },
    )
    print(f"Completed: {args.out}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int, nargs="+", default=[30, 100, 300])
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(940001, 940011)))
    parser.add_argument("--oos-size", type=int, default=5000)
    parser.add_argument("--oos-seed", type=int, default=930001)
    parser.add_argument("--oos-cache", type=Path, help="Optional existing sorted OOS NPZ")
    parser.add_argument("--oos-baseline-seed", type=int, default=930002)
    parser.add_argument(
        "--oos-baseline-cache", type=Path, help="Optional existing sorted second OOS NPZ"
    )
    parser.add_argument(
        "--training-cache-dir",
        type=Path,
        help="Optional directory containing existing training_seed_<seed>_S<max>.npz caches",
    )
    parser.add_argument("--histogram-bins", type=int, default=20)
    parser.add_argument("--histogram-smoothing", type=float, default=0.5)
    parser.add_argument("--coverage-tolerance-pp", type=float, default=2.5)
    parser.add_argument("--minimum-gain-captured", type=float, default=0.60)
    parser.add_argument("--limit", type=int, help="Smoke test: use the first N fixed solutions")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
