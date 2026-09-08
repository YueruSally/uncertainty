#!/usr/bin/env python3
"""Paired CCP100/CCP300 re-optimisation pilot with common OOS-5000 validation.

The pilot performs three paired repetitions by default. Within each pair, CCP100
uses the exact first 100 scenarios of the CCP300 master sample and both methods
share the same NSGA-II seed, path library, operator settings and evaluation
budget. All original training Pareto decisions are retained. OOS nondominated
fronts are constructed only inside HV/IGD+ metric calculations; no candidate is
removed after validation.
"""
from __future__ import annotations

import argparse
import csv
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import tempfile
import time
from typing import Any, Sequence

import numpy as np

import baseline_uncertainty as base
from run_ccp_candidate_pool import decision_signature, nondominated_indices
from run_ev_ccp_oos_pilot import (
    candidate_rows,
    json_candidate,
    scenario_digest,
    scenario_prefix,
    summarise,
)
from run_formal_ev_vs_ccp_s30_30runs import restore_individual


ROOT = Path(__file__).resolve().parent
DEFAULT_OUT = ROOT / "ccp_s100_s300_reoptimization_pilot_3runs"
METHODS = ("CCP100", "CCP300")
OBJECTIVES = ("cost", "emission", "makespan")
VIEWS = ("mean", "q90")
DEFAULT_PAIRS = tuple(
    {"replicate": i + 1, "algorithm_seed": 980101 + i, "training_master_seed": 980201 + i}
    for i in range(5)
)
DEFAULT_VALIDATION_SEED = 980999
HV_REFERENCE = (1.2, 1.2, 1.2)


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def exact_hypervolume_3d(points, reference=HV_REFERENCE) -> float:
    """Exact union volume of minimisation boxes [point, reference] in 3-D."""
    ref = np.asarray(reference, dtype=float)
    pts = np.asarray(points, dtype=float)
    if pts.size == 0:
        return 0.0
    pts = pts.reshape((-1, 3))
    pts = pts[np.all(pts <= ref, axis=1)]
    if not len(pts):
        return 0.0
    xs = sorted(set(float(x) for x in pts[:, 0] if x < ref[0])) + [float(ref[0])]
    volume = 0.0
    for left, right in zip(xs[:-1], xs[1:]):
        active = pts[pts[:, 0] <= left + 1e-15]
        ys = sorted(set(float(y) for y in active[:, 1] if y < ref[1])) + [float(ref[1])]
        area = 0.0
        for low, high in zip(ys[:-1], ys[1:]):
            z = min(
                (float(point[2]) for point in active if point[1] <= low + 1e-15),
                default=float(ref[2]),
            )
            area += (high - low) * max(0.0, float(ref[2]) - z)
        volume += (right - left) * area
    return float(volume)


def coverage_diagnostic(values, training_q90: float, alpha: float) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError("OOS objective array must be non-empty and finite")
    coverage = float(np.mean(array <= training_q90))
    signed_gap_pp = 100.0 * (coverage - alpha)
    oos_q90 = base.empirical_ccp_quantile(array, alpha)
    q90_signed_error_pct = (
        100.0 * (training_q90 - oos_q90) / oos_q90 if abs(oos_q90) > 1e-15 else None
    )
    return {
        "coverage": coverage,
        "exceedance_rate": 1.0 - coverage,
        "coverage_signed_gap_pp": signed_gap_pp,
        "coverage_absolute_gap_pp": abs(signed_gap_pp),
        "oos_q90": float(oos_q90),
        "training_q90": float(training_q90),
        "q90_signed_error_pct": q90_signed_error_pct,
        "q90_absolute_error_pct": (
            abs(q90_signed_error_pct) if q90_signed_error_pct is not None else None
        ),
    }


def numeric_summary(values) -> dict[str, float | None]:
    array = np.asarray(
        [float(value) for value in values if value is not None and np.isfinite(float(value))],
        dtype=float,
    )
    if array.size == 0:
        return {"mean": None, "sd": None, "median": None, "minimum": None, "maximum": None}
    return {
        "mean": float(np.mean(array)),
        "sd": float(np.std(array, ddof=1)) if array.size > 1 else 0.0,
        "median": float(np.median(array)),
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
    }


def _objective_point(row: dict, view: str) -> tuple[float, float, float]:
    return tuple(float(row[objective][view]) for objective in OBJECTIVES)


def validated_quality(all_summaries: list[dict]) -> tuple[dict, dict]:
    """Compute all run/method HV and IGD+ using one global scale per OOS view."""
    quality: dict[tuple[int, str, str], dict] = {}
    scaling: dict[str, dict] = {}
    for view in VIEWS:
        all_points = [_objective_point(row, view) for row in all_summaries]
        pooled_indices = nondominated_indices(all_points)[0]
        pooled = np.asarray([all_points[index] for index in pooled_indices], dtype=float)
        ideal = np.min(pooled, axis=0)
        nadir = np.max(pooled, axis=0)
        span = np.where(nadir - ideal > 1e-12, nadir - ideal, 1.0)
        pooled_normalised = ((pooled - ideal) / span).tolist()
        scaling[view] = {
            "ideal": ideal.tolist(),
            "nadir": nadir.tolist(),
            "hypervolume_reference": list(HV_REFERENCE),
            "global_oos_reference_front_size": len(pooled),
        }
        for replicate in sorted({int(row["replicate"]) for row in all_summaries}):
            for method in METHODS:
                group = [
                    row for row in all_summaries
                    if int(row["replicate"]) == replicate and row["method"] == method
                ]
                points = [_objective_point(row, view) for row in group]
                front_indices = nondominated_indices(points)[0]
                front = np.asarray([points[index] for index in front_indices], dtype=float)
                normalised = ((front - ideal) / span).tolist()
                quality[(replicate, method, view)] = {
                    "hypervolume": exact_hypervolume_3d(normalised),
                    "igd_plus": float(base.igd_plus(pooled_normalised, normalised)),
                    "metric_front_size": len(front_indices),
                    "original_candidate_count": len(group),
                }
    return quality, scaling


def expected_method_metadata(
    replicate: int,
    method: str,
    seed_row: dict,
    sample_size: int,
    args,
) -> dict:
    return {
        "replicate": replicate,
        "method": method,
        "sample_size": sample_size,
        "algorithm_seed": seed_row["algorithm_seed"],
        "training_master_seed": seed_row["training_master_seed"],
        "population": args.pop,
        "generations": args.gens,
        "alpha": args.alpha,
        "formulation": "separate marginal empirical q90 objectives",
    }


def valid_method_completion(method_dir: Path, expected: dict) -> bool:
    marker = method_dir / "COMPLETE.json"
    candidates = method_dir / "final_feasible_nondominated.json"
    configuration = method_dir / "configuration.json"
    if not all(path.is_file() for path in (marker, candidates, configuration)):
        return False
    try:
        marker_data = json.loads(marker.read_text())
        config_data = json.loads(configuration.read_text())
        json.loads(candidates.read_text())
    except Exception:
        return False
    return (
        all(marker_data.get(key) == value and config_data.get(key) == value
            for key, value in expected.items())
        and marker_data.get("candidate_sha256") == file_digest(candidates)
        and marker_data.get("configuration_sha256") == file_digest(configuration)
        and marker_data.get("completed") is True
    )


def load_completed_candidates(
    method_dir: Path,
    path_lib,
    timetable_dict,
    arc_lookup,
) -> list[dict]:
    payload = json.loads((method_dir / "final_feasible_nondominated.json").read_text())
    rows = []
    for row in payload:
        restored = restore_individual(row, path_lib, timetable_dict, arc_lookup)
        if decision_signature(restored) != row["decision_fingerprint"]:
            raise RuntimeError("Restored decision fingerprint mismatch")
        rows.append({**row, "individual": restored})
    return rows


def load_jsonl_recovering_final_partial_line(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    valid_lines = []
    lines = path.read_text(encoding="utf-8").splitlines()
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            if index != len(lines) - 1:
                raise RuntimeError(f"Invalid non-final JSONL record in {path}")
            break
        rows.append(row)
        valid_lines.append(json.dumps(row, separators=(",", ":"), allow_nan=False))
    if len(valid_lines) != len([line for line in lines if line.strip()]):
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text("\n".join(valid_lines) + ("\n" if valid_lines else ""), encoding="utf-8")
        tmp.replace(path)
    return rows


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def validation_key(row: dict) -> tuple[int, str, str]:
    return (int(row["replicate"]), row["method"], row["source_solution_id"])


def build_run_rows(
    summaries: list[dict],
    quality: dict,
    runtimes: dict,
    replicate_count: int,
    alpha: float,
) -> list[dict]:
    rows = []
    for replicate in range(1, replicate_count + 1):
        for method in METHODS:
            group = [
                row for row in summaries
                if int(row["replicate"]) == replicate and row["method"] == method
            ]
            if not group:
                raise RuntimeError(f"No OOS summaries for replicate={replicate}, method={method}")
            row = {
                "replicate": replicate,
                "method": method,
                "sample_size": 100 if method == "CCP100" else 300,
                "optimisation_runtime_seconds": runtimes[
                    f"replicate_{replicate:02d}_{method}_optimisation_seconds"
                ],
                "candidate_count": len(group),
            }
            for view in VIEWS:
                q = quality[(replicate, method, view)]
                row[f"oos_{view}_hypervolume"] = q["hypervolume"]
                row[f"oos_{view}_igd_plus"] = q["igd_plus"]
                row[f"oos_{view}_metric_front_size"] = q["metric_front_size"]
            all_abs_gaps = []
            for objective in OBJECTIVES:
                coverages = [item[objective]["coverage"] for item in group]
                gaps = [item[objective]["coverage_absolute_gap_pp"] for item in group]
                errors = [item[objective]["q90_absolute_error_pct"] for item in group]
                row[f"{objective}_mean_coverage"] = float(np.mean(coverages))
                row[f"{objective}_median_coverage"] = float(np.median(coverages))
                row[f"{objective}_mean_absolute_coverage_gap_pp"] = float(np.mean(gaps))
                row[f"{objective}_median_q90_absolute_error_pct"] = float(np.median(errors))
                row[f"{objective}_solutions_at_or_above_nominal_fraction"] = float(
                    np.mean(np.asarray(coverages) >= alpha)
                )
                all_abs_gaps.extend(gaps)
            row["all_objectives_mean_absolute_coverage_gap_pp"] = float(np.mean(all_abs_gaps))
            rows.append(row)
    return rows


def build_paired_rows(run_rows: list[dict], replicate_count: int) -> list[dict]:
    paired = []
    metric_specs = {
        "optimisation_runtime_seconds": "S300_minus_S100",
        "oos_mean_hypervolume": "S300_minus_S100",
        "oos_mean_igd_plus": "S300_minus_S100",
        "oos_q90_hypervolume": "S300_minus_S100",
        "oos_q90_igd_plus": "S300_minus_S100",
        "all_objectives_mean_absolute_coverage_gap_pp": "S300_minus_S100",
    }
    for replicate in range(1, replicate_count + 1):
        left = next(
            row for row in run_rows if row["replicate"] == replicate and row["method"] == "CCP100"
        )
        right = next(
            row for row in run_rows if row["replicate"] == replicate and row["method"] == "CCP300"
        )
        row = {
            "replicate": replicate,
            "algorithm_seed": DEFAULT_PAIRS[replicate - 1]["algorithm_seed"],
            "training_master_seed": DEFAULT_PAIRS[replicate - 1]["training_master_seed"],
        }
        for metric, suffix in metric_specs.items():
            row[f"CCP100_{metric}"] = left[metric]
            row[f"CCP300_{metric}"] = right[metric]
            row[f"{metric}_{suffix}"] = right[metric] - left[metric]
        for objective in OBJECTIVES:
            metric = f"{objective}_mean_coverage"
            row[f"CCP100_{metric}"] = left[metric]
            row[f"CCP300_{metric}"] = right[metric]
            row[f"{metric}_S300_minus_S100"] = right[metric] - left[metric]
        paired.append(row)
    return paired


def aggregate_results(run_rows: list[dict], paired_rows: list[dict]) -> dict:
    metrics = (
        "optimisation_runtime_seconds",
        "candidate_count",
        "oos_mean_hypervolume",
        "oos_mean_igd_plus",
        "oos_q90_hypervolume",
        "oos_q90_igd_plus",
        "all_objectives_mean_absolute_coverage_gap_pp",
    )
    for objective in OBJECTIVES:
        metrics += (
            f"{objective}_mean_coverage",
            f"{objective}_mean_absolute_coverage_gap_pp",
            f"{objective}_median_q90_absolute_error_pct",
        )
    by_method = {}
    for method in METHODS:
        group = [row for row in run_rows if row["method"] == method]
        by_method[method] = {
            metric: numeric_summary([row[metric] for row in group]) for metric in metrics
        }

    win_specs = {
        "oos_mean_hypervolume": "higher",
        "oos_mean_igd_plus": "lower",
        "oos_q90_hypervolume": "higher",
        "oos_q90_igd_plus": "lower",
        "all_objectives_mean_absolute_coverage_gap_pp": "lower",
        "optimisation_runtime_seconds": "lower",
    }
    wins = {}
    for metric, direction in win_specs.items():
        s100 = s300 = ties = 0
        differences = []
        for pair in paired_rows:
            left = pair[f"CCP100_{metric}"]
            right = pair[f"CCP300_{metric}"]
            differences.append(right - left)
            if math.isclose(left, right, rel_tol=1e-12, abs_tol=1e-12):
                ties += 1
            elif (left > right) == (direction == "higher"):
                s100 += 1
            else:
                s300 += 1
        wins[metric] = {
            "preferred_direction": direction,
            "CCP100_wins": s100,
            "CCP300_wins": s300,
            "ties": ties,
            "S300_minus_S100": numeric_summary(differences),
        }
    return {
        "scope": "three-pair small-budget re-optimisation pilot; descriptive, not inferential",
        "by_method": by_method,
        "paired_wins": wins,
    }


def plot_paired_metrics(out: Path, run_rows: list[dict]) -> None:
    import matplotlib.pyplot as plt

    specs = [
        ("optimisation_runtime_seconds", "Optimisation runtime (s)", None),
        ("oos_mean_hypervolume", "OOS-mean HV (higher better)", None),
        ("oos_q90_hypervolume", "OOS-q90 HV (higher better)", None),
        ("oos_mean_igd_plus", "OOS-mean IGD+ (lower better)", None),
        ("oos_q90_igd_plus", "OOS-q90 IGD+ (lower better)", None),
        (
            "all_objectives_mean_absolute_coverage_gap_pp",
            "Mean absolute coverage gap (pp)",
            None,
        ),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8.5))
    for ax, (metric, title, _) in zip(axes.flat, specs):
        for replicate in sorted({row["replicate"] for row in run_rows}):
            group = {
                row["method"]: row for row in run_rows if row["replicate"] == replicate
            }
            ax.plot(
                [100, 300],
                [group["CCP100"][metric], group["CCP300"][metric]],
                marker="o",
                linewidth=1.5,
                label=f"replicate {replicate}",
            )
        ax.set_xticks([100, 300])
        ax.set_xlabel("Training sample size S")
        ax.set_title(title)
        ax.grid(alpha=0.25)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Paired CCP100 vs CCP300 re-optimisation pilot")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out / "paired_quality_runtime.png", dpi=180)
    plt.close(fig)


def plot_coverage(out: Path, run_rows: list[dict], alpha: float) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.3))
    for ax, objective in zip(axes, OBJECTIVES):
        metric = f"{objective}_mean_coverage"
        for replicate in sorted({row["replicate"] for row in run_rows}):
            group = {
                row["method"]: row for row in run_rows if row["replicate"] == replicate
            }
            ax.plot(
                [100, 300],
                [100.0 * group["CCP100"][metric], 100.0 * group["CCP300"][metric]],
                marker="o",
                linewidth=1.5,
                label=f"replicate {replicate}",
            )
        ax.axhline(100.0 * alpha, color="black", linestyle="--", linewidth=1)
        ax.set_xticks([100, 300])
        ax.set_xlabel("Training sample size S")
        ax.set_ylabel("Mean OOS coverage (%)")
        ax.set_title(objective.title())
        ax.grid(alpha=0.25)
    axes[0].legend(fontsize=8)
    fig.suptitle("Coverage of training q90 thresholds under one common OOS-5000 sample")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out / "paired_coverage.png", dpi=180)
    plt.close(fig)


def write_report(out: Path, run_rows: list[dict], aggregate: dict, args) -> None:
    lines = [
        "# CCP100 versus CCP300 re-optimisation pilot",
        "",
        f"This descriptive pilot uses {args.replicates} paired repetitions, population "
        f"{args.pop}, {args.gens} generations and one fresh common OOS-{args.validation_scenarios} "
        f"sample. It is not a formal inferential experiment.",
        "",
        "All original training Pareto candidates are retained. OOS nondominated fronts are used only for HV/IGD+ geometry and do not filter the saved candidate set.",
        "",
        "| Replicate | Method | Runtime (s) | Candidates | Mean HV | Mean IGD+ | q90 HV | q90 IGD+ | Cost cov. | Emission cov. | Makespan cov. | Mean abs. gap (pp) |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in run_rows:
        lines.append(
            f"| {row['replicate']} | {row['method']} | "
            f"{row['optimisation_runtime_seconds']:.2f} | {row['candidate_count']} | "
            f"{row['oos_mean_hypervolume']:.6f} | {row['oos_mean_igd_plus']:.6f} | "
            f"{row['oos_q90_hypervolume']:.6f} | {row['oos_q90_igd_plus']:.6f} | "
            f"{100.0 * row['cost_mean_coverage']:.2f}% | "
            f"{100.0 * row['emission_mean_coverage']:.2f}% | "
            f"{100.0 * row['makespan_mean_coverage']:.2f}% | "
            f"{row['all_objectives_mean_absolute_coverage_gap_pp']:.3f} |"
        )
    lines += [
        "",
        "## Descriptive paired win counts",
        "",
        "| Metric | Preferred | CCP100 wins | CCP300 wins | Ties |",
        "|---|---|---:|---:|---:|",
    ]
    for metric, result in aggregate["paired_wins"].items():
        lines.append(
            f"| {metric} | {result['preferred_direction']} | "
            f"{result['CCP100_wins']} | {result['CCP300_wins']} | {result['ties']} |"
        )
    lines += [
        "",
        "With only three paired repetitions, win counts and differences are descriptive. Do not report significance tests or claim that either sample size is definitively superior.",
    ]
    (out / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/data_expanded.xlsx")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--replicates", type=int, default=3, choices=range(1, 6))
    parser.add_argument("--pop", type=int, default=100)
    parser.add_argument("--gens", type=int, default=100)
    parser.add_argument("--alpha", type=float, default=0.90)
    parser.add_argument("--validation-scenarios", type=int, default=5000)
    parser.add_argument("--validation-seed", type=int, default=DEFAULT_VALIDATION_SEED)
    parser.add_argument("--expected-batches", type=int, default=40)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.pop < 2 or args.gens < 1 or args.validation_scenarios < 2:
        raise ValueError("pop>=2, gens>=1 and validation-scenarios>=2 required")
    if not 0.0 < args.alpha <= 1.0:
        raise ValueError("alpha must lie in (0, 1]")
    seeds = list(DEFAULT_PAIRS[: args.replicates])
    used_seeds = [
        value
        for row in seeds
        for value in (row["algorithm_seed"], row["training_master_seed"])
    ] + [args.validation_seed]
    if len(used_seeds) != len(set(used_seeds)):
        raise ValueError("All algorithm, training and validation seeds must be distinct")

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics").mkdir(exist_ok=True)
    (out / "validation").mkdir(exist_ok=True)

    base.BORDER_EVENT_DEFINITIONS = base.load_border_event_definitions(
        base.DEFAULT_BORDER_EVENT_DATA_FILE
    )
    network = base.load_network_from_extended(args.data)
    (
        node_names,
        node_region,
        node_hold_cost,
        node_proc_cost,
        node_trans_cost,
        arcs,
        timetables,
        batches,
        waiting_cost,
        wait_emission,
        carbon_tax,
        emission_factors,
        mode_speeds,
        trans_map,
        border_delay_map,
        theta_rm,
        _mode_speed_source,
    ) = network
    if len(batches) != args.expected_batches:
        raise ValueError(f"Expected {args.expected_batches} batches, got {len(batches)}")
    for batch in batches:
        batch.penalty_per_teu_h = base.DEFAULT_LATE_PENALTY_USD_PER_TEU_H
    timetable_dict = base.build_timetable_dict(timetables)
    arc_lookup = base.build_arc_lookup(arcs)
    random.seed(0)
    np.random.seed(0)
    path_lib = base.build_path_library(
        node_names, node_region, arcs, batches, timetable_dict, arc_lookup
    )
    base.sanity_check_path_lib(batches, path_lib)
    ev_set = base.build_expected_value_scenario_set(
        arcs,
        border_delay_map,
        seed=0,
        border_event_definitions=base.BORDER_EVENT_DEFINITIONS,
    )
    shared_reliable = base.build_reliable_path_options(
        batches,
        path_lib,
        timetable_dict,
        trans_map,
        border_delay_map,
        ev_set,
        mode="ev",
    )

    masters = {}
    samples100 = {}
    training_digests = []
    for seed_row in seeds:
        replicate = seed_row["replicate"]
        master = base.build_scenario_set(
            arcs,
            border_delay_map,
            300,
            seed_row["training_master_seed"],
            stochastic=True,
            border_event_definitions=base.BORDER_EVENT_DEFINITIONS,
        )
        sample100 = scenario_prefix(master, 100)
        prefix_ok = (
            all(
                np.array_equal(sample100.travel_multiplier[key], master.travel_multiplier[key][:100])
                for key in sample100.travel_multiplier
            )
            and all(
                np.array_equal(sample100.border_delay_h[key], master.border_delay_h[key][:100])
                for key in sample100.border_delay_h
            )
        )
        if not prefix_ok:
            raise RuntimeError(f"Replicate {replicate}: S100 is not exact S300 prefix")
        masters[replicate] = master
        samples100[replicate] = sample100
        training_digests.append(
            {
                "replicate": replicate,
                "S100": scenario_digest(sample100),
                "S300": scenario_digest(master),
                "S100_exact_S300_prefix": True,
            }
        )

    validation = base.build_scenario_set(
        arcs,
        border_delay_map,
        args.validation_scenarios,
        args.validation_seed,
        stochastic=True,
        border_event_definitions=base.BORDER_EVENT_DEFINITIONS,
    )
    validation_digest = scenario_digest(validation)
    if validation_digest in {
        entry[size] for entry in training_digests for size in ("S100", "S300")
    }:
        raise RuntimeError("Validation scenario digest matches a training digest")

    config = {
        "label": "paired CCP100/CCP300 small-budget re-optimisation pilot",
        "scope": "sample-size selection pilot; not formal inferential experiment",
        "population": args.pop,
        "generations": args.gens,
        "alpha": args.alpha,
        "replicates": args.replicates,
        "training_sizes": [100, 300],
        "validation_size": args.validation_scenarios,
        "validation_seed": args.validation_seed,
        "validation_digest": validation_digest,
        "replicate_seeds": seeds,
        "training_scenario_digests": training_digests,
        "nested_training_prefixes": True,
        "paired_algorithm_seeds": True,
        "path_library_seed": 0,
        "common_oos_for_all_candidates": True,
        "new_optimisation": True,
        "post_oos_candidate_filtering": False,
        "scenario_level_results_written": False,
        "metric_fronts_used_only_for_hv_igd_geometry": True,
        "operators": {
            "crossover_rate": base.CROSSOVER_RATE,
            "mutation_rate": base.MUTATION_RATE,
        },
        "waiting": base.waiting_emission_configuration(),
        "input_sha256": {
            str(path): file_digest(path)
            for path in (
                Path(args.data),
                Path(base.DEFAULT_BORDER_EVENT_DATA_FILE),
                ROOT / "baseline_uncertainty.py",
                Path(__file__).resolve(),
            )
        },
    }
    config_path = out / "configuration.json"
    if config_path.exists():
        if json.loads(config_path.read_text()) != config:
            raise RuntimeError("Output configuration differs; use a new --out directory")
    else:
        if any(path.name != "metrics" and path.name != "validation" for path in out.iterdir()):
            raise RuntimeError("Non-empty output without matching configuration; use a new --out")
        atomic_json(config_path, config)

    eval_kwargs = {
        "node_hold_cost": node_hold_cost,
        "node_proc_cost": node_proc_cost,
        "carbon_tax_map": carbon_tax,
        "trans_map": trans_map,
        "border_delay_map": border_delay_map,
        "theta_rm": theta_rm,
        "node_trans_cost": node_trans_cost,
    }

    all_candidates = []
    runtimes = {}
    for seed_row in seeds:
        replicate = seed_row["replicate"]
        scenarios_by_method = {
            "CCP100": samples100[replicate],
            "CCP300": masters[replicate],
        }
        for method in METHODS:
            sample_size = 100 if method == "CCP100" else 300
            method_dir = out / f"replicate_{replicate:02d}" / method.lower()
            method_dir.mkdir(parents=True, exist_ok=True)
            expected = expected_method_metadata(
                replicate, method, seed_row, sample_size, args
            )
            if valid_method_completion(method_dir, expected):
                rows = load_completed_candidates(
                    method_dir, path_lib, timetable_dict, arc_lookup
                )
                marker = json.loads((method_dir / "COMPLETE.json").read_text())
                runtimes[
                    f"replicate_{replicate:02d}_{method}_optimisation_seconds"
                ] = marker["runtime_seconds"]
                all_candidates.extend(rows)
                print(
                    f"[RESUME] replicate={replicate} method={method} "
                    f"candidates={len(rows)}",
                    flush=True,
                )
                continue

            method_config_path = method_dir / "configuration.json"
            if method_config_path.exists():
                existing = json.loads(method_config_path.read_text())
                if any(existing.get(key) != value for key, value in expected.items()):
                    raise RuntimeError(f"Incompatible incomplete output: {method_dir}")
            atomic_json(
                method_config_path,
                {
                    **expected,
                    "training_scenario_digest": scenario_digest(
                        scenarios_by_method[method]
                    ),
                    "seeds": {
                        "algorithm": seed_row["algorithm_seed"],
                        "training_master": seed_row["training_master_seed"],
                        "path_library": 0,
                    },
                },
            )
            base.ACTIVE_SCENARIO_SET = scenarios_by_method[method]
            base._PATH_SCENARIO_CACHE = {}
            base.RISK_METRIC = "ccp"
            base.CONFIDENCE_COST = args.alpha
            base.CONFIDENCE_EMISSION = args.alpha
            base.CONFIDENCE_TIME = args.alpha
            random.seed(seed_row["algorithm_seed"])
            np.random.seed(seed_row["algorithm_seed"])
            print(
                f"[OPTIMISE] replicate={replicate} method={method} "
                f"S={sample_size} pop={args.pop} gens={args.gens}",
                flush=True,
            )
            started = time.perf_counter()
            population = base.run_nsga2(
                node_names,
                node_region,
                node_hold_cost,
                node_proc_cost,
                node_trans_cost,
                arcs,
                timetables,
                batches,
                waiting_cost,
                wait_emission,
                carbon_tax,
                emission_factors,
                mode_speeds,
                trans_map,
                border_delay_map,
                theta_rm,
                path_lib,
                shared_reliable,
                pop_size=args.pop,
                generations=args.gens,
            )[0]
            runtime = time.perf_counter() - started
            fronts = base.fast_non_dominated_sort(population)
            front = [
                deepcopy(population[index])
                for index in fronts[0]
                if population[index].feasible
            ]
            if not front:
                raise RuntimeError(
                    f"No feasible first-front candidate for replicate={replicate}, method={method}"
                )
            row_config = {
                "population": args.pop,
                "generations": args.gens,
                "alpha": args.alpha,
                "seeds": {
                    "algorithm": seed_row["algorithm_seed"],
                    "training_master": seed_row["training_master_seed"],
                    "path_library": 0,
                },
            }
            rows = candidate_rows(
                method,
                f"reopt-pilot-r{replicate:02d}-{method}",
                front,
                row_config,
            )
            for row in rows:
                row["replicate"] = replicate
                row["training_sample_size"] = sample_size
            candidate_path = method_dir / "final_feasible_nondominated.json"
            atomic_json(candidate_path, [json_candidate(row) for row in rows])
            marker = {
                **expected,
                "candidate_count": len(rows),
                "runtime_seconds": runtime,
                "candidate_sha256": file_digest(candidate_path),
                "configuration_sha256": file_digest(method_config_path),
                "completed": True,
            }
            atomic_json(method_dir / "COMPLETE.json", marker)
            runtimes[
                f"replicate_{replicate:02d}_{method}_optimisation_seconds"
            ] = runtime
            all_candidates.extend(rows)

    if not all_candidates:
        raise RuntimeError("No optimisation candidates were produced")
    atomic_json(
        out / "candidate_provenance.json",
        [json_candidate(row) for row in all_candidates],
    )

    validation_config = {
        "size": args.validation_scenarios,
        "seed": args.validation_seed,
        "digest": validation_digest,
        "candidate_count": len(all_candidates),
        "one_common_scenario_set": True,
        "post_oos_candidate_filtering": False,
        "scenario_level_results_written": False,
    }
    validation_config_path = out / "validation" / "configuration.json"
    if validation_config_path.exists():
        if json.loads(validation_config_path.read_text()) != validation_config:
            raise RuntimeError("Validation configuration differs from existing output")
    else:
        atomic_json(validation_config_path, validation_config)

    checkpoint_path = out / "validation" / "per_candidate_summary.jsonl"
    summaries = load_jsonl_recovering_final_partial_line(checkpoint_path)
    completed_keys = {validation_key(row) for row in summaries}
    candidate_keys = {validation_key(row) for row in all_candidates}
    if not completed_keys <= candidate_keys:
        raise RuntimeError("Validation checkpoint contains unknown candidate keys")

    base.ACTIVE_SCENARIO_SET = validation
    base._PATH_SCENARIO_CACHE = {}
    base.RISK_METRIC = "ccp"
    validation_started = time.perf_counter()
    for index, row in enumerate(all_candidates, start=1):
        key = validation_key(row)
        if key in completed_keys:
            continue
        fixed = deepcopy(row["individual"])
        before = row["decision_fingerprint"]
        if decision_signature(fixed) != before:
            raise RuntimeError("Candidate fingerprint mismatch before OOS evaluation")
        base.evaluate_individual(
            fixed,
            batches,
            arcs,
            timetable_dict,
            waiting_cost,
            wait_emission,
            **eval_kwargs,
        )
        after = decision_signature(fixed)
        if after != before:
            raise RuntimeError("OOS evaluation changed a fixed decision")
        summary = {
            "replicate": int(row["replicate"]),
            "method": row["method"],
            "training_sample_size": int(row["training_sample_size"]),
            "source_solution_id": row["source_solution_id"],
            "decision_fingerprint": before,
            "training_objectives": row["optimisation_objectives"],
        }
        for objective, values in (
            ("cost", fixed.cost_s),
            ("emission", fixed.emission_s),
            ("makespan", fixed.makespan_s),
        ):
            distribution = summarise(values)
            coverage = coverage_diagnostic(
                values,
                float(row["optimisation_objectives"][objective]),
                args.alpha,
            )
            summary[objective] = {**distribution, **coverage}
        summary["decision_fingerprint_unchanged"] = True
        append_jsonl(checkpoint_path, summary)
        summaries.append(summary)
        completed_keys.add(key)
        print(
            f"[OOS] {len(completed_keys)}/{len(all_candidates)} "
            f"replicate={row['replicate']} method={row['method']}",
            flush=True,
        )
    validation_runtime_this_invocation = time.perf_counter() - validation_started
    runtimes["validation_seconds_this_invocation"] = validation_runtime_this_invocation
    runtimes["validation_candidate_count"] = len(summaries)
    if completed_keys != candidate_keys:
        raise RuntimeError("OOS validation did not cover every candidate")
    if not all(row["decision_fingerprint_unchanged"] for row in summaries):
        raise RuntimeError("At least one OOS decision fingerprint changed")

    summaries.sort(key=validation_key)
    atomic_json(out / "validation" / "per_candidate_summary.json", summaries)
    flat_candidate_rows = []
    for row in summaries:
        flat = {
            "replicate": row["replicate"],
            "method": row["method"],
            "training_sample_size": row["training_sample_size"],
            "source_solution_id": row["source_solution_id"],
            "decision_fingerprint": row["decision_fingerprint"],
        }
        for objective in OBJECTIVES:
            for field in (
                "mean",
                "q90",
                "coverage",
                "exceedance_rate",
                "coverage_signed_gap_pp",
                "coverage_absolute_gap_pp",
                "training_q90",
                "q90_signed_error_pct",
                "q90_absolute_error_pct",
            ):
                flat[f"{objective}_{field}"] = row[objective][field]
        flat_candidate_rows.append(flat)
    write_csv(out / "validation" / "per_candidate_summary.csv", flat_candidate_rows)

    quality, scaling = validated_quality(summaries)
    atomic_json(out / "metrics" / "normalisation.json", scaling)
    run_rows = build_run_rows(
        summaries, quality, runtimes, args.replicates, args.alpha
    )
    paired_rows = build_paired_rows(run_rows, args.replicates)
    aggregate = aggregate_results(run_rows, paired_rows)
    write_csv(out / "metrics" / "per_run_summary.csv", run_rows)
    write_csv(out / "metrics" / "paired_comparison.csv", paired_rows)
    atomic_json(out / "metrics" / "aggregate_summary.json", aggregate)
    atomic_json(out / "runtime_summary.json", runtimes)
    plot_paired_metrics(out, run_rows)
    plot_coverage(out, run_rows, args.alpha)
    write_report(out, run_rows, aggregate, args)

    complete = {
        "replicates": args.replicates,
        "methods": list(METHODS),
        "optimisations": args.replicates * len(METHODS),
        "population": args.pop,
        "generations": args.gens,
        "validation_scenarios": args.validation_scenarios,
        "validation_seed": args.validation_seed,
        "validation_digest": validation_digest,
        "candidate_count": len(summaries),
        "all_original_training_pareto_candidates_retained": True,
        "post_oos_candidate_filtering": False,
        "all_decision_fingerprints_unchanged": True,
        "scenario_level_results_written": False,
        "pilot_not_formal_inference": True,
        "completed": True,
    }
    atomic_json(out / "COMPLETE.json", complete)
    print(json.dumps(complete, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
