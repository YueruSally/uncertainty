#!/usr/bin/env python3
"""Five-pair full-size CCP30/CCP50 scenario-size selection study."""

from __future__ import annotations

import argparse
import csv
from copy import deepcopy
import gzip
import json
from pathlib import Path
import random
import time
from typing import Any, Sequence

import numpy as np

import baseline_uncertainty as base
from run_ccp_candidate_pool import decision_signature, nondominated_indices
from run_ev_ccp_oos_pilot import (
    candidate_rows, json_candidate, scenario_digest, scenario_prefix, summarise,
)

DEFAULT_PAIRS = tuple((871001 + i, 872001 + i) for i in range(5))
DEFAULT_VALIDATION_SEED = 879999
HV_REFERENCE = (1.2, 1.2, 1.2)


def exact_hypervolume_3d(points, reference=HV_REFERENCE):
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
            z = min((float(p[2]) for p in active if p[1] <= low + 1e-15),
                    default=float(ref[2]))
            area += (high - low) * max(0.0, float(ref[2]) - z)
        volume += (right - left) * area
    return float(volume)


def percent_errors(summaries, objective):
    values = []
    for item in summaries:
        training = float(item["training_objectives"][objective])
        validation = float(item[objective]["q90"])
        values.append(abs(100.0 * (validation - training) / training)
                      if abs(training) > 1e-15 else float("nan"))
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return {
        "median_absolute_percentage_error": float(np.median(arr)),
        "mean_absolute_percentage_error": float(np.mean(arr)),
        "maximum_absolute_percentage_error": float(np.max(arr)),
    }


def metric_delta(validation, training):
    absolute = float(validation - training)
    return {"absolute": absolute,
            "percentage": (100.0 * absolute / training if abs(training) > 1e-15 else None)}


def validated_quality(method_summaries):
    all_summaries = method_summaries["CCP30"] + method_summaries["CCP50"]
    all_points = [(x["cost"]["q90"], x["emission"]["q90"], x["makespan"]["q90"])
                  for x in all_summaries]
    pooled_indices = nondominated_indices(all_points)[0]
    pooled = np.asarray([all_points[i] for i in pooled_indices], dtype=float)
    ideal, nadir = np.min(pooled, axis=0), np.max(pooled, axis=0)
    span = np.where(nadir - ideal > 1e-12, nadir - ideal, 1.0)
    pooled_norm = ((pooled - ideal) / span).tolist()
    result = {}
    for method, rows in method_summaries.items():
        pts = [(x["cost"]["q90"], x["emission"]["q90"], x["makespan"]["q90"])
               for x in rows]
        front_indices = nondominated_indices(pts)[0]
        front = np.asarray([pts[i] for i in front_indices], dtype=float)
        norm = ((front - ideal) / span).tolist()
        result[method] = {
            "hypervolume": exact_hypervolume_3d(norm),
            "igd_plus": base.igd_plus(pooled_norm, norm),
            "validated_nondominated_size": len(front_indices),
        }
    return result, {"ideal": ideal.tolist(), "nadir": nadir.tolist(),
                    "hypervolume_reference": list(HV_REFERENCE),
                    "pooled_reference_size": len(pooled)}


def parse_args(argv: Sequence[str] | None = None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default="data/data_expanded.xlsx")
    p.add_argument("--out", default="ccp_s30_s50_selection_5runs")
    p.add_argument("--pop", type=int, default=300)
    p.add_argument("--gens", type=int, default=500)
    p.add_argument("--alpha", type=float, default=.90)
    p.add_argument("--validation-scenarios", type=int, default=5000)
    p.add_argument("--expected-batches", type=int, default=40)
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.pop < 2 or args.gens < 1 or args.validation_scenarios < 1:
        raise ValueError("pop>=2, gens>=1 and validation-scenarios>=1 required")
    seeds = [{"replicate": i + 1, "algorithm": a, "training_S50_master": t}
             for i, (a, t) in enumerate(DEFAULT_PAIRS)]
    used = [x for pair in DEFAULT_PAIRS for x in pair] + [DEFAULT_VALIDATION_SEED]
    if len(used) != len(set(used)):
        raise ValueError("all new algorithm, training, and validation seeds must differ")
    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {out}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "validation").mkdir()

    config: dict[str, Any] = {
        "label": "five-pair CCP scenario-size selection study; not formal experiment",
        "population": args.pop, "generations": args.gens, "alpha": args.alpha,
        "training_sizes": [30, 50], "validation_size": args.validation_scenarios,
        "replicate_seeds": seeds, "selection_validation_seed": DEFAULT_VALIDATION_SEED,
        "path_library_seed": 0, "candidate_pool_limit": None,
        "selection_validation_reuse_prohibition":
            "must not be presented as final independent validation for formal study",
        "quality_metric_definition": json.loads(
            (Path(__file__).with_name("CCP_S30_S50_SELECTION_PLAN.json")).read_text())[
                "quality_metrics"],
    }

    base.BORDER_EVENT_DEFINITIONS = base.load_border_event_definitions(
        base.DEFAULT_BORDER_EVENT_DATA_FILE)
    network = base.load_network_from_extended(args.data)
    (node_names, node_region, node_hold_cost, node_proc_cost, node_trans_cost,
     arcs, timetables, batches, waiting_cost, wait_emission, carbon_tax,
     emission_factors, mode_speeds, trans_map, border_delay_map, theta_rm,
     _mode_speed_source) = network
    if len(batches) != args.expected_batches:
        raise ValueError(f"expected {args.expected_batches} batches, got {len(batches)}")
    for batch in batches:
        batch.penalty_per_teu_h = base.DEFAULT_LATE_PENALTY_USD_PER_TEU_H
    tt_dict, arc_lookup = base.build_timetable_dict(timetables), base.build_arc_lookup(arcs)
    random.seed(0); np.random.seed(0)
    path_lib = base.build_path_library(node_names, node_region, arcs, batches,
                                       tt_dict, arc_lookup)
    base.sanity_check_path_lib(batches, path_lib)
    ev_set = base.build_expected_value_scenario_set(
        arcs, border_delay_map, seed=0,
        border_event_definitions=base.BORDER_EVENT_DEFINITIONS)
    shared_reliable = base.build_reliable_path_options(
        batches, path_lib, tt_dict, trans_map, border_delay_map, ev_set, mode="ev")
    validation = base.build_scenario_set(
        arcs, border_delay_map, args.validation_scenarios, DEFAULT_VALIDATION_SEED,
        stochastic=True, border_event_definitions=base.BORDER_EVENT_DEFINITIONS)
    config["validation_scenario_digest"] = scenario_digest(validation)
    config["waiting_emission_g_per_teu_h"] = float(wait_emission)
    config["preflight"] = {"paired_algorithm_seeds": True,
                           "validation_seed_distinct_from_all_training": True,
                           "shared_path_library_and_option_ordering": True}
    (out / "configuration.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (out / "predeclared_seed_manifest.json").write_text(
        json.dumps({"replicates": seeds, "selection_validation": DEFAULT_VALIDATION_SEED,
                    "path_library": 0}, indent=2), encoding="utf-8")

    eval_kwargs = dict(node_hold_cost=node_hold_cost, node_proc_cost=node_proc_cost,
                       carbon_tax_map=carbon_tax, trans_map=trans_map,
                       border_delay_map=border_delay_map, theta_rm=theta_rm,
                       node_trans_cost=node_trans_cost)
    all_candidates, runtimes = [], {}
    prefix_checks, training_digests = [], []
    for seed_row in seeds:
        replicate = seed_row["replicate"]
        rep_dir = out / f"replicate_{replicate:02d}"
        rep_dir.mkdir()
        master = base.build_scenario_set(
            arcs, border_delay_map, 50, seed_row["training_S50_master"], stochastic=True,
            border_event_definitions=base.BORDER_EVENT_DEFINITIONS)
        sample30 = scenario_prefix(master, 30)
        prefix_ok = (all(np.array_equal(sample30.travel_multiplier[k], master.travel_multiplier[k][:30])
                         for k in sample30.travel_multiplier) and
                     all(np.array_equal(sample30.border_delay_h[k], master.border_delay_h[k][:30])
                         for k in sample30.border_delay_h))
        if not prefix_ok:
            raise RuntimeError(f"replicate {replicate}: S30 is not exact S50 prefix")
        prefix_checks.append(prefix_ok)
        training_digests.append({"replicate": replicate, "S30": scenario_digest(sample30),
                                 "S50": scenario_digest(master)})
        for method, scenarios in (("CCP30", sample30), ("CCP50", master)):
            method_dir = rep_dir / method.lower(); method_dir.mkdir()
            base.ACTIVE_SCENARIO_SET = scenarios
            base._PATH_SCENARIO_CACHE = {}
            base.RISK_METRIC = "ccp"
            base.CONFIDENCE_COST = base.CONFIDENCE_EMISSION = base.CONFIDENCE_TIME = args.alpha
            random.seed(seed_row["algorithm"]); np.random.seed(seed_row["algorithm"])
            started = time.perf_counter()
            population = base.run_nsga2(
                node_names, node_region, node_hold_cost, node_proc_cost,
                node_trans_cost, arcs, timetables, batches, waiting_cost,
                wait_emission, carbon_tax, emission_factors, mode_speeds,
                trans_map, border_delay_map, theta_rm, path_lib, shared_reliable,
                pop_size=args.pop, generations=args.gens)[0]
            runtime = time.perf_counter() - started
            key = f"replicate_{replicate:02d}_{method}"
            runtimes[key] = runtime
            fronts = base.fast_non_dominated_sort(population)
            front = [deepcopy(population[i]) for i in fronts[0] if population[i].feasible]
            row_config = {"population": args.pop, "generations": args.gens,
                          "alpha": args.alpha,
                          "seeds": {"algorithm": seed_row["algorithm"],
                                    "training_S50_master": seed_row["training_S50_master"],
                                    "path_library": 0}}
            rows = candidate_rows(method, f"selection-r{replicate:02d}-{method}",
                                  front, row_config)
            for row in rows: row["replicate"] = replicate
            all_candidates.extend(rows)
            (method_dir / "final_feasible_nondominated.json").write_text(
                json.dumps([json_candidate(x) for x in rows], indent=2), encoding="utf-8")
    config["preflight"]["all_S30_exact_prefix_checks"] = all(prefix_checks)
    config["training_scenario_digests"] = training_digests
    (out / "configuration.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (out / "candidate_provenance.json").write_text(
        json.dumps([json_candidate(x) for x in all_candidates], indent=2), encoding="utf-8")
    if not all_candidates:
        raise RuntimeError("no final feasible nondominated candidates")

    stage_started = time.perf_counter()
    base.ACTIVE_SCENARIO_SET, base._PATH_SCENARIO_CACHE, base.RISK_METRIC = validation, {}, "ccp"
    summaries = []
    scenario_file = out / "validation" / "scenario_level_results.csv.gz"
    with gzip.open(scenario_file, "wt", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["replicate", "method", "source_solution_id",
            "decision_fingerprint", "scenario_id", "cost", "emission", "makespan"])
        writer.writeheader()
        for row in all_candidates:
            fixed, before = deepcopy(row["individual"]), row["decision_fingerprint"]
            if decision_signature(fixed) != before:
                raise RuntimeError("candidate provenance fingerprint mismatch before Stage C")
            base.evaluate_individual(fixed, batches, arcs, tt_dict, waiting_cost,
                                     wait_emission, **eval_kwargs)
            after = decision_signature(fixed)
            if after != before:
                raise RuntimeError("Stage C altered a fixed decision")
            for sid, values in enumerate(zip(fixed.cost_s, fixed.emission_s, fixed.makespan_s)):
                writer.writerow(dict(replicate=row["replicate"], method=row["method"],
                    source_solution_id=row["source_solution_id"], decision_fingerprint=before,
                    scenario_id=sid, cost=float(values[0]), emission=float(values[1]),
                    makespan=float(values[2])))
            objective_summaries = {"cost": summarise(fixed.cost_s),
                                   "emission": summarise(fixed.emission_s),
                                   "makespan": summarise(fixed.makespan_s)}
            summaries.append({"replicate": row["replicate"], "method": row["method"],
                "source_solution_id": row["source_solution_id"], "decision_fingerprint": before,
                "stage_c_signature_before": before, "stage_c_signature_after": after,
                "training_objectives": row["optimisation_objectives"], **objective_summaries,
                "generalisation_delta": {name: metric_delta(vals["q90"],
                    row["optimisation_objectives"][name]) for name, vals in objective_summaries.items()},
                "emission_constant": bool(np.ptp(fixed.emission_s) <= 1e-12 * max(
                    1.0, float(np.max(np.abs(fixed.emission_s))))),
                "punctuality_diagnostic_min_batch_on_time_probability": (
                    float(min(fixed.batch_on_time_prob.values())) if fixed.batch_on_time_prob else None)})
    runtimes["stage_c_validation"] = time.perf_counter() - stage_started
    (out / "validation" / "per_candidate_summary.json").write_text(
        json.dumps(summaries, indent=2), encoding="utf-8")

    paired_rows = []
    per_run = []
    for replicate in range(1, 6):
        methods = {m: [x for x in summaries if x["replicate"] == replicate and x["method"] == m]
                   for m in ("CCP30", "CCP50")}
        quality, scaling = validated_quality(methods)
        entry: dict[str, Any] = {"replicate": replicate,
            "algorithm_seed": seeds[replicate - 1]["algorithm"],
            "training_seed": seeds[replicate - 1]["training_S50_master"],
            "normalisation": scaling}
        for method in ("CCP30", "CCP50"):
            entry[method] = {"runtime_seconds": runtimes[f"replicate_{replicate:02d}_{method}"],
                "candidate_count": len(methods[method]), "quality": quality[method],
                "cost_q90_generalisation_error": percent_errors(methods[method], "cost"),
                "makespan_q90_generalisation_error": percent_errors(methods[method], "makespan"),
                "emission_q90_generalisation_error": percent_errors(methods[method], "emission")}
        entry["CCP50_minus_CCP30"] = {
            "runtime_seconds": entry["CCP50"]["runtime_seconds"] - entry["CCP30"]["runtime_seconds"],
            "hypervolume": quality["CCP50"]["hypervolume"] - quality["CCP30"]["hypervolume"],
            "igd_plus": quality["CCP50"]["igd_plus"] - quality["CCP30"]["igd_plus"],
            "median_abs_pct_cost_q90_error": entry["CCP50"]["cost_q90_generalisation_error"]["median_absolute_percentage_error"] - entry["CCP30"]["cost_q90_generalisation_error"]["median_absolute_percentage_error"],
            "median_abs_pct_makespan_q90_error": entry["CCP50"]["makespan_q90_generalisation_error"]["median_absolute_percentage_error"] - entry["CCP30"]["makespan_q90_generalisation_error"]["median_absolute_percentage_error"]}
        paired_rows.append(entry); per_run.extend([{"replicate": replicate, "method": m, **entry[m]} for m in ("CCP30", "CCP50")])
    (out / "per_run_validation_summaries.json").write_text(json.dumps(per_run, indent=2), encoding="utf-8")
    (out / "paired_comparison.json").write_text(json.dumps(paired_rows, indent=2), encoding="utf-8")
    flat_fields = ["replicate", "algorithm_seed", "training_seed", "CCP30_runtime", "CCP50_runtime",
        "runtime_delta", "CCP30_candidates", "CCP50_candidates", "CCP30_HV", "CCP50_HV", "HV_delta",
        "CCP30_IGD_plus", "CCP50_IGD_plus", "IGD_plus_delta", "CCP30_cost_median_APE",
        "CCP50_cost_median_APE", "cost_median_APE_delta", "CCP30_makespan_median_APE",
        "CCP50_makespan_median_APE", "makespan_median_APE_delta"]
    with (out / "paired_comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=flat_fields); writer.writeheader()
        for x in paired_rows:
            d=x["CCP50_minus_CCP30"]; a=x["CCP30"]; b=x["CCP50"]
            writer.writerow(dict(replicate=x["replicate"], algorithm_seed=x["algorithm_seed"], training_seed=x["training_seed"],
                CCP30_runtime=a["runtime_seconds"], CCP50_runtime=b["runtime_seconds"], runtime_delta=d["runtime_seconds"],
                CCP30_candidates=a["candidate_count"], CCP50_candidates=b["candidate_count"],
                CCP30_HV=a["quality"]["hypervolume"], CCP50_HV=b["quality"]["hypervolume"], HV_delta=d["hypervolume"],
                CCP30_IGD_plus=a["quality"]["igd_plus"], CCP50_IGD_plus=b["quality"]["igd_plus"], IGD_plus_delta=d["igd_plus"],
                CCP30_cost_median_APE=a["cost_q90_generalisation_error"]["median_absolute_percentage_error"], CCP50_cost_median_APE=b["cost_q90_generalisation_error"]["median_absolute_percentage_error"], cost_median_APE_delta=d["median_abs_pct_cost_q90_error"],
                CCP30_makespan_median_APE=a["makespan_q90_generalisation_error"]["median_absolute_percentage_error"], CCP50_makespan_median_APE=b["makespan_q90_generalisation_error"]["median_absolute_percentage_error"], makespan_median_APE_delta=d["median_abs_pct_makespan_q90_error"]))

    def wins(metric, lower=False):
        a = sum((x["CCP30"]["quality"][metric] < x["CCP50"]["quality"][metric]) if lower else
                (x["CCP30"]["quality"][metric] > x["CCP50"]["quality"][metric]) for x in paired_rows)
        b = sum((x["CCP50"]["quality"][metric] < x["CCP30"]["quality"][metric]) if lower else
                (x["CCP50"]["quality"][metric] > x["CCP30"]["quality"][metric]) for x in paired_rows)
        return {"CCP30": a, "CCP50": b, "ties": 5-a-b}
    runtime_pct = [abs(x["CCP50_minus_CCP30"]["runtime_seconds"])/x["CCP30"]["runtime_seconds"]*100 for x in paired_rows]
    aggregate = {"HV_wins": wins("hypervolume"), "IGD_plus_wins": wins("igd_plus", lower=True),
        "lower_median_cost_q90_error": {m: sum(x[m]["cost_q90_generalisation_error"]["median_absolute_percentage_error"] < x[{"CCP30":"CCP50","CCP50":"CCP30"}[m]]["cost_q90_generalisation_error"]["median_absolute_percentage_error"] for x in paired_rows) for m in ("CCP30","CCP50")},
        "lower_median_makespan_q90_error": {m: sum(x[m]["makespan_q90_generalisation_error"]["median_absolute_percentage_error"] < x[{"CCP30":"CCP50","CCP50":"CCP30"}[m]]["makespan_q90_generalisation_error"]["median_absolute_percentage_error"] for x in paired_rows) for m in ("CCP30","CCP50")},
        "median_absolute_paired_runtime_difference_percent": float(np.median(runtime_pct)),
        "runtime_effectively_similar_by_predeclared_rule": bool(np.median(runtime_pct) <= 5.0),
        "all_stage_c_fingerprints_unchanged": all(x["stage_c_signature_before"] == x["stage_c_signature_after"] for x in summaries),
        "all_fixed_candidate_emissions_scenario_invariant": all(x["emission_constant"] for x in summaries),
        "total_candidate_scenario_evaluations": len(summaries) * args.validation_scenarios,
        "punctuality_is_diagnostic_only": True}
    (out / "aggregate_comparison_summary.json").write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
    (out / "runtime_summary.json").write_text(json.dumps(runtimes, indent=2), encoding="utf-8")
    (out / "README.md").write_text("# CCP S30/S50 selection study\n\nFive new paired full-size optimisation replicates compare S30 and S50. This is an engineering scenario-size selection study, not the formal experiment. Each pair shares its NSGA-II seed and uses an exact S30 prefix of one S50 master sample. All candidates are frozen for one common independent S=5000 selection-only validation set. That set must never be described or reused as the formal study's final validation set. Punctuality is diagnostic only.\n", encoding="utf-8")
    print(json.dumps({"aggregate": aggregate, "paired": paired_rows}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
