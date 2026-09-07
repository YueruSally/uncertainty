#!/usr/bin/env python3
"""Validation-only OOS comparison of the formal Run-1 EV and CCP30 fronts."""
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import baseline_uncertainty as base
from run_ccp_candidate_pool import decision_signature
from run_ev_ccp_oos_pilot import scenario_digest
from run_formal_ev_vs_ccp_s30_30runs import restore_individual


ROOT = Path(__file__).resolve().parent
FORMAL = ROOT / "formal_ev_vs_ccp_s30_30runs"
RUN1 = FORMAL / "run_01"
DEFAULT_OUT = FORMAL / "run1_oos_validation_all_original"
VALIDATION_SEED = 930001
OBJECTIVES = ("cost", "emission", "makespan")
METHODS = ("EV", "CCP30")
WAITING_CONFIG = base.waiting_emission_configuration()


def nondominated_mask(points):
    """Minimisation mask in the full three-objective space; retain duplicates."""
    points = np.asarray(points, dtype=float)
    keep = np.ones(len(points), dtype=bool)
    for i, point in enumerate(points):
        weakly_better = np.all(points <= point, axis=1)
        strictly_better = np.any(points < point, axis=1)
        keep[i] = not np.any(weakly_better & strictly_better)
    return keep


def load_rows(ev_limit=None, ccp_limit=None):
    rows = []
    for method, limit in (("EV", ev_limit), ("CCP30", ccp_limit)):
        path = RUN1 / method / "final_feasible_nondominated.json"
        method_rows = json.loads(path.read_text(encoding="utf-8"))
        if not all(row["method"] == method and row["run_id"] == 1
                   for row in method_rows):
            raise RuntimeError(f"Run-1 provenance mismatch in {path}")
        rows.extend(method_rows if limit is None else method_rows[:limit])
    return rows


def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def all_solution_statistics(solutions):
    """Return one unfiltered OOS row for every fixed original solution."""
    rows = []
    for solution in solutions:
        row = {"solution_id": solution["solution_id"],
               "method": solution["method"]}
        for statistic, reducer in (("mean", np.mean), ("median", np.median)):
            for objective in OBJECTIVES:
                row[f"oos_{statistic}_{objective}"] = float(
                    reducer(solution[f"{objective}_s"]))
        for objective in OBJECTIVES:
            row[f"oos_q90_{objective}"] = float(
                base.empirical_ccp_quantile(
                    np.asarray(solution[f"{objective}_s"], dtype=float), .9))
        rows.append(row)
    return rows


def plot_all_means(path, rows, x_objective, y_objective):
    fig, axis = plt.subplots(figsize=(8, 6))
    for method, marker in (("EV", "o"), ("CCP30", "^")):
        selected = [row for row in rows if row["method"] == method]
        axis.scatter([r[f"oos_mean_{x_objective}"] for r in selected],
                     [r[f"oos_mean_{y_objective}"] for r in selected],
                     s=22, alpha=.7, marker=marker, label=method)
    axis.set_xlabel(f"OOS mean {x_objective}")
    axis.set_ylabel(f"OOS mean {y_objective}")
    axis.grid(alpha=.2)
    axis.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_all_means_3d(path, rows):
    fig = plt.figure(figsize=(9, 7))
    axis = fig.add_subplot(111, projection="3d")
    for method, marker in (("EV", "o"), ("CCP30", "^")):
        selected = [row for row in rows if row["method"] == method]
        axis.scatter([r["oos_mean_cost"] for r in selected],
                     [r["oos_mean_emission"] for r in selected],
                     [r["oos_mean_makespan"] for r in selected],
                     s=18, alpha=.65, marker=marker, label=method)
    axis.set_xlabel("OOS mean cost")
    axis.set_ylabel("OOS mean emissions")
    axis.set_zlabel("OOS mean makespan")
    axis.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def pareto_appearance_rows(solutions, scenario_count):
    """Count pooled first-front membership independently in every scenario."""
    if not solutions:
        return []
    objective_matrices = {
        objective: np.vstack([row[f"{objective}_s"] for row in solutions])
        for objective in OBJECTIVES
    }
    expected_shape = (len(solutions), scenario_count)
    if any(matrix.shape != expected_shape
           for matrix in objective_matrices.values()):
        raise RuntimeError("OOS objective arrays do not match the scenario count")

    appearances = np.zeros(len(solutions), dtype=np.int64)
    for scenario_index in range(scenario_count):
        points = np.column_stack([
            objective_matrices[objective][:, scenario_index]
            for objective in OBJECTIVES
        ])
        appearances += nondominated_mask(points)

    rows = [{
        "solution_id": row["solution_id"],
        "method": row["method"],
        "pareto_appearances": int(count),
        "pareto_frequency": float(count / scenario_count),
    } for row, count in zip(solutions, appearances)]
    return sorted(rows, key=lambda row: row["pareto_frequency"], reverse=True)


def solution_objective_summary_rows(solutions, frequency_rows):
    """Summarise OOS means and pooled mean-objective Pareto membership."""
    frequency_by_id = {row["solution_id"]: row for row in frequency_rows}
    if len(frequency_by_id) != len(frequency_rows):
        raise RuntimeError("Pareto appearance rows contain duplicate solution IDs")

    rows = []
    mean_points = []
    for solution in solutions:
        solution_id = solution["solution_id"]
        if solution_id not in frequency_by_id:
            raise RuntimeError(f"missing Pareto appearance row for {solution_id}")
        means = [float(np.mean(solution[f"{objective}_s"]))
                 for objective in OBJECTIVES]
        frequency = frequency_by_id[solution_id]
        rows.append({
            "solution_id": solution_id,
            "method": solution["method"],
            "oos_mean_cost": means[0],
            "oos_mean_emission": means[1],
            "oos_mean_makespan": means[2],
            "pareto_appearances": frequency["pareto_appearances"],
            "pareto_frequency": frequency["pareto_frequency"],
        })
        mean_points.append(means)

    for row, is_nondominated in zip(rows, nondominated_mask(mean_points)):
        row["oos_mean_pareto"] = bool(is_nondominated)
    return rows


def run(scenarios=5000, out=DEFAULT_OUT, ev_limit=None, ccp_limit=None):
    if scenarios < 1:
        raise ValueError("validation scenario count must be at least 1")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)

    base.BORDER_EVENT_DEFINITIONS = base.load_border_event_definitions(
        base.DEFAULT_BORDER_EVENT_DATA_FILE)
    net = base.load_network_from_extended(ROOT / "data/data_expanded.xlsx")
    (node_names, node_region, node_hold_cost, node_proc_cost, node_trans_cost,
     arcs, timetables, batches, waiting_cost, wait_emission, carbon_tax,
     emission_factors, mode_speeds, trans_map, border_delay_map, theta_rm, _) = net
    wait_emission = base.WAIT_EMISSION_gCO2_per_TEU_H_DEFAULT
    base.print_waiting_emission_configuration()
    for batch in batches:
        batch.penalty_per_teu_h = base.DEFAULT_LATE_PENALTY_USD_PER_TEU_H
    tt = base.build_timetable_dict(timetables)
    arc_lookup = base.build_arc_lookup(arcs)
    random.seed(0)
    np.random.seed(0)
    paths = base.build_path_library(
        node_names, node_region, arcs, batches, tt, arc_lookup)
    base.sanity_check_path_lib(batches, paths)

    validation = base.build_scenario_set(
        arcs, border_delay_map, scenarios, VALIDATION_SEED, stochastic=True,
        border_event_definitions=base.BORDER_EVENT_DEFINITIONS)
    common_digest = scenario_digest(validation)
    base.ACTIVE_SCENARIO_SET = validation
    base._PATH_SCENARIO_CACHE = {}
    base.RISK_METRIC = "ccp"  # controls summaries only; never feasibility selection

    evaluated = []
    scenario_object_ids = set()
    unchanged_fingerprints = 0
    for source in load_rows(ev_limit, ccp_limit):
        if (base.ACTIVE_SCENARIO_SET is not validation
                or base.ACTIVE_SCENARIO_SET.seed != VALIDATION_SEED
                or scenario_digest(base.ACTIVE_SCENARIO_SET) != common_digest):
            raise RuntimeError("common validation ScenarioSet changed")
        scenario_object_ids.add(id(base.ACTIVE_SCENARIO_SET))
        individual = restore_individual(source, paths, tt, arc_lookup)
        before = decision_signature(individual)
        base.evaluate_individual(
            individual, batches, arcs, tt, waiting_cost, wait_emission,
            node_hold_cost=node_hold_cost, node_proc_cost=node_proc_cost,
            carbon_tax_map=carbon_tax, trans_map=trans_map,
            border_delay_map=border_delay_map, theta_rm=theta_rm,
            node_trans_cost=node_trans_cost)
        after = decision_signature(individual)
        if before != after:
            raise RuntimeError("OOS validation changed a fixed decision")
        unchanged_fingerprints += 1
        evaluated.append({
            "solution_id": source["source_solution_id"],
            "method": source["method"],
            "cost_s": np.asarray(individual.cost_s, dtype=float),
            "emission_s": np.asarray(individual.emission_s, dtype=float),
            "makespan_s": np.asarray(individual.makespan_s, dtype=float),
        })

    if len(scenario_object_ids) != 1:
        raise RuntimeError("solutions did not all use the same ScenarioSet object")
    expected_rows = len(load_rows(ev_limit, ccp_limit))
    if len(evaluated) != expected_rows:
        raise RuntimeError(f"expected {expected_rows} output rows, got {len(evaluated)}")
    solution_ids = [row["solution_id"] for row in evaluated]
    if len(set(solution_ids)) != len(solution_ids):
        raise RuntimeError("Run-1 solution IDs are not unique across EV and CCP30")

    summaries = all_solution_statistics(evaluated)
    summary_fields = ["solution_id", "method"]
    for statistic in ("mean", "median", "q90"):
        summary_fields.extend(f"oos_{statistic}_{objective}"
                              for objective in OBJECTIVES)
    write_csv(out / "run1_all_original_solutions_oos_statistics.csv",
              summaries, summary_fields)
    for x, y in (("cost", "emission"), ("cost", "makespan"),
                 ("emission", "makespan")):
        plot_all_means(out / f"all_original_oos_mean_{x}_vs_{y}.png",
                       summaries, x, y)
    plot_all_means_3d(
        out / "supporting_all_original_oos_mean_cost_emission_makespan_3d.png",
        summaries)
    method_counts = {method: sum(row["method"] == method for row in evaluated)
                     for method in METHODS}
    print(f"EV solutions loaded: {method_counts['EV']}", flush=True)
    print(f"CCP30 solutions loaded: {method_counts['CCP30']}", flush=True)
    print(f"Total solutions: {len(evaluated)}", flush=True)
    print(f"OOS scenarios: {scenarios}", flush=True)
    print(f"Validation seed: {VALIDATION_SEED}", flush=True)
    (out / "validation_configuration.json").write_text(json.dumps({
        "run_id": 1, "methods": list(METHODS), "scenario_count": scenarios,
        "validation_seed": VALIDATION_SEED, "scenario_digest": common_digest,
        "one_common_scenario_set_for_all_solutions": True,
        "common_scenario_set_object_count": len(scenario_object_ids),
        "scenario_level_results_written": False,
        "solution_count": len(evaluated),
        "fingerprints_unchanged_before_after": unchanged_fingerprints,
        "post_validation_pareto_filtering": False,
        "analysis": ["all_original_fixed_solutions_oos_statistics",
                     "all_original_fixed_solutions_oos_mean_plots"],
        **WAITING_CONFIG,
    }, indent=2) + "\n", encoding="utf-8")
    return summaries


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenarios", type=int, default=5000)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--ev-limit", type=int)
    parser.add_argument("--ccp-limit", type=int)
    args = parser.parse_args()
    rows = run(args.scenarios, args.output_dir,
               args.ev_limit, args.ccp_limit)
    print(f"Validated {len(rows)} fixed Run-1 solutions; saved unfiltered "
          f"OOS statistics and mean-objective plots to "
          f"{args.output_dir}")


if __name__ == "__main__":
    main()
