#!/usr/bin/env python3
"""OOS coverage of stored CCP30 S=30 q90 thresholds; no optimisation/filtering."""
from __future__ import annotations

import csv
import json
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import baseline_uncertainty as base
from run_ccp_candidate_pool import decision_signature
from run_ev_ccp_oos_pilot import scenario_digest
from run_formal_ev_vs_ccp_s30_30runs import restore_individual
from run_ccp30_q90_oos_validation import (
    ALPHA, OBJECTIVES, OOS_SCENARIOS, OOS_SEED, ROOT, load_ccp30_rows,
    audit_training_semantics,
)


OUT = (ROOT / "formal_ev_vs_ccp_s30_30runs" /
       "run1_oos_validation_all_original")
CSV_NAME = "ccp30_q90_oos_coverage_by_solution.csv"
SUMMARY_NAME = "ccp30_q90_oos_coverage_summary.json"
FIGURE_NAME = "ccp30_q90_oos_mean_coverage.png"


def coverage_row(source, objective_arrays):
    row = {"solution_id": source["source_solution_id"]}
    for objective, values in zip(OBJECTIVES, objective_arrays):
        threshold = float(source["optimisation_objectives"][objective])
        values = np.asarray(values, dtype=float)
        if values.size != OOS_SCENARIOS:
            raise RuntimeError("coverage requires exactly 5000 OOS outcomes")
        covered = int(np.count_nonzero(values <= threshold))
        exceeded = int(np.count_nonzero(values > threshold))
        if covered + exceeded != OOS_SCENARIOS:
            raise RuntimeError("non-finite outcome broke coverage partition")
        row[f"q90_30_{objective}"] = threshold
        row[f"oos_coverage_{objective}"] = covered / OOS_SCENARIOS
        row[f"oos_exceedance_{objective}"] = exceeded / OOS_SCENARIOS
    return row


def summarise(rows):
    result = {}
    for objective in OBJECTIVES:
        coverage = np.asarray(
            [row[f"oos_coverage_{objective}"] for row in rows], dtype=float)
        exceedance = np.asarray(
            [row[f"oos_exceedance_{objective}"] for row in rows], dtype=float)
        result[objective] = {
            "mean_oos_coverage": float(np.mean(coverage)),
            "median_oos_coverage": float(np.median(coverage)),
            "minimum_oos_coverage": float(np.min(coverage)),
            "maximum_oos_coverage": float(np.max(coverage)),
            "mean_oos_exceedance_rate": float(np.mean(exceedance)),
            "median_oos_exceedance_rate": float(np.median(exceedance)),
            "coverage_below_0_90_count": int(np.count_nonzero(coverage < ALPHA)),
            "coverage_above_0_90_count": int(np.count_nonzero(coverage > ALPHA)),
            "coverage_exactly_0_90_count": int(np.count_nonzero(coverage == ALPHA)),
        }
    result.update({
        "solution_count": len(rows),
        "training_scenarios": 30,
        "training_seed": 920001,
        "oos_scenarios": OOS_SCENARIOS,
        "oos_seed": OOS_SEED,
        "target_coverage": ALPHA,
        "coverage_rule": "count(Y_oos <= q90_30) / 5000",
        "exceedance_rule": "count(Y_oos > q90_30) / 5000",
        "post_validation_pareto_filtering": False,
        "waiting_emission_gCO2_per_TEU_h": 320.50,
        "carbon_cost_waiting_emission": True,
    })
    return result


def write_outputs(rows, summary, out=OUT):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    fields = ["solution_id"]
    for objective in OBJECTIVES:
        fields.extend((f"q90_30_{objective}", f"oos_coverage_{objective}",
                       f"oos_exceedance_{objective}"))
    with (out / CSV_NAME).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (out / SUMMARY_NAME).write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    means = [100 * summary[objective]["mean_oos_coverage"]
             for objective in OBJECTIVES]
    fig, axis = plt.subplots(figsize=(7.5, 5.5))
    bars = axis.bar(("Cost", "Emissions", "Makespan"), means,
                    color=("#4c78a8", "#f58518", "#54a24b"))
    axis.axhline(100 * ALPHA, color="black", linestyle="--",
                 label="90% reference")
    axis.set_ylabel("Mean OOS coverage (%)")
    axis.set_ylim(min(80, min(means) - 2), 100)
    axis.bar_label(bars, labels=[f"{value:.2f}%" for value in means],
                   padding=3)
    axis.legend()
    axis.grid(axis="y", alpha=.2)
    fig.tight_layout()
    fig.savefig(out / FIGURE_NAME, dpi=180)
    plt.close(fig)


def run(out=OUT):
    sources = load_ccp30_rows()
    audit_training_semantics(sources)
    base.BORDER_EVENT_DEFINITIONS = base.load_border_event_definitions(
        base.DEFAULT_BORDER_EVENT_DATA_FILE)
    net = base.load_network_from_extended(ROOT / "data/data_expanded.xlsx")
    (node_names, node_region, node_hold_cost, node_proc_cost, node_trans_cost,
     arcs, timetables, batches, waiting_cost, _workbook_wait_emission,
     carbon_tax, _emission_factors, _mode_speeds, trans_map, border_delay_map,
     theta_rm, _) = net
    wait_emission = base.WAIT_EMISSION_gCO2_per_TEU_H_DEFAULT
    config = base.waiting_emission_configuration()
    if wait_emission != 320.50 or not config["carbon_cost_waiting_emission"]:
        raise RuntimeError("Scheme-B waiting-emission configuration mismatch")
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
        arcs, border_delay_map, OOS_SCENARIOS, OOS_SEED, stochastic=True,
        border_event_definitions=base.BORDER_EVENT_DEFINITIONS)
    digest = scenario_digest(validation)
    base.ACTIVE_SCENARIO_SET = validation
    base._PATH_SCENARIO_CACHE = {}
    base.RISK_METRIC = "ccp"
    base.CONFIDENCE_COST = base.CONFIDENCE_EMISSION = base.CONFIDENCE_TIME = ALPHA

    rows = []
    for source in sources:
        if (base.ACTIVE_SCENARIO_SET is not validation or
                scenario_digest(validation) != digest):
            raise RuntimeError("common OOS ScenarioSet changed")
        individual = restore_individual(source, paths, tt, arc_lookup)
        before = decision_signature(individual)
        base.evaluate_individual(
            individual, batches, arcs, tt, waiting_cost, wait_emission,
            node_hold_cost=node_hold_cost, node_proc_cost=node_proc_cost,
            carbon_tax_map=carbon_tax, trans_map=trans_map,
            border_delay_map=border_delay_map, theta_rm=theta_rm,
            node_trans_cost=node_trans_cost)
        after = decision_signature(individual)
        if before != source["decision_fingerprint"] or after != before:
            raise RuntimeError("coverage validation changed a fixed decision")
        rows.append(coverage_row(source, (
            individual.cost_s, individual.emission_s, individual.makespan_s)))
    if len(rows) != len(sources) or len({r["solution_id"] for r in rows}) != len(sources):
        raise RuntimeError("CCP30 coverage output is incomplete")
    summary = summarise(rows)
    write_outputs(rows, summary, out)
    print(f"Calculated OOS q90 coverage for {len(rows)} fixed CCP30 solutions; "
          "no filtering applied.", flush=True)
    return rows, summary


if __name__ == "__main__":
    run()
