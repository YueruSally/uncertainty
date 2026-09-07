#!/usr/bin/env python3
"""Validation-only comparison of fixed Run-1 CCP30 training and OOS q90."""
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


ROOT = Path(__file__).resolve().parent
SOURCE = (ROOT / "formal_ev_vs_ccp_s30_30runs" / "run_01" / "CCP30"
          / "final_feasible_nondominated.json")
SOURCE_CONFIG = SOURCE.with_name("configuration.json")
DEFAULT_OUT = (ROOT / "formal_ev_vs_ccp_s30_30runs"
               / "run1_oos_validation_all_original")
OBJECTIVES = ("cost", "emission", "makespan")
TRAINING_SCENARIOS = 30
TRAINING_SEED = 920001
OOS_SCENARIOS = 5000
OOS_SEED = 930001
ALPHA = 0.9


def load_ccp30_rows(path=SOURCE):
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    if not rows:
        raise RuntimeError("no original CCP30 solutions found")
    if not all(row.get("method") == "CCP30" and row.get("run_id") == 1
               for row in rows):
        raise RuntimeError("fixed-solution source contains non-Run-1 CCP30 rows")
    ids = [row["source_solution_id"] for row in rows]
    if len(set(ids)) != len(ids):
        raise RuntimeError("CCP30 source solution IDs are not unique")
    return rows


def audit_training_semantics(rows, config_path=SOURCE_CONFIG):
    formal_config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    expected = {
        "method": "CCP30", "run_id": 1, "training_scenarios": 30,
        "training_seed": 920001, "alpha": 0.9,
    }
    if any(formal_config.get(key) != value for key, value in expected.items()):
        raise RuntimeError("formal CCP30 training configuration mismatch")
    if "separate empirical q90 objectives" not in formal_config.get(
            "formulation", ""):
        raise RuntimeError("formal CCP30 risk metric is not empirical q90")
    for row in rows:
        config = row.get("configuration", {})
        seeds = row.get("random_seeds", {})
        if config.get("alpha") != ALPHA:
            raise RuntimeError("CCP30 alpha is not 0.9")
        if seeds.get("training") != TRAINING_SEED:
            raise RuntimeError("CCP30 training seed is not 920001")
        stored = row.get("optimisation_objectives", {})
        if set(stored) != set(OBJECTIVES):
            raise RuntimeError("stored optimisation objectives are incomplete")
    # This is the formal experiment's exact S=30/order-statistic audit gate.
    if base.empirical_ccp_quantile(np.arange(TRAINING_SCENARIOS), ALPHA) != 26.0:
        raise RuntimeError("unexpected empirical CCP quantile semantics")


def ccp_q90(cost_s, emission_s, makespan_s):
    """Use the optimization's single aggregation implementation unchanged."""
    arrays = [np.asarray(values, dtype=float)
              for values in (cost_s, emission_s, makespan_s)]
    if any(values.size != OOS_SCENARIOS for values in arrays):
        raise RuntimeError("OOS objective arrays must each contain 5000 outcomes")
    return base.aggregate_scenario_objectives(
        *arrays, *(float(np.mean(values)) for values in arrays),
        mode="ccp", confidence_cost=ALPHA, confidence_emission=ALPHA,
        confidence_time=ALPHA)


def relative_error_pct(q90_30, q90_5000):
    if q90_5000 == 0:
        raise ZeroDivisionError("q90_5000 must be nonzero for relative error")
    return abs(q90_30 - q90_5000) / q90_5000 * 100.0


def comparison_row(source, q90_5000):
    """Join stored training objectives to q90 values computed from OOS arrays."""
    stored = source["optimisation_objectives"]
    row = {"solution_id": source["source_solution_id"]}
    for objective, oos_value in zip(OBJECTIVES, q90_5000):
        training_value = float(stored[objective])
        oos_value = float(oos_value)
        row[f"q90_30_{objective}"] = training_value
        row[f"q90_5000_{objective}"] = oos_value
        row[f"relative_error_{objective}_pct"] = relative_error_pct(
            training_value, oos_value)
    return row


def write_csv(path, rows):
    fields = ["solution_id"]
    for objective in OBJECTIVES:
        fields.extend([f"q90_30_{objective}", f"q90_5000_{objective}",
                       f"relative_error_{objective}_pct"])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_scatter(path, rows, objective):
    x = np.asarray([row[f"q90_30_{objective}"] for row in rows])
    y = np.asarray([row[f"q90_5000_{objective}"] for row in rows])
    lower = float(min(x.min(), y.min()))
    upper = float(max(x.max(), y.max()))
    padding = (upper - lower) * 0.04 or 1.0
    limits = (lower - padding, upper + padding)
    fig, axis = plt.subplots(figsize=(7, 7))
    axis.scatter(x, y, s=20, alpha=0.7)
    axis.plot(limits, limits, linestyle="--", color="black", label="y = x")
    axis.set_xlim(limits)
    axis.set_ylim(limits)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel(f"S=30 training q90 {objective}")
    axis.set_ylabel(f"S=5000 OOS q90 {objective}")
    axis.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def run(out=DEFAULT_OUT):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
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
    if wait_emission != 320.50 or not base.waiting_emission_configuration()[
            "carbon_cost_waiting_emission"]:
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
    unchanged = 0
    scenario_objects = set()
    for source in sources:
        if (base.ACTIVE_SCENARIO_SET is not validation
                or validation.size != OOS_SCENARIOS
                or validation.seed != OOS_SEED
                or scenario_digest(validation) != digest):
            raise RuntimeError("common OOS ScenarioSet changed")
        scenario_objects.add(id(base.ACTIVE_SCENARIO_SET))
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
            raise RuntimeError("OOS validation changed a fixed decision")
        unchanged += 1
        rows.append(comparison_row(source, ccp_q90(
            individual.cost_s, individual.emission_s, individual.makespan_s)))

    if len(rows) != len(sources) or len(scenario_objects) != 1:
        raise RuntimeError("validation completeness/common-scenario audit failed")
    write_csv(out / "ccp30_q90_s30_vs_s5000.csv", rows)
    for objective in OBJECTIVES:
        write_scatter(out / f"ccp30_q90_{objective}_s30_vs_s5000.png",
                      rows, objective)
    summary = {
        objective: {
            "mean_relative_error_pct": float(np.mean([
                row[f"relative_error_{objective}_pct"] for row in rows])),
            "median_relative_error_pct": float(np.median([
                row[f"relative_error_{objective}_pct"] for row in rows])),
            "maximum_relative_error_pct": float(np.max([
                row[f"relative_error_{objective}_pct"] for row in rows])),
        } for objective in OBJECTIVES
    }
    summary.update({
        "solution_count": len(rows),
        "post_validation_pareto_filtering": False,
        "training_scenarios": TRAINING_SCENARIOS,
        "training_seed": TRAINING_SEED,
        "oos_scenarios": OOS_SCENARIOS,
        "oos_seed": OOS_SEED,
        "alpha": ALPHA,
        "waiting_emission_gCO2_per_TEU_h": 320.50,
        "carbon_cost_waiting_emission": True,
    })
    (out / "ccp30_q90_s30_vs_s5000_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Validated {len(rows)} fixed CCP30 solutions; "
          f"fingerprints unchanged: {unchanged}; common ScenarioSet objects: "
          f"{len(scenario_objects)}", flush=True)
    return rows, summary


if __name__ == "__main__":
    run()
