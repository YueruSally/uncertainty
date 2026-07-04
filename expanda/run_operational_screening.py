#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Operational uncertainty coarse screening for the expanded 40-batch network.

The script keeps baseline3.py as the optimization core. It first solves a
deterministic baseline, then injects one candidate operational uncertainty at a
time and re-optimizes the routing plan. A candidate is retained when it affects
service performance and/or changes routing decisions.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import baseline3 as B


ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = ROOT / "outputs"


CANDIDATES: dict[str, dict[str, str]] = {
    "C1": {
        "name": "line_haul_travel_time",
        "label": "Line-haul travel time uncertainty",
        "mechanism": "realized_time",
        "literature_class": "transit time / travel time reliability",
        "operational_meaning": (
            "Realized movement time on rail, road and water main-haul arcs "
            "deviates from the planned travel time."
        ),
    },
    "C2": {
        "name": "border_processing_time",
        "label": "Border processing and break-of-gauge time uncertainty",
        "mechanism": "realized_time",
        "literature_class": "service/transfer time + queueing delay",
        "operational_meaning": (
            "Customs, inspection, break-of-gauge handling and queueing at "
            "border nodes vary during execution."
        ),
    },
    "C3": {
        "name": "terminal_transshipment_time",
        "label": "Terminal/transshipment handling time uncertainty",
        "mechanism": "realized_time",
        "literature_class": "transfer time / terminal handling time",
        "operational_meaning": (
            "Handling and transfer processing time at terminals, ports and "
            "hubs varies during execution."
        ),
    },
    "C4": {
        "name": "capacity_slot_availability",
        "label": "Service capacity and slot availability uncertainty",
        "mechanism": "service_feasibility",
        "literature_class": "capacity uncertainty",
        "operational_meaning": (
            "Short-term available capacity on links, nodes, services, slots "
            "or wagons is lower than planned."
        ),
    },
    "C5": {
        "name": "shipment_volume",
        "label": "Shipment volume uncertainty",
        "mechanism": "shipment_planning",
        "literature_class": "demand uncertainty",
        "operational_meaning": (
            "Execution-stage batch quantities or short-term OD shipment "
            "volumes deviate from the planned quantities."
        ),
    },
    "C6": {
        "name": "corridor_link_node_disruption",
        "label": "Corridor/link/node disruption uncertainty",
        "mechanism": "network_disruption",
        "literature_class": "link-node failure / disruption",
        "operational_meaning": (
            "A recurrent operational disruption temporarily reduces the "
            "availability of key corridors, links, border nodes or terminals."
        ),
    },
}


LEVELS: dict[str, dict[str, float]] = {
    "low": {
        "line_time_spread": 0.08,
        "border_time_spread": 0.15,
        "terminal_time_spread": 0.15,
        "capacity_loss": 0.10,
        "demand_growth": 0.10,
        "disruption_capacity_factor": 0.50,
        "disruption_delay_spread": 0.25,
        "disruption_nodes": 1,
    },
    "medium": {
        "line_time_spread": 0.18,
        "border_time_spread": 0.35,
        "terminal_time_spread": 0.30,
        "capacity_loss": 0.25,
        "demand_growth": 0.25,
        "disruption_capacity_factor": 0.25,
        "disruption_delay_spread": 0.50,
        "disruption_nodes": 2,
    },
    "high": {
        "line_time_spread": 0.30,
        "border_time_spread": 0.60,
        "terminal_time_spread": 0.50,
        "capacity_loss": 0.40,
        "demand_growth": 0.40,
        "disruption_capacity_factor": 0.05,
        "disruption_delay_spread": 0.85,
        "disruption_nodes": 3,
    },
}


MODES: dict[str, dict[str, Any]] = {
    "smoke": {
        "pop": 20,
        "gens": 5,
        "seeds": [2026],
        "levels": ["medium"],
    },
    "quick": {
        "pop": 60,
        "gens": 25,
        "seeds": [2026, 7],
        "levels": ["low", "medium", "high"],
    },
    "coarse": {
        "pop": 90,
        "gens": 50,
        "seeds": [2026, 7, 99],
        "levels": ["low", "medium", "high"],
    },
    "full": {
        "pop": 180,
        "gens": 120,
        "seeds": [2026, 7, 99, 1000, 42],
        "levels": ["low", "medium", "high"],
    },
}


DEFAULT_DISRUPTION_NODES = [
    "Khorgos",
    "Alashankou",
    "Erenhot",
    "Manzhouli",
    "Brest",
    "Malaszewicze",
]


@dataclass
class ModelInputs:
    node_names: list[str]
    node_region: dict[str, str]
    node_hold_cost: dict[str, float]
    node_proc_cost: dict[str, float]
    node_trans_cost: dict[str, float]
    arcs: list[B.Arc]
    timetables: list[B.TimetableEntry]
    batches: list[B.Batch]
    waiting_cost_per_teu_h: float
    wait_emis_g_per_teu_h: float
    carbon_tax_map: dict
    emission_factor_map: dict
    mode_speeds_map: dict
    trans_map: dict
    border_delay_map: dict
    theta_rm: dict
    border_capacity: dict[str, float]
    background_flow: dict[str, float]


def _candidate_seed(seed: int, candidate: str, level: str) -> int:
    return int(seed + 1009 * int(candidate[1:]) + 9173 * list(LEVELS).index(level))


def _delay_multiplier(rng: np.random.Generator, spread: float) -> float:
    if spread <= 0:
        return 1.0
    return float(rng.triangular(1.0, 1.0 + 0.50 * spread, 1.0 + spread))


def _capacity_multiplier(rng: np.random.Generator, loss: float) -> float:
    if loss <= 0:
        return 1.0
    return float(rng.triangular(max(0.01, 1.0 - loss), 1.0 - 0.50 * loss, 1.0))


def _demand_multiplier(rng: np.random.Generator, growth: float) -> float:
    if growth <= 0:
        return 1.0
    return float(rng.triangular(1.0, 1.0 + 0.50 * growth, 1.0 + growth))


def set_capacity_globals(inputs: ModelInputs) -> None:
    B.BORDER_CAPACITY = deepcopy(inputs.border_capacity)
    B.BACKGROUND_FLOW = deepcopy(inputs.background_flow)


def load_inputs(data_path: Path, expected_batches: int) -> ModelInputs:
    loaded = B.load_network_from_extended(str(data_path))
    inputs = ModelInputs(
        node_names=loaded[0],
        node_region=loaded[1],
        node_hold_cost=loaded[2],
        node_proc_cost=loaded[3],
        node_trans_cost=loaded[4],
        arcs=loaded[5],
        timetables=loaded[6],
        batches=loaded[7],
        waiting_cost_per_teu_h=loaded[8],
        wait_emis_g_per_teu_h=loaded[9],
        carbon_tax_map=loaded[10],
        emission_factor_map=loaded[11],
        mode_speeds_map=loaded[12],
        trans_map=loaded[13],
        border_delay_map=loaded[14],
        theta_rm=loaded[15],
        border_capacity=deepcopy(B.BORDER_CAPACITY),
        background_flow=deepcopy(B.BACKGROUND_FLOW),
    )
    if expected_batches and len(inputs.batches) != expected_batches:
        raise ValueError(
            f"Expected {expected_batches} batches, got {len(inputs.batches)} "
            f"from {data_path}"
        )
    return inputs


def build_path_library(inputs: ModelInputs, seed: int) -> dict:
    set_capacity_globals(inputs)
    tt_dict = B.build_timetable_dict(inputs.timetables)
    arc_lookup = B.build_arc_lookup(inputs.arcs)
    random.seed(seed)
    np.random.seed(seed)
    path_lib = B.build_path_library(
        inputs.node_names,
        inputs.node_region,
        inputs.arcs,
        inputs.batches,
        tt_dict,
        arc_lookup,
    )
    B.sanity_check_path_lib(inputs.batches, path_lib)
    return path_lib


def apply_candidate(
    base_inputs: ModelInputs,
    candidate: str,
    level: str,
    seed: int,
    disruption_pool: list[str],
) -> tuple[ModelInputs, dict[str, Any]]:
    inputs = deepcopy(base_inputs)
    cfg = LEVELS[level]
    rng = np.random.default_rng(_candidate_seed(seed, candidate, level))
    params: dict[str, Any] = {"candidate": candidate, "level": level}

    if candidate == "C1":
        multipliers = []
        spread = float(cfg["line_time_spread"])
        for arc in inputs.arcs:
            m = _delay_multiplier(rng, spread)
            arc.speed_kmh = max(1.0, arc.speed_kmh / m)
            multipliers.append(m)
        params.update({"time_spread": spread, "mean_multiplier": float(np.mean(multipliers))})

    elif candidate == "C2":
        multipliers = []
        spread = float(cfg["border_time_spread"])
        new_map = {}
        for key, value in inputs.border_delay_map.items():
            node, mode = key
            if node in B.BREAK_OF_GAUGE_NODES:
                m = _delay_multiplier(rng, spread)
                new_map[key] = float(value) * m
                multipliers.append(m)
            else:
                new_map[key] = value
        inputs.border_delay_map = new_map
        params.update({"time_spread": spread, "mean_multiplier": _safe_mean(multipliers)})

    elif candidate == "C3":
        multipliers = []
        spread = float(cfg["terminal_time_spread"])
        for rec in inputs.trans_map.values():
            m = _delay_multiplier(rng, spread)
            rec["time_h"] = float(rec.get("time_h", 0.0)) * m
            multipliers.append(m)
        params.update({"time_spread": spread, "mean_multiplier": _safe_mean(multipliers)})

    elif candidate == "C4":
        multipliers = []
        loss = float(cfg["capacity_loss"])
        for arc in inputs.arcs:
            m = _capacity_multiplier(rng, loss)
            arc.capacity = max(0.0, float(arc.capacity) * m)
            multipliers.append(m)
        for node in list(inputs.border_capacity.keys()):
            m = _capacity_multiplier(rng, loss)
            inputs.border_capacity[node] = max(0.0, float(inputs.border_capacity[node]) * m)
            multipliers.append(m)
        params.update({"capacity_loss": loss, "mean_capacity_multiplier": _safe_mean(multipliers)})

    elif candidate == "C5":
        multipliers = []
        growth = float(cfg["demand_growth"])
        for batch in inputs.batches:
            m = _demand_multiplier(rng, growth)
            batch.quantity = float(batch.quantity) * m
            multipliers.append(m)
        params.update({"demand_growth": growth, "mean_quantity_multiplier": _safe_mean(multipliers)})

    elif candidate == "C6":
        node_count = int(cfg["disruption_nodes"])
        cap_factor = float(cfg["disruption_capacity_factor"])
        delay_spread = float(cfg["disruption_delay_spread"])
        available_nodes = [
            n for n in disruption_pool
            if n in inputs.node_names or n in inputs.border_capacity
        ]
        if not available_nodes:
            available_nodes = [n for n in B.BREAK_OF_GAUGE_NODES if n in inputs.node_names]
        node_count = min(max(1, node_count), len(available_nodes))
        disrupted_nodes = sorted(rng.choice(available_nodes, size=node_count, replace=False).tolist())
        for arc in inputs.arcs:
            if arc.from_node in disrupted_nodes or arc.to_node in disrupted_nodes:
                arc.capacity = max(0.0, float(arc.capacity) * cap_factor)
        for node in disrupted_nodes:
            if node in inputs.border_capacity:
                inputs.border_capacity[node] = max(0.0, float(inputs.border_capacity[node]) * cap_factor)
        new_map = {}
        for key, value in inputs.border_delay_map.items():
            node, _mode = key
            if node in disrupted_nodes:
                new_map[key] = float(value) * _delay_multiplier(rng, delay_spread)
            else:
                new_map[key] = value
        inputs.border_delay_map = new_map
        params.update({
            "disrupted_nodes": disrupted_nodes,
            "capacity_factor": cap_factor,
            "delay_spread": delay_spread,
        })

    else:
        raise ValueError(f"Unknown candidate: {candidate}")

    return inputs, params


def _safe_mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 1.0


def run_one_model(
    inputs: ModelInputs,
    path_lib: dict,
    pop_size: int,
    generations: int,
    seed: int,
) -> tuple[B.Individual, dict[str, Any]]:
    set_capacity_globals(inputs)
    random.seed(seed)
    np.random.seed(seed)
    start = time.perf_counter()
    (
        population,
        pareto,
        _front_hist,
        feasible_ratio_hist,
        feasible_ratio_strict_hist,
        _vio_mean_hist,
        boost_trigger_hist,
        boost_new_feas_hist,
        _pareto_size_hist,
        _feasible_count_hist,
        runtime_s,
    ) = B.run_nsga2(
        inputs.node_names,
        inputs.node_region,
        inputs.node_hold_cost,
        inputs.node_proc_cost,
        inputs.node_trans_cost,
        inputs.arcs,
        inputs.timetables,
        inputs.batches,
        inputs.waiting_cost_per_teu_h,
        inputs.wait_emis_g_per_teu_h,
        inputs.carbon_tax_map,
        inputs.emission_factor_map,
        inputs.mode_speeds_map,
        inputs.trans_map,
        inputs.border_delay_map,
        inputs.theta_rm,
        path_lib,
        pop_size=pop_size,
        generations=generations,
    )
    representative = select_representative(population, pareto)
    meta = {
        "runtime_s": float(runtime_s),
        "wall_s": float(time.perf_counter() - start),
        "pareto_size": int(len(pareto)),
        "final_feasible_ratio": float(feasible_ratio_hist[-1]) if feasible_ratio_hist else 0.0,
        "final_feasible_ratio_strict": (
            float(feasible_ratio_strict_hist[-1]) if feasible_ratio_strict_hist else 0.0
        ),
        "boost_gens_triggered": int(sum(boost_trigger_hist)),
        "boost_new_feasible_total": int(sum(boost_new_feas_hist)),
    }
    return representative, meta


def select_representative(population: list[B.Individual], pareto: list[B.Individual]) -> B.Individual:
    feasible_pareto = [ind for ind in pareto if ind.feasible]
    if feasible_pareto:
        return min(feasible_pareto, key=lambda ind: ind.objectives[0])
    feasible_pop = [ind for ind in population if ind.feasible]
    if feasible_pop:
        return min(feasible_pop, key=lambda ind: ind.objectives[0])
    return min(population, key=lambda ind: (ind.penalty, ind.objectives[0]))


def weighted_quantile(values: list[float], weights: list[float], q: float) -> float:
    if not values:
        return 0.0
    arr = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    order = np.argsort(arr)
    arr = arr[order]
    w = w[order]
    cum = np.cumsum(w)
    if cum[-1] <= 0:
        return float(arr[-1])
    return float(arr[np.searchsorted(cum, q * cum[-1], side="left")])


def solution_metrics(ind: B.Individual, inputs: ModelInputs) -> dict[str, Any]:
    set_capacity_globals(inputs)
    tt_dict = B.build_timetable_dict(inputs.timetables)
    arc_flow_map: dict = {}
    node_flow_map: dict = {}
    total_teu = sum(float(b.quantity) for b in inputs.batches)
    on_time_teu = 0.0
    infeasible_teu = 0.0
    lateness_values: list[float] = []
    lateness_weights: list[float] = []

    for batch in inputs.batches:
        key = (batch.origin, batch.destination, batch.batch_id)
        for alloc in ind.od_allocations.get(key, []):
            flow = float(alloc.share) * float(batch.quantity)
            if flow <= 1e-12:
                continue
            travel_time, _node_wait, miss_tt = B.simulate_path_time_capacity(
                alloc.path,
                batch,
                flow,
                tt_dict,
                arc_flow_map,
                trans_map=inputs.trans_map,
                border_delay_map=inputs.border_delay_map,
                node_flow_map=node_flow_map,
            )
            if math.isinf(travel_time) or miss_tt:
                infeasible_teu += flow
                lateness_values.append(1.0e6)
                lateness_weights.append(flow)
                continue
            lateness = max(0.0, float(batch.ET) + float(travel_time) - float(batch.LT))
            if lateness <= 1e-9:
                on_time_teu += flow
            lateness_values.append(lateness)
            lateness_weights.append(flow)

    late_teu_h = float(ind.vio_breakdown.get("late_teu_h", 0.0))
    return {
        "cost": float(ind.objectives[0]),
        "emission_gCO2": float(ind.objectives[1]),
        "time_h": float(ind.objectives[2]),
        "penalty": float(ind.penalty),
        "feasible": bool(ind.feasible),
        "total_teu": float(total_teu),
        "on_time_rate": float(on_time_teu / total_teu) if total_teu > 0 else 0.0,
        "infeasible_teu": float(infeasible_teu),
        "late_teu_h": late_teu_h,
        "late_h_per_teu": float(late_teu_h / total_teu) if total_teu > 0 else 0.0,
        "p90_lateness_h": weighted_quantile(lateness_values, lateness_weights, 0.90),
        "p95_lateness_h": weighted_quantile(lateness_values, lateness_weights, 0.95),
        "max_border_util": float(ind.vio_breakdown.get("max_border_util", 0.0)),
        "border_cap_excess": float(ind.vio_breakdown.get("border_cap_excess", 0.0)),
        "arc_cap_excess": float(ind.vio_breakdown.get("cap_excess", 0.0)),
    }


def allocation_share_map(ind: B.Individual, batch: B.Batch) -> dict[tuple, float]:
    key = (batch.origin, batch.destination, batch.batch_id)
    out: dict[tuple, float] = {}
    for alloc in ind.od_allocations.get(key, []):
        signature = (tuple(alloc.path.nodes), tuple(alloc.path.modes))
        out[signature] = out.get(signature, 0.0) + float(alloc.share)
    return out


def route_change_ratio(
    baseline_ind: B.Individual,
    scenario_ind: B.Individual,
    baseline_inputs: ModelInputs,
    scenario_inputs: ModelInputs,
) -> float:
    baseline_by_id = {b.batch_id: b for b in baseline_inputs.batches}
    changed_teu = 0.0
    total_teu = 0.0
    for batch in scenario_inputs.batches:
        base_batch = baseline_by_id.get(batch.batch_id)
        if base_batch is None:
            continue
        base_map = allocation_share_map(baseline_ind, base_batch)
        scenario_map = allocation_share_map(scenario_ind, batch)
        overlap = 0.0
        for signature, scenario_share in scenario_map.items():
            overlap += min(float(scenario_share), float(base_map.get(signature, 0.0)))
        batch_change = max(0.0, 1.0 - min(1.0, overlap))
        changed_teu += float(batch.quantity) * batch_change
        total_teu += float(batch.quantity)
    return float(changed_teu / total_teu) if total_teu > 0 else 0.0


def build_row(
    candidate: str,
    level: str,
    seed: int,
    params: dict[str, Any],
    metrics: dict[str, Any],
    meta: dict[str, Any],
    baseline_metrics: dict[str, Any],
    route_change: float,
) -> dict[str, Any]:
    row = {
        "candidate": candidate,
        "candidate_name": CANDIDATES[candidate]["name"] if candidate != "baseline" else "baseline",
        "candidate_label": CANDIDATES[candidate]["label"] if candidate != "baseline" else "Baseline",
        "level": level,
        "seed": seed,
        "route_change_ratio": route_change,
        **metrics,
        **meta,
        "scenario_params_json": json.dumps(params, ensure_ascii=False, sort_keys=True),
    }
    if candidate != "baseline":
        row.update({
            "cost_delta_rel": rel_delta(metrics["cost"], baseline_metrics["cost"]),
            "time_delta_rel": rel_delta(metrics["time_h"], baseline_metrics["time_h"]),
            "emission_delta_rel": rel_delta(metrics["emission_gCO2"], baseline_metrics["emission_gCO2"]),
            "on_time_drop": float(baseline_metrics["on_time_rate"] - metrics["on_time_rate"]),
            "late_h_per_teu_delta": float(
                metrics["late_h_per_teu"] - baseline_metrics["late_h_per_teu"]
            ),
            "p95_lateness_delta_h": float(
                metrics["p95_lateness_h"] - baseline_metrics["p95_lateness_h"]
            ),
        })
    else:
        row.update({
            "cost_delta_rel": 0.0,
            "time_delta_rel": 0.0,
            "emission_delta_rel": 0.0,
            "on_time_drop": 0.0,
            "late_h_per_teu_delta": 0.0,
            "p95_lateness_delta_h": 0.0,
        })
    return row


def rel_delta(value: float, base: float) -> float:
    if abs(base) <= 1e-12:
        return 0.0
    return float((value - base) / abs(base))


def summarize_results(
    rows: list[dict[str, Any]],
    kpi_threshold: float,
    route_threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.DataFrame(rows)
    scenarios = df[df["candidate"] != "baseline"].copy()
    if scenarios.empty:
        return pd.DataFrame(), pd.DataFrame()

    summary = (
        scenarios
        .groupby(["candidate", "candidate_name", "candidate_label", "level"], dropna=False)
        .agg(
            runs=("seed", "count"),
            cost_delta_rel_mean=("cost_delta_rel", "mean"),
            time_delta_rel_mean=("time_delta_rel", "mean"),
            emission_delta_rel_mean=("emission_delta_rel", "mean"),
            on_time_drop_mean=("on_time_drop", "mean"),
            late_h_per_teu_delta_mean=("late_h_per_teu_delta", "mean"),
            p95_lateness_delta_h_mean=("p95_lateness_delta_h", "mean"),
            route_change_ratio_mean=("route_change_ratio", "mean"),
            infeasible_teu_mean=("infeasible_teu", "mean"),
            feasible_ratio_mean=("final_feasible_ratio", "mean"),
            runtime_s_mean=("runtime_s", "mean"),
        )
        .reset_index()
    )
    summary["kpi_score"] = summary.apply(kpi_score, axis=1)
    summary["kpi_material"] = summary["kpi_score"] >= float(kpi_threshold)
    summary["decision_material"] = summary["route_change_ratio_mean"] >= float(route_threshold)
    summary["retain_for_refined_screening"] = (
        summary["kpi_material"] | summary["decision_material"]
    )
    summary["screening_class"] = summary.apply(classify_row, axis=1)

    ranking = (
        summary
        .groupby(["candidate", "candidate_name", "candidate_label"], dropna=False)
        .agg(
            max_kpi_score=("kpi_score", "max"),
            mean_kpi_score=("kpi_score", "mean"),
            max_route_change_ratio=("route_change_ratio_mean", "max"),
            mean_route_change_ratio=("route_change_ratio_mean", "mean"),
            retained_levels=("retain_for_refined_screening", "sum"),
            tested_levels=("level", "count"),
        )
        .reset_index()
    )
    ranking["retain_for_refined_screening"] = ranking["retained_levels"] > 0
    ranking["priority_score"] = (
        ranking["max_kpi_score"] + ranking["max_route_change_ratio"]
    )
    ranking = ranking.sort_values(
        ["retain_for_refined_screening", "priority_score", "max_kpi_score"],
        ascending=[False, False, False],
    )
    return summary, ranking


def kpi_score(row: pd.Series) -> float:
    return float(max(
        abs(float(row["cost_delta_rel_mean"])),
        abs(float(row["time_delta_rel_mean"])),
        max(0.0, float(row["on_time_drop_mean"])),
        abs(float(row["late_h_per_teu_delta_mean"])) / 24.0,
        abs(float(row["p95_lateness_delta_h_mean"])) / 168.0,
    ))


def classify_row(row: pd.Series) -> str:
    kpi = bool(row["kpi_material"])
    decision = bool(row["decision_material"])
    if kpi and decision:
        return "core_decision_relevant"
    if kpi:
        return "performance_risk"
    if decision:
        return "strategy_shift"
    return "screen_out"


def write_report(
    out_dir: Path,
    data_path: Path,
    args: argparse.Namespace,
    summary: pd.DataFrame,
    ranking: pd.DataFrame,
) -> None:
    lines = [
        "# Operational Uncertainty Coarse Screening",
        "",
        "This experiment identifies decision-relevant operational uncertainty "
        "candidates for the subsequent stochastic/robust routing model.",
        "",
        "## Run Configuration",
        "",
        f"- Data: `{data_path}`",
        f"- Expected batches: `{args.expected_batches}`",
        f"- Population: `{args.pop}`",
        f"- Generations: `{args.gens}`",
        f"- Seeds: `{', '.join(map(str, args.seeds))}`",
        f"- Levels: `{', '.join(args.levels)}`",
        f"- KPI threshold: `{args.kpi_threshold}`",
        f"- Route-change threshold: `{args.route_threshold}`",
        "",
        "## Candidate Pool",
        "",
        "| ID | Candidate | Mechanism | Literature class |",
        "|---|---|---|---|",
    ]
    for cid, meta in CANDIDATES.items():
        lines.append(
            f"| {cid} | {meta['label']} | {meta['mechanism']} | "
            f"{meta['literature_class']} |"
        )

    lines.extend(["", "## Ranking", ""])
    if ranking.empty:
        lines.append("No scenario rows were produced.")
    else:
        cols = [
            "candidate",
            "candidate_label",
            "max_kpi_score",
            "max_route_change_ratio",
            "retained_levels",
            "retain_for_refined_screening",
        ]
        lines.append(markdown_table(ranking[cols]))

    lines.extend([
        "",
        "## Selection Logic",
        "",
        "A candidate is retained if it materially affects operational KPIs "
        "and/or materially changes routing decisions. Factors that only affect "
        "capacity-induced waiting should not be entered into the final model "
        "together with their realized delay representation, to avoid double "
        "counting the same mechanism.",
    ])
    if not summary.empty:
        lines.extend(["", "## Level Summary", "", markdown_table(summary)])

    (out_dir / "screening_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return ""
    headers = [str(c) for c in df.columns]
    rows = []
    for _, row in df.iterrows():
        rows.append([format_markdown_cell(row[c]) for c in df.columns])
    out = []
    out.append("| " + " | ".join(headers) + " |")
    out.append("| " + " | ".join("---" for _ in headers) + " |")
    for row in rows:
        out.append("| " + " | ".join(row) + " |")
    return "\n".join(out)


def format_markdown_cell(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    if pd.isna(value):
        return ""
    text = str(value)
    return text.replace("|", "\\|").replace("\n", " ")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run operational uncertainty coarse screening on the 40-batch expanded data."
    )
    parser.add_argument("--data", default=str(ROOT / "data" / "data_expanded.xlsx"))
    parser.add_argument("--out", default="")
    parser.add_argument("--mode", choices=MODES.keys(), default="quick")
    parser.add_argument("--candidates", nargs="+", default=list(CANDIDATES.keys()))
    parser.add_argument("--levels", nargs="+", default=None, choices=list(LEVELS.keys()))
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--pop", type=int, default=0)
    parser.add_argument("--gens", type=int, default=0)
    parser.add_argument("--expected-batches", type=int, default=40)
    parser.add_argument("--kpi-threshold", type=float, default=0.03)
    parser.add_argument("--route-threshold", type=float, default=0.10)
    parser.add_argument("--disruption-nodes", nargs="+", default=DEFAULT_DISRUPTION_NODES)
    args = parser.parse_args()

    mode_cfg = MODES[args.mode]
    args.pop = args.pop or int(mode_cfg["pop"])
    args.gens = args.gens or int(mode_cfg["gens"])
    args.seeds = args.seeds or list(mode_cfg["seeds"])
    args.levels = args.levels or list(mode_cfg["levels"])
    bad = sorted(set(args.candidates) - set(CANDIDATES))
    if bad:
        raise ValueError(f"Unknown candidates: {bad}")
    return args


def main() -> None:
    args = parse_args()
    data_path = Path(args.data).expanduser().resolve()
    if not data_path.exists():
        raise FileNotFoundError(data_path)

    out_dir = Path(args.out).expanduser() if args.out else OUTPUT_ROOT / f"operational_screening_{args.mode}"
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("Operational Uncertainty Coarse Screening")
    print("=" * 78)
    print(f"Data       : {data_path}")
    print(f"Output     : {out_dir}")
    print(f"Mode       : {args.mode}")
    print(f"Candidates : {', '.join(args.candidates)}")
    print(f"Levels     : {', '.join(args.levels)}")
    print(f"Seeds      : {', '.join(map(str, args.seeds))}")
    print(f"GA         : pop={args.pop}, gens={args.gens}")
    print("=" * 78)

    base_inputs = load_inputs(data_path, expected_batches=args.expected_batches)
    print(f"[INIT] Loaded {len(base_inputs.batches)} batches.")

    base_path_lib = build_path_library(base_inputs, seed=0)
    baseline_by_seed: dict[int, tuple[B.Individual, dict[str, Any]]] = {}
    rows: list[dict[str, Any]] = []

    for seed in args.seeds:
        print("\n" + "=" * 78)
        print(f"[BASELINE] seed={seed}")
        print("=" * 78)
        baseline_ind, baseline_meta = run_one_model(
            base_inputs,
            base_path_lib,
            pop_size=args.pop,
            generations=args.gens,
            seed=seed,
        )
        baseline_metrics = solution_metrics(baseline_ind, base_inputs)
        baseline_by_seed[seed] = (baseline_ind, baseline_metrics)
        rows.append(build_row(
            candidate="baseline",
            level="baseline",
            seed=seed,
            params={"baseline": True},
            metrics=baseline_metrics,
            meta=baseline_meta,
            baseline_metrics=baseline_metrics,
            route_change=0.0,
        ))

    for candidate in args.candidates:
        for level in args.levels:
            for seed in args.seeds:
                print("\n" + "=" * 78)
                print(f"[SCREEN] {candidate} {CANDIDATES[candidate]['name']} | level={level} | seed={seed}")
                print("=" * 78)
                scenario_inputs, scenario_params = apply_candidate(
                    base_inputs,
                    candidate=candidate,
                    level=level,
                    seed=seed,
                    disruption_pool=args.disruption_nodes,
                )
                scenario_path_lib = build_path_library(
                    scenario_inputs,
                    seed=_candidate_seed(seed, candidate, level),
                )
                scenario_ind, scenario_meta = run_one_model(
                    scenario_inputs,
                    scenario_path_lib,
                    pop_size=args.pop,
                    generations=args.gens,
                    seed=_candidate_seed(seed, candidate, level),
                )
                scenario_metrics = solution_metrics(scenario_ind, scenario_inputs)
                baseline_ind, baseline_metrics = baseline_by_seed[seed]
                change = route_change_ratio(
                    baseline_ind,
                    scenario_ind,
                    base_inputs,
                    scenario_inputs,
                )
                rows.append(build_row(
                    candidate=candidate,
                    level=level,
                    seed=seed,
                    params=scenario_params,
                    metrics=scenario_metrics,
                    meta=scenario_meta,
                    baseline_metrics=baseline_metrics,
                    route_change=change,
                ))

    results = pd.DataFrame(rows)
    summary, ranking = summarize_results(
        rows,
        kpi_threshold=args.kpi_threshold,
        route_threshold=args.route_threshold,
    )

    results_csv = out_dir / "screening_results.csv"
    summary_csv = out_dir / "screening_summary.csv"
    ranking_csv = out_dir / "screening_ranking.csv"
    results.to_csv(results_csv, index=False)
    summary.to_csv(summary_csv, index=False)
    ranking.to_csv(ranking_csv, index=False)

    manifest = {
        "data": str(data_path),
        "output": str(out_dir),
        "mode": args.mode,
        "pop": args.pop,
        "gens": args.gens,
        "seeds": args.seeds,
        "levels": args.levels,
        "candidates": args.candidates,
        "expected_batches": args.expected_batches,
        "kpi_threshold": args.kpi_threshold,
        "route_threshold": args.route_threshold,
        "candidate_definitions": CANDIDATES,
        "level_definitions": LEVELS,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    write_report(out_dir, data_path, args, summary, ranking)

    print("\n" + "=" * 78)
    print("[DONE] Operational coarse screening complete.")
    print(f"[EXPORT] {results_csv}")
    print(f"[EXPORT] {summary_csv}")
    print(f"[EXPORT] {ranking_csv}")
    print(f"[EXPORT] {out_dir / 'screening_report.md'}")
    if not ranking.empty:
        print("\n[RANKING]")
        show_cols = [
            "candidate",
            "candidate_label",
            "max_kpi_score",
            "max_route_change_ratio",
            "retain_for_refined_screening",
        ]
        print(ranking[show_cols].to_string(index=False))


if __name__ == "__main__":
    main()
