#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Refined operational uncertainty experiments after coarse screening.

This is the second-stage experiment. It focuses on the coarse-screened core
operational uncertainties and uses a fixed path-library topology across all
scenarios so route-change metrics are not inflated by repeated random path
search.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import baseline3 as B
import run_operational_screening as S


ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = ROOT / "outputs"


REFINED_LEVELS: dict[str, dict[str, float]] = {
    "r1": {
        "intensity": 1,
        "boundary_capacity_loss": 0.08,
        "boundary_background_growth": 0.05,
        "line_time_spread": 0.05,
        "border_time_spread": 0.10,
        "terminal_time_spread": 0.10,
        "capacity_loss": 0.08,
        "demand_growth": 0.05,
        "disruption_capacity_factor": 0.70,
        "disruption_delay_spread": 0.15,
        "disruption_nodes": 1,
    },
    "r2": {
        "intensity": 2,
        "boundary_capacity_loss": 0.15,
        "boundary_background_growth": 0.12,
        "line_time_spread": 0.12,
        "border_time_spread": 0.22,
        "terminal_time_spread": 0.18,
        "capacity_loss": 0.15,
        "demand_growth": 0.12,
        "disruption_capacity_factor": 0.50,
        "disruption_delay_spread": 0.30,
        "disruption_nodes": 1,
    },
    "r3": {
        "intensity": 3,
        "boundary_capacity_loss": 0.25,
        "boundary_background_growth": 0.25,
        "line_time_spread": 0.18,
        "border_time_spread": 0.35,
        "terminal_time_spread": 0.30,
        "capacity_loss": 0.25,
        "demand_growth": 0.25,
        "disruption_capacity_factor": 0.25,
        "disruption_delay_spread": 0.50,
        "disruption_nodes": 2,
    },
    "r4": {
        "intensity": 4,
        "boundary_capacity_loss": 0.35,
        "boundary_background_growth": 0.40,
        "line_time_spread": 0.25,
        "border_time_spread": 0.50,
        "terminal_time_spread": 0.42,
        "capacity_loss": 0.35,
        "demand_growth": 0.35,
        "disruption_capacity_factor": 0.10,
        "disruption_delay_spread": 0.70,
        "disruption_nodes": 2,
    },
    "r5": {
        "intensity": 5,
        "boundary_capacity_loss": 0.45,
        "boundary_background_growth": 0.60,
        "line_time_spread": 0.35,
        "border_time_spread": 0.70,
        "terminal_time_spread": 0.55,
        "capacity_loss": 0.45,
        "demand_growth": 0.45,
        "disruption_capacity_factor": 0.02,
        "disruption_delay_spread": 1.00,
        "disruption_nodes": 3,
    },
}


MODES: dict[str, dict[str, Any]] = {
    "smoke": {
        "pop": 24,
        "gens": 6,
        "seeds": [2026],
        "levels": ["r3"],
        "candidates": ["M1", "M2"],
        "combo_levels": [],
    },
    "quick": {
        "pop": 70,
        "gens": 30,
        "seeds": [2026, 7],
        "levels": ["r1", "r2", "r3", "r4", "r5"],
        "candidates": ["M1", "M2"],
        "combo_levels": ["r3"],
    },
    "refined": {
        "pop": 120,
        "gens": 80,
        "seeds": [2026, 7, 99, 1000],
        "levels": ["r1", "r2", "r3", "r4", "r5"],
        "candidates": ["M1", "M2"],
        "combo_levels": ["r3", "r5"],
    },
    "full": {
        "pop": 180,
        "gens": 150,
        "seeds": [2026, 7, 99, 1000, 42],
        "levels": ["r1", "r2", "r3", "r4", "r5"],
        "candidates": ["M1", "M2", "B1", "B2"],
        "combo_levels": ["r3", "r5"],
    },
}


DEFAULT_COMBINATIONS = [
    ("M1", "M2"),
]


SCENARIO_DEFS: dict[str, dict[str, str]] = {
    "M1": {
        "label": "Boundary bottleneck congestion",
        "role": "core",
        "description": (
            "Border capacity pressure creates endogenous processing delay "
            "through a BPR congestion function; this merges C4 and C2 into "
            "one mechanism."
        ),
    },
    "M2": {
        "label": "Corridor/link disruption",
        "role": "core",
        "description": "Recurrent link, node or corridor disruption.",
    },
    "B1": {
        "label": "Line-haul travel time benchmark",
        "role": "benchmark",
        "description": "Benchmark for established travel-time reliability uncertainty.",
    },
    "B2": {
        "label": "Shipment volume benchmark",
        "role": "benchmark",
        "description": "Benchmark for established demand/volume uncertainty.",
    },
}

BPR_ALPHA = 0.15
BPR_BETA = 4.0


def patch_refined_levels() -> None:
    S.LEVELS = REFINED_LEVELS


def scenario_seed(seed: int, scenario_id: str, level: str) -> int:
    token = sum((idx + 1) * ord(ch) for idx, ch in enumerate(scenario_id))
    return int(seed + 1009 * token + 9173 * list(REFINED_LEVELS).index(level))


def path_library_signatures(path_lib: dict) -> dict[tuple[str, str], list[tuple[tuple[str, ...], tuple[str, ...]]]]:
    signatures = {}
    for od, paths in path_lib.items():
        signatures[od] = [
            (tuple(path.nodes), tuple(path.modes))
            for path in paths
        ]
    return signatures


def rebuild_path_library_from_signatures(
    inputs: S.ModelInputs,
    signatures: dict[tuple[str, str], list[tuple[tuple[str, ...], tuple[str, ...]]]],
) -> dict:
    tt_dict = B.build_timetable_dict(inputs.timetables)
    arc_lookup = B.build_arc_lookup(inputs.arcs)
    rebuilt = {}
    next_pid = 0
    for od, sigs in signatures.items():
        origin, destination = od
        paths = []
        seen = set()
        for nodes, modes in sigs:
            if (nodes, modes) in seen:
                continue
            seen.add((nodes, modes))
            path = B.rebuild_path_from_nodes_modes(
                origin,
                destination,
                list(nodes),
                list(modes),
                tt_dict,
                arc_lookup,
                allow_road_fallback=False,
            )
            if path is None:
                continue
            path.path_id = next_pid
            next_pid += 1
            paths.append(path)
        if paths:
            rebuilt[od] = paths
    B.sanity_check_path_lib(inputs.batches, rebuilt)
    return rebuilt


def bpr_delay_multiplier(utilisation: float, alpha: float = BPR_ALPHA, beta: float = BPR_BETA) -> float:
    u = max(0.0, float(utilisation))
    return float(1.0 + alpha * (u ** beta))


def apply_boundary_bottleneck_congestion(
    inputs: S.ModelInputs,
    level: str,
    cascade_nodes: list[str] | None = None,
) -> tuple[S.ModelInputs, dict[str, Any]]:
    cfg = REFINED_LEVELS[level]
    capacity_loss = float(cfg["boundary_capacity_loss"])
    background_growth = float(cfg["boundary_background_growth"])
    cascade_nodes = cascade_nodes or []
    active_nodes = sorted(
        n for n in B.BREAK_OF_GAUGE_NODES
        if n in inputs.border_capacity or any(k[0] == n for k in inputs.border_delay_map)
    )

    cascade_factor = 0.0
    if cascade_nodes:
        disrupted_border_count = sum(1 for n in cascade_nodes if n in active_nodes)
        cascade_factor = 0.06 * float(cfg["intensity"]) * max(1, disrupted_border_count)

    node_params: dict[str, dict[str, float]] = {}
    for node in active_nodes:
        base_capacity = float(inputs.border_capacity.get(node, 0.0))
        if base_capacity > 0.0:
            effective_capacity = max(1.0, base_capacity * max(0.02, 1.0 - capacity_loss))
            inputs.border_capacity[node] = effective_capacity
        else:
            effective_capacity = 0.0

        base_background = float(inputs.background_flow.get(node, 0.0))
        bg_multiplier = 1.0 + background_growth
        if cascade_factor > 0.0 and node not in cascade_nodes:
            bg_multiplier += cascade_factor
        effective_background = base_background * bg_multiplier
        if effective_background > 0.0:
            inputs.background_flow[node] = effective_background

        utilisation = (
            effective_background / effective_capacity
            if effective_capacity > 0.0 else 0.0
        )
        delay_multiplier = bpr_delay_multiplier(utilisation)
        node_params[node] = {
            "base_capacity": base_capacity,
            "effective_capacity": effective_capacity,
            "base_background": base_background,
            "effective_background": effective_background,
            "utilisation": utilisation,
            "bpr_delay_multiplier": delay_multiplier,
        }

    new_delay_map = {}
    for key, value in inputs.border_delay_map.items():
        node, _mode = key
        multiplier = node_params.get(node, {}).get("bpr_delay_multiplier", 1.0)
        new_delay_map[key] = float(value) * multiplier
    inputs.border_delay_map = new_delay_map

    utilisations = [p["utilisation"] for p in node_params.values()]
    multipliers = [p["bpr_delay_multiplier"] for p in node_params.values()]
    params = {
        "mechanism": "M1_boundary_bottleneck_congestion",
        "capacity_loss": capacity_loss,
        "background_growth": background_growth,
        "bpr_alpha": BPR_ALPHA,
        "bpr_beta": BPR_BETA,
        "cascade_nodes": list(cascade_nodes),
        "cascade_background_factor": cascade_factor,
        "mean_utilisation": float(np.mean(utilisations)) if utilisations else 0.0,
        "max_utilisation": float(np.max(utilisations)) if utilisations else 0.0,
        "mean_bpr_delay_multiplier": float(np.mean(multipliers)) if multipliers else 1.0,
        "max_bpr_delay_multiplier": float(np.max(multipliers)) if multipliers else 1.0,
        "node_params": node_params,
    }
    return inputs, params


def apply_atomic_scenario(
    inputs: S.ModelInputs,
    scenario_token: str,
    level: str,
    seed: int,
    disruption_nodes: list[str],
) -> tuple[S.ModelInputs, dict[str, Any]]:
    if scenario_token == "M1":
        return apply_boundary_bottleneck_congestion(inputs, level)
    if scenario_token == "M2":
        return S.apply_candidate(inputs, "C6", level, seed, disruption_nodes)
    if scenario_token == "B1":
        return S.apply_candidate(inputs, "C1", level, seed, disruption_nodes)
    if scenario_token == "B2":
        return S.apply_candidate(inputs, "C5", level, seed, disruption_nodes)
    if scenario_token in S.CANDIDATES:
        return S.apply_candidate(inputs, scenario_token, level, seed, disruption_nodes)
    raise ValueError(f"Unknown refined scenario token: {scenario_token}")


def apply_scenario(
    base_inputs: S.ModelInputs,
    scenario_candidates: list[str],
    level: str,
    seed: int,
    disruption_nodes: list[str],
) -> tuple[S.ModelInputs, dict[str, Any]]:
    patch_refined_levels()
    scenario_inputs = deepcopy(base_inputs)
    params: dict[str, Any] = {
        "level": level,
        "intensity": REFINED_LEVELS[level]["intensity"],
        "candidates": scenario_candidates,
        "candidate_params": {},
    }
    ordered_candidates = [c for c in scenario_candidates if c != "M1"]
    if "M1" in scenario_candidates:
        ordered_candidates.append("M1")

    cascade_nodes: list[str] = []
    for candidate in ordered_candidates:
        if candidate == "M1":
            scenario_inputs, candidate_params = apply_boundary_bottleneck_congestion(
                scenario_inputs,
                level=level,
                cascade_nodes=cascade_nodes if "M2" in scenario_candidates else None,
            )
        else:
            scenario_inputs, candidate_params = apply_atomic_scenario(
                scenario_inputs,
                scenario_token=candidate,
                level=level,
                seed=seed,
                disruption_nodes=disruption_nodes,
            )
            if candidate == "M2":
                cascade_nodes = list(candidate_params.get("disrupted_nodes", []))
        params["candidate_params"][candidate] = candidate_params
    return scenario_inputs, params


def candidate_label(scenario_id: str) -> str:
    if scenario_id in SCENARIO_DEFS:
        return SCENARIO_DEFS[scenario_id]["label"]
    if "+" not in scenario_id:
        return S.CANDIDATES[scenario_id]["label"]
    return " + ".join(candidate_label(c) for c in scenario_id.split("+"))


def scenario_type(scenario_id: str) -> str:
    if scenario_id == "baseline":
        return "baseline"
    return "combination" if "+" in scenario_id else "single"


def path_border_key(path: B.Path) -> str:
    for node in path.nodes:
        if node in B.BREAK_OF_GAUGE_NODES:
            return node
    return "other"


def batch_border_share_map(ind: B.Individual, batch: B.Batch) -> dict[str, float]:
    key = (batch.origin, batch.destination, batch.batch_id)
    out: dict[str, float] = {}
    for alloc in ind.od_allocations.get(key, []):
        border = path_border_key(alloc.path)
        out[border] = out.get(border, 0.0) + float(alloc.share)
    return out


def network_border_share_map(ind: B.Individual, inputs: S.ModelInputs) -> dict[str, float]:
    total = 0.0
    out: dict[str, float] = {}
    for batch in inputs.batches:
        for border, share in batch_border_share_map(ind, batch).items():
            flow = float(batch.quantity) * float(share)
            out[border] = out.get(border, 0.0) + flow
            total += flow
    if total <= 1e-12:
        return out
    return {k: v / total for k, v in out.items()}


def rerouted_volume_share_by_border(
    baseline_ind: B.Individual,
    scenario_ind: B.Individual,
    baseline_inputs: S.ModelInputs,
    scenario_inputs: S.ModelInputs,
) -> float:
    baseline_by_id = {b.batch_id: b for b in baseline_inputs.batches}
    rerouted_teu = 0.0
    total_teu = 0.0
    for batch in scenario_inputs.batches:
        base_batch = baseline_by_id.get(batch.batch_id)
        if base_batch is None:
            continue
        base_map = batch_border_share_map(baseline_ind, base_batch)
        scenario_map = batch_border_share_map(scenario_ind, batch)
        overlap = 0.0
        for border, scenario_share in scenario_map.items():
            overlap += min(float(scenario_share), float(base_map.get(border, 0.0)))
        rerouted = max(0.0, 1.0 - min(1.0, overlap))
        rerouted_teu += float(batch.quantity) * rerouted
        total_teu += float(batch.quantity)
    return float(rerouted_teu / total_teu) if total_teu > 0 else 0.0


def border_share_shift(
    baseline_ind: B.Individual,
    scenario_ind: B.Individual,
    baseline_inputs: S.ModelInputs,
    scenario_inputs: S.ModelInputs,
) -> float:
    base = network_border_share_map(baseline_ind, baseline_inputs)
    scenario = network_border_share_map(scenario_ind, scenario_inputs)
    keys = set(base) | set(scenario)
    return float(sum(abs(float(scenario.get(k, 0.0)) - float(base.get(k, 0.0))) for k in keys))


def decision_metrics(
    baseline_ind: B.Individual,
    scenario_ind: B.Individual,
    baseline_inputs: S.ModelInputs,
    scenario_inputs: S.ModelInputs,
    baseline_metrics: dict[str, Any],
    scenario_metrics: dict[str, Any],
) -> dict[str, float]:
    recovery_cost_delta = float(scenario_metrics["cost"] - baseline_metrics["cost"])
    recovery_cost_delta_rel = S.rel_delta(scenario_metrics["cost"], baseline_metrics["cost"])
    return {
        "path_route_change_ratio": S.route_change_ratio(
            baseline_ind,
            scenario_ind,
            baseline_inputs,
            scenario_inputs,
        ),
        "rerouted_volume_share": rerouted_volume_share_by_border(
            baseline_ind,
            scenario_ind,
            baseline_inputs,
            scenario_inputs,
        ),
        "border_share_shift": border_share_shift(
            baseline_ind,
            scenario_ind,
            baseline_inputs,
            scenario_inputs,
        ),
        "recovery_cost_delta": recovery_cost_delta,
        "recovery_cost_delta_rel": recovery_cost_delta_rel,
    }


def build_refined_row(
    scenario_id: str,
    level: str,
    seed: int,
    params: dict[str, Any],
    metrics: dict[str, Any],
    meta: dict[str, Any],
    baseline_metrics: dict[str, Any],
    decision: dict[str, float],
) -> dict[str, Any]:
    row = {
        "scenario_id": scenario_id,
        "scenario_type": scenario_type(scenario_id),
        "candidate_label": "Baseline" if scenario_id == "baseline" else candidate_label(scenario_id),
        "level": level,
        "intensity": REFINED_LEVELS.get(level, {}).get("intensity", 0),
        "seed": int(seed),
        "route_change_ratio": float(decision.get("path_route_change_ratio", 0.0)),
        "path_route_change_ratio": float(decision.get("path_route_change_ratio", 0.0)),
        "rerouted_volume_share": float(decision.get("rerouted_volume_share", 0.0)),
        "border_share_shift": float(decision.get("border_share_shift", 0.0)),
        "recovery_cost_delta": float(decision.get("recovery_cost_delta", 0.0)),
        "recovery_cost_delta_rel": float(decision.get("recovery_cost_delta_rel", 0.0)),
        **metrics,
        **meta,
        "scenario_params_json": json.dumps(params, ensure_ascii=False, sort_keys=True),
    }
    if scenario_id == "baseline":
        row.update({
            "cost_delta_rel": 0.0,
            "time_delta_rel": 0.0,
            "emission_delta_rel": 0.0,
            "on_time_drop": 0.0,
            "late_h_per_teu_delta": 0.0,
            "p95_lateness_delta_h": 0.0,
        })
    else:
        row.update({
            "cost_delta_rel": S.rel_delta(metrics["cost"], baseline_metrics["cost"]),
            "time_delta_rel": S.rel_delta(metrics["time_h"], baseline_metrics["time_h"]),
            "emission_delta_rel": S.rel_delta(metrics["emission_gCO2"], baseline_metrics["emission_gCO2"]),
            "on_time_drop": float(baseline_metrics["on_time_rate"] - metrics["on_time_rate"]),
            "late_h_per_teu_delta": float(
                metrics["late_h_per_teu"] - baseline_metrics["late_h_per_teu"]
            ),
            "p95_lateness_delta_h": float(
                metrics["p95_lateness_h"] - baseline_metrics["p95_lateness_h"]
            ),
        })
    return row


def refined_kpi_score(row: pd.Series) -> float:
    return float(max(
        abs(float(row["cost_delta_rel_mean"])),
        abs(float(row["time_delta_rel_mean"])),
        max(0.0, float(row["on_time_drop_mean"])),
        abs(float(row["late_h_per_teu_delta_mean"])) / 24.0,
        abs(float(row["p95_lateness_delta_h_mean"])) / 168.0,
        abs(float(row["infeasible_teu_mean"])) / max(1.0, abs(float(row["total_teu_mean"]))),
    ))


def summarize_refined(rows: list[dict[str, Any]]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = pd.DataFrame(rows)
    scenarios = df[df["scenario_id"] != "baseline"].copy()
    if scenarios.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    summary = (
        scenarios
        .groupby(["scenario_id", "scenario_type", "candidate_label", "level", "intensity"], dropna=False)
        .agg(
            runs=("seed", "count"),
            cost_delta_rel_mean=("cost_delta_rel", "mean"),
            time_delta_rel_mean=("time_delta_rel", "mean"),
            emission_delta_rel_mean=("emission_delta_rel", "mean"),
            on_time_drop_mean=("on_time_drop", "mean"),
            on_time_rate_mean=("on_time_rate", "mean"),
            late_h_per_teu_delta_mean=("late_h_per_teu_delta", "mean"),
            p95_lateness_delta_h_mean=("p95_lateness_delta_h", "mean"),
            route_change_ratio_mean=("route_change_ratio", "mean"),
            path_route_change_ratio_mean=("path_route_change_ratio", "mean"),
            rerouted_volume_share_mean=("rerouted_volume_share", "mean"),
            border_share_shift_mean=("border_share_shift", "mean"),
            recovery_cost_delta_mean=("recovery_cost_delta", "mean"),
            recovery_cost_delta_rel_mean=("recovery_cost_delta_rel", "mean"),
            infeasible_teu_mean=("infeasible_teu", "mean"),
            total_teu_mean=("total_teu", "mean"),
            feasible_ratio_mean=("final_feasible_ratio", "mean"),
            runtime_s_mean=("runtime_s", "mean"),
        )
        .reset_index()
        .sort_values(["scenario_type", "scenario_id", "intensity"])
    )
    summary["kpi_score"] = summary.apply(refined_kpi_score, axis=1)

    single = summary[summary["scenario_type"] == "single"].copy()
    ranking_rows = []
    for scenario_id, sub in single.groupby("scenario_id", dropna=False):
        sub = sub.sort_values("intensity")
        kpis = sub["kpi_score"].to_numpy(dtype=float)
        route = sub["rerouted_volume_share_mean"].to_numpy(dtype=float)
        recovery = sub["recovery_cost_delta_rel_mean"].to_numpy(dtype=float)
        monotonic_steps = 0
        if len(kpis) > 1:
            monotonic_steps = int(sum(kpis[i] >= kpis[i - 1] - 1e-9 for i in range(1, len(kpis))))
        monotonicity = monotonic_steps / max(1, len(kpis) - 1)
        ranking_rows.append({
            "scenario_id": scenario_id,
            "candidate_label": str(sub["candidate_label"].iloc[0]),
            "max_kpi_score": float(np.max(kpis)) if len(kpis) else 0.0,
            "mean_kpi_score": float(np.mean(kpis)) if len(kpis) else 0.0,
            "max_rerouted_volume_share": float(np.max(route)) if len(route) else 0.0,
            "mean_rerouted_volume_share": float(np.mean(route)) if len(route) else 0.0,
            "max_recovery_cost_delta_rel": float(np.max(recovery)) if len(recovery) else 0.0,
            "mean_recovery_cost_delta_rel": float(np.mean(recovery)) if len(recovery) else 0.0,
            "monotonicity_score": float(monotonicity),
            "recommended_role": recommended_role(str(scenario_id), float(np.max(kpis)) if len(kpis) else 0.0),
        })
    ranking = pd.DataFrame(ranking_rows)
    if not ranking.empty:
        ranking["role_priority"] = ranking["recommended_role"].map({
            "core": 0,
            "benchmark": 1,
            "secondary": 2,
        }).fillna(9)
        ranking = ranking.sort_values(
            ["role_priority", "max_kpi_score", "max_recovery_cost_delta_rel", "monotonicity_score"],
            ascending=[True, False, False, False],
        )

    combo = summary[summary["scenario_type"] == "combination"].copy()
    if not combo.empty:
        combo = combo.sort_values(["scenario_id", "intensity"])
    return summary, ranking, combo


def recommended_role(scenario_id: str, max_kpi_score: float) -> str:
    if scenario_id in SCENARIO_DEFS:
        return SCENARIO_DEFS[scenario_id]["role"]
    if scenario_id in {"C6"} and max_kpi_score >= 0.02:
        return "core"
    if scenario_id in {"C1", "C5"}:
        return "benchmark"
    return "secondary"


def write_report(
    out_dir: Path,
    args: argparse.Namespace,
    summary: pd.DataFrame,
    ranking: pd.DataFrame,
    combination_summary: pd.DataFrame,
) -> None:
    lines = [
        "# Refined Operational Uncertainty Experiments",
        "",
        "This second-stage experiment follows the coarse screening result. It "
        "uses a fixed path-library topology across baseline and scenarios, so "
        "route-change metrics reflect optimization responses rather than repeated "
        "random path generation.",
        "",
        "## Run Configuration",
        "",
        f"- Mode: `{args.mode}`",
        f"- Data: `{args.data}`",
        f"- Expected batches: `{args.expected_batches}`",
        f"- Population: `{args.pop}`",
        f"- Generations: `{args.gens}`",
        f"- Seeds: `{', '.join(map(str, args.seeds))}`",
        f"- Single-factor candidates: `{', '.join(args.candidates)}`",
        f"- Fine levels: `{', '.join(args.levels)}`",
        f"- Combination levels: `{', '.join(args.combo_levels) if args.combo_levels else 'none'}`",
        "",
        "## Interpretation",
        "",
        "- M1 is boundary bottleneck congestion: border capacity pressure creates endogenous delay through a BPR function.",
        "- M2 is corridor/link disruption.",
        "- M1+M2 is the main cascade scenario: disruption pushes flow toward remaining borders and amplifies bottleneck congestion.",
        "- On-time delivery is retained as a service KPI. Recovery cost is the main economic decision indicator; rerouted volume and border-share shift are supporting flow-reallocation indicators.",
        "- B1 and B2 can be added as benchmarks, because line-haul travel-time and demand uncertainty are established uncertainty classes.",
        "- C3 is not included by default because coarse screening showed the weakest KPI impact.",
        "",
        "## Refined Ranking",
        "",
    ]
    if ranking.empty:
        lines.append("No refined ranking rows were produced.")
    else:
        lines.append(S.markdown_table(ranking))
    if not combination_summary.empty:
        lines.extend(["", "## Combination Scenarios", "", S.markdown_table(combination_summary)])
    if not summary.empty:
        lines.extend(["", "## Single-Factor and Level Summary", "", S.markdown_table(summary)])
    (out_dir / "refined_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run refined operational uncertainty experiments after coarse screening."
    )
    parser.add_argument("--data", default=str(ROOT / "data" / "data_expanded.xlsx"))
    parser.add_argument("--out", default="")
    parser.add_argument("--mode", choices=MODES.keys(), default="refined")
    parser.add_argument("--candidates", nargs="+", default=None)
    parser.add_argument("--levels", nargs="+", default=None, choices=list(REFINED_LEVELS.keys()))
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--pop", type=int, default=0)
    parser.add_argument("--gens", type=int, default=0)
    parser.add_argument("--expected-batches", type=int, default=40)
    parser.add_argument("--combo-levels", nargs="+", default=None, choices=list(REFINED_LEVELS.keys()))
    parser.add_argument("--no-combinations", action="store_true")
    parser.add_argument("--disruption-nodes", nargs="+", default=S.DEFAULT_DISRUPTION_NODES)
    args = parser.parse_args()

    mode_cfg = MODES[args.mode]
    args.pop = args.pop or int(mode_cfg["pop"])
    args.gens = args.gens or int(mode_cfg["gens"])
    args.seeds = args.seeds or list(mode_cfg["seeds"])
    args.levels = args.levels or list(mode_cfg["levels"])
    args.candidates = args.candidates or list(mode_cfg["candidates"])
    args.combo_levels = [] if args.no_combinations else (
        args.combo_levels if args.combo_levels is not None else list(mode_cfg["combo_levels"])
    )
    allowed = set(S.CANDIDATES) | set(SCENARIO_DEFS)
    bad = sorted(set(args.candidates) - allowed)
    if bad:
        raise ValueError(f"Unknown candidates: {bad}")
    return args


def main() -> None:
    patch_refined_levels()
    args = parse_args()
    data_path = Path(args.data).expanduser().resolve()
    if not data_path.exists():
        raise FileNotFoundError(data_path)

    out_dir = Path(args.out).expanduser() if args.out else OUTPUT_ROOT / f"operational_refined_{args.mode}"
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("Refined Operational Uncertainty Experiments")
    print("=" * 78)
    print(f"Data       : {data_path}")
    print(f"Output     : {out_dir}")
    print(f"Mode       : {args.mode}")
    print(f"Candidates : {', '.join(args.candidates)}")
    print(f"Levels     : {', '.join(args.levels)}")
    print(f"Combo Lvls : {', '.join(args.combo_levels) if args.combo_levels else 'none'}")
    print(f"Seeds      : {', '.join(map(str, args.seeds))}")
    print(f"GA         : pop={args.pop}, gens={args.gens}")
    print("=" * 78)

    base_inputs = S.load_inputs(data_path, expected_batches=args.expected_batches)
    print(f"[INIT] Loaded {len(base_inputs.batches)} batches.")

    base_path_lib = S.build_path_library(base_inputs, seed=0)
    signatures = path_library_signatures(base_path_lib)
    print(f"[INIT] Fixed path signatures for {len(signatures)} OD pairs.")

    baseline_by_seed: dict[int, tuple[B.Individual, dict[str, Any], dict[str, Any]]] = {}
    rows: list[dict[str, Any]] = []

    for seed in args.seeds:
        print("\n" + "=" * 78)
        print(f"[BASELINE] seed={seed}")
        print("=" * 78)
        path_lib = rebuild_path_library_from_signatures(base_inputs, signatures)
        baseline_ind, baseline_meta = S.run_one_model(
            base_inputs,
            path_lib,
            pop_size=args.pop,
            generations=args.gens,
            seed=seed,
        )
        baseline_metrics = S.solution_metrics(baseline_ind, base_inputs)
        baseline_by_seed[seed] = (baseline_ind, baseline_metrics, baseline_meta)
        rows.append(build_refined_row(
            scenario_id="baseline",
            level="baseline",
            seed=seed,
            params={"baseline": True},
            metrics=baseline_metrics,
            meta=baseline_meta,
            baseline_metrics=baseline_metrics,
            decision={
                "path_route_change_ratio": 0.0,
                "rerouted_volume_share": 0.0,
                "border_share_shift": 0.0,
                "recovery_cost_delta": 0.0,
                "recovery_cost_delta_rel": 0.0,
            },
        ))

    scenario_specs: list[tuple[str, list[str], list[str]]] = [
        (candidate, [candidate], list(args.levels))
        for candidate in args.candidates
    ]
    for combo in DEFAULT_COMBINATIONS:
        if args.combo_levels and all(c in args.candidates for c in combo):
            scenario_specs.append(("+".join(combo), list(combo), list(args.combo_levels)))

    for scenario_id, scenario_candidates, levels in scenario_specs:
        for level in levels:
            for seed in args.seeds:
                print("\n" + "=" * 78)
                print(f"[REFINED] {scenario_id} | level={level} | seed={seed}")
                print("=" * 78)
                scenario_inputs, params = apply_scenario(
                    base_inputs,
                    scenario_candidates=scenario_candidates,
                    level=level,
                    seed=seed,
                    disruption_nodes=args.disruption_nodes,
                )
                path_lib = rebuild_path_library_from_signatures(scenario_inputs, signatures)
                run_seed = scenario_seed(seed, scenario_id, level)
                scenario_ind, scenario_meta = S.run_one_model(
                    scenario_inputs,
                    path_lib,
                    pop_size=args.pop,
                    generations=args.gens,
                    seed=run_seed,
                )
                scenario_metrics = S.solution_metrics(scenario_ind, scenario_inputs)
                baseline_ind, baseline_metrics, _baseline_meta = baseline_by_seed[seed]
                decision = decision_metrics(
                    baseline_ind,
                    scenario_ind,
                    base_inputs,
                    scenario_inputs,
                    baseline_metrics,
                    scenario_metrics,
                )
                rows.append(build_refined_row(
                    scenario_id=scenario_id,
                    level=level,
                    seed=seed,
                    params=params,
                    metrics=scenario_metrics,
                    meta=scenario_meta,
                    baseline_metrics=baseline_metrics,
                    decision=decision,
                ))

    results = pd.DataFrame(rows)
    summary, ranking, combination_summary = summarize_refined(rows)

    results_path = out_dir / "refined_results.csv"
    summary_path = out_dir / "refined_summary.csv"
    ranking_path = out_dir / "refined_ranking.csv"
    combo_path = out_dir / "refined_combination_summary.csv"
    results.to_csv(results_path, index=False)
    summary.to_csv(summary_path, index=False)
    ranking.to_csv(ranking_path, index=False)
    combination_summary.to_csv(combo_path, index=False)

    manifest = {
        "data": str(data_path),
        "output": str(out_dir),
        "mode": args.mode,
        "pop": args.pop,
        "gens": args.gens,
        "seeds": args.seeds,
        "levels": args.levels,
        "candidates": args.candidates,
        "combo_levels": args.combo_levels,
        "expected_batches": args.expected_batches,
        "fixed_path_library": True,
        "candidate_definitions": S.CANDIDATES,
        "scenario_definitions": SCENARIO_DEFS,
        "level_definitions": REFINED_LEVELS,
        "m1_congestion_formula": "border_delay = base_border_delay * (1 + alpha * utilisation ** beta)",
        "m1_bpr_alpha": BPR_ALPHA,
        "m1_bpr_beta": BPR_BETA,
        "default_combinations": ["+".join(c) for c in DEFAULT_COMBINATIONS],
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    write_report(out_dir, args, summary, ranking, combination_summary)

    print("\n" + "=" * 78)
    print("[DONE] Refined operational experiments complete.")
    print(f"[EXPORT] {results_path}")
    print(f"[EXPORT] {summary_path}")
    print(f"[EXPORT] {ranking_path}")
    print(f"[EXPORT] {combo_path}")
    print(f"[EXPORT] {out_dir / 'refined_report.md'}")
    if not ranking.empty:
        print("\n[REFINED RANKING]")
        show_cols = [
            "scenario_id",
            "candidate_label",
            "max_kpi_score",
            "max_recovery_cost_delta_rel",
            "max_rerouted_volume_share",
            "monotonicity_score",
            "recommended_role",
        ]
        print(ranking[show_cols].to_string(index=False))


if __name__ == "__main__":
    main()
