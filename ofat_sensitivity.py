#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Batch OFAT sensitivity runner for the V1 multimodal baseline.

This script keeps the baseline model deterministic and unchanged. It creates
temporary copies of data_expanded.xlsx, perturbs one parameter group at a time,
runs baseline3_v1.py's NSGA-II functions, and exports comparable summaries.

Typical quick run:
    python ofat_sensitivity.py --preset congestion --levels -0.3 -0.15 0 0.15 0.3

Full parameter sweep:
    python ofat_sensitivity.py --preset full --seeds 2026 2027 2028
"""
import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

import baseline3_v1 as B


BREAK_OF_GAUGE_NODES = set(B.BREAK_OF_GAUGE_NODES)
TRACK_NODES = [
    "Khorgos", "Alashankou", "Erenhot", "Manzhouli", "Zabaykalsk",
    "Dostyk", "Altynkol", "Brest", "Malaszewicze",
]

GROUPS_CONGESTION = [
    "P1_border_delay_rail",
    "P2_border_capacity",
    "P3_background_flow",
]

GROUPS_FULL = GROUPS_CONGESTION + [
    "P4_transfer_time",
    "P5_transfer_cost",
    "P6_service_frequency",
    "P7_transport_cost",
    "P8_emission_factor",
    "P9_quantity",
    "P10_latest_time",
]


def read_workbook(path: str) -> Dict[str, pd.DataFrame]:
    xls = pd.ExcelFile(path)
    return {sheet: pd.read_excel(xls, sheet) for sheet in xls.sheet_names}


def write_workbook(sheets: Dict[str, pd.DataFrame], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for sheet, df in sheets.items():
            df.to_excel(writer, sheet_name=sheet, index=False)


def node_mask(df: pd.DataFrame) -> pd.Series:
    return df["EnglishName"].astype(str).str.strip().isin(BREAK_OF_GAUGE_NODES)


def multiply_existing(df: pd.DataFrame, column: str, mask: pd.Series, factor: float) -> None:
    if column not in df.columns:
        return
    df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0.0).astype(float)
    vals = df.loc[mask, column]
    df.loc[mask, column] = vals * factor


def apply_scenario(sheets: Dict[str, pd.DataFrame], group: str, factor: float) -> None:
    """Apply one OFAT perturbation in-place."""
    if group == "baseline":
        return

    if group == "P1_border_delay_rail":
        df = sheets["Node_Border"]
        multiply_existing(df, "BorderDelay_rail_h", node_mask(df), factor)
        return

    if group == "P2_border_capacity":
        df = sheets["Node_Border"]
        multiply_existing(df, "BorderCapacity_TEUday", node_mask(df), factor)
        return

    if group == "P3_background_flow":
        df = sheets["Node_Border"]
        multiply_existing(df, "BackgroundFlow_TEUday", node_mask(df), factor)
        return

    if group == "P4_transfer_time":
        if "Transshipment" in sheets and "TransferTime_h" in sheets["Transshipment"].columns:
            sheets["Transshipment"]["TransferTime_h"] = (
                pd.to_numeric(sheets["Transshipment"]["TransferTime_h"], errors="coerce")
                .fillna(0.0) * factor
            )
        return

    if group == "P5_transfer_cost":
        if "Transshipment" in sheets and "TransferCost_USD_per_TEU" in sheets["Transshipment"].columns:
            sheets["Transshipment"]["TransferCost_USD_per_TEU"] = (
                pd.to_numeric(sheets["Transshipment"]["TransferCost_USD_per_TEU"], errors="coerce")
                .fillna(0.0) * factor
            )
        return

    if group == "P6_service_frequency":
        if "Timetable" in sheets:
            df = sheets["Timetable"]
            modes = df["Mode"].astype(str).str.lower().isin(["rail", "water"])
            if "Frequency_per_week" in df.columns:
                df["Frequency_per_week"] = (
                    pd.to_numeric(df["Frequency_per_week"], errors="coerce")
                    .fillna(1.0)
                    .astype(float)
                )
                freq = df.loc[modes, "Frequency_per_week"]
                new_freq = np.maximum(freq * factor, 0.1)
                df.loc[modes, "Frequency_per_week"] = new_freq
                if "Headway_Hours" in df.columns:
                    df["Headway_Hours"] = (
                        pd.to_numeric(df["Headway_Hours"], errors="coerce")
                        .fillna(168.0)
                        .astype(float)
                    )
                    df.loc[modes, "Headway_Hours"] = 168.0 / new_freq
            elif "Headway_Hours" in df.columns:
                # Higher factor means better frequency, therefore shorter headway.
                df["Headway_Hours"] = (
                    pd.to_numeric(df["Headway_Hours"], errors="coerce")
                    .fillna(168.0)
                    .astype(float)
                )
                headway = df.loc[modes, "Headway_Hours"]
                df.loc[modes, "Headway_Hours"] = headway / max(factor, 0.1)
        return

    if group == "P7_transport_cost":
        if "Arcs_All" in sheets and "Cost_$_per_km" in sheets["Arcs_All"].columns:
            sheets["Arcs_All"]["Cost_$_per_km"] = (
                pd.to_numeric(sheets["Arcs_All"]["Cost_$_per_km"], errors="coerce")
                .fillna(0.0) * factor
            )
        return

    if group == "P8_emission_factor":
        if "Arcs_All" in sheets and "Emission_gCO2_per_tkm" in sheets["Arcs_All"].columns:
            sheets["Arcs_All"]["Emission_gCO2_per_tkm"] = (
                pd.to_numeric(sheets["Arcs_All"]["Emission_gCO2_per_tkm"], errors="coerce")
                .fillna(0.0) * factor
            )
        if "Emission_Factors" in sheets:
            df = sheets["Emission_Factors"]
            for col in ["gCO2_per_tonne_km", "gCO2_per_TEU_km_assuming10t"]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0) * factor
        return

    if group == "P9_quantity":
        if "Batches" in sheets and "QuantityTEU" in sheets["Batches"].columns:
            sheets["Batches"]["QuantityTEU"] = (
                pd.to_numeric(sheets["Batches"]["QuantityTEU"], errors="coerce")
                .fillna(0.0) * factor
            )
        return

    if group == "P10_latest_time":
        if "Batches" in sheets and "LT" in sheets["Batches"].columns:
            sheets["Batches"]["LT"] = (
                pd.to_numeric(sheets["Batches"]["LT"], errors="coerce")
                .fillna(0.0) * factor
            )
        return

    raise ValueError(f"Unknown parameter group: {group}")


def load_context(workbook: str) -> Dict:
    (node_names, node_region, node_hold_cost, node_proc_cost, node_trans_cost,
     arcs, timetables, batches,
     wc, we, carbon_tax_map, emission_factor_map, mode_speeds_map,
     trans_map, border_delay_map, theta_rm) = B.load_network_from_extended(workbook)
    return dict(
        node_names=node_names, node_region=node_region,
        node_hold_cost=node_hold_cost, node_proc_cost=node_proc_cost,
        node_trans_cost=node_trans_cost, arcs=arcs, timetables=timetables,
        batches=batches, wc=wc, we=we, carbon_tax_map=carbon_tax_map,
        emission_factor_map=emission_factor_map, mode_speeds_map=mode_speeds_map,
        trans_map=trans_map, border_delay_map=border_delay_map, theta_rm=theta_rm,
    )


def pick_representative(population, pareto):
    """Use feasible min-cost Pareto solution; otherwise min-penalty solution."""
    feasible = [ind for ind in pareto if ind.feasible]
    if feasible:
        return min(feasible, key=lambda x: x.objectives[0])
    return min(population, key=lambda x: x.penalty)


def allocation_signature(ind, batches) -> Dict[int, Tuple]:
    sig = {}
    for batch in batches:
        key = (batch.origin, batch.destination, batch.batch_id)
        paths = []
        for alloc in ind.od_allocations.get(key, []):
            if alloc.share <= 1e-9:
                continue
            paths.append((
                round(float(alloc.share), 3),
                tuple(alloc.path.nodes),
                tuple(alloc.path.modes),
            ))
        sig[int(batch.batch_id)] = tuple(sorted(paths))
    return sig


def route_change_ratio(sig, baseline_sig) -> float:
    if not baseline_sig:
        return 0.0
    changed = 0
    total = 0
    for bid, base_paths in baseline_sig.items():
        total += 1
        if sig.get(bid) != base_paths:
            changed += 1
    return changed / max(total, 1)


def summarize(ind, batches, scenario: Dict, baseline_sig=None) -> Dict:
    bf = getattr(ind, "border_flow", {}) or {}
    bu = getattr(ind, "border_util", {}) or {}
    vb = ind.vio_breakdown or {}
    sig = allocation_signature(ind, batches)
    total_demand = float(sum(batch.quantity for batch in batches))
    tracked_flow = sum(float(bf.get(node, 0.0)) for node in TRACK_NODES)

    row = {
        "scenario_id": scenario["scenario_id"],
        "group": scenario["group"],
        "delta": scenario["delta"],
        "factor": scenario["factor"],
        "seed": scenario["seed"],
        "feasible": bool(ind.feasible),
        "penalty": float(ind.penalty),
        "cost": float(ind.objectives[0]),
        "emission_gCO2": float(ind.objectives[1]),
        "time_h": float(ind.objectives[2]),
        "max_border_util": float(vb.get("max_border_util", 0.0)),
        "border_cap_excess": float(vb.get("border_cap_excess", 0.0)),
        "arc_cap_excess": float(vb.get("cap_excess", 0.0)),
        "late_teu_h": float(vb.get("late_teu_h", 0.0)),
        "route_change_ratio": route_change_ratio(sig, baseline_sig),
        "total_demand": total_demand,
        "share_other": max(0.0, total_demand - tracked_flow) / total_demand if total_demand else 0.0,
    }
    for node in TRACK_NODES:
        flow = float(bf.get(node, 0.0))
        row[f"flow_{node}"] = flow
        row[f"share_{node}"] = flow / total_demand if total_demand else 0.0
        row[f"util_{node}"] = float(bu.get(node, 0.0))
    return row


def run_one(workbook: str, scenario: Dict, args, baseline_sig=None) -> Tuple[Dict, Dict]:
    ctx = load_context(workbook)
    batches = ctx["batches"][:args.batches] if args.batches else ctx["batches"]

    B.DFS_MAX_PATHS_PER_OD = args.dfs_paths
    B.PATHS_TOPK_PER_CRITERION = args.topk
    B.PATH_LIB_CAP_TOTAL = args.path_cap

    tt_dict = B.build_timetable_dict(ctx["timetables"])
    arc_lookup = B.build_arc_lookup(ctx["arcs"])

    random.seed(args.path_seed)
    np.random.seed(args.path_seed)
    path_lib = B.build_path_library(
        ctx["node_names"], ctx["node_region"], ctx["arcs"], batches, tt_dict, arc_lookup
    )
    B.sanity_check_path_lib(batches, path_lib)

    random.seed(scenario["seed"])
    np.random.seed(scenario["seed"])
    out = B.run_nsga2(
        ctx["node_names"], ctx["node_region"], ctx["node_hold_cost"],
        ctx["node_proc_cost"], ctx["node_trans_cost"],
        ctx["arcs"], ctx["timetables"], batches,
        ctx["wc"], ctx["we"], ctx["carbon_tax_map"], ctx["emission_factor_map"],
        ctx["mode_speeds_map"], ctx["trans_map"], ctx["border_delay_map"],
        ctx["theta_rm"], path_lib, pop_size=args.pop, generations=args.gens,
    )
    population, pareto = out[0], out[1]
    rep = pick_representative(population, pareto)

    eval_kwargs = dict(
        node_hold_cost=ctx["node_hold_cost"],
        node_proc_cost=ctx["node_proc_cost"],
        carbon_tax_map=ctx["carbon_tax_map"],
        trans_map=ctx["trans_map"],
        border_delay_map=ctx["border_delay_map"],
        theta_rm=ctx["theta_rm"],
        node_trans_cost=ctx["node_trans_cost"],
    )
    B.evaluate_individual(rep, batches, ctx["arcs"], tt_dict, ctx["wc"], ctx["we"], **eval_kwargs)

    row = summarize(rep, batches, scenario, baseline_sig=baseline_sig)
    row["pareto_size"] = len(pareto)
    row["n_feasible_pop"] = sum(1 for ind in population if ind.feasible)
    row["runtime_s"] = float(out[-1])
    return row, allocation_signature(rep, batches)


def build_scenarios(groups: Iterable[str], levels: List[float], seeds: List[int]) -> List[Dict]:
    scenarios = []
    for seed in seeds:
        scenarios.append({
            "scenario_id": f"baseline_seed{seed}",
            "group": "baseline",
            "delta": 0.0,
            "factor": 1.0,
            "seed": seed,
        })
        for group in groups:
            for delta in levels:
                if abs(delta) < 1e-12:
                    continue
                factor = 1.0 + float(delta)
                scenarios.append({
                    "scenario_id": f"{group}_{delta:+.2f}_seed{seed}".replace("+", "p").replace("-", "m"),
                    "group": group,
                    "delta": float(delta),
                    "factor": factor,
                    "seed": seed,
                })
    return scenarios


def main():
    parser = argparse.ArgumentParser(description="Run OFAT sensitivity experiments for baseline3_v1.")
    parser.add_argument("--data", default="data_expanded.xlsx")
    parser.add_argument("--out", default="ofat_out")
    parser.add_argument("--preset", choices=["congestion", "full"], default="congestion")
    parser.add_argument("--groups", nargs="*", default=None,
                        help="Optional explicit groups, overriding --preset.")
    parser.add_argument("--levels", nargs="+", type=float,
                        default=[-0.30, -0.15, 0.0, 0.15, 0.30])
    parser.add_argument("--seeds", nargs="+", type=int, default=[2026])
    parser.add_argument("--batches", type=int, default=14)
    parser.add_argument("--pop", type=int, default=140)
    parser.add_argument("--gens", type=int, default=140)
    parser.add_argument("--dfs-paths", type=int, default=120)
    parser.add_argument("--topk", type=int, default=12)
    parser.add_argument("--path-cap", type=int, default=36)
    parser.add_argument("--path-seed", type=int, default=0)
    parser.add_argument("--keep-workbooks", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out)
    workbook_dir = out_dir / "workbooks"
    out_dir.mkdir(parents=True, exist_ok=True)

    groups = args.groups if args.groups else (
        GROUPS_CONGESTION if args.preset == "congestion" else GROUPS_FULL
    )
    scenarios = build_scenarios(groups, args.levels, args.seeds)
    base_sheets = read_workbook(args.data)

    rows = []
    baseline_sigs: Dict[int, Dict] = {}
    started = time.perf_counter()

    manifest = {
        "data": args.data,
        "preset": args.preset,
        "groups": list(groups),
        "levels": args.levels,
        "seeds": args.seeds,
        "batches": args.batches,
        "pop": args.pop,
        "gens": args.gens,
        "dfs_paths": args.dfs_paths,
        "topk": args.topk,
        "path_cap": args.path_cap,
        "created_at_unix": time.time(),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    for idx, scenario in enumerate(scenarios, start=1):
        print(f"\n=== [{idx}/{len(scenarios)}] {scenario['scenario_id']} ===")
        sheets = {name: df.copy(deep=True) for name, df in base_sheets.items()}
        apply_scenario(sheets, scenario["group"], scenario["factor"])
        wb_path = workbook_dir / f"{scenario['scenario_id']}.xlsx"
        write_workbook(sheets, wb_path)

        baseline_sig = None if scenario["group"] == "baseline" else baseline_sigs.get(scenario["seed"])
        row, sig = run_one(str(wb_path), scenario, args, baseline_sig=baseline_sig)
        rows.append(row)
        if scenario["group"] == "baseline":
            baseline_sigs[scenario["seed"]] = sig

        pd.DataFrame(rows).to_csv(out_dir / "ofat_results_partial.csv", index=False)
        (out_dir / "latest_row.json").write_text(json.dumps(row, indent=2), encoding="utf-8")

        print(
            f"[DONE] feasible={row['feasible']} cost={row['cost']:.3e} "
            f"time={row['time_h']:.1f} max_util={row['max_border_util']:.3f} "
            f"route_change={row['route_change_ratio']:.2%}"
        )

        if not args.keep_workbooks:
            try:
                wb_path.unlink()
            except OSError:
                pass

    df = pd.DataFrame(rows)
    result_csv = out_dir / "ofat_results.csv"
    df.to_csv(result_csv, index=False)

    flow_cols = [c for c in df.columns if c.startswith("flow_")]
    util_cols = [c for c in df.columns if c.startswith("util_")]
    summary_cols = [
        "cost", "emission_gCO2", "time_h", "max_border_util",
        "border_cap_excess", "late_teu_h", "route_change_ratio",
    ]
    summary = (
        df.groupby(["group", "delta"], dropna=False)[summary_cols]
        .mean(numeric_only=True)
        .reset_index()
    )
    summary.to_csv(out_dir / "ofat_group_summary.csv", index=False)

    node_summary = (
        df.groupby(["group", "delta"], dropna=False)[flow_cols + util_cols]
        .mean(numeric_only=True)
        .reset_index()
    )
    node_summary.to_csv(out_dir / "ofat_node_summary.csv", index=False)

    elapsed = time.perf_counter() - started
    print(f"\n=== OFAT complete in {elapsed:.1f}s ===")
    print(f"Results: {result_csv}")
    print(f"Summary: {out_dir / 'ofat_group_summary.csv'}")
    print(f"Node summary: {out_dir / 'ofat_node_summary.csv'}")


if __name__ == "__main__":
    main()
