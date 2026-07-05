#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Summarize route choices from Pareto JSON outputs for the network expansion test.

The script selects one representative solution per scenario using a balanced
normalized objective score, then exports:
  - solution-level objective and feasibility summary
  - border/node flow comparison
  - main route allocation per batch
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = ROOT / "outputs"
SCENARIOS = ("original", "expanded")


def scenario_dir_name(scenario: str, mode: str, tag: str) -> str:
    return f"{scenario}_{tag}_{mode}" if tag else f"{scenario}_{mode}"


def load_points(path: Path) -> list[dict]:
    if not path.exists():
        print(f"[SKIP] Missing {path}")
        return []
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, list) else []


def select_balanced(points: list[dict]) -> dict | None:
    feasible = [p for p in points if p.get("feasible", False)]
    candidates = feasible or points
    if not candidates:
        return None

    keys = ("cost", "emission_gCO2", "time_h")
    ranges = {}
    for key in keys:
        vals = [float(p["objectives"].get(key, 0.0)) for p in candidates]
        lo, hi = min(vals), max(vals)
        ranges[key] = (lo, hi)

    def score(p: dict) -> float:
        total = 0.0
        for key in keys:
            val = float(p["objectives"].get(key, 0.0))
            lo, hi = ranges[key]
            total += 0.0 if hi <= lo else (val - lo) / (hi - lo)
        return total

    return min(candidates, key=score)


def solution_row(scenario: str, point: dict | None) -> dict:
    if not point:
        return {"scenario": scenario, "has_solution": False}
    obj = point.get("objectives", {})
    vio = point.get("vio_breakdown", {})
    return {
        "scenario": scenario,
        "has_solution": True,
        "feasible": bool(point.get("feasible", False)),
        "cost": float(obj.get("cost", 0.0)),
        "emission_gCO2": float(obj.get("emission_gCO2", 0.0)),
        "time_h": float(obj.get("time_h", 0.0)),
        "penalty": float(obj.get("penalty", 0.0)),
        "late_teu_h": float(vio.get("late_teu_h", 0.0)),
        "max_border_util": float(vio.get("max_border_util", 0.0)),
        "border_cap_excess": float(vio.get("border_cap_excess", 0.0)),
    }


def border_rows(scenario: str, point: dict | None) -> list[dict]:
    if not point:
        return []
    flows = point.get("border_flow", {}) or {}
    utils = point.get("border_util", {}) or {}
    rows = []
    for node, flow in sorted(flows.items(), key=lambda kv: (-float(kv[1]), kv[0])):
        rows.append({
            "scenario": scenario,
            "node": node,
            "flow_teu": float(flow),
            "utilisation": float(utils.get(node, 0.0)),
        })
    return rows


def allocation_rows(scenario: str, point: dict | None) -> list[dict]:
    if not point:
        return []
    rows = []
    for alloc in point.get("allocations", []):
        paths = alloc.get("paths", []) or []
        for rank, path in enumerate(sorted(paths, key=lambda p: -float(p.get("share", 0.0))), 1):
            share = float(path.get("share", 0.0))
            qty = float(alloc.get("quantity_teu", 0.0))
            rows.append({
                "scenario": scenario,
                "batch_id": alloc.get("batch_id"),
                "origin": alloc.get("origin"),
                "destination": alloc.get("destination"),
                "quantity_teu": qty,
                "path_rank": rank,
                "share": share,
                "allocated_teu": qty * share,
                "nodes": " -> ".join(path.get("nodes", [])),
                "modes": " -> ".join(path.get("modes", [])),
                "base_cost_per_teu": float(path.get("base_cost_per_teu", 0.0)),
                "base_emission_per_teu": float(path.get("base_emission_per_teu", 0.0)),
                "base_travel_time_h": float(path.get("base_travel_time_h", 0.0)),
            })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize route choices for network scenarios.")
    parser.add_argument("--mode", default="quick")
    parser.add_argument("--tag", default="b20")
    args = parser.parse_args()

    solution_rows = []
    all_border_rows = []
    all_allocation_rows = []

    for scenario in SCENARIOS:
        path = OUTPUT_ROOT / scenario_dir_name(scenario, args.mode, args.tag) / "pareto_points.json"
        point = select_balanced(load_points(path))
        solution_rows.append(solution_row(scenario, point))
        all_border_rows.extend(border_rows(scenario, point))
        all_allocation_rows.extend(allocation_rows(scenario, point))

    out_stem = f"network_routes_{args.tag}_{args.mode}" if args.tag else f"network_routes_{args.mode}"
    out_xlsx = OUTPUT_ROOT / f"{out_stem}.xlsx"
    out_csv = OUTPUT_ROOT / f"{out_stem}_batch_routes.csv"
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    df_solution = pd.DataFrame(solution_rows)
    df_border = pd.DataFrame(all_border_rows)
    df_alloc = pd.DataFrame(all_allocation_rows)

    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
        df_solution.to_excel(writer, sheet_name="SolutionSummary", index=False)
        df_border.to_excel(writer, sheet_name="BorderFlows", index=False)
        df_alloc.to_excel(writer, sheet_name="BatchRoutes", index=False)

    df_alloc.to_csv(out_csv, index=False)
    print(df_solution.to_string(index=False))
    print(f"\n[EXPORT] {out_xlsx}")
    print(f"[EXPORT] {out_csv}")


if __name__ == "__main__":
    main()
