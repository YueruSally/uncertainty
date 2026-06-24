#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bottleneck stress experiment for P1/P2/P3.

This is the focused follow-up to the OFAT sweep. It tests whether congestion
becomes a strong driver once bottleneck utilisation is pushed near the critical
range. The model remains deterministic and uses the min-cost feasible Pareto
solution as the representative solution.

Full run:
    python3 bottleneck_stress.py run

Quick correctness check:
    python3 bottleneck_stress.py run --test

Outputs:
    stress_out/stress_results.csv
    stress_out/stress_summary.csv
    stress_out/stress_interpretation.txt
"""
import argparse
import contextlib
import io
import json
import os
import random
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

import baseline3_v1 as B


DATA = "data_expanded.xlsx"
OUTD = Path("stress_out")
CSV = OUTD / "stress_results.csv"

ENTRY_NODES = ["Khorgos", "Alashankou", "Erenhot", "Manzhouli"]
TRACK_NODES = ENTRY_NODES + ["Dostyk", "Altynkol", "Zabaykalsk", "Brest", "Malaszewicze"]

SCENARIOS = [
    ("BASE", "baseline", 1.00),
    ("P2_border_capacity", "capacity_x0.70", 0.70),
    ("P2_border_capacity", "capacity_x0.50", 0.50),
    ("P2_border_capacity", "capacity_x0.30", 0.30),
    ("P3_background_flow", "background_x1.30", 1.30),
    ("P3_background_flow", "background_x1.50", 1.50),
    ("P3_background_flow", "background_x2.00", 2.00),
    ("P1_border_delay_rail", "rail_delay_x1.30", 1.30),
    ("P1_border_delay_rail", "rail_delay_x1.50", 1.50),
    ("P1_border_delay_rail", "rail_delay_x2.00", 2.00),
]


def parse_args():
    parser = argparse.ArgumentParser(description="Run bottleneck stress tests for P1/P2/P3.")
    parser.add_argument("cmd", nargs="?", default="run", choices=["run", "analyze"])
    parser.add_argument("--test", action="store_true", help="Tiny correctness check.")
    parser.add_argument("--quick", action="store_true", help="Medium run.")
    parser.add_argument("--data", default=DATA)
    parser.add_argument("--out", default=str(OUTD))
    return parser.parse_args()


def config_from_args(args):
    if args.test:
        return dict(batches=12, pop=40, gen=30, seeds=[2026],
                    dfs=60, topk=8, cap=24)
    if args.quick:
        return dict(batches=14, pop=90, gen=80, seeds=[2026, 7],
                    dfs=120, topk=12, cap=36)
    return dict(batches=None, pop=200, gen=150, seeds=[2026, 7, 99],
                dfs=150, topk=15, cap=45)


def load_master(data_file, cfg):
    (node_names, node_region, nhc, npc, ntc, arcs, timetables, batches,
     wc, we, ctm, efm, msm, trans_map, bdm, theta) = B.load_network_from_extended(data_file)
    if cfg["batches"]:
        batches = batches[:cfg["batches"]]

    master = dict(
        node_names=node_names, node_region=node_region, nhc=nhc, npc=npc, ntc=ntc,
        arcs=arcs, timetables=timetables, batches=batches, wc=wc, we=we,
        ctm=ctm, efm=efm, msm=msm, trans_map=trans_map, bdm=bdm, theta=theta,
        cap0=dict(B.BORDER_CAPACITY), bg0=dict(B.BACKGROUND_FLOW),
    )
    return master


def restore_globals(master):
    B.BORDER_CAPACITY.clear()
    B.BORDER_CAPACITY.update(master["cap0"])
    B.BACKGROUND_FLOW.clear()
    B.BACKGROUND_FLOW.update(master["bg0"])


def build_inputs(master, group, factor):
    bdm = dict(master["bdm"])
    if group == "P1_border_delay_rail":
        for key in list(bdm.keys()):
            node, mode = key
            if node in B.BREAK_OF_GAUGE_NODES and mode == "rail":
                bdm[key] = bdm[key] * factor
    elif group == "P2_border_capacity":
        for node in B.BREAK_OF_GAUGE_NODES:
            if node in master["cap0"]:
                B.BORDER_CAPACITY[node] = master["cap0"][node] * factor
    elif group == "P3_background_flow":
        for node in B.BREAK_OF_GAUGE_NODES:
            if node in master["bg0"]:
                B.BACKGROUND_FLOW[node] = master["bg0"][node] * factor
    elif group == "BASE":
        pass
    else:
        raise ValueError(group)
    return bdm


def pick_rep(population, pareto):
    feasible = [ind for ind in pareto if ind.feasible]
    if feasible:
        return min(feasible, key=lambda x: x.objectives[0])
    return min(population, key=lambda x: x.penalty)


def allocation_signature(ind, batches):
    sig = {}
    for batch in batches:
        key = (batch.origin, batch.destination, batch.batch_id)
        paths = []
        for alloc in ind.od_allocations.get(key, []):
            if alloc.share <= 1e-9:
                continue
            paths.append((round(float(alloc.share), 3),
                          tuple(alloc.path.nodes),
                          tuple(alloc.path.modes)))
        sig[int(batch.batch_id)] = tuple(sorted(paths))
    return sig


def route_change_ratio(sig, baseline_sig):
    if not baseline_sig:
        return 0.0
    total = len(baseline_sig)
    changed = sum(1 for bid, base in baseline_sig.items() if sig.get(bid) != base)
    return changed / max(total, 1)


def run_one(master, path_lib, cfg, scenario, seed, baseline_sig=None):
    group, label, factor = scenario
    restore_globals(master)
    bdm = build_inputs(master, group, factor)
    batches = master["batches"]
    tts = master["timetables"]
    tt_dict = B.build_timetable_dict(tts)
    eval_kwargs = dict(
        node_hold_cost=master["nhc"],
        node_proc_cost=master["npc"],
        carbon_tax_map=master["ctm"],
        trans_map=master["trans_map"],
        border_delay_map=bdm,
        theta_rm=master["theta"],
        node_trans_cost=master["ntc"],
    )

    random.seed(seed)
    np.random.seed(seed)
    with contextlib.redirect_stdout(io.StringIO()):
        out = B.run_nsga2(
            master["node_names"], master["node_region"], master["nhc"], master["npc"], master["ntc"],
            master["arcs"], tts, batches, master["wc"], master["we"], master["ctm"], master["efm"],
            master["msm"], master["trans_map"], bdm, master["theta"], path_lib,
            pop_size=cfg["pop"], generations=cfg["gen"],
        )
        population, pareto = out[0], out[1]
        rep = pick_rep(population, pareto)
        B.evaluate_individual(rep, batches, master["arcs"], tt_dict,
                              master["wc"], master["we"], **eval_kwargs)

    bf = getattr(rep, "border_flow", {}) or {}
    bu = getattr(rep, "border_util", {}) or {}
    vb = rep.vio_breakdown or {}
    total_demand = sum(b.quantity for b in batches)
    entry_total = sum(float(bf.get(node, 0.0)) for node in ENTRY_NODES)
    sea_other = max(0.0, total_demand - entry_total)
    sig = allocation_signature(rep, batches)

    row = {
        "group": group,
        "scenario": label,
        "factor": factor,
        "seed": seed,
        "feasible": bool(rep.feasible),
        "cost": float(rep.objectives[0]),
        "emission_gCO2": float(rep.objectives[1]),
        "time_h": float(rep.objectives[2]),
        "penalty": float(rep.penalty),
        "late_teu_h": float(vb.get("late_teu_h", 0.0)),
        "max_border_util": float(vb.get("max_border_util", 0.0)),
        "border_cap_excess": float(vb.get("border_cap_excess", 0.0)),
        "total_demand": float(total_demand),
        "share_sea_other": float(sea_other / total_demand) if total_demand else 0.0,
        "route_change_ratio": route_change_ratio(sig, baseline_sig),
    }
    for node in TRACK_NODES:
        flow = float(bf.get(node, 0.0))
        row[f"flow_{node}"] = round(flow, 1)
        row[f"share_{node}"] = round(flow / total_demand, 4) if total_demand else 0.0
        row[f"util_{node}"] = round(float(bu.get(node, 0.0)), 4)
    return row, sig


def done_keys(csv_path):
    if not csv_path.exists():
        return set()
    df = pd.read_csv(csv_path)
    required = {"group", "scenario", "factor", "seed"}
    if not required.issubset(df.columns):
        stamp = time.strftime("%Y%m%d_%H%M%S")
        backup = csv_path.with_name(f"stress_results_legacy_{stamp}.csv")
        csv_path.rename(backup)
        print(f"[STRESS] Existing incompatible result CSV moved to {backup}")
        return set()
    return set(zip(df["group"], df["scenario"], df["factor"], df["seed"]))


def append_row(csv_path, row):
    pd.DataFrame([row]).to_csv(csv_path, mode="a", header=not csv_path.exists(), index=False)


def run(args):
    cfg = config_from_args(args)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "stress_results.csv"

    manifest = dict(data=args.data, config=cfg, scenarios=SCENARIOS,
                    representative="min-cost feasible Pareto solution",
                    created_at_unix=time.time())
    (out_dir / "stress_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"[STRESS] pop={cfg['pop']} gen={cfg['gen']} batches={cfg['batches'] or 'all'} seeds={cfg['seeds']}")
    master = load_master(args.data, cfg)
    B.DFS_MAX_PATHS_PER_OD = cfg["dfs"]
    B.PATHS_TOPK_PER_CRITERION = cfg["topk"]
    B.PATH_LIB_CAP_TOTAL = cfg["cap"]

    tt_dict = B.build_timetable_dict(master["timetables"])
    arc_lookup = B.build_arc_lookup(master["arcs"])
    random.seed(0)
    np.random.seed(0)
    path_lib = B.build_path_library(master["node_names"], master["node_region"],
                                    master["arcs"], master["batches"], tt_dict, arc_lookup)
    B.sanity_check_path_lib(master["batches"], path_lib)

    baseline_sig = {}
    done = done_keys(csv_path)
    plan = [(scenario, seed) for seed in cfg["seeds"] for scenario in SCENARIOS]
    todo = [(s, seed) for s, seed in plan if (s[0], s[1], s[2], seed) not in done]
    print(f"[STRESS] total={len(plan)} done={len(done)} remaining={len(todo)}")

    for i, (scenario, seed) in enumerate(todo, 1):
        t0 = time.time()
        base_sig = None if scenario[0] == "BASE" else baseline_sig.get(seed)
        row, sig = run_one(master, path_lib, cfg, scenario, seed, baseline_sig=base_sig)
        if scenario[0] == "BASE":
            baseline_sig[seed] = sig
        append_row(csv_path, row)
        print(
            f"[STRESS] ({i}/{len(todo)}) {scenario[1]} seed={seed} "
            f"feas={row['feasible']} max_u={row['max_border_util']:.3f} "
            f"route_change={row['route_change_ratio']:.2%} "
            f"K/A/E/M={row['share_Khorgos']:.2f}/{row['share_Alashankou']:.2f}/"
            f"{row['share_Erenhot']:.2f}/{row['share_Manzhouli']:.2f} "
            f"({time.time() - t0:.0f}s)",
            flush=True,
        )

    analyze(args)


def analyze(args):
    out_dir = Path(args.out)
    csv_path = out_dir / "stress_results.csv"
    if not csv_path.exists():
        print("[STRESS] no results yet")
        return
    df = pd.read_csv(csv_path)
    summary_cols = ["cost", "emission_gCO2", "time_h", "max_border_util",
                    "border_cap_excess", "late_teu_h", "route_change_ratio",
                    "share_sea_other"]
    for node in TRACK_NODES:
        summary_cols.extend([f"flow_{node}", f"share_{node}", f"util_{node}"])
    summary = df.groupby(["group", "scenario", "factor"], dropna=False)[summary_cols].mean(numeric_only=True).reset_index()
    summary.to_csv(out_dir / "stress_summary.csv", index=False)

    text_path = out_dir / "stress_interpretation.txt"
    with text_path.open("w", encoding="utf-8") as f:
        f.write("BOTTLENECK STRESS TEST SUMMARY\n\n")
        f.write("Decision rule:\n")
        f.write("- If max utilisation approaches/exceeds 0.9 and route_change/share movement jumps, congestion is a strong driver.\n")
        f.write("- If utilisation is high but route_change stays low, shift the main uncertainty story toward P10/P9.\n\n")
        f.write(summary[["group", "scenario", "factor", "max_border_util", "route_change_ratio",
                         "share_Khorgos", "share_Alashankou", "share_Erenhot", "share_Manzhouli",
                         "share_sea_other"]].to_string(index=False))
        f.write("\n")
    print(f"[STRESS] wrote {out_dir / 'stress_summary.csv'}")
    print(f"[STRESS] wrote {text_path}")


if __name__ == "__main__":
    args = parse_args()
    if args.cmd == "analyze":
        analyze(args)
    else:
        run(args)
