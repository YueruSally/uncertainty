#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OFAT (One-Factor-At-A-Time) parameter sensitivity experiment
============================================================
Runs the V1-wired deterministic model (baseline3_v1.py) on data_expanded.xlsx,
perturbs ONE class of parameter at a time (-30/-15/+15/+30%), repeats over
several random seeds, and records for the chosen representative solution:

  * flow & SHARE through each parallel break-of-gauge entry
    (Khorgos / Alashankou / Erenhot / Manzhouli) + Brest + sea/other
  * peak border utilisation u_j
  * objectives  (cost / emission / time)  + late penalty + feasibility

Then ranks the parameters by how much they move (a) corridor SHARES — the
thesis mechanism — and (b) the objectives. The top parameter(s) are the
natural candidates for the later uncertainty model.

USAGE
-----
  python3 ofat_experiment.py run            # full sweep (resumable)
  python3 ofat_experiment.py run --quick    # medium, ~minutes
  python3 ofat_experiment.py run --test     # tiny, for a correctness check
  python3 ofat_experiment.py analyze        # (re)build summary from CSV

The sweep is RESUMABLE: every completed run is appended to
ofat_out/ofat_results.csv immediately, and already-done (param,level,seed)
combinations are skipped on re-launch. So if it is interrupted, just run the
same command again and it continues. `analyze` runs automatically once the
sweep is complete.
"""
import os, sys, json, time, copy, contextlib, io, shutil
from dataclasses import replace
import numpy as np
import pandas as pd
import random

import baseline3_v1 as B

DATA = "data_expanded.xlsx"
OUTD = "ofat_out"
CSV  = os.path.join(OUTD, "ofat_results.csv")
os.makedirs(OUTD, exist_ok=True)
REQUIRED_RESULT_COLUMNS = {"param", "level_pct", "seed"}

# ════════════════════════════════════════════════════════════════════
# CONFIG  (defaults = FULL scale, as chosen)
# ════════════════════════════════════════════════════════════════════
MODE = "full"
if "--quick" in sys.argv: MODE = "quick"
if "--test"  in sys.argv: MODE = "test"

if MODE == "full":
    N_BATCHES = None              # None = use all 40 batches
    POP, GEN  = 200, 150
    SEEDS     = [2026, 7, 99]
    B.DFS_MAX_PATHS_PER_OD, B.PATHS_TOPK_PER_CRITERION, B.PATH_LIB_CAP_TOTAL = 150, 15, 45
elif MODE == "quick":
    N_BATCHES = 14
    POP, GEN  = 90, 80
    SEEDS     = [2026, 7]
    B.DFS_MAX_PATHS_PER_OD, B.PATHS_TOPK_PER_CRITERION, B.PATH_LIB_CAP_TOTAL = 120, 12, 36
else:  # test
    N_BATCHES = 12
    POP, GEN  = 40, 30
    SEEDS     = [2026]
    B.DFS_MAX_PATHS_PER_OD, B.PATHS_TOPK_PER_CRITERION, B.PATH_LIB_CAP_TOTAL = 60, 8, 24

LEVELS = [-30, -15, 15, 30]       # percent; baseline (0) handled separately

# Parameters to sweep (the "uncertainty candidates")
PARAMS = ["P1_border_delay_rail", "P2_border_capacity", "P3_background_flow",
          "P4_transfer_time", "P5_transfer_cost", "P6_frequency",
          "P7_cost_per_km", "P8_emission_factor", "P9_demand", "P10_time_window_LT"]
if MODE == "test":
    PARAMS = ["P2_border_capacity", "P7_cost_per_km"]   # just exercise both code paths

# Parallel break-of-gauge ENTRY nodes (where parallel-corridor rerouting shows)
ENTRIES = ["Khorgos", "Alashankou", "Erenhot", "Manzhouli"]
TRACK   = ENTRIES + ["Brest"]


# ════════════════════════════════════════════════════════════════════
# Master data (loaded once)
# ════════════════════════════════════════════════════════════════════
def load_master():
    (node_names, node_region, nhc, npc, ntc, arcs, timetables, batches,
     wc, we, ctm, efm, msm, trans_map, bdm, theta) = B.load_network_from_extended(DATA)
    if N_BATCHES:
        batches = batches[:N_BATCHES]
    M = dict(node_names=node_names, node_region=node_region, nhc=nhc, npc=npc,
             ntc=ntc, arcs=arcs, timetables=timetables, batches=batches,
             wc=wc, we=we, ctm=ctm, efm=efm, msm=msm, trans_map=trans_map,
             bdm=bdm, theta=theta)
    # snapshot mutable masters we perturb-in-place
    M["arc_cost0"] = [a.cost_per_teu_km for a in arcs]
    M["arc_emis0"] = [a.emission_per_teu_km for a in arcs]
    M["cap0"] = dict(B.BORDER_CAPACITY)
    M["bg0"]  = dict(B.BACKGROUND_FLOW)
    return M


def recompute_path_metrics(path_lib):
    """Paths cache base cost/emission/time from their Arc objects; recompute
    after arc attributes have been scaled (P7/P8)."""
    for paths in path_lib.values():
        for p in paths:
            p.base_cost_per_teu     = sum(a.cost_per_teu_km * a.distance for a in p.arcs)
            p.base_emission_per_teu = sum(a.emission_per_teu_km * a.distance for a in p.arcs)
            p.base_travel_time_h    = sum(a.distance / max(a.speed_kmh, 1.0) for a in p.arcs)


def restore_arcs(M, path_lib):
    for a, c, e in zip(M["arcs"], M["arc_cost0"], M["arc_emis0"]):
        a.cost_per_teu_km = c
        a.emission_per_teu_km = e
    recompute_path_metrics(path_lib)
    B.BORDER_CAPACITY.clear(); B.BORDER_CAPACITY.update(M["cap0"])
    B.BACKGROUND_FLOW.clear(); B.BACKGROUND_FLOW.update(M["bg0"])


# ════════════════════════════════════════════════════════════════════
# Apply one perturbation -> returns the (possibly modified) run inputs
# ════════════════════════════════════════════════════════════════════
def build_inputs(M, path_lib, param, factor):
    """factor = 1+level/100. Returns dict of inputs for run_nsga2 + eval."""
    bdm   = copy.deepcopy(M["bdm"])
    trans = copy.deepcopy(M["trans_map"])
    tts   = list(M["timetables"])
    batches = list(M["batches"])
    BoG = B.BREAK_OF_GAUGE_NODES

    if param == "P1_border_delay_rail":
        for k in list(bdm.keys()):
            if k[0] in BoG and k[1] == "rail":
                bdm[k] = bdm[k] * factor
    elif param == "P2_border_capacity":
        for n in BoG:
            if n in B.BORDER_CAPACITY:
                B.BORDER_CAPACITY[n] = M["cap0"][n] * factor
    elif param == "P3_background_flow":
        for n in BoG:
            if n in B.BACKGROUND_FLOW:
                B.BACKGROUND_FLOW[n] = M["bg0"][n] * factor
    elif param == "P4_transfer_time":
        for k in trans: trans[k]["time_h"] = trans[k].get("time_h", 0.0) * factor
    elif param == "P5_transfer_cost":
        for k in trans: trans[k]["cost_per_teu"] = trans[k].get("cost_per_teu", 0.0) * factor
    elif param == "P6_frequency":
        new = []
        for t in tts:
            f = max(t.frequency_per_week * factor, 1e-6)
            new.append(replace(t, frequency_per_week=f, headway_hours=168.0 / f))
        tts = new
    elif param == "P7_cost_per_km":
        for a in M["arcs"]: a.cost_per_teu_km *= factor
        recompute_path_metrics(path_lib)
    elif param == "P8_emission_factor":
        for a in M["arcs"]: a.emission_per_teu_km *= factor
        recompute_path_metrics(path_lib)
    elif param == "P9_demand":
        batches = [replace(b, quantity=b.quantity * factor) for b in batches]
    elif param == "P10_time_window_LT":
        batches = [replace(b, LT=b.LT * factor) for b in batches]
    elif param == "BASE":
        pass
    else:
        raise ValueError(param)

    return dict(bdm=bdm, trans=trans, tts=tts, batches=batches)


def pick_rep(population, pareto):
    feas = [i for i in pareto if i.feasible]
    if feas:
        return min(feas, key=lambda x: x.objectives[0])
    return min(population, key=lambda x: x.penalty)


def run_one(M, path_lib, param, level, seed):
    factor = 1.0 + level / 100.0
    inp = build_inputs(M, path_lib, param, factor)
    bdm, trans, tts, batches = inp["bdm"], inp["trans"], inp["tts"], inp["batches"]
    ek = dict(node_hold_cost=M["nhc"], node_proc_cost=M["npc"], carbon_tax_map=M["ctm"],
              trans_map=trans, border_delay_map=bdm, theta_rm=M["theta"],
              node_trans_cost=M["ntc"])
    random.seed(seed); np.random.seed(seed)
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            out = B.run_nsga2(
                M["node_names"], M["node_region"], M["nhc"], M["npc"], M["ntc"],
                M["arcs"], tts, batches, M["wc"], M["we"], M["ctm"], M["efm"],
                M["msm"], trans, bdm, M["theta"], path_lib,
                pop_size=POP, generations=GEN)
            population, pareto = out[0], out[1]
            rep = pick_rep(population, pareto)
            B.evaluate_individual(rep, batches, M["arcs"],
                                  B.build_timetable_dict(tts), M["wc"], M["we"], **ek)
    finally:
        if param in ("P7_cost_per_km", "P8_emission_factor",
                     "P2_border_capacity", "P3_background_flow"):
            restore_arcs(M, path_lib)

    bf = getattr(rep, "border_flow", {}) or {}
    bu = getattr(rep, "border_util", {}) or {}
    total = sum(b.quantity for b in batches)
    entry_flow = {e: float(bf.get(e, 0.0)) for e in ENTRIES}
    sea_other = max(0.0, total - sum(entry_flow.values()))
    row = {
        "param": param, "level_pct": level, "seed": seed,
        "feasible": bool(rep.feasible),
        "cost": float(rep.objectives[0]),
        "emission_gCO2": float(rep.objectives[1]),
        "time_h": float(rep.objectives[2]),
        "late_teu_h": float(rep.vio_breakdown.get("late_teu_h", 0.0)),
        "penalty": float(rep.penalty),
        "max_border_util": float(rep.vio_breakdown.get("max_border_util", 0.0)),
        "border_cap_excess": float(rep.vio_breakdown.get("border_cap_excess", 0.0)),
        "total_demand": float(total),
    }
    for e in ENTRIES:
        row[f"flow_{e}"]  = round(entry_flow[e], 1)
        row[f"share_{e}"] = round(entry_flow[e] / total, 4) if total > 0 else 0.0
        row[f"util_{e}"]  = round(float(bu.get(e, 0.0)), 4)
    row["flow_Brest"]  = round(float(bf.get("Brest", 0.0)), 1)
    row["util_Brest"]  = round(float(bu.get("Brest", 0.0)), 4)
    row["share_sea"]   = round(sea_other / total, 4) if total > 0 else 0.0
    return row


# ════════════════════════════════════════════════════════════════════
# Sweep (resumable) + analyze
# ════════════════════════════════════════════════════════════════════
def done_keys():
    if not os.path.exists(CSV):
        return set()
    df = pd.read_csv(CSV)
    if not REQUIRED_RESULT_COLUMNS.issubset(df.columns):
        stamp = time.strftime("%Y%m%d_%H%M%S")
        backup = os.path.join(OUTD, f"ofat_results_legacy_{stamp}.csv")
        shutil.move(CSV, backup)
        print(f"[OFAT] Existing result CSV has old/incompatible columns; moved to {backup}")
        return set()
    return set(zip(df["param"], df["level_pct"], df["seed"]))


def append_row(row):
    df = pd.DataFrame([row])
    df.to_csv(CSV, mode="a", header=not os.path.exists(CSV), index=False)


def build_plan():
    plan = [("BASE", 0, s) for s in SEEDS]
    for p in PARAMS:
        for lv in LEVELS:
            for s in SEEDS:
                plan.append((p, lv, s))
    return plan


def sweep():
    print(f"[OFAT] MODE={MODE}  pop={POP} gen={GEN} seeds={SEEDS} "
          f"batches={N_BATCHES or 'all'} params={len(PARAMS)}")
    M = load_master()
    tt = B.build_timetable_dict(M["timetables"]); al = B.build_arc_lookup(M["arcs"])
    random.seed(0); np.random.seed(0)
    path_lib = B.build_path_library(M["node_names"], M["node_region"],
                                    M["arcs"], M["batches"], tt, al)
    B.sanity_check_path_lib(M["batches"], path_lib)

    plan = build_plan()
    done = done_keys()
    todo = [t for t in plan if t not in done]
    print(f"[OFAT] total runs={len(plan)}  done={len(done)}  remaining={len(todo)}")
    for i, (param, lv, seed) in enumerate(todo, 1):
        t0 = time.time()
        row = run_one(M, path_lib, param, lv, seed)
        append_row(row)
        print(f"[OFAT] ({i}/{len(todo)}) {param} {lv:+d}% seed={seed} "
              f"feas={row['feasible']} cost={row['cost']:.3e} "
              f"shares K/A/E/M="
              f"{row['share_Khorgos']:.2f}/{row['share_Alashankou']:.2f}/"
              f"{row['share_Erenhot']:.2f}/{row['share_Manzhouli']:.2f} "
              f"sea={row['share_sea']:.2f}  ({time.time()-t0:.0f}s)", flush=True)

    if len(done_keys()) >= len(plan):
        print("[OFAT] sweep complete -> analyze")
        analyze()
    else:
        print("[OFAT] partial; re-run the same command to continue.")


def analyze():
    if not os.path.exists(CSV):
        print("[OFAT] no results yet."); return
    df = pd.read_csv(CSV)
    if not REQUIRED_RESULT_COLUMNS.issubset(df.columns):
        print(f"[OFAT] incompatible result CSV columns: {list(df.columns)}")
        print("[OFAT] remove or rename ofat_out/ofat_results.csv, then rerun.")
        return
    share_cols = [f"share_{e}" for e in ENTRIES] + ["share_sea"]
    obj_cols   = ["cost", "emission_gCO2", "time_h"]
    agg_cols   = share_cols + obj_cols + ["max_border_util", "feasible",
                                          "border_cap_excess", "late_teu_h"]
    # mean over seeds per (param, level)
    g = df.groupby(["param", "level_pct"])[agg_cols].mean().reset_index()
    g.to_csv(os.path.join(OUTD, "summary_by_param_level.csv"), index=False)

    base = g[g["param"] == "BASE"]
    if base.empty:
        print("[OFAT] no BASE rows; cannot rank."); return
    b = base.iloc[0]

    rank = []
    for p in [x for x in g["param"].unique() if x != "BASE"]:
        sub = g[g["param"] == p]
        # share swing: max over levels of total |share - base| across entries+sea
        share_swing = 0.0
        for _, r in sub.iterrows():
            s = sum(abs(r[c] - b[c]) for c in share_cols)
            share_swing = max(share_swing, s)
        # objective swing: max relative change across the 3 objectives
        obj_swing = 0.0
        for _, r in sub.iterrows():
            for c in obj_cols:
                if b[c] > 0:
                    obj_swing = max(obj_swing, abs(r[c] - b[c]) / b[c])
        util_swing = max(abs(sub["max_border_util"] - b["max_border_util"]).max(), 0.0)
        rank.append({"param": p,
                     "share_swing": round(share_swing, 4),
                     "obj_swing_rel": round(obj_swing, 4),
                     "util_swing": round(float(util_swing), 4)})
    rk = pd.DataFrame(rank).sort_values("share_swing", ascending=False)
    rk.to_csv(os.path.join(OUTD, "sensitivity_ranking.csv"), index=False)

    with open(os.path.join(OUTD, "sensitivity_ranking.txt"), "w") as f:
        f.write("OFAT SENSITIVITY RANKING\n")
        f.write(f"(mode={MODE}, seeds={SEEDS}, levels={LEVELS})\n\n")
        f.write(f"Full-mode setting: batches={N_BATCHES or 'all'}, pop={POP}, gen={GEN}, representative=min-cost feasible Pareto solution.\n\n")
        f.write("share_swing  = max total reallocation of corridor share vs baseline\n")
        f.write("               (THE thesis-mechanism signal — higher = more rerouting)\n")
        f.write("obj_swing_rel= max relative change in cost/emission/time vs baseline\n")
        f.write("util_swing   = max change in peak border utilisation vs baseline\n\n")
        f.write(rk.to_string(index=False))
        f.write("\n\nINTERPRETATION\n")
        if not rk.empty:
            top = rk.iloc[0]["param"]
            f.write(f"  Most influential on corridor rerouting: {top}\n")
            f.write(f"  -> strongest candidate for the uncertainty source.\n")
            cong = rk[rk["param"].str.startswith(("P1", "P2", "P3"))]
            f.write(f"  Congestion-coupled params (P1/P2/P3) share_swing: "
                    f"{dict(zip(cong['param'], cong['share_swing']))}\n")
    print("[OFAT] wrote summary_by_param_level.csv, sensitivity_ranking.csv/.txt")
    print(rk.to_string(index=False))


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "run"
    if cmd == "analyze":
        analyze()
    else:
        sweep()
