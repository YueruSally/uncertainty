#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Smoke test for the V1 wiring in baseline3_v1.py.

Goal — verify (without adding any congestion-delay function yet):
  1. Per-mode border delays are now LOADED and APPLIED at break-of-gauge
     nodes (incl. non-CN Brest etc.) — they were silently dropped before.
  2. BorderCapacity + BackgroundFlow now feed a node capacity soft
     constraint and a reported utilisation u_j.
  3. Tightening a bottleneck's capacity REROUTES flow to parallel
     corridors / sea — i.e. the knob actually moves the solution.

Runs at deliberately small scale (this is a wiring check, not the real
experiment). Writes results to smoke_result.json + smoke_result.txt.
"""
import json, random, time, copy, os, sys
import numpy as np
import baseline3_v1 as B

DATA = "data_expanded.xlsx"
OUTD = "smoke_out"
os.makedirs(OUTD, exist_ok=True)

SCENARIOS = {
    "A": ("A_baseline", None),
    "B": ("B_khorgos_tight", {"Khorgos": 400}),
    "C": ("C_brest_tight", {"Brest": 380}),
}

# ── reduced (but adequate) scale for a wiring check ─────────────────
B.DFS_MAX_PATHS_PER_OD     = 120
B.PATHS_TOPK_PER_CRITERION = 12
B.PATH_LIB_CAP_TOTAL       = 36
N_BATCHES   = 14
POP, GENS   = 140, 140
SEED        = 2026

EASTERN_ENTRIES = ["Khorgos", "Alashankou", "Erenhot", "Manzhouli", "Zabaykalsk"]
TRACK_NODES     = EASTERN_ENTRIES + ["Dostyk", "Altynkol", "Brest", "Malaszewicze"]


def load():
    (node_names, node_region, node_hold_cost, node_proc_cost, node_trans_cost,
     arcs, timetables, batches,
     wc, we, carbon_tax_map, emission_factor_map, mode_speeds_map,
     trans_map, border_delay_map, theta_rm) = B.load_network_from_extended(DATA)
    return dict(node_names=node_names, node_region=node_region,
                node_hold_cost=node_hold_cost, node_proc_cost=node_proc_cost,
                node_trans_cost=node_trans_cost, arcs=arcs, timetables=timetables,
                batches=batches, wc=wc, we=we, carbon_tax_map=carbon_tax_map,
                emission_factor_map=emission_factor_map, mode_speeds_map=mode_speeds_map,
                trans_map=trans_map, border_delay_map=border_delay_map, theta_rm=theta_rm)


def pick_representative(population, pareto):
    """Feasible rank-0 min-cost; else min-penalty individual."""
    feas = [i for i in pareto if i.feasible]
    if feas:
        return min(feas, key=lambda x: x.objectives[0]), True
    return min(population, key=lambda x: x.penalty), False


def summarize(ind):
    bf = getattr(ind, "border_flow", {}) or {}
    bu = getattr(ind, "border_util", {}) or {}
    vb = ind.vio_breakdown
    return {
        "feasible": bool(ind.feasible),
        "penalty": float(ind.penalty),
        "cost": float(ind.objectives[0]),
        "emission_gCO2": float(ind.objectives[1]),
        "time_h": float(ind.objectives[2]),
        "max_border_util": float(vb.get("max_border_util", 0.0)),
        "border_cap_excess": float(vb.get("border_cap_excess", 0.0)),
        "arc_cap_excess": float(vb.get("cap_excess", 0.0)),
        "late_teu_h": float(vb.get("late_teu_h", 0.0)),
        "flow_through": {n: round(float(bf.get(n, 0.0)), 1) for n in TRACK_NODES},
        "peak_util": {n: round(float(bu.get(n, 0.0)), 3) for n in TRACK_NODES},
    }


def run_scenario(name, ctx, path_lib, batches, eval_kwargs, cap_overrides=None):
    # apply capacity overrides on the module global, then restore after
    saved = dict(B.BORDER_CAPACITY)
    if cap_overrides:
        for k, v in cap_overrides.items():
            B.BORDER_CAPACITY[k] = v
    random.seed(SEED); np.random.seed(SEED)
    t0 = time.perf_counter()
    out = B.run_nsga2(
        ctx["node_names"], ctx["node_region"], ctx["node_hold_cost"],
        ctx["node_proc_cost"], ctx["node_trans_cost"],
        ctx["arcs"], ctx["timetables"], batches,
        ctx["wc"], ctx["we"], ctx["carbon_tax_map"], ctx["emission_factor_map"],
        ctx["mode_speeds_map"], ctx["trans_map"], ctx["border_delay_map"],
        ctx["theta_rm"], path_lib, pop_size=POP, generations=GENS)
    population, pareto = out[0], out[1]
    rep, feas = pick_representative(population, pareto)
    # re-evaluate representative under the (possibly overridden) caps so its
    # border_flow / border_util attributes are guaranteed populated
    B.evaluate_individual(rep, batches, ctx["arcs"],
                          B.build_timetable_dict(ctx["timetables"]),
                          ctx["wc"], ctx["we"], **eval_kwargs)
    res = summarize(rep)
    res["scenario"] = name
    res["cap_used"] = {n: B.BORDER_CAPACITY.get(n, 0.0) for n in TRACK_NODES}
    res["runtime_s"] = round(time.perf_counter() - t0, 1)
    res["pareto_size"] = len(pareto)
    res["n_feasible_pop"] = sum(1 for i in population if i.feasible)
    # restore
    B.BORDER_CAPACITY.clear(); B.BORDER_CAPACITY.update(saved)
    return res


def main():
    scen = sys.argv[1] if len(sys.argv) > 1 else "A"
    scen_name, overrides = SCENARIOS[scen]
    log = open(os.path.join(OUTD, f"progress_{scen}.txt"), "w")
    def say(*a):
        print(*a); print(*a, file=log); log.flush()

    say(f"=== V1 WIRING SMOKE TEST — scenario {scen} ({scen_name}) ===")
    ctx = load()

    # check #1: are border delays actually loaded for break-of-gauge nodes?
    bdm = ctx["border_delay_map"]
    sample = {k: v for k, v in bdm.items() if k[0] in B.BREAK_OF_GAUGE_NODES}
    say(f"[CHECK1] border_delay_map entries total = {len(bdm)}")
    say(f"[CHECK1] break-of-gauge delay entries (sample): "
        f"{dict(list(sample.items())[:8])}")
    say(f"[CHECK1] Brest rail delay = {bdm.get(('Brest','rail'))}, "
        f"road = {bdm.get(('Brest','road'))}")
    say(f"[CHECK2] BORDER_CAPACITY active(BoG) = "
        f"{ {n:B.BORDER_CAPACITY.get(n) for n in B.BREAK_OF_GAUGE_NODES if n in B.BORDER_CAPACITY} }")
    say(f"[CHECK2] BACKGROUND_FLOW active(BoG) = "
        f"{ {n:B.BACKGROUND_FLOW.get(n) for n in B.BREAK_OF_GAUGE_NODES if n in B.BACKGROUND_FLOW} }")

    # subsample batches for speed
    batches = ctx["batches"][:N_BATCHES]
    say(f"[INFO] using {len(batches)} batches, pop={POP}, gens={GENS}")

    tt_dict = B.build_timetable_dict(ctx["timetables"])
    arc_lookup = B.build_arc_lookup(ctx["arcs"])
    random.seed(0); np.random.seed(0)
    path_lib = B.build_path_library(ctx["node_names"], ctx["node_region"],
                                    ctx["arcs"], batches, tt_dict, arc_lookup)
    B.sanity_check_path_lib(batches, path_lib)

    eval_kwargs = dict(
        node_hold_cost=ctx["node_hold_cost"], node_proc_cost=ctx["node_proc_cost"],
        carbon_tax_map=ctx["carbon_tax_map"], trans_map=ctx["trans_map"],
        border_delay_map=ctx["border_delay_map"], theta_rm=ctx["theta_rm"],
        node_trans_cost=ctx["node_trans_cost"])

    say(f"\n--- Scenario {scen}: {scen_name} overrides={overrides} ---")
    r = run_scenario(scen_name, ctx, path_lib, batches, eval_kwargs,
                     cap_overrides=overrides)

    with open(os.path.join(OUTD, f"result_{scen}.json"), "w") as f:
        json.dump(r, f, indent=2, ensure_ascii=False)

    say(f"\n[DONE {scen}] feasible={r['feasible']} pareto={r['pareto_size']} "
        f"cost={r['cost']:.3e} time={r['time_h']:.1f}h")
    say(f"  max_border_util={r['max_border_util']:.3f} "
        f"border_cap_excess={r['border_cap_excess']:.1f}")
    for n in TRACK_NODES:
        say(f"   {n:<14} flow={r['flow_through'][n]:<8} "
            f"util={r['peak_util'][n]:<6} cap={r['cap_used'][n]}")
    log.close()


if __name__ == "__main__":
    main()
