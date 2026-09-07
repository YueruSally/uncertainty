#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""10-scenario sanity check for the deterministic/EV/CCP evaluation pipeline.

DIAGNOSTIC / REPORTING SCRIPT ONLY. It imports and calls the REAL
production functions in baseline_uncertainty.py as the sole source of
truth for every number reported:

  - load_network_from_extended, build_path_library   (network + paths)
  - greedy_initial_individual                         (ONE fixed candidate
                                                         solution -- NOT
                                                         NSGA-II, no
                                                         population search)
  - configure_scenario_set / build_scenario_set        (the frozen S=10
                                                         scenario set)
  - simulate_path_over_scenarios                       (per-scenario
                                                         arrival arrays)
  - evaluate_individual + aggregate_scenario_objectives (EV and CCP
                                                         aggregate results)
  - empirical_ccp_quantile                             (CCP objective
                                                         quantiles, called
                                                         indirectly via
                                                         evaluate_individual)

The per-scenario, per-node trace below (`instrumented_trace`) is an
INSTRUMENTED REPLAY of simulate_path_over_scenarios's own loop: it calls
the exact same helper functions (scenario_set.border_for_arc,
scenario_set.travel, next_departure_time_programB, nominal_arc_travel_time)
in the exact same order, purely to expose intermediate per-node state for
human-readable reporting. It does not reimplement any decision logic and
does not compute any number that overrides what the production functions
themselves produce.

Does NOT run NSGA-II. Does NOT run population-based optimisation. Does
NOT run 50/100/200 scenarios or 30 independent runs.
"""
import json
import random
import sys
from pathlib import Path as FSPath

import numpy as np

sys.path.insert(0, str(FSPath(__file__).resolve().parent))
import baseline_uncertainty as model  # noqa: E402

HERE = FSPath(__file__).resolve().parent
DATA_FILE = HERE / "data" / "data_expanded_b20.xlsx"
OUT_DIR = HERE / "sanity_10"
SEED = 42
S = 10

FLOAT_TOL = 1e-9


def _close(a, b, tol=FLOAT_TOL):
    """Value comparison with an explicit floating-point tolerance -- never
    object identity, never a bare `==` on two independently computed
    floats."""
    if a is None or b is None:
        return a == b
    return abs(float(a) - float(b)) <= tol


def _tuples_close(a, b, tol=FLOAT_TOL):
    return len(a) == len(b) and all(_close(x, y, tol) for x, y in zip(a, b))


def _dicts_close(a, b, tol=FLOAT_TOL):
    return set(a.keys()) == set(b.keys()) and all(
        _close(a[k], b[k], tol) for k in a)


def load_everything():
    (node_names, node_region, node_hold_cost, node_proc_cost, node_trans_cost,
     arcs, timetables, batches, waiting_cost_per_teu_h, wait_emis_g_per_teu_h,
     carbon_tax_map, emission_factor_map, mode_speeds_map, trans_map,
     border_delay_map, theta_rm, mode_speed_source
     ) = model.load_network_from_extended(str(DATA_FILE))
    tt_dict = model.build_timetable_dict(timetables)
    arc_lookup = model.build_arc_lookup(arcs)
    random.seed(0)
    np.random.seed(0)
    path_lib = model.build_path_library(
        node_names, node_region, arcs, batches, tt_dict, arc_lookup)
    return dict(
        node_names=node_names, node_region=node_region,
        node_hold_cost=node_hold_cost, node_proc_cost=node_proc_cost,
        node_trans_cost=node_trans_cost, arcs=arcs, timetables=timetables,
        batches=batches, waiting_cost_per_teu_h=waiting_cost_per_teu_h,
        wait_emis_g_per_teu_h=wait_emis_g_per_teu_h,
        carbon_tax_map=carbon_tax_map, emission_factor_map=emission_factor_map,
        mode_speeds_map=mode_speeds_map, trans_map=trans_map,
        border_delay_map=border_delay_map, theta_rm=theta_rm,
        mode_speed_source=mode_speed_source, tt_dict=tt_dict,
        arc_lookup=arc_lookup, path_lib=path_lib,
    )


def pick_representative_allocation(individual, batches):
    """Prefer a batch whose fixed path includes a non-road (rail/water)
    arc, so timetable/waiting/missed-departure diagnostics are meaningful.
    Falls back to the first allocated batch otherwise."""
    batches_by_id = {b.batch_id: b for b in batches}
    fallback = None
    for key, allocs in individual.od_allocations.items():
        if not allocs:
            continue
        origin, destination, batch_id = key
        alloc = allocs[0]
        if fallback is None:
            fallback = (batches_by_id[batch_id], alloc.path)
        if any(arc.mode != "road" for arc in alloc.path.arcs):
            return batches_by_id[batch_id], alloc.path
    return fallback


def instrumented_trace(path, batch, tt_dict, trans_map, scenario_set):
    """Per-scenario, per-node replay using the REAL helper functions.
    Also flags a 'missed departure' at rail/water nodes: a scenario misses
    the nominal (multiplier=1, mean border delay) departure slot if its
    stochastic readiness time at that node falls after the nominal
    departure that a zero-noise scenario would have caught, forcing it
    onto a later headway slot."""
    nominal_events = _replay_one_scenario(
        path, batch, tt_dict, trans_map, scenario_set, scenario=None)
    nominal_departure_by_node = {
        e["node"]: e["scheduled_departure_h"] for e in nominal_events
        if e.get("scheduled_departure_h") is not None}

    traces = []
    for s in range(scenario_set.size):
        events = _replay_one_scenario(
            path, batch, tt_dict, trans_map, scenario_set, scenario=s)
        valid = all("error" not in e for e in events)
        for e in events:
            nominal_dep = nominal_departure_by_node.get(e["node"])
            e["missed_nominal_departure"] = bool(
                e["mode"] != "road" and nominal_dep is not None
                and e["scheduled_departure_h"] is not None
                and e["scheduled_departure_h"] > nominal_dep + 1e-9)
        traces.append(dict(
            scenario_id=s,
            valid=valid,
            final_arrival_h=(events[-1]["node_arrival_h"] if valid and events
                              else None),
            node_events=events,
        ))
    return traces


def _replay_one_scenario(path, batch, tt_dict, trans_map, scenario_set,
                          scenario):
    """scenario=None means the NOMINAL replay: multiplier=1.0 and mean
    border delay (via scenario_set.border_event_mean_h), used only to
    define what a 'missed departure' means relative to. scenario=int uses
    the real frozen draw for that scenario index, via scenario_set.travel
    / scenario_set.border_for_arc exactly as simulate_path_over_scenarios
    does."""
    t = float(batch.ET)
    prev_arc = None
    incurred_border_events = set()
    events = []
    for arc in path.arcs:
        node = arc.from_node
        transfer_wait = 0.0
        if prev_arc is not None and prev_arc.mode != arc.mode:
            rec = trans_map.get((node, prev_arc.mode, arc.mode), {})
            transfer_wait = max(0.0, model.safe_float(rec.get("time_h"), 0.0))
            t += transfer_wait

        if scenario is None:
            event_key = scenario_set.border_event_for_arc(arc)
            bd = (0.0 if event_key is None else
                  float(scenario_set.border_event_mean_h.get(event_key, 0.0)))
        else:
            event_key, bd = scenario_set.border_for_arc(arc, scenario)
        border_delay_applied = 0.0
        if event_key is not None and event_key not in incurred_border_events:
            incurred_border_events.add(event_key)
            if bd > 0.0:
                t += bd
                border_delay_applied = bd

        readiness_time = t
        entries = [] if arc.mode == "road" else \
            tt_dict.get((node, arc.to_node, arc.mode), [])
        if arc.mode != "road" and not entries:
            events.append(dict(
                node=node, to_node=arc.to_node, mode=arc.mode,
                error="missing_timetable"))
            break

        departure = t if not entries else \
            model.next_departure_time_programB(t, entries)
        wait_h = max(0.0, departure - t)

        multiplier = 1.0 if scenario is None else scenario_set.travel(arc, scenario)
        nominal_travel_h = model.nominal_arc_travel_time(arc, tt_dict)
        realized_travel_h = nominal_travel_h * multiplier
        arrival_next = departure + realized_travel_h

        events.append(dict(
            node=node, to_node=arc.to_node, mode=arc.mode,
            transfer_wait_h=transfer_wait,
            border_event=("|".join(event_key) if event_key else None),
            border_delay_h=border_delay_applied,
            readiness_time_h=readiness_time,
            scheduled_departure_h=departure,
            schedule_wait_h=wait_h,
            travel_multiplier=multiplier,
            nominal_travel_h=nominal_travel_h,
            realized_travel_h=realized_travel_h,
            node_arrival_h=arrival_next,
        ))
        t = arrival_next
        prev_arc = arc
    return events


def run_ev_and_ccp(env, individual, batches, arcs, tt_dict, scenario_set):
    """Evaluate EV and CCP against the GIVEN scenario_set.

    Explicitly installs `scenario_set` as model.ACTIVE_SCENARIO_SET before
    every evaluation (previous bug: this function relied on the CALLER
    having already set the global correctly, so a caller could pass one
    scenario_set object here while a *different* object silently stayed
    active -- evaluation would then silently run against the wrong
    scenarios). model.ACTIVE_SCENARIO_SET is intentionally left pointing
    at `scenario_set` after this function returns (not restored to
    whatever was active before), so the caller can rely on "the scenario
    set these results came from is still the active one" -- only
    RISK_METRIC (a pure mode toggle, not scenario identity) is restored.
    """
    results = {}
    for mode in ("ev", "ccp"):
        previous_mode = model.RISK_METRIC
        model.RISK_METRIC = mode
        model.ACTIVE_SCENARIO_SET = scenario_set
        try:
            ind = model.Individual(od_allocations={
                key: list(allocs)
                for key, allocs in individual.od_allocations.items()})
            model.evaluate_individual(
                ind, batches, arcs, tt_dict,
                waiting_cost_per_teu_h=env["waiting_cost_per_teu_h"],
                wait_emis_g_per_teu_h=env["wait_emis_g_per_teu_h"],
                node_hold_cost=env["node_hold_cost"],
                node_proc_cost=env["node_proc_cost"],
                carbon_tax_map=env["carbon_tax_map"],
                trans_map=env["trans_map"],
                border_delay_map=env["border_delay_map"],
                theta_rm=env["theta_rm"],
                node_trans_cost=env["node_trans_cost"],
            )
        finally:
            model.RISK_METRIC = previous_mode
        results[mode] = dict(
            objectives=list(ind.objectives),
            penalty=ind.penalty,
            feasible=ind.feasible,
            feasible_hard=ind.feasible_hard,
            normalized_violation=ind.normalized_violation,
            vio_breakdown=ind.vio_breakdown,
            batch_on_time_prob=ind.batch_on_time_prob,
            batch_max_lateness_h=ind.batch_max_lateness_h,
        )
        results[mode]["active_scenario_set_id"] = id(model.ACTIVE_SCENARIO_SET)
    return results


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    env = load_everything()
    arcs, batches, tt_dict = env["arcs"], env["batches"], env["tt_dict"]
    path_lib = env["path_lib"]

    # ONE fixed candidate solution -- greedy_initial_individual is a real,
    # deterministic, existing constructor. No NSGA-II, no population.
    individual = model.greedy_initial_individual(batches, path_lib)
    model.repair_missing_allocations(individual, batches, path_lib)

    rep_batch, rep_path = pick_representative_allocation(individual, batches)

    border_event_definitions = model.load_border_event_definitions(
        model.DEFAULT_BORDER_EVENT_DATA_FILE)

    # Configure the frozen scenario set ONCE. EV and CCP below reuse the
    # exact same model.ACTIVE_SCENARIO_SET object (identity, not just
    # equality) -- see run_ev_and_ccp, which never calls
    # configure_scenario_set again.
    scenario_set = model.configure_scenario_set(
        arcs=arcs, border_delay_map=env["border_delay_map"], size=S,
        seed=SEED, stochastic=True,
        border_event_definitions=border_event_definitions)
    frozen_scenario_set_id = id(model.ACTIVE_SCENARIO_SET)

    trace = instrumented_trace(
        rep_path, rep_batch, tt_dict, env["trans_map"], scenario_set)

    ev_ccp = run_ev_and_ccp(
        env, individual, batches, arcs, tt_dict, scenario_set)

    # --- Reproducibility check ---------------------------------------
    # 1. Independently rebuild a SECOND scenario set with the same seed.
    # 2. Compare its travel_multiplier / border_delay_h arrays to the
    #    original's (value comparison, not object identity).
    # 3. Explicitly INSTALL the rebuilt object as ACTIVE_SCENARIO_SET
    #    (via run_ev_and_ccp, which now does this itself -- see the fix
    #    to run_ev_and_ccp above) and re-evaluate for real against it.
    # 4. Compare the two independently produced aggregate outputs.
    scenario_set_repeat = model.build_scenario_set(
        arcs=arcs, border_delay_map=env["border_delay_map"], size=S,
        seed=SEED, stochastic=True,
        border_event_definitions=border_event_definitions)
    assert scenario_set_repeat is not scenario_set, (
        "reproducibility check requires two INDEPENDENT objects, not the "
        "same one reused")

    repro_ok = True
    repro_details = {}
    for key, arr in scenario_set.travel_multiplier.items():
        arr2 = scenario_set_repeat.travel_multiplier.get(key)
        same = arr2 is not None and np.array_equal(arr, arr2)
        repro_details[f"travel_multiplier[{key}]"] = bool(same)
        repro_ok = repro_ok and same
    for key, arr in scenario_set.border_delay_h.items():
        arr2 = scenario_set_repeat.border_delay_h.get(key)
        same = arr2 is not None and np.array_equal(arr, arr2)
        repro_details[f"border_delay_h[{key}]"] = bool(same)
        repro_ok = repro_ok and same

    repro_trace = instrumented_trace(
        rep_path, rep_batch, tt_dict, env["trans_map"], scenario_set_repeat)
    trace_identical = (trace == repro_trace)

    # run_ev_and_ccp now explicitly installs scenario_set_repeat as
    # ACTIVE_SCENARIO_SET before evaluating -- this genuinely exercises
    # the independently rebuilt object, not the original.
    ev_ccp_repeat = run_ev_and_ccp(
        env, individual, batches, arcs, tt_dict, scenario_set_repeat)
    repeat_used_rebuilt_object = (
        ev_ccp_repeat["ev"]["active_scenario_set_id"]
        == ev_ccp_repeat["ccp"]["active_scenario_set_id"]
        == id(scenario_set_repeat)
        != frozen_scenario_set_id)

    aggregates_identical = {
        mode: (
            _tuples_close(ev_ccp[mode]["objectives"],
                          ev_ccp_repeat[mode]["objectives"])
            and _close(ev_ccp[mode]["penalty"], ev_ccp_repeat[mode]["penalty"])
            and ev_ccp[mode]["feasible"] == ev_ccp_repeat[mode]["feasible"]
            and ev_ccp[mode]["feasible_hard"] == ev_ccp_repeat[mode]["feasible_hard"]
            and _close(ev_ccp[mode]["normalized_violation"],
                       ev_ccp_repeat[mode]["normalized_violation"])
            and _dicts_close(ev_ccp[mode]["batch_on_time_prob"],
                              ev_ccp_repeat[mode]["batch_on_time_prob"])
            and _dicts_close(ev_ccp[mode]["batch_max_lateness_h"],
                              ev_ccp_repeat[mode]["batch_max_lateness_h"])
        )
        for mode in ("ev", "ccp")
    }

    # Restore the original scenario set as active before continuing, so
    # later sections start from a known, original state.
    model.ACTIVE_SCENARIO_SET = scenario_set

    # --- State-isolation check -----------------------------------------
    # Part 1: BORDER_CAPACITY / BACKGROUND_FLOW / tt_dict must be
    # unmutated by evaluation.
    border_capacity_before = dict(model.BORDER_CAPACITY)
    background_flow_before = dict(model.BACKGROUND_FLOW)
    tt_dict_before = {k: list(v) for k, v in tt_dict.items()}
    _ = run_ev_and_ccp(env, individual, batches, arcs, tt_dict, scenario_set)
    state_isolation_ok = (
        model.BORDER_CAPACITY == border_capacity_before
        and model.BACKGROUND_FLOW == background_flow_before
        and {k: list(v) for k, v in tt_dict.items()} == tt_dict_before
    )

    # Part 2: genuine cross-scenario-set isolation. Take independent
    # (deep-copied) snapshots of the ORIGINAL set's arrays BEFORE
    # touching an alternate set, actually run a REAL evaluation while a
    # genuinely different alternate set (different seed) is active, then
    # verify the original's arrays are unchanged by value (not identity)
    # and that re-evaluating against the restored original reproduces
    # the very first original evaluation.
    original_travel_snapshot = {
        k: v.copy() for k, v in scenario_set.travel_multiplier.items()}
    original_border_snapshot = {
        k: v.copy() for k, v in scenario_set.border_delay_h.items()}

    alternate_scenario_set = model.build_scenario_set(
        arcs=arcs, border_delay_map=env["border_delay_map"], size=5,
        seed=43, stochastic=True,
        border_event_definitions=border_event_definitions)
    assert alternate_scenario_set is not scenario_set

    # Actually evaluate for real while the alternate set is active (this
    # is the missing step from the previous, no-op version of this check).
    alt_result = run_ev_and_ccp(
        env, individual, batches, arcs, tt_dict, alternate_scenario_set)
    alt_evaluation_used_alternate_set = (
        alt_result["ev"]["active_scenario_set_id"]
        == alt_result["ccp"]["active_scenario_set_id"]
        == id(alternate_scenario_set))

    original_travel_unchanged = all(
        key in scenario_set.travel_multiplier
        and np.array_equal(scenario_set.travel_multiplier[key], snapshot)
        for key, snapshot in original_travel_snapshot.items())
    original_border_unchanged = all(
        key in scenario_set.border_delay_h
        and np.array_equal(scenario_set.border_delay_h[key], snapshot)
        for key, snapshot in original_border_snapshot.items())

    # Restore the original set and re-evaluate; results must reproduce
    # the FIRST original evaluation (ev_ccp) computed at the very top of
    # main(), not merely be internally self-consistent.
    ev_ccp_after_alternate = run_ev_and_ccp(
        env, individual, batches, arcs, tt_dict, scenario_set)
    restored_matches_first_original_evaluation = {
        mode: (
            _tuples_close(ev_ccp[mode]["objectives"],
                          ev_ccp_after_alternate[mode]["objectives"])
            and _close(ev_ccp[mode]["penalty"],
                       ev_ccp_after_alternate[mode]["penalty"])
            and ev_ccp[mode]["feasible"] == ev_ccp_after_alternate[mode]["feasible"]
        )
        for mode in ("ev", "ccp")
    }

    original_arrays_unchanged = (
        original_travel_unchanged and original_border_unchanged)

    # --- Maritime rule check ---
    water_events = {
        "|".join(k): v for k, v in scenario_set.border_event_mean_h.items()
        if k[2] == "water"}
    water_delay_arrays_zero = all(
        bool(np.all(scenario_set.border_delay_h.get(k, np.zeros(S)) == 0.0))
        for k in scenario_set.border_event_mean_h if k[2] == "water")

    report = {
        "config": {
            "scenario_count": S, "seed": SEED,
            "data_file": str(DATA_FILE),
            "risk_metrics_evaluated": ["ev", "ccp"],
            "mode_time_cv": model.MODE_TIME_CV,
            "border_delay_cv": model.BORDER_DELAY_CV,
            "representative_batch_id": rep_batch.batch_id,
            "representative_od": [rep_batch.origin, rep_batch.destination],
            "representative_path_nodes": rep_path.nodes,
            "representative_path_modes": rep_path.modes,
            "representative_batch_ET": rep_batch.ET,
            "representative_batch_LT": rep_batch.LT,
            "representative_batch_penalty_per_teu_h": rep_batch.penalty_per_teu_h,
        },
        "per_scenario_trace": trace,
        "ev_result": ev_ccp["ev"],
        "ccp_result": ev_ccp["ccp"],
        "reproducibility": {
            "scenario_arrays_identical": repro_ok,
            "scenario_array_details": repro_details,
            "instrumented_trace_identical": trace_identical,
            # The rebuilt object was genuinely installed and evaluated
            # (not the original silently reused) -- see repeat_used_rebuilt_object.
            "repeat_evaluation_used_independently_rebuilt_object":
                repeat_used_rebuilt_object,
            "aggregate_results_identical": aggregates_identical,
            "note": (
                "scenario_set_repeat is a SEPARATE object from scenario_set "
                "(assert scenario_set_repeat is not scenario_set passed); "
                "run_ev_and_ccp explicitly installs it as "
                "ACTIVE_SCENARIO_SET before evaluating, and "
                "repeat_evaluation_used_independently_rebuilt_object "
                "confirms the id() seen during that evaluation was the "
                "rebuilt object's, not the original's."),
        },
        "shared_scenario_check": {
            "ev_active_scenario_set_id": ev_ccp["ev"]["active_scenario_set_id"],
            "ccp_active_scenario_set_id": ev_ccp["ccp"]["active_scenario_set_id"],
            "same_object_identity": (
                ev_ccp["ev"]["active_scenario_set_id"]
                == ev_ccp["ccp"]["active_scenario_set_id"]
                == frozen_scenario_set_id),
        },
        "state_isolation_check": {
            "border_capacity_background_flow_tt_dict_unmutated": state_isolation_ok,
            "alternate_set_seed": 43,
            "alternate_evaluation_genuinely_used_alternate_set":
                alt_evaluation_used_alternate_set,
            "original_scenario_arrays_unchanged_after_alternate_set_use":
                original_arrays_unchanged,
            "original_travel_multiplier_unchanged": original_travel_unchanged,
            "original_border_delay_h_unchanged": original_border_unchanged,
            "restored_original_reproduces_first_evaluation":
                restored_matches_first_original_evaluation,
            "note": (
                "Snapshots of scenario_set's travel_multiplier/"
                "border_delay_h were taken by value (array .copy()) BEFORE "
                "activating a genuinely different alternate_scenario_set "
                "(seed=43); a real EV+CCP evaluation was run while the "
                "alternate set was active (see "
                "alternate_evaluation_genuinely_used_alternate_set); the "
                "original set was then restored and re-evaluated, and its "
                "results are compared against the very FIRST original "
                "evaluation (ev_result/ccp_result above), not against "
                "themselves."),
        },
        "maritime_rule_check": {
            "water_border_events_found": water_events,
            "all_water_border_delay_arrays_zero": water_delay_arrays_zero,
            "water_travel_time_cv": model.MODE_TIME_CV["water"],
            "water_border_delay_cv": model.BORDER_DELAY_CV["water"],
        },
    }

    (OUT_DIR / "sanity_10_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8")
    print(f"[SANITY] Wrote {OUT_DIR / 'sanity_10_report.json'}")
    print(f"[SANITY] Representative batch: {rep_batch.batch_id} "
          f"{rep_batch.origin}->{rep_batch.destination} "
          f"modes={rep_path.modes}")
    print(f"[SANITY] EV objectives: {ev_ccp['ev']['objectives']}")
    print(f"[SANITY] CCP objectives: {ev_ccp['ccp']['objectives']}")
    print(f"[SANITY] Reproducible (arrays+trace): "
          f"{repro_ok and trace_identical}  "
          f"repeat_used_rebuilt_object={repeat_used_rebuilt_object}  "
          f"aggregates_identical={aggregates_identical}")
    print(f"[SANITY] Same frozen scenario set for EV/CCP: "
          f"{report['shared_scenario_check']['same_object_identity']}")
    print(f"[SANITY] State isolation (border_capacity/background_flow/"
          f"tt_dict) OK: {state_isolation_ok}")
    print(f"[SANITY] Alternate-set (seed=43) evaluation genuinely used the "
          f"alternate object: {alt_evaluation_used_alternate_set}")
    print(f"[SANITY] Original scenario arrays unchanged after alternate-set "
          f"use: {original_arrays_unchanged}  "
          f"(travel={original_travel_unchanged} "
          f"border={original_border_unchanged})")
    print(f"[SANITY] Restored original reproduces first evaluation: "
          f"{restored_matches_first_original_evaluation}")
    print(f"[SANITY] Water border delay arrays all zero: "
          f"{water_delay_arrays_zero} (events={list(water_events.keys())})")


if __name__ == "__main__":
    main()
