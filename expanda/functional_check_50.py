#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""S=50 functional uncertainty experiment.

DIAGNOSTIC / REPORTING SCRIPT ONLY -- imports and calls the REAL,
already-checkpointed production functions in baseline_uncertainty.py, and
reuses the already-Codex-audited helper functions from sanity_check_10.py
(load_everything, pick_representative_allocation, instrumented_trace,
run_ev_and_ccp, the float-tolerance comparison helpers) rather than
reimplementing them. Does NOT modify baseline_uncertainty.py or
sanity_check_10.py.

Does NOT run NSGA-II. Does NOT run population-based optimisation. Does
NOT run 30 independent runs, algorithm comparisons, or S=100/200.

The full-solution per-scenario cost/emission/makespan arrays
(full_solution_scenario_arrays) mirror evaluate_individual's own
accumulation formula field-for-field (same deterministic transport cost,
carbon cost, schedule-wait cost/emission, border-delay processing cost,
late-penalty cost, transshipment cost) because evaluate_individual itself
does not expose the raw per-scenario arrays outside its local scope (only
the reduced EV mean / CCP quantile). This reconstruction is CROSS-VALIDATED
against the real evaluate_individual's own EV-mode scenario_cost_mean /
scenario_emission_mean / scenario_time_mean at runtime (see
`assert_cross_validated`) -- the script aborts loudly rather than report
silently-wrong numbers if the reconstruction and the real function ever
disagree beyond floating-point tolerance.
"""
import csv
import json
import sys
from pathlib import Path as FSPath

import numpy as np

sys.path.insert(0, str(FSPath(__file__).resolve().parent))
import baseline_uncertainty as model  # noqa: E402
import sanity_check_10 as sc10  # noqa: E402  (reuse audited helpers)

HERE = FSPath(__file__).resolve().parent
OUT_DIR = HERE / "functional_50"
SEED = 42
S = 50


def full_solution_scenario_arrays(env, individual, batches, arcs, tt_dict,
                                   scenario_set):
    """Reconstruct the FULL-solution per-scenario cost_s/emission_s/
    makespan_s arrays, field-for-field identical to evaluate_individual's
    own accumulation (see module docstring). Returns (cost_s, emission_s,
    makespan_s, diagnostics)."""
    size = scenario_set.size
    cost_s = np.zeros(size, dtype=float)
    emission_s = np.zeros(size, dtype=float)
    makespan_s = np.zeros(size, dtype=float)
    node_hold_cost = env["node_hold_cost"]
    node_proc_cost = env["node_proc_cost"]
    carbon_tax_map = env["carbon_tax_map"]
    trans_map = env["trans_map"]
    border_delay_map = env["border_delay_map"]
    theta_rm = env["theta_rm"]
    node_trans_cost = env["node_trans_cost"]
    waiting_cost_per_teu_h = env["waiting_cost_per_teu_h"]
    wait_emis_g_per_teu_h = env["wait_emis_g_per_teu_h"]

    missed_departure_events = []  # (batch_id, scenario, node, mode)

    for batch in batches:
        key = (batch.origin, batch.destination, batch.batch_id)
        allocs = [a for a in individual.od_allocations.get(key, [])
                  if a.share > 1e-12]
        if not allocs:
            continue
        batch_arrival_s = np.full(size, batch.ET, dtype=float)
        y_jmn_k = {}
        for alloc in allocs:
            flow = alloc.share * batch.quantity
            path = alloc.path

            path_result = model.simulate_path_over_scenarios(
                path, batch, tt_dict, trans_map, border_delay_map,
                scenario_set)
            batch_arrival_s = np.maximum(
                batch_arrival_s, path_result.arrival_h)

            cost_s += path.base_cost_per_teu * flow
            emission_s += path.base_emission_per_teu * flow

            carbon_cost = 0.0
            for arc in path.arcs:
                region = getattr(arc, "from_region", "")
                if theta_rm.get((region, arc.mode), 1) == 0:
                    continue
                tax_rate = float(carbon_tax_map.get(region, 0.0))
                emission_tons = (
                    arc.emission_per_teu_km * arc.distance * flow / 1e6)
                carbon_cost += emission_tons * tax_rate
            cost_s += carbon_cost

            for node, wait_values in path_result.schedule_wait_h.items():
                hold_rate = node_hold_cost.get(node, waiting_cost_per_teu_h)
                cost_s += hold_rate * flow * wait_values
                emission_s += wait_emis_g_per_teu_h * flow * wait_values

            for node, delay_values in path_result.border_delay_h.items():
                proc_rate = node_proc_cost.get(node, 0.0)
                cost_s += proc_rate * flow * delay_values

            path_lateness = np.maximum(0.0, path_result.arrival_h - batch.LT)
            cost_s += batch.penalty_per_teu_h * flow * path_lateness

            for i in range(len(path.arcs) - 1):
                if path.arcs[i].mode != path.arcs[i + 1].mode:
                    node = path.arcs[i + 1].from_node
                    m_in, m_out = path.arcs[i].mode, path.arcs[i + 1].mode
                    y_jmn_k[(node, m_in, m_out)] = (
                        y_jmn_k.get((node, m_in, m_out), 0.0) + flow)

            # Missed-departure aggregation for this batch/path, reusing the
            # already-audited sc10 replay logic.
            trace = sc10.instrumented_trace(
                path, batch, tt_dict, trans_map, scenario_set)
            for tr in trace:
                for ev in tr["node_events"]:
                    if ev.get("missed_nominal_departure"):
                        missed_departure_events.append(dict(
                            batch_id=batch.batch_id, scenario=tr["scenario_id"],
                            node=ev["node"], to_node=ev["to_node"],
                            mode=ev["mode"],
                            border_delay_h=ev.get("border_delay_h", 0.0)))

        for (node, m_in, m_out), transfer_flow in y_jmn_k.items():
            rec = trans_map.get((node, m_in, m_out), {})
            unit_cost = model.safe_float(rec.get("cost_per_teu"), 0.0)
            transfer_h = model.safe_float(rec.get("time_h"), 0.0)
            cost_s += unit_cost * transfer_flow
            cost_s += node_trans_cost.get(node, 0.0) * transfer_flow * transfer_h

        makespan_s = np.maximum(makespan_s, batch_arrival_s)

    return cost_s, emission_s, makespan_s, missed_departure_events


def assert_cross_validated(reconstructed_mean, real_mean, label, tol=1e-6):
    rel_tol = max(tol, abs(real_mean) * 1e-9)
    if abs(reconstructed_mean - real_mean) > rel_tol:
        raise AssertionError(
            f"Cross-validation FAILED for {label}: reconstructed "
            f"mean={reconstructed_mean!r} vs real evaluate_individual "
            f"mean={real_mean!r} (diff={abs(reconstructed_mean-real_mean)!r} "
            f"> tol={rel_tol!r}). Refusing to report unverified numbers.")


def quantiles_summary(arr):
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return dict(mean=None, std=None, min=None, max=None,
                     p5=None, p50=None, p95=None, n_finite=0, n_total=len(arr))
    return dict(
        mean=float(np.mean(finite)), std=float(np.std(finite)),
        min=float(np.min(finite)), max=float(np.max(finite)),
        p5=float(np.percentile(finite, 5)),
        p50=float(np.percentile(finite, 50)),
        p95=float(np.percentile(finite, 95)),
        n_finite=int(finite.size), n_total=int(len(arr)),
    )


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    env = sc10.load_everything()
    arcs, batches, tt_dict = env["arcs"], env["batches"], env["tt_dict"]
    path_lib = env["path_lib"]

    # SAME fixed candidate solution as the S=10 sanity check: identical,
    # deterministic construction (greedy_initial_individual on the same
    # batches/path_lib, itself built with the same fixed random.seed(0)/
    # np.random.seed(0) inside load_everything) -- reproduces the exact
    # same individual, byte for byte, without needing to serialize it.
    individual = model.greedy_initial_individual(batches, path_lib)
    model.repair_missing_allocations(individual, batches, path_lib)
    rep_batch, rep_path = sc10.pick_representative_allocation(
        individual, batches)
    assert rep_batch.batch_id == 1 and rep_path.nodes == [
        "Xi'an", "Erenhot", "Moscow", "Malaszewicze", "Duisburg", "Berlin"], (
        "Representative batch/path differs from the S=10 sanity check -- "
        "the 'same fixed candidate solution' guarantee has been violated.")

    border_event_definitions = model.load_border_event_definitions(
        model.DEFAULT_BORDER_EVENT_DATA_FILE)

    # --- A. Deterministic mode (S=1, stochastic=False) ---
    previous_mode = model.RISK_METRIC
    model.RISK_METRIC = "deterministic"
    det_scenario_set = model.configure_scenario_set(
        arcs=arcs, border_delay_map=env["border_delay_map"], size=1,
        seed=SEED, stochastic=False,
        border_event_definitions=border_event_definitions)
    det_ind = model.Individual(od_allocations={
        key: list(a) for key, a in individual.od_allocations.items()})
    model.evaluate_individual(
        det_ind, batches, arcs, tt_dict,
        waiting_cost_per_teu_h=env["waiting_cost_per_teu_h"],
        wait_emis_g_per_teu_h=env["wait_emis_g_per_teu_h"],
        node_hold_cost=env["node_hold_cost"], node_proc_cost=env["node_proc_cost"],
        carbon_tax_map=env["carbon_tax_map"], trans_map=env["trans_map"],
        border_delay_map=env["border_delay_map"], theta_rm=env["theta_rm"],
        node_trans_cost=env["node_trans_cost"])
    det_trace = sc10._replay_one_scenario(
        rep_path, rep_batch, tt_dict, env["trans_map"], det_scenario_set,
        scenario=0)
    det_arrival = det_trace[-1]["node_arrival_h"] if det_trace else None
    det_lateness = (max(0.0, det_arrival - rep_batch.LT)
                     if det_arrival is not None else None)
    model.RISK_METRIC = previous_mode

    # --- Frozen S=50 scenario set, shared by EV and CCP ---
    scenario_set = model.configure_scenario_set(
        arcs=arcs, border_delay_map=env["border_delay_map"], size=S,
        seed=SEED, stochastic=True,
        border_event_definitions=border_event_definitions)
    frozen_id = id(model.ACTIVE_SCENARIO_SET)

    ev_ccp = sc10.run_ev_and_ccp(
        env, individual, batches, arcs, tt_dict, scenario_set)
    model.ACTIVE_SCENARIO_SET = scenario_set  # restore after run_ev_and_ccp

    # --- Full-solution per-scenario arrays (reconstructed, cross-validated) ---
    cost_s, emission_s, makespan_s, missed_events = \
        full_solution_scenario_arrays(
            env, individual, batches, arcs, tt_dict, scenario_set)
    assert_cross_validated(
        float(np.mean(cost_s)), ev_ccp["ev"]["vio_breakdown"]["scenario_cost_mean"],
        "full-solution cost_s mean vs evaluate_individual scenario_cost_mean")
    assert_cross_validated(
        float(np.mean(emission_s)),
        ev_ccp["ev"]["vio_breakdown"]["scenario_emission_mean"],
        "full-solution emission_s mean vs evaluate_individual scenario_emission_mean")
    assert_cross_validated(
        float(np.mean(makespan_s)),
        ev_ccp["ev"]["vio_breakdown"]["scenario_time_mean"],
        "full-solution makespan_s mean vs evaluate_individual scenario_time_mean")

    # --- Representative-batch per-scenario arrival/lateness ---
    rep_result = model.simulate_path_over_scenarios(
        rep_path, rep_batch, tt_dict, env["trans_map"],
        env["border_delay_map"], scenario_set)
    rep_arrival_s = rep_result.arrival_h
    rep_lateness_s = np.maximum(0.0, rep_arrival_s - rep_batch.LT)

    # --- Scenario-level CCP satisfaction indicator (representative batch) ---
    rep_satisfied = rep_arrival_s <= rep_batch.LT

    # --- D. Missed-departure aggregation ---
    stage_counts = {}
    border_delays_at_missed = []
    for e in missed_events:
        stage = f"{e['node']}->{e['to_node']} ({e['mode']})"
        stage_counts[stage] = stage_counts.get(stage, 0) + 1
        border_delays_at_missed.append(e["border_delay_h"])
    n_scenarios_with_any_missed = len({
        (e["batch_id"], e["scenario"]) for e in missed_events})
    n_batch_scenario_pairs = len(batches) * S

    # --- F. Reproducibility: independent rebuild ---
    scenario_set_repeat = model.build_scenario_set(
        arcs=arcs, border_delay_map=env["border_delay_map"], size=S,
        seed=SEED, stochastic=True,
        border_event_definitions=border_event_definitions)
    assert scenario_set_repeat is not scenario_set
    repro_ok = True
    for key, arr in scenario_set.travel_multiplier.items():
        arr2 = scenario_set_repeat.travel_multiplier.get(key)
        repro_ok = repro_ok and arr2 is not None and np.array_equal(arr, arr2)
    for key, arr in scenario_set.border_delay_h.items():
        arr2 = scenario_set_repeat.border_delay_h.get(key)
        repro_ok = repro_ok and arr2 is not None and np.array_equal(arr, arr2)
    ev_ccp_repeat = sc10.run_ev_and_ccp(
        env, individual, batches, arcs, tt_dict, scenario_set_repeat)
    repeat_used_rebuilt_object = (
        ev_ccp_repeat["ev"]["active_scenario_set_id"]
        == ev_ccp_repeat["ccp"]["active_scenario_set_id"]
        == id(scenario_set_repeat) != frozen_id)
    aggregates_identical = {
        mode: (
            sc10._tuples_close(ev_ccp[mode]["objectives"],
                                ev_ccp_repeat[mode]["objectives"])
            and sc10._close(ev_ccp[mode]["penalty"], ev_ccp_repeat[mode]["penalty"])
            and ev_ccp[mode]["feasible"] == ev_ccp_repeat[mode]["feasible"]
            and sc10._dicts_close(ev_ccp[mode]["batch_on_time_prob"],
                                   ev_ccp_repeat[mode]["batch_on_time_prob"])
        )
        for mode in ("ev", "ccp")
    }
    model.ACTIVE_SCENARIO_SET = scenario_set  # restore

    # --- Manifest via the REAL production build_scenario_manifest ---
    manifest_ev = model.build_scenario_manifest(
        risk_metric="ev", scenario_set=scenario_set,
        confidence_cost=model.CONFIDENCE_COST,
        confidence_emission=model.CONFIDENCE_EMISSION,
        confidence_time=model.CONFIDENCE_TIME,
        confidence_ontime=model.CONFIDENCE_ONTIME,
        mode_time_cv=model.MODE_TIME_CV,
        mode_time_cap_factor=model.MODE_TIME_MAX_FACTOR,
        mode_speed_kmh=env["mode_speeds_map"],
        mode_speed_source=env["mode_speed_source"],
        mode_speed_requested_cli_override_kmh={
            "road": None, "rail": None, "water": None},
        border_delay_cv=model.BORDER_DELAY_CV,
        border_delay_cap_factor=model.BORDER_DELAY_MAX_FACTOR,
        border_event_data_file=str(model.DEFAULT_BORDER_EVENT_DATA_FILE),
        border_event_definitions=border_event_definitions,
        max_late_ratio=model.MAX_LATE_RATIO,
        max_late_h_override=model.MAX_LATE_H_OVERRIDE,
        late_penalty_source="workbook_default",
        late_penalty_input_basis="hourly_rate",
        late_penalty_baseline_usd_per_teu_h=model.DEFAULT_LATE_PENALTY_USD_PER_TEU_H,
        use_input_late_penalties=False,
        payload_tonnes_per_teu=model.PAYLOAD_TONNES_PER_TEU,
        wait_emission_g_per_teu_h=env["wait_emis_g_per_teu_h"],
        feasibility_seed_fraction=0.0,
        feasibility_search_restarts=0,
        feasibility_search_iterations=0,
    )

    # --- Maritime rule check ---
    water_events = {
        "|".join(k): v for k, v in scenario_set.border_event_mean_h.items()
        if k[2] == "water"}
    water_delay_zero = all(
        bool(np.all(scenario_set.border_delay_h.get(k, np.zeros(S)) == 0.0))
        for k in scenario_set.border_event_mean_h if k[2] == "water")

    # ============================= Save evidence =============================
    with (OUT_DIR / "scenario_level_results.csv").open("w", newline="",
                                                          encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "scenario_id", "rep_travel_multipliers_5arcs", "rep_border_delay_h_sum",
            "rep_final_arrival_h", "rep_lateness_h", "rep_satisfies_LT",
            "full_solution_total_cost", "full_solution_emissions_g",
            "full_solution_makespan_h",
        ])
        for s in range(S):
            mults = [round(scenario_set.travel(arc, s), 6) for arc in rep_path.arcs]
            border_sum = sum(
                float(scenario_set.border_for_arc(arc, s)[1]) for arc in rep_path.arcs)
            writer.writerow([
                s, mults, round(border_sum, 6),
                round(float(rep_arrival_s[s]), 6), round(float(rep_lateness_s[s]), 6),
                bool(rep_satisfied[s]),
                round(float(cost_s[s]), 6), round(float(emission_s[s]), 6),
                round(float(makespan_s[s]), 6),
            ])

    ev_stats = dict(
        arrival_h=quantiles_summary(rep_arrival_s),
        lateness_h=quantiles_summary(rep_lateness_s),
        full_solution_cost=quantiles_summary(cost_s),
        full_solution_emissions=quantiles_summary(emission_s),
        full_solution_makespan=quantiles_summary(makespan_s),
    )

    summary = {
        "config": {
            "scenario_count": S, "seed": SEED,
            "data_file": str(sc10.DATA_FILE),
            "risk_metrics_evaluated": ["deterministic", "ev", "ccp"],
            "mode_time_cv": model.MODE_TIME_CV,
            "border_delay_cv": model.BORDER_DELAY_CV,
            "confidence_ontime": model.CONFIDENCE_ONTIME,
            "confidence_cost": model.CONFIDENCE_COST,
            "confidence_emission": model.CONFIDENCE_EMISSION,
            "confidence_time": model.CONFIDENCE_TIME,
            "epsilon_ontime": 1.0 - model.CONFIDENCE_ONTIME,
            "representative_batch_id": rep_batch.batch_id,
            "representative_od": [rep_batch.origin, rep_batch.destination],
            "representative_path_nodes": rep_path.nodes,
            "representative_path_modes": rep_path.modes,
            "effective_mode_speed_kmh": env["mode_speeds_map"],
            "effective_mode_speed_source": env["mode_speed_source"],
        },
        "A_deterministic": {
            "objectives_cost_emission_makespan": list(det_ind.objectives),
            "feasible": det_ind.feasible,
            "feasible_hard": det_ind.feasible_hard,
            "representative_arrival_h": det_arrival,
            "representative_lateness_h": det_lateness,
        },
        "B_expected_value": {
            "mean_cost": ev_ccp["ev"]["objectives"][0],
            "mean_emissions": ev_ccp["ev"]["objectives"][1],
            "mean_makespan": ev_ccp["ev"]["objectives"][2],
            "representative_mean_arrival_h": ev_stats["arrival_h"]["mean"],
            "representative_mean_lateness_h": ev_stats["lateness_h"]["mean"],
            "distributions": ev_stats,
        },
        "C_ccp": {
            "objectives_cost_emission_makespan": ev_ccp["ccp"]["objectives"],
            "confidence_ontime": model.CONFIDENCE_ONTIME,
            "epsilon_ontime": 1.0 - model.CONFIDENCE_ONTIME,
            "representative_batch": {
                "satisfied_count": int(np.sum(rep_satisfied)),
                "violated_count": int(S - np.sum(rep_satisfied)),
                "empirical_satisfaction_probability": float(np.mean(rep_satisfied)),
                "passes_chance_constraint": bool(
                    np.mean(rep_satisfied) >= model.CONFIDENCE_ONTIME),
            },
            "full_solution": {
                "min_on_time_prob": ev_ccp["ccp"]["vio_breakdown"]["min_on_time_prob"],
                "chance_vio": ev_ccp["ccp"]["vio_breakdown"]["chance_vio"],
                "max_late_excess_h": ev_ccp["ccp"]["vio_breakdown"]["max_late_excess_h"],
                "penalty": ev_ccp["ccp"]["penalty"],
                "normalized_violation": ev_ccp["ccp"]["normalized_violation"],
                "feasible": ev_ccp["ccp"]["feasible"],
                "feasible_hard": ev_ccp["ccp"]["feasible_hard"],
            },
        },
        "D_scenario_behaviour": {
            "n_missed_departure_events": len(missed_events),
            "n_distinct_batch_scenario_pairs_with_a_missed_departure":
                n_scenarios_with_any_missed,
            "n_batch_scenario_pairs_total": n_batch_scenario_pairs,
            "missed_departure_share_of_batch_scenario_pairs": (
                n_scenarios_with_any_missed / n_batch_scenario_pairs
                if n_batch_scenario_pairs else None),
            "missed_departures_by_stage": dict(
                sorted(stage_counts.items(), key=lambda kv: -kv[1])),
            "border_delay_h_at_missed_departures": dict(
                mean=(float(np.mean(border_delays_at_missed))
                      if border_delays_at_missed else None),
                min=(float(np.min(border_delays_at_missed))
                     if border_delays_at_missed else None),
                max=(float(np.max(border_delays_at_missed))
                     if border_delays_at_missed else None),
                n=len(border_delays_at_missed)),
        },
        "F_reproducibility": {
            "scenario_arrays_identical": repro_ok,
            "repeat_used_independently_rebuilt_object": repeat_used_rebuilt_object,
            "aggregate_results_identical": aggregates_identical,
        },
        "G_evidence": {
            "shared_scenario_set_for_ev_and_ccp": (
                ev_ccp["ev"]["active_scenario_set_id"]
                == ev_ccp["ccp"]["active_scenario_set_id"] == frozen_id),
            "manifest": manifest_ev,
        },
        "maritime_rule_check": {
            "water_border_events_found": water_events,
            "all_water_border_delay_arrays_zero": water_delay_zero,
            "water_travel_time_cv": model.MODE_TIME_CV["water"],
            "water_border_delay_cv": model.BORDER_DELAY_CV["water"],
        },
        "cross_validation": {
            "full_solution_cost_mean_matches_evaluate_individual": True,
            "full_solution_emission_mean_matches_evaluate_individual": True,
            "full_solution_makespan_mean_matches_evaluate_individual": True,
            "note": ("Assertions above already passed (script would have "
                     "raised AssertionError and aborted before writing "
                     "this file otherwise)."),
        },
    }

    (OUT_DIR / "aggregate_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8")
    (OUT_DIR / "manifest.json").write_text(
        json.dumps(manifest_ev, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8")

    print(f"[FUNC50] Wrote {OUT_DIR}/aggregate_summary.json, "
          f"scenario_level_results.csv, manifest.json")
    print(f"[FUNC50] A. Deterministic objectives: {det_ind.objectives}  "
          f"feasible={det_ind.feasible}")
    print(f"[FUNC50] B. EV objectives: {ev_ccp['ev']['objectives']}")
    print(f"[FUNC50] C. CCP objectives: {ev_ccp['ccp']['objectives']}  "
          f"rep_satisfaction={float(np.mean(rep_satisfied)):.3f} "
          f"(threshold={model.CONFIDENCE_ONTIME})")
    print(f"[FUNC50] D. Missed-departure events: {len(missed_events)} across "
          f"{n_scenarios_with_any_missed}/{n_batch_scenario_pairs} "
          f"batch-scenario pairs")
    print(f"[FUNC50] F. Reproducible: {repro_ok}  "
          f"repeat_used_rebuilt_object={repeat_used_rebuilt_object}  "
          f"aggregates_identical={aggregates_identical}")
    print(f"[FUNC50] Same frozen scenario set for EV/CCP: "
          f"{summary['G_evidence']['shared_scenario_set_for_ev_and_ccp']}")
    print(f"[FUNC50] Water border delay arrays all zero: {water_delay_zero} "
          f"(events={list(water_events.keys())})")


if __name__ == "__main__":
    main()
