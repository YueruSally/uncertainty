#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Feasible-reference check: construct ONE deterministic-feasible candidate
solution (via the existing, checkpointed capacity-aware min-conflicts
search -- NOT NSGA-II, no population), then evaluate that SAME solution
under deterministic / EV(S=50) / CCP(S=50) using the already-validated
S=50 scenario framework.

DIAGNOSTIC / REPORTING SCRIPT ONLY. Reuses the already-checkpointed and
Codex-audited helpers from sanity_check_10.py and functional_check_50.py
rather than reimplementing them. Does NOT modify baseline_uncertainty.py,
sanity_check_10.py, or functional_check_50.py.

Does NOT run NSGA-II, population-based optimisation, S=100/200, or 30
independent runs.
"""
import json
import sys
from pathlib import Path as FSPath

import numpy as np

sys.path.insert(0, str(FSPath(__file__).resolve().parent))
import baseline_uncertainty as model  # noqa: E402
import sanity_check_10 as sc10  # noqa: E402
import functional_check_50 as func50  # noqa: E402

HERE = FSPath(__file__).resolve().parent
OUT_DIR = HERE / "feasible_reference_50"
SEED = 42
S = 50


def build_deterministic_feasible_reference(env, batches, path_lib, arcs,
                                            tt_dict, border_event_definitions,
                                            scenario_set_for_diagnostics):
    """Construct one candidate solution using the EXISTING, checkpointed
    capacity-aware min-conflicts search (find_capacity_aware_choice /
    capacity_aware_initial_individual) -- not NSGA-II, no population.

    reliable_options is built with mode="deterministic" so the screen is
    ONLY the genuinely hard requirements (timetable validity + nominal
    capacity footprint), NOT the CCP-only probability/max-lateness screen
    -- exactly matching the approved mode-aware gating in
    build_reliable_path_options. No constraint is weakened: this simply
    selects which real hard constraints apply to the SEARCH's screening
    step; the resulting individual is independently re-checked against
    the full, unmodified evaluate_individual()'s hard_ok afterward.
    """
    reliable_options = model.build_reliable_path_options(
        batches, path_lib, tt_dict, env["trans_map"], env["border_delay_map"],
        scenario_set_for_diagnostics, mode="deterministic")
    arc_caps = {(a.from_node, a.to_node, a.mode): a.capacity for a in arcs}

    # Existing function's own default restarts/iterations first (no
    # parameter weakening); only escalate restarts if genuinely needed,
    # using the SAME existing search mechanism.
    individual, excess = model.capacity_aware_initial_individual(
        batches, reliable_options, arc_caps)
    attempts_log = [dict(restarts=model.FEASIBILITY_SEARCH_RESTARTS,
                          iterations=model.FEASIBILITY_SEARCH_ITERATIONS,
                          excess=excess)]
    escalation = 1
    while (individual is None or excess > 1e-9) and escalation <= 4:
        escalation += 1
        restarts = model.FEASIBILITY_SEARCH_RESTARTS * escalation
        choice, excess = model.find_capacity_aware_choice(
            batches, reliable_options, arc_caps,
            restarts=restarts, iterations=model.FEASIBILITY_SEARCH_ITERATIONS)
        attempts_log.append(dict(restarts=restarts,
                                  iterations=model.FEASIBILITY_SEARCH_ITERATIONS,
                                  excess=excess))
        if choice is not None:
            individual = model.Individual()
            for batch, option_index in zip(batches, choice):
                key = (batch.origin, batch.destination, batch.batch_id)
                option = reliable_options[key][option_index]
                individual.od_allocations[key] = [
                    model.PathAllocation(path=option.path, share=1.0)]
    return individual, excess, reliable_options, attempts_log


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    env = sc10.load_everything()
    arcs, batches, tt_dict = env["arcs"], env["batches"], env["tt_dict"]
    path_lib = env["path_lib"]

    border_event_definitions = model.load_border_event_definitions(
        model.DEFAULT_BORDER_EVENT_DATA_FILE)

    # The validated, frozen S=50 scenario set (seed=42) -- SAME framework
    # as functional_check_50.py, not modified, not regenerated with
    # different parameters.
    scenario_set = model.configure_scenario_set(
        arcs=arcs, border_delay_map=env["border_delay_map"], size=S,
        seed=SEED, stochastic=True,
        border_event_definitions=border_event_definitions)
    frozen_id = id(model.ACTIVE_SCENARIO_SET)

    # --- Step 1: deterministic-feasible reference solution ---
    individual, search_excess, reliable_options, attempts_log = \
        build_deterministic_feasible_reference(
            env, batches, path_lib, arcs, tt_dict, border_event_definitions,
            scenario_set)
    construction_found_zero_excess_choice = (
        individual is not None and search_excess <= 1e-9)

    # Independent, authoritative re-check via the REAL, unmodified
    # evaluate_individual() in deterministic mode -- this is the actual
    # feasibility ground truth, not the search heuristic's own score.
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
    model.RISK_METRIC = previous_mode

    genuinely_deterministic_feasible = bool(det_ind.feasible_hard)
    if not genuinely_deterministic_feasible:
        # Do NOT weaken constraints or silently accept an infeasible
        # reference -- report the failure honestly and stop before
        # proceeding to Step 2, per "do not weaken constraints to obtain
        # feasibility".
        report_failure = {
            "step1_result": "NO_FEASIBLE_REFERENCE_FOUND",
            "search_excess": search_excess,
            "attempts_log": attempts_log,
            "evaluate_individual_vio_breakdown": det_ind.vio_breakdown,
            "evaluate_individual_feasible_hard": det_ind.feasible_hard,
        }
        (OUT_DIR / "step1_failure_report.json").write_text(
            json.dumps(report_failure, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8")
        print("[FEAS-REF] FAILED to construct a genuinely deterministic-"
              "feasible reference solution with the existing capacity-aware "
              "search. See step1_failure_report.json. Stopping without "
              "proceeding to Step 2 -- constraints were not weakened.")
        return

    path_allocations_report = {
        f"{origin}->{dest} (batch {bid})": dict(
            path_nodes=allocs[0].path.nodes, path_modes=allocs[0].path.modes,
            share=allocs[0].share)
        for (origin, dest, bid), allocs in individual.od_allocations.items()
    }

    # --- Step 2: evaluate the SAME solution under deterministic/EV/CCP ---
    ev_ccp = sc10.run_ev_and_ccp(
        env, individual, batches, arcs, tt_dict, scenario_set)
    model.ACTIVE_SCENARIO_SET = scenario_set

    cost_s, emission_s, makespan_s, missed_events = \
        func50.full_solution_scenario_arrays(
            env, individual, batches, arcs, tt_dict, scenario_set)
    func50.assert_cross_validated(
        float(np.mean(cost_s)), ev_ccp["ev"]["vio_breakdown"]["scenario_cost_mean"],
        "reference-solution cost_s mean vs evaluate_individual scenario_cost_mean")
    func50.assert_cross_validated(
        float(np.mean(emission_s)),
        ev_ccp["ev"]["vio_breakdown"]["scenario_emission_mean"],
        "reference-solution emission_s mean vs evaluate_individual scenario_emission_mean")
    func50.assert_cross_validated(
        float(np.mean(makespan_s)),
        ev_ccp["ev"]["vio_breakdown"]["scenario_time_mean"],
        "reference-solution makespan_s mean vs evaluate_individual scenario_time_mean")

    # Aggregate per-batch lateness across the whole solution, for the EV
    # "mean lateness" requirement (batch-level lateness arrays, summed
    # per scenario across batches using the SAME simulate_path_over_scenarios
    # calls already performed inside full_solution_scenario_arrays via
    # instrumented_trace -- recomputed directly here for clarity).
    total_lateness_s = np.zeros(S, dtype=float)
    for batch in batches:
        key = (batch.origin, batch.destination, batch.batch_id)
        allocs = [a for a in individual.od_allocations.get(key, [])
                  if a.share > 1e-12]
        if not allocs:
            continue
        batch_arrival_s = np.full(S, batch.ET, dtype=float)
        for alloc in allocs:
            result = model.simulate_path_over_scenarios(
                alloc.path, batch, tt_dict, env["trans_map"],
                env["border_delay_map"], scenario_set)
            batch_arrival_s = np.maximum(batch_arrival_s, result.arrival_h)
        total_lateness_s += np.maximum(0.0, batch_arrival_s - batch.LT)

    # --- Step 3: max_late_excess_h semantics, verified against the ACTUAL
    # code (not inferred from the name) ---
    max_late_excess_semantics = {
        "raw_value_this_run": ev_ccp["ccp"]["vio_breakdown"]["max_late_excess_h"],
        "source_lines": "baseline_uncertainty.py evaluate_individual(), "
                          "inside the per-batch loop (approx. lines 2314-2320)",
        "verbatim_code": (
            "batch_lateness = np.maximum(0.0, batch_arrival_s - batch.LT)\n"
            "late_limit = batch_max_lateness_h(batch)\n"
            "batch_excess = np.maximum(0.0, batch_lateness - late_limit)\n"
            "if np.any(np.isfinite(batch_excess)):\n"
            "    max_late_excess_h += float(np.max(batch_excess))\n"
            "else:\n"
            "    max_late_excess_h = float(\"inf\")"
        ),
        "interpretation": (
            "NEITHER a single global maximum NOR a plain sum-across-scenarios. "
            "For EACH batch: batch_excess[s] = max(0, lateness_h[s] - "
            "batch's own max-allowed-lateness). Then np.max(batch_excess) "
            "takes the WORST-CASE SCENARIO for that one batch (a max over "
            "the S=50 scenario axis). That per-batch worst-case value is "
            "then ACCUMULATED WITH += ACROSS BATCHES. So the final "
            "max_late_excess_h is a SUM ACROSS BATCHES of each batch's own "
            "MAXIMUM-OVER-SCENARIOS lateness-excess-beyond-its-own-limit. "
            "It is a sum-of-per-batch-maxima, not a true single maximum, "
            "not a sum-across-scenarios, and (as stored in vio_breakdown) "
            "NOT yet penalty-scaled -- the USD scaling by "
            "PEN_MAX_LATE_EXCESS_PER_H happens separately when building "
            "`penalty`, only under CCP mode."
        ),
        "edge_case": (
            "If a batch's arrival is np.inf in EVERY scenario (e.g. that "
            "batch's allocation missed its timetable entirely), "
            "np.any(np.isfinite(batch_excess)) is False and the ENTIRE "
            "accumulator is reassigned to float('inf'), discarding whatever "
            "finite contributions had already been summed from other "
            "batches processed earlier in the loop."
        ),
        "name_assessment": (
            "The name 'max_late_excess_h' is misleading on its own: a reader "
            "would reasonably expect a single global maximum (e.g. the worst "
            "lateness-excess across the whole solution), but the quantity is "
            "actually a SUM of per-batch maxima. No production code renamed "
            "in this check, per instructions."
        ),
    }

    # --- Reproducibility of THIS reference solution's aggregates (same
    # mechanism already validated in the S=10/S=50 rounds) ---
    scenario_set_repeat = model.build_scenario_set(
        arcs=arcs, border_delay_map=env["border_delay_map"], size=S,
        seed=SEED, stochastic=True,
        border_event_definitions=border_event_definitions)
    assert scenario_set_repeat is not scenario_set
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
        )
        for mode in ("ev", "ccp")
    }
    model.ACTIVE_SCENARIO_SET = scenario_set

    ev_stats = dict(
        total_lateness_h=func50.quantiles_summary(total_lateness_s),
        full_solution_cost=func50.quantiles_summary(cost_s),
        full_solution_emissions=func50.quantiles_summary(emission_s),
        full_solution_makespan=func50.quantiles_summary(makespan_s),
    )

    report = {
        "config": {
            "scenario_count": S, "seed": SEED,
            "risk_metrics_evaluated": ["deterministic", "ev", "ccp"],
            "confidence_ontime": model.CONFIDENCE_ONTIME,
            "epsilon_ontime": 1.0 - model.CONFIDENCE_ONTIME,
            "mode_time_cv": model.MODE_TIME_CV,
            "border_delay_cv": model.BORDER_DELAY_CV,
        },
        "step1_reference_construction": {
            "method": (
                "Existing checkpointed capacity-aware min-conflicts search: "
                "build_reliable_path_options(mode='deterministic') "
                "[timetable-validity + nominal-capacity screen only, NOT "
                "CCP probability/max-lateness] -> "
                "capacity_aware_initial_individual() / "
                "find_capacity_aware_choice() (random-restart min-conflicts "
                "over each batch's independently-reliable single-path "
                "options). NOT NSGA-II; no population; no genetic operators."
            ),
            "attempts_log": attempts_log,
            "search_reported_capacity_excess": search_excess,
            "path_allocations": path_allocations_report,
            "independent_reevaluation_via_real_evaluate_individual": {
                "feasible": det_ind.feasible,
                "feasible_hard": det_ind.feasible_hard,
                "vio_breakdown": det_ind.vio_breakdown,
            },
            "genuinely_deterministic_feasible": genuinely_deterministic_feasible,
        },
        "A_deterministic": {
            "objectives_cost_emission_makespan": list(det_ind.objectives),
            "feasible": det_ind.feasible,
            "feasible_hard": det_ind.feasible_hard,
            "vio_breakdown": det_ind.vio_breakdown,
        },
        "B_expected_value": {
            "mean_cost": ev_ccp["ev"]["objectives"][0],
            "mean_emissions": ev_ccp["ev"]["objectives"][1],
            "mean_makespan": ev_ccp["ev"]["objectives"][2],
            "mean_total_lateness_h": ev_stats["total_lateness_h"]["mean"],
            "distributions": ev_stats,
        },
        "C_ccp": {
            "objectives_cost_emission_makespan": ev_ccp["ccp"]["objectives"],
            "confidence_ontime": model.CONFIDENCE_ONTIME,
            "epsilon_ontime": 1.0 - model.CONFIDENCE_ONTIME,
            "min_on_time_prob": ev_ccp["ccp"]["vio_breakdown"]["min_on_time_prob"],
            "chance_vio": ev_ccp["ccp"]["vio_breakdown"]["chance_vio"],
            "max_late_excess_h_raw": ev_ccp["ccp"]["vio_breakdown"]["max_late_excess_h"],
            "penalty": ev_ccp["ccp"]["penalty"],
            "normalized_violation": ev_ccp["ccp"]["normalized_violation"],
            "feasible": ev_ccp["ccp"]["feasible"],
            "feasible_hard": ev_ccp["ccp"]["feasible_hard"],
            "batch_on_time_prob": ev_ccp["ccp"]["batch_on_time_prob"],
            "batch_max_lateness_h": ev_ccp["ccp"]["batch_max_lateness_h"],
        },
        "step3_max_late_excess_h_semantics": max_late_excess_semantics,
        "reproducibility": {
            "repeat_used_independently_rebuilt_object": repeat_used_rebuilt_object,
            "aggregate_results_identical": aggregates_identical,
        },
        "shared_scenario_check": {
            "ev_active_scenario_set_id": ev_ccp["ev"]["active_scenario_set_id"],
            "ccp_active_scenario_set_id": ev_ccp["ccp"]["active_scenario_set_id"],
            "same_object_identity": (
                ev_ccp["ev"]["active_scenario_set_id"]
                == ev_ccp["ccp"]["active_scenario_set_id"] == frozen_id),
        },
    }

    (OUT_DIR / "feasible_reference_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8")

    print(f"[FEAS-REF] Wrote {OUT_DIR}/feasible_reference_report.json")
    print(f"[FEAS-REF] Step1: genuinely_deterministic_feasible="
          f"{genuinely_deterministic_feasible}  attempts={len(attempts_log)}  "
          f"final_search_excess={search_excess}")
    print(f"[FEAS-REF] A. Deterministic objectives: {det_ind.objectives}  "
          f"feasible={det_ind.feasible}  feasible_hard={det_ind.feasible_hard}")
    print(f"[FEAS-REF] B. EV objectives: {ev_ccp['ev']['objectives']}")
    print(f"[FEAS-REF] C. CCP objectives: {ev_ccp['ccp']['objectives']}  "
          f"min_on_time_prob={ev_ccp['ccp']['vio_breakdown']['min_on_time_prob']}  "
          f"CCP feasible={ev_ccp['ccp']['feasible']}")
    print(f"[FEAS-REF] Step3 max_late_excess_h raw value: "
          f"{ev_ccp['ccp']['vio_breakdown']['max_late_excess_h']} "
          f"(see step3_max_late_excess_h_semantics in the JSON for full "
          f"derivation)")
    print(f"[FEAS-REF] Reproducible: repeat_used_rebuilt_object="
          f"{repeat_used_rebuilt_object}  aggregates_identical={aggregates_identical}")
    print(f"[FEAS-REF] Same frozen scenario set for EV/CCP: "
          f"{report['shared_scenario_check']['same_object_identity']}")


if __name__ == "__main__":
    main()
