#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Scenario-count sensitivity analysis: S=50 vs S=100 vs S=200.

DIAGNOSTIC / REPORTING SCRIPT ONLY. Imports and calls the REAL,
already-checkpointed production functions in baseline_uncertainty.py, and
reuses the already-Codex-audited helper functions from sanity_check_10.py,
functional_check_50.py, and feasible_reference_check_50.py rather than
reimplementing them. Does NOT modify baseline_uncertainty.py or any of
those three scripts.

Does NOT run NSGA-II. Does NOT run population-based optimisation. Does
NOT run 30 independent optimisation runs, SPEA2/MOEA-D comparisons, or
any parameter tuning. Does NOT create any new optimised solution and does
NOT re-run the randomized capacity-aware search: the ONE candidate
evaluated here is LOADED DIRECTLY from the already-checkpointed
feasible_reference_50/feasible_reference_report.json (its path_nodes/
path_modes/share per batch), by looking up the matching Path object in
the current path_lib -- not by re-invoking
capacity_aware_initial_individual/find_capacity_aware_choice. This avoids
any dependency on global random/np.random module state (those functions
consume the module-level `random` generator, which this script's call
sequence does not control) and is more faithful to "load [the existing
candidate] without running a new optimisation". The loaded candidate is
independently re-verified as genuinely deterministic-feasible via the
REAL, unmodified evaluate_individual() before use; the script hard-fails
if the checkpoint report is missing/incomplete or the loaded candidate is
not feasible_hard.

Candidate-count limitation (explicit, per instructions): only ONE
genuinely deterministic-feasible fixed candidate solution exists anywhere
in this session's validated outputs (feasible_reference_50/
feasible_reference_report.json). No second or third deterministic-feasible
fixed solution exists to load without running a new optimisation search.
This analysis therefore evaluates scenario-count sensitivity for that
SINGLE candidate only -- generality across multiple candidate solutions is
NOT established here and is explicitly flagged as a limitation throughout
the outputs.

Nesting design: for each Monte-Carlo seed, build ONE master S=200 scenario
set via the real model.build_scenario_set(). S=50 and S=100 are then
derived as literal array PREFIXES (first 50 / first 100 elements) of the
master S=200 arrays -- not independently regenerated. This guarantees
S50 subset-of S100 subset-of S200 by construction (array slicing), and the
script additionally re-verifies this explicitly and independently for
every seed before using the subsets for anything.
"""
import csv
import json
import math
import sys
from pathlib import Path as FSPath

import numpy as np

sys.path.insert(0, str(FSPath(__file__).resolve().parent))
import baseline_uncertainty as model  # noqa: E402
import sanity_check_10 as sc10  # noqa: E402  (reuse audited helpers)
import functional_check_50 as func50  # noqa: E402  (reuse audited helpers)
import feasible_reference_check_50 as feasref  # noqa: E402  (reuse audited helpers)

HERE = FSPath(__file__).resolve().parent
OUT_DIR = HERE / "scenario_count_sensitivity"

SEEDS = [42, 43, 44, 45, 46]
S_VALUES = [50, 100, 200]
MASTER_S = 200
Z_95 = 1.959963984540054  # two-sided 95% normal quantile, for Wilson CI


# ════════════════════════════════════════════════════════
# Small numeric helpers
# ════════════════════════════════════════════════════════

def wilson_ci(successes: int, n: int, z: float = Z_95):
    """95% Wilson score confidence interval for a binomial proportion.
    Statistical diagnostic ONLY -- never used to alter the production CCP
    pass/fail rule, which remains min_on_time_prob >= CONFIDENCE_ONTIME
    exactly as implemented in evaluate_individual()."""
    if n <= 0:
        return None, None
    phat = successes / n
    denom = 1.0 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    margin = (z * math.sqrt((phat * (1.0 - phat) + z * z / (4 * n)) / n)) / denom
    lo = max(0.0, center - margin)
    hi = min(1.0, center + margin)
    return float(lo), float(hi)


def _stats(values):
    arr = np.asarray([v for v in values if v is not None], dtype=float)
    if arr.size == 0:
        return dict(mean=None, std=None, min=None, max=None, cv=None, n=0)
    mean = float(np.mean(arr))
    std = float(np.std(arr))
    cv = (std / mean) if abs(mean) > 1e-12 else None
    return dict(mean=mean, std=std, min=float(np.min(arr)), max=float(np.max(arr)),
                cv=cv, n=int(arr.size))


def _abs_pct(before, after):
    absdiff = after - before
    pct = (absdiff / before * 100.0) if abs(before) > 1e-12 else None
    return dict(absolute=absdiff, percent=pct)


# ════════════════════════════════════════════════════════
# Nested-subset construction and verification
# ════════════════════════════════════════════════════════

def make_nested_subset(master: model.ScenarioSet, n: int) -> model.ScenarioSet:
    """Derive a size-n subset of `master` as a literal array PREFIX (first
    n elements) -- never an independent redraw."""
    assert 1 <= n <= master.size, f"n={n} out of range for master.size={master.size}"
    return model.ScenarioSet(
        size=n,
        seed=master.seed,
        travel_multiplier={k: v[:n].copy() for k, v in master.travel_multiplier.items()},
        border_delay_h={k: v[:n].copy() for k, v in master.border_delay_h.items()},
        arc_border_event=dict(master.arc_border_event),
        border_event_mean_h=dict(master.border_event_mean_h),
        stochastic=master.stochastic,
    )


def verify_prefix_subset(small: model.ScenarioSet, large: model.ScenarioSet):
    """Independently re-verify (not merely assume) that every array in
    `small` equals the first `small.size` elements of the corresponding
    array in `large`, for travel_multiplier AND border_delay_h -- including
    that the two sets share EXACTLY the same keys (not just that every
    small-side key has a large-side match) and the same non-array metadata
    (arc_border_event, border_event_mean_h, seed, stochastic)."""
    mismatches = []
    if set(small.travel_multiplier) != set(large.travel_multiplier):
        mismatches.append("travel_multiplier key sets differ")
    if set(small.border_delay_h) != set(large.border_delay_h):
        mismatches.append("border_delay_h key sets differ")
    if small.arc_border_event != large.arc_border_event:
        mismatches.append("arc_border_event differs")
    if small.border_event_mean_h != large.border_event_mean_h:
        mismatches.append("border_event_mean_h differs")
    if small.seed != large.seed:
        mismatches.append("seed differs")
    if small.stochastic != large.stochastic:
        mismatches.append("stochastic differs")
    for key, arr in small.travel_multiplier.items():
        large_arr = large.travel_multiplier.get(key)
        ok = (large_arr is not None
              and large_arr.shape[0] >= small.size
              and np.array_equal(arr, large_arr[:small.size]))
        if not ok:
            mismatches.append(f"travel_multiplier[{key}]")
    for key, arr in small.border_delay_h.items():
        large_arr = large.border_delay_h.get(key)
        ok = (large_arr is not None
              and large_arr.shape[0] >= small.size
              and np.array_equal(arr, large_arr[:small.size]))
        if not ok:
            mismatches.append(f"border_delay_h[{key}]")
    return dict(is_prefix_subset=(len(mismatches) == 0), mismatched_keys=mismatches,
                small_size=small.size, large_size=large.size)


# ════════════════════════════════════════════════════════
# Sampling diagnostics
# ════════════════════════════════════════════════════════

def sampling_diagnostics_for_scenario_set(scen: model.ScenarioSet):
    """Empirical mean/std/CV of the configured uncertain variables, pooled
    by mode. travel_multiplier is already on the raw multiplier scale
    (mean approx. 1), so it is pooled directly. border_delay_h = base_mean_h
    * multiplier, and base_mean_h differs PER EVENT -- pooling raw
    border_delay_h values across multiple events with different means would
    mix between-event mean differences into the pooled variance (a scale
    mixture, NOT scale-invariant). Each event's array is therefore
    normalized by its own positive base mean BEFORE pooling, recovering the
    underlying lognormal multiplier's CV regardless of how many
    differently-scaled events are pooled into the same mode."""
    rows = []
    pooled_travel = {"road": [], "rail": [], "water": []}
    for key, arr in scen.travel_multiplier.items():
        mode = key[2]
        if mode in pooled_travel:
            pooled_travel[mode].append(arr)
    for mode, arrs in pooled_travel.items():
        if not arrs:
            continue
        pooled = np.concatenate(arrs)
        s = _stats(pooled.tolist())
        rows.append(dict(variable="travel_multiplier", mode=mode,
                          empirical_mean=s["mean"], empirical_std=s["std"],
                          empirical_cv=s["cv"],
                          configured_cv=model.MODE_TIME_CV.get(mode), n=s["n"]))

    pooled_border = {"road": [], "rail": [], "water": []}
    skipped_zero_mean_events = []
    for key, mean_h in scen.border_event_mean_h.items():
        mode = key[2]
        if mode not in pooled_border:
            continue
        if mean_h <= 0.0:
            skipped_zero_mean_events.append("|".join(key))
            continue
        arr = scen.border_delay_h.get(key)
        if arr is None:
            continue
        # Normalize by this event's own positive mean before pooling (see
        # docstring) -- recovers the underlying multiplier scale.
        pooled_border[mode].append(arr / mean_h)
    for mode, arrs in pooled_border.items():
        if not arrs:
            continue
        pooled = np.concatenate(arrs)
        s = _stats(pooled.tolist())
        rows.append(dict(variable="border_delay_multiplier_normalized", mode=mode,
                          empirical_mean=s["mean"], empirical_std=s["std"],
                          empirical_cv=s["cv"],
                          configured_cv=model.BORDER_DELAY_CV.get(mode), n=s["n"]))

    water_all_zero = all(
        bool(np.all(arr == 0.0))
        for key, arr in scen.border_delay_h.items() if key[2] == "water")
    return rows, water_all_zero, skipped_zero_mean_events


# ════════════════════════════════════════════════════════
# Solution-level per-scenario helper arrays (mirrors production formulas,
# same pattern already used and Codex-audited in feasible_reference_check_50.py)
# ════════════════════════════════════════════════════════

def total_lateness_array(env, individual, batches, tt_dict, scen):
    total_lateness_s = np.zeros(scen.size, dtype=float)
    for batch in batches:
        key = (batch.origin, batch.destination, batch.batch_id)
        allocs = [a for a in individual.od_allocations.get(key, [])
                  if a.share > 1e-12]
        if not allocs:
            continue
        batch_arrival_s = np.full(scen.size, batch.ET, dtype=float)
        for alloc in allocs:
            result = model.simulate_path_over_scenarios(
                alloc.path, batch, tt_dict, env["trans_map"],
                env["border_delay_map"], scen)
            batch_arrival_s = np.maximum(batch_arrival_s, result.arrival_h)
        total_lateness_s += np.maximum(0.0, batch_arrival_s - batch.LT)
    return total_lateness_s


def batch_arrival_array(env, individual, batches, tt_dict, scen, batch_id):
    batch = next(b for b in batches if b.batch_id == batch_id)
    key = (batch.origin, batch.destination, batch.batch_id)
    allocs = [a for a in individual.od_allocations.get(key, [])
              if a.share > 1e-12]
    batch_arrival_s = np.full(scen.size, batch.ET, dtype=float)
    for alloc in allocs:
        result = model.simulate_path_over_scenarios(
            alloc.path, batch, tt_dict, env["trans_map"],
            env["border_delay_map"], scen)
        batch_arrival_s = np.maximum(batch_arrival_s, result.arrival_h)
    return batch, batch_arrival_s


# ════════════════════════════════════════════════════════
# Candidate loading (DIRECT load from the ALREADY-checkpointed
# feasible_reference_check_50.py report -- no randomized search is re-run)
# ════════════════════════════════════════════════════════

CHECKPOINT_REPORT_PATH = HERE / "feasible_reference_50" / "feasible_reference_report.json"


def load_candidate_from_checkpoint(batches, path_lib):
    """Load the single validated candidate's fixed path allocations DIRECTLY
    from the already-checkpointed feasible_reference_50/
    feasible_reference_report.json (path_nodes/path_modes/share per batch),
    by matching each entry to a Path object in the CURRENT path_lib. This
    does NOT re-invoke capacity_aware_initial_individual/
    find_capacity_aware_choice (the randomized min-conflicts search), so it
    has no dependency on global random/np.random module state. Hard-fails
    (raises RuntimeError) if the checkpoint report, its path_allocations
    section, any batch's entry, or a matching Path is missing -- never
    silently proceeds with an unverified or partial candidate."""
    if not CHECKPOINT_REPORT_PATH.exists():
        raise RuntimeError(
            f"Checkpointed reference report not found at "
            f"{CHECKPOINT_REPORT_PATH}. Refusing to fabricate a candidate: "
            "only an already-validated deterministic-feasible fixed "
            "candidate may be used at this stage, loaded without running a "
            "new optimisation.")
    saved = json.loads(CHECKPOINT_REPORT_PATH.read_text(encoding="utf-8"))
    saved_alloc = saved.get("step1_reference_construction", {}).get("path_allocations")
    if not saved_alloc:
        raise RuntimeError(
            f"Checkpointed reference report at {CHECKPOINT_REPORT_PATH} has "
            "no step1_reference_construction.path_allocations. Refusing to "
            "proceed with an unverified candidate.")

    individual = model.Individual()
    for batch in batches:
        key_str = f"{batch.origin}->{batch.destination} (batch {batch.batch_id})"
        entry = saved_alloc.get(key_str)
        if entry is None:
            raise RuntimeError(
                f"Checkpointed reference report is missing an allocation "
                f"for {key_str!r}. Refusing to proceed with an incomplete "
                "candidate.")
        match = None
        for path in path_lib.get((batch.origin, batch.destination), []):
            if path.nodes == entry["path_nodes"] and path.modes == entry["path_modes"]:
                match = path
                break
        if match is None:
            raise RuntimeError(
                f"No path in the current path_lib for {batch.origin}->"
                f"{batch.destination} matches the checkpointed path_nodes/"
                f"path_modes for {key_str!r} (nodes={entry['path_nodes']}, "
                f"modes={entry['path_modes']}). The path library may have "
                "changed since the checkpoint was created. Refusing to "
                "proceed with an unverified candidate.")
        okey = (batch.origin, batch.destination, batch.batch_id)
        individual.od_allocations[okey] = [
            model.PathAllocation(path=match, share=float(entry["share"]))]
    return individual, saved_alloc


def verify_candidate_feasibility(env, individual, batches, arcs, tt_dict,
                                  border_event_definitions):
    """Independently re-verify (via the REAL, unmodified evaluate_individual
    in deterministic mode) that the loaded candidate is still genuinely
    deterministic-feasible under the current code/data -- a pure re-check,
    not a re-derivation, and no constraint is weakened or bypassed.
    RISK_METRIC is restored via try/finally so a mid-evaluation exception
    cannot leave the global mode mutated."""
    previous_mode = model.RISK_METRIC
    try:
        model.RISK_METRIC = "deterministic"
        model.configure_scenario_set(
            arcs=arcs, border_delay_map=env["border_delay_map"], size=1,
            seed=feasref.SEED, stochastic=False,
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
    finally:
        model.RISK_METRIC = previous_mode
    return det_ind


# ════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    env = sc10.load_everything()
    arcs, batches, tt_dict = env["arcs"], env["batches"], env["tt_dict"]
    path_lib = env["path_lib"]
    border_event_definitions = model.load_border_event_definitions(
        model.DEFAULT_BORDER_EVENT_DATA_FILE)

    # --- Candidate: the SINGLE validated deterministic-feasible reference,
    # loaded DIRECTLY from the checkpoint (no randomized search re-run). ---
    individual, saved_alloc = load_candidate_from_checkpoint(batches, path_lib)
    det_ind = verify_candidate_feasibility(
        env, individual, batches, arcs, tt_dict, border_event_definitions)
    genuinely_deterministic_feasible = bool(det_ind.feasible_hard)
    if not genuinely_deterministic_feasible:
        raise RuntimeError(
            "Candidate loaded from the checkpointed feasible_reference_50 "
            "report is NOT genuinely deterministic-feasible under the "
            "current code/data (feasible_hard=False). Refusing to proceed "
            f"with a weakened or infeasible reference solution. "
            f"vio_breakdown: {det_ind.vio_breakdown}")

    path_allocations_report = {
        f"{origin}->{dest} (batch {bid})": dict(
            path_nodes=allocs[0].path.nodes, path_modes=allocs[0].path.modes,
            share=allocs[0].share)
        for (origin, dest, bid), allocs in individual.od_allocations.items()
    }
    identical_to_checkpoint = (path_allocations_report == saved_alloc)
    if not identical_to_checkpoint:
        # Structurally this should be impossible -- the individual was built
        # directly FROM saved_alloc above -- so this guards against a
        # serialization/rounding round-trip mismatch, not against a search
        # producing a different candidate (there is no search here anymore).
        mismatched = sorted(
            k for k in (set(saved_alloc) | set(path_allocations_report))
            if saved_alloc.get(k) != path_allocations_report.get(k))
        raise RuntimeError(
            "Internal consistency check failed: the loaded candidate does "
            f"not round-trip back to identical path_allocations. Mismatched "
            f"keys: {mismatched}. Refusing to proceed with an unverified "
            "candidate.")
    print(f"[SCS] Candidate loaded from checkpoint: "
          f"genuinely_deterministic_feasible={genuinely_deterministic_feasible}  "
          f"identical_to_checkpoint={identical_to_checkpoint}")

    candidate_id = "reference_1"
    candidate_limitation_note = (
        "Only ONE genuinely deterministic-feasible fixed candidate solution "
        "exists in previously validated outputs (feasible_reference_50/"
        "feasible_reference_report.json). No second or third such candidate "
        "was available to load without running a new optimisation search, "
        "per the explicit instruction not to create new optimised solutions "
        "for this stage. This scenario-count sensitivity analysis therefore "
        "covers ONE candidate only; generality across multiple candidate "
        "solutions or across the Pareto front is NOT established by this "
        "analysis.")

    # --- Deterministic baseline (S-independent anchor, same candidate) ---
    deterministic_baseline = dict(
        objectives_cost_emission_makespan=list(det_ind.objectives),
        feasible=det_ind.feasible, feasible_hard=det_ind.feasible_hard,
        vio_breakdown=det_ind.vio_breakdown,
    )

    per_row_records = []       # per_candidate_seed_S.csv
    ccp_prob_records = []      # ccp_probability_summary.csv
    sampling_records = []      # sampling_diagnostics.csv
    subset_check_by_seed = {}
    ev_paired_by_seed = {}
    ccp_stability_by_seed = {}
    cross_validation_failures = []
    representative_manifest = None
    cache_safety_spot_check = dict(
        seed=SEEDS[0], S=50, ev_objectives_match=None, ccp_objectives_match=None,
        batch_on_time_prob_match=None, min_on_time_prob_match=None,
        feasible_hard_match=None)

    for seed in SEEDS:
        master = model.build_scenario_set(
            arcs=arcs, border_delay_map=env["border_delay_map"], size=MASTER_S,
            seed=seed, stochastic=True,
            border_event_definitions=border_event_definitions)

        # NOTE on simulate_path_over_scenarios's module-level _PATH_SCENARIO_CACHE
        # (baseline_uncertainty.py:239, keyed by (seed, size, stochastic,
        # batch_id, ET, topology, trans_signature)): this script installs each
        # subset as model.ACTIVE_SCENARIO_SET by direct assignment (via
        # run_ev_and_ccp), NOT via configure_scenario_set, so the cache is
        # never explicitly cleared between (seed, S) combinations. This is
        # safe here because the cache key already includes BOTH seed and
        # size, and every ScenarioSet this script ever constructs for a given
        # (seed, size) pair is deterministically derived from that same pair
        # (build_scenario_set(seed, size) or a literal array-prefix slice of
        # it) -- so no two constructed sets ever share a (seed, size) cache
        # key while holding different array contents (this sweep only ever
        # constructs the S=200 master via build_scenario_set plus literal
        # array-prefix slices for S=50/S=100 -- no independent
        # build_scenario_set(seed, size) call is ever made for a (seed,
        # size) pair also produced by slicing). An explicit spot-check
        # further below (seed=42, S=50) directly clears the cache and forces
        # a fresh recomputation of that SAME object to confirm this for that
        # one pair.
        subsets = {S: make_nested_subset(master, S) for S in S_VALUES if S != MASTER_S}
        subsets[MASTER_S] = master  # S=200 is the master itself, unsliced

        v50_in_100 = verify_prefix_subset(subsets[50], subsets[100])
        v100_in_200 = verify_prefix_subset(subsets[100], subsets[200])
        v50_in_200 = verify_prefix_subset(subsets[50], subsets[200])
        subset_check_by_seed[seed] = dict(
            S50_subset_of_S100=v50_in_100, S100_subset_of_S200=v100_in_200,
            S50_subset_of_S200=v50_in_200,
            all_pass=(v50_in_100["is_prefix_subset"]
                       and v100_in_200["is_prefix_subset"]
                       and v50_in_200["is_prefix_subset"]))
        if not subset_check_by_seed[seed]["all_pass"]:
            raise RuntimeError(
                f"Nested-subset verification FAILED for seed={seed}: "
                f"{subset_check_by_seed[seed]}. Refusing to report results "
                "built on an unverified scenario-nesting assumption.")

        ev_by_S = {}
        ccp_by_S = {}

        for S in S_VALUES:
            scen = subsets[S]

            ev_ccp = sc10.run_ev_and_ccp(env, individual, batches, arcs, tt_dict, scen)
            model.ACTIVE_SCENARIO_SET = scen

            # --- Cross-validate EV cost/emission/makespan means against an
            # independently reconstructed full-solution per-scenario array,
            # exactly as done in functional_check_50.py / feasible_reference_check_50.py.
            cost_s, emission_s, makespan_s, _ = func50.full_solution_scenario_arrays(
                env, individual, batches, arcs, tt_dict, scen)
            cv_ok = True
            for label, recon, real in (
                ("cost", float(np.mean(cost_s)),
                 ev_ccp["ev"]["vio_breakdown"]["scenario_cost_mean"]),
                ("emission", float(np.mean(emission_s)),
                 ev_ccp["ev"]["vio_breakdown"]["scenario_emission_mean"]),
                ("makespan", float(np.mean(makespan_s)),
                 ev_ccp["ev"]["vio_breakdown"]["scenario_time_mean"]),
            ):
                try:
                    func50.assert_cross_validated(
                        recon, real, f"seed={seed} S={S} {label}")
                except AssertionError as exc:
                    cv_ok = False
                    cross_validation_failures.append(str(exc))

            total_lateness_s = total_lateness_array(env, individual, batches, tt_dict, scen)

            ev_cost = ev_ccp["ev"]["objectives"][0]
            ev_emis = ev_ccp["ev"]["objectives"][1]
            ev_time = ev_ccp["ev"]["objectives"][2]
            ev_total_lateness_h = float(np.mean(total_lateness_s))
            ev_by_S[S] = dict(cost=ev_cost, emissions=ev_emis, makespan=ev_time,
                               total_lateness_h=ev_total_lateness_h)

            # --- CCP: bottleneck batch, satisfied/violated counts, Wilson CI ---
            batch_on_time_prob = ev_ccp["ccp"]["batch_on_time_prob"]
            min_on_time_prob = ev_ccp["ccp"]["vio_breakdown"]["min_on_time_prob"]
            bottleneck_batch_id = min(batch_on_time_prob, key=batch_on_time_prob.get)
            _, arrival_s = batch_arrival_array(
                env, individual, batches, tt_dict, scen, bottleneck_batch_id)
            batch_obj = next(b for b in batches if b.batch_id == bottleneck_batch_id)
            satisfied_bool = arrival_s <= batch_obj.LT
            satisfied_count = int(np.sum(satisfied_bool))
            violated_count = int(scen.size - satisfied_count)
            empirical_prob = satisfied_count / scen.size
            if abs(empirical_prob - batch_on_time_prob[bottleneck_batch_id]) > 1e-9:
                cross_validation_failures.append(
                    f"seed={seed} S={S}: recomputed bottleneck satisfied-fraction "
                    f"{empirical_prob!r} != production batch_on_time_prob"
                    f"[{bottleneck_batch_id}]={batch_on_time_prob[bottleneck_batch_id]!r}")
                cv_ok = False
            if abs(batch_on_time_prob[bottleneck_batch_id] - min_on_time_prob) > 1e-9:
                cross_validation_failures.append(
                    f"seed={seed} S={S}: bottleneck batch_on_time_prob "
                    f"{batch_on_time_prob[bottleneck_batch_id]!r} != vio_breakdown "
                    f"min_on_time_prob={min_on_time_prob!r}")
                cv_ok = False

            wilson_lo, wilson_hi = wilson_ci(satisfied_count, scen.size)
            chance_constraint_pass = bool(min_on_time_prob + 1e-12 >= model.CONFIDENCE_ONTIME)

            ccp_cost, ccp_emis, ccp_time = ev_ccp["ccp"]["objectives"]
            ccp_by_S[S] = dict(
                cost=ccp_cost, emissions=ccp_emis, makespan=ccp_time,
                min_on_time_prob=min_on_time_prob,
                chance_constraint_pass=chance_constraint_pass,
                feasible_hard=ev_ccp["ccp"]["feasible_hard"],
                feasible_soft=ev_ccp["ccp"]["feasible"],
            )

            ccp_prob_records.append(dict(
                seed=seed, S=S, min_on_time_prob=min_on_time_prob,
                bottleneck_batch_id=bottleneck_batch_id,
                satisfied_count=satisfied_count, violated_count=violated_count,
                n_total=scen.size, wilson_lo95=wilson_lo, wilson_hi95=wilson_hi,
                chance_constraint_pass=chance_constraint_pass,
                feasible_hard=ev_ccp["ccp"]["feasible_hard"]))

            per_row_records.append(dict(
                candidate_id=candidate_id, seed=seed, S=S,
                ev_cost=ev_cost, ev_emissions=ev_emis, ev_makespan=ev_time,
                ev_total_lateness_h=ev_total_lateness_h,
                ccp_cost=ccp_cost, ccp_emissions=ccp_emis, ccp_makespan=ccp_time,
                ccp_min_on_time_prob=min_on_time_prob,
                ccp_bottleneck_batch_id=bottleneck_batch_id,
                ccp_satisfied_count=satisfied_count, ccp_violated_count=violated_count,
                ccp_wilson_lo95=wilson_lo, ccp_wilson_hi95=wilson_hi,
                ccp_chance_constraint_pass=chance_constraint_pass,
                ccp_feasible_hard=ev_ccp["ccp"]["feasible_hard"],
                ccp_feasible_soft=ev_ccp["ccp"]["feasible"],
                ccp_chance_vio=ev_ccp["ccp"]["vio_breakdown"]["chance_vio"],
                ccp_max_late_excess_h=ev_ccp["ccp"]["vio_breakdown"]["max_late_excess_h"],
                ccp_penalty=ev_ccp["ccp"]["penalty"],
                ccp_normalized_violation=ev_ccp["ccp"]["normalized_violation"],
                cross_validation_ok=cv_ok,
            ))

            if seed == SEEDS[0] and S == 50:
                # Cache-safety spot-check: FORCE a fully uncached
                # recomputation of this EXACT (seed, S) ScenarioSet object
                # (not a differently-constructed one -- an independent
                # build_scenario_set(seed=42, size=50) call would NOT match
                # this array-prefix subset, since build_scenario_set draws
                # sequentially across ALL arc keys before starting border
                # events, so a size=50 call and a size=200-then-sliced-to-50
                # call diverge after the very first drawn array; comparing
                # against such an independent build would be invalid). This
                # clears _PATH_SCENARIO_CACHE and re-evaluates the SAME
                # `scen` object already used above, directly exercising
                # simulate_path_over_scenarios's (seed, size, ...) cache-key
                # correctness for this pair rather than merely assuming it.
                model._PATH_SCENARIO_CACHE = {}
                spot_ev_ccp = sc10.run_ev_and_ccp(
                    env, individual, batches, arcs, tt_dict, scen)
                model.ACTIVE_SCENARIO_SET = scen
                # Compare objectives AND the CCP diagnostics that most
                # directly depend on the per-scenario arrival arrays
                # (batch_on_time_prob, min_on_time_prob, feasible_hard) --
                # objectives alone could theoretically stay unchanged while
                # an arrival-array-derived probability silently diverged.
                cache_safety_spot_check["ev_objectives_match"] = sc10._tuples_close(
                    tuple(ev_ccp["ev"]["objectives"]),
                    tuple(spot_ev_ccp["ev"]["objectives"]))
                cache_safety_spot_check["ccp_objectives_match"] = sc10._tuples_close(
                    tuple(ev_ccp["ccp"]["objectives"]),
                    tuple(spot_ev_ccp["ccp"]["objectives"]))
                cache_safety_spot_check["batch_on_time_prob_match"] = sc10._dicts_close(
                    ev_ccp["ccp"]["batch_on_time_prob"],
                    spot_ev_ccp["ccp"]["batch_on_time_prob"])
                cache_safety_spot_check["min_on_time_prob_match"] = sc10._close(
                    ev_ccp["ccp"]["vio_breakdown"]["min_on_time_prob"],
                    spot_ev_ccp["ccp"]["vio_breakdown"]["min_on_time_prob"])
                cache_safety_spot_check["feasible_hard_match"] = (
                    ev_ccp["ccp"]["feasible_hard"] == spot_ev_ccp["ccp"]["feasible_hard"])
                if not all(cache_safety_spot_check[k] for k in (
                        "ev_objectives_match", "ccp_objectives_match",
                        "batch_on_time_prob_match", "min_on_time_prob_match",
                        "feasible_hard_match")):
                    cross_validation_failures.append(
                        f"Cache-safety spot-check FAILED for seed={seed} S={S}: "
                        f"forced-fresh (cache-cleared) recomputation "
                        f"ev={spot_ev_ccp['ev']['objectives']} "
                        f"ccp={spot_ev_ccp['ccp']['objectives']} "
                        f"batch_on_time_prob={spot_ev_ccp['ccp']['batch_on_time_prob']} "
                        f"feasible_hard={spot_ev_ccp['ccp']['feasible_hard']} does not "
                        f"match the original cached-path evaluation "
                        f"ev={ev_ccp['ev']['objectives']} "
                        f"ccp={ev_ccp['ccp']['objectives']} "
                        f"batch_on_time_prob={ev_ccp['ccp']['batch_on_time_prob']} "
                        f"feasible_hard={ev_ccp['ccp']['feasible_hard']}")

            diag_rows, water_all_zero, skipped_zero_events = \
                sampling_diagnostics_for_scenario_set(scen)
            for row in diag_rows:
                sampling_records.append(dict(seed=seed, S=S, **row))
            if not water_all_zero:
                cross_validation_failures.append(
                    f"seed={seed} S={S}: maritime rule violated -- nonzero "
                    "water border-delay array found (deterministic maritime "
                    "border-delay mean is expected to be zero for this dataset)")

            if S == 200 and seed == SEEDS[0]:
                representative_manifest = model.build_scenario_manifest(
                    risk_metric="ccp", scenario_set=scen,
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

        # --- Paired EV changes within this seed ---
        ev_pairs = {}
        for obj in ("cost", "emissions", "makespan", "total_lateness_h"):
            ev_pairs[obj] = dict(
                S50_to_S100=_abs_pct(ev_by_S[50][obj], ev_by_S[100][obj]),
                S100_to_S200=_abs_pct(ev_by_S[100][obj], ev_by_S[200][obj]),
                S50_to_S200=_abs_pct(ev_by_S[50][obj], ev_by_S[200][obj]),
            )
        ev_paired_by_seed[seed] = ev_pairs

        # --- CCP classification stability across S for this seed ---
        chance_seq = [ccp_by_S[S]["chance_constraint_pass"] for S in S_VALUES]
        hard_seq = [ccp_by_S[S]["feasible_hard"] for S in S_VALUES]
        ccp_stability_by_seed[seed] = dict(
            chance_constraint_pass_by_S={S: ccp_by_S[S]["chance_constraint_pass"]
                                          for S in S_VALUES},
            feasible_hard_by_S={S: ccp_by_S[S]["feasible_hard"] for S in S_VALUES},
            chance_constraint_classification_stable=(len(set(chance_seq)) == 1),
            feasible_hard_classification_stable=(len(set(hard_seq)) == 1),
        )

    if cross_validation_failures:
        raise RuntimeError(
            "Cross-validation FAILED for one or more (seed, S) combinations. "
            "Refusing to report unverified numbers:\n" +
            "\n".join(cross_validation_failures))

    # --- Aggregate across the 5 seeds, per S ---
    aggregate_by_S = {}
    for S in S_VALUES:
        rows_here = [r for r in per_row_records if r["S"] == S]
        aggregate_by_S[S] = dict(
            n_seeds=len(rows_here),
            ev_cost=_stats([r["ev_cost"] for r in rows_here]),
            ev_emissions=_stats([r["ev_emissions"] for r in rows_here]),
            ev_makespan=_stats([r["ev_makespan"] for r in rows_here]),
            ev_total_lateness_h=_stats([r["ev_total_lateness_h"] for r in rows_here]),
            ccp_cost=_stats([r["ccp_cost"] for r in rows_here]),
            ccp_emissions=_stats([r["ccp_emissions"] for r in rows_here]),
            ccp_makespan=_stats([r["ccp_makespan"] for r in rows_here]),
            ccp_min_on_time_prob=_stats([r["ccp_min_on_time_prob"] for r in rows_here]),
            n_seeds_passing_chance_constraint=sum(
                1 for r in rows_here if r["ccp_chance_constraint_pass"]),
            n_seeds_passing_feasible_hard=sum(
                1 for r in rows_here if r["ccp_feasible_hard"]),
            n_seeds_total=len(rows_here),
        )

    # --- Aggregate paired-diff summary across seeds (extra evidence) ---
    paired_diff_aggregate = {}
    for obj in ("cost", "emissions", "makespan", "total_lateness_h"):
        paired_diff_aggregate[obj] = {}
        for transition in ("S50_to_S100", "S100_to_S200", "S50_to_S200"):
            pct_values = [ev_paired_by_seed[seed][obj][transition]["percent"]
                          for seed in SEEDS
                          if ev_paired_by_seed[seed][obj][transition]["percent"] is not None]
            paired_diff_aggregate[obj][transition] = dict(
                percent_abs_stats=_stats([abs(v) for v in pct_values]))

    n_seeds_chance_stable = sum(
        1 for seed in SEEDS
        if ccp_stability_by_seed[seed]["chance_constraint_classification_stable"])
    n_seeds_hard_stable = sum(
        1 for seed in SEEDS
        if ccp_stability_by_seed[seed]["feasible_hard_classification_stable"])

    summary = {
        "config": {
            "seeds": SEEDS, "scenario_counts": S_VALUES, "master_scenario_count": MASTER_S,
            "confidence_ontime": model.CONFIDENCE_ONTIME,
            "epsilon_ontime": 1.0 - model.CONFIDENCE_ONTIME,
            "mode_time_cv": model.MODE_TIME_CV,
            "border_delay_cv": model.BORDER_DELAY_CV,
            "distribution": "lognormal (mean-one, capped)",
        },
        "candidate": {
            "candidate_id": candidate_id,
            "candidate_count": 1,
            "limitation_note": candidate_limitation_note,
            "loaded_directly_from_checkpoint": True,
            "genuinely_deterministic_feasible": genuinely_deterministic_feasible,
            "identical_to_checkpointed_feasible_reference": identical_to_checkpoint,
            "path_allocations": path_allocations_report,
        },
        "deterministic_baseline": deterministic_baseline,
        "nested_subset_verification_by_seed": subset_check_by_seed,
        "aggregate_by_S": aggregate_by_S,
        "ev_paired_changes_by_seed": ev_paired_by_seed,
        "ev_paired_changes_aggregate_across_seeds": paired_diff_aggregate,
        "ccp_classification_stability_by_seed": ccp_stability_by_seed,
        "ccp_classification_stability_overall": {
            "n_seeds_chance_constraint_classification_stable": n_seeds_chance_stable,
            "n_seeds_feasible_hard_classification_stable": n_seeds_hard_stable,
            "n_seeds_total": len(SEEDS),
        },
        "cross_validation": {
            "status": "ALL PASSED",
            "cache_safety_spot_check": cache_safety_spot_check,
            "note": ("EV cost/emission/makespan means were cross-validated against "
                     "an independently reconstructed full-solution per-scenario "
                     "array for every (seed, S) combination; the CCP bottleneck "
                     "batch's recomputed satisfied-fraction was cross-validated "
                     "against evaluate_individual's own batch_on_time_prob and "
                     "min_on_time_prob. cache_safety_spot_check additionally "
                     "clears _PATH_SCENARIO_CACHE and forces a fresh, uncached "
                     "recomputation of the SAME seed=42/S=50 ScenarioSet object "
                     "already used in the sweep, confirming objectives, "
                     "batch_on_time_prob, min_on_time_prob, and feasible_hard "
                     "all reproduce -- a spot-check of that one (seed, S) pair, "
                     "not a proof covering every combination swept (the "
                     "argument for the remaining combinations is the "
                     "cache-key/construction-pattern argument in the code "
                     "comment above the subset loop). The script would have "
                     "raised RuntimeError and aborted before writing this file "
                     "otherwise."),
        },
    }

    # ============================= Save evidence =============================
    with (OUT_DIR / "per_candidate_seed_S.csv").open("w", newline="", encoding="utf-8") as fh:
        fieldnames = list(per_row_records[0].keys())
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(per_row_records)

    with (OUT_DIR / "ccp_probability_summary.csv").open("w", newline="", encoding="utf-8") as fh:
        fieldnames = list(ccp_prob_records[0].keys())
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(ccp_prob_records)

    with (OUT_DIR / "sampling_diagnostics.csv").open("w", newline="", encoding="utf-8") as fh:
        fieldnames = list(sampling_records[0].keys())
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sampling_records)

    with (OUT_DIR / "aggregate_by_S.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        header = ["S", "n_seeds"]
        metric_names = ["ev_cost", "ev_emissions", "ev_makespan", "ev_total_lateness_h",
                         "ccp_cost", "ccp_emissions", "ccp_makespan", "ccp_min_on_time_prob"]
        for m in metric_names:
            header += [f"{m}_mean", f"{m}_std", f"{m}_min", f"{m}_max", f"{m}_cv"]
        header += ["n_seeds_passing_chance_constraint", "n_seeds_passing_feasible_hard"]
        writer.writerow(header)
        for S in S_VALUES:
            agg = aggregate_by_S[S]
            row = [S, agg["n_seeds"]]
            for m in metric_names:
                s = agg[m]
                row += [s["mean"], s["std"], s["min"], s["max"], s["cv"]]
            row += [agg["n_seeds_passing_chance_constraint"],
                    agg["n_seeds_passing_feasible_hard"]]
            writer.writerow(row)

    (OUT_DIR / "scenario_count_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    (OUT_DIR / "manifest.json").write_text(
        json.dumps(dict(
            analysis="scenario_count_sensitivity",
            seeds=SEEDS, scenario_counts=S_VALUES, master_scenario_count=MASTER_S,
            candidate_id=candidate_id,
            representative_production_manifest_seed=SEEDS[0],
            representative_production_manifest_S=MASTER_S,
            representative_production_manifest=representative_manifest,
        ), ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    print(f"[SCS] Wrote evidence files to {OUT_DIR}/")
    print(f"[SCS] Candidate limitation: {candidate_limitation_note}")
    for S in S_VALUES:
        agg = aggregate_by_S[S]
        print(f"[SCS] S={S}: EV cost mean={agg['ev_cost']['mean']:.2f} "
              f"cv={agg['ev_cost']['cv']}  "
              f"CCP min_on_time_prob mean={agg['ccp_min_on_time_prob']['mean']:.4f}  "
              f"seeds_passing_chance_constraint="
              f"{agg['n_seeds_passing_chance_constraint']}/{agg['n_seeds_total']}  "
              f"seeds_passing_feasible_hard="
              f"{agg['n_seeds_passing_feasible_hard']}/{agg['n_seeds_total']}")
    print(f"[SCS] CCP classification stability across S=50->100->200: "
          f"chance-constraint stable in {n_seeds_chance_stable}/{len(SEEDS)} seeds, "
          f"feasible_hard stable in {n_seeds_hard_stable}/{len(SEEDS)} seeds")


if __name__ == "__main__":
    main()
