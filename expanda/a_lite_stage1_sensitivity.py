#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""A-lite STAGE 1 -- fixed-candidate CCP scenario-count sensitivity.

SCIENTIFIC QUESTION (and ONLY this question)
--------------------------------------------
Does the finite scenario count S materially affect (a) the empirical CCP
probability estimate and (b) the CCP feasibility CLASSIFICATION, for FIXED
candidate solutions near the alpha=0.90 boundary?

This is explicitly NOT:
  - an optimisation experiment (no search of any kind is performed),
  - an SAA convergence proof (no claim that any S is "sufficient"),
  - evidence about NSGA-II search-time selection bias (that is Stage 2).

DESIGN
------
10 frozen candidates (S1-01..S1-10) x 10 fresh master seeds x S in {50,100,200}.
For each master seed exactly ONE master ScenarioSet of size 200 is built; the
S=50 and S=100 sets are literal array PREFIXES of that master, never
independent redraws. A previous audit established that
build_scenario_set(seed, 50) is NOT the prefix of build_scenario_set(seed, 200)
-- the shared RNG stream advances by `size` per arc key -- so independent
regeneration is forbidden here and is never performed.

Because S50/S100/S200 are nested they are POSITIVELY CORRELATED by
construction (deliberate common-random-numbers coupling, which is what
isolates the effect of ADDING scenarios). They must NOT be treated as
independent samples in tests or confidence intervals. Independent
replication comes ONLY from the 10 distinct master seeds.

This script does NOT modify baseline_uncertainty.py or any other production
module; it imports and calls them unchanged, and reuses already-audited
helpers (sanity_check_10.load_everything, scenario_count_sensitivity's
make_nested_subset / verify_prefix_subset / wilson_ci) rather than copying
production logic.

EXECUTION MODES
---------------
  --self-test   integrity only: provenance, candidate load, border-event
                configuration, and nested-prefix verification for all seeds.
                Performs ZERO candidate evaluations. Safe to run any time.
  --run         the full Stage-1 study.
  --sanity-run  a deliberately reduced run; refuses to start unless
                --i-have-approval is also given, because a reduced
                scientific sample requires explicit sign-off.
"""
import argparse
import csv
import hashlib
import json
import statistics
import sys
from pathlib import Path as FSPath

import numpy as np

HERE = FSPath(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import baseline_uncertainty as model  # noqa: E402
import sanity_check_10 as sc10  # noqa: E402  (audited loader; builds no scenarios)
import scenario_count_sensitivity as scs  # noqa: E402  (audited nesting helpers)

# ════════════════════════════════════════════════════════
# DECLARED CONFIGURATION -- fixed BEFORE any result is inspected
# ════════════════════════════════════════════════════════

# Ten FRESH master seeds. Deliberately disjoint from every seed previously
# used anywhere in this project: 42 (candidate-location scan and smoke
# --mc-seed), 42-46 (scenario_count_sensitivity), 1000003 (MC_BASE_SEED and
# the archive runs), 1000 (smoke GA seed) and 0 (path-library seeding).
# A contiguous block is used so the choice is transparently not cherry-picked.
MASTER_SEEDS = [700001, 700002, 700003, 700004, 700005,
                700006, 700007, 700008, 700009, 700010]

S_VALUES = [50, 100, 200]
MASTER_S = 200
RISK_METRIC = "ccp"

# The CCP threshold itself. production reads the mutable global
# model.CONFIDENCE_ONTIME; forcing RISK_METRIC alone would NOT protect the
# classification semantics if that global ever changed. Asserted before any
# evaluation, recorded in the manifest, and used to independently recompute
# chance_vio for cross-checking. Stage 1 never changes it.
EXPECTED_CONFIDENCE_ONTIME = 0.90

# ── PREDECLARED MATERIALITY CRITERIA ─────────────────────────────────────
# Declared BEFORE any Stage-1 result is inspected so that "materially
# affects" cannot be redefined after seeing the data. These are
# interpretation thresholds for the write-up, NOT tuning knobs: nothing in
# the code branches on them and no candidate or seed is filtered by them.
MATERIALITY = {
    "mean_abs_prob_delta_vs_S200": 0.02,
    "classification_flip_rate_vs_S200": 0.10,
    "rationale": (
        "A mean |p_hat_S - p_hat_200| at or above 0.02 is one full S=50 "
        "quantisation step: a practically meaningful shift in the "
        "probability ESTIMATE at the scale the estimator can even resolve. "
        "It does not by itself imply any candidate crossed the 0.90 "
        "boundary -- the separately reported classification flip rate is "
        "what establishes that. A classification flip rate at or above 0.10 "
        "(i.e. at least one in ten) means at least one in ten fixed "
        "candidate x seed cases would be labelled differently by the choice "
        "of S alone. Either condition is reported as a MATERIAL effect of "
        "scenario count; both below is reported as immaterial at the scales "
        "tested. Thresholds are inclusive (>=). This is a reporting rule "
        "only: nothing branches on it and no candidate or seed is filtered "
        "by it."),
}

# Integrity assertions against the CURRENT validated production configuration.
# These are NOT model logic and NOT parameters -- they are tripwires. A
# previous scan silently produced 4 events / 10 arcs because
# BORDER_EVENT_DEFINITIONS was left empty, which would have invalidated every
# probability. If the production configuration stops reproducing these, the
# run must stop rather than quietly produce different physics.
EXPECTED_BORDER_EVENTS = 258
EXPECTED_BORDER_ARCS = 268

# Frozen candidate provenance (see commit feb4cf2).
CHECKPOINT = HERE / "a_lite_stage1_candidates.json"
EXPECTED_CANDIDATES_SHA256 = (
    "ce214df2f78007f19f39c3a342924d2c6eee54501f4bd88485ffb72118279f33")
EXPECTED_CANDIDATE_IDS = [f"S1-{i:02d}" for i in range(1, 11)]

OUT_DIR = HERE / "a_lite_stage1_results"
# A reduced sanity run must never land in, or overwrite, the real results.
SANITY_OUT_DIR = HERE / "a_lite_stage1_sanity"

# Numerical tolerances.
TOL_HARD = 1e-12          # CCP hard-constraint tolerance (matches production)
TOL_CAP = 1e-9            # capacity tolerance (matches production)
TOL_MONOTONE = 1e-9       # tolerance for the nested max-lateness monotonicity

# ── HELD-OUT VALIDATION: DESIGNED, DELIBERATELY NOT RUN ──────────────────
# A held-out set must be independent of (a) candidate selection and (b) all
# Stage-1 master seeds, and must never be used to choose candidates or to
# tune Stage-1 decision criteria after results are inspected. Its size and
# seed are intentionally left unset until the implementation and its
# computational cost have been reviewed. Any attempt to use it fails closed.
HELD_OUT_SEED = None
HELD_OUT_S = None
HELD_OUT_ENABLED = False

# Every seed that must never be reused as a Stage-1 or held-out seed.
FORBIDDEN_SEEDS = {0, 42, 43, 44, 45, 46, 1000, 1000003}


class Stage1IntegrityError(RuntimeError):
    """Any fail-closed condition. Never caught, never downgraded."""


def fail(msg):
    raise Stage1IntegrityError(msg)


# ════════════════════════════════════════════════════════
# Provenance / candidate loading
# ════════════════════════════════════════════════════════

def canonical_decision_from_checkpoint(entry):
    """The checkpoint already stores the canonical decision structure."""
    return entry["decision_allocations"]


def candidates_canonical_sha256(cands):
    blob = json.dumps(cands, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def load_frozen_checkpoint():
    """Load and hard-verify the frozen 10-candidate checkpoint."""
    if not CHECKPOINT.exists():
        fail(f"frozen candidate checkpoint not found: {CHECKPOINT}")
    doc = json.loads(CHECKPOINT.read_text(encoding="utf-8"))
    cands = doc.get("candidates")
    if not isinstance(cands, list):
        fail("checkpoint has no 'candidates' array")
    got = candidates_canonical_sha256(cands)
    if got != EXPECTED_CANDIDATES_SHA256:
        fail(f"candidate-array SHA-256 mismatch\n  expected "
             f"{EXPECTED_CANDIDATES_SHA256}\n  got      {got}")
    if len(cands) != 10:
        fail(f"expected 10 frozen candidates, got {len(cands)}")
    ids = [c["stage1_candidate_id"] for c in cands]
    if ids != EXPECTED_CANDIDATE_IDS:
        fail(f"candidate IDs/order changed: {ids}")
    if len(set(ids)) != len(ids):
        fail(f"duplicate candidate IDs: {ids}")
    return doc, cands


def individual_decision_fingerprint(ind):
    """Order-insensitive canonical fingerprint of an Individual's decisions."""
    parts = []
    for (origin, dest, bid), allocs in sorted(ind.od_allocations.items(),
                                              key=lambda kv: kv[0][2]):
        paths = sorted(
            (tuple(a.path.nodes), tuple(a.path.modes), float(a.share),
             # include arc identity and the derived base metrics so that
             # mutation of a shared Path/Arc object is also detected, not
             # just a change of nodes/modes/shares
             tuple((x.from_node, x.to_node, x.mode, float(x.distance))
                   for x in a.path.arcs),
             float(a.path.base_cost_per_teu),
             float(a.path.base_emission_per_teu),
             float(a.path.base_travel_time_h))
            for a in allocs)
        parts.append((int(bid), origin, dest, tuple(paths)))
    return hashlib.sha256(repr(parts).encode("utf-8")).hexdigest()


def build_candidate_individual(entry, batches, path_lib, tt_dict, arc_lookup,
                               stats):
    """Hybrid loader: reuse an exact path_lib match where one exists,
    otherwise reconstruct with the production function and NO road
    fallback. Exact nodes/modes equality is required; there is no silent
    repair path."""
    lib = {}
    for od, paths in path_lib.items():
        for p in paths:
            lib[(od, tuple(p.nodes), tuple(p.modes))] = p

    by_bid = {int(a[0]): a for a in entry["decision_allocations"]}
    ind = model.Individual()
    for b in batches:
        rec = by_bid.get(int(b.batch_id))
        if rec is None:
            fail(f"{entry['stage1_candidate_id']}: missing batch {b.batch_id}")
        _bid, origin, dest, paths = rec
        if origin != b.origin or dest != b.destination:
            fail(f"{entry['stage1_candidate_id']}: batch {b.batch_id} OD "
                 f"mismatch ({origin}->{dest} vs {b.origin}->{b.destination})")
        allocs = []
        for nodes, modes, share in paths:
            nodes, modes = list(nodes), list(modes)
            hit = lib.get(((b.origin, b.destination), tuple(nodes), tuple(modes)))
            if hit is not None:
                stats["lib_hit"] += 1
                path = hit
            else:
                path = model.rebuild_path_from_nodes_modes(
                    b.origin, b.destination, nodes, modes, tt_dict, arc_lookup,
                    allow_road_fallback=False)
                if path is None:
                    fail(f"{entry['stage1_candidate_id']}: batch {b.batch_id} "
                         f"path could not be reconstructed exactly "
                         f"(nodes={nodes}, modes={modes})")
                stats["rebuilt"] += 1
            if path.nodes != nodes or path.modes != modes:
                fail(f"{entry['stage1_candidate_id']}: batch {b.batch_id} "
                     "reconstructed path identity differs from the frozen one")
            allocs.append(model.PathAllocation(path=path, share=float(share)))
        ind.od_allocations[(b.origin, b.destination, b.batch_id)] = allocs
    return ind


# ════════════════════════════════════════════════════════
# Environment / border-event configuration
# ════════════════════════════════════════════════════════

def prepare_environment():
    """Load network + path library, and explicitly install the production
    border-event definitions.

    baseline_uncertainty.BORDER_EVENT_DEFINITIONS is {} at module level and
    is populated by production main() from --border-event-data. A previous
    candidate-location scan silently produced a ScenarioSet with 4 events /
    10 arcs because this step was omitted, which would have invalidated
    every probability it reported. This function therefore performs the
    load explicitly and the caller asserts the resulting mapping.
    """
    env = sc10.load_everything()
    if model.ACTIVE_SCENARIO_SET is not None:
        fail("a ScenarioSet was already installed before Stage-1 setup")
    # The CCP threshold is a mutable global and IS the classification
    # semantics under test; verify it rather than assuming the default.
    if float(model.CONFIDENCE_ONTIME) != EXPECTED_CONFIDENCE_ONTIME:
        fail(f"CONFIDENCE_ONTIME is {model.CONFIDENCE_ONTIME!r}, expected "
             f"{EXPECTED_CONFIDENCE_ONTIME!r}. Stage 1 is defined at "
             "alpha=0.90 and must not run against a different threshold.")
    defs = model.load_border_event_definitions(
        model.DEFAULT_BORDER_EVENT_DATA_FILE)
    if not defs:
        fail("border-event definitions loaded empty from "
             f"{model.DEFAULT_BORDER_EVENT_DATA_FILE}")
    model.BORDER_EVENT_DEFINITIONS = defs
    env["border_event_definitions"] = defs
    return env


def build_master(env, seed):
    """Build exactly ONE master ScenarioSet of size MASTER_S for `seed`, and
    assert it reproduces the validated border-event mapping."""
    if seed in FORBIDDEN_SEEDS:
        fail(f"seed {seed} is reserved/previously used and must not be reused")
    master = model.build_scenario_set(
        arcs=env["arcs"], border_delay_map=env["border_delay_map"],
        size=MASTER_S, seed=seed, stochastic=True,
        border_event_definitions=env["border_event_definitions"])
    n_events = len(master.border_event_mean_h)
    n_arcs = len(master.arc_border_event)
    if n_events != EXPECTED_BORDER_EVENTS or n_arcs != EXPECTED_BORDER_ARCS:
        fail(f"border-event configuration mismatch for seed {seed}: got "
             f"{n_events} events / {n_arcs} arcs, expected "
             f"{EXPECTED_BORDER_EVENTS} / {EXPECTED_BORDER_ARCS}. Refusing to "
             "run against a different stochastic configuration than the "
             "validated one. MIGRATION: if the network or border-event data "
             "has legitimately changed, this is a NEW experiment identity -- "
             "update EXPECTED_BORDER_EVENTS/EXPECTED_BORDER_ARCS in a "
             "reviewed change, record the new provenance, and re-derive the "
             "frozen candidate panel. Never relax this check to make an "
             "existing run proceed.")
    if master.size != MASTER_S or master.seed != seed or not master.stochastic:
        fail(f"unexpected master ScenarioSet metadata for seed {seed}")
    return master


def nested_subsets(master, seed):
    """S50/S100 as literal prefixes of `master`; S200 IS the master.
    Every pairwise nesting relation is independently re-verified."""
    subsets = {S: scs.make_nested_subset(master, S)
               for S in S_VALUES if S != MASTER_S}
    subsets[MASTER_S] = master

    report = {}
    for small, large in ((50, 100), (100, 200), (50, 200)):
        v = scs.verify_prefix_subset(subsets[small], subsets[large])
        report[f"S{small}_subset_of_S{large}"] = v
        if not v["is_prefix_subset"]:
            fail(f"nested-prefix verification FAILED for seed {seed} "
                 f"(S{small} in S{large}): {v['mismatched_keys']}")
    for S, scen in subsets.items():
        if scen.size != S or scen.seed != seed or not scen.stochastic:
            fail(f"subset metadata wrong for seed {seed} S={S}")
        if len(scen.border_event_mean_h) != EXPECTED_BORDER_EVENTS:
            fail(f"subset border_event_mean_h changed for seed {seed} S={S}")
        if len(scen.arc_border_event) != EXPECTED_BORDER_ARCS:
            fail(f"subset arc_border_event changed for seed {seed} S={S}")
        for key, arr in scen.travel_multiplier.items():
            if arr.shape[0] != S:
                fail(f"travel_multiplier[{key}] length {arr.shape[0]} != {S}")
        for key, arr in scen.border_delay_h.items():
            if arr.shape[0] != S:
                fail(f"border_delay_h[{key}] length {arr.shape[0]} != {S}")
    report["all_pass"] = True
    return subsets, report


def install_scenario_set(scen):
    """MANDATORY cache-safe installation.

    _PATH_SCENARIO_CACHE is keyed by (seed, size, stochastic, batch_id, ET,
    topology, trans_signature) and does NOT include the scenario array
    contents, so two different ScenarioSets sharing (seed, size, stochastic)
    would silently collide. Every installation therefore clears the cache
    first, then installs, then asserts identity.
    """
    model._PATH_SCENARIO_CACHE = {}
    model.ACTIVE_SCENARIO_SET = scen
    if model.ACTIVE_SCENARIO_SET is not scen:
        fail("ACTIVE_SCENARIO_SET is not the intended object after install")
    if model._PATH_SCENARIO_CACHE:
        fail("path-scenario cache non-empty immediately after install")


def assert_cache_confined(scen):
    """After evaluating against `scen`, every cache entry must belong to it."""
    keys = {k[:3] for k in model._PATH_SCENARIO_CACHE}
    allowed = {(scen.seed, scen.size, scen.stochastic)}
    if not keys <= allowed:
        fail(f"path-scenario cache holds foreign scenario keys: "
             f"{keys - allowed}")


# ════════════════════════════════════════════════════════
# Evaluation of one candidate against one installed ScenarioSet
# ════════════════════════════════════════════════════════

def evaluate_candidate(env, ind, batches, arcs, tt_dict, scen, cand_id, seed, S):
    """Evaluate ONE candidate under CCP semantics against the ALREADY
    INSTALLED `scen`. Every fail-closed condition raises."""
    if model.ACTIVE_SCENARIO_SET is not scen:
        fail(f"{cand_id}: active ScenarioSet is not the intended one")

    prev = model.RISK_METRIC
    model.RISK_METRIC = RISK_METRIC
    try:
        clone = model.Individual(od_allocations={
            k: list(v) for k, v in ind.od_allocations.items()})
        model.evaluate_individual(
            clone, batches, arcs, tt_dict,
            waiting_cost_per_teu_h=env["waiting_cost_per_teu_h"],
            wait_emis_g_per_teu_h=env["wait_emis_g_per_teu_h"],
            node_hold_cost=env["node_hold_cost"],
            node_proc_cost=env["node_proc_cost"],
            carbon_tax_map=env["carbon_tax_map"], trans_map=env["trans_map"],
            border_delay_map=env["border_delay_map"], theta_rm=env["theta_rm"],
            node_trans_cost=env["node_trans_cost"])
    finally:
        model.RISK_METRIC = prev

    if model.ACTIVE_SCENARIO_SET is not scen:
        fail(f"{cand_id}: ScenarioSet changed during evaluation")

    v = clone.vio_breakdown
    if not v:
        fail(f"{cand_id}: evaluator returned no vio_breakdown")
    required = ("min_on_time_prob", "chance_vio", "max_late_excess_h",
                "miss_alloc", "miss_tt", "cap_excess", "border_cap_excess")
    missing = [k for k in required if k not in v]
    if missing:
        fail(f"{cand_id}: vio_breakdown missing required field(s) {missing}")
    probs = clone.batch_on_time_prob
    if not probs:
        fail(f"{cand_id}: missing per-batch probability vector")
    if len(probs) != len(batches):
        fail(f"{cand_id}: probability vector has {len(probs)} entries, "
             f"expected {len(batches)}")
    expected_bids = {int(b.batch_id) for b in batches}
    got_bids = {int(k) for k in probs}
    if got_bids != expected_bids:
        fail(f"{cand_id}: probability vector batch IDs do not match the batch "
             f"set (missing={sorted(expected_bids - got_bids)}, "
             f"unexpected={sorted(got_bids - expected_bids)})")
    for bid, p in probs.items():
        pf = float(p)
        if not np.isfinite(pf):
            fail(f"{cand_id}: non-finite probability for batch {bid}: {p}")
        if pf < 0.0 or pf > 1.0:
            fail(f"{cand_id}: probability outside [0,1] for batch {bid}: {pf}")

    pmin = min(float(p) for p in probs.values())
    # Cross-check our recomputed minimum against the evaluator's own value.
    # Finiteness MUST be checked first: abs(pmin - nan) > tol is False, so a
    # NaN reported_min would otherwise slip through the comparison.
    reported_min = float(v["min_on_time_prob"])
    if not np.isfinite(reported_min):
        fail(f"{cand_id}: non-finite reported min_on_time_prob = {reported_min}")
    if abs(pmin - reported_min) > 1e-12:
        fail(f"{cand_id}: recomputed min_on_time_prob {pmin!r} disagrees with "
             f"the evaluator's {reported_min!r}")
    binding = sorted(int(b) for b, p in probs.items()
                     if abs(float(p) - pmin) <= 1e-15)
    mle = float(v["max_late_excess_h"])
    chance_vio = float(v["chance_vio"])
    for name, val in (("max_late_excess_h", mle), ("chance_vio", chance_vio),
                      ("min_on_time_prob", pmin)):
        if not np.isfinite(val):
            fail(f"{cand_id}: non-finite {name} = {val}")

    # Independently recompute the chance violation from the probability
    # vector at the asserted alpha. This protects the CLASSIFICATION
    # semantics, not merely the reported minimum.
    recomputed_vio = sum(max(0.0, EXPECTED_CONFIDENCE_ONTIME - float(p))
                         for p in probs.values())
    if abs(recomputed_vio - chance_vio) > 1e-9:
        fail(f"{cand_id}: recomputed chance_vio {recomputed_vio!r} disagrees "
             f"with the evaluator's {chance_vio!r} at alpha="
             f"{EXPECTED_CONFIDENCE_ONTIME}; the effective CCP threshold may "
             "differ from the asserted one")

    # Non-CCP hard-constraint diagnostics must themselves be well-formed.
    hard = {}
    for name in ("miss_alloc", "miss_tt", "cap_excess", "border_cap_excess"):
        val = float(v[name])
        if not np.isfinite(val):
            fail(f"{cand_id}: non-finite {name} = {val}")
        if val < 0.0:
            fail(f"{cand_id}: negative {name} = {val}")
        hard[name] = val
    for name, val in (("obj_cost", clone.objectives[0]),
                      ("obj_emission", clone.objectives[1]),
                      ("obj_time", clone.objectives[2]),
                      ("penalty", clone.penalty)):
        if not np.isfinite(float(val)):
            fail(f"{cand_id}: non-finite {name} = {val}")

    prob_pass = bool(chance_vio <= TOL_HARD)
    mlate_pass = bool(mle <= TOL_HARD)
    nonccp_pass = bool(hard["miss_alloc"] == 0 and hard["miss_tt"] == 0
                       and hard["cap_excess"] <= TOL_CAP
                       and hard["border_cap_excess"] <= TOL_CAP)
    # feasible_hard must equal the conjunction of its components under CCP;
    # if it does not, our decomposition and production's disagree.
    if bool(clone.feasible_hard) != (prob_pass and mlate_pass and nonccp_pass):
        fail(f"{cand_id}: feasible_hard={clone.feasible_hard} disagrees with "
             f"components (prob={prob_pass}, maxlate={mlate_pass}, "
             f"nonccp={nonccp_pass})")

    return dict(
        stage1_candidate_id=cand_id, master_seed=seed, S=S,
        min_on_time_prob=pmin,
        batch_on_time_prob={str(k): float(x) for k, x in sorted(probs.items())},
        binding_batches=binding,
        binding_batch=binding[0],
        binding_batch_set=",".join(map(str, binding)),
        chance_vio_total=chance_vio,
        chance_constraint_pass=prob_pass,
        max_late_excess_h=mle,
        maxlate_screen_pass=mlate_pass,
        miss_alloc=hard["miss_alloc"], miss_tt=hard["miss_tt"],
        cap_excess=hard["cap_excess"],
        border_cap_excess=hard["border_cap_excess"],
        nonccp_hard_pass=nonccp_pass,
        feasible_hard=bool(clone.feasible_hard),
        # descriptive only -- NEVER used to select, exclude or reorder
        obj_cost=float(clone.objectives[0]),
        obj_emission_gCO2=float(clone.objectives[1]),
        obj_time_h=float(clone.objectives[2]),
        penalty=float(clone.penalty),
    )


def check_maxlate_monotonicity(rows_for_candidate_seed, cand_id, seed):
    """Under EXACT nested prefixes, max_late_excess_h is a sum over batches
    of a max over scenarios, so it is deterministically NON-DECREASING in S.
    A violation means the nesting or the installation is broken."""
    by_S = {r["S"]: r["max_late_excess_h"] for r in rows_for_candidate_seed}
    for a, b in ((50, 100), (100, 200)):
        if a in by_S and b in by_S and by_S[b] < by_S[a] - TOL_MONOTONE:
            fail(f"max_late_excess_h monotonicity VIOLATED for {cand_id} "
                 f"seed {seed}: S{a}={by_S[a]!r} > S{b}={by_S[b]!r}. Under "
                 "exact nested prefixes this is impossible; treating as an "
                 "integrity failure rather than a result.")


# ════════════════════════════════════════════════════════
# Summaries
# ════════════════════════════════════════════════════════

SANITY_WARNING = (
    "SANITY RUN -- PLUMBING/DATA-FLOW CHECK ONLY. n=1 master seed. Spread "
    "statistics are degenerate (SD=0), pass rates are 0/1, and the "
    "material_by_* booleans are NOT scientific findings. Do not interpret, "
    "quote or plot any value in this file.")
FULL_RUN_NOTE = (
    "Full Stage-1 study. Nested S values are positively correlated by "
    "construction; independent replication comes only from distinct master "
    "seeds.")


def summarise(rows, cand_meta, mode):
    """Per-candidate-x-S and per-S summaries.

    Nested S values are positively correlated by construction; nothing here
    treats them as independent samples. Across-seed spread is the only
    replication used, and no confidence interval is computed across S.
    """
    per_cs, per_s = [], []
    by_cs = {}
    for r in rows:
        by_cs.setdefault((r["stage1_candidate_id"], r["S"]), []).append(r)

    scope = SANITY_WARNING if mode == "sanity-run" else FULL_RUN_NOTE

    for (cid, S), rs in sorted(by_cs.items()):
        ps = [r["min_on_time_prob"] for r in rs]
        passes = [r["chance_constraint_pass"] for r in rs]
        meta = cand_meta[cid]
        per_cs.append(dict(
            interpretation_scope=scope,
            stage1_candidate_id=cid, S=S, n_seeds=len(rs),
            scan_min_on_time_prob=meta["scan_min_on_time_prob"],
            scan_binding_batch=meta["binding_batch"],
            observed_binding_batches=sorted({r["binding_batch"] for r in rs}),
            observed_binding_batch_sets=sorted({r["binding_batch_set"] for r in rs}),
            mean_min_on_time_prob=statistics.fmean(ps),
            median_min_on_time_prob=statistics.median(ps),
            sd_min_on_time_prob=(statistics.stdev(ps) if len(ps) > 1 else 0.0),
            min_min_on_time_prob=min(ps), max_min_on_time_prob=max(ps),
            ccp_pass_rate_across_seeds=sum(passes) / len(passes),
            maxlate_pass_rate=sum(r["maxlate_screen_pass"] for r in rs) / len(rs),
            nonccp_hard_pass_rate=sum(r["nonccp_hard_pass"] for r in rs) / len(rs),
            feasible_hard_rate=sum(r["feasible_hard"] for r in rs) / len(rs),
        ))

    ref = {(r["stage1_candidate_id"], r["master_seed"]): r
           for r in rows if r["S"] == MASTER_S}
    for S in S_VALUES:
        sub = [r for r in rows if r["S"] == S]
        agree = flips_pf = flips_fp = 0
        absdiff, signdiff = [], []
        bind_changes, mlate_changes, hard_changes = 0, 0, 0
        for r in sub:
            base = ref.get((r["stage1_candidate_id"], r["master_seed"]))
            if base is None:
                fail(f"missing S={MASTER_S} reference for "
                     f"{r['stage1_candidate_id']} seed {r['master_seed']}")
            if r["chance_constraint_pass"] == base["chance_constraint_pass"]:
                agree += 1
            elif r["chance_constraint_pass"]:
                flips_pf += 1          # S pass -> S200 fail
            else:
                flips_fp += 1          # S fail -> S200 pass
            d = r["min_on_time_prob"] - base["min_on_time_prob"]
            signdiff.append(d)          # signed: exposes systematic drift
            absdiff.append(abs(d))
            bind_changes += (r["binding_batch_set"] != base["binding_batch_set"])
            mlate_changes += (r["maxlate_screen_pass"] != base["maxlate_screen_pass"])
            hard_changes += (r["nonccp_hard_pass"] != base["nonccp_hard_pass"])
        mean_abs = statistics.fmean(absdiff) if absdiff else None
        flip_rate = ((flips_pf + flips_fp) / len(sub)) if sub else None
        per_s.append(dict(
            interpretation_scope=scope,
            S=S, n_rows=len(sub),
            classification_agreement_with_S200=agree / len(sub) if sub else None,
            n_pass_at_S_but_fail_at_S200=flips_pf,
            n_fail_at_S_but_pass_at_S200=flips_fp,
            rate_pass_at_S_but_fail_at_S200=flips_pf / len(sub) if sub else None,
            rate_fail_at_S_but_pass_at_S200=flips_fp / len(sub) if sub else None,
            mean_abs_diff_vs_S200=mean_abs,
            max_abs_diff_vs_S200=max(absdiff) if absdiff else None,
            mean_signed_diff_vs_S200=statistics.fmean(signdiff) if signdiff else None,
            median_signed_diff_vs_S200=statistics.median(signdiff) if signdiff else None,
            min_signed_diff_vs_S200=min(signdiff) if signdiff else None,
            max_signed_diff_vs_S200=max(signdiff) if signdiff else None,
            classification_flip_rate_vs_S200=flip_rate,
            materiality_predeclared=MATERIALITY,
            material_by_prob_delta=(
                None if mean_abs is None
                else bool(mean_abs >= MATERIALITY["mean_abs_prob_delta_vs_S200"])),
            material_by_flip_rate=(
                None if flip_rate is None
                else bool(flip_rate >= MATERIALITY["classification_flip_rate_vs_S200"])),
            n_binding_batch_set_changes_vs_S200=bind_changes,
            n_maxlate_screen_changes_vs_S200=mlate_changes,
            n_nonccp_hard_changes_vs_S200=hard_changes,
            comparison_note=(
                "Comparisons are WITHIN the same nested master seed, so S and "
                "S=200 share the first S scenarios and are positively "
                "correlated by construction. These are paired agreement "
                "counts, NOT independent-sample statistics."),
        ))
    return per_cs, per_s


# ════════════════════════════════════════════════════════
# Driver
# ════════════════════════════════════════════════════════

def write_outputs(rows, per_cs, per_s, nesting, manifest, out_dir):
    OUT_DIR = out_dir
    if OUT_DIR.exists() and any(OUT_DIR.iterdir()):
        fail(f"output directory {OUT_DIR} already exists and is non-empty. "
             "Refusing to overwrite previous results; move or remove them "
             "deliberately first.")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "raw_rows.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    if rows:
        flat_cols = [k for k in rows[0] if k not in ("batch_on_time_prob",
                                                     "binding_batches")]
        with (OUT_DIR / "raw_rows.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=flat_cols + ["binding_batches"])
            w.writeheader()
            for r in rows:
                row = {k: r[k] for k in flat_cols}
                row["binding_batches"] = "|".join(map(str, r["binding_batches"]))
                w.writerow(row)
    (OUT_DIR / "per_candidate_summary.json").write_text(json.dumps(per_cs, indent=2), encoding="utf-8")
    (OUT_DIR / "per_S_summary.json").write_text(json.dumps(per_s, indent=2), encoding="utf-8")
    (OUT_DIR / "nested_prefix_verification.json").write_text(json.dumps(nesting, indent=2), encoding="utf-8")
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[OUT] {OUT_DIR}")


def base_manifest(doc, mode, seeds):
    return {
        "stage": "A-lite Stage 1",
        "mode": mode,
        "sanity_run": (mode == "sanity-run"),
        "is_full_study": (mode == "full-run"),
        "scientific_question": (
            "Does finite scenario count S materially affect the empirical CCP "
            "probability estimate and the CCP feasibility classification for "
            "FIXED candidates near the alpha=0.90 boundary?"),
        "explicitly_not": [
            "an optimisation experiment (no search is performed)",
            "an SAA convergence proof (no claim any S is sufficient)",
            "evidence about NSGA-II search-time selection bias (Stage 2)",
        ],
        "master_seeds": seeds,
        "seed_provenance": (
            "Ten fresh seeds declared before any Stage-1 result was inspected, "
            "disjoint from every previously used seed "
            f"({sorted(FORBIDDEN_SEEDS)})."),
        "S_values": S_VALUES,
        "master_S": MASTER_S,
        "risk_metric": RISK_METRIC,
        "confidence_ontime_alpha": float(model.CONFIDENCE_ONTIME),
        "confidence_ontime_asserted": EXPECTED_CONFIDENCE_ONTIME,
        "predeclared_materiality": MATERIALITY,
        "nesting": (
            "For each seed exactly ONE master ScenarioSet of size 200 is built; "
            "S=50 and S=100 are literal array prefixes of it. Independent "
            "build_scenario_set(seed, S) calls are never made, because they do "
            "NOT form prefixes."),
        "correlation_caveat": (
            "S50/S100/S200 are nested and therefore positively correlated by "
            "construction. They are NOT independent samples; no test or "
            "confidence interval treats them as such. Independent replication "
            "comes only from the 10 distinct master seeds."),
        "candidate_provenance": {
            "checkpoint": str(CHECKPOINT),
            "candidates_canonical_sha256": EXPECTED_CANDIDATES_SHA256,
            "candidate_ids": EXPECTED_CANDIDATE_IDS,
            "upstream_pool_scope": doc.get("upstream_pool_scope", {}).get("narrowing"),
            "scan_caveat": doc.get("scan_probability_caveat"),
        },
        "binding_batch_limitation": doc.get("known_binding_batch_limitation"),
        "border_event_integrity": {
            "expected_events": EXPECTED_BORDER_EVENTS,
            "expected_arcs": EXPECTED_BORDER_ARCS,
            "note": ("Asserted per seed as a tripwire against the validated "
                     "configuration; not model logic and not a parameter."),
        },
        "held_out_validation": {
            "designed": True, "executed": False,
            "seed": HELD_OUT_SEED, "S": HELD_OUT_S,
            "enabled": HELD_OUT_ENABLED,
            "rule": ("Must be independent of candidate selection and of all "
                     "Stage-1 master seeds, and must never be used to choose "
                     "candidates or to tune decision criteria after results "
                     "are inspected. Size/seed deliberately unset pending "
                     "review; any use fails closed."),
        },
    }


def run_study(env, doc, cands, seeds, s_values, mode, out_dir):
    batches, arcs, tt_dict = env["batches"], env["arcs"], env["tt_dict"]
    path_lib, arc_lookup = env["path_lib"], env["arc_lookup"]

    stats = {"lib_hit": 0, "rebuilt": 0}
    inds, cand_meta = {}, {}
    for e in cands:
        cid = e["stage1_candidate_id"]
        inds[cid] = build_candidate_individual(
            e, batches, path_lib, tt_dict, arc_lookup, stats)
        cand_meta[cid] = e
    # Defensive: evaluate_individual does not mutate od_allocations (verified
    # against the source), but the same Individual objects are reused across
    # every seed x S, so any future change to that would silently corrupt the
    # whole study. Fingerprint them and re-check at the end.
    decisions_before = {cid: individual_decision_fingerprint(i)
                        for cid, i in inds.items()}
    print(f"[LOAD] {len(inds)} candidates; path_lib reuse={stats['lib_hit']} "
          f"reconstructed={stats['rebuilt']}")

    rows, nesting = [], {}
    for seed in seeds:
        master = build_master(env, seed)
        subsets, report = nested_subsets(master, seed)
        nesting[str(seed)] = report
        for S in s_values:
            scen = subsets[S]
            install_scenario_set(scen)
            for cid in EXPECTED_CANDIDATE_IDS:
                if cid not in inds:
                    continue
                rows.append(evaluate_candidate(
                    env, inds[cid], batches, arcs, tt_dict, scen, cid, seed, S))
            assert_cache_confined(scen)
        for cid in inds:
            check_maxlate_monotonicity(
                [r for r in rows
                 if r["stage1_candidate_id"] == cid and r["master_seed"] == seed],
                cid, seed)
        print(f"[SEED {seed}] done ({len(s_values)} S values x {len(inds)} candidates)")

    for cid, before in decisions_before.items():
        if individual_decision_fingerprint(inds[cid]) != before:
            fail(f"{cid}: candidate decision structure was mutated during "
                 "evaluation; results cannot be trusted")

    expected = len(seeds) * len(s_values) * len(inds)
    if len(rows) != expected:
        fail(f"expected {expected} rows, produced {len(rows)}")
    per_cs, per_s = summarise(rows, cand_meta, mode)
    man = base_manifest(doc, mode, seeds)
    man["n_rows"] = len(rows)
    man["candidates_evaluated"] = sorted(inds)
    man["candidate_loading"] = dict(path_lib_reuse=stats["lib_hit"],
                                    reconstructed=stats["rebuilt"])
    write_outputs(rows, per_cs, per_s, nesting, man, out_dir)
    return rows


def self_test(env, doc, cands):
    """Integrity-only. Performs ZERO candidate evaluations."""
    print("=== STAGE-1 SELF-TEST (no candidate evaluation) ===")
    batches = env["batches"]
    stats = {"lib_hit": 0, "rebuilt": 0}
    for e in cands:
        build_candidate_individual(e, batches, env["path_lib"], env["tt_dict"],
                                   env["arc_lookup"], stats)
    print(f"  PASS  10 candidates reconstruct exactly "
          f"(path_lib reuse={stats['lib_hit']} rebuilt={stats['rebuilt']})")

    if set(MASTER_SEEDS) & FORBIDDEN_SEEDS:
        fail("Stage-1 seeds overlap previously used seeds")
    if len(set(MASTER_SEEDS)) != 10:
        fail("Stage-1 seeds are not 10 distinct values")
    print(f"  PASS  10 fresh distinct seeds, disjoint from {sorted(FORBIDDEN_SEEDS)}")

    for seed in MASTER_SEEDS:
        master = build_master(env, seed)
        subsets, _ = nested_subsets(master, seed)
        install_scenario_set(subsets[50])
        install_scenario_set(subsets[100])
        install_scenario_set(subsets[200])
    print(f"  PASS  border-event config {EXPECTED_BORDER_EVENTS}/"
          f"{EXPECTED_BORDER_ARCS} on all {len(MASTER_SEEDS)} seeds")
    print("  PASS  nested prefixes S50 subset S100 subset S200 verified on all seeds")
    print("  PASS  cache-safe install works for every subset")

    if HELD_OUT_ENABLED or HELD_OUT_SEED is not None or HELD_OUT_S is not None:
        fail("held-out validation must remain disabled in this phase")
    print("  PASS  held-out validation designed but disabled")
    print("\nSELF-TEST PASSED -- 0 candidate evaluations performed")


def main():
    ap = argparse.ArgumentParser(description="A-lite Stage 1")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--self-test", action="store_true")
    g.add_argument("--run", action="store_true")
    g.add_argument("--sanity-run", action="store_true")
    ap.add_argument("--i-have-approval", action="store_true")
    ap.add_argument("--sanity-seeds", type=int, default=1)
    ap.add_argument("--sanity-candidates", type=int, default=2)
    a = ap.parse_args()

    env = prepare_environment()
    doc, cands = load_frozen_checkpoint()
    print(f"[PROV] candidate-array SHA-256 verified "
          f"{EXPECTED_CANDIDATES_SHA256[:16]}...  ids S1-01..S1-10")

    if a.self_test:
        self_test(env, doc, cands)
        return
    if a.sanity_run:
        if not a.i_have_approval:
            fail("--sanity-run runs a reduced SCIENTIFIC sample and requires "
                 "explicit sign-off: pass --i-have-approval.")
        if not 1 <= a.sanity_seeds <= len(MASTER_SEEDS):
            fail(f"--sanity-seeds must be in 1..{len(MASTER_SEEDS)}, "
                 f"got {a.sanity_seeds}")
        if not 1 <= a.sanity_candidates <= len(cands):
            fail(f"--sanity-candidates must be in 1..{len(cands)}, "
                 f"got {a.sanity_candidates}")
        run_study(env, doc, cands[:a.sanity_candidates],
                  MASTER_SEEDS[:a.sanity_seeds], S_VALUES, "sanity-run",
                  SANITY_OUT_DIR)
        return
    if not a.i_have_approval:
        fail("--run executes the full Stage-1 study and requires explicit "
             "sign-off: pass --i-have-approval.")
    run_study(env, doc, cands, MASTER_SEEDS, S_VALUES, "full-run", OUT_DIR)


if __name__ == "__main__":
    main()
