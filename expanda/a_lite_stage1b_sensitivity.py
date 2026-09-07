#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""A-lite STAGE 1B -- fixed-candidate CCP scenario-count ADEQUACY study.

SCIENTIFIC QUESTION (and ONLY this question)
--------------------------------------------
Is a scenario count S in {200, 500} ADEQUATE relative to a larger empirical
reference S_ref = 1000, for the 10 FIXED Stage-1 candidates, judged by the
frozen gates G1 (probability stability), G2dagger (candidate-level one-sided
optimism) and G3 (OFF-boundary classification stability)?

This is explicitly NOT:
  - an optimisation experiment (no search of any kind is performed),
  - an SAA convergence proof (S=1000 is an empirical reference, NOT ground
    truth, and no theoretical convergence claim is made),
  - evidence about NSGA-II search-time selection bias (that is Stage 2).

FROZEN METHODOLOGY
------------------
Every rule implemented here is fixed by, and traceable to,
    expanda/A_LITE_STAGE1B_METHODOLOGY_AMENDMENT.md
whose SHA-256 is pinned below and verified before any evaluation. This driver
implements that document and does not reinterpret it. If an implementation
need ever conflicts with it, the correct action is to STOP and report a
methodology/implementation mismatch -- never to adjust a rule here.

DESIGN
------
10 frozen candidates (S1-01..S1-10) x 10 NEW master seeds x S in {200,500,1000}
= 300 raw rows. For each master seed exactly ONE master ScenarioSet of size
1000 is built; S=200 and S=500 are literal array PREFIXES of that master,
never independent redraws. build_scenario_set(seed, 200) is NOT the prefix of
build_scenario_set(seed, 1000) -- the shared RNG stream advances by `size` per
arc key -- so independent regeneration is PROHIBITED and never performed.

Because S200/S500/S1000 are nested they are POSITIVELY CORRELATED by
construction (deliberate common-random-numbers coupling, which is what
isolates the effect of ADDING scenarios). They must NOT be treated as
independent samples. Independent replication comes ONLY from the 10 distinct
master seeds. The 100 candidate x seed cells are crossed-clustered (all 10
candidates share one ScenarioSet per seed), so effective replication is of
order 10, not 100.

GATING vs NON-GATING -- structurally enforced
---------------------------------------------
The ONLY scientific acceptance gates are G1, G2dagger and G3. Every statistic
this module produces carries an explicit `gating` boolean, and the verdict is
computed exclusively from statistics with gating=True (asserted at runtime).
The 0.01 optimism tripwire, the seed-cluster bootstrap and the
leave-one-candidate-out (LOCO) sensitivity analysis are DESCRIPTIVE ONLY:
they can never create a PASS, a FAIL or an INDETERMINATE category, and they
never block Stage 2 independently. There is NO INDETERMINATE category.

REUSE
-----
This script does NOT modify baseline_uncertainty.py or any other production
module. It imports the audited Stage-1 driver and reuses its evaluation,
provenance, cache-safety and candidate-loading helpers unchanged rather than
copying production logic.

EXECUTION MODES
---------------
  --self-test   integrity only: frozen-methodology hash, candidate checkpoint,
                seed list, nesting construction, border configuration, gate
                arithmetic on synthetic cases, tie semantics and non-gating
                semantics. Performs ZERO candidate evaluations.
  --run         the full Stage-1B study (requires --i-have-approval).
  --sanity-run  a deliberately tiny plumbing run (requires --i-have-approval);
                its outputs are scientifically UNINTERPRETABLE and are written
                to a separate directory.
"""
import argparse
import csv
import hashlib
import json
import math
import statistics
import subprocess
import sys
from fractions import Fraction
from pathlib import Path as FSPath
from types import MappingProxyType

import numpy as np

HERE = FSPath(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import baseline_uncertainty as model  # noqa: E402
import scenario_count_sensitivity as scs  # noqa: E402  (audited nesting helpers)
import a_lite_stage1_sensitivity as s1  # noqa: E402  (audited Stage-1 driver)

# ════════════════════════════════════════════════════════
# FROZEN METHODOLOGY PROVENANCE
# ════════════════════════════════════════════════════════

AMENDMENT = HERE / "A_LITE_STAGE1B_METHODOLOGY_AMENDMENT.md"
EXPECTED_AMENDMENT_SHA256 = (
    "df85d40b041f3a3882c2bca1adeec9b2b2b06b6f2452a920b4c23d9ab21a0188")
# The commit that froze the methodology. Recorded for provenance and used to
# cross-check the amendment blob. Deliberately NOT compared against HEAD:
# HEAD legitimately advances (e.g. when this driver is committed), whereas the
# amendment CONTENT must not change. Content-addressing is the invariant.
METHODOLOGY_FREEZE_COMMIT = "369585eb6d781dc4f69cf05960842bdded26ee09"
AMENDMENT_REPO_PATH = "expanda/A_LITE_STAGE1B_METHODOLOGY_AMENDMENT.md"

# ════════════════════════════════════════════════════════
# DECLARED CONFIGURATION -- fixed BEFORE any Stage-1B result exists
# ════════════════════════════════════════════════════════

# Ten NEW master seeds. A contiguous block is used so the choice is
# transparently not cherry-picked. Disjoint from every seed used anywhere in
# this project, INCLUDING all ten Stage-1 seeds (700001..700010).
MASTER_SEEDS = [810001, 810002, 810003, 810004, 810005,
                810006, 810007, 810008, 810009, 810010]

# Every seed that must never be reused as a Stage-1B master seed.
FORBIDDEN_SEEDS = set(s1.FORBIDDEN_SEEDS) | set(s1.MASTER_SEEDS)

S_VALUES = [200, 500, 1000]
MASTER_S = 1000
S_REF = 1000                 # the reference for EVERY gating comparison
GATING_S = [200, 500]        # ladder order: S200 first, then S500
DIAGNOSTIC_PAIR = (200, 500)  # reported, NEVER gates

RISK_METRIC = "ccp"
EXPECTED_CONFIDENCE_ONTIME = 0.90   # alpha; tie rule is INCLUSIVE (p >= alpha)
ALPHA_EXACT = Fraction(9, 10)       # alpha as an exact rational

EXPECTED_N_CANDIDATES = 10
EXPECTED_N_SEEDS = 10
EXPECTED_N_CELLS = EXPECTED_N_CANDIDATES * EXPECTED_N_SEEDS          # 100
EXPECTED_N_ROWS = EXPECTED_N_CELLS * len(S_VALUES)                   # 300

# ── FROZEN GATE THRESHOLDS (amendment sections 2, 3, 7) ──────────────────
# All three gates are STRICT: a statistic exactly at its threshold FAILS.
THRESHOLD_G1 = 0.02          # mean |dp| over 100 cells
THRESHOLD_G2DAGGER = 0.02    # max_c D_c
THRESHOLD_G3 = 0.10          # OFF-stratum gross flip rate
# Non-gating tripwire (amendment section 4). INCLUSIVE: |T| >= 0.01 trips.
TRIPWIRE_T = 0.01

# ── FROZEN STRATA (amendment section 7; pre-registered from Stage-1) ─────
# Membership is FROZEN. Candidates are never relabelled after results.
KNIFE_CANDIDATES = ("S1-01", "S1-02", "S1-03", "S1-05", "S1-06")
OFF_CANDIDATES = ("S1-04", "S1-07", "S1-08", "S1-09", "S1-10")

# Descriptive-only bootstrap configuration. This RNG seed resamples SEED
# LABELS for a descriptive interval; it is never used to generate scenarios
# and cannot affect any gate.
BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_RNG_SEED = 20260817
BOOTSTRAP_CI = 0.80

OUT_DIR = HERE / "a_lite_stage1b_results"
SANITY_OUT_DIR = HERE / "a_lite_stage1b_sanity"

TOL_HARD = s1.TOL_HARD        # 1e-12, matches production
TOL_MONOTONE = s1.TOL_MONOTONE

# The ScenarioSet fields that carry per-scenario arrays. If the production
# dataclass ever gains another scenario-indexed field, prefix verification
# would silently stop covering it -- so the schema is pinned and checked.
EXPECTED_SCENARIOSET_FIELDS = {
    "size", "seed", "travel_multiplier", "border_delay_h",
    "arc_border_event", "border_event_mean_h", "stochastic",
}
SCENARIO_INDEXED_FIELDS = ("travel_multiplier", "border_delay_h")


class Stage1BIntegrityError(RuntimeError):
    """Any fail-closed condition. Never caught, never downgraded."""


def fail(msg):
    raise Stage1BIntegrityError(msg)


# ════════════════════════════════════════════════════════
# Frozen-methodology verification
# ════════════════════════════════════════════════════════

def sha256_file(path):
    return hashlib.sha256(FSPath(path).read_bytes()).hexdigest()


def verify_frozen_methodology(require_git=True):
    """Fail closed unless the frozen amendment is byte-identical to the
    version this driver was written against.

    Two independent checks:
      (1) the working-tree file hashes to the pinned SHA-256;
      (2) the blob stored at the methodology-freeze commit hashes to the same
          value, so a working-tree edit cannot masquerade as the frozen text.
    """
    if not AMENDMENT.exists():
        fail(f"frozen methodology document not found: {AMENDMENT}")
    got = sha256_file(AMENDMENT)
    if got != EXPECTED_AMENDMENT_SHA256:
        fail("FROZEN METHODOLOGY HASH MISMATCH\n"
             f"  expected {EXPECTED_AMENDMENT_SHA256}\n"
             f"  got      {got}\n"
             "  The methodology amendment has changed. Stage 1B implements a "
             "frozen document and must not run against a modified one. If the "
             "change is intended, it is a NEW methodology identity: re-review "
             "it, re-freeze it, and update EXPECTED_AMENDMENT_SHA256 in a "
             "reviewed change. Never relax this check to make a run proceed.")
    committed = None
    try:
        blob = subprocess.run(
            ["git", "cat-file", "-p",
             f"{METHODOLOGY_FREEZE_COMMIT}:{AMENDMENT_REPO_PATH}"],
            cwd=str(HERE.parent), capture_output=True, check=True)
        committed = hashlib.sha256(blob.stdout).hexdigest()
    except Exception as exc:                      # noqa: BLE001
        if require_git:
            fail("cannot read the frozen amendment from the methodology-freeze "
                 f"commit {METHODOLOGY_FREEZE_COMMIT}: {exc}. Refusing to run "
                 "without verifiable methodology provenance.")
    if committed is not None and committed != EXPECTED_AMENDMENT_SHA256:
        fail("the amendment blob at the methodology-freeze commit does not "
             f"match the pinned hash (got {committed}). Provenance is broken.")
    return {"amendment_path": str(AMENDMENT),
            "amendment_sha256": got,
            "freeze_commit": METHODOLOGY_FREEZE_COMMIT,
            "freeze_commit_blob_sha256": committed,
            "verified": True}


def verify_seed_list(seeds):
    """The seed list is declared in source BEFORE any Stage-1B result exists."""
    if len(seeds) != EXPECTED_N_SEEDS:
        fail(f"expected exactly {EXPECTED_N_SEEDS} master seeds, got {len(seeds)}")
    if len(set(seeds)) != len(seeds):
        fail(f"duplicate master seed in {seeds}")
    clash = sorted(set(seeds) & FORBIDDEN_SEEDS)
    if clash:
        fail(f"master seeds {clash} are reserved/previously used (including all "
             f"Stage-1 seeds {sorted(s1.MASTER_SEEDS)}) and must not be reused")
    for s in seeds:
        if not isinstance(s, int) or isinstance(s, bool) or s < 0:
            fail(f"master seed {s!r} is not a non-negative int")
    return True


def verify_strata():
    knife, off = set(KNIFE_CANDIDATES), set(OFF_CANDIDATES)
    if knife & off:
        fail(f"KNIFE and OFF strata overlap: {sorted(knife & off)}")
    if knife | off != set(s1.EXPECTED_CANDIDATE_IDS):
        fail("KNIFE u OFF does not equal the frozen candidate panel")
    if len(knife) != 5 or len(off) != 5:
        fail(f"strata sizes wrong: KNIFE={len(knife)} OFF={len(off)}, expected 5/5")
    return True


def stratum_of(cid):
    if cid in KNIFE_CANDIDATES:
        return "KNIFE"
    if cid in OFF_CANDIDATES:
        return "OFF"
    fail(f"candidate {cid} belongs to no frozen stratum")


# ════════════════════════════════════════════════════════
# Scenario construction: ONE S=1000 master per seed, literal prefixes
# ════════════════════════════════════════════════════════

def assert_scenarioset_schema(scen):
    fields = set(getattr(scen, "__dataclass_fields__", {}))
    if fields != EXPECTED_SCENARIOSET_FIELDS:
        fail("ScenarioSet schema changed: "
             f"unexpected={sorted(fields - EXPECTED_SCENARIOSET_FIELDS)} "
             f"missing={sorted(EXPECTED_SCENARIOSET_FIELDS - fields)}. Prefix "
             "verification covers only the known scenario-indexed fields "
             f"{SCENARIO_INDEXED_FIELDS}; a new one would be silently "
             "unverified. Refusing to run.")


def build_master(env, seed):
    """Build exactly ONE master ScenarioSet of size 1000 for `seed`.

    This is the ONLY call to build_scenario_set anywhere in Stage 1B. S=200
    and S=500 are derived from this object by literal prefix slicing.
    """
    verify_seed_list(MASTER_SEEDS)
    if seed in FORBIDDEN_SEEDS:
        fail(f"seed {seed} is reserved/previously used and must not be reused")
    master = model.build_scenario_set(
        arcs=env["arcs"], border_delay_map=env["border_delay_map"],
        size=MASTER_S, seed=seed, stochastic=True,
        border_event_definitions=env["border_event_definitions"])
    assert_scenarioset_schema(master)
    n_events = len(master.border_event_mean_h)
    n_arcs = len(master.arc_border_event)
    if (n_events != s1.EXPECTED_BORDER_EVENTS
            or n_arcs != s1.EXPECTED_BORDER_ARCS):
        fail(f"border-event configuration mismatch for seed {seed}: got "
             f"{n_events} events / {n_arcs} arcs, expected "
             f"{s1.EXPECTED_BORDER_EVENTS} / {s1.EXPECTED_BORDER_ARCS}. "
             "Refusing to run against a different stochastic configuration "
             "than the validated one.")
    if master.size != MASTER_S or master.seed != seed or not master.stochastic:
        fail(f"unexpected master ScenarioSet metadata for seed {seed}")
    for name in SCENARIO_INDEXED_FIELDS:
        for key, arr in getattr(master, name).items():
            if arr.shape[0] != MASTER_S:
                fail(f"master {name}[{key}] length {arr.shape[0]} != {MASTER_S}")
            if not np.all(np.isfinite(arr)):
                fail(f"master {name}[{key}] contains non-finite values")
    return master


def nested_subsets(master, seed, s_values=None):
    """S200/S500 as literal prefixes of `master`; S1000 IS the master.

    Every pairwise nesting relation is independently re-verified, on every
    scenario-indexed field, with a schema tripwire so a new field cannot go
    unverified.
    """
    s_values = list(S_VALUES if s_values is None else s_values)
    if MASTER_S not in s_values:
        fail(f"the master size {MASTER_S} must be part of the S grid")
    subsets = {S: scs.make_nested_subset(master, S)
               for S in s_values if S != MASTER_S}
    subsets[MASTER_S] = master

    report = {}
    pairs = [(a, b) for i, a in enumerate(sorted(s_values))
             for b in sorted(s_values)[i + 1:]]
    for small, large in pairs:
        v = scs.verify_prefix_subset(subsets[small], subsets[large])
        report[f"S{small}_subset_of_S{large}"] = v
        if not v["is_prefix_subset"]:
            fail(f"nested-prefix verification FAILED for seed {seed} "
                 f"(S{small} in S{large}): {v['mismatched_keys']}")
    for S, scen in subsets.items():
        assert_scenarioset_schema(scen)
        if scen.size != S or scen.seed != seed or not scen.stochastic:
            fail(f"subset metadata wrong for seed {seed} S={S}")
        if len(scen.border_event_mean_h) != s1.EXPECTED_BORDER_EVENTS:
            fail(f"subset border_event_mean_h changed for seed {seed} S={S}")
        if len(scen.arc_border_event) != s1.EXPECTED_BORDER_ARCS:
            fail(f"subset arc_border_event changed for seed {seed} S={S}")
        for name in SCENARIO_INDEXED_FIELDS:
            for key, arr in getattr(scen, name).items():
                if arr.shape[0] != S:
                    fail(f"{name}[{key}] length {arr.shape[0]} != {S} "
                         f"(seed {seed})")
    report["all_pass"] = True
    report["verified_pairs"] = [f"S{a}_subset_of_S{b}" for a, b in pairs]
    report["scenario_indexed_fields"] = list(SCENARIO_INDEXED_FIELDS)
    return subsets, report


def check_maxlate_monotonicity(rows_for_candidate_seed, cand_id, seed, s_values):
    """Under EXACT nested prefixes max_late_excess_h is a sum over batches of a
    max over scenarios, so it is deterministically NON-DECREASING in S."""
    by_S = {r["S"]: r["max_late_excess_h"] for r in rows_for_candidate_seed}
    ordered = sorted(s_values)
    for a, b in zip(ordered, ordered[1:]):
        if a in by_S and b in by_S and by_S[b] < by_S[a] - TOL_MONOTONE:
            fail(f"max_late_excess_h monotonicity VIOLATED for {cand_id} "
                 f"seed {seed}: S{a}={by_S[a]!r} > S{b}={by_S[b]!r}. Under "
                 "exact nested prefixes this is impossible; treating as an "
                 "integrity failure rather than a result.")


# ════════════════════════════════════════════════════════
# Cells: the paired data structure every gate is computed from
# ════════════════════════════════════════════════════════

def ccp_pass(chance_vio_total):
    """The FROZEN inclusive CCP probability decision, exactly as production
    computes it: pass iff the summed shortfall is within TOL_HARD.

    On the attainable grid (every p_b is a multiple of 1/S with S <= 1000, so
    every positive shortfall is >= 1e-3) this is identical to the inclusive
    rule p_hat >= alpha, and therefore p_hat == 0.90 counts as a PASS.
    """
    v = float(chance_vio_total)
    if not np.isfinite(v):
        fail(f"non-finite chance_vio_total {v!r}: cannot classify")
    return bool(v <= TOL_HARD)


def _freeze_row(row):
    """A deeply read-only view of one validated result row.

    Sequences become tuples and the mapping becomes a MappingProxyType, so a
    holder of the certified grid cannot edit a cell after it was validated.
    Only the grid's own snapshot is frozen; the raw `rows` list that
    write_outputs() serialises is untouched.
    """
    frozen = {}
    for k, v in row.items():
        frozen[k] = tuple(v) if isinstance(v, list) else v
    return MappingProxyType(frozen)


class ValidatedGrid:
    """Proof that the FULL declared cell grid passed fail-closed validation.

    Gate evaluation requires an instance of this type. A plain dict of rows
    can therefore never reach a gate: there is no code path that computes a
    verdict without the complete-grid precheck having succeeded first.

    The type is PROOF-CARRYING, not merely a marker, and the proof has to
    survive the moment of construction:

      * construction RE-RUNS the full fail-closed check over its contents, so
        fabricating an instance around an incomplete or defective index RAISES
        instead of laundering unvalidated data into a gate;
      * the validated rows are SNAPSHOTTED AND FROZEN (MappingProxyType over
        rows whose sequences are tuples), so neither the caller's original
        rows nor the certified copy can be edited after the check passed;
      * the instance itself is immutable and the class is FINAL, so no
        subclass can validate a legitimate index and then override lookup to
        substitute different contents.

    Validation can therefore not be skipped, undone, or substituted by
    constructing, mutating or subclassing the container.
    """

    __slots__ = ("_index", "candidates", "seeds", "s_values", "n_cells")

    def __init_subclass__(cls, **kwargs):
        fail("ValidatedGrid is final and must not be subclassed: a subclass "
             "could pass validation on one index and then override lookup to "
             "serve different contents to the gates.")

    def __init__(self, index, candidates, seeds, s_values, duplicates=()):
        candidates = tuple(candidates)
        seeds = tuple(int(s) for s in seeds)
        s_values = tuple(int(S) for S in s_values)
        report = grid_defect_report(index, candidates, seeds, s_values,
                                    duplicates)
        if report is not None:
            fail("ValidatedGrid may only be constructed from a grid that "
                 "passes the fail-closed precheck; refusing to certify an "
                 "invalid grid as validated. "
                 f"counts={json.dumps(report['counts'])}")
        # Snapshot AND freeze what was validated. A copy alone would still let
        # the holder edit the certified grid; the certificate has to describe
        # something that can no longer change.
        object.__setattr__(self, "_index", MappingProxyType(
            {k: _freeze_row(v) for k, v in index.items()}))
        object.__setattr__(self, "candidates", candidates)
        object.__setattr__(self, "seeds", seeds)
        object.__setattr__(self, "s_values", s_values)
        object.__setattr__(self, "n_cells", len(self._index))

    @property
    def index(self):
        """Read-only view of the certified snapshot. Assignment into it raises
        TypeError, so `vg.index[key] = ...` cannot rewrite a validated cell."""
        return self._index

    def __setattr__(self, name, value):
        fail(f"ValidatedGrid is immutable; refusing to rebind {name!r} after "
             "certification")

    def __delattr__(self, name):
        fail("ValidatedGrid is immutable; refusing to delete "
             f"{name!r} after certification")

    def __getitem__(self, key):
        return self._index[key]

    def validation_record(self):
        """The explicit, machine-readable SUCCESSFUL-validation status that
        the manifest and the report must carry. An absent record means the
        run cannot claim its grid was validated."""
        expected = (len(self.candidates) * len(self.seeds)
                    * len(self.s_values))
        return {
            "status": "VALIDATED",
            "rule": ("amendment section 2.2 item 10: every declared "
                     "(candidate, seed, S) cell present exactly once, "
                     "probability finite in [0,1], chance_vio_total finite "
                     "and >= 0, and the stored chance_constraint_pass equal "
                     "to the recomputed frozen inclusive predicate"),
            "expected_cells": expected,
            "validated_cells": self.n_cells,
            "n_candidates": len(self.candidates),
            "n_seeds": len(self.seeds),
            "s_values": list(self.s_values),
            "imputed_cells": 0, "dropped_cells": 0, "reduced_denominators": 0,
        }


def index_rows(rows):
    """Index raw rows by (candidate_id, master_seed, S), collecting duplicates
    rather than stopping at the first one."""
    idx, duplicates = {}, []
    for r in rows:
        key = (r["candidate_id"], int(r["master_seed"]), int(r["S"]))
        if key in idx:
            duplicates.append(key)
        idx[key] = r
    return idx, duplicates


def cell_defects(row, key):
    """ALL validity defects of one cell (amendment section 2.2 item 10).

    Returns a list so the caller can enumerate every offending triple instead
    of aborting on the first one.
    """
    defects = []
    p = row.get("min_on_time_prob")
    try:
        pf = float(p)
    except (TypeError, ValueError):
        return [f"min_on_time_prob not numeric ({p!r})"]
    if not np.isfinite(pf):
        defects.append(f"min_on_time_prob non-finite ({p!r})")
    elif not (0.0 <= pf <= 1.0):
        defects.append(f"min_on_time_prob {pf!r} outside [0,1]")
    cv = row.get("chance_vio_total")
    try:
        cvf = float(cv)
    except (TypeError, ValueError):
        defects.append(f"chance_vio_total not numeric ({cv!r})")
        cvf = None
    if cvf is not None:
        if not np.isfinite(cvf):
            defects.append(f"chance_vio_total non-finite ({cv!r})")
        elif cvf < 0.0:
            defects.append(f"negative chance_vio_total {cv!r}")
        else:
            stored = row.get("chance_constraint_pass")
            # Strict: a non-boolean must never be coerced. bool("False") is
            # True, so accepting a string here would invert a classification.
            if not isinstance(stored, (bool, np.bool_)):
                defects.append(
                    f"chance_constraint_pass is {type(stored).__name__} "
                    f"({stored!r}), not a boolean; refusing to coerce")
            elif bool(stored) != ccp_pass(cvf):
                defects.append(
                    f"stored chance_constraint_pass={stored!r} disagrees with "
                    "the frozen inclusive rule recomputed from "
                    f"chance_vio_total ({ccp_pass(cvf)!r})")
    return defects


def validate_cell(row, key):
    """Single-cell convenience wrapper; raises on the first defect found."""
    defects = cell_defects(row, key)
    if defects:
        fail(f"cell {key}: " + "; ".join(defects))
    return True


def grid_defect_report(index, candidates, seeds, s_values, duplicates=()):
    """EVERY defect of a candidate x seed x S grid, enumerated in ONE pass.

    Returns None when the grid is clean, otherwise the machine-readable
    validation-failure report of amendment section 2.2 item 10. Validation
    never stops at the first defect: missing, duplicate, unexpected and
    per-cell-invalid triples are all collected, so an operator sees the whole
    picture rather than one symptom at a time.

    This is the SINGLE source of truth for grid validity. Both
    require_complete_grid() and ValidatedGrid construction go through it, so
    there is no route to a certified grid that skipped it.
    """
    expected, missing = [], []
    for c in candidates:
        for s in seeds:
            for S in s_values:
                key = (c, int(s), int(S))
                expected.append(key)
                if key not in index:
                    missing.append(key)
    extra = sorted(set(index) - set(expected))
    invalid = []
    for key in expected:
        row = index.get(key)
        if row is None:
            continue
        defects = cell_defects(row, key)
        if defects:
            invalid.append({"cell": list(key), "defects": defects})
    duplicates = [tuple(k) for k in duplicates]
    if not (missing or extra or duplicates or invalid):
        return None
    return {
        "status": "INVALID_RUN",
        "stage": "A-lite Stage 1B",
        "rule": ("Per the frozen methodology (amendment section 2.2 item "
                 "10): if any of the declared cells is invalid, the ENTIRE "
                 "Stage-1B run is INVALID; neither S200 nor S500 may pass, "
                 "and Stage 2 must not start. No scientific result "
                 "artifacts are written. Denominators are never reduced, "
                 "values are never imputed, cells are never dropped."),
        "expected_cells": len(expected),
        "missing_cells": [list(k) for k in missing],
        "duplicate_cells": [list(k) for k in duplicates],
        "unexpected_cells": [list(k) for k in extra],
        "invalid_cells": invalid,
        "counts": {"missing": len(missing), "duplicate": len(duplicates),
                   "unexpected": len(extra), "invalid": len(invalid)},
    }


def require_complete_grid(rows, candidates, seeds, s_values, out_dir=None):
    """Fail-closed completeness over the FULL declared grid.

    No imputation, no dropping, no reduced denominator. Every offending
    triple is enumerated -- validation does NOT stop at the first defect --
    and, if `out_dir` is given, a dedicated validation-failure artifact
    containing only the invalid status and the offending triples is written
    before the run is aborted. No scientific result artifact is written.
    """
    idx, duplicates = index_rows(rows)
    report = grid_defect_report(idx, candidates, seeds, s_values, duplicates)
    if report is not None:
        missing = [tuple(k) for k in report["missing_cells"]]
        extra = [tuple(k) for k in report["unexpected_cells"]]
        invalid = report["invalid_cells"]
        n_expected = report["expected_cells"]
        written = None
        if out_dir is not None:
            out_dir = FSPath(out_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            target = out_dir / "VALIDATION_FAILURE.json"
            target.write_text(json.dumps(report, indent=2, default=json_default),
                              encoding="utf-8")
            written = str(target)
        fail("INVALID RUN -- the entire Stage-1B run is invalid and Stage 2 "
             "must not start.\n"
             f"  missing={len(missing)} duplicate={len(duplicates)} "
             f"unexpected={len(extra)} invalid={len(invalid)} "
             f"of {n_expected} declared cells\n"
             f"  offending triples: {json.dumps(report['counts'])}\n"
             f"  first missing:   {missing[:3]}\n"
             f"  first duplicate: {duplicates[:3]}\n"
             f"  first invalid:   {[i['cell'] for i in invalid[:3]]}\n"
             f"  validation-failure artifact: {written}\n"
             "  Denominators are never reduced and no value is ever imputed.")
    return ValidatedGrid(idx, candidates, seeds, s_values, duplicates)


_CELLS_TOKEN = object()


class ValidatedCells(dict):
    """Paired (candidate, seed) cells that carry PROOF OF PROVENANCE.

    Every gate requires this exact type. It can only be produced by the two
    audited constructors in this module:

      * build_cells(), which reads a ValidatedGrid and therefore inherits a
        completed fail-closed validation, and
      * _synthetic_cells(), which realises test values on the ATTAINABLE
        empirical grid and rejects anything a real estimator could not emit.

    A hand-assembled mapping is refused however well-formed it looks -- exact
    Fractions and genuine booleans are NOT sufficient, because the property a
    gate needs is not "well-shaped" but "descended from a validated grid".
    The mapping and every cell in it are IMMUTABLE and the class is FINAL, so
    a caller cannot take legitimate cells, swap one valid Fraction or boolean
    for another, and then obtain a verdict on the altered data.

    STATED LIMITATION. This closes accidental and casual bypasses, which is
    what the methodology's "gates may never be computed on an unvalidated
    grid" requires. It is NOT a security boundary: any caller executing inside
    this process can rebind module attributes -- the construction token, a
    gate function, or even THRESHOLD_G1_EXACT -- and no Python-level
    construction can prevent that. Provenance here is an engineering
    guarantee against mistakes, not a defence against a hostile in-process
    actor.
    """

    __slots__ = ()

    _MUTATION = ("ValidatedCells is immutable: a certified cell mapping may "
                 "not be edited after construction, because a gate would then "
                 "be computed on data that never passed validation. Build a "
                 "different mapping instead.")

    def __init_subclass__(cls, **kwargs):
        fail("ValidatedCells is final and must not be subclassed: a subclass "
             "could override lookup and substitute cell contents after the "
             "mapping was certified.")

    def __init__(self, mapping, token=None):
        if token is not _CELLS_TOKEN:
            fail("ValidatedCells may only be produced by build_cells() (from a "
                 "ValidatedGrid) or by _synthetic_cells(); a gate may never be "
                 "computed on a hand-assembled or forged cell mapping, however "
                 "well-formed its exact values and boolean flags appear.")
        # dict.__init__ does not route through __setitem__, so the frozen
        # cells are installed once and can never be replaced afterwards.
        super().__init__({k: MappingProxyType(dict(v))
                          for k, v in mapping.items()})

    def __setitem__(self, *a, **k):
        fail(self._MUTATION)

    def __delitem__(self, *a, **k):
        fail(self._MUTATION)

    def update(self, *a, **k):
        fail(self._MUTATION)

    def setdefault(self, *a, **k):
        fail(self._MUTATION)

    def pop(self, *a, **k):
        fail(self._MUTATION)

    def popitem(self, *a, **k):
        fail(self._MUTATION)

    def clear(self, *a, **k):
        fail(self._MUTATION)

    def __ior__(self, other):
        fail(self._MUTATION)


def _require_cells(cells, where):
    """Every gate entry point starts here: no provenance, no verdict."""
    if type(cells) is not ValidatedCells:
        fail(f"{where} requires ValidatedCells descended from a completed "
             "fail-closed validation (build_cells() on a ValidatedGrid, or "
             "_synthetic_cells()); refusing to compute a gate on a "
             f"{type(cells).__name__}.")


def build_cells(idx, S, S_ref, candidates, seeds):
    """The paired (candidate, seed) structure for ONE comparison.

    Requires a ValidatedGrid: cells can only be built from a grid that has
    already passed the full fail-closed precheck. The grid's own validated
    snapshot is read directly, never through __getitem__, so no override can
    interpose between the certificate and the gate.
    """
    if type(idx) is not ValidatedGrid:
        fail("build_cells requires a ValidatedGrid; gates may never be "
             "computed on a grid that has not passed require_complete_grid()")
    store = idx.index
    cells = {}
    for c in candidates:
        for s in seeds:
            a, b = store[(c, int(s), int(S))], store[(c, int(s), int(S_ref))]
            p_s, p_r = float(a["min_on_time_prob"]), float(b["min_on_time_prob"])
            # Exact rationals drive every gate; the floats are for reporting.
            ex_s, ex_r = exact_p(p_s, S), exact_p(p_r, S_ref)
            cells[(c, int(s))] = {
                "p_S": p_s, "p_ref": p_r, "dp": float(ex_s - ex_r),
                "p_S_exact": ex_s, "p_ref_exact": ex_r, "dp_exact": ex_s - ex_r,
                "pass_S": bool(a["chance_constraint_pass"]),
                "pass_ref": bool(b["chance_constraint_pass"]),
                "binding_S": a["binding_batch_set"],
                "binding_ref": b["binding_batch_set"],
                "mlate_S": float(a["max_late_excess_h"]),
                "mlate_ref": float(b["max_late_excess_h"]),
                "nonccp_S": bool(a["nonccp_hard_pass"]),
                "nonccp_ref": bool(b["nonccp_hard_pass"]),
            }
    return ValidatedCells(cells, _CELLS_TOKEN)


# ════════════════════════════════════════════════════════
# GATES -- pure functions. G1, G2dagger, G3 are the ONLY gates.
# ════════════════════════════════════════════════════════

def exact_p(p, S):
    """Recover p_hat EXACTLY as the rational k/S it mathematically is.

    Every p_hat is a minimum over batch-level empirical proportions, so it is
    exactly k/S for an integer k. Production returns it as a float, and
    binary floating point cannot represent those values exactly. That matters
    here because the frozen gates are STRICT (`< threshold`), so behaviour
    exactly AT a threshold is part of the specification -- and float
    subtraction breaks it: 0.95 - 0.93 evaluates to 0.019999999999999907,
    a mathematically exact 0.02 that would silently PASS a specified FAIL.
    (0.92 - 0.90 rounds the other way, so the error is not even consistent.)

    Recovering the integer numerator and doing all gate arithmetic in exact
    rationals makes the strict comparison mean what the amendment says. This
    is faithfulness to the frozen rule, not a change to it.
    """
    S = int(S)
    if S <= 0:
        fail(f"invalid scenario count {S}")
    pf = float(p)
    if not np.isfinite(pf):
        fail(f"non-finite probability {p!r}: cannot form an exact value")
    scaled = pf * S
    k = int(round(scaled))
    if abs(scaled - k) > 1e-6:
        fail(f"probability {pf!r} is not a multiple of 1/{S} (k={scaled!r}). "
             "p_hat must be an empirical proportion over exactly S scenarios; "
             "a non-grid value means the estimator or the scenario count is "
             "not what this study assumes.")
    if not 0 <= k <= S:
        fail(f"probability numerator {k} outside [0,{S}]")
    return Fraction(k, S)


def _mean_exact(values, denominator):
    """Exact rational mean. No rounding anywhere."""
    if denominator <= 0:
        fail("mean with non-positive denominator")
    total = Fraction(0)
    for v in values:
        total += v
    return total / Fraction(int(denominator))


# Frozen thresholds as EXACT rationals, so the strict comparisons are exact.
THRESHOLD_G1_EXACT = Fraction(2, 100)          # 0.02
THRESHOLD_G2DAGGER_EXACT = Fraction(2, 100)    # 0.02
THRESHOLD_G3_EXACT = Fraction(1, 10)           # 0.10
TRIPWIRE_T_EXACT = Fraction(1, 100)            # 0.01


def _dps(cells, candidates, seeds):
    """Exact signed paired differences, in a fixed deterministic order."""
    _require_cells(cells, "paired-difference extraction")
    out = []
    for c in candidates:
        for s in seeds:
            cell = cells.get((c, int(s)))
            if cell is None:
                fail(f"missing cell ({c},{s}); denominators are never reduced")
            dp = cell.get("dp_exact")
            if dp is None or not isinstance(dp, Fraction):
                fail(f"cell ({c},{s}) has no exact dp; gates require exact "
                     "rational inputs")
            out.append(dp)
    return out


def compute_g1(cells, candidates, seeds, n_cells_expected=None):
    """G1: mean over ALL paired cells of |p_S - p_ref| < 0.02 (STRICT, exact).

    `n_cells_expected` pins the FROZEN denominator (100 on a full run). When
    it is given, a reduced candidate or seed list is refused outright rather
    than quietly averaging over 90 cells: the denominator is a constant of the
    methodology, never a count of what happened to be supplied.
    """
    _require_cells(cells, "G1")
    dps = _dps(cells, candidates, seeds)
    n = len(candidates) * len(seeds)
    if n_cells_expected is not None and n != n_cells_expected:
        fail(f"G1 denominator must be exactly {n_cells_expected} cells, got "
             f"{n} ({len(candidates)} candidates x {len(seeds)} seeds); "
             "denominators are never reduced to fit the data supplied")
    if len(dps) != n:
        fail(f"G1 denominator wrong: {len(dps)} != {n}")
    value = _mean_exact((abs(d) for d in dps), n)
    return {"name": "G1", "gating": True,
            "statistic": "mean_abs_paired_probability_difference",
            "value": float(value), "value_exact": str(value),
            "threshold": THRESHOLD_G1,
            "comparison": "strict_less_than_exact_rational", "denominator": n,
            "passed": bool(value < THRESHOLD_G1_EXACT)}


def compute_d_c(cells, candidates, seeds, n_seeds_expected=None):
    """D_c(S,ref) = (1/N_s) * sum_s [p_S - p_ref], SIGNED.

    The denominator is the DECLARED number of master seeds and is never a
    surviving-row count. Positive = smaller-S optimism.
    """
    _require_cells(cells, "D_c")
    n_s = len(seeds)
    if n_seeds_expected is not None and n_s != n_seeds_expected:
        fail(f"D_c denominator must be exactly {n_seeds_expected} master "
             f"seeds, got {n_s}")
    out = {}
    for c in candidates:
        vals = []
        for s in seeds:
            cell = cells.get((c, int(s)))
            if cell is None:
                fail(f"D_c: candidate {c} missing seed {s}; the comparison is "
                     "INVALID -- the denominator is never reduced")
            dp = cell.get("dp_exact")
            if dp is None or not isinstance(dp, Fraction):
                fail(f"D_c: cell ({c},{s}) has no exact dp")
            vals.append(dp)
        if len(vals) != n_s:
            fail(f"D_c denominator wrong for {c}: {len(vals)} != {n_s}")
        out[c] = _mean_exact(vals, n_s)
    return out


def compute_a_c(cells, candidates, seeds):
    """Per-candidate MEAN ABSOLUTE difference, so a global aggregate cannot
    hide a single candidate. Non-gating."""
    _require_cells(cells, "A_c")
    out = {}
    for c in candidates:
        vals = [abs(cells[(c, int(s))]["dp_exact"]) for s in seeds]
        out[c] = _mean_exact(vals, len(seeds))
    return out


def per_candidate_full_table(cells, candidates, seeds):
    """The MANDATORY per-candidate diagnostic table (amendment section 5).

    All ten rows, never truncated: signed D_c, absolute A_c, the worst and
    best single seed, flip counts and their signed decomposition, boundary
    mass, and the frozen stratum label.
    """
    d_c = compute_d_c(cells, candidates, seeds)
    a_c = compute_a_c(cells, candidates, seeds)
    table = {}
    for c in candidates:
        dps = [cells[(c, int(s))]["dp_exact"] for s in seeds]
        nflip = netflip = nb_s = nb_r = 0
        for s in seeds:
            cell = cells[(c, int(s))]
            if cell["pass_S"] != cell["pass_ref"]:
                nflip += 1
                netflip += 1 if cell["pass_S"] else -1
            if cell["p_S_exact"] == ALPHA_EXACT:
                nb_s += 1
            if cell["p_ref_exact"] == ALPHA_EXACT:
                nb_r += 1
        table[c] = {
            "stratum": stratum_of(c),
            "D_c": float(d_c[c]), "D_c_exact": str(d_c[c]),
            "A_c": float(a_c[c]), "A_c_exact": str(a_c[c]),
            "Dmax_c": float(max(dps)), "Dmin_c": float(min(dps)),
            "nflip_c": nflip, "netflip_c": netflip,
            "nboundary_c_at_S": nb_s, "nboundary_c_at_ref": nb_r,
            "n_seeds": len(seeds),
        }
    return table


def per_seed_breakdown(cells, candidates, seeds):
    """Complete per-seed breakdown (amendment section 7.3 item 14)."""
    out = {}
    for s in seeds:
        dps = [cells[(c, int(s))]["dp_exact"] for c in candidates]
        nflip = ff = fi = 0
        for c in candidates:
            cell = cells[(c, int(s))]
            if cell["pass_S"] != cell["pass_ref"]:
                nflip += 1
                if cell["pass_S"]:
                    ff += 1
                else:
                    fi += 1
        out[str(s)] = {
            "mean_signed": float(_mean_exact(dps, len(candidates))),
            "mean_abs": float(_mean_exact([abs(d) for d in dps],
                                          len(candidates))),
            "max_abs": float(max(abs(d) for d in dps)),
            "gross_flip_count": nflip,
            "false_feasible_count": ff, "false_infeasible_count": fi,
            "signed_net_classification_imbalance": ff - fi,
            "n_candidates": len(candidates),
        }
    return out


def compute_g2dagger(cells, candidates, seeds, n_seeds_expected=None,
                     n_candidates_expected=None):
    """G2dagger: max_c D_c < 0.02 (STRICT). One-sided by construction.

    `n_seeds_expected` pins D_c's denominator (10) and
    `n_candidates_expected` pins the panel the outer max ranges over (10), so
    neither can shrink to whatever was supplied.
    """
    _require_cells(cells, "G2dagger")
    if n_candidates_expected is not None and \
            len(candidates) != n_candidates_expected:
        fail(f"G2dagger must range over exactly {n_candidates_expected} "
             f"candidates, got {len(candidates)}; the frozen panel is never "
             "filtered, reweighted or extended")
    d_c = compute_d_c(cells, candidates, seeds, n_seeds_expected)
    if not d_c:
        fail("G2dagger: no candidates")
    argmax = max(d_c, key=lambda c: d_c[c])
    argmin = min(d_c, key=lambda c: d_c[c])
    value = d_c[argmax]
    # The maximum can be attained by more than one candidate. `argmax` stays
    # the deterministic first-in-frozen-order choice, but reporting it alone
    # would conceal a tie and read as "the uniquely worst candidate", so every
    # co-argmax is recorded too. This is reporting only: the gate value and
    # the verdict do not depend on how a tie is broken.
    tied = [c for c in candidates if d_c[c] == value]
    return {"name": "G2dagger", "gating": True,
            "statistic": "max_over_candidates_of_signed_mean_seed_difference",
            "value": float(value), "value_exact": str(value),
            "threshold": THRESHOLD_G2DAGGER,
            "comparison": "strict_less_than_exact_rational",
            "argmax_candidate": argmax,
            "argmax_stratum": stratum_of(argmax),
            "argmax_candidates": tied,
            "argmax_is_tied": bool(len(tied) > 1),
            "min_D_c": float(d_c[argmin]), "argmin_candidate": argmin,
            "seed_denominator": len(seeds),
            "per_candidate_D_c": {c: float(d_c[c]) for c in candidates},
            "per_candidate_D_c_exact": {c: str(d_c[c]) for c in candidates},
            "passed": bool(value < THRESHOLD_G2DAGGER_EXACT)}


def flip_stats(cells, candidates, seeds):
    """Gross flips and their signed decomposition over a candidate subset."""
    _require_cells(cells, "classification-flip statistics")
    n = len(candidates) * len(seeds)
    if n == 0:
        fail("flip_stats: empty cell set")
    gross = false_feasible = false_infeasible = 0
    for c in candidates:
        for s in seeds:
            cell = cells.get((c, int(s)))
            if cell is None:
                fail(f"flip_stats: missing cell ({c},{s}); denominators are "
                     "never reduced")
            # Fail closed exactly as the probability gates do: a classification
            # gate may only read validated boolean decisions. Truthiness would
            # silently invert a label (bool("False") is True).
            ps, pr = cell.get("pass_S"), cell.get("pass_ref")
            if not isinstance(ps, (bool, np.bool_)) or \
               not isinstance(pr, (bool, np.bool_)):
                fail(f"cell ({c},{s}) has non-boolean pass flags "
                     f"({ps!r}, {pr!r}); classification gates require "
                     "validated boolean decisions and never coerce")
            if cell["pass_S"] != cell["pass_ref"]:
                gross += 1
                if cell["pass_S"] and not cell["pass_ref"]:
                    false_feasible += 1
                else:
                    false_infeasible += 1
    return {"n_cells": n, "gross_flip_count": gross,
            "gross_flip_rate": gross / n,
            "false_feasible_count": false_feasible,
            "false_infeasible_count": false_infeasible,
            "signed_net_classification_imbalance":
                false_feasible - false_infeasible,
            "signed_net_imbalance_rate": (false_feasible - false_infeasible) / n}


def compute_g3(cells, seeds, off_candidates=OFF_CANDIDATES,
               n_cells_expected=None):
    """G3: OFF-stratum gross classification flip rate < 0.10 (STRICT).

    With 5 OFF candidates x 10 seeds = 50 cells: 0-4 flips PASS, >=5 FAIL.
    `n_cells_expected` pins that FROZEN 50-cell denominator, so a truncated
    OFF stratum cannot silently rescale the rate.
    """
    _require_cells(cells, "G3")
    off_candidates = list(off_candidates)
    if not off_candidates:
        fail("G3 has an EMPTY OFF stratum: the gate is undefined without "
             "OFF-stratum cells. A run whose candidate subset contains no OFF "
             "candidate cannot evaluate G3 and must not report a gate verdict.")
    n = len(off_candidates) * len(seeds)
    if n_cells_expected is not None and n != n_cells_expected:
        fail(f"G3 denominator must be exactly {n_cells_expected} OFF-stratum "
             f"cells, got {n} ({len(off_candidates)} OFF candidates x "
             f"{len(seeds)} seeds); the flip rate is never rescaled to a "
             "reduced stratum")
    st = flip_stats(cells, off_candidates, seeds)
    value = Fraction(st["gross_flip_count"], st["n_cells"])
    return {"name": "G3", "gating": True,
            "statistic": "OFF_stratum_gross_classification_flip_rate",
            "value": float(value), "value_exact": str(value),
            "threshold": THRESHOLD_G3,
            "comparison": "strict_less_than_exact_rational",
            "denominator": st["n_cells"],
            "flip_count": st["gross_flip_count"],
            "stratum_membership": list(off_candidates),
            "passed": bool(value < THRESHOLD_G3_EXACT)}


# ════════════════════════════════════════════════════════
# NON-GATING diagnostics. Every one carries gating=False.
# ════════════════════════════════════════════════════════

def compute_tripwire(cells, candidates, seeds, n_cells_expected=None):
    """T = mean over ALL cells of signed (p_S - p_ref). NON-GATING.

    Inclusive: T >= +0.01 trips (optimism), T <= -0.01 trips (pessimism).
    Tripping causes investigation/reporting ONLY. It can never make a gate
    pass or fail, never creates an INDETERMINATE category, and never blocks
    Stage 2 independently.
    """
    _require_cells(cells, "optimism tripwire")
    dps = _dps(cells, candidates, seeds)
    n = len(candidates) * len(seeds)
    if n_cells_expected is not None and n != n_cells_expected:
        fail(f"tripwire denominator must be exactly {n_cells_expected} cells, "
             f"got {n}; the denominator is a fixed constant, never a "
             "surviving-row count")
    T = _mean_exact(dps, n)
    tripped_opt = bool(T >= TRIPWIRE_T_EXACT)
    tripped_pes = bool(T <= -TRIPWIRE_T_EXACT)
    return {"name": "optimism_tripwire_0.01", "gating": False,
            "role": "NON_GATING_investigation_and_reporting_only",
            "statistic": "global_mean_signed_probability_difference",
            "value": float(T), "value_exact": str(T),
            "threshold": TRIPWIRE_T,
            "comparison": "inclusive_absolute_at_or_above",
            "denominator": n,
            "tripped": bool(tripped_opt or tripped_pes),
            "tripped_optimism": tripped_opt,
            "tripped_pessimism": tripped_pes,
            "direction": ("optimism" if tripped_opt else
                          "pessimism" if tripped_pes else "not_tripped"),
            "consequence": ("Report only. Must never make G1/G2dagger/G3 pass "
                            "or fail, never creates INDETERMINATE, never "
                            "blocks Stage 2 independently. If tripped, a "
                            "decomposition by candidate, seed and stratum is "
                            "mandatory; if tripped while the gate PASSES this "
                            "must be prominent in the summary and carried as "
                            "an explicit Stage-2 limitation.")}


def tripwire_decomposition(cells, candidates, seeds):
    """Mandatory decomposition when the tripwire trips. Non-gating."""
    by_candidate = {c: float(_mean_exact([cells[(c, int(s))]["dp_exact"]
                                          for s in seeds], len(seeds)))
                    for c in candidates}
    by_seed = {str(s): float(_mean_exact([cells[(c, int(s))]["dp_exact"]
                                          for c in candidates],
                                         len(candidates))) for s in seeds}
    by_stratum = {}
    for name, members in (("KNIFE", KNIFE_CANDIDATES), ("OFF", OFF_CANDIDATES)):
        members = [c for c in members if c in candidates]
        if members:
            vals = [cells[(c, int(s))]["dp_exact"]
                    for c in members for s in seeds]
            by_stratum[name] = float(_mean_exact(vals, len(vals)))
    return {"gating": False, "role": "NON_GATING_reporting_only",
            "mean_signed_by_candidate": by_candidate,
            "mean_signed_by_seed": by_seed,
            "mean_signed_by_stratum": by_stratum}


def compute_bootstrap(cells, candidates, seeds, n_resamples=BOOTSTRAP_RESAMPLES):
    """Seed-cluster bootstrap. DESCRIPTIVE ONLY.

    Resamples the master seeds (the only independent replication) with
    replacement. It NEVER determines PASS or FAIL, and there is NO
    INDETERMINATE category: if an interval straddles a threshold the gate
    verdict still stands exactly as computed by G1/G2dagger/G3, and the
    fragility is reported as a stated limitation.
    """
    rng = np.random.default_rng(BOOTSTRAP_RNG_SEED)
    seeds = list(seeds)
    g1s, g2s, off_rates = [], [], []
    for _ in range(int(n_resamples)):
        drawn = [seeds[i] for i in rng.integers(0, len(seeds), len(seeds))]
        # Resampling with replacement duplicates seed labels; build a cell
        # view with distinct synthetic labels so duplicates are kept.
        view, labels = {}, []
        for j, s in enumerate(drawn):
            for c in candidates:
                view[(c, j)] = cells[(c, int(s))]
            labels.append(j)
        # A relabelled view of ALREADY-validated cells: provenance is
        # inherited, so it is re-certified rather than laundered.
        view = ValidatedCells(view, _CELLS_TOKEN)
        g1s.append(compute_g1(view, candidates, labels)["value"])
        g2s.append(compute_g2dagger(view, candidates, labels)["value"])
        off = [c for c in OFF_CANDIDATES if c in candidates]
        if off:
            off_rates.append(flip_stats(view, off, labels)["gross_flip_rate"])
    lo_q, hi_q = (1.0 - BOOTSTRAP_CI) / 2.0, 1.0 - (1.0 - BOOTSTRAP_CI) / 2.0

    def interval(vals):
        if not vals:
            return None
        return {"lo": float(np.quantile(vals, lo_q)),
                "hi": float(np.quantile(vals, hi_q)),
                "median": float(np.median(vals))}

    return {"name": "seed_cluster_bootstrap", "gating": False,
            "role": "DESCRIPTIVE_ONLY",
            "resamples": int(n_resamples), "ci_level": BOOTSTRAP_CI,
            "rng_seed": BOOTSTRAP_RNG_SEED,
            "G1_interval": interval(g1s),
            "G2dagger_interval": interval(g2s),
            "OFF_flip_rate_interval": interval(off_rates),
            "note": ("Descriptive robustness only. Does NOT create an "
                     "INDETERMINATE category, does NOT override G1/G2dagger/G3, "
                     "cannot turn PASS into FAIL or FAIL into PASS, and cannot "
                     "block Stage 2 independently.")}


def compute_loco(cells, candidates, seeds):
    """Leave-one-candidate-out sensitivity. DESCRIPTIVE ONLY.

    Exposes how far G1 / max_c D_c / the OFF flip rate depend on any single
    candidate. Never gates.
    """
    out = {}
    for omitted in candidates:
        kept = [c for c in candidates if c != omitted]
        if not kept:
            continue
        entry = {
            "G1_value": compute_g1(cells, kept, seeds)["value"],
            "G2dagger_value": compute_g2dagger(cells, kept, seeds)["value"],
        }
        off = [c for c in OFF_CANDIDATES if c in kept]
        entry["OFF_flip_rate"] = (flip_stats(cells, off, seeds)["gross_flip_rate"]
                                  if off else None)
        entry["omitted_stratum"] = stratum_of(omitted)
        out[omitted] = entry
    return {"name": "leave_one_candidate_out", "gating": False,
            "role": "DESCRIPTIVE_ONLY", "per_omitted_candidate": out,
            "note": ("Descriptive robustness only. Does NOT create an "
                     "INDETERMINATE category, does NOT override G1/G2dagger/G3, "
                     "cannot turn PASS into FAIL or FAIL into PASS, and cannot "
                     "block Stage 2 independently.")}


STAGE1_RESULTS = HERE / "a_lite_stage1_results" / "raw_rows.csv"


def sqrt_rate_check(cells, candidates, seeds, S, S_ref):
    """Descriptive convergence-behaviour check (amendment section 7.3 item 15).

    For nested prefixes the paired difference of two Monte-Carlo means behaves
    like sqrt(1/S - 1/S_ref) up to a constant. Reporting observed mean|dp|
    against that scale shows whether the estimator is behaving like a
    converging estimator at all. Purely descriptive: it never gates.
    """
    absd = [abs(d) for d in _dps(cells, candidates, seeds)]
    observed = float(_mean_exact(absd, len(absd)))
    scale = math.sqrt(1.0 / S - 1.0 / S_ref) if S < S_ref else float("nan")
    return {"gating": False, "role": "DESCRIPTIVE_ONLY",
            "observed_mean_abs_dp": observed,
            "sqrt_scale_sqrt_1overS_minus_1overSref": scale,
            "implied_constant": (observed / scale
                                 if scale and np.isfinite(scale) and scale > 0
                                 else None),
            "note": ("The implied constant should be comparable across "
                     "comparisons if the estimator is converging at the "
                     "Monte-Carlo rate. Descriptive only; never gates.")}


def stage1_ratio_matched_reference(ratio=0.5):
    """Stage-1's mean|dp| for the ratio-matched pair (amendment 7.3 item 16).

    Stage-1's S100-vs-S200 has the same S/S_ref ratio (0.5) as Stage-1B's
    S500-vs-S1000, so the two are directly comparable as a scale-invariance
    diagnostic. Reads the EXISTING Stage-1 results read-only; it never writes
    or modifies them, and it never gates.
    """
    if not STAGE1_RESULTS.exists():
        return {"available": False,
                "reason": f"Stage-1 results not found at {STAGE1_RESULTS}"}
    try:
        with STAGE1_RESULTS.open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        by = {}
        for r in rows:
            by[(r["stage1_candidate_id"], int(r["master_seed"]),
                int(r["S"]))] = float(r["min_on_time_prob"])
        pairs = [(100, 200)] if ratio == 0.5 else []
        out = {}
        for S, S_ref in pairs:
            diffs = []
            for (c, s, sv) in by:
                if sv != S or (c, s, S_ref) not in by:
                    continue
                diffs.append(abs(exact_p(by[(c, s, S)], S)
                                 - exact_p(by[(c, s, S_ref)], S_ref)))
            if diffs:
                out[f"S{S}_vs_S{S_ref}"] = {
                    "ratio": S / S_ref, "n_cells": len(diffs),
                    "mean_abs_dp": float(_mean_exact(diffs, len(diffs)))}
        return {"available": True, "source": str(STAGE1_RESULTS),
                "stage1_reference": out}
    except Exception as exc:                                # noqa: BLE001
        return {"available": False, "reason": f"could not read Stage-1: {exc}"}


def ratio_matched_diagnostic(cells, candidates, seeds, S, S_ref):
    """Stage-1B vs the ratio-matched Stage-1 contrast. Descriptive only."""
    absd = [abs(d) for d in _dps(cells, candidates, seeds)]
    observed = float(_mean_exact(absd, len(absd)))
    ratio = S / S_ref
    entry = {"gating": False, "role": "DESCRIPTIVE_ONLY",
             "this_comparison": {"S": S, "S_ref": S_ref, "ratio": ratio,
                                 "mean_abs_dp": observed},
             "is_ratio_matched_to_stage1": bool(abs(ratio - 0.5) < 1e-12),
             "note": ("S500-vs-S1000 (ratio 0.5) is ratio-matched to Stage-1's "
                      "S100-vs-S200 (ratio 0.5); comparing them tests scale "
                      "invariance. Descriptive only; never gates.")}
    if entry["is_ratio_matched_to_stage1"]:
        entry["stage1_reference"] = stage1_ratio_matched_reference(0.5)
    return entry


def descriptive_diagnostics(cells, candidates, seeds):
    """All remaining mandatory NON-GATING diagnostics."""
    dps = _dps(cells, candidates, seeds)
    absd = [abs(d) for d in dps]
    absf = [float(d) for d in absd]
    knife = [c for c in KNIFE_CANDIDATES if c in candidates]
    off = [c for c in OFF_CANDIDATES if c in candidates]

    boundary_S = boundary_ref = 0
    binding_changes = mlate_changes = nonccp_changes = 0
    per_cell_boundary = {}
    for c in candidates:
        nb_s = nb_r = 0
        for s in seeds:
            cell = cells[(c, int(s))]
            if cell["p_S_exact"] == ALPHA_EXACT:
                boundary_S += 1
                nb_s += 1
            if cell["p_ref_exact"] == ALPHA_EXACT:
                boundary_ref += 1
                nb_r += 1
            if cell["binding_S"] != cell["binding_ref"]:
                binding_changes += 1
            if abs(cell["mlate_S"] - cell["mlate_ref"]) > TOL_MONOTONE:
                mlate_changes += 1
            if cell["nonccp_S"] != cell["nonccp_ref"]:
                nonccp_changes += 1
        per_cell_boundary[c] = {"at_S": nb_s, "at_ref": nb_r}

    # Two candidates can be structurally DISTINCT decisions and still produce
    # the identical p_hat series, if their differences never touch the binding
    # batch. That lowers the effective diversity of the panel for this
    # statistic below the nominal candidate count, so it is detected and
    # reported. It is NEVER a reason to drop, merge or reweight a candidate:
    # the frozen panel and its denominators stand.
    series = {c: tuple((cells[(c, int(s))]["p_S_exact"],
                        cells[(c, int(s))]["p_ref_exact"]) for s in seeds)
              for c in candidates}
    cl = list(candidates)
    identical_pairs = [[a, b] for i, a in enumerate(cl) for b in cl[i + 1:]
                       if series[a] == series[b]]

    return {
        "gating": False, "role": "DESCRIPTIVE_ONLY",
        "statistically_identical_candidate_pairs": identical_pairs,
        "n_distinct_probability_series": len(set(series.values())),
        "n_candidates": len(cl),
        "mean_abs_probability_difference": float(_mean_exact(absd, len(absd))),
        "median_abs_probability_difference": float(statistics.median(absf)),
        "max_abs_probability_difference": float(max(absd)),
        "mean_signed_probability_difference": float(_mean_exact(dps, len(dps))),
        "per_candidate_signed_D_c": {c: float(v) for c, v in
                                     compute_d_c(cells, candidates, seeds).items()},
        "per_candidate_mean_abs_A_c": {c: float(v) for c, v in
                                       compute_a_c(cells, candidates, seeds).items()},
        "per_candidate_full_table": per_candidate_full_table(
            cells, candidates, seeds),
        "per_seed_breakdown": per_seed_breakdown(cells, candidates, seeds),
        "all_pair_flips": flip_stats(cells, list(candidates), seeds),
        "KNIFE_flips": flip_stats(cells, knife, seeds) if knife else None,
        "OFF_flips": flip_stats(cells, off, seeds) if off else None,
        "exact_boundary_mass_at_alpha": {
            "at_S": boundary_S, "at_ref": boundary_ref,
            "per_candidate": per_cell_boundary},
        "binding_batch_set_changes": binding_changes,
        "max_lateness_changes": mlate_changes,
        "nonccp_hard_pass_changes": nonccp_changes,
    }


# ════════════════════════════════════════════════════════
# Comparison + ladder
# ════════════════════════════════════════════════════════

def stated_limitations(comp, scientific=True):
    """MANDATORY non-gating disclosures (amendment sections 5, 7.1, 7.3).

    The frozen methodology requires that a straddling bootstrap band, a
    verdict that hinges on one candidate, a concealed argmax tie and a panel
    whose effective diversity is below its nominal size are all REPORTED --
    while the verdict "still stands as computed by G1 AND G2dagger AND G3".
    Nothing here can create, overturn or qualify a gate result; every entry
    is a disclosure obligation, and an empty list is itself a finding.

    A stated limitation is a STAGE-2 DISCLOSURE OBLIGATION, so it can only
    arise from a run that produces a verdict. A non-scientific run produces
    none, and emitting "MANDATORY disclosure ... carry forward" wording there
    would be exactly the leak the sanity tripwire note already prevents.
    """
    if not scientific:
        return ["NON-SCIENTIFIC RUN -- DO NOT QUOTE. A stated limitation is a "
                "Stage-2 disclosure obligation and can only arise from a run "
                "that produces a verdict. This run produces none, so no "
                "limitation is asserted. Any fragility visible in its "
                "descriptive diagnostics is a plumbing observation only and "
                "must never be carried forward."]
    out = []
    g, ng = comp["gates"], comp["non_gating"]

    g2 = g["G2dagger"]
    if g2.get("argmax_is_tied"):
        out.append(
            f"max_c D_c = {g2['value_exact']} is attained by MORE THAN ONE "
            f"candidate ({', '.join(g2['argmax_candidates'])}). The single "
            "reported argmax is a deterministic first-in-frozen-order choice "
            "and must NOT be read as the uniquely worst candidate.")

    for a, b in (ng["diagnostics"].get(
            "statistically_identical_candidate_pairs") or []):
        out.append(
            f"Candidates {a} and {b} are STRUCTURALLY DISTINCT decisions that "
            "produce an identical p_hat series in every (seed, S) cell of "
            "this comparison. Effective panel diversity for the CCP "
            "worst-batch statistic is therefore below the nominal candidate "
            "count, and candidate-level replication claims must be qualified. "
            "The frozen panel is NOT reselected and no denominator is "
            "reduced on this basis.")

    bs = ng.get("bootstrap")
    if bs:
        for name, key, thr in (("G1", "G1_interval", THRESHOLD_G1),
                               ("max_c D_c", "G2dagger_interval",
                                THRESHOLD_G2DAGGER),
                               ("OFF-stratum flip rate", "OFF_flip_rate_interval",
                                THRESHOLD_G3)):
            iv = bs.get(key)
            if iv and iv["lo"] < thr < iv["hi"]:
                out.append(
                    f"The DESCRIPTIVE seed-cluster bootstrap band for {name} "
                    f"[{iv['lo']:.6f}, {iv['hi']:.6f}] STRADDLES its {thr} "
                    "threshold. Per amendment section 7.1 the gate verdict "
                    "stands exactly as computed; this fragility is a stated "
                    "limitation to carry forward and may never overturn a "
                    "verdict or create an INDETERMINATE category.")

    # A verdict "hinges" on a candidate only when omitting it would FLIP that
    # component across its threshold. Merely staying on the same side is not a
    # hinge: for an already-failing component, a LOCO refit that is still at
    # or above the threshold demonstrates nothing. Both directions count.
    lo = (ng.get("loco") or {}).get("per_omitted_candidate", {})
    for name, key, gate, thr in (
            ("G1", "G1_value", "G1", THRESHOLD_G1),
            ("max_c D_c", "G2dagger_value", "G2dagger", THRESHOLD_G2DAGGER),
            ("OFF-stratum flip rate", "OFF_flip_rate", "G3", THRESHOLD_G3)):
        full_passes = bool(g[gate]["passed"])
        flips = sorted(c for c, r in lo.items()
                       if r.get(key) is not None
                       and (r[key] >= thr) == full_passes)
        if flips:
            direction = ("from PASS to FAIL" if full_passes
                         else "from FAIL to PASS")
            out.append(
                f"Leave-one-candidate-out: omitting {', '.join(flips)} would "
                f"move {name} across its {thr} threshold {direction}, so this "
                "component depends materially on the full frozen panel. "
                "DESCRIPTIVE ONLY -- it never changes the computed verdict.")
    return out


def evaluate_comparison(idx, S, S_ref, candidates, seeds, gating,
                        n_seeds_expected=None, with_bootstrap=True,
                        scientific=None):
    """All statistics for ONE comparison.

    `gating=False` marks a comparison that may NEVER influence the ladder
    (S200-vs-S500). The gate verdict is computed ONLY from statistics whose
    own `gating` flag is True.

    `scientific` says whether this comparison belongs to a run that produces a
    real verdict. Only such a run can create a Stage-2 disclosure obligation;
    a sanity run gets an explicit do-not-quote note instead. It defaults to
    the same full-run signal that pins the frozen dimensions, and run_study
    passes it explicitly.
    """
    cells = build_cells(idx, S, S_ref, candidates, seeds)
    # A full run pins EVERY frozen dimension, not just the seed count: 100
    # cells for G1 and T, 10 seeds x 10 candidates for D_c/G2dagger, and 50
    # OFF-stratum cells for G3. Sanity mode deliberately passes None, and its
    # outputs are stamped scientifically uninterpretable.
    full = n_seeds_expected is not None
    if scientific is None:
        scientific = full
    n_cells_exp = EXPECTED_N_CELLS if full else None
    n_cands_exp = EXPECTED_N_CANDIDATES if full else None
    n_off_exp = len(OFF_CANDIDATES) * n_seeds_expected if full else None

    g1 = compute_g1(cells, candidates, seeds, n_cells_exp)
    g2 = compute_g2dagger(cells, candidates, seeds, n_seeds_expected,
                          n_cands_exp)
    off = [c for c in OFF_CANDIDATES if c in candidates]
    g3 = compute_g3(cells, seeds, off, n_off_exp)
    trip = compute_tripwire(cells, candidates, seeds, n_cells_exp)

    gates = [g1, g2, g3]
    for g in gates:
        if not g.get("gating"):
            fail(f"{g['name']} lost its gating flag; refusing to decide")
    passed = all(bool(g["passed"]) for g in gates)

    out = {
        "comparison": f"S{S}_vs_S{S_ref}",
        "S": S, "S_ref": S_ref,
        "is_gating_comparison": bool(gating),
        "gate_verdict": ("PASS" if passed else "FAIL") if gating else "NOT_A_GATE",
        "gates": {"G1": g1, "G2dagger": g2, "G3": g3},
        "gate_rule": "PASS iff G1 AND G2dagger AND G3 all pass (all STRICT <)",
        "indeterminate_category_exists": False,
        "non_gating": {
            "optimism_tripwire": trip,
            "tripwire_decomposition": (
                tripwire_decomposition(cells, candidates, seeds)
                if trip["tripped"] else None),
            "diagnostics": descriptive_diagnostics(cells, candidates, seeds),
            "sqrt_rate_check": sqrt_rate_check(cells, candidates, seeds,
                                               S, S_ref),
            "ratio_matched_scale_invariance": ratio_matched_diagnostic(
                cells, candidates, seeds, S, S_ref),
            "loco": compute_loco(cells, candidates, seeds),
            "bootstrap": (compute_bootstrap(cells, candidates, seeds)
                          if with_bootstrap else None),
        },
    }
    if gating:
        out["tripwire_tripped_while_gate_passed"] = bool(
            trip["tripped"] and passed)
        if out["tripwire_tripped_while_gate_passed"] and scientific:
            out["MANDATORY_STAGE2_LIMITATION"] = (
                f"The 0.01 optimism tripwire TRIPPED (T={trip['value']:.6f}, "
                f"{trip['direction']}) while the scientific gate PASSED. This "
                "is non-blocking but MUST be stated prominently in the summary "
                "and carried explicitly as a Stage-2 limitation: the accepted "
                "operating point carries a directional bias of this size and "
                "Stage-2 CCP labels inherit it.")
        elif out["tripwire_tripped_while_gate_passed"]:
            # A non-scientific run has no accepted operating point, no valid
            # ladder decision and no verdict to qualify, so it cannot create a
            # Stage-2 disclosure obligation. Emitting the scientific wording
            # here would be a route for an uninterpretable number to be quoted
            # as a real limitation.
            out["SANITY_TRIPWIRE_NOTE"] = (
                f"The 0.01 optimism tripwire fired (T={trip['value']:.6f}, "
                f"{trip['direction']}) in a NON-SCIENTIFIC run. This confirms "
                "only that the tripwire and its reporting path work. There is "
                "NO accepted operating point, NO valid ladder decision and NO "
                "scientific verdict in such a run, so this value MUST NOT be "
                "quoted and MUST NOT be carried forward as a Stage-2 "
                "limitation. Only a full run can create that obligation.")
    # Computed LAST, from the finished comparison, and purely additive: it
    # reads the gate results but can never alter them.
    out["stated_limitations"] = stated_limitations(out, scientific)
    out["stated_limitations_are_scientific"] = bool(scientific)
    return out


def decide_ladder(comparisons):
    """Frozen Stage-2 ladder. S200 first; S500 only if S200 FAILS.

    S200-vs-S500 is diagnostic only and can never enter this decision.
    There is NO INDETERMINATE category.
    """
    by_S = {}
    for comp in comparisons:
        if not comp["is_gating_comparison"]:
            continue
        if comp["S_ref"] != S_REF:
            fail(f"gating comparison against S_ref={comp['S_ref']}; every gate "
                 f"must use S_ref={S_REF}")
        by_S[comp["S"]] = comp
    for S in GATING_S:
        if S not in by_S:
            fail(f"missing gating comparison for S={S}")

    first, second = GATING_S[0], GATING_S[1]
    if by_S[first]["gate_verdict"] == "PASS":
        return {
            "selected_S": first, "stage2_may_proceed": True,
            "s500_gate_consulted": False,
            "rationale": (f"S{first} vs S{S_REF} passed G1 AND G2dagger AND G3. "
                          f"Per the frozen ladder S{second} is NOT evaluated as "
                          "a replacement gate decision; all its results remain "
                          "part of the fixed study and are reported."),
            "diagnostic_only_pair": f"S{DIAGNOSTIC_PAIR[0]}_vs_S{DIAGNOSTIC_PAIR[1]}",
        }
    if by_S[second]["gate_verdict"] == "PASS":
        return {
            "selected_S": second, "stage2_may_proceed": True,
            "s500_gate_consulted": True,
            "rationale": (f"S{first} vs S{S_REF} FAILED; S{second} vs S{S_REF} "
                          "passed the same three gates."),
            "diagnostic_only_pair": f"S{DIAGNOSTIC_PAIR[0]}_vs_S{DIAGNOSTIC_PAIR[1]}",
        }
    return {
        "selected_S": None, "stage2_may_proceed": False,
        "s500_gate_consulted": True,
        "rationale": (f"Both S{first} and S{second} FAILED against S{S_REF}. "
                      "Scenario-count adequacy is UNRESOLVED and Stage 2 must "
                      f"NOT begin. S{S_REF} is NOT declared sufficient merely "
                      "for being the largest tested."),
        "diagnostic_only_pair": f"S{DIAGNOSTIC_PAIR[0]}_vs_S{DIAGNOSTIC_PAIR[1]}",
    }


# ════════════════════════════════════════════════════════
# Outputs
# ════════════════════════════════════════════════════════

SANITY_WARNING = (
    "SANITY RUN -- PLUMBING/DATA-FLOW CHECK ONLY. Reduced candidates/seeds. "
    "Every gate value, D_c, flip rate, tripwire and bootstrap interval in this "
    "file is SCIENTIFICALLY UNINTERPRETABLE and must never be quoted, plotted "
    "or treated as a Stage-1B result.")
FULL_RUN_NOTE = (
    "Full Stage-1B study. Nested S values are positively correlated by "
    "construction; independent replication comes only from the 10 distinct "
    "master seeds, and the 100 cells are crossed-clustered (effective "
    "replication of order 10, not 100).")


def json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"unserialisable {type(o)!r}")


def write_outputs(rows, comparisons, ladder, nesting, manifest, out_dir):
    out_dir = FSPath(out_dir)
    if out_dir.exists() and any(out_dir.iterdir()):
        fail(f"output directory {out_dir} already exists and is non-empty. "
             "Refusing to overwrite previous results; move or remove them "
             "deliberately first.")
    out_dir.mkdir(parents=True, exist_ok=True)
    dump = lambda name, obj: (out_dir / name).write_text(          # noqa: E731
        json.dumps(obj, indent=2, default=json_default), encoding="utf-8")

    dump("raw_rows.json", rows)
    if rows:
        skip = ("batch_on_time_prob", "binding_batches")
        flat = [k for k in rows[0] if k not in skip]
        with (out_dir / "raw_rows.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=flat + ["binding_batches"])
            w.writeheader()
            for r in rows:
                row = {k: r[k] for k in flat}
                row["binding_batches"] = "|".join(map(str, r["binding_batches"]))
                w.writerow(row)
    dump("comparisons.json", comparisons)
    dump("ladder_decision.json", ladder)
    dump("nested_prefix_verification.json", nesting)
    dump("manifest.json", manifest)
    # The per-candidate table is a MANDATORY standalone artifact: its absence
    # invalidates the run, and it must never be truncated to a "worst few".
    per_cand = {}
    for comp in comparisons:
        d = comp["non_gating"]["diagnostics"]
        per_cand[comp["comparison"]] = {
            "full_table": d["per_candidate_full_table"],
            "per_seed_breakdown": d["per_seed_breakdown"],
            "per_candidate_signed_D_c": d["per_candidate_signed_D_c"],
            "per_candidate_mean_abs_A_c": d["per_candidate_mean_abs_A_c"],
            "max_c_D_c": comp["gates"]["G2dagger"]["value"],
            "argmax_candidate": comp["gates"]["G2dagger"]["argmax_candidate"],
            "argmax_stratum": comp["gates"]["G2dagger"]["argmax_stratum"],
            "strata": {c: stratum_of(c) for c in d["per_candidate_signed_D_c"]},
        }
    dump("per_candidate_diagnostics.json", per_cand)
    (out_dir / "STAGE1B_REPORT.md").write_text(
        render_report(comparisons, ladder, manifest), encoding="utf-8")
    print(f"[OUT] {out_dir}")


def _fmt(x, nd=6):
    return "n/a" if x is None else f"{float(x):+.{nd}f}"


def render_report(comparisons, ladder, manifest):
    """The MANDATORY human-readable report.

    Every per-candidate table is printed in FULL (all rows, never truncated,
    never summarised by its mean), and every verdict line quotes max_c D_c
    together with its argmax candidate id, as section 5 requires.
    """
    L = []
    add = L.append
    add(f"# A-lite Stage-1B Report ({manifest.get('mode', '?')})")
    add("")
    if manifest.get("sanity_run"):
        add(f"> **{SANITY_WARNING}**")
        add("")
    prov = manifest.get("frozen_methodology", {})
    add(f"- Frozen methodology SHA-256: `{prov.get('amendment_sha256')}`")
    add(f"- Methodology freeze commit: `{prov.get('freeze_commit')}` "
        f"(blob SHA-256 `{prov.get('freeze_commit_blob_sha256')}`)")
    val = manifest.get("grid_validation") or {}
    status = val.get("status", "NOT_RECORDED")
    if status == "VALIDATED":
        add(f"- Grid validation: **VALIDATED** — "
            f"{val.get('validated_cells')}/{val.get('expected_cells')} "
            "declared cells present exactly once and valid; "
            f"imputed {val.get('imputed_cells')}, "
            f"dropped {val.get('dropped_cells')}, "
            f"reduced denominators {val.get('reduced_denominators')}")
    else:
        # Never print success wording for a record that does not assert
        # success: an absent certificate must read as an absent certificate.
        add(f"- Grid validation: **{status}** — "
            f"{val.get('rule', 'no validation record was supplied')}. "
            "**This run must NOT be quoted or cited as validated, and no "
            "gate verdict in it may be relied upon.**")
    add(f"- Master seeds: {manifest.get('master_seeds')}")
    add(f"- S grid: {manifest.get('S_values')}   reference S_ref = "
        f"{manifest.get('S_ref')}")
    add(f"- alpha = {manifest.get('confidence_ontime_asserted')} "
        f"(inclusive: p_hat >= alpha PASSES)")
    add("- Gates: **G1 AND G2dagger AND G3**, all STRICT `<`. "
        "There is NO INDETERMINATE category.")
    add("- **S1000 is an empirical reference, not ground truth.** No SAA or "
        "theoretical convergence claim is made for it; a PASS means the "
        "estimate and the label do not move materially between S and S1000 "
        "along a fixed realisation.")
    add("- **Bootstrap, LOCO and the 0.01 tripwire are non-gating** "
        "(DESCRIPTIVE ONLY / reporting only). They never create a PASS, a "
        "FAIL or an INDETERMINATE category and never block Stage 2.")
    add("- All-pair and KNIFE gross flip rates are **non-gating** diagnostics; "
        "only the OFF-stratum rate gates, as G3.")
    add("")
    add("## Ladder decision")
    add("")
    add(f"- selected S: **{ladder.get('selected_S')}**")
    add(f"- Stage 2 may proceed: **{ladder.get('stage2_may_proceed')}**")
    add(f"- S500 gate consulted: {ladder.get('s500_gate_consulted')}")
    add(f"- {ladder.get('rationale', '')}")
    add("")

    for comp in comparisons:
        g = comp["gates"]
        add(f"## {comp['comparison']}"
            + ("" if comp["is_gating_comparison"] else "  (DIAGNOSTIC ONLY)"))
        add("")
        add(f"**Verdict: {comp['gate_verdict']}** — "
            f"G1 = {g['G1']['value']:.6f} ({'pass' if g['G1']['passed'] else 'FAIL'}), "
            f"max_c D_c = {g['G2dagger']['value']:+.6f} "
            f"[{g['G2dagger']['argmax_candidate']}, "
            f"{g['G2dagger']['argmax_stratum']}"
            + (f"; TIED with {', '.join(c for c in g['G2dagger']['argmax_candidates'] if c != g['G2dagger']['argmax_candidate'])}"
               if g['G2dagger'].get('argmax_is_tied') else "")
            + "] "
            f"({'pass' if g['G2dagger']['passed'] else 'FAIL'}), "
            f"G3 = {g['G3']['value']:.4f} "
            f"({g['G3']['flip_count']}/{g['G3']['denominator']}) "
            f"({'pass' if g['G3']['passed'] else 'FAIL'})")
        add("")
        if comp.get("MANDATORY_STAGE2_LIMITATION"):
            add(f"> **TRIPWIRE TRIPPED WHILE GATE PASSED.** "
                f"{comp['MANDATORY_STAGE2_LIMITATION']}")
            add("")
        elif comp.get("SANITY_TRIPWIRE_NOTE"):
            add(f"> **TRIPWIRE FIRED IN A NON-SCIENTIFIC RUN — DO NOT QUOTE.** "
                f"{comp['SANITY_TRIPWIRE_NOTE']}")
            add("")
        lims = comp.get("stated_limitations") or []
        sci = comp.get("stated_limitations_are_scientific", True)
        add("### Stated limitations (MANDATORY disclosure, non-gating)" if sci
            else "### Stated limitations — NOT APPLICABLE (non-scientific run)")
        add("")
        if lims and not sci:
            for limitation in lims:
                add(f"- {limitation}")
        elif lims:
            add("These are reporting obligations. Per amendment sections 7.1 "
                "and 7.3 the gate verdict above **stands exactly as "
                "computed**; nothing here overturns it or creates an "
                "INDETERMINATE category.")
            add("")
            for limitation in lims:
                add(f"- {limitation}")
        else:
            add("- None triggered for this comparison: no argmax tie, no "
                "statistically identical candidate pair, no bootstrap band "
                "straddling a threshold, and no leave-one-candidate-out "
                "refit reaching a threshold.")
        add("")
        d = comp["non_gating"]["diagnostics"]
        trip = comp["non_gating"]["optimism_tripwire"]
        add("### Per-candidate diagnostics (all rows, non-gating except as noted)")
        add("")
        add("| candidate | stratum | D_c | A_c | Dmax_c | Dmin_c | nflip_c | "
            "netflip_c | bound@S | bound@ref |")
        add("|---|---|---|---|---|---|---|---|---|---|")
        for cid, row in sorted(d["per_candidate_full_table"].items()):
            add(f"| {cid} | {row['stratum']} | {_fmt(row['D_c'])} | "
                f"{row['A_c']:.6f} | {_fmt(row['Dmax_c'])} | "
                f"{_fmt(row['Dmin_c'])} | {row['nflip_c']} | "
                f"{row['netflip_c']:+d} | {row['nboundary_c_at_S']} | "
                f"{row['nboundary_c_at_ref']} |")
        add("")
        add("### Panel diagnostics (non-gating)")
        add("")
        ap, kn, off = d["all_pair_flips"], d["KNIFE_flips"], d["OFF_flips"]
        add(f"- mean |dp| = {d['mean_abs_probability_difference']:.6f}, "
            f"median = {d['median_abs_probability_difference']:.6f}, "
            f"max = {d['max_abs_probability_difference']:.6f}")
        add(f"- mean signed dp (T) = {d['mean_signed_probability_difference']:+.6f}")
        add(f"- all-pair flips {ap['gross_flip_count']}/{ap['n_cells']} "
            f"({ap['gross_flip_rate']:.4f})")
        if kn:
            add(f"- KNIFE flips {kn['gross_flip_count']}/{kn['n_cells']} "
                f"({kn['gross_flip_rate']:.4f})  [non-gating]")
        if off:
            add(f"- OFF flips {off['gross_flip_count']}/{off['n_cells']} "
                f"({off['gross_flip_rate']:.4f})  [G3]")
            add(f"- false-feasible {off['false_feasible_count']}, "
                f"false-infeasible {off['false_infeasible_count']}, "
                f"net imbalance {off['signed_net_classification_imbalance']:+d}")
        add(f"- exact p_hat=alpha mass: {d['exact_boundary_mass_at_alpha']['at_S']} "
            f"at S, {d['exact_boundary_mass_at_alpha']['at_ref']} at ref")
        add(f"- binding-batch-set changes {d['binding_batch_set_changes']}, "
            f"max-lateness changes {d['max_lateness_changes']}, "
            f"non-CCP hard-pass changes {d['nonccp_hard_pass_changes']}")
        add(f"- 0.01 tripwire: T = {trip['value']:+.6f} -> "
            f"**{'TRIPPED (' + trip['direction'] + ')' if trip['tripped'] else 'not tripped'}** "
            "[NON-GATING: reporting/investigation only]")
        sq = comp["non_gating"]["sqrt_rate_check"]
        add(f"- sqrt-rate check: observed {sq['observed_mean_abs_dp']:.6f} vs "
            f"scale {sq['sqrt_scale_sqrt_1overS_minus_1overSref']:.6f} "
            f"(implied constant {_fmt(sq['implied_constant'])})")
        rm = comp["non_gating"]["ratio_matched_scale_invariance"]
        add(f"- ratio-matched to Stage-1: {rm['is_ratio_matched_to_stage1']}"
            + (f", Stage-1 reference "
               f"{rm.get('stage1_reference', {}).get('stage1_reference')}"
               if rm["is_ratio_matched_to_stage1"] else ""))
        bs = comp["non_gating"]["bootstrap"]
        if bs:
            add(f"- seed-cluster bootstrap (DESCRIPTIVE ONLY): "
                f"G1 {bs['G1_interval']}, G2dagger {bs['G2dagger_interval']}")
        add("")
        add("### Per-seed breakdown (all seeds, non-gating)")
        add("")
        add("| seed | mean signed | mean abs | max abs | flips | "
            "false-feasible | false-infeasible | net imbalance |")
        add("|---|---|---|---|---|---|---|---|")
        for sid, r in sorted(d["per_seed_breakdown"].items()):
            add(f"| {sid} | {_fmt(r['mean_signed'])} | {r['mean_abs']:.6f} | "
                f"{r['max_abs']:.6f} | {r['gross_flip_count']} | "
                f"{r['false_feasible_count']} | {r['false_infeasible_count']} | "
                f"{r['signed_net_classification_imbalance']:+d} |")
        add("")
        lo = comp["non_gating"]["loco"]
        add("### Leave-one-candidate-out (LOCO) — DESCRIPTIVE ONLY, never gates")
        add("")
        add("| omitted | stratum | G1 | max_c D_c | OFF flip rate |")
        add("|---|---|---|---|---|")
        for cid, r in sorted(lo["per_omitted_candidate"].items()):
            add(f"| {cid} | {r['omitted_stratum']} | {r['G1_value']:.6f} | "
                f"{_fmt(r['G2dagger_value'])} | "
                + (f"{r['OFF_flip_rate']:.4f} |" if r["OFF_flip_rate"] is not None
                   else "n/a |"))
        add("")
    return "\n".join(L) + "\n"


def base_manifest(doc, mode, seeds, candidates, s_values, provenance,
                  validation=None):
    return {
        "stage": "A-lite Stage 1B",
        # An EXPLICIT successful-validation record. A run that cannot show
        # this record cannot claim its grid was validated; absence is itself
        # the reportable status.
        "grid_validation": validation or {
            "status": "NOT_RECORDED",
            "rule": ("no ValidatedGrid validation record was supplied; the "
                     "run must not be quoted as validated"),
        },
        "mode": mode,
        "sanity_run": (mode == "sanity-run"),
        "is_full_study": (mode == "full-run"),
        "scientifically_interpretable": (mode == "full-run"),
        "warning": SANITY_WARNING if mode == "sanity-run" else FULL_RUN_NOTE,
        "scientific_question": (
            "Is a scenario count S in {200,500} ADEQUATE relative to the larger "
            "empirical reference S_ref=1000, for the 10 FIXED Stage-1 "
            "candidates, under the frozen gates G1, G2dagger and G3?"),
        "explicitly_not": [
            "an optimisation experiment (no search is performed)",
            "an SAA convergence proof; S=1000 is an empirical reference and is "
            "explicitly NOT ground truth",
            "evidence about NSGA-II search-time selection bias (Stage 2)",
        ],
        "frozen_methodology": provenance,
        "master_seeds": list(seeds),
        "seed_provenance": (
            "Ten NEW seeds declared in source before any Stage-1B result "
            "existed; a contiguous block so the choice is transparently not "
            "cherry-picked. Disjoint from every previously used seed including "
            f"all Stage-1 seeds ({sorted(FORBIDDEN_SEEDS)})."),
        "S_values": list(s_values),
        "master_S": MASTER_S,
        "S_ref": S_REF,
        "risk_metric": RISK_METRIC,
        "confidence_ontime_alpha": float(model.CONFIDENCE_ONTIME),
        "confidence_ontime_asserted": EXPECTED_CONFIDENCE_ONTIME,
        "tie_rule": ("INCLUSIVE: p_hat >= 0.90 is a CCP probability PASS, so "
                     "p_hat == 0.90 PASSES. Implemented as the production "
                     f"predicate chance_vio_total <= {TOL_HARD}."),
        "gates": {
            "G1": {"statistic": "mean |p_S - p_1000| over all 100 cells",
                   "threshold": THRESHOLD_G1, "strict": True, "gating": True},
            "G2dagger": {"statistic": "max_c D_c(S,1000)",
                         "D_c": "(1/10) sum_s [p_{c,s,S} - p_{c,s,1000}], signed",
                         "threshold": THRESHOLD_G2DAGGER, "strict": True,
                         "gating": True},
            "G3": {"statistic": "OFF-stratum gross classification flip rate",
                   "threshold": THRESHOLD_G3, "strict": True, "gating": True},
            "rule": "PASS iff G1 AND G2dagger AND G3",
        },
        "non_gating": {
            "optimism_tripwire_0.01": "reporting/investigation only",
            "seed_cluster_bootstrap": "DESCRIPTIVE ONLY",
            "leave_one_candidate_out": "DESCRIPTIVE ONLY",
            "all_pair_flips": "diagnostic only",
            "KNIFE_flips": "diagnostic only",
            "S200_vs_S500": "diagnostic only; never enters the ladder",
            "objectives": ("cost/emission/time/penalty are DESCRIPTIVE ONLY and "
                           "never influence candidate selection or any gate"),
        },
        "indeterminate_category_exists": False,
        "strata": {"KNIFE": list(KNIFE_CANDIDATES), "OFF": list(OFF_CANDIDATES),
                   "provenance": ("pre-registered from Stage-1 pooled estimates, "
                                  "out-of-sample w.r.t. Stage-1B's new seeds; "
                                  "membership is FROZEN and candidates are "
                                  "never relabelled after results")},
        "nesting": (
            f"For each seed exactly ONE master ScenarioSet of size {MASTER_S} is "
            "built; S=200 and S=500 are literal array prefixes of it. "
            "Independent build_scenario_set(seed, S) calls are never made, "
            "because they do NOT form prefixes."),
        "correlation_caveat": (
            "S200/S500/S1000 are nested and therefore positively correlated by "
            "construction. They are NOT independent samples. Independent "
            "replication comes only from the distinct master seeds; the 100 "
            "cells are crossed-clustered (effective replication of order 10)."),
        "candidate_provenance": {
            "checkpoint": str(s1.CHECKPOINT),
            "candidates_canonical_sha256": s1.EXPECTED_CANDIDATES_SHA256,
            "candidate_ids": list(candidates),
            "panel_origin": "frozen Stage-1 panel; NOT reselected for Stage 1B",
        },
        "border_event_integrity": {
            "expected_events": s1.EXPECTED_BORDER_EVENTS,
            "expected_arcs": s1.EXPECTED_BORDER_ARCS,
        },
        "expected_row_count": len(candidates) * len(seeds) * len(s_values),
    }


# ════════════════════════════════════════════════════════
# Study driver
# ════════════════════════════════════════════════════════

def stratified_subset(cands, k):
    """Pick a k-candidate sanity subset that spans BOTH frozen strata.

    G3 is defined on the OFF stratum, so a subset drawn in file order (which
    starts S1-01, S1-02 -- both KNIFE) would leave G3 with an empty stratum
    and no gate could be computed. Sanity mode exists to exercise the whole
    pipeline, so it must contain at least one OFF candidate.
    """
    if k < 2:
        fail("a sanity subset must contain at least 2 candidates so that both "
             "the KNIFE and OFF strata are represented")
    by_id = {c["stage1_candidate_id"]: c for c in cands}
    knife = [c for c in KNIFE_CANDIDATES if c in by_id]
    off = [c for c in OFF_CANDIDATES if c in by_id]
    picked, i = [], 0
    while len(picked) < k and (i < len(knife) or i < len(off)):
        if i < len(knife) and len(picked) < k:
            picked.append(knife[i])
        if i < len(off) and len(picked) < k:
            picked.append(off[i])
        i += 1
    if len(picked) < k:
        fail(f"cannot build a stratified subset of {k} candidates")
    if not [c for c in picked if c in OFF_CANDIDATES]:
        fail("stratified subset contains no OFF candidate; G3 would be undefined")
    return [by_id[c] for c in picked]


def run_study(env, doc, cands, seeds, s_values, mode, out_dir):
    provenance = verify_frozen_methodology()
    verify_strata()
    if mode == "full-run":
        # Pure configuration checks first, cheapest and most specific first, so
        # a misconfigured full run reports the actual violation.
        if list(s_values) != S_VALUES:
            fail(f"full run requires the frozen S grid {S_VALUES}, got {s_values}")
        verify_seed_list(seeds)
        if len(cands) != EXPECTED_N_CANDIDATES:
            fail(f"full run requires exactly {EXPECTED_N_CANDIDATES} candidates, "
                 f"got {len(cands)}")

    batches, arcs, tt_dict = env["batches"], env["arcs"], env["tt_dict"]
    stats = {"lib_hit": 0, "rebuilt": 0}
    inds, cand_ids = {}, []
    for e in cands:
        cid = e["stage1_candidate_id"]
        inds[cid] = s1.build_candidate_individual(
            e, batches, env["path_lib"], tt_dict, env["arc_lookup"], stats)
        cand_ids.append(cid)
    fingerprints = {cid: s1.individual_decision_fingerprint(i)
                    for cid, i in inds.items()}
    print(f"[LOAD] {len(inds)} candidates; path_lib reuse={stats['lib_hit']} "
          f"reconstructed={stats['rebuilt']}")

    rows, nesting = [], {}
    for seed in seeds:
        master = build_master(env, seed)
        subsets, report = nested_subsets(master, seed, s_values)
        nesting[str(seed)] = report
        for S in sorted(s_values):
            scen = subsets[S]
            s1.install_scenario_set(scen)          # clears the path cache first
            for cid in cand_ids:
                row = s1.evaluate_candidate(
                    env, inds[cid], batches, arcs, tt_dict, scen, cid, seed, S)
                row["candidate_id"] = row.pop("stage1_candidate_id")
                row["candidate_fingerprint"] = fingerprints[cid]
                row["stratum"] = stratum_of(cid)
                rows.append(row)
            s1.assert_cache_confined(scen)
        for cid in cand_ids:
            check_maxlate_monotonicity(
                [r for r in rows
                 if r["candidate_id"] == cid and r["master_seed"] == seed],
                cid, seed, s_values)
        print(f"[SEED {seed}] done ({len(s_values)} S values x {len(inds)} candidates)")

    for cid, before in fingerprints.items():
        if s1.individual_decision_fingerprint(inds[cid]) != before:
            fail(f"{cid}: candidate decision structure was mutated during "
                 "evaluation; results cannot be trusted")

    expected = len(seeds) * len(s_values) * len(inds)
    if len(rows) != expected:
        fail(f"expected {expected} rows, produced {len(rows)}")
    if mode == "full-run" and len(rows) != EXPECTED_N_ROWS:
        fail(f"full run must produce exactly {EXPECTED_N_ROWS} rows")

    idx = require_complete_grid(rows, cand_ids, seeds, s_values,
                                out_dir=out_dir)
    n_exp = EXPECTED_N_SEEDS if mode == "full-run" else None
    # Only a full run produces a verdict, so only a full run can create a
    # Stage-2 disclosure obligation. Passed explicitly rather than inferred.
    scientific = (mode == "full-run")

    comparisons = []
    for S in sorted(s_values):
        if S == S_REF:
            continue
        comparisons.append(evaluate_comparison(
            idx, S, S_REF, cand_ids, seeds, gating=(S in GATING_S),
            n_seeds_expected=n_exp, scientific=scientific))
    a, b = DIAGNOSTIC_PAIR
    if a in s_values and b in s_values:
        comparisons.append(evaluate_comparison(
            idx, a, b, cand_ids, seeds, gating=False, n_seeds_expected=n_exp,
            scientific=scientific))

    ladder = (decide_ladder(comparisons) if mode == "full-run" else {
        "selected_S": None, "stage2_may_proceed": False,
        "rationale": "SANITY RUN -- no ladder decision is computed or valid."})

    manifest = base_manifest(doc, mode, seeds, cand_ids, s_values, provenance,
                             validation=idx.validation_record())
    manifest["n_rows"] = len(rows)
    write_outputs(rows, comparisons, ladder, nesting, manifest, out_dir)
    return rows, comparisons, ladder


# ════════════════════════════════════════════════════════
# Self-test: ZERO candidate evaluations
# ════════════════════════════════════════════════════════

def _synthetic_cells(values_by_candidate, seeds, S=1000, S_ref=1000):
    """Build cells from per-(candidate,seed) dp values, for testing ONLY.

    The dp values are realised on the ATTAINABLE empirical grid: p_ref is set
    to alpha = k/S_ref and p_S to (alpha + dp), and both are required to be
    exact multiples of 1/S and 1/S_ref respectively. A dp that cannot occur
    for real empirical proportions is rejected rather than silently faked, so
    synthetic gate tests exercise exactly the arithmetic the real path uses.
    """
    cells = {}
    for c, per_seed in values_by_candidate.items():
        for s, spec in zip(seeds, per_seed):
            if isinstance(spec, tuple):
                dp, pass_S, pass_ref = spec
            else:
                dp, pass_S, pass_ref = spec, True, True
            dp_f = Fraction(dp).limit_denominator(10 ** 9) \
                if not isinstance(dp, Fraction) else dp
            ex_r = ALPHA_EXACT
            ex_s = ex_r + dp_f
            for name, val, size in (("p_ref", ex_r, S_ref), ("p_S", ex_s, S)):
                scaled = val * size
                if scaled.denominator != 1:
                    fail(f"synthetic {name}={val} is not on the 1/{size} "
                         "empirical grid; synthetic cells must use attainable "
                         "empirical proportions")
                if not 0 <= val <= 1:
                    fail(f"synthetic {name}={val} outside [0,1]")
            cells[(c, int(s))] = {
                "p_S": float(ex_s), "p_ref": float(ex_r),
                "p_S_exact": ex_s, "p_ref_exact": ex_r,
                "dp": float(ex_s - ex_r), "dp_exact": ex_s - ex_r,
                "pass_S": pass_S, "pass_ref": pass_ref,
                "binding_S": "1", "binding_ref": "1",
                "mlate_S": 0.0, "mlate_ref": 0.0,
                "nonccp_S": True, "nonccp_ref": True,
            }
    return ValidatedCells(cells, _CELLS_TOKEN)


def self_test_gate_arithmetic():
    """Synthetic verification of the frozen gate arithmetic and semantics."""
    seeds = list(range(10))
    cands = list(s1.EXPECTED_CANDIDATE_IDS)

    # -- D_c denominator, sign convention and G2dagger maximum --
    spec = {c: [0.0] * 10 for c in cands}
    spec["S1-04"] = [0.03] * 10          # one optimistic OFF candidate
    spec["S1-01"] = [-0.05] * 10         # strong pessimism must NOT gate
    cells = _synthetic_cells(spec, seeds)
    d_c = compute_d_c(cells, cands, seeds, EXPECTED_N_SEEDS)
    assert abs(d_c["S1-04"] - 0.03) < 1e-12, "D_c signed mean wrong"
    assert abs(d_c["S1-01"] + 0.05) < 1e-12, "D_c must keep negative sign"
    g2 = compute_g2dagger(cells, cands, seeds, EXPECTED_N_SEEDS)
    assert g2["argmax_candidate"] == "S1-04", "G2dagger argmax wrong"
    assert not g2["passed"], "G2dagger must FAIL at 0.03"
    assert g2["seed_denominator"] == 10, "D_c denominator must be 10"

    # A single optimistic candidate must NOT be hidden by the global mean.
    g1 = compute_g1(cells, cands, seeds)
    assert g1["passed"], "G1 should pass here (dilution) -- G2dagger is the catch"

    # -- strict threshold semantics: exactly at the threshold FAILS --
    spec = {c: [0.0] * 10 for c in cands}
    spec["S1-04"] = [THRESHOLD_G2DAGGER] * 10
    cells = _synthetic_cells(spec, seeds)
    assert not compute_g2dagger(cells, cands, seeds)["passed"], \
        "G2dagger must FAIL at exactly 0.02"
    # 19/1000 is the largest ATTAINABLE empirical value below 0.02 on the
    # S=1000 grid. An arbitrary 0.02-1e-9 perturbation is not a possible
    # empirical difference and is rejected by _synthetic_cells.
    spec["S1-04"] = [Fraction(19, 1000)] * 10
    cells = _synthetic_cells(spec, seeds)
    assert compute_g2dagger(cells, cands, seeds)["passed"], \
        "G2dagger must PASS at the attainable 19/1000 just below 0.02"

    spec = {c: [(THRESHOLD_G1, True, True)] * 10 for c in cands}
    cells = _synthetic_cells(spec, seeds)
    assert not compute_g1(cells, cands, seeds)["passed"], \
        "G1 must FAIL at exactly 0.02"

    # -- G3: 4/50 OFF flips PASS, 5/50 FAIL; KNIFE flips must NOT gate --
    def off_flip_spec(n_off_flips, n_knife_flips=0):
        spec, made = {}, {"off": 0, "knife": 0}
        for c in cands:
            per = []
            for _ in seeds:
                flip = False
                if c in OFF_CANDIDATES and made["off"] < n_off_flips:
                    flip, made["off"] = True, made["off"] + 1
                elif c in KNIFE_CANDIDATES and made["knife"] < n_knife_flips:
                    flip, made["knife"] = True, made["knife"] + 1
                per.append((0.0, True, not flip))
            spec[c] = per
        return _synthetic_cells(spec, seeds)

    g3 = compute_g3(off_flip_spec(4), seeds)
    assert g3["denominator"] == 50 and g3["flip_count"] == 4 and g3["passed"], \
        "G3 must PASS at 4/50"
    g3 = compute_g3(off_flip_spec(5), seeds)
    assert g3["flip_count"] == 5 and not g3["passed"], "G3 must FAIL at 5/50"
    cells = off_flip_spec(0, n_knife_flips=50)
    assert compute_g3(cells, seeds)["passed"], \
        "KNIFE flips must NEVER gate G3"
    allp = flip_stats(cells, cands, seeds)
    assert allp["gross_flip_rate"] == 0.5, "all-pair flip rate wrong"

    # -- false-feasible / false-infeasible / net imbalance --
    spec = {c: [(0.0, True, True)] * 10 for c in cands}
    spec["S1-07"] = [(0.0, True, False)] * 10     # accepted at S, rejected at ref
    spec["S1-08"] = [(0.0, False, True)] * 3 + [(0.0, True, True)] * 7
    cells = _synthetic_cells(spec, seeds)
    st = flip_stats(cells, cands, seeds)
    assert st["false_feasible_count"] == 10, "false-feasible count wrong"
    assert st["false_infeasible_count"] == 3, "false-infeasible count wrong"
    assert st["signed_net_classification_imbalance"] == 7, "net imbalance wrong"

    # -- tie rule: p_hat == 0.90 is a PASS --
    assert ccp_pass(0.0), "zero shortfall must be a CCP PASS (inclusive rule)"
    assert ccp_pass(TOL_HARD), "shortfall at the tolerance must still PASS"
    assert not ccp_pass(1e-3), "a real shortfall must FAIL"
    row = {"min_on_time_prob": 0.90, "chance_vio_total": 0.0,
           "chance_constraint_pass": True}
    validate_cell(row, ("synthetic", 0, 200))
    bad = dict(row, chance_constraint_pass=False)
    try:
        validate_cell(bad, ("synthetic", 0, 200))
    except Stage1BIntegrityError:
        pass
    else:
        fail("tie p_hat=0.90 misclassified as FAIL was not caught")

    # -- tripwire is NON-GATING --
    spec = {c: [0.015] * 10 for c in cands}       # T = 0.015 -> tripped
    cells = _synthetic_cells(spec, seeds)
    trip = compute_tripwire(cells, cands, seeds)
    assert trip["tripped"] and trip["gating"] is False, "tripwire must not gate"
    assert compute_g1(cells, cands, seeds)["passed"], "G1 unaffected by tripwire"
    assert compute_g2dagger(cells, cands, seeds)["passed"], \
        "G2dagger unaffected by tripwire"
    spec = {c: [0.005] * 10 for c in cands}
    assert not compute_tripwire(_synthetic_cells(spec, seeds), cands,
                                seeds)["tripped"], "tripwire false positive"
    spec = {c: [-0.02] * 10 for c in cands}
    t2 = compute_tripwire(_synthetic_cells(spec, seeds), cands, seeds)
    assert t2["tripped_pessimism"] and not t2["tripped_optimism"], \
        "mirror pessimism tripwire wrong"

    # -- bootstrap and LOCO are descriptive only --
    spec = {c: [0.001 * (i + 1)] * 10 for i, c in enumerate(cands)}
    cells = _synthetic_cells(spec, seeds)
    bs = compute_bootstrap(cells, cands, seeds, n_resamples=50)
    lo = compute_loco(cells, cands, seeds)
    for obj in (bs, lo):
        assert obj["gating"] is False and obj["role"] == "DESCRIPTIVE_ONLY", \
            f"{obj['name']} must be descriptive only"
        assert "passed" not in obj and "verdict" not in obj, \
            f"{obj['name']} must not carry a verdict"
    return True


def self_test_ladder_arithmetic():
    """The frozen ladder, including that S200-vs-S500 can never gate."""
    def comp(S, S_ref, verdict, gating=True):
        return {"comparison": f"S{S}_vs_S{S_ref}", "S": S, "S_ref": S_ref,
                "is_gating_comparison": gating, "gate_verdict": verdict}

    d = decide_ladder([comp(200, 1000, "PASS"), comp(500, 1000, "FAIL")])
    assert d["selected_S"] == 200 and d["stage2_may_proceed"], "S200 pass"
    assert d["s500_gate_consulted"] is False, "S500 must not be consulted"

    d = decide_ladder([comp(200, 1000, "FAIL"), comp(500, 1000, "PASS")])
    assert d["selected_S"] == 500 and d["stage2_may_proceed"], "S500 fallback"

    d = decide_ladder([comp(200, 1000, "FAIL"), comp(500, 1000, "FAIL")])
    assert d["selected_S"] is None and not d["stage2_may_proceed"], \
        "both fail -> Stage 2 must not start"

    # A passing diagnostic S200-vs-S500 must never rescue a failed ladder.
    d = decide_ladder([comp(200, 1000, "FAIL"), comp(500, 1000, "FAIL"),
                       comp(200, 500, "PASS", gating=False)])
    assert d["selected_S"] is None, "S200-vs-S500 must never gate"
    return True


def self_test(env, doc, cands, skip_scenarios=False):
    """Integrity-only. Performs ZERO candidate evaluations."""
    print("=== STAGE-1B SELF-TEST (no candidate evaluation) ===")
    prov = verify_frozen_methodology()
    print(f"  PASS  frozen methodology SHA-256 {prov['amendment_sha256'][:16]}... "
          f"matches the freeze commit {METHODOLOGY_FREEZE_COMMIT[:9]}")

    stats = {"lib_hit": 0, "rebuilt": 0}
    for e in cands:
        s1.build_candidate_individual(e, env["batches"], env["path_lib"],
                                      env["tt_dict"], env["arc_lookup"], stats)
    print(f"  PASS  {len(cands)} candidates reconstruct exactly from the frozen "
          f"checkpoint (path_lib reuse={stats['lib_hit']} rebuilt={stats['rebuilt']})")

    verify_seed_list(MASTER_SEEDS)
    print(f"  PASS  {len(MASTER_SEEDS)} new distinct seeds {MASTER_SEEDS[0]}.."
          f"{MASTER_SEEDS[-1]}, disjoint from {sorted(FORBIDDEN_SEEDS)}")

    verify_strata()
    print(f"  PASS  frozen strata KNIFE={list(KNIFE_CANDIDATES)} "
          f"OFF={list(OFF_CANDIDATES)}")

    if float(model.CONFIDENCE_ONTIME) != EXPECTED_CONFIDENCE_ONTIME:
        fail(f"CONFIDENCE_ONTIME is {model.CONFIDENCE_ONTIME!r}, expected "
             f"{EXPECTED_CONFIDENCE_ONTIME!r}")
    print(f"  PASS  alpha = {EXPECTED_CONFIDENCE_ONTIME} asserted against production")

    self_test_gate_arithmetic()
    print("  PASS  gate arithmetic: D_c sign/denominator, G2dagger maximum, "
          "strict thresholds (exactly 0.02 FAILS), G3 4/50 vs 5/50")
    print("  PASS  tie semantics: p_hat == 0.90 classified as CCP PASS")
    print("  PASS  non-gating semantics: tripwire, bootstrap, LOCO, KNIFE and "
          "all-pair flips never gate")
    self_test_ladder_arithmetic()
    print("  PASS  ladder arithmetic: S200 -> S500 -> stop; S200-vs-S500 never gates")

    if skip_scenarios:
        print("  SKIP  scenario construction (--skip-scenarios)")
    else:
        for seed in MASTER_SEEDS:
            master = build_master(env, seed)
            subsets, _ = nested_subsets(master, seed)
            for S in S_VALUES:
                s1.install_scenario_set(subsets[S])
        print(f"  PASS  border-event config {s1.EXPECTED_BORDER_EVENTS}/"
              f"{s1.EXPECTED_BORDER_ARCS} on all {len(MASTER_SEEDS)} seeds")
        print(f"  PASS  one S{MASTER_S} master per seed; S200 and S500 verified "
              "as literal prefixes (all pairwise nestings, both scenario-indexed "
              "fields)")
        print("  PASS  cache-safe install works for every subset")

    if OUT_DIR.exists() and any(OUT_DIR.iterdir()):
        fail(f"the reserved full-study output directory {OUT_DIR} is already "
             "non-empty; a full run would refuse to overwrite it")
    print(f"  PASS  reserved output directory {OUT_DIR.name} is absent or empty")
    print("\nSELF-TEST PASSED -- 0 candidate evaluations performed")
    return True


def main():
    ap = argparse.ArgumentParser(description="A-lite Stage 1B")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--self-test", action="store_true")
    g.add_argument("--run", action="store_true")
    g.add_argument("--sanity-run", action="store_true")
    ap.add_argument("--i-have-approval", action="store_true")
    ap.add_argument("--skip-scenarios", action="store_true",
                    help="self-test only: skip ScenarioSet construction")
    ap.add_argument("--sanity-seeds", type=int, default=1)
    ap.add_argument("--sanity-candidates", type=int, default=2)
    ap.add_argument("--sanity-S", type=int, nargs="+", default=None,
                    help="sanity only: reduced S grid, must include the master")
    a = ap.parse_args()

    # Policy gates are checked BEFORE any expensive work, so an unapproved run
    # fails immediately rather than after loading the network.
    if a.run and not a.i_have_approval:
        fail("--run executes the full Stage-1B study (300 evaluations) and "
             "requires explicit sign-off: pass --i-have-approval.")
    if a.sanity_run and not a.i_have_approval:
        fail("--sanity-run executes candidate evaluations and requires "
             "explicit sign-off: pass --i-have-approval.")

    verify_frozen_methodology()
    env = s1.prepare_environment()
    doc, cands = s1.load_frozen_checkpoint()
    print(f"[PROV] candidate-array SHA-256 verified "
          f"{s1.EXPECTED_CANDIDATES_SHA256[:16]}...  ids S1-01..S1-10")

    if a.self_test:
        self_test(env, doc, cands, skip_scenarios=a.skip_scenarios)
        return
    if a.sanity_run:
        if not 1 <= a.sanity_seeds <= len(MASTER_SEEDS):
            fail(f"--sanity-seeds must be in 1..{len(MASTER_SEEDS)}")
        if not 1 <= a.sanity_candidates <= len(cands):
            fail(f"--sanity-candidates must be in 1..{len(cands)}")
        s_values = a.sanity_S or S_VALUES
        if MASTER_S not in s_values:
            fail(f"the sanity S grid must include the master size {MASTER_S}")
        run_study(env, doc, stratified_subset(cands, a.sanity_candidates),
                  MASTER_SEEDS[:a.sanity_seeds], sorted(s_values),
                  "sanity-run", SANITY_OUT_DIR)
        return
    run_study(env, doc, cands, MASTER_SEEDS, S_VALUES, "full-run", OUT_DIR)


if __name__ == "__main__":
    main()
