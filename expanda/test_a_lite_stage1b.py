#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Fail-closed tests for the A-lite Stage-1B driver.

Every test asserts that a violation of the FROZEN methodology raises
Stage1BIntegrityError rather than being silently tolerated, or that a frozen
rule computes exactly what the amendment specifies.

Two suites:
  * FAST  -- pure gate/ladder/provenance logic; no network, no ScenarioSet.
  * SLOW  -- integration checks that need the real environment (prefix
             construction, border-event tripwire, cache safety). Skipped
             unless STAGE1B_INTEGRATION=1, so the fast suite stays runnable.

Run:  python test_a_lite_stage1b.py
      STAGE1B_INTEGRATION=1 python test_a_lite_stage1b.py
"""
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path as FSPath

import numpy as np
from fractions import Fraction

HERE = FSPath(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import a_lite_stage1b_sensitivity as s1b   # noqa: E402
import a_lite_stage1_sensitivity as s1     # noqa: E402
import scenario_count_sensitivity as scs   # noqa: E402
import baseline_uncertainty as model       # noqa: E402

IntegrityError = s1b.Stage1BIntegrityError
SEEDS = list(range(10))
CANDS = list(s1.EXPECTED_CANDIDATE_IDS)
INTEGRATION = os.environ.get("STAGE1B_INTEGRATION") == "1"


def cells(spec, seeds=SEEDS):
    return s1b._synthetic_cells(spec, seeds)


def flat(value, cands=CANDS, n=10):
    return {c: [value] * n for c in cands}


# ══════════════════════════════════════════════════════════
# 1. Frozen-methodology and candidate provenance
# ══════════════════════════════════════════════════════════

class TestProvenance(unittest.TestCase):

    def test_amendment_hash_matches_frozen_value(self):
        self.assertEqual(s1b.sha256_file(s1b.AMENDMENT),
                         s1b.EXPECTED_AMENDMENT_SHA256)
        prov = s1b.verify_frozen_methodology()
        self.assertTrue(prov["verified"])
        self.assertEqual(prov["freeze_commit_blob_sha256"],
                         s1b.EXPECTED_AMENDMENT_SHA256)

    def test_amendment_hash_mismatch_fails_closed(self):
        original = s1b.EXPECTED_AMENDMENT_SHA256
        s1b.EXPECTED_AMENDMENT_SHA256 = "0" * 64
        try:
            with self.assertRaises(IntegrityError) as ctx:
                s1b.verify_frozen_methodology()
            self.assertIn("FROZEN METHODOLOGY HASH MISMATCH", str(ctx.exception))
        finally:
            s1b.EXPECTED_AMENDMENT_SHA256 = original

    def test_missing_amendment_fails_closed(self):
        original = s1b.AMENDMENT
        s1b.AMENDMENT = HERE / "__no_such_amendment__.md"
        try:
            with self.assertRaises(IntegrityError):
                s1b.verify_frozen_methodology()
        finally:
            s1b.AMENDMENT = original

    def test_bad_freeze_commit_provenance_fails_closed(self):
        original = s1b.METHODOLOGY_FREEZE_COMMIT
        s1b.METHODOLOGY_FREEZE_COMMIT = "0" * 40
        try:
            with self.assertRaises(IntegrityError) as ctx:
                s1b.verify_frozen_methodology(require_git=True)
            self.assertIn("methodology-freeze commit", str(ctx.exception))
        finally:
            s1b.METHODOLOGY_FREEZE_COMMIT = original

    def test_candidate_checkpoint_hash_mismatch_fails_closed(self):
        original = s1.EXPECTED_CANDIDATES_SHA256
        s1.EXPECTED_CANDIDATES_SHA256 = "f" * 64
        try:
            with self.assertRaises(s1.Stage1IntegrityError):
                s1.load_frozen_checkpoint()
        finally:
            s1.EXPECTED_CANDIDATES_SHA256 = original

    def test_candidate_checkpoint_is_intact(self):
        doc, cands = s1.load_frozen_checkpoint()
        self.assertEqual(len(cands), 10)
        self.assertEqual([c["stage1_candidate_id"] for c in cands],
                         s1.EXPECTED_CANDIDATE_IDS)

    def test_wrong_candidate_count_or_order_fails_closed(self):
        original = s1.EXPECTED_CANDIDATE_IDS
        s1.EXPECTED_CANDIDATE_IDS = list(reversed(original))
        try:
            with self.assertRaises(s1.Stage1IntegrityError):
                s1.load_frozen_checkpoint()
        finally:
            s1.EXPECTED_CANDIDATE_IDS = original


# ══════════════════════════════════════════════════════════
# 2. Seeds, strata, S grid, alpha
# ══════════════════════════════════════════════════════════

class TestDeclaredConfiguration(unittest.TestCase):

    def test_seed_list_is_ten_fresh_disjoint_seeds(self):
        self.assertTrue(s1b.verify_seed_list(s1b.MASTER_SEEDS))
        self.assertEqual(len(s1b.MASTER_SEEDS), 10)
        self.assertEqual(len(set(s1b.MASTER_SEEDS)), 10)
        self.assertFalse(set(s1b.MASTER_SEEDS) & s1b.FORBIDDEN_SEEDS)

    def test_stage1_seeds_are_forbidden_for_stage1b(self):
        for seed in s1.MASTER_SEEDS:
            self.assertIn(seed, s1b.FORBIDDEN_SEEDS)
        for seed in (0, 42, 43, 44, 45, 46, 1000, 1000003):
            self.assertIn(seed, s1b.FORBIDDEN_SEEDS)

    def test_forbidden_seed_fails_closed(self):
        with self.assertRaises(IntegrityError):
            s1b.verify_seed_list([700001] + s1b.MASTER_SEEDS[1:])
        with self.assertRaises(IntegrityError):
            s1b.verify_seed_list([42] + s1b.MASTER_SEEDS[1:])

    def test_duplicate_seed_fails_closed(self):
        dup = list(s1b.MASTER_SEEDS)
        dup[1] = dup[0]
        with self.assertRaises(IntegrityError):
            s1b.verify_seed_list(dup)

    def test_wrong_seed_count_fails_closed(self):
        with self.assertRaises(IntegrityError):
            s1b.verify_seed_list(s1b.MASTER_SEEDS[:9])
        with self.assertRaises(IntegrityError):
            s1b.verify_seed_list(s1b.MASTER_SEEDS + [810011])

    def test_frozen_s_grid(self):
        self.assertEqual(s1b.S_VALUES, [200, 500, 1000])
        self.assertEqual(s1b.MASTER_S, 1000)
        self.assertEqual(s1b.S_REF, 1000)
        self.assertEqual(s1b.GATING_S, [200, 500])

    def test_frozen_strata_membership(self):
        self.assertTrue(s1b.verify_strata())
        self.assertEqual(set(s1b.KNIFE_CANDIDATES),
                         {"S1-01", "S1-02", "S1-03", "S1-05", "S1-06"})
        self.assertEqual(set(s1b.OFF_CANDIDATES),
                         {"S1-04", "S1-07", "S1-08", "S1-09", "S1-10"})

    def test_incorrect_off_membership_fails_closed(self):
        original = s1b.OFF_CANDIDATES
        s1b.OFF_CANDIDATES = ("S1-04", "S1-07", "S1-08", "S1-09", "S1-01")
        try:
            with self.assertRaises(IntegrityError):
                s1b.verify_strata()
        finally:
            s1b.OFF_CANDIDATES = original

    def test_alpha_is_090_and_thresholds_are_frozen(self):
        self.assertEqual(s1b.EXPECTED_CONFIDENCE_ONTIME, 0.90)
        self.assertEqual(float(model.CONFIDENCE_ONTIME), 0.90)
        self.assertEqual(s1b.THRESHOLD_G1, 0.02)
        self.assertEqual(s1b.THRESHOLD_G2DAGGER, 0.02)
        self.assertEqual(s1b.THRESHOLD_G3, 0.10)
        self.assertEqual(s1b.TRIPWIRE_T, 0.01)

    def test_wrong_alpha_fails_closed_in_prepare_environment(self):
        original = model.CONFIDENCE_ONTIME
        model.CONFIDENCE_ONTIME = 0.85
        try:
            with self.assertRaises(s1.Stage1IntegrityError):
                s1.prepare_environment()
        finally:
            model.CONFIDENCE_ONTIME = original


# ══════════════════════════════════════════════════════════
# 3. Tie semantics (p_hat >= 0.90 is a PASS)
# ══════════════════════════════════════════════════════════

class TestTieRule(unittest.TestCase):

    def test_exact_boundary_is_a_pass(self):
        self.assertTrue(s1b.ccp_pass(0.0))
        self.assertTrue(s1b.ccp_pass(s1b.TOL_HARD))

    def test_real_shortfall_is_a_fail(self):
        self.assertFalse(s1b.ccp_pass(1e-3))
        self.assertFalse(s1b.ccp_pass(0.1))

    def test_nonfinite_shortfall_fails_closed(self):
        for bad in (float("nan"), float("inf")):
            with self.assertRaises(IntegrityError):
                s1b.ccp_pass(bad)

    def test_tie_misclassified_as_fail_is_caught(self):
        row = {"min_on_time_prob": 0.90, "chance_vio_total": 0.0,
               "chance_constraint_pass": False}
        with self.assertRaises(IntegrityError) as ctx:
            s1b.validate_cell(row, ("S1-01", 810001, 200))
        self.assertIn("disagrees with the frozen", str(ctx.exception))

    def test_tie_correctly_classified_passes(self):
        row = {"min_on_time_prob": 0.90, "chance_vio_total": 0.0,
               "chance_constraint_pass": True}
        self.assertTrue(s1b.validate_cell(row, ("S1-01", 810001, 200)))

    def test_alpha_times_S_is_integer_so_the_tie_is_attainable(self):
        for S in s1b.S_VALUES:
            self.assertEqual((s1b.EXPECTED_CONFIDENCE_ONTIME * S) % 1, 0.0)


# ══════════════════════════════════════════════════════════
# 4. Cell validity / fail-closed completeness
# ══════════════════════════════════════════════════════════

def row(cid, seed, S, p, cv=None, **kw):
    cv = (max(0.0, 0.90 - p) if cv is None else cv)
    base = {"candidate_id": cid, "master_seed": seed, "S": S,
            "min_on_time_prob": p, "chance_vio_total": cv,
            "chance_constraint_pass": s1b.ccp_pass(cv),
            "binding_batch_set": "1", "max_late_excess_h": 0.0,
            "nonccp_hard_pass": True}
    base.update(kw)
    return base


def full_grid(p_by_S=None):
    p_by_S = p_by_S or {200: 0.90, 500: 0.90, 1000: 0.90}
    return [row(c, s, S, p_by_S[S])
            for c in CANDS for s in s1b.MASTER_SEEDS for S in s1b.S_VALUES]


class TestGridValidity(unittest.TestCase):

    def test_complete_grid_is_accepted(self):
        idx = s1b.require_complete_grid(full_grid(), CANDS, s1b.MASTER_SEEDS,
                                        s1b.S_VALUES)
        self.assertIsInstance(idx, s1b.ValidatedGrid)
        self.assertEqual(idx.n_cells, 300)

    def test_missing_cell_invalidates_the_whole_run(self):
        rows = full_grid()[:-1]
        with self.assertRaises(IntegrityError) as ctx:
            s1b.require_complete_grid(rows, CANDS, s1b.MASTER_SEEDS, s1b.S_VALUES)
        msg = str(ctx.exception)
        self.assertIn("INVALID RUN", msg)
        self.assertIn("never reduced", msg)

    def test_duplicate_row_fails_closed(self):
        rows = full_grid()
        rows.append(dict(rows[0]))
        with self.assertRaises(IntegrityError):
            s1b.require_complete_grid(rows, CANDS, s1b.MASTER_SEEDS, s1b.S_VALUES)

    def test_extra_out_of_grid_cell_fails_closed(self):
        rows = full_grid()
        rows.append(row("S1-01", 999999, 200, 0.90))
        with self.assertRaises(IntegrityError):
            s1b.require_complete_grid(rows, CANDS, s1b.MASTER_SEEDS, s1b.S_VALUES)

    def test_nan_and_inf_probability_fail_closed(self):
        for bad in (float("nan"), float("inf")):
            with self.assertRaises(IntegrityError):
                s1b.validate_cell(row("S1-01", 810001, 200, bad, cv=0.0),
                                  ("S1-01", 810001, 200))

    def test_probability_outside_unit_interval_fails_closed(self):
        for bad in (-0.01, 1.01):
            with self.assertRaises(IntegrityError):
                s1b.validate_cell(row("S1-01", 810001, 200, bad, cv=0.0),
                                  ("S1-01", 810001, 200))

    def test_negative_or_nonfinite_chance_vio_fails_closed(self):
        with self.assertRaises(IntegrityError):
            s1b.validate_cell(row("S1-01", 810001, 200, 0.9, cv=-1e-6),
                              ("S1-01", 810001, 200))
        with self.assertRaises(IntegrityError):
            s1b.validate_cell(row("S1-01", 810001, 200, 0.9, cv=float("nan")),
                              ("S1-01", 810001, 200))

    def test_inconsistent_pass_flag_fails_closed(self):
        bad = row("S1-01", 810001, 200, 0.85, cv=0.05)
        bad["chance_constraint_pass"] = True     # should be False
        with self.assertRaises(IntegrityError):
            s1b.validate_cell(bad, ("S1-01", 810001, 200))


# ══════════════════════════════════════════════════════════
# 5. G1
# ══════════════════════════════════════════════════════════

class TestG1(unittest.TestCase):

    def test_value_and_denominator(self):
        g1 = s1b.compute_g1(cells(flat(0.01)), CANDS, SEEDS)
        self.assertAlmostEqual(g1["value"], 0.01, places=12)
        self.assertEqual(g1["denominator"], 100)
        self.assertTrue(g1["gating"])

    def test_uses_absolute_value(self):
        spec = {c: [0.01] * 5 + [-0.01] * 5 for c in CANDS}
        self.assertAlmostEqual(s1b.compute_g1(cells(spec), CANDS, SEEDS)["value"],
                               0.01, places=12)

    def test_strict_threshold_exactly_002_fails(self):
        self.assertFalse(s1b.compute_g1(cells(flat(0.02)), CANDS, SEEDS)["passed"])

    def test_just_below_threshold_passes(self):
        self.assertTrue(
            s1b.compute_g1(cells(flat(Fraction(19, 1000))),
                           CANDS, SEEDS)["passed"])

    def test_missing_cell_fails_closed(self):
        # Built short rather than deleted after the fact: a certified cell
        # mapping is immutable, so the absence has to be genuine.
        spec = flat(0.01)
        spec[CANDS[0]] = spec[CANDS[0]][:-1]        # one seed never supplied
        c = cells(spec)
        self.assertEqual(len(c), 99)
        with self.assertRaises(IntegrityError):
            s1b.compute_g1(c, CANDS, SEEDS)


# ══════════════════════════════════════════════════════════
# 6. D_c and G2dagger
# ══════════════════════════════════════════════════════════

class TestDcAndG2Dagger(unittest.TestCase):

    def test_dc_is_signed_mean_over_seeds(self):
        spec = flat(0.0)
        spec["S1-04"] = [0.02] * 5 + [-0.01] * 5      # mean = +0.005
        d = s1b.compute_d_c(cells(spec), CANDS, SEEDS, 10)
        self.assertAlmostEqual(d["S1-04"], 0.005, places=12)

    def test_dc_keeps_negative_sign(self):
        spec = flat(0.0)
        spec["S1-01"] = [-0.04] * 10
        d = s1b.compute_d_c(cells(spec), CANDS, SEEDS, 10)
        self.assertAlmostEqual(d["S1-01"], -0.04, places=12)

    def test_positive_means_smaller_S_optimism(self):
        c = cells({"S1-04": [0.03] * 10}, SEEDS)
        self.assertGreater(c[("S1-04", 0)]["p_S"], c[("S1-04", 0)]["p_ref"])
        d = s1b.compute_d_c(c, ["S1-04"], SEEDS)
        self.assertGreater(d["S1-04"], 0)

    def test_denominator_is_always_ten(self):
        with self.assertRaises(IntegrityError) as ctx:
            s1b.compute_d_c(cells(flat(0.0, n=9), SEEDS[:9]), CANDS,
                            SEEDS[:9], 10)
        self.assertIn("exactly 10", str(ctx.exception))

    def test_missing_seed_invalidates_rather_than_reducing_denominator(self):
        spec = flat(0.01)
        spec["S1-04"] = spec["S1-04"][:-1]          # 9 seeds, never 10
        c = cells(spec)
        self.assertEqual(len(c), 99)
        with self.assertRaises(IntegrityError) as ctx:
            s1b.compute_d_c(c, CANDS, SEEDS, 10)
        self.assertIn("never reduced", str(ctx.exception))

    def test_missing_candidate_invalidates(self):
        spec = flat(0.01)
        del spec["S1-09"]                           # candidate never supplied
        c = cells(spec)
        self.assertEqual(len(c), 90)
        with self.assertRaises(IntegrityError):
            s1b.compute_d_c(c, CANDS, SEEDS, 10)

    def test_g2dagger_is_the_maximum_not_the_mean(self):
        spec = flat(0.0)
        spec["S1-08"] = [0.03] * 10
        g2 = s1b.compute_g2dagger(cells(spec), CANDS, SEEDS, 10)
        self.assertAlmostEqual(g2["value"], 0.03, places=12)
        self.assertEqual(g2["argmax_candidate"], "S1-08")
        self.assertEqual(g2["argmax_stratum"], "OFF")

    def test_g2dagger_is_one_sided_pessimism_never_fails_it(self):
        spec = flat(-0.5)
        g2 = s1b.compute_g2dagger(cells(spec), CANDS, SEEDS, 10)
        self.assertTrue(g2["passed"])
        self.assertLess(g2["min_D_c"], 0)

    def test_g2dagger_catches_what_g1_dilutes(self):
        spec = flat(0.0)
        spec["S1-04"] = [0.03] * 10
        c = cells(spec)
        self.assertTrue(s1b.compute_g1(c, CANDS, SEEDS)["passed"])
        self.assertFalse(s1b.compute_g2dagger(c, CANDS, SEEDS, 10)["passed"])

    def test_strict_threshold_exactly_002_fails(self):
        spec = flat(0.0)
        spec["S1-04"] = [0.02] * 10
        self.assertFalse(
            s1b.compute_g2dagger(cells(spec), CANDS, SEEDS, 10)["passed"])

    def test_just_below_threshold_passes(self):
        spec = flat(0.0)
        spec["S1-04"] = [Fraction(19, 1000)] * 10
        self.assertTrue(
            s1b.compute_g2dagger(cells(spec), CANDS, SEEDS, 10)["passed"])

    def test_all_ten_dc_values_are_reported(self):
        g2 = s1b.compute_g2dagger(cells(flat(0.001)), CANDS, SEEDS, 10)
        self.assertEqual(sorted(g2["per_candidate_D_c"]), sorted(CANDS))
        self.assertEqual(len(g2["per_candidate_D_c"]), 10)

    def test_per_candidate_absolute_differences_reported(self):
        spec = {c: [0.01] * 5 + [-0.01] * 5 for c in CANDS}
        a_c = s1b.compute_a_c(cells(spec), CANDS, SEEDS)
        self.assertEqual(len(a_c), 10)
        for c in CANDS:
            self.assertAlmostEqual(a_c[c], 0.01, places=12)
        d_c = s1b.compute_d_c(cells(spec), CANDS, SEEDS, 10)
        self.assertAlmostEqual(d_c["S1-01"], 0.0, places=12)


# ══════════════════════════════════════════════════════════
# 7. G3 and flip decomposition
# ══════════════════════════════════════════════════════════

def flip_cells(n_off=0, n_knife=0):
    made = {"off": 0, "knife": 0}
    spec = {}
    for c in CANDS:
        per = []
        for _ in SEEDS:
            flip = False
            if c in s1b.OFF_CANDIDATES and made["off"] < n_off:
                flip, made["off"] = True, made["off"] + 1
            elif c in s1b.KNIFE_CANDIDATES and made["knife"] < n_knife:
                flip, made["knife"] = True, made["knife"] + 1
            per.append((0.0, True, not flip))
        spec[c] = per
    return cells(spec)


class TestG3(unittest.TestCase):

    def test_denominator_is_fifty_off_cells(self):
        self.assertEqual(s1b.compute_g3(flip_cells(0), SEEDS)["denominator"], 50)

    def test_four_flips_pass_five_flips_fail(self):
        self.assertTrue(s1b.compute_g3(flip_cells(4), SEEDS)["passed"])
        self.assertFalse(s1b.compute_g3(flip_cells(5), SEEDS)["passed"])

    def test_rate_exactly_010_fails(self):
        g3 = s1b.compute_g3(flip_cells(5), SEEDS)
        self.assertAlmostEqual(g3["value"], 0.10, places=12)
        self.assertFalse(g3["passed"])

    def test_knife_flips_never_gate(self):
        g3 = s1b.compute_g3(flip_cells(0, n_knife=50), SEEDS)
        self.assertTrue(g3["passed"])
        self.assertEqual(g3["flip_count"], 0)

    def test_all_pair_flips_never_gate(self):
        c = flip_cells(0, n_knife=50)
        self.assertEqual(s1b.flip_stats(c, CANDS, SEEDS)["gross_flip_rate"], 0.5)
        self.assertTrue(s1b.compute_g3(c, SEEDS)["passed"])

    def test_false_feasible_and_infeasible_decomposition(self):
        spec = {c: [(0.0, True, True)] * 10 for c in CANDS}
        spec["S1-07"] = [(0.0, True, False)] * 10      # false-feasible
        spec["S1-08"] = [(0.0, False, True)] * 4 + [(0.0, True, True)] * 6
        st = s1b.flip_stats(cells(spec), CANDS, SEEDS)
        self.assertEqual(st["false_feasible_count"], 10)
        self.assertEqual(st["false_infeasible_count"], 4)
        self.assertEqual(st["signed_net_classification_imbalance"], 6)
        self.assertEqual(st["gross_flip_count"], 14)


# ══════════════════════════════════════════════════════════
# 8. Non-gating: tripwire, bootstrap, LOCO
# ══════════════════════════════════════════════════════════

class TestNonGating(unittest.TestCase):

    def test_tripwire_value_and_denominator(self):
        t = s1b.compute_tripwire(cells(flat(0.012)), CANDS, SEEDS)
        self.assertAlmostEqual(t["value"], 0.012, places=12)
        self.assertEqual(t["denominator"], 100)

    def test_tripwire_is_inclusive_at_001(self):
        self.assertTrue(
            s1b.compute_tripwire(cells(flat(0.01)), CANDS, SEEDS)["tripped"])
        # 9/1000 is the attainable grid neighbour just below 0.01;
        # 0.0099 would put p_S at 0.9099, off the 1/1000 grid.
        self.assertFalse(
            s1b.compute_tripwire(cells(flat(Fraction(9, 1000))),
                                 CANDS, SEEDS)["tripped"])

    def test_mirror_pessimism_tripwire(self):
        t = s1b.compute_tripwire(cells(flat(-0.02)), CANDS, SEEDS)
        self.assertTrue(t["tripped"])
        self.assertTrue(t["tripped_pessimism"])
        self.assertFalse(t["tripped_optimism"])
        self.assertEqual(t["direction"], "pessimism")

    def test_tripwire_is_never_gating(self):
        t = s1b.compute_tripwire(cells(flat(0.015)), CANDS, SEEDS)
        self.assertFalse(t["gating"])
        self.assertNotIn("passed", t)

    def test_tripwire_cannot_change_a_gate_verdict(self):
        c = cells(flat(0.015))          # tripped, but all gates pass
        self.assertTrue(s1b.compute_tripwire(c, CANDS, SEEDS)["tripped"])
        self.assertTrue(s1b.compute_g1(c, CANDS, SEEDS)["passed"])
        self.assertTrue(s1b.compute_g2dagger(c, CANDS, SEEDS, 10)["passed"])
        self.assertTrue(s1b.compute_g3(c, SEEDS)["passed"])

    def test_bootstrap_and_loco_are_descriptive_only(self):
        c = cells({cd: [0.001 * (i + 1)] * 10 for i, cd in enumerate(CANDS)})
        for obj in (s1b.compute_bootstrap(c, CANDS, SEEDS, n_resamples=40),
                    s1b.compute_loco(c, CANDS, SEEDS)):
            self.assertFalse(obj["gating"])
            self.assertEqual(obj["role"], "DESCRIPTIVE_ONLY")
            self.assertNotIn("passed", obj)
            self.assertNotIn("verdict", obj)
            self.assertNotIn("INDETERMINATE", json.dumps(obj, default=str).upper()
                             .replace("NOT CREATE AN INDETERMINATE", ""))

    def test_loco_reports_every_candidate(self):
        lo = s1b.compute_loco(cells(flat(0.001)), CANDS, SEEDS)
        self.assertEqual(sorted(lo["per_omitted_candidate"]), sorted(CANDS))

    def test_bootstrap_is_deterministic_for_a_fixed_seed(self):
        c = cells(flat(0.005))
        a = s1b.compute_bootstrap(c, CANDS, SEEDS, n_resamples=50)
        b = s1b.compute_bootstrap(c, CANDS, SEEDS, n_resamples=50)
        self.assertEqual(a["G1_interval"], b["G1_interval"])

    def test_no_indeterminate_category_exists(self):
        idx = s1b.require_complete_grid(full_grid(), CANDS, s1b.MASTER_SEEDS,
                                        s1b.S_VALUES)
        comp = s1b.evaluate_comparison(idx, 200, 1000, CANDS, s1b.MASTER_SEEDS,
                                       gating=True, n_seeds_expected=10,
                                       with_bootstrap=False)
        self.assertIn(comp["gate_verdict"], ("PASS", "FAIL"))
        self.assertFalse(comp["indeterminate_category_exists"])


# ══════════════════════════════════════════════════════════
# 9. Comparison assembly and the Stage-2 ladder
# ══════════════════════════════════════════════════════════

def comp(S, S_ref, verdict, gating=True):
    return {"comparison": f"S{S}_vs_S{S_ref}", "S": S, "S_ref": S_ref,
            "is_gating_comparison": gating, "gate_verdict": verdict}


class TestLadder(unittest.TestCase):

    def test_s200_pass_selects_200_without_consulting_s500(self):
        d = s1b.decide_ladder([comp(200, 1000, "PASS"), comp(500, 1000, "FAIL")])
        self.assertEqual(d["selected_S"], 200)
        self.assertTrue(d["stage2_may_proceed"])
        self.assertFalse(d["s500_gate_consulted"])

    def test_s200_fail_falls_through_to_s500(self):
        d = s1b.decide_ladder([comp(200, 1000, "FAIL"), comp(500, 1000, "PASS")])
        self.assertEqual(d["selected_S"], 500)
        self.assertTrue(d["s500_gate_consulted"])

    def test_both_fail_blocks_stage2(self):
        d = s1b.decide_ladder([comp(200, 1000, "FAIL"), comp(500, 1000, "FAIL")])
        self.assertIsNone(d["selected_S"])
        self.assertFalse(d["stage2_may_proceed"])

    def test_s200_vs_s500_can_never_gate(self):
        d = s1b.decide_ladder([comp(200, 1000, "FAIL"), comp(500, 1000, "FAIL"),
                               comp(200, 500, "PASS", gating=False)])
        self.assertIsNone(d["selected_S"])

    def test_s200_vs_s500_marked_non_gating_in_output(self):
        idx = s1b.require_complete_grid(full_grid(), CANDS, s1b.MASTER_SEEDS,
                                        s1b.S_VALUES)
        c = s1b.evaluate_comparison(idx, 200, 500, CANDS, s1b.MASTER_SEEDS,
                                    gating=False, with_bootstrap=False)
        self.assertFalse(c["is_gating_comparison"])
        self.assertEqual(c["gate_verdict"], "NOT_A_GATE")

    def test_missing_gating_comparison_fails_closed(self):
        with self.assertRaises(IntegrityError):
            s1b.decide_ladder([comp(200, 1000, "PASS")])

    def test_gate_against_wrong_reference_fails_closed(self):
        with self.assertRaises(IntegrityError) as ctx:
            s1b.decide_ladder([comp(200, 500, "PASS"), comp(500, 1000, "PASS")])
        self.assertIn("S_ref", str(ctx.exception))

    def test_verdict_uses_conjunction_of_three_gates(self):
        idx = s1b.require_complete_grid(
            full_grid({200: 0.90, 500: 0.90, 1000: 0.90}),
            CANDS, s1b.MASTER_SEEDS, s1b.S_VALUES)
        c = s1b.evaluate_comparison(idx, 200, 1000, CANDS, s1b.MASTER_SEEDS,
                                    gating=True, n_seeds_expected=10,
                                    with_bootstrap=False)
        self.assertEqual(c["gate_verdict"], "PASS")
        for name in ("G1", "G2dagger", "G3"):
            self.assertTrue(c["gates"][name]["gating"])
            self.assertTrue(c["gates"][name]["passed"])

    def test_one_failing_gate_fails_the_comparison(self):
        grid = full_grid({200: 0.95, 500: 0.90, 1000: 0.90})
        idx = s1b.require_complete_grid(grid, CANDS, s1b.MASTER_SEEDS,
                                        s1b.S_VALUES)
        c = s1b.evaluate_comparison(idx, 200, 1000, CANDS, s1b.MASTER_SEEDS,
                                    gating=True, n_seeds_expected=10,
                                    with_bootstrap=False)
        self.assertEqual(c["gate_verdict"], "FAIL")

    def test_tripwire_tripped_while_gate_passed_is_escalated(self):
        # 0.915 IS attainable at S=200 (183/200); 0.912 was not (182.4).
        grid = full_grid({200: 0.915, 500: 0.90, 1000: 0.90})
        idx = s1b.require_complete_grid(grid, CANDS, s1b.MASTER_SEEDS,
                                        s1b.S_VALUES)
        c = s1b.evaluate_comparison(idx, 200, 1000, CANDS, s1b.MASTER_SEEDS,
                                    gating=True, n_seeds_expected=10,
                                    with_bootstrap=False)
        self.assertEqual(c["gate_verdict"], "PASS")
        self.assertTrue(c["non_gating"]["optimism_tripwire"]["tripped"])
        self.assertTrue(c["tripwire_tripped_while_gate_passed"])
        self.assertIn("MANDATORY_STAGE2_LIMITATION", c)
        self.assertIsNotNone(c["non_gating"]["tripwire_decomposition"])

    def test_gate_flag_stripped_is_detected(self):
        real = s1b.compute_g1

        def fake(*args, **kwargs):
            out = real(*args, **kwargs)
            out["gating"] = False
            return out
        s1b.compute_g1 = fake
        try:
            idx = s1b.require_complete_grid(full_grid(), CANDS,
                                            s1b.MASTER_SEEDS, s1b.S_VALUES)
            with self.assertRaises(IntegrityError):
                s1b.evaluate_comparison(idx, 200, 1000, CANDS, s1b.MASTER_SEEDS,
                                        gating=True, with_bootstrap=False)
        finally:
            s1b.compute_g1 = real


# ══════════════════════════════════════════════════════════
# 10. Output protection and run policy
# ══════════════════════════════════════════════════════════

class TestOutputAndPolicy(unittest.TestCase):

    def test_refuses_to_overwrite_non_empty_output_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = FSPath(tmp) / "results"
            d.mkdir()
            (d / "raw_rows.csv").write_text("pre-existing", encoding="utf-8")
            with self.assertRaises(IntegrityError) as ctx:
                s1b.write_outputs([], [], {}, {}, {}, d)
            self.assertIn("Refusing to overwrite", str(ctx.exception))

    def test_writes_into_an_empty_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = FSPath(tmp) / "results"
            s1b.write_outputs([], [], {"selected_S": None}, {}, {"stage": "x"}, d)
            self.assertTrue((d / "manifest.json").exists())
            self.assertTrue((d / "per_candidate_diagnostics.json").exists())

    def test_sanity_and_full_output_directories_are_distinct(self):
        self.assertNotEqual(s1b.OUT_DIR, s1b.SANITY_OUT_DIR)

    def test_full_run_without_approval_is_refused(self):
        r = subprocess.run(
            [sys.executable, str(HERE / "a_lite_stage1b_sensitivity.py"), "--run"],
            capture_output=True, text=True, cwd=str(HERE))
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("--i-have-approval", r.stdout + r.stderr)

    def test_sanity_run_without_approval_is_refused(self):
        r = subprocess.run(
            [sys.executable, str(HERE / "a_lite_stage1b_sensitivity.py"),
             "--sanity-run"], capture_output=True, text=True, cwd=str(HERE))
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("--i-have-approval", r.stdout + r.stderr)

    def test_a_mode_is_required(self):
        r = subprocess.run(
            [sys.executable, str(HERE / "a_lite_stage1b_sensitivity.py")],
            capture_output=True, text=True, cwd=str(HERE))
        self.assertNotEqual(r.returncode, 0)

    def test_full_run_rejects_a_reduced_configuration(self):
        """A 'full-run' with fewer than 10 seeds must fail before any work."""
        with self.assertRaises(IntegrityError) as ctx:
            s1b.run_study(None, {}, [], s1b.MASTER_SEEDS[:2], s1b.S_VALUES,
                          "full-run", s1b.SANITY_OUT_DIR)
        self.assertIn("master seeds", str(ctx.exception))

    def test_full_run_rejects_a_reduced_s_grid(self):
        with self.assertRaises(IntegrityError) as ctx:
            s1b.run_study(None, {}, [], s1b.MASTER_SEEDS, [200, 1000],
                          "full-run", s1b.SANITY_OUT_DIR)
        self.assertIn("frozen S grid", str(ctx.exception))

    def test_full_run_rejects_a_reduced_candidate_panel(self):
        with self.assertRaises(IntegrityError) as ctx:
            s1b.run_study(None, {}, [], s1b.MASTER_SEEDS, s1b.S_VALUES,
                          "full-run", s1b.SANITY_OUT_DIR)
        self.assertIn("exactly 10 candidates", str(ctx.exception))

    def test_synthetic_full_run_gate_arithmetic_selftests_pass(self):
        self.assertTrue(s1b.self_test_gate_arithmetic())
        self.assertTrue(s1b.self_test_ladder_arithmetic())


# ══════════════════════════════════════════════════════════
# 11. Integration: scenario construction, nesting, cache
# ══════════════════════════════════════════════════════════

@unittest.skipUnless(INTEGRATION, "set STAGE1B_INTEGRATION=1 to run")
class TestScenarioIntegration(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.env = s1.prepare_environment()
        cls.seed = s1b.MASTER_SEEDS[0]
        cls.master = s1b.build_master(cls.env, cls.seed)

    def test_master_is_size_1000_with_validated_border_config(self):
        self.assertEqual(self.master.size, 1000)
        self.assertEqual(len(self.master.border_event_mean_h),
                         s1.EXPECTED_BORDER_EVENTS)
        self.assertEqual(len(self.master.arc_border_event),
                         s1.EXPECTED_BORDER_ARCS)

    def test_prefixes_are_literal_and_verified(self):
        subsets, report = s1b.nested_subsets(self.master, self.seed)
        self.assertTrue(report["all_pass"])
        for S in s1b.S_VALUES:
            self.assertEqual(subsets[S].size, S)
        for key, arr in subsets[200].travel_multiplier.items():
            self.assertTrue(np.array_equal(
                arr, self.master.travel_multiplier[key][:200]))
        for key, arr in subsets[500].border_delay_h.items():
            self.assertTrue(np.array_equal(
                arr, self.master.border_delay_h[key][:500]))

    def test_independent_regeneration_is_not_a_prefix(self):
        """The reason independent build_scenario_set calls are PROHIBITED."""
        independent = model.build_scenario_set(
            arcs=self.env["arcs"], border_delay_map=self.env["border_delay_map"],
            size=200, seed=self.seed, stochastic=True,
            border_event_definitions=self.env["border_event_definitions"])
        v = scs.verify_prefix_subset(independent, self.master)
        self.assertFalse(v["is_prefix_subset"])

    def test_corrupted_prefix_is_detected(self):
        subsets, _ = s1b.nested_subsets(self.master, self.seed)
        bad = scs.make_nested_subset(self.master, 200)
        key = sorted(bad.travel_multiplier)[0]
        bad.travel_multiplier[key] = bad.travel_multiplier[key].copy()
        bad.travel_multiplier[key][7] += 0.5
        v = scs.verify_prefix_subset(bad, self.master)
        self.assertFalse(v["is_prefix_subset"])

    def test_truncated_array_is_detected(self):
        """A short array must fail prefix verification against the master."""
        bad = scs.make_nested_subset(self.master, 200)
        key = sorted(bad.travel_multiplier)[0]
        bad.travel_multiplier[key] = bad.travel_multiplier[key][:100]
        v = scs.verify_prefix_subset(self.master, bad)
        self.assertFalse(v["is_prefix_subset"])

    def test_border_event_tripwire_fails_closed(self):
        original = s1.EXPECTED_BORDER_EVENTS
        s1.EXPECTED_BORDER_EVENTS = original + 1
        try:
            with self.assertRaises(IntegrityError) as ctx:
                s1b.build_master(self.env, s1b.MASTER_SEEDS[1])
            self.assertIn("border-event configuration mismatch",
                          str(ctx.exception))
        finally:
            s1.EXPECTED_BORDER_EVENTS = original

    def test_scenarioset_schema_drift_fails_closed(self):
        original = set(s1b.EXPECTED_SCENARIOSET_FIELDS)
        s1b.EXPECTED_SCENARIOSET_FIELDS = original | {"a_new_field"}
        try:
            with self.assertRaises(IntegrityError) as ctx:
                s1b.assert_scenarioset_schema(self.master)
            self.assertIn("ScenarioSet schema changed", str(ctx.exception))
        finally:
            s1b.EXPECTED_SCENARIOSET_FIELDS = original

    def test_install_clears_stale_scenario_cache(self):
        subsets, _ = s1b.nested_subsets(self.master, self.seed)
        model._PATH_SCENARIO_CACHE = {("stale", 1, True, 0): "poison"}
        s1.install_scenario_set(subsets[200])
        self.assertEqual(model._PATH_SCENARIO_CACHE, {})
        self.assertIs(model.ACTIVE_SCENARIO_SET, subsets[200])

    def test_foreign_cache_entry_is_detected(self):
        subsets, _ = s1b.nested_subsets(self.master, self.seed)
        s1.install_scenario_set(subsets[200])
        model._PATH_SCENARIO_CACHE[("foreign", 999, True, 0)] = "x"
        try:
            with self.assertRaises(s1.Stage1IntegrityError):
                s1.assert_cache_confined(subsets[200])
        finally:
            model._PATH_SCENARIO_CACHE = {}

    def test_seed_streams_differ_across_master_seeds(self):
        other = s1b.build_master(self.env, s1b.MASTER_SEEDS[1])
        key = sorted(self.master.travel_multiplier)[0]
        self.assertFalse(np.array_equal(self.master.travel_multiplier[key],
                                        other.travel_multiplier[key]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
