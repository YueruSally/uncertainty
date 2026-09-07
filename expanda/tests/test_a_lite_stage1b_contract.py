#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Stage-1B CONTRACT coverage (Parts A-I).

Companion to ``expanda/test_a_lite_stage1b.py``. That file proves the
fail-closed guards fire; this file proves the FROZEN SCIENTIFIC CONTRACT of
`expanda/A_LITE_STAGE1B_METHODOLOGY_AMENDMENT.md`
(sha256 df85d40b...0188, freeze commit 369585eb) holds exactly:

  A  G1  exact 0.02 boundary, incl. the 0.95-0.93 float-representation trap
  B  G2dagger exact 0.02 boundary on max_c D_c, denominator exactly 10
  C  G3  exact 0.10 boundary on the OFF stratum, denominator exactly 50
  D  exact_p valid/invalid contract
  E  complete-grid / ValidatedGrid contract, incl. bypass resistance
  F  VALIDATION_FAILURE.json and abort-before-gates
  G  fixed denominators 10 / 100 / 50 / 100, never reduced
  H  non-gating isolation and boolean gate aggregation (full truth table)
  I  boolean typing, sanity KNIFE/OFF composition, report completeness

DESIGN RULES OBSERVED THROUGHOUT
  * Every probability is an ATTAINABLE empirical proportion k/S. A value that
    a min-over-batch-proportions estimator could not produce is never used,
    because a gate proved on impossible data proves nothing.
  * Tests drive the REAL public implementation (`require_complete_grid`,
    `build_cells`, `compute_g1`, `compute_g2dagger`, `compute_g3`,
    `compute_tripwire`, `evaluate_comparison`, `decide_ladder`,
    `render_report`, `base_manifest`). No gate arithmetic is reimplemented
    here; expected values are stated as literal frozen constants.
  * Boundary tests assert BOTH sides (at the threshold -> FAIL, nearest
    attainable value below -> PASS), so a `<` silently widened to `<=` -- or
    inverted -- cannot leave the suite green.

Runs NO Stage-1B study: zero candidate evaluations, zero ScenarioSets, no
sanity run, no full run. Nothing on disk outside pytest's tmp_path is written.

Run:  ../.venv/bin/python -m pytest tests/test_a_lite_stage1b_contract.py -q
"""
import copy
import hashlib
import json
import os
import sys
from fractions import Fraction
from pathlib import Path as FSPath

import numpy as np
import pytest

HERE = FSPath(__file__).resolve().parent.parent


def _tree_inventory(d):
    """Typed content inventory of an output tree: relative path -> entry.

    Entries are ("file", sha256), ("dir", None) or ("symlink", target), so a
    deletion, a rewrite, a delete-and-recreate, an empty subdirectory, and a
    relinked symlink are all detected -- not merely file creation. None when
    the directory itself is absent. Symlinks are classified before
    directories, since Path.is_dir() follows them.
    """
    d = FSPath(d)
    # lexists, not exists: a BROKEN root symlink must register as present,
    # otherwise swapping an absent root for a dangling link reads as "absent".
    if not os.path.lexists(d):
        return None
    # The ROOT entry is inventoried too, so replacing a real directory with a
    # symlink to an identical-content tree cannot produce an equal inventory.
    out = {".": ("symlink", os.readlink(d)) if d.is_symlink()
           else ("dir", None) if d.is_dir() else
           ("file", hashlib.sha256(d.read_bytes()).hexdigest())}
    if not d.is_dir() or d.is_symlink():
        return out
    for p in sorted(d.rglob("*")):
        rel = str(p.relative_to(d))
        if p.is_symlink():
            out[rel] = ("symlink", os.readlink(p))
        elif p.is_dir():
            out[rel] = ("dir", None)
        else:
            out[rel] = ("file", hashlib.sha256(p.read_bytes()).hexdigest())
    return out


# Snapshot taken BEFORE the Stage-1B driver is imported, so an import-time
# side effect cannot be absorbed into the baseline. Paths are derived from the
# repository layout here and cross-checked against the driver's own constants
# once it is imported (see the guard tests at the end of this file).
_OUT_DIR_PATHS = (HERE / "a_lite_stage1b_results", HERE / "a_lite_stage1b_sanity")
_OUT_TREES_AT_IMPORT = {str(d): _tree_inventory(d) for d in _OUT_DIR_PATHS}

sys.path.insert(0, str(HERE))
import a_lite_stage1b_sensitivity as s1b   # noqa: E402
import a_lite_stage1_sensitivity as s1     # noqa: E402

IE = s1b.Stage1BIntegrityError

ALPHA = 0.90
CANDS = list(s1.EXPECTED_CANDIDATE_IDS)          # S1-01 .. S1-10, frozen order
SEEDS = list(s1b.MASTER_SEEDS)                   # the 10 declared master seeds
S_VALUES = list(s1b.S_VALUES)                    # [200, 500, 1000]
KNIFE = list(s1b.KNIFE_CANDIDATES)
OFF = list(s1b.OFF_CANDIDATES)


# ══════════════════════════════════════════════════════════════════
# Fixtures: raw rows on the ATTAINABLE k/S grid only
# ══════════════════════════════════════════════════════════════════

def make_row(cid, seed, S, p, **override):
    """One raw result row as the driver's evaluator emits it.

    `chance_vio_total` is the production shortfall sum for a single binding
    batch, and `chance_constraint_pass` is recomputed by the driver's own
    frozen predicate -- so the row is self-consistent by construction and the
    inclusive tie rule (p_hat >= alpha PASSES) is exercised, never asserted.
    """
    p = float(p)
    cv = max(0.0, ALPHA - p)
    row = {"candidate_id": cid, "master_seed": int(seed), "S": int(S),
           "min_on_time_prob": p, "chance_vio_total": cv,
           "chance_constraint_pass": s1b.ccp_pass(cv),
           "binding_batch_set": "1", "max_late_excess_h": 0.0,
           "nonccp_hard_pass": True}
    row.update(override)
    return row


def build_rows(p_fn, candidates=None, seeds=None, s_values=None):
    """Full declared grid from a p_fn(candidate, seed_index, S) -> probability."""
    candidates = CANDS if candidates is None else candidates
    seeds = SEEDS if seeds is None else seeds
    s_values = S_VALUES if s_values is None else s_values
    return [make_row(c, s, S, p_fn(c, i, S))
            for c in candidates
            for i, s in enumerate(seeds)
            for S in s_values]


def grid(p_fn, **kw):
    """Rows -> the real fail-closed precheck -> a ValidatedGrid."""
    candidates = kw.pop("candidates", CANDS)
    seeds = kw.pop("seeds", SEEDS)
    s_values = kw.pop("s_values", S_VALUES)
    rows = build_rows(p_fn, candidates, seeds, s_values)
    return s1b.require_complete_grid(rows, candidates, seeds, s_values)


def cells_of(p_fn, S=200, S_ref=1000, candidates=None, seeds=None):
    """The paired cell structure for one comparison, via the real path."""
    candidates = CANDS if candidates is None else candidates
    seeds = SEEDS if seeds is None else seeds
    vg = grid(p_fn, candidates=candidates, seeds=seeds)
    return s1b.build_cells(vg, S, S_ref, candidates, seeds)


def flat_p(p200, p500, p1000):
    return lambda c, i, S: {200: p200, 500: p500, 1000: p1000}[S]


# ══════════════════════════════════════════════════════════════════
# A. EXACT G1 BOUNDARY  (mean |dp| over 100 cells, STRICT < 0.02)
# ══════════════════════════════════════════════════════════════════

class TestPartA_G1ExactBoundary:

    def test_g1_exactly_at_002_fails(self):
        """p_S = 0.92 @S200 (184/200) vs p_ref = 0.90 @S1000 (900/1000)."""
        g1 = s1b.compute_g1(cells_of(flat_p(0.92, 0.90, 0.90)), CANDS, SEEDS)
        assert Fraction(g1["value_exact"]) == Fraction(1, 50)
        assert g1["threshold"] == 0.02
        assert g1["passed"] is False, "G1 == 0.02 must FAIL: the gate is strict <"

    def test_g1_nearest_attainable_value_below_002_passes(self):
        """99 cells at 0.02 and one at 0.015 -> 0.01995, the closest the
        S=200 grid gets to the threshold from below. PASS."""
        def p_fn(c, i, S):
            if S != 200:
                return 0.90
            return 0.915 if (c == CANDS[0] and i == 0) else 0.92
        g1 = s1b.compute_g1(cells_of(p_fn), CANDS, SEEDS)
        assert Fraction(g1["value_exact"]) == Fraction(399, 20000)   # 0.01995
        assert g1["passed"] is True

    def test_g1_just_above_002_fails(self):
        """One cell pushed to 0.025 -> 0.02005. FAIL, so the pass region is
        genuinely bounded above and not merely 'anything small'."""
        def p_fn(c, i, S):
            if S != 200:
                return 0.90
            return 0.925 if (c == CANDS[0] and i == 0) else 0.92
        g1 = s1b.compute_g1(cells_of(p_fn), CANDS, SEEDS)
        assert Fraction(g1["value_exact"]) == Fraction(401, 20000)   # 0.02005
        assert g1["passed"] is False

    def test_095_minus_093_float_trap_cannot_rescue_a_failing_gate(self):
        """The representational case named by the frozen protocol.

        In binary floating point 0.95 - 0.93 == 0.019999999999999907, which is
        strictly BELOW 0.02 and would PASS. The exact empirical value is
        19/20 - 93/100 = 1/50 = 0.02 exactly, which must FAIL.
        """
        # 1. the trap is real: naive float arithmetic would pass this gate
        naive = sum([0.95 - 0.93] * 100) / 100
        assert naive < 0.02, "premise of the test: float arithmetic passes"
        assert (0.95 - 0.93) == pytest.approx(0.019999999999999907, abs=1e-18)

        # 2. exact reconstruction from recovered empirical numerators
        ex_s, ex_r = s1b.exact_p(0.95, 200), s1b.exact_p(0.93, 1000)
        assert (ex_s, ex_r) == (Fraction(190, 200), Fraction(930, 1000))
        assert ex_s - ex_r == Fraction(1, 50)

        # 3. the real gate, driven end to end, FAILS
        g1 = s1b.compute_g1(cells_of(flat_p(0.95, 0.93, 0.93)), CANDS, SEEDS)
        assert Fraction(g1["value_exact"]) == Fraction(1, 50)
        assert g1["passed"] is False
        assert g1["comparison"] == "strict_less_than_exact_rational"

    def test_gate_inputs_are_exact_rationals_not_floats(self):
        """Every cell carries a Fraction dp; the float fields are reporting
        only. Proven on the trap values, where the two disagree."""
        cells = cells_of(flat_p(0.95, 0.93, 0.93))
        cell = cells[(CANDS[0], SEEDS[0])]
        assert isinstance(cell["dp_exact"], Fraction)
        assert isinstance(cell["p_S_exact"], Fraction)
        assert cell["dp_exact"] == Fraction(1, 50)
        # the reported float is the lossy view; the gate never reads it
        assert cell["p_S"] - cell["p_ref"] < 0.02 <= float(cell["dp_exact"])

    def test_g1_uses_absolute_value_so_signs_cannot_cancel(self):
        """Half the seeds +0.02, half -0.02: signed mean is 0 but G1 is 0.02
        and must FAIL. Guards against G1 silently becoming a signed mean."""
        def p_fn(c, i, S):
            if S != 200:
                return 0.95
            return 0.97 if i % 2 == 0 else 0.93
        cells = cells_of(p_fn)
        g1 = s1b.compute_g1(cells, CANDS, SEEDS)
        trip = s1b.compute_tripwire(cells, CANDS, SEEDS)
        assert Fraction(trip["value_exact"]) == 0, "signed mean cancels to 0"
        assert Fraction(g1["value_exact"]) == Fraction(1, 50)
        assert g1["passed"] is False


# ══════════════════════════════════════════════════════════════════
# B. EXACT G2dagger BOUNDARY  (max_c D_c, STRICT < 0.02, denominator 10)
# ══════════════════════════════════════════════════════════════════

# Ten DIFFERENT per-seed deviations averaging to exactly 0.02, all on the
# 1/200 grid. Mixed on purpose: ten identical cells would not distinguish a
# mean from a max, a first element or a last element.
MIXED_DPS_MEAN_002 = [0.005, 0.010, 0.015, 0.020, 0.025,
                      0.030, 0.035, 0.020, 0.020, 0.020]     # sum 0.200
MIXED_DPS_MEAN_00195 = MIXED_DPS_MEAN_002[:-1] + [0.015]     # sum 0.195

TARGET = "S1-04"        # an OFF candidate, so a stratum mix-up is visible


def d_c_grid(dps, target=TARGET, base=0.90):
    """All candidates flat at `base`; `target` carries the given per-seed dps."""
    def p_fn(c, i, S):
        if S == 200 and c == target:
            return round(base + dps[i], 6)
        return base
    return p_fn


class TestPartB_G2DaggerExactBoundary:

    def test_d_c_is_the_mean_over_ten_seeds_of_mixed_values(self):
        cells = cells_of(d_c_grid(MIXED_DPS_MEAN_002))
        d_c = s1b.compute_d_c(cells, CANDS, SEEDS, s1b.EXPECTED_N_SEEDS)
        assert d_c[TARGET] == Fraction(1, 50)
        assert len(set(MIXED_DPS_MEAN_002)) > 1, "must be a genuine average"
        assert all(d_c[c] == 0 for c in CANDS if c != TARGET)

    def test_g2dagger_exactly_at_002_fails(self):
        cells = cells_of(d_c_grid(MIXED_DPS_MEAN_002))
        g2 = s1b.compute_g2dagger(cells, CANDS, SEEDS, s1b.EXPECTED_N_SEEDS)
        assert Fraction(g2["value_exact"]) == Fraction(1, 50)
        assert g2["argmax_candidate"] == TARGET
        assert g2["argmax_stratum"] == "OFF"
        assert g2["threshold"] == 0.02
        assert g2["passed"] is False, "max_c D_c == 0.02 must FAIL (strict <)"

    def test_g2dagger_below_002_passes(self):
        cells = cells_of(d_c_grid(MIXED_DPS_MEAN_00195))
        g2 = s1b.compute_g2dagger(cells, CANDS, SEEDS, s1b.EXPECTED_N_SEEDS)
        assert Fraction(g2["value_exact"]) == Fraction(39, 2000)     # 0.0195
        assert g2["passed"] is True

    def test_g2dagger_fails_while_g1_passes_so_dilution_cannot_hide_it(self):
        """The whole point of G2dagger: a single-candidate excursion the
        100-cell mean dilutes to 0.002."""
        cells = cells_of(d_c_grid(MIXED_DPS_MEAN_002))
        g1 = s1b.compute_g1(cells, CANDS, SEEDS)
        g2 = s1b.compute_g2dagger(cells, CANDS, SEEDS, s1b.EXPECTED_N_SEEDS)
        assert Fraction(g1["value_exact"]) == Fraction(1, 500)       # 0.002
        assert g1["passed"] is True and g2["passed"] is False

    def test_g2dagger_is_a_max_not_a_mean_over_candidates(self):
        """Nine candidates at -0.05 cannot average away one at +0.02."""
        def p_fn(c, i, S):
            if S != 200:
                return 0.90
            if c == TARGET:
                return round(0.90 + MIXED_DPS_MEAN_002[i], 6)
            return 0.85
        cells = cells_of(p_fn)
        g2 = s1b.compute_g2dagger(cells, CANDS, SEEDS, s1b.EXPECTED_N_SEEDS)
        assert Fraction(g2["value_exact"]) == Fraction(1, 50)
        assert g2["min_D_c"] == pytest.approx(-0.05)
        assert g2["passed"] is False

    def test_g2dagger_is_one_sided_pessimism_never_gates(self):
        """Every candidate at -0.20: extreme, but conservative, so PASS."""
        cells = cells_of(flat_p(0.70, 0.90, 0.90))
        g2 = s1b.compute_g2dagger(cells, CANDS, SEEDS, s1b.EXPECTED_N_SEEDS)
        assert Fraction(g2["value_exact"]) == Fraction(-1, 5)
        assert g2["passed"] is True

    def test_d_c_denominator_is_exactly_ten_and_reported(self):
        cells = cells_of(d_c_grid(MIXED_DPS_MEAN_002))
        g2 = s1b.compute_g2dagger(cells, CANDS, SEEDS, s1b.EXPECTED_N_SEEDS)
        assert g2["seed_denominator"] == 10
        assert len(g2["per_candidate_D_c"]) == 10
        assert len(g2["per_candidate_D_c_exact"]) == 10
        assert sorted(g2["per_candidate_D_c"]) == sorted(CANDS)

    def test_d_c_refuses_a_reduced_seed_denominator(self):
        """Nine seeds must not silently become the denominator."""
        vg = grid(d_c_grid(MIXED_DPS_MEAN_002))
        nine = SEEDS[:9]
        cells = s1b.build_cells(vg, 200, 1000, CANDS, nine)
        with pytest.raises(IE, match="exactly 10 master"):
            s1b.compute_d_c(cells, CANDS, nine, s1b.EXPECTED_N_SEEDS)
        with pytest.raises(IE, match="exactly 10 master"):
            s1b.compute_g2dagger(cells, CANDS, nine, s1b.EXPECTED_N_SEEDS)


# ══════════════════════════════════════════════════════════════════
# C. EXACT G3 BOUNDARY  (OFF-stratum gross flip rate, STRICT < 0.10)
# ══════════════════════════════════════════════════════════════════

def flip_grid(n_off_flips, knife_flips=False):
    """`n_off_flips` OFF cells classified PASS at S200 and FAIL at S1000.

    The flip is produced by real probabilities either side of alpha
    (p_S = 0.90 -> PASS by the inclusive rule; p_ref = 0.895 -> FAIL), never
    by writing a pass flag directly.
    """
    flipped = [(OFF[k % len(OFF)], k // len(OFF)) for k in range(n_off_flips)]

    def p_fn(c, i, S):
        if knife_flips and c in KNIFE:
            return 0.90 if S == 200 else 0.895
        if (c, i) in flipped and S == 1000:
            return 0.895
        return 0.90
    return p_fn


class TestPartC_G3ExactBoundary:

    def test_g3_four_flips_of_fifty_passes(self):
        cells = cells_of(flip_grid(4))
        g3 = s1b.compute_g3(cells, SEEDS, OFF)
        assert g3["denominator"] == 50 and g3["flip_count"] == 4
        assert Fraction(g3["value_exact"]) == Fraction(2, 25)        # 0.08
        assert g3["passed"] is True

    def test_g3_exactly_five_flips_of_fifty_is_010_and_fails(self):
        cells = cells_of(flip_grid(5))
        g3 = s1b.compute_g3(cells, SEEDS, OFF)
        assert g3["denominator"] == 50 and g3["flip_count"] == 5
        assert Fraction(g3["value_exact"]) == Fraction(1, 10)        # 0.10
        assert g3["threshold"] == 0.10
        assert g3["passed"] is False, "0.10 must FAIL: the gate is strict <"

    def test_g3_six_flips_also_fails(self):
        g3 = s1b.compute_g3(cells_of(flip_grid(6)), SEEDS, OFF)
        assert Fraction(g3["value_exact"]) == Fraction(3, 25)        # 0.12
        assert g3["passed"] is False

    def test_g3_zero_flips_passes(self):
        g3 = s1b.compute_g3(cells_of(flip_grid(0)), SEEDS, OFF)
        assert g3["flip_count"] == 0 and g3["passed"] is True

    def test_g3_is_the_off_stratum_gate_per_the_frozen_protocol(self):
        g3 = s1b.compute_g3(cells_of(flip_grid(5)), SEEDS, OFF)
        assert g3["name"] == "G3"
        assert g3["statistic"] == "OFF_stratum_gross_classification_flip_rate"
        assert g3["stratum_membership"] == OFF
        assert set(OFF) == {"S1-04", "S1-07", "S1-08", "S1-09", "S1-10"}
        assert set(KNIFE) & set(OFF) == set()

    def test_knife_flips_never_enter_g3(self):
        """All 50 KNIFE cells flip; G3 still reads only the OFF stratum."""
        cells = cells_of(flip_grid(4, knife_flips=True))
        g3 = s1b.compute_g3(cells, SEEDS, OFF)
        allpair = s1b.flip_stats(cells, CANDS, SEEDS)
        knife = s1b.flip_stats(cells, KNIFE, SEEDS)
        assert knife["gross_flip_count"] == 50
        assert allpair["gross_flip_count"] == 54
        assert allpair["gross_flip_rate"] == 0.54
        assert g3["flip_count"] == 4 and g3["passed"] is True

    def test_classification_uses_the_inclusive_tie_rule_at_alpha(self):
        """p_hat == 0.90 is a PASS, so a cell at exactly alpha never flips."""
        assert s1b.ccp_pass(0.0) is True                     # p_hat == alpha
        assert s1b.ccp_pass(s1b.TOL_HARD) is True
        assert s1b.ccp_pass(1e-3) is False                   # p_hat == alpha-1e-3
        cells = cells_of(flat_p(0.90, 0.90, 0.90))
        for c in CANDS:
            for s in SEEDS:
                assert cells[(c, s)]["pass_S"] is True
                assert cells[(c, s)]["pass_ref"] is True
        assert s1b.compute_g3(cells, SEEDS, OFF)["flip_count"] == 0

    def test_flip_decomposition_is_directional(self):
        cells = cells_of(flip_grid(5))
        st = s1b.flip_stats(cells, OFF, SEEDS)
        assert st["false_feasible_count"] == 5      # accepted at S, rejected at ref
        assert st["false_infeasible_count"] == 0
        assert st["signed_net_classification_imbalance"] == 5

    def test_g3_refuses_an_empty_off_stratum(self):
        cells = cells_of(flip_grid(0))
        with pytest.raises(IE, match="EMPTY OFF stratum"):
            s1b.compute_g3(cells, SEEDS, [])


# ══════════════════════════════════════════════════════════════════
# D. exact_p CONTRACT
# ══════════════════════════════════════════════════════════════════

VALID_EXACT_P = [
    (0.0, 200, 0), (0.0, 500, 0), (0.0, 1000, 0),
    (1.0, 200, 200), (1.0, 500, 500), (1.0, 1000, 1000),
    (0.90, 200, 180), (0.90, 500, 450), (0.90, 1000, 900),
    (0.92, 200, 184), (0.95, 200, 190), (0.005, 200, 1),
    (0.918, 500, 459), (0.999, 1000, 999), (0.001, 1000, 1),
    (0.93, 1000, 930), (0.895, 1000, 895), (0.5, 200, 100),
]

INVALID_EXACT_P = [
    (0.9099, 1000, "not a multiple"),      # off the 1/1000 grid
    (0.912, 200, "not a multiple"),        # 182.4 scenarios
    (1.0 / 3.0, 500, "not a multiple"),
    (0.9 + 1e-5, 1000, "not a multiple"),   # beyond the recovery tolerance
    (-0.01, 200, "outside"),
    (-1.0, 1000, "outside"),
    (1.01, 200, "outside"),
    (2.0, 1000, "outside"),
    (float("nan"), 200, "non-finite"),
    (float("inf"), 500, "non-finite"),
    (float("-inf"), 1000, "non-finite"),
]


class TestPartD_ExactP:

    @pytest.mark.parametrize("p,S,k", VALID_EXACT_P)
    def test_valid_values_recover_the_exact_numerator(self, p, S, k):
        got = s1b.exact_p(p, S)
        assert isinstance(got, Fraction)
        assert got == Fraction(k, S)
        assert got * S == k, "the recovered numerator must be exactly k"
        assert 0 <= got <= 1

    @pytest.mark.parametrize("p,S,msg", INVALID_EXACT_P)
    def test_invalid_values_fail_closed(self, p, S, msg):
        with pytest.raises(IE, match=msg):
            s1b.exact_p(p, S)

    @pytest.mark.parametrize("S", [0, -1, -200])
    def test_non_positive_scenario_count_fails_closed(self, S):
        with pytest.raises(IE, match="invalid scenario count"):
            s1b.exact_p(0.9, S)

    def test_numerator_recovery_tolerance_is_bounded_and_documented(self):
        """`exact_p` recovers the integer numerator a float is a rounding of,
        so it deliberately absorbs deviations up to |p*S - k| <= 1e-6 -- the
        scale of float round-trip error, ~10 orders of magnitude smaller than
        the 1/S grid spacing it must never confuse. Larger deviations, which
        cannot be rounding, are rejected.

        Consequence worth stating: at S=1000 the near-threshold float
        0.02 - 1e-9 is recovered as the exact 20/1000, i.e. as 0.02 -- so it
        is a FAIL, never a sneaky pass. Attainability of synthetic dp values
        is enforced separately by `_synthetic_cells`.
        """
        assert s1b.exact_p(0.9 + 1e-9, 1000) == Fraction(900, 1000)
        assert s1b.exact_p(0.9 - 1e-9, 1000) == Fraction(900, 1000)
        with pytest.raises(IE, match="not a multiple"):
            s1b.exact_p(0.9 + 1e-5, 1000)
        # the tolerance is far below the grid spacing, so no two distinct
        # attainable proportions can ever collide
        assert s1b.exact_p(0.900, 1000) != s1b.exact_p(0.901, 1000)
        near_002 = s1b.exact_p(0.92 - 1e-9, 200) - s1b.exact_p(0.90, 1000)
        assert near_002 == Fraction(1, 50), "recovered as exactly 0.02"

    def test_the_same_decimal_is_valid_at_one_S_and_invalid_at_another(self):
        """Grid membership is a property of (p, S), not of p alone."""
        assert s1b.exact_p(0.912, 500) == Fraction(456, 500)
        with pytest.raises(IE, match="not a multiple"):
            s1b.exact_p(0.912, 200)

    def test_exact_p_is_used_by_the_real_cell_builder(self):
        """A row whose probability is off its own 1/S grid is rejected when
        the comparison is built, not silently rounded."""
        rows = build_rows(flat_p(0.90, 0.90, 0.90))
        for r in rows:
            if r["S"] == 200 and r["candidate_id"] == CANDS[0] \
                    and r["master_seed"] == SEEDS[0]:
                r["min_on_time_prob"] = 0.9123      # 182.46 scenarios
        vg = s1b.require_complete_grid(rows, CANDS, SEEDS, S_VALUES)
        with pytest.raises(IE, match="not a multiple of 1/200"):
            s1b.build_cells(vg, 200, 1000, CANDS, SEEDS)


# ══════════════════════════════════════════════════════════════════
# E. COMPLETE GRID / ValidatedGrid CONTRACT
# ══════════════════════════════════════════════════════════════════

class TestPartE_ValidatedGrid:

    def test_complete_grid_returns_a_validated_grid(self):
        vg = grid(flat_p(0.90, 0.90, 0.90))
        assert isinstance(vg, s1b.ValidatedGrid)
        assert vg.n_cells == 300 == s1b.EXPECTED_N_ROWS
        assert list(vg.candidates) == CANDS
        assert list(vg.seeds) == SEEDS
        assert list(vg.s_values) == S_VALUES

    def test_the_certified_grid_snapshot_is_deeply_immutable(self):
        vg = grid(flat_p(0.92, 0.90, 0.90))
        key = (CANDS[0], SEEDS[0], 200)
        with pytest.raises(TypeError):
            vg.index[key] = {"min_on_time_prob": 0.10}
        with pytest.raises(TypeError):
            del vg.index[key]
        with pytest.raises(TypeError):
            vg.index[key]["min_on_time_prob"] = 0.10
        with pytest.raises(IE, match="immutable"):
            vg.n_cells = 1
        with pytest.raises(IE, match="immutable"):
            vg.candidates = []
        assert vg.index[key]["min_on_time_prob"] == 0.92

    def test_the_certified_cell_mapping_is_deeply_immutable(self):
        """A caller cannot take legitimate cells and swap one valid Fraction
        or boolean for another valid one before asking for a verdict."""
        cells = cells_of(flat_p(0.92, 0.90, 0.90))
        key = (CANDS[0], SEEDS[0])
        for mutate in (lambda: cells.__setitem__(key, {}),
                       lambda: cells.__delitem__(key),
                       lambda: cells.update({key: {}}),
                       lambda: cells.pop(key),
                       lambda: cells.popitem(),
                       lambda: cells.clear(),
                       lambda: cells.setdefault(("x", 1), {})):
            with pytest.raises(IE, match="immutable"):
                mutate()
        with pytest.raises(TypeError):       # the cell itself, not just the map
            cells[key]["dp_exact"] = Fraction(1, 4)
        with pytest.raises(TypeError):
            cells[key]["pass_S"] = False
        assert cells[key]["dp_exact"] == Fraction(1, 50)
        assert Fraction(s1b.compute_g1(cells, CANDS, SEEDS)["value_exact"]) \
            == Fraction(1, 50)

    def test_duplicates_are_retained_and_reported_never_deduplicated(self):
        rows = build_rows(flat_p(0.90, 0.90, 0.90))
        dup_key = (rows[0]["candidate_id"], rows[0]["master_seed"],
                   rows[0]["S"])
        rows.append(dict(rows[0]))
        idx, duplicates = s1b.index_rows(rows)
        assert duplicates == [dup_key], "the duplicate must be reported"
        report = s1b.grid_defect_report(idx, CANDS, SEEDS, S_VALUES, duplicates)
        assert report is not None
        assert [tuple(k) for k in report["duplicate_cells"]] == [dup_key]
        assert report["counts"]["duplicate"] == 1
        with pytest.raises(IE, match="INVALID RUN"):
            s1b.require_complete_grid(rows, CANDS, SEEDS, S_VALUES)

    def test_all_defect_classes_are_enumerated_in_one_pass(self):
        """Six simultaneous defect classes; validation must report them all,
        not abort on the first one."""
        rows = build_rows(flat_p(0.90, 0.90, 0.90))
        missing_key = ("S1-10", SEEDS[9], 1000)
        rows = [r for r in rows
                if (r["candidate_id"], r["master_seed"], r["S"]) != missing_key]
        dup_key = ("S1-01", SEEDS[0], 200)
        rows.append(make_row(*dup_key, 0.90))                  # duplicate
        rows.append(make_row("S1-99", SEEDS[0], 200, 0.90))    # unknown candidate
        rows.append(make_row("S1-01", 999999, 200, 0.90))      # unknown seed
        rows.append(make_row("S1-01", SEEDS[0], 333, 0.90))    # unknown S
        bad_p = ("S1-02", SEEDS[1], 200)
        bad_flag = ("S1-03", SEEDS[2], 500)
        for r in rows:
            key = (r["candidate_id"], r["master_seed"], r["S"])
            if key == bad_p:
                r["min_on_time_prob"] = 1.5                    # outside [0,1]
            elif key == bad_flag:
                r["chance_constraint_pass"] = "False"          # not a boolean

        idx, duplicates = s1b.index_rows(rows)
        rep = s1b.grid_defect_report(idx, CANDS, SEEDS, S_VALUES, duplicates)

        assert [tuple(k) for k in rep["missing_cells"]] == [missing_key]
        assert [tuple(k) for k in rep["duplicate_cells"]] == [dup_key]
        unexpected = {tuple(k) for k in rep["unexpected_cells"]}
        assert unexpected == {("S1-99", SEEDS[0], 200),
                              ("S1-01", 999999, 200),
                              ("S1-01", SEEDS[0], 333)}
        invalid = {tuple(e["cell"]) for e in rep["invalid_cells"]}
        assert invalid == {bad_p, bad_flag}, "both invalid cells, not just one"
        assert rep["counts"] == {"missing": 1, "duplicate": 1,
                                 "unexpected": 3, "invalid": 2}
        assert rep["expected_cells"] == 300
        assert rep["status"] == "INVALID_RUN"

    def test_one_cell_reports_every_one_of_its_own_defects(self):
        bad = make_row("S1-01", SEEDS[0], 200, 0.90)
        bad["min_on_time_prob"] = 1.5
        bad["chance_vio_total"] = -1.0
        defects = s1b.cell_defects(bad, ("S1-01", SEEDS[0], 200))
        assert len(defects) == 2
        assert any("outside [0,1]" in d for d in defects)
        assert any("negative chance_vio_total" in d for d in defects)

    # ── bypass resistance ────────────────────────────────────────────

    def test_a_fabricated_validated_grid_over_an_incomplete_index_is_refused(self):
        rows = build_rows(flat_p(0.90, 0.90, 0.90))[:-1]
        idx, _ = s1b.index_rows(rows)
        with pytest.raises(IE, match="only be constructed from a grid"):
            s1b.ValidatedGrid(idx, CANDS, SEEDS, S_VALUES)

    def test_a_fabricated_validated_grid_over_a_defective_index_is_refused(self):
        rows = build_rows(flat_p(0.90, 0.90, 0.90))
        rows[0]["chance_constraint_pass"] = "False"
        idx, _ = s1b.index_rows(rows)
        with pytest.raises(IE, match="only be constructed from a grid"):
            s1b.ValidatedGrid(idx, CANDS, SEEDS, S_VALUES)

    def test_a_fabricated_validated_grid_cannot_launder_duplicates(self):
        rows = build_rows(flat_p(0.90, 0.90, 0.90))
        idx, _ = s1b.index_rows(rows)
        dup = [("S1-01", SEEDS[0], 200)]
        with pytest.raises(IE, match="only be constructed from a grid"):
            s1b.ValidatedGrid(idx, CANDS, SEEDS, S_VALUES, duplicates=dup)

    def test_validated_grid_cannot_be_subclassed_at_all(self):
        """A subclass could validate a legitimate index and then override
        lookup to serve different rows to the gates, so the class is final and
        refuses at class-definition time."""
        with pytest.raises(IE, match="final and must not be subclassed"):
            class Sneaky(s1b.ValidatedGrid):        # noqa: F841
                def __getitem__(self, key):
                    return {"min_on_time_prob": 0.90}

    def test_mutating_the_rows_after_certification_cannot_reach_a_gate(self):
        """The certificate covers a SNAPSHOT. Rewriting the caller's rows
        after validation must not change any gate input."""
        rows = build_rows(flat_p(0.92, 0.90, 0.90))
        vg = s1b.require_complete_grid(rows, CANDS, SEEDS, S_VALUES)
        before = s1b.compute_g1(s1b.build_cells(vg, 200, 1000, CANDS, SEEDS),
                                CANDS, SEEDS)["value_exact"]
        # the caller still holds the very row objects that were certified
        for r in rows:
            r["min_on_time_prob"] = 0.50
            r["chance_vio_total"] = 0.40
            r["chance_constraint_pass"] = False
        after = s1b.compute_g1(s1b.build_cells(vg, 200, 1000, CANDS, SEEDS),
                               CANDS, SEEDS)["value_exact"]
        assert after == before == str(Fraction(1, 50))
        cells = s1b.build_cells(vg, 200, 1000, CANDS, SEEDS)
        assert cells[(CANDS[0], SEEDS[0])]["p_S"] == 0.92
        assert cells[(CANDS[0], SEEDS[0])]["pass_S"] is True

    def test_mutating_the_source_index_after_certification_is_not_seen(self):
        rows = build_rows(flat_p(0.92, 0.90, 0.90))
        idx, dups = s1b.index_rows(rows)
        vg = s1b.ValidatedGrid(idx, CANDS, SEEDS, S_VALUES, dups)
        idx[("S1-01", SEEDS[0], 200)]["min_on_time_prob"] = 0.10
        del idx[("S1-02", SEEDS[0], 200)]
        cells = s1b.build_cells(vg, 200, 1000, CANDS, SEEDS)
        assert vg.n_cells == 300
        assert cells[("S1-01", SEEDS[0])]["p_S"] == 0.92
        assert Fraction(s1b.compute_g1(cells, CANDS, SEEDS)["value_exact"]) \
            == Fraction(1, 50)

    def test_a_lookalike_object_is_not_accepted_as_a_validated_grid(self):
        class Lookalike:
            def __init__(self, index):
                self.index = index

            def __getitem__(self, key):
                return self.index[key]

        rows = build_rows(flat_p(0.90, 0.90, 0.90))
        idx, _ = s1b.index_rows(rows)
        for impostor in (Lookalike(idx), idx, list(idx.values()), None):
            with pytest.raises(IE, match="requires a ValidatedGrid"):
                s1b.build_cells(impostor, 200, 1000, CANDS, SEEDS)
            with pytest.raises(IE, match="requires a ValidatedGrid"):
                s1b.evaluate_comparison(impostor, 200, 1000, CANDS, SEEDS,
                                        gating=True, with_bootstrap=False)

    # ── gate-before-validation rejection at every gate entry point ───

    @staticmethod
    def _every_gate(cells):
        """Every public gate entry point, as callables."""
        return [("G1", lambda: s1b.compute_g1(cells, CANDS, SEEDS)),
                ("D_c", lambda: s1b.compute_d_c(cells, CANDS, SEEDS,
                                                s1b.EXPECTED_N_SEEDS)),
                ("G2dagger", lambda: s1b.compute_g2dagger(
                    cells, CANDS, SEEDS, s1b.EXPECTED_N_SEEDS)),
                ("G3", lambda: s1b.compute_g3(cells, SEEDS, OFF)),
                ("tripwire", lambda: s1b.compute_tripwire(cells, CANDS, SEEDS)),
                ("flip_stats", lambda: s1b.flip_stats(cells, CANDS, SEEDS)),
                ("A_c", lambda: s1b.compute_a_c(cells, CANDS, SEEDS))]

    def test_raw_rows_are_refused_by_every_gate_entry_point(self):
        rows = build_rows(flat_p(0.92, 0.90, 0.90))
        raw = {(r["candidate_id"], r["master_seed"]): r
               for r in rows if r["S"] == 200}
        for name, call in self._every_gate(raw):
            with pytest.raises(IE, match="requires ValidatedCells"):
                call()

    def test_forged_but_perfectly_formed_cells_are_refused_by_every_gate(self):
        """The decisive bypass test: a mapping with genuine Fractions, genuine
        booleans and the complete 100-cell shape -- indistinguishable from
        real cells by inspection -- is still refused, because what a gate
        requires is PROVENANCE, not plausibility."""
        real = cells_of(flat_p(0.90, 0.90, 0.90))
        forged = {k: dict(v) for k, v in real.items()}       # plain dict
        forged[(CANDS[0], SEEDS[0])]["dp_exact"] = Fraction(1, 2)
        assert type(forged) is dict
        assert all(isinstance(c["dp_exact"], Fraction) for c in forged.values())
        assert all(isinstance(c["pass_S"], bool) for c in forged.values())
        assert len(forged) == 100
        for name, call in self._every_gate(forged):
            with pytest.raises(IE, match="requires ValidatedCells"):
                call()

    def test_the_certified_cell_type_cannot_be_constructed_or_subclassed(self):
        real = cells_of(flat_p(0.90, 0.90, 0.90))
        assert type(real) is s1b.ValidatedCells
        with pytest.raises(IE, match="may only be produced by build_cells"):
            s1b.ValidatedCells({k: dict(v) for k, v in real.items()})
        with pytest.raises(IE, match="may only be produced by build_cells"):
            s1b.ValidatedCells(dict(real), token=object())
        with pytest.raises(IE, match="final and must not be subclassed"):
            class Forge(s1b.ValidatedCells):        # noqa: F841
                pass

    def test_float_only_cells_are_refused_by_the_probability_gates(self):
        """The gates never fall back to binary floating point.

        Reaching this defensive guard requires the module-internal
        construction token, because the two real constructors cannot emit a
        float dp and a certified mapping can no longer be edited. That is the
        documented residual limitation (see ValidatedCells' docstring): the
        provenance contract stops accidental bypass, not an in-process caller
        who deliberately reaches for module internals.
        """
        real = cells_of(flat_p(0.92, 0.90, 0.90))
        lossy = s1b.ValidatedCells(
            {k: dict(v, dp_exact=float(v["dp_exact"])) for k, v in real.items()},
            s1b._CELLS_TOKEN)
        for fn in (lambda: s1b.compute_g1(lossy, CANDS, SEEDS),
                   lambda: s1b.compute_d_c(lossy, CANDS, SEEDS),
                   lambda: s1b.compute_tripwire(lossy, CANDS, SEEDS)):
            with pytest.raises(IE, match="exact"):
                fn()

    def test_the_provenance_boundary_is_documented_not_overclaimed(self):
        """Honesty check on the contract itself. The construction token is an
        ordinary module attribute, so an in-process caller CAN forge cells --
        exactly as it could rebind a threshold or a gate function. The class
        must therefore document that it is an engineering guarantee against
        mistakes, not a security boundary, rather than claiming more."""
        doc = s1b.ValidatedCells.__doc__
        assert "STATED LIMITATION" in doc
        assert "NOT a security boundary" in doc
        assert "rebind module attributes" in doc

    def test_ladder_refuses_to_decide_without_gating_comparisons(self):
        vg = grid(flat_p(0.90, 0.90, 0.90))
        diag = s1b.evaluate_comparison(vg, 200, 500, CANDS, SEEDS,
                                       gating=False, with_bootstrap=False)
        assert diag["gate_verdict"] == "NOT_A_GATE"
        with pytest.raises(IE, match="missing gating comparison"):
            s1b.decide_ladder([diag])
        with pytest.raises(IE, match="missing gating comparison"):
            s1b.decide_ladder([])

    def test_ladder_refuses_a_gate_against_the_wrong_reference(self):
        vg = grid(flat_p(0.90, 0.90, 0.90))
        wrong = s1b.evaluate_comparison(vg, 200, 500, CANDS, SEEDS,
                                        gating=True, with_bootstrap=False)
        with pytest.raises(IE, match=r"S_ref=500|must use S_ref"):
            s1b.decide_ladder([wrong])


# ══════════════════════════════════════════════════════════════════
# F. VALIDATION_FAILURE.json
# ══════════════════════════════════════════════════════════════════

class TestPartF_ValidationFailureArtifact:

    @staticmethod
    def _defective_rows():
        rows = build_rows(flat_p(0.90, 0.90, 0.90))
        rows = [r for r in rows
                if (r["candidate_id"], r["master_seed"], r["S"])
                != ("S1-10", SEEDS[9], 1000)]
        rows.append(dict(rows[0]))
        rows.append(make_row("S1-77", SEEDS[0], 200, 0.90))
        rows[5]["min_on_time_prob"] = float("nan")
        return rows

    def test_validation_failure_artifact_is_written_with_every_category(self, tmp_path):
        out = tmp_path / "results"
        with pytest.raises(IE, match="INVALID RUN"):
            s1b.require_complete_grid(self._defective_rows(), CANDS, SEEDS,
                                      S_VALUES, out_dir=out)
        target = out / "VALIDATION_FAILURE.json"
        assert target.exists()
        rep = json.loads(target.read_text(encoding="utf-8"))
        assert rep["status"] == "INVALID_RUN"
        assert rep["stage"] == "A-lite Stage 1B"
        assert rep["expected_cells"] == 300
        for key in ("missing_cells", "duplicate_cells", "unexpected_cells",
                    "invalid_cells", "counts", "rule"):
            assert key in rep and rep[key], f"{key} missing or empty"
        assert rep["counts"] == {"missing": 1, "duplicate": 1,
                                 "unexpected": 1, "invalid": 1}
        assert "never reduced" in rep["rule"]
        assert "Stage 2 must not start" in rep["rule"]

    def test_no_scientific_artifact_is_written_on_an_invalid_run(self, tmp_path):
        out = tmp_path / "results"
        with pytest.raises(IE):
            s1b.require_complete_grid(self._defective_rows(), CANDS, SEEDS,
                                      S_VALUES, out_dir=out)
        assert {p.name for p in out.iterdir()} == {"VALIDATION_FAILURE.json"}
        for forbidden in ("STAGE1B_REPORT.md", "manifest.json",
                          "comparisons.json", "ladder_decision.json",
                          "per_candidate_diagnostics.json", "raw_rows.json"):
            assert not (out / forbidden).exists()

    def test_an_invalid_run_aborts_before_any_gate_is_computed(self, tmp_path, monkeypatch):
        fired = []
        for name in ("compute_g1", "compute_g2dagger", "compute_g3",
                     "compute_tripwire", "build_cells", "decide_ladder"):
            monkeypatch.setattr(s1b, name,
                                lambda *a, _n=name, **k: fired.append(_n))
        out = tmp_path / "results"
        with pytest.raises(IE, match="INVALID RUN"):
            vg = s1b.require_complete_grid(self._defective_rows(), CANDS,
                                           SEEDS, S_VALUES, out_dir=out)
            s1b.evaluate_comparison(vg, 200, 1000, CANDS, SEEDS, gating=True)
        assert fired == [], f"gate machinery ran on an invalid run: {fired}"

    def test_write_outputs_refuses_to_overwrite_a_non_empty_directory(self, tmp_path):
        out = tmp_path / "results"
        out.mkdir()
        (out / "previous_result.json").write_text("{}", encoding="utf-8")
        with pytest.raises(IE, match="already exists and is non-empty"):
            s1b.write_outputs([], [], {}, {}, {}, out)
        assert {p.name for p in out.iterdir()} == {"previous_result.json"}


# ══════════════════════════════════════════════════════════════════
# G. FIXED DENOMINATORS -- 10 / 100 / 50 / 100, never reduced
# ══════════════════════════════════════════════════════════════════

class TestPartG_FixedDenominators:

    def test_the_four_frozen_denominators(self):
        cells = cells_of(flip_grid(4))
        g1 = s1b.compute_g1(cells, CANDS, SEEDS)
        g2 = s1b.compute_g2dagger(cells, CANDS, SEEDS, s1b.EXPECTED_N_SEEDS)
        g3 = s1b.compute_g3(cells, SEEDS, OFF)
        trip = s1b.compute_tripwire(cells, CANDS, SEEDS)
        assert g1["denominator"] == 100 == s1b.EXPECTED_N_CELLS
        assert g2["seed_denominator"] == 10 == s1b.EXPECTED_N_SEEDS
        assert g3["denominator"] == 50 == len(OFF) * len(SEEDS)
        assert trip["denominator"] == 100

    def test_removing_one_cell_invalidates_the_run_instead_of_shrinking_a_denominator(self):
        rows = build_rows(flat_p(0.90, 0.90, 0.90))
        dropped = rows.pop()
        key = (dropped["candidate_id"], dropped["master_seed"], dropped["S"])
        with pytest.raises(IE, match="INVALID RUN") as ctx:
            s1b.require_complete_grid(rows, CANDS, SEEDS, S_VALUES)
        assert "never reduced" in str(ctx.value)
        # and the certified container cannot be fabricated around the gap
        idx, _ = s1b.index_rows(rows)
        with pytest.raises(IE):
            s1b.ValidatedGrid(idx, CANDS, SEEDS, S_VALUES)
        rep = s1b.grid_defect_report(idx, CANDS, SEEDS, S_VALUES)
        assert [tuple(k) for k in rep["missing_cells"]] == [key]

    def test_a_missing_paired_cell_is_refused_by_the_probability_gates(self):
        # Genuinely never constructed, not deleted afterwards: a certified
        # mapping is immutable, so the gap has to be real.
        spec = {c: [Fraction(0)] * 10 for c in CANDS}
        spec[CANDS[0]] = spec[CANDS[0]][:-1]
        cells = s1b._synthetic_cells(spec, SEEDS)
        assert len(cells) == 99
        with pytest.raises(IE, match="never reduced"):
            s1b.compute_g1(cells, CANDS, SEEDS)
        with pytest.raises(IE, match="never reduced"):
            s1b.compute_d_c(cells, CANDS, SEEDS)
        with pytest.raises(IE, match="never reduced"):
            s1b.compute_tripwire(cells, CANDS, SEEDS)
        with pytest.raises(IE, match="never reduced"):
            s1b.flip_stats(cells, CANDS, SEEDS)

    def test_g1_denominator_is_the_declared_grid_not_the_rows_present(self):
        """With nine seeds the mean would be over 90 cells. The frozen G1
        denominator is 100, so the nine-seed shape must never reach a
        verdict -- it is rejected upstream by the grid precheck."""
        rows = build_rows(flat_p(0.90, 0.90, 0.90), seeds=SEEDS[:9])
        with pytest.raises(IE, match="INVALID RUN"):
            s1b.require_complete_grid(rows, CANDS, SEEDS, S_VALUES)

    def test_g3_denominator_is_fifty_even_when_no_off_cell_flips(self):
        g3 = s1b.compute_g3(cells_of(flip_grid(0)), SEEDS, OFF)
        assert g3["denominator"] == 50
        assert Fraction(g3["value_exact"]) == 0

    # ── direct calls with complete-looking but REDUCED inputs ────────

    def test_every_gate_refuses_a_reduced_dimension_directly(self):
        """Not only upstream: each gate itself refuses to produce a verdict on
        90 / 9 / 40 cells when the frozen dimension is asserted, so a caller
        cannot obtain a 99/49/9-style denominator by calling the gate API."""
        vg = grid(flat_p(0.90, 0.90, 0.90))
        nine_seeds, nine_cands = SEEDS[:9], CANDS[:9]
        four_off = OFF[:4]

        cells9s = s1b.build_cells(vg, 200, 1000, CANDS, nine_seeds)
        with pytest.raises(IE, match="exactly 100 cells"):
            s1b.compute_g1(cells9s, CANDS, nine_seeds, s1b.EXPECTED_N_CELLS)
        with pytest.raises(IE, match="exactly 100 cells"):
            s1b.compute_tripwire(cells9s, CANDS, nine_seeds,
                                 s1b.EXPECTED_N_CELLS)
        with pytest.raises(IE, match="exactly 10 master"):
            s1b.compute_d_c(cells9s, CANDS, nine_seeds, s1b.EXPECTED_N_SEEDS)

        cells9c = s1b.build_cells(vg, 200, 1000, nine_cands, SEEDS)
        with pytest.raises(IE, match="exactly 100 cells"):
            s1b.compute_g1(cells9c, nine_cands, SEEDS, s1b.EXPECTED_N_CELLS)
        with pytest.raises(IE, match="exactly 10 .*candidates"):
            s1b.compute_g2dagger(cells9c, nine_cands, SEEDS,
                                 s1b.EXPECTED_N_SEEDS,
                                 s1b.EXPECTED_N_CANDIDATES)

        cells_off4 = s1b.build_cells(vg, 200, 1000, four_off, SEEDS)
        with pytest.raises(IE, match="exactly 50 OFF-stratum cells"):
            s1b.compute_g3(cells_off4, SEEDS, four_off, 50)
        with pytest.raises(IE, match="exactly 50 OFF-stratum cells"):
            s1b.compute_g3(cells9s, nine_seeds, OFF, 50)

    def test_a_full_run_comparison_asserts_every_frozen_dimension(self):
        """evaluate_comparison in full-run mode (n_seeds_expected given) pins
        100 cells, 10 seeds, 10 candidates and 50 OFF cells at once."""
        vg = grid(flat_p(0.90, 0.90, 0.90))
        for cands, seeds in ((CANDS[:9], SEEDS), (CANDS, SEEDS[:9])):
            with pytest.raises(IE, match="exactly (100 cells|10 )"):
                s1b.evaluate_comparison(
                    vg, 200, 1000, cands, seeds, gating=True,
                    n_seeds_expected=s1b.EXPECTED_N_SEEDS,
                    with_bootstrap=False)
        # the full declared shape is accepted
        comp = s1b.evaluate_comparison(
            vg, 200, 1000, CANDS, SEEDS, gating=True,
            n_seeds_expected=s1b.EXPECTED_N_SEEDS, with_bootstrap=False)
        assert comp["gates"]["G1"]["denominator"] == 100
        assert comp["gates"]["G3"]["denominator"] == 50
        assert comp["gates"]["G2dagger"]["seed_denominator"] == 10


# ══════════════════════════════════════════════════════════════════
# H. NON-GATING ISOLATION AND BOOLEAN AGGREGATION
# ══════════════════════════════════════════════════════════════════

def truth_grid(fail_g1, fail_g2, fail_g3):
    """A grid that fails exactly the requested subset of {G1, G2dagger, G3}.

    fail_g1 : per-seed +-0.02 around a raised reference -> mean |dp| == 0.02
              exactly, signed mean 0, and every probability above alpha so no
              classification flips are induced.
    fail_g2 : one candidate carries MIXED_DPS_MEAN_002 -> D_c == 0.02 exactly.
    fail_g3 : five OFF cells rejected at the reference and accepted at S
              -> 5/50 == 0.10 exactly.
    """
    base = 0.95 if fail_g1 else 0.90
    flipped = {(OFF[k], 0) for k in range(5)} if fail_g3 else set()

    def p_fn(c, i, S):
        if S == 1000 and (c, i) in flipped:
            return 0.89                       # below alpha -> rejected at ref
        if S != 200:
            return base
        if fail_g2 and c == "S1-01":
            return round(base + MIXED_DPS_MEAN_002[i], 6)
        if fail_g1:
            return round(base + (0.02 if i % 2 == 0 else -0.02), 6)
        return base
    return p_fn


TRUTH_TABLE = [(a, b, c) for a in (False, True)
               for b in (False, True) for c in (False, True)]


class TestPartH_AggregationAndNonGating:

    @pytest.mark.parametrize("fail_g1,fail_g2,fail_g3", TRUTH_TABLE)
    def test_verdict_is_exactly_the_conjunction_of_the_three_gates(
            self, fail_g1, fail_g2, fail_g3):
        vg = grid(truth_grid(fail_g1, fail_g2, fail_g3))
        comp = s1b.evaluate_comparison(vg, 200, 1000, CANDS, SEEDS,
                                       gating=True,
                                       n_seeds_expected=s1b.EXPECTED_N_SEEDS,
                                       with_bootstrap=False)
        g = comp["gates"]
        assert g["G1"]["passed"] is not fail_g1
        assert g["G2dagger"]["passed"] is not fail_g2
        assert g["G3"]["passed"] is not fail_g3
        expected = "PASS" if not (fail_g1 or fail_g2 or fail_g3) else "FAIL"
        assert comp["gate_verdict"] == expected
        assert comp["indeterminate_category_exists"] is False
        assert all(gate["gating"] is True for gate in g.values())

    def test_each_single_gate_failure_is_sufficient_to_fail(self):
        for triple in [(True, False, False), (False, True, False),
                       (False, False, True)]:
            vg = grid(truth_grid(*triple))
            comp = s1b.evaluate_comparison(vg, 200, 1000, CANDS, SEEDS,
                                           gating=True, with_bootstrap=False)
            assert comp["gate_verdict"] == "FAIL", triple
            failing = [n for n, gate in comp["gates"].items()
                       if not gate["passed"]]
            assert len(failing) == 1, (triple, failing)

    # ── Case A: gates pass while every diagnostic screams ────────────

    @staticmethod
    def _case_a_grid():
        """All three gates pass; tripwire tripped, KNIFE flip rate 1.0,
        all-pair flip rate 0.5, and the S500 comparison catastrophic."""
        def p_fn(c, i, S):
            if S == 200:
                return 0.905 if c in KNIFE else 0.915
            if S == 500:
                return 0.80                      # ruins S500 and S200-vs-S500
            return 0.89 if c in KNIFE else 0.90   # KNIFE rejected at reference
        return p_fn

    def _case_a_comparisons(self, with_bootstrap=False):
        vg = grid(self._case_a_grid())
        c200 = s1b.evaluate_comparison(vg, 200, 1000, CANDS, SEEDS, gating=True,
                                       n_seeds_expected=s1b.EXPECTED_N_SEEDS,
                                       with_bootstrap=with_bootstrap)
        c500 = s1b.evaluate_comparison(vg, 500, 1000, CANDS, SEEDS, gating=True,
                                       n_seeds_expected=s1b.EXPECTED_N_SEEDS,
                                       with_bootstrap=False)
        cdiag = s1b.evaluate_comparison(vg, 200, 500, CANDS, SEEDS,
                                        gating=False, with_bootstrap=False)
        return [c200, c500, cdiag]

    def test_case_a_extreme_diagnostics_do_not_prevent_a_pass(self):
        c200, c500, cdiag = self._case_a_comparisons()
        g = c200["gates"]
        assert Fraction(g["G1"]["value_exact"]) == Fraction(3, 200)   # 0.015
        assert Fraction(g["G2dagger"]["value_exact"]) == Fraction(3, 200)
        assert g["G3"]["flip_count"] == 0
        assert c200["gate_verdict"] == "PASS"

        d = c200["non_gating"]["diagnostics"]
        trip = c200["non_gating"]["optimism_tripwire"]
        assert trip["tripped"] is True and trip["direction"] == "optimism"
        assert d["all_pair_flips"]["gross_flip_rate"] == 0.5
        assert d["KNIFE_flips"]["gross_flip_rate"] == 1.0
        assert d["OFF_flips"]["gross_flip_rate"] == 0.0
        assert c500["gate_verdict"] == "FAIL"          # would-be alternative
        assert cdiag["gate_verdict"] == "NOT_A_GATE"

        ladder = s1b.decide_ladder([c200, c500, cdiag])
        assert ladder["selected_S"] == 200
        assert ladder["stage2_may_proceed"] is True
        assert ladder["s500_gate_consulted"] is False

    def test_a_tripped_tripwire_inside_a_passing_gate_is_disclosed_not_gating(self):
        c200 = self._case_a_comparisons()[0]
        assert c200["tripwire_tripped_while_gate_passed"] is True
        assert "MANDATORY_STAGE2_LIMITATION" in c200
        assert "non-blocking" in c200["MANDATORY_STAGE2_LIMITATION"]
        assert c200["gate_verdict"] == "PASS"

    def test_a_tripped_tripwire_in_a_non_scientific_run_creates_no_obligation(self):
        """A sanity run has no accepted operating point, no valid ladder
        decision and no verdict to qualify, so a tripped tripwire there must
        NOT emit the Stage-2 disclosure obligation. It gets an explicit
        do-not-quote note instead, so an uninterpretable number cannot be
        quoted as a real limitation."""
        vg = grid(self._case_a_grid())
        sanity = s1b.evaluate_comparison(vg, 200, 1000, CANDS, SEEDS,
                                         gating=True, with_bootstrap=False,
                                         scientific=False)
        assert sanity["tripwire_tripped_while_gate_passed"] is True
        assert "MANDATORY_STAGE2_LIMITATION" not in sanity
        note = sanity["SANITY_TRIPWIRE_NOTE"]
        assert "MUST NOT be quoted" in note
        assert "MUST NOT be carried forward as a Stage-2 limitation" in note

        # the same comparison in a scientific run DOES create the obligation
        real = s1b.evaluate_comparison(vg, 200, 1000, CANDS, SEEDS,
                                       gating=True,
                                       n_seeds_expected=s1b.EXPECTED_N_SEEDS,
                                       with_bootstrap=False, scientific=True)
        assert "MANDATORY_STAGE2_LIMITATION" in real
        assert "SANITY_TRIPWIRE_NOTE" not in real

    def test_scientific_defaults_to_the_full_run_signal(self):
        """Omitting `scientific` must not silently grant a reduced run the
        authority to create a Stage-2 obligation."""
        vg = grid(self._case_a_grid())
        implied_sanity = s1b.evaluate_comparison(
            vg, 200, 1000, CANDS, SEEDS, gating=True, with_bootstrap=False)
        assert "MANDATORY_STAGE2_LIMITATION" not in implied_sanity
        assert "SANITY_TRIPWIRE_NOTE" in implied_sanity
        implied_full = s1b.evaluate_comparison(
            vg, 200, 1000, CANDS, SEEDS, gating=True,
            n_seeds_expected=s1b.EXPECTED_N_SEEDS, with_bootstrap=False)
        assert "MANDATORY_STAGE2_LIMITATION" in implied_full

    def test_the_sanity_note_is_rendered_and_marked_do_not_quote(self):
        vg = grid(self._case_a_grid())
        comps = [s1b.evaluate_comparison(vg, S, 1000, CANDS, SEEDS,
                                         gating=True, with_bootstrap=False,
                                         scientific=False)
                 for S in (200, 500)]
        md = s1b.render_report(comps, {"selected_S": None,
                                       "stage2_may_proceed": False,
                                       "rationale": "SANITY RUN"},
                               {"mode": "sanity-run", "sanity_run": True})
        assert "TRIPWIRE FIRED IN A NON-SCIENTIFIC RUN — DO NOT QUOTE" in md
        assert "TRIPWIRE TRIPPED WHILE GATE PASSED" not in md
        assert "carried explicitly as a Stage-2 limitation" not in md

    # ── mandatory stated-limitation disclosures (sections 5, 7.1, 7.3) ──

    def test_an_argmax_tie_is_disclosed_and_never_changes_the_gate(self):
        """max_c D_c can be attained by several candidates. The deterministic
        first-in-frozen-order argmax is kept, but the tie must be reported so
        it is not read as a unique worst candidate."""
        dps = MIXED_DPS_MEAN_00195
        def p_fn(c, i, S):
            if S == 200 and c in ("S1-04", "S1-07"):
                return round(0.90 + dps[i], 6)
            return 0.90
        cells = cells_of(p_fn)
        g2 = s1b.compute_g2dagger(cells, CANDS, SEEDS, s1b.EXPECTED_N_SEEDS)
        assert g2["argmax_is_tied"] is True
        assert g2["argmax_candidates"] == ["S1-04", "S1-07"]
        assert g2["argmax_candidate"] == "S1-04"       # first in frozen order
        assert Fraction(g2["value_exact"]) == Fraction(39, 2000)
        assert g2["passed"] is True                    # tie changes nothing

        vg = grid(p_fn)
        comp = s1b.evaluate_comparison(vg, 200, 1000, CANDS, SEEDS, gating=True,
                                       n_seeds_expected=s1b.EXPECTED_N_SEEDS,
                                       with_bootstrap=False, scientific=True)
        assert any("attained by MORE THAN ONE candidate" in L
                   for L in comp["stated_limitations"])
        assert comp["gate_verdict"] == "PASS"

    def test_a_unique_argmax_reports_no_tie(self):
        cells = cells_of(d_c_grid(MIXED_DPS_MEAN_00195))
        g2 = s1b.compute_g2dagger(cells, CANDS, SEEDS, s1b.EXPECTED_N_SEEDS)
        assert g2["argmax_is_tied"] is False
        assert g2["argmax_candidates"] == [TARGET]

    def test_statistically_identical_candidates_are_detected_and_disclosed(self):
        """Two structurally distinct candidates can produce an identical p_hat
        series. That lowers effective panel diversity and must be disclosed --
        but must NOT reduce any denominator or drop a candidate."""
        vg = grid(flat_p(0.915, 0.90, 0.90))       # all 10 identical by design
        comp = s1b.evaluate_comparison(vg, 200, 1000, CANDS, SEEDS, gating=True,
                                       n_seeds_expected=s1b.EXPECTED_N_SEEDS,
                                       with_bootstrap=False, scientific=True)
        d = comp["non_gating"]["diagnostics"]
        assert d["n_candidates"] == 10
        assert d["n_distinct_probability_series"] == 1
        assert len(d["statistically_identical_candidate_pairs"]) == 45
        assert any("identical p_hat series" in L
                   for L in comp["stated_limitations"])
        assert any("no denominator is reduced" in L
                   for L in comp["stated_limitations"])
        # denominators are untouched by the disclosure
        assert comp["gates"]["G1"]["denominator"] == 100
        assert comp["gates"]["G2dagger"]["seed_denominator"] == 10
        assert comp["gates"]["G3"]["denominator"] == 50

    def test_a_fully_diverse_panel_reports_no_identical_pairs(self):
        def p_fn(c, i, S):
            return 0.90 if S != 200 else round(0.90 + 0.005 * CANDS.index(c), 6)
        comp = s1b.evaluate_comparison(grid(p_fn), 200, 1000, CANDS, SEEDS,
                                       gating=True, with_bootstrap=False,
                                       n_seeds_expected=s1b.EXPECTED_N_SEEDS,
                                       scientific=True)
        d = comp["non_gating"]["diagnostics"]
        assert d["statistically_identical_candidate_pairs"] == []
        assert d["n_distinct_probability_series"] == 10
        assert not any("identical p_hat series" in L
                       for L in comp["stated_limitations"])

    def test_a_straddling_bootstrap_band_is_disclosed_but_never_gates(self):
        """Amendment 7.1: a band straddling a threshold is a stated
        limitation; the verdict stands exactly as computed."""
        comp = dict(self._case_a_comparisons(with_bootstrap=False)[0])
        comp["non_gating"] = dict(comp["non_gating"])
        comp["non_gating"]["bootstrap"] = {
            "G1_interval": {"lo": 0.010, "hi": 0.030, "median": 0.015},
            "G2dagger_interval": {"lo": 0.010, "hi": 0.030, "median": 0.015},
            "OFF_flip_rate_interval": {"lo": 0.04, "hi": 0.12, "median": 0.08},
        }
        lims = s1b.stated_limitations(comp)
        assert sum("STRADDLES" in L for L in lims) == 3
        assert all("stands exactly as computed" in L
                   for L in lims if "STRADDLES" in L)
        assert comp["gate_verdict"] == "PASS"       # untouched

    def test_a_loco_refit_that_would_flip_a_passing_gate_is_disclosed(self):
        comp = dict(self._case_a_comparisons(with_bootstrap=False)[0])
        assert comp["gates"]["G3"]["passed"] is True
        comp["non_gating"] = dict(comp["non_gating"])
        comp["non_gating"]["loco"] = {"per_omitted_candidate": {
            "S1-02": {"G1_value": 0.001, "G2dagger_value": 0.001,
                      "OFF_flip_rate": 0.10},        # crosses -> hinge
            "S1-03": {"G1_value": 0.001, "G2dagger_value": 0.001,
                      "OFF_flip_rate": 0.02}}}       # stays below -> not
        lims = s1b.stated_limitations(comp)
        hit = [L for L in lims if "Leave-one-candidate-out" in L]
        assert len(hit) == 1 and "S1-02" in hit[0] and "S1-03" not in hit[0]
        assert "from PASS to FAIL" in hit[0] and "DESCRIPTIVE" in hit[0]

    def test_a_loco_refit_that_flips_nothing_on_a_failing_gate_is_not_a_hinge(self):
        """Over-generalisation guard: for an ALREADY-FAILING component, a LOCO
        refit that is still above the threshold demonstrates nothing and must
        NOT be reported as a hinge. Only a refit that would cross counts."""
        vg = grid(truth_grid(False, False, True))          # G3 fails: 5/50
        comp = s1b.evaluate_comparison(vg, 200, 1000, CANDS, SEEDS, gating=True,
                                       n_seeds_expected=s1b.EXPECTED_N_SEEDS,
                                       with_bootstrap=False, scientific=True)
        assert comp["gates"]["G3"]["passed"] is False
        comp["non_gating"] = dict(comp["non_gating"])
        comp["non_gating"]["loco"] = {"per_omitted_candidate": {
            "S1-07": {"G1_value": 0.001, "G2dagger_value": 0.001,
                      "OFF_flip_rate": 0.125},       # still failing -> NOT a hinge
            "S1-04": {"G1_value": 0.001, "G2dagger_value": 0.001,
                      "OFF_flip_rate": 0.00}}}       # would rescue -> hinge
        lims = s1b.stated_limitations(comp)
        hit = [L for L in lims if "Leave-one-candidate-out" in L]
        assert len(hit) == 1, lims
        assert "S1-04" in hit[0] and "S1-07" not in hit[0]
        assert "from FAIL to PASS" in hit[0]

    def test_a_non_scientific_run_asserts_no_stated_limitation(self):
        """A stated limitation is a Stage-2 disclosure obligation, so a run
        that produces no verdict must not emit one -- otherwise the sanity
        leak the tripwire note prevents would reappear through this channel."""
        vg = grid(self._case_a_grid())
        sanity = s1b.evaluate_comparison(vg, 200, 1000, CANDS, SEEDS,
                                         gating=True, with_bootstrap=True,
                                         scientific=False)
        assert sanity["stated_limitations_are_scientific"] is False
        lims = sanity["stated_limitations"]
        assert len(lims) == 1
        assert lims[0].startswith("NON-SCIENTIFIC RUN -- DO NOT QUOTE")
        assert "must never be carried forward" in lims[0]
        for banned in ("MANDATORY", "carry forward and", "stands exactly as computed"):
            assert not any(banned in L for L in lims), banned

        real = s1b.evaluate_comparison(vg, 200, 1000, CANDS, SEEDS, gating=True,
                                       n_seeds_expected=s1b.EXPECTED_N_SEEDS,
                                       with_bootstrap=True, scientific=True)
        assert real["stated_limitations_are_scientific"] is True
        assert not any("DO NOT QUOTE" in L for L in real["stated_limitations"])

    def test_a_non_scientific_report_does_not_claim_mandatory_disclosure(self):
        vg = grid(self._case_a_grid())
        comps = [s1b.evaluate_comparison(vg, S, 1000, CANDS, SEEDS, gating=True,
                                         with_bootstrap=True, scientific=False)
                 for S in (200, 500)]
        md = s1b.render_report(comps, {"selected_S": None,
                                       "stage2_may_proceed": False,
                                       "rationale": "SANITY RUN"},
                               {"mode": "sanity-run", "sanity_run": True})
        assert "Stated limitations — NOT APPLICABLE (non-scientific run)" in md
        assert "MANDATORY disclosure" not in md
        assert "NON-SCIENTIFIC RUN -- DO NOT QUOTE" in md

    def test_a_clean_comparison_states_that_nothing_was_triggered(self):
        """A genuinely clean panel: ten DISTINCT probability series, a unique
        argmax, and every statistic far from its threshold."""
        def p_fn(c, i, S):
            # p_ref differs per candidate on the 1/1000 grid -> all distinct
            return round(0.90 + 0.001 * CANDS.index(c), 6) if S == 1000 else 0.90
        vg = grid(p_fn)
        comp = s1b.evaluate_comparison(vg, 200, 1000, CANDS, SEEDS, gating=True,
                                       n_seeds_expected=s1b.EXPECTED_N_SEEDS,
                                       with_bootstrap=True, scientific=True)
        assert comp["gate_verdict"] == "PASS"
        d = comp["non_gating"]["diagnostics"]
        assert d["n_distinct_probability_series"] == 10
        assert comp["gates"]["G2dagger"]["argmax_is_tied"] is False
        assert comp["stated_limitations"] == []
        md = s1b.render_report([comp], {"selected_S": 200,
                                        "stage2_may_proceed": True,
                                        "rationale": "x"},
                               {"mode": "full-run"})
        assert "### Stated limitations (MANDATORY disclosure, non-gating)" in md
        assert "None triggered for this comparison" in md

    def test_stated_limitations_cannot_alter_any_gate_or_the_ladder(self):
        """The disclosures are computed from a finished comparison and are
        purely additive."""
        comps = self._case_a_comparisons(with_bootstrap=True)
        baseline_ladder = s1b.decide_ladder(copy.deepcopy(comps))
        baseline_gates = [copy.deepcopy(c["gates"]) for c in comps]
        for c in comps:
            assert isinstance(c["stated_limitations"], list)
        sabotaged = copy.deepcopy(comps)
        for c in sabotaged:
            c["stated_limitations"] = ["FABRICATED " * 5] * 20
        assert s1b.decide_ladder(sabotaged) == baseline_ladder
        assert [c["gates"] for c in sabotaged] == baseline_gates

    def test_mutating_every_non_gating_field_cannot_change_the_ladder(self):
        comps = self._case_a_comparisons(with_bootstrap=True)
        baseline = s1b.decide_ladder(copy.deepcopy(comps))
        sabotaged = copy.deepcopy(comps)
        for comp in sabotaged:
            ng = comp["non_gating"]
            ng["optimism_tripwire"]["tripped"] = True
            ng["optimism_tripwire"]["value"] = 99.0
            ng["loco"]["per_omitted_candidate"] = {"S1-01": {"G1_value": 99.0}}
            ng["bootstrap"] = {"G1_interval": [99.0, 99.0],
                               "G2dagger_interval": [99.0, 99.0]}
            ng["diagnostics"]["all_pair_flips"]["gross_flip_rate"] = 1.0
            ng["diagnostics"]["KNIFE_flips"]["gross_flip_rate"] = 1.0
            ng["sqrt_rate_check"]["observed_mean_abs_dp"] = 99.0
            ng["ratio_matched_scale_invariance"]["this_comparison"] = None
        assert s1b.decide_ladder(sabotaged) == baseline

    def test_the_ladder_reads_nothing_at_all_from_the_non_gating_block(self):
        """Stronger than mutation: DELETING every non-gating statistic leaves
        the ladder decision identical, so no diagnostic is an input at all."""
        comps = self._case_a_comparisons(with_bootstrap=True)
        baseline = s1b.decide_ladder(copy.deepcopy(comps))
        stripped = copy.deepcopy(comps)
        for comp in stripped:
            del comp["non_gating"]
            comp.pop("tripwire_tripped_while_gate_passed", None)
            comp.pop("MANDATORY_STAGE2_LIMITATION", None)
        assert s1b.decide_ladder(stripped) == baseline
        assert baseline["selected_S"] == 200

    def test_the_diagnostic_only_pair_can_never_gate(self):
        """S200-vs-S500 fails all three gate statistics and is still ignored."""
        cdiag = self._case_a_comparisons()[2]
        assert cdiag["is_gating_comparison"] is False
        assert cdiag["gate_verdict"] == "NOT_A_GATE"
        assert not cdiag["gates"]["G1"]["passed"]
        c200, c500 = self._case_a_comparisons()[:2]
        assert s1b.decide_ladder([c200, c500, cdiag])["selected_S"] == 200
        assert s1b.decide_ladder([c200, c500])["selected_S"] == 200

    # ── Case B: diagnostics benign, one gate fails ───────────────────

    @pytest.mark.parametrize("triple,failing", [
        ((True, False, False), "G1"),
        ((False, True, False), "G2dagger"),
        ((False, False, True), "G3"),
    ])
    def test_case_b_a_single_gate_failure_blocks_despite_benign_diagnostics(
            self, triple, failing):
        vg = grid(truth_grid(*triple))
        comp = s1b.evaluate_comparison(vg, 200, 1000, CANDS, SEEDS, gating=True,
                                       with_bootstrap=False)
        trip = comp["non_gating"]["optimism_tripwire"]
        assert trip["tripped"] is False, "diagnostics must be benign here"
        assert comp["gates"][failing]["passed"] is False
        assert comp["gate_verdict"] == "FAIL"
        assert "MANDATORY_STAGE2_LIMITATION" not in comp

    def test_both_gating_comparisons_failing_blocks_stage_two(self):
        """ONE real grid in which the five OFF cells are rejected at the
        reference and accepted at both S200 and S500, so G3 == 0.10 fails
        BOTH gating comparisons. No comparison object is hand-assembled."""
        vg = grid(truth_grid(False, False, True))
        comps = [s1b.evaluate_comparison(vg, S, 1000, CANDS, SEEDS, gating=True,
                                         n_seeds_expected=s1b.EXPECTED_N_SEEDS,
                                         with_bootstrap=False)
                 for S in (200, 500)]
        for comp in comps:
            assert Fraction(comp["gates"]["G3"]["value_exact"]) == Fraction(1, 10)
            assert comp["gate_verdict"] == "FAIL"
        ladder = s1b.decide_ladder(comps)
        assert ladder["selected_S"] is None
        assert ladder["stage2_may_proceed"] is False
        assert ladder["s500_gate_consulted"] is True
        assert "NOT declared sufficient" in ladder["rationale"]

    def test_s500_is_selected_only_when_s200_fails(self):
        """ONE real grid: the D_c == 0.02 excursion exists only at S200, so
        S200 fails G2dagger while S500 passes all three gates."""
        vg = grid(truth_grid(False, True, False))
        c200 = s1b.evaluate_comparison(vg, 200, 1000, CANDS, SEEDS, gating=True,
                                       n_seeds_expected=s1b.EXPECTED_N_SEEDS,
                                       with_bootstrap=False)
        c500 = s1b.evaluate_comparison(vg, 500, 1000, CANDS, SEEDS, gating=True,
                                       n_seeds_expected=s1b.EXPECTED_N_SEEDS,
                                       with_bootstrap=False)
        assert c200["gate_verdict"] == "FAIL"
        assert c200["gates"]["G2dagger"]["passed"] is False
        assert c500["gate_verdict"] == "PASS"
        ladder = s1b.decide_ladder([c200, c500])
        assert ladder["selected_S"] == 500
        assert ladder["stage2_may_proceed"] is True
        assert ladder["s500_gate_consulted"] is True

    # ── every non-gating statistic declares itself non-gating ────────

    def test_every_non_gating_statistic_is_flagged_non_gating(self):
        c200 = self._case_a_comparisons(with_bootstrap=True)[0]
        ng = c200["non_gating"]
        assert ng["optimism_tripwire"]["gating"] is False
        assert ng["optimism_tripwire"]["role"].startswith("NON_GATING")
        assert ng["diagnostics"]["gating"] is False
        assert ng["diagnostics"]["role"] == "DESCRIPTIVE_ONLY"
        assert ng["loco"]["gating"] is False
        assert ng["loco"]["role"] == "DESCRIPTIVE_ONLY"
        assert ng["bootstrap"]["gating"] is False
        assert ng["bootstrap"]["role"] == "DESCRIPTIVE_ONLY"
        assert ng["sqrt_rate_check"]["gating"] is False
        assert ng["ratio_matched_scale_invariance"]["gating"] is False
        assert ng["tripwire_decomposition"]["gating"] is False
        assert len(ng["loco"]["per_omitted_candidate"]) == 10

    def test_a_gate_that_loses_its_gating_flag_is_refused(self):
        """The verdict is derived only from statistics carrying gating=True,
        and that invariant is asserted at runtime."""
        vg = grid(truth_grid(False, False, False))
        real = s1b.compute_g1

        def stripped(*a, **k):
            out = real(*a, **k)
            out["gating"] = False
            return out
        try:
            s1b.compute_g1 = stripped
            with pytest.raises(IE, match="lost its gating flag"):
                s1b.evaluate_comparison(vg, 200, 1000, CANDS, SEEDS,
                                        gating=True, with_bootstrap=False)
        finally:
            s1b.compute_g1 = real

    def test_no_indeterminate_category_is_ever_produced(self):
        verdicts = set()
        for triple in TRUTH_TABLE:
            vg = grid(truth_grid(*triple))
            verdicts.add(s1b.evaluate_comparison(
                vg, 200, 1000, CANDS, SEEDS, gating=True,
                with_bootstrap=False)["gate_verdict"])
        assert verdicts == {"PASS", "FAIL"}


# ══════════════════════════════════════════════════════════════════
# I. BOOLEAN TYPING / SANITY COMPOSITION / REPORT CONTRACT
# ══════════════════════════════════════════════════════════════════

class TestPartI1_BooleanTyping:

    @pytest.mark.parametrize("flag", [True, np.bool_(True)])
    def test_true_booleans_are_accepted(self, flag):
        row = make_row("S1-01", SEEDS[0], 200, 0.90)
        row["chance_constraint_pass"] = flag
        assert s1b.cell_defects(row, ("S1-01", SEEDS[0], 200)) == []

    @pytest.mark.parametrize("flag", [False, np.bool_(False)])
    def test_false_booleans_are_accepted(self, flag):
        row = make_row("S1-01", SEEDS[0], 200, 0.85)
        row["chance_constraint_pass"] = flag
        assert s1b.cell_defects(row, ("S1-01", SEEDS[0], 200)) == []

    @pytest.mark.parametrize("flag", ["True", "False", "", 1, 0, 1.0, 0.0,
                                      None, [], np.int64(1)])
    def test_non_boolean_flags_are_refused_without_coercion(self, flag):
        row = make_row("S1-01", SEEDS[0], 200, 0.90)
        row["chance_constraint_pass"] = flag
        defects = s1b.cell_defects(row, ("S1-01", SEEDS[0], 200))
        assert any("not a boolean" in d for d in defects), (flag, defects)

    def test_the_string_False_is_refused_because_truthiness_would_invert(self):
        assert bool("False") is True, "premise: the naive coercion is wrong"
        row = make_row("S1-01", SEEDS[0], 200, 0.85)     # genuinely a FAIL
        row["chance_constraint_pass"] = "False"
        with pytest.raises(IE, match="refusing to coerce"):
            s1b.validate_cell(row, ("S1-01", SEEDS[0], 200))

    def test_a_flag_disagreeing_with_the_frozen_predicate_is_refused(self):
        row = make_row("S1-01", SEEDS[0], 200, 0.85)
        row["chance_constraint_pass"] = True             # should be False
        with pytest.raises(IE, match="disagrees with"):
            s1b.validate_cell(row, ("S1-01", SEEDS[0], 200))
        row = make_row("S1-01", SEEDS[0], 200, 0.90)
        row["chance_constraint_pass"] = False            # should be True
        with pytest.raises(IE, match="disagrees with"):
            s1b.validate_cell(row, ("S1-01", SEEDS[0], 200))

    def test_classification_gates_refuse_non_boolean_pass_flags(self):
        """Built with a non-boolean decision from the start, via the module's
        own synthetic constructor. Provenance is intact, so the boolean guard
        is what must fire -- and "False" is the adversarial case, since
        bool("False") is True and coercion would invert the label."""
        spec = {c: [(Fraction(0), True, True)] * 10 for c in CANDS}
        spec[OFF[0]] = ([(Fraction(0), "False", True)]
                        + [(Fraction(0), True, True)] * 9)
        cells = s1b._synthetic_cells(spec, SEEDS)
        assert type(cells) is s1b.ValidatedCells
        with pytest.raises(IE, match="non-boolean pass flags"):
            s1b.compute_g3(cells, SEEDS, OFF)
        with pytest.raises(IE, match="non-boolean pass flags"):
            s1b.flip_stats(cells, CANDS, SEEDS)


@pytest.fixture(scope="module")
def cands():
    """The 10 frozen Stage-1 candidates, read-only from the checkpoint."""
    _doc, cands = s1.load_frozen_checkpoint()
    return cands


class TestPartI2_SanityComposition:

    @pytest.mark.parametrize("k", range(2, 11))
    def test_every_sanity_subset_spans_both_strata(self, cands, k):
        """G3 is defined on the OFF stratum, so a sanity subset without an OFF
        candidate would leave the gate undefined. k=2 is the driver default."""
        picked = [c["stage1_candidate_id"]
                  for c in s1b.stratified_subset(cands, k)]
        assert len(picked) == k and len(set(picked)) == k
        assert [c for c in picked if c in KNIFE], picked
        assert [c for c in picked if c in OFF], picked

    def test_a_one_candidate_sanity_subset_is_refused(self, cands):
        with pytest.raises(IE, match="at least 2 candidates"):
            s1b.stratified_subset(cands, 1)

    def test_a_knife_only_panel_cannot_produce_a_sanity_subset(self, cands):
        knife_only = [c for c in cands
                      if c["stage1_candidate_id"] in KNIFE]
        with pytest.raises(IE, match="no OFF candidate|cannot build"):
            s1b.stratified_subset(knife_only, 2)

    def test_the_frozen_strata_partition_the_panel(self):
        assert s1b.verify_strata() is True
        assert len(KNIFE) == len(OFF) == 5
        assert sorted(KNIFE + OFF) == sorted(s1.EXPECTED_CANDIDATE_IDS)


@pytest.fixture(scope="module")
def rendered():
    """A full synthetic Stage-1B report: three comparisons, the ladder, the
    manifest and the rendered markdown. NO study is run; the rows are
    synthetic attainable proportions."""
    def p_fn(c, i, S):
        if S == 200:
            return 0.905 if c in KNIFE else 0.915
        if S == 500:
            return 0.91
        return 0.89 if c in KNIFE else 0.90
    vg = grid(p_fn)
    comps = [s1b.evaluate_comparison(vg, S, 1000, CANDS, SEEDS, gating=True,
                                     n_seeds_expected=s1b.EXPECTED_N_SEEDS,
                                     with_bootstrap=True)
             for S in (200, 500)]
    comps.append(s1b.evaluate_comparison(vg, 200, 500, CANDS, SEEDS,
                                         gating=False, with_bootstrap=False))
    ladder = s1b.decide_ladder(comps)
    provenance = s1b.verify_frozen_methodology()
    manifest = s1b.base_manifest(None, "full-run", SEEDS, CANDS, S_VALUES,
                                 provenance,
                                 validation=vg.validation_record())
    manifest["n_rows"] = 300
    return {"comparisons": comps, "ladder": ladder, "manifest": manifest,
            "md": s1b.render_report(comps, ladder, manifest)}


class TestPartI3_ReportContract:

    def test_manifest_carries_the_stage1b_provenance_and_configuration(self, rendered):
        m = rendered["manifest"]
        assert m["stage"] == "A-lite Stage 1B"
        fm = m["frozen_methodology"]
        assert fm["amendment_sha256"] == s1b.EXPECTED_AMENDMENT_SHA256
        assert fm["freeze_commit"] == s1b.METHODOLOGY_FREEZE_COMMIT
        assert fm["verified"] is True
        assert m["S_values"] == [200, 500, 1000]
        assert m["S_ref"] == 1000 and m["master_S"] == 1000
        assert len(m["master_seeds"]) == 10 == s1b.EXPECTED_N_SEEDS
        assert len(m["candidate_provenance"]["candidate_ids"]) == 10
        assert m["candidate_provenance"]["candidates_canonical_sha256"] == \
            s1.EXPECTED_CANDIDATES_SHA256
        assert m["expected_row_count"] == 300
        assert m["confidence_ontime_asserted"] == 0.90
        assert m["indeterminate_category_exists"] is False
        assert m["gates"]["G1"]["threshold"] == 0.02
        assert m["gates"]["G2dagger"]["threshold"] == 0.02
        assert m["gates"]["G3"]["threshold"] == 0.10
        assert all(m["gates"][g]["strict"] for g in ("G1", "G2dagger", "G3"))
        assert m["gates"]["rule"] == "PASS iff G1 AND G2dagger AND G3"
        assert "DESCRIPTIVE ONLY" in m["non_gating"]["seed_cluster_bootstrap"]
        assert "DESCRIPTIVE ONLY" in m["non_gating"]["leave_one_candidate_out"]
        assert "never enters the ladder" in m["non_gating"]["S200_vs_S500"]

    def test_manifest_carries_an_explicit_successful_validation_record(self, rendered):
        v = rendered["manifest"]["grid_validation"]
        assert v["status"] == "VALIDATED"
        assert v["expected_cells"] == 300 == v["validated_cells"]
        assert v["n_candidates"] == 10 and v["n_seeds"] == 10
        assert v["s_values"] == [200, 500, 1000]
        assert v["imputed_cells"] == 0
        assert v["dropped_cells"] == 0
        assert v["reduced_denominators"] == 0
        assert "amendment section 2.2 item 10" in v["rule"]

    def test_a_manifest_without_a_validation_record_says_so_explicitly(self):
        """Absence of the record must be a visible status, never a silent
        implication that the grid was validated."""
        m = s1b.base_manifest(None, "full-run", SEEDS, CANDS, S_VALUES,
                              {"amendment_sha256": "x"})
        assert m["grid_validation"]["status"] == "NOT_RECORDED"
        assert "must not be quoted as validated" in m["grid_validation"]["rule"]

    def test_a_report_without_a_validation_record_never_reads_as_validated(
            self, rendered):
        """The fallback must not inherit the success wording: a missing
        certificate has to render as a missing certificate."""
        manifest = copy.deepcopy(rendered["manifest"])
        manifest["grid_validation"] = s1b.base_manifest(
            None, "full-run", SEEDS, CANDS, S_VALUES,
            {"amendment_sha256": "x"})["grid_validation"]
        md = s1b.render_report(rendered["comparisons"], rendered["ladder"],
                               manifest)
        assert "Grid validation: **NOT_RECORDED**" in md
        assert "must NOT be quoted or cited as validated" in md
        assert "declared cells present exactly once and valid" not in md
        assert "**VALIDATED**" not in md

    def test_report_states_the_freeze_commit_and_validation_status(self, rendered):
        md = rendered["md"]
        assert s1b.METHODOLOGY_FREEZE_COMMIT in md
        assert "Methodology freeze commit" in md
        assert "Grid validation: **VALIDATED**" in md
        assert "300/300 declared cells present exactly once and valid" in md
        assert "imputed 0, dropped 0, reduced denominators 0" in md

    def test_report_contains_provenance_and_grid_configuration(self, rendered):
        md = rendered["md"]
        assert s1b.EXPECTED_AMENDMENT_SHA256 in md
        assert "[200, 500, 1000]" in md
        assert "S_ref = 1000" in md
        for seed in SEEDS:
            assert str(seed) in md
        assert "alpha = 0.9" in md
        assert "p_hat >= alpha PASSES" in md

    def test_report_prints_all_ten_candidates_untruncated(self, rendered):
        md = rendered["md"]
        for c in CANDS:
            assert f"| {c} |" in md, c
        # three comparisons x 10 candidate rows, plus 3 x 10 LOCO rows
        assert sum(md.count(f"| {c} |") for c in CANDS) == 60
        for truncation in ("top 3", "...", "truncated", "worst candidate only"):
            assert truncation not in md

    def test_report_contains_every_mandatory_per_candidate_statistic(self, rendered):
        md = rendered["md"]
        for column in ("D_c", "A_c", "Dmax_c", "Dmin_c", "nflip_c",
                       "netflip_c", "bound@S", "bound@ref", "stratum"):
            assert column in md, column

    def test_report_contains_the_three_gates_with_the_argmax_candidate(self, rendered):
        md = rendered["md"]
        assert "G1 = " in md and "max_c D_c = " in md and "G3 = " in md
        argmax = rendered["comparisons"][0]["gates"]["G2dagger"]["argmax_candidate"]
        assert f"[{argmax}, " in md
        assert "Gates: **G1 AND G2dagger AND G3**" in md

    def test_report_contains_every_mandatory_panel_diagnostic(self, rendered):
        md = rendered["md"]
        for probe in ("all-pair flips", "KNIFE flips", "OFF flips",
                      "false-feasible", "false-infeasible", "net imbalance",
                      "exact p_hat=alpha mass", "0.01 tripwire",
                      "seed-cluster bootstrap (DESCRIPTIVE ONLY)",
                      "sqrt-rate check", "ratio-matched to Stage-1",
                      "Per-seed breakdown", "Leave-one-candidate-out"):
            assert probe in md, probe

    def test_report_contains_the_full_per_seed_and_loco_tables(self, rendered):
        md = rendered["md"]
        loco = rendered["comparisons"][0]["non_gating"]["loco"]
        assert len(loco["per_omitted_candidate"]) == 10
        per_seed = rendered["comparisons"][0]["non_gating"]["diagnostics"][
            "per_seed_breakdown"]
        assert len(per_seed) == 10
        for seed in SEEDS:
            assert f"| {seed} |" in md, seed

    def test_report_carries_the_diagnostic_only_comparison(self, rendered):
        md = rendered["md"]
        assert "S200_vs_S500" in md
        assert "(DIAGNOSTIC ONLY)" in md

    def test_report_states_s1000_is_not_ground_truth(self, rendered):
        assert ("**S1000 is an empirical reference, not ground truth.**"
                in rendered["md"])

    def test_report_states_which_statistics_are_non_gating(self, rendered):
        md = rendered["md"]
        assert ("**Bootstrap, LOCO and the 0.01 tripwire are non-gating**"
                in md)
        assert ("All-pair and KNIFE gross flip rates are **non-gating** "
                "diagnostics; only the OFF-stratum rate gates, as G3." in md)
        assert "NO INDETERMINATE category" in md

    def test_report_states_the_ladder_decision(self, rendered):
        md, ladder = rendered["md"], rendered["ladder"]
        assert "## Ladder decision" in md
        assert f"selected S: **{ladder['selected_S']}**" in md
        assert f"Stage 2 may proceed: **{ladder['stage2_may_proceed']}**" in md


# ══════════════════════════════════════════════════════════════════
# Guard: this contract suite never runs a study and never mutates state
# ══════════════════════════════════════════════════════════════════

def test_this_suite_creates_or_mutates_no_stage1b_output():
    """This suite must not execute a study of any kind, nor disturb one.

    Stated as a CONTENT DELTA against a pre-import snapshot, not as absolute
    absence: a legitimate sanity run creates SANITY_OUT_DIR, and the contract
    suite must stay valid after one has happened. What must never change is
    that running these tests creates, deletes or rewrites nothing there.
    """
    for d, before in _OUT_TREES_AT_IMPORT.items():
        after = _tree_inventory(d)
        if before is None:
            assert after is None, (
                f"{d} was created while the contract suite ran: this suite "
                "must never execute a sanity run or a full study")
        else:
            assert after is not None, f"{d} was deleted by the contract suite"
            assert after == before, (
                f"{d} contents changed while the contract suite ran "
                f"(added={sorted(set(after) - set(before))}, "
                f"removed={sorted(set(before) - set(after))}, "
                f"modified={sorted(k for k in set(after) & set(before) if after[k] != before[k])})")


def test_the_snapshotted_paths_are_the_drivers_real_output_directories():
    """The pre-import snapshot is only meaningful if it covers the very
    directories the driver writes to."""
    assert set(_OUT_TREES_AT_IMPORT) == {str(FSPath(s1b.OUT_DIR)),
                                         str(FSPath(s1b.SANITY_OUT_DIR))}


def test_the_sanity_and_full_study_output_trees_never_alias():
    """The invariant that holds BOTH before and after the authorised full run.

    Absolute absence of the full-study directory was a true invariant only
    until that study was authorised and run; it is not a property of the code
    and must not be asserted as one. What IS permanent is that sanity output
    and scientific output are distinct trees, so a sanity number can never be
    read back as a scientific one.
    """
    full, sanity = FSPath(s1b.OUT_DIR), FSPath(s1b.SANITY_OUT_DIR)
    assert full != sanity
    assert full.name != sanity.name
    assert not str(full).startswith(str(sanity) + os.sep)
    assert not str(sanity).startswith(str(full) + os.sep)


def test_frozen_methodology_is_intact():
    prov = s1b.verify_frozen_methodology()
    assert prov["amendment_sha256"] == s1b.EXPECTED_AMENDMENT_SHA256
    assert prov["freeze_commit_blob_sha256"] == s1b.EXPECTED_AMENDMENT_SHA256
