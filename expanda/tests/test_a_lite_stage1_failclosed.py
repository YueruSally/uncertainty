#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Fail-closed tests for the A-lite Stage-1 sensitivity script.

A fail-closed design is worthless unless the guards are shown to actually
fire, so every test here INJECTS a fault and asserts Stage1IntegrityError.
Nothing on disk is modified: faults are injected into in-memory copies or
via monkeypatching, and the real checkpoint / production modules are never
written to.

Runs NO Stage-1 study: no candidate is evaluated except inside the two
tests that deliberately feed a corrupted evaluator, and those use a stub.
"""
import copy
import json
import sys
from pathlib import Path as FSPath

import numpy as np
import pytest

HERE = FSPath(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
import baseline_uncertainty as model  # noqa: E402
import a_lite_stage1_sensitivity as st1  # noqa: E402

IE = st1.Stage1IntegrityError


@pytest.fixture(scope="module")
def doc_cands():
    return st1.load_frozen_checkpoint()


# ── provenance guards ────────────────────────────────────────────────────

def test_real_checkpoint_loads_clean(doc_cands):
    doc, cands = doc_cands
    assert len(cands) == 10
    assert [c["stage1_candidate_id"] for c in cands] == st1.EXPECTED_CANDIDATE_IDS


def test_candidate_hash_mismatch_fails(doc_cands, monkeypatch, tmp_path):
    _, cands = doc_cands
    bad = copy.deepcopy(cands)
    # perturb a share -> canonical hash must change
    bad[0]["decision_allocations"][0][3][0][2] = 0.5
    p = tmp_path / "bad.json"
    p.write_text(json.dumps({"candidates": bad}), encoding="utf-8")
    monkeypatch.setattr(st1, "CHECKPOINT", p)
    with pytest.raises(IE, match="SHA-256 mismatch"):
        st1.load_frozen_checkpoint()


def test_wrong_candidate_count_fails(doc_cands, monkeypatch, tmp_path):
    _, cands = doc_cands
    p = tmp_path / "short.json"
    p.write_text(json.dumps({"candidates": cands[:9]}), encoding="utf-8")
    monkeypatch.setattr(st1, "CHECKPOINT", p)
    monkeypatch.setattr(st1, "EXPECTED_CANDIDATES_SHA256",
                        st1.candidates_canonical_sha256(cands[:9]))
    with pytest.raises(IE, match="expected 10 frozen candidates"):
        st1.load_frozen_checkpoint()


def test_renamed_candidate_id_fails(doc_cands, monkeypatch, tmp_path):
    _, cands = doc_cands
    bad = copy.deepcopy(cands)
    bad[3]["stage1_candidate_id"] = "S1-99"
    p = tmp_path / "ids.json"
    p.write_text(json.dumps({"candidates": bad}), encoding="utf-8")
    monkeypatch.setattr(st1, "CHECKPOINT", p)
    monkeypatch.setattr(st1, "EXPECTED_CANDIDATES_SHA256",
                        st1.candidates_canonical_sha256(bad))
    with pytest.raises(IE, match="IDs/order changed"):
        st1.load_frozen_checkpoint()


def test_missing_checkpoint_fails(monkeypatch, tmp_path):
    monkeypatch.setattr(st1, "CHECKPOINT", tmp_path / "nope.json")
    with pytest.raises(IE, match="not found"):
        st1.load_frozen_checkpoint()


# ── candidate reconstruction guards ──────────────────────────────────────

@pytest.fixture(scope="module")
def env():
    return st1.prepare_environment()


def test_unreconstructable_path_fails(env, doc_cands):
    _, cands = doc_cands
    bad = copy.deepcopy(cands[0])
    # a node that cannot exist in the network -> no arc, no fallback
    bad["decision_allocations"][0][3][0][0] = ["NOWHERE_A", "NOWHERE_B"]
    bad["decision_allocations"][0][3][0][1] = ["road"]
    with pytest.raises(IE, match="could not be reconstructed exactly"):
        st1.build_candidate_individual(
            bad, env["batches"], env["path_lib"], env["tt_dict"],
            env["arc_lookup"], {"lib_hit": 0, "rebuilt": 0})


def test_missing_batch_fails(env, doc_cands):
    _, cands = doc_cands
    bad = copy.deepcopy(cands[0])
    bad["decision_allocations"] = bad["decision_allocations"][:-1]
    with pytest.raises(IE, match="missing batch"):
        st1.build_candidate_individual(
            bad, env["batches"], env["path_lib"], env["tt_dict"],
            env["arc_lookup"], {"lib_hit": 0, "rebuilt": 0})


# ── border-event configuration guard ─────────────────────────────────────

def test_empty_border_definitions_fail(env, monkeypatch):
    """The exact hazard that silently corrupted an earlier scan."""
    bad_env = dict(env)
    bad_env["border_event_definitions"] = {}
    with pytest.raises(IE, match="border-event configuration mismatch"):
        st1.build_master(bad_env, 700001)


def test_forbidden_seed_rejected(env):
    for s in (42, 43, 1000003):
        with pytest.raises(IE, match="reserved/previously used"):
            st1.build_master(env, s)


# ── nested-prefix guard ──────────────────────────────────────────────────

def test_corrupted_prefix_fails(env):
    master = st1.build_master(env, 700001)
    subsets, report = st1.nested_subsets(master, 700001)
    assert report["all_pass"] is True

    tampered = scs_tamper(subsets[50])
    bad = {50: tampered, 100: subsets[100], 200: subsets[200]}
    v = _verify(bad[50], bad[100])
    assert not v["is_prefix_subset"]


def scs_tamper(scen):
    """Return a copy of `scen` with one travel multiplier perturbed."""
    tm = {k: v.copy() for k, v in scen.travel_multiplier.items()}
    first = sorted(tm)[0]
    tm[first][0] += 0.123
    return model.ScenarioSet(
        size=scen.size, seed=scen.seed, travel_multiplier=tm,
        border_delay_h={k: v.copy() for k, v in scen.border_delay_h.items()},
        arc_border_event=dict(scen.arc_border_event),
        border_event_mean_h=dict(scen.border_event_mean_h),
        stochastic=scen.stochastic)


def _verify(small, large):
    import scenario_count_sensitivity as scs
    return scs.verify_prefix_subset(small, large)


def test_nested_subsets_are_true_prefixes(env):
    master = st1.build_master(env, 700002)
    subsets, _ = st1.nested_subsets(master, 700002)
    for S in (50, 100):
        for k, arr in subsets[S].travel_multiplier.items():
            assert np.array_equal(arr, master.travel_multiplier[k][:S])
        for k, arr in subsets[S].border_delay_h.items():
            assert np.array_equal(arr, master.border_delay_h[k][:S])
    assert subsets[200] is master


def test_independent_rebuild_is_NOT_a_prefix(env):
    """Guards the reason prefixes are mandatory: an independently built
    S=50 set is NOT the prefix of an independently built S=200 set."""
    m200 = st1.build_master(env, 700003)
    m50 = model.build_scenario_set(
        arcs=env["arcs"], border_delay_map=env["border_delay_map"], size=50,
        seed=700003, stochastic=True,
        border_event_definitions=env["border_event_definitions"])
    diffs = [k for k, arr in m50.travel_multiplier.items()
             if not np.array_equal(arr, m200.travel_multiplier[k][:50])]
    assert diffs, ("independent rebuild unexpectedly matched the prefix; the "
                   "nesting rationale must be re-examined")


# ── cache-safety guards ──────────────────────────────────────────────────

def test_install_clears_cache_and_sets_active(env):
    master = st1.build_master(env, 700004)
    subsets, _ = st1.nested_subsets(master, 700004)
    model._PATH_SCENARIO_CACHE = {("poison",): "stale"}
    st1.install_scenario_set(subsets[100])
    assert model._PATH_SCENARIO_CACHE == {}
    assert model.ACTIVE_SCENARIO_SET is subsets[100]


def test_foreign_cache_key_detected(env):
    master = st1.build_master(env, 700005)
    subsets, _ = st1.nested_subsets(master, 700005)
    st1.install_scenario_set(subsets[50])
    model._PATH_SCENARIO_CACHE[(999999, 7, True, 1, 0.0, (), ())] = "foreign"
    with pytest.raises(IE, match="foreign scenario keys"):
        st1.assert_cache_confined(subsets[50])
    model._PATH_SCENARIO_CACHE = {}


# ── monotonicity guard ───────────────────────────────────────────────────

def test_maxlate_monotonicity_violation_fails():
    rows = [{"S": 50, "max_late_excess_h": 5.0},
            {"S": 100, "max_late_excess_h": 4.0},
            {"S": 200, "max_late_excess_h": 9.0}]
    with pytest.raises(IE, match="monotonicity VIOLATED"):
        st1.check_maxlate_monotonicity(rows, "S1-01", 700001)


def test_maxlate_monotonicity_ok_passes():
    rows = [{"S": 50, "max_late_excess_h": 1.0},
            {"S": 100, "max_late_excess_h": 1.0},
            {"S": 200, "max_late_excess_h": 2.5}]
    st1.check_maxlate_monotonicity(rows, "S1-01", 700001)


# ── evaluator output guards (stubbed; no real study run) ─────────────────

class _Stub:
    def __init__(self, probs, mle=0.0, drop_fields=()):
        self.batch_on_time_prob = probs
        vb = {
            "max_late_excess_h": mle, "chance_vio": 0.0, "miss_alloc": 0.0,
            "miss_tt": 0.0, "cap_excess": 0.0, "border_cap_excess": 0.0,
            "min_on_time_prob": (min(probs.values()) if probs else 0.0)}
        for f in drop_fields:
            vb.pop(f, None)
        self.vio_breakdown = vb
        self.feasible_hard = True
        self.objectives = (1.0, 2.0, 3.0)
        self.penalty = 0.0


def _run_eval_with(monkeypatch, env, stub):
    master = st1.build_master(env, 700006)
    subsets, _ = st1.nested_subsets(master, 700006)
    scen = subsets[50]
    st1.install_scenario_set(scen)
    monkeypatch.setattr(model, "Individual", lambda **kw: stub)
    monkeypatch.setattr(model, "evaluate_individual", lambda *a, **k: None)
    ind = model.Individual()
    ind.od_allocations = {}
    return st1.evaluate_candidate(env, ind, env["batches"], env["arcs"],
                                  env["tt_dict"], scen, "S1-01", 700006, 50)


def test_nan_probability_fails(env, monkeypatch):
    probs = {i: 0.95 for i in range(1, 21)}
    probs[7] = float("nan")
    with pytest.raises(IE, match="non-finite probability"):
        _run_eval_with(monkeypatch, env, _Stub(probs))


def test_probability_out_of_range_fails(env, monkeypatch):
    probs = {i: 0.95 for i in range(1, 21)}
    probs[3] = 1.4
    with pytest.raises(IE, match=r"outside \[0,1\]"):
        _run_eval_with(monkeypatch, env, _Stub(probs))


def test_missing_probability_vector_fails(env, monkeypatch):
    with pytest.raises(IE, match="missing per-batch probability vector"):
        _run_eval_with(monkeypatch, env, _Stub({}))


def test_truncated_probability_vector_fails(env, monkeypatch):
    probs = {i: 0.95 for i in range(1, 12)}
    with pytest.raises(IE, match="probability vector has"):
        _run_eval_with(monkeypatch, env, _Stub(probs))


def test_infinite_maxlate_fails(env, monkeypatch):
    probs = {i: 0.95 for i in range(1, 21)}
    with pytest.raises(IE, match="non-finite max_late_excess_h"):
        _run_eval_with(monkeypatch, env, _Stub(probs, mle=float("inf")))


def test_missing_vio_breakdown_field_fails(env, monkeypatch):
    """A missing evaluator field must fail closed, not raise KeyError."""
    probs = {i: 0.95 for i in range(1, 21)}
    with pytest.raises(IE, match="missing required field"):
        _run_eval_with(monkeypatch, env,
                       _Stub(probs, drop_fields=("max_late_excess_h",)))


def test_min_prob_disagreement_fails(env, monkeypatch):
    """Recomputed minimum must agree with the evaluator's own value."""
    probs = {i: 0.95 for i in range(1, 21)}
    stub = _Stub(probs)
    stub.vio_breakdown["min_on_time_prob"] = 0.5   # deliberately inconsistent
    with pytest.raises(IE, match="disagrees with"):
        _run_eval_with(monkeypatch, env, stub)


# ── alpha / classification-semantics guards ──────────────────────────────

def test_alpha_is_asserted_not_assumed(monkeypatch):
    """CONFIDENCE_ONTIME is a mutable global and IS the semantics under
    test; a changed threshold must stop the run.

    Earlier tests in this module install ScenarioSets, so the global is
    reset here to reach the alpha guard rather than the earlier
    'already installed' guard.
    """
    monkeypatch.setattr(model, "ACTIVE_SCENARIO_SET", None)
    monkeypatch.setattr(model, "CONFIDENCE_ONTIME", 0.85)
    with pytest.raises(IE, match="CONFIDENCE_ONTIME"):
        st1.prepare_environment()


def test_preexisting_scenario_set_blocks_setup(monkeypatch):
    """Independently confirm the earlier guard also fires."""
    monkeypatch.setattr(model, "ACTIVE_SCENARIO_SET", object())
    with pytest.raises(IE, match="already installed"):
        st1.prepare_environment()


def test_chance_vio_disagreement_fails(env, monkeypatch):
    """Recomputed sum(max(0, alpha - p_i)) must match the evaluator."""
    probs = {i: 0.95 for i in range(1, 21)}
    probs[5] = 0.80                      # true shortfall 0.10
    stub = _Stub(probs)
    stub.vio_breakdown["chance_vio"] = 0.0   # evaluator claims no violation
    with pytest.raises(IE, match="recomputed chance_vio"):
        _run_eval_with(monkeypatch, env, stub)


def test_batch_id_set_mismatch_fails(env, monkeypatch):
    """Right length, wrong batch IDs must not pass."""
    probs = {i: 0.95 for i in range(1, 21)}
    probs[999] = probs.pop(20)
    with pytest.raises(IE, match="batch IDs do not match"):
        _run_eval_with(monkeypatch, env, _Stub(probs))


def test_nan_reported_min_fails(env, monkeypatch):
    """abs(pmin - nan) > tol is False, so NaN must be caught explicitly."""
    probs = {i: 0.95 for i in range(1, 21)}
    stub = _Stub(probs)
    stub.vio_breakdown["min_on_time_prob"] = float("nan")
    with pytest.raises(IE, match="non-finite reported min_on_time_prob"):
        _run_eval_with(monkeypatch, env, stub)


def test_negative_hard_diagnostic_fails(env, monkeypatch):
    probs = {i: 0.95 for i in range(1, 21)}
    stub = _Stub(probs)
    stub.vio_breakdown["cap_excess"] = -1.0
    with pytest.raises(IE, match="negative cap_excess"):
        _run_eval_with(monkeypatch, env, stub)


def test_nonfinite_hard_diagnostic_fails(env, monkeypatch):
    probs = {i: 0.95 for i in range(1, 21)}
    stub = _Stub(probs)
    stub.vio_breakdown["border_cap_excess"] = float("inf")
    with pytest.raises(IE, match="non-finite border_cap_excess"):
        _run_eval_with(monkeypatch, env, stub)


def test_feasible_hard_inconsistent_fails(env, monkeypatch):
    """feasible_hard must equal the conjunction of its components."""
    probs = {i: 0.95 for i in range(1, 21)}
    stub = _Stub(probs)
    stub.vio_breakdown["cap_excess"] = 10.0   # non-CCP violation present
    stub.feasible_hard = True                 # but claims hard-feasible
    with pytest.raises(IE, match="disagrees with components"):
        _run_eval_with(monkeypatch, env, stub)


def test_nonfinite_objective_fails(env, monkeypatch):
    probs = {i: 0.95 for i in range(1, 21)}
    stub = _Stub(probs)
    stub.objectives = (float("inf"), 2.0, 3.0)
    with pytest.raises(IE, match="non-finite obj_cost"):
        _run_eval_with(monkeypatch, env, stub)


# ── run-control guards ───────────────────────────────────────────────────

def test_write_outputs_refuses_to_overwrite(tmp_path):
    d = tmp_path / "results"
    d.mkdir()
    (d / "existing.json").write_text("{}", encoding="utf-8")
    with pytest.raises(IE, match="Refusing to overwrite"):
        st1.write_outputs([], [], [], {}, {}, d)


def test_sanity_and_full_output_dirs_differ():
    assert st1.OUT_DIR != st1.SANITY_OUT_DIR


def test_predeclared_materiality_present():
    assert st1.MATERIALITY["mean_abs_prob_delta_vs_S200"] == 0.02
    assert st1.MATERIALITY["classification_flip_rate_vs_S200"] == 0.10


def test_expected_alpha_declared():
    assert st1.EXPECTED_CONFIDENCE_ONTIME == 0.90


# ── held-out separation ──────────────────────────────────────────────────

def test_held_out_disabled_and_seed_disjoint():
    assert st1.HELD_OUT_ENABLED is False
    assert st1.HELD_OUT_SEED is None and st1.HELD_OUT_S is None
    assert not (set(st1.MASTER_SEEDS) & st1.FORBIDDEN_SEEDS)
    assert len(set(st1.MASTER_SEEDS)) == 10


def test_seeds_declared_and_stable():
    assert st1.MASTER_SEEDS == [700001, 700002, 700003, 700004, 700005,
                                700006, 700007, 700008, 700009, 700010]
