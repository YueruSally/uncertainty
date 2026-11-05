#!/usr/bin/env python3
"""One frozen-policy search. Called in an isolated process by the coordinator."""
import argparse
from pathlib import Path
import time

import torch

from m_operator_closed_loop import FrozenPathPolicy, FrozenOperatorController, ProfiledBase
from m_operator_offline import digest, write_json
from run_learning_components_deterministic import _normalizer
from transformer_ppo.episode import run_episode

METHODS = ("BASE", "HEURISTIC_BALANCED", "TRANSFORMER_PRETRAINED")


def execute(data, instance, normalization, bundle, out, seed, method, plan_sha):
    started = time.perf_counter()
    torch.set_num_threads(1)
    policy = None if method == "BASE" else FrozenPathPolicy(bundle, heuristic=method == "HEURISTIC_BALANCED")
    loading = time.perf_counter() - started
    result = run_episode(data, instance, out, policy, _normalizer(normalization), seed,
        evaluation_budget=400, pop_size=20, generations=1000,
        evaluation_mode="ccp", scenario_count=100, ccp_confidence=.9,
        uncertainty_profile="transport_border_saa_v1", component="BASE" if policy is None else "M",
        inference_controller_factories={"BASE": ProfiledBase, "M": FrozenOperatorController})
    if result["ppo_updates"] or result.get("counterfactual_evaluation_count", 0) or result.get("candidate_audit_evaluations", 0):
        raise RuntimeError("frozen search must not train or evaluate counterfactual candidates")
    if not 396 <= result["real_objective_evaluation_count"] <= 400 or result["stop_reason"] != "evaluation_budget":
        raise RuntimeError("search did not exhaust the common B400 budget within offspring-pair reserve")
    wall = time.perf_counter() - started
    objective = result["objective_evaluation_seconds"]
    model = result["feature_construction_seconds"] + result["inference_seconds"]
    hv = result["profile_hv_seconds"] + result["final_hv_seconds"]
    result.update(strategy=method, algorithm_seed=seed, plan_sha256=plan_sha,
        frozen_policy_sha256=digest(bundle), worker_seconds=wall,
        policy_loading_seconds=loading,
        timing_seconds={"ccp_objective_evaluation": objective,
            "state_construction_and_inference": model, "hv_and_archive": hv,
            "startup_io_and_other_search": max(0., wall - objective - model - hv)},
        new_training_steps=0)
    write_json(out / "RUN_COMPLETE.json", result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "instance", "normalization", "bundle", "out"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--method", choices=METHODS, required=True)
    p.add_argument("--plan-sha", required=True)
    a = p.parse_args()
    r = execute(a.data, a.instance, a.normalization, a.bundle, a.out, a.seed, a.method, a.plan_sha)
    print(f"[M-CLOSED-RUN] {a.method} HV={r['final_hypervolume']:.8g} seconds={r['optimisation_seconds']:.2f}", flush=True)


if __name__ == "__main__":
    main()
