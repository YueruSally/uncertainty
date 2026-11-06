#!/usr/bin/env python3
"""Frozen size-200 K8 ranker versus Balanced on three new dev instances."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import fcntl
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import threading
import time

SOURCE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE))

import numpy as np
import torch

from m_operator_offline import digest, read_json, write_json
from m_prescreen_scale_policy import (METHODS, FrozenScaleRanker,
    ScaleSearchController, ScaleSearchPolicy)
from run_learning_components_deterministic import _normalizer
from run_m_operator_closed_loop_v1 import load_run
from transformer_ppo.episode import run_episode
from transformer_ppo.instance_catalog import canonical_digest, read_instance


INSTANCES = tuple(f"prescreen-search-S0-{index:03d}" for index in range(1, 4))
SEEDS = tuple(range(918001, 918011))


def log(message):
    print(f"[M-K8-SCALE-SEARCH] {message}", flush=True)


def validate_run(row, method=None):
    audit = row.get("k8_scale_search", {})
    if method is not None and row.get("strategy") != method:
        raise ValueError("scale-search method identity mismatch")
    if (row.get("real_objective_evaluation_count") != 400
            or row.get("evaluation_budget") != 400
            or row.get("ppo_updates", 0)
            or row.get("training_feedback_evaluations", 0)
            or row.get("counterfactual_evaluation_count", 0)
            or audit.get("method") != row.get("strategy")
            or audit.get("hidden_candidate_objective_evaluations") != 0
            or audit.get("pool_events", 0) < 1
            or audit.get("pool_candidates", 0) < audit.get("pool_events", 0)
            or not math.isfinite(row.get("final_hypervolume", math.nan))
            or row.get("final_hypervolume", 0.0) <= 0.0):
        raise ValueError("invalid frozen K8 scale-search run")


def run_one(a):
    torch.set_num_threads(1)
    info, _ = read_instance(a.instance)
    predictor = None
    if a.method == "TRANSFORMER_K8_SIZE200":
        predictor = FrozenScaleRanker(read_json(a.bundle), digest)
    policy = ScaleSearchPolicy(info["instance_id"], a.seed, a.method,
                               predictor=predictor)
    holders = []

    def factory(*args, **kwargs):
        controller = ScaleSearchController(*args, **kwargs)
        holders.append(controller)
        return controller

    result = run_episode(a.data, a.instance, a.out, policy,
        _normalizer(a.normalization), a.seed, evaluation_budget=400,
        pop_size=20, generations=1000, evaluation_mode="ccp", component="M",
        scenario_count=100, ccp_confidence=.9,
        uncertainty_profile="transport_border_saa_v1",
        inference_controller_factories={"M": factory})
    if len(holders) != 1:
        raise RuntimeError("scale-search controller was not installed once")
    result.update(strategy=a.method, algorithm_seed=a.seed,
        plan_sha256=a.plan_sha, new_training_steps=0, new_ppo_updates=0,
        learning_enabled=False,
        role="bounded frozen size-200 K8 development search")
    validate_run(result, a.method)
    write_json(a.out/"RUN_COMPLETE.json", result)


def bootstrap_interval(values, seed):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    sampled = values[rng.integers(0, len(values), (20_000, len(values)))]
    means = sampled.mean(1)
    return [float(np.quantile(means, .025)),
            float(np.quantile(means, .975))]


def summarize(runs, instances=INSTANCES, seeds=SEEDS):
    keyed = {(row["instance"], row["algorithm_seed"], row["strategy"]): row
             for row in runs}
    expected = {(name, seed, method) for name in instances for seed in seeds
                for method in METHODS}
    if len(keyed) != len(runs) or set(keyed) != expected:
        raise ValueError("incomplete or duplicate scale-search grid")
    per_instance, paired, table = [], [], []
    total_switches = 0
    for instance_index, name in enumerate(instances):
        values = []
        for seed in seeds:
            baseline = keyed[name, seed, "HEURISTIC_BALANCED"]
            transformer = keyed[name, seed, "TRANSFORMER_K8_SIZE200"]
            for row in (baseline, transformer):
                validate_run(row, row["strategy"])
                if row["pairing"] != baseline["pairing"]:
                    raise ValueError("paired initialization or scenarios changed")
            gain = 100.0*(transformer["final_hypervolume"] /
                          baseline["final_hypervolume"]-1.0)
            values.append(gain)
            paired.append({"instance": name, "algorithm_seed": seed,
                "transformer_minus_balanced_percent": gain,
                "balanced_hv": baseline["final_hypervolume"],
                "transformer_hv": transformer["final_hypervolume"]})
        interval = bootstrap_interval(values, 919001+instance_index)
        switches = sum(keyed[name, seed, "TRANSFORMER_K8_SIZE200"]
            ["k8_scale_search"]["switches"] for seed in seeds)
        total_switches += switches
        item = {"instance": name, "paired_seeds": len(seeds),
            "mean_paired_percent_vs_balanced": statistics.fmean(values),
            "paired_seed_bootstrap_95ci_low": interval[0],
            "paired_seed_bootstrap_95ci_high": interval[1],
            "wins": sum(value > 1e-12 for value in values),
            "ties": sum(abs(value) <= 1e-12 for value in values),
            "losses": sum(value < -1e-12 for value in values),
            "transformer_switches": switches,
            "mean_balanced_hv": statistics.fmean(
                keyed[name, seed, "HEURISTIC_BALANCED"]["final_hypervolume"]
                for seed in seeds),
            "mean_transformer_hv": statistics.fmean(
                keyed[name, seed, "TRANSFORMER_K8_SIZE200"]["final_hypervolume"]
                for seed in seeds)}
        per_instance.append(item)
        for method in METHODS:
            selected = [keyed[name, seed, method] for seed in seeds]
            table.append({"instance": name, "method": method,
                "runs": len(selected),
                "mean_hv": statistics.fmean(
                    row["final_hypervolume"] for row in selected),
                "mean_seconds": statistics.fmean(
                    row["optimisation_seconds"] for row in selected),
                "mean_paired_percent_vs_balanced": (
                    0.0 if method == "HEURISTIC_BALANCED" else
                    statistics.fmean(values))})
    instance_means = [row["mean_paired_percent_vs_balanced"]
                      for row in per_instance]
    pooled = statistics.fmean(instance_means)
    passed = (pooled > 0.0 and all(value > 0.0 for value in instance_means)
              and total_switches > 0)
    return {"completed": True, "pairing_exact": True,
        "per_instance": per_instance,
        "pooled_equal_instance": {
            "mean_paired_percent_vs_balanced": pooled,
            "positive_instances": sum(value > 0 for value in instance_means),
            "instance_count": len(instance_means)},
        "table": table, "paired_runs": paired,
        "total_transformer_switches": total_switches,
        "development_search_pass": passed,
        "conclusion": ("ADMIT_FROZEN_K8_MODEL_TO_BOUNDED_PPO"
                       if passed else
                       "FROZEN_K8_MODEL_DID_NOT_BEAT_BALANCED_ON_ALL_INSTANCES"),
        "next_action": ("add bounded PPO and retain no-PPO ablation"
                        if passed else
                        "stop K8 ranker; do not open locked final test")}


def frozen_plan(a):
    protocol = read_json(SOURCE/"experiments/m_prescreen_protocol_v1.json")
    design = protocol.get("scale_bounded_search", {})
    if (design.get("instances") != list(INSTANCES)
            or design.get("algorithm_seeds") != list(SEEDS)
            or design.get("methods") != list(METHODS)):
        raise ValueError("bounded-search protocol differs from implementation")
    bundle = read_json(a.bundle)
    curve = read_json(a.curve_root/"QUICK_RESULT.json")
    model = bundle.get("models", {}).get("200", {})
    if (curve.get("curve", [{}])[-1].get("training_instances") != 200
            or curve["curve"][-1].get("mean_net_gain_all_states", 0) <= 0
            or curve["curve"][-1].get("positive_instances", 0) < 12
            or bundle.get("protocol") != "m-prescreen-scale-learning-curve-v1"
            or len(model.get("checkpoints", ())) != 3
            or not (a.curve_root/"TRAINING_COMPLETE").exists()):
        raise ValueError("size-200 exploratory admission evidence changed")
    if (bundle.get("normalization_sha256") != digest(a.normalization)
            or Path(bundle.get("collection", "")).resolve()
            != a.collection.resolve()):
        raise ValueError("bundle inputs differ from requested search")
    collection_plan = read_json(a.collection/"PLAN.json")
    if (collection_plan.get("train_count") != 200
            or collection_plan.get("validation_count") != 20):
        raise ValueError("scale collection changed")
    catalog = read_json(a.search_instances/"catalog.json")
    identity = dict(catalog); claimed = identity.pop("catalog_digest", None)
    if (canonical_digest(identity) != claimed
            or catalog.get("count") != 3
            or catalog.get("final_test_instances_used") is not False):
        raise ValueError("bounded-search instance catalog changed")
    catalog_names = [row["instance_id"] for row in catalog["instances"]]
    if catalog_names != list(INSTANCES):
        raise ValueError("bounded-search instances changed")
    inputs = [a.data.resolve(), a.normalization.resolve(), a.bundle.resolve(),
        a.collection/"PLAN.json", a.collection/"QUICK_RESULT.json",
        a.search_instances/"catalog.json", Path(__file__).resolve(),
        SOURCE/"m_prescreen_scale_policy.py", SOURCE/"m_prescreen_k8.py",
        SOURCE/"m_prescreen_surrogate.py",
        SOURCE/"experiments/m_prescreen_protocol_v1.json",
        SOURCE/"data/border_crossing_events.csv"]
    inputs.extend(Path(path) for path in model["checkpoints"])
    inputs.extend(a.search_instances/f"{name}.json" for name in INSTANCES)
    return {"protocol": "m-prescreen-scale-bounded-search-v1",
        "instances": list(INSTANCES), "algorithm_seeds": list(SEEDS),
        "methods": list(METHODS), "run_count": 60,
        "scenario_count": 100, "ccp_confidence": .9,
        "population_size": 20, "evaluation_budget": 400,
        "workers": a.workers, "threads_per_worker": 1,
        "run_timeout_seconds": 300, "invocation_limit_seconds": 1800,
        "automatic_retries": 0, "new_training_steps": 0,
        "hidden_candidate_objective_evaluations": 0,
        "model_size_training_instances": 200,
        "model_selection": "fixed final epoch, fixed three-seed ensemble",
        "admission_qualification": (
            "exploratory exception after positive heldout mean and 12/20 "
            "positive instances; original monotonic learning-curve gate failed"),
        "pass_rule": design["pass"],
        "development_only": True, "opens_locked_final_test": False,
        "data": str(a.data.resolve()),
        "normalization": str(a.normalization.resolve()),
        "collection": str(a.collection.resolve()),
        "curve_root": str(a.curve_root.resolve()),
        "bundle": str(a.bundle.resolve()),
        "search_instances": str(a.search_instances.resolve()),
        "input_sha256": {str(path.resolve()): digest(path) for path in inputs},
        "torch_version": str(torch.__version__),
        "numpy_version": str(np.__version__)}


def execute(a):
    root = a.out.resolve(); plan = frozen_plan(a)
    if (root/"PLAN.json").exists() and read_json(root/"PLAN.json") != plan:
        raise ValueError("frozen bounded-search inputs changed")
    write_json(root/"PLAN.json", plan)
    if (root/"PIPELINE_COMPLETE").exists():
        log("already complete; see QUICK_RESULT.json")
        return
    plan_sha = digest(root/"PLAN.json")
    (root/"logs").mkdir(exist_ok=True)
    started, deadline = time.monotonic(), time.monotonic()+1800
    status = {"stage": "search", "completed_runs": 0, "total_runs": 60}
    lock, stop = threading.Lock(), threading.Event()
    env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
           "OPENBLAS_NUM_THREADS": "1", "PYTHONUNBUFFERED": "1"}
    log("60 searches: 3 untouched dev instances x 10 seeds x 2 methods; CCP100/pop20/B400")

    def group(name, seed):
        rows = []
        order = METHODS if SEEDS.index(seed) % 2 == 0 else METHODS[::-1]
        for method in order:
            if stop.is_set() or time.monotonic() >= deadline:
                raise TimeoutError("30-minute invocation limit reached")
            folder = root/"runs"/name/f"seed_{seed}"/method
            reused = (folder/"RUN_COMPLETE.json").exists()
            if not reused:
                if folder.exists():
                    quarantine = root/"interrupted_runs"/(
                        f"{name}_{seed}_{method}_{time.time_ns()}")
                    quarantine.parent.mkdir(parents=True, exist_ok=True)
                    folder.rename(quarantine)
                logfile = root/"logs"/f"{name}_{seed}_{method}_{time.time_ns()}.log"
                command = [sys.executable, "-u", str(Path(__file__).resolve()),
                    "--worker", "--data", str(a.data.resolve()),
                    "--search-instances", str(a.search_instances.resolve()),
                    "--collection", str(a.collection.resolve()),
                    "--curve-root", str(a.curve_root.resolve()),
                    "--normalization", str(a.normalization.resolve()),
                    "--bundle", str(a.bundle.resolve()), "--out", str(folder),
                    "--instance", str((a.search_instances/f"{name}.json").resolve()),
                    "--seed", str(seed), "--method", method,
                    "--plan-sha", plan_sha]
                with logfile.open("w") as handle, subprocess.Popen(command,
                        cwd=SOURCE, env=env, stdout=handle,
                        stderr=subprocess.STDOUT) as process:
                    run_started = heartbeat = time.monotonic()
                    while process.poll() is None:
                        now = time.monotonic()
                        if (stop.is_set() or now >= deadline
                                or now-run_started >= 300):
                            process.kill(); process.wait()
                            raise TimeoutError(f"run stopped; see {logfile}")
                        if now-heartbeat >= 20:
                            log(f"running {name}/{seed}/{method} pid={process.pid} elapsed={now-run_started:.0f}s")
                            heartbeat = now
                        time.sleep(.2)
                    if process.returncode:
                        tail = "\n".join(logfile.read_text(
                            errors="replace").splitlines()[-20:])
                        raise RuntimeError(f"worker failed: {logfile}\n{tail}")
            row = load_run(folder, name, plan_sha)
            validate_run(row, method)
            rows.append(row)
            with lock:
                status["completed_runs"] += 1
                status["elapsed_seconds"] = time.monotonic()-started
                write_json(root/"STATUS.json", status)
                audit = row["k8_scale_search"]
                log(f"{status['completed_runs']}/60 {'reused' if reused else 'complete'} {name}/{seed}/{method} HV={row['final_hypervolume']:.7g} switches={audit['switches']}")
        return rows

    runs = []
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        futures = [pool.submit(group, name, seed) for name in INSTANCES
                   for seed in SEEDS]
        try:
            for future in as_completed(futures):
                runs.extend(future.result())
        except BaseException:
            stop.set()
            for future in futures:
                future.cancel()
            raise
    report = summarize(runs)
    report.update(design=plan, new_searches=len(runs),
        new_training_steps=0, new_ppo_updates=0,
        search_objective_evaluations=sum(
            row["real_objective_evaluation_count"] for row in runs),
        hidden_candidate_objective_evaluations=0,
        elapsed_seconds_this_invocation=time.monotonic()-started)
    write_json(root/"COMPLETE.json", {**report, "runs": runs})
    write_json(root/"QUICK_RESULT.json", report)
    with (root/"RESULT_TABLE.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(report["table"][0]))
        writer.writeheader(); writer.writerows(report["table"])
    write_json(root/"STATUS.json", {**status, "stage": "complete",
        "conclusion": report["conclusion"]})
    (root/"PIPELINE_COMPLETE").write_text(report["conclusion"]+"\n")
    print((root/"RESULT_TABLE.csv").read_text(), flush=True)
    log(f"COMPLETE gain={report['pooled_equal_instance']['mean_paired_percent_vs_balanced']:+.3f}% {report['conclusion']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path,
        default=SOURCE/"data/data_expanded.xlsx")
    for name in ("search-instances", "collection", "curve-root",
                 "normalization", "bundle", "out"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--workers", choices=(1, 2, 3, 4), default=3,
                        type=int)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--instance", type=Path)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--method", choices=METHODS)
    parser.add_argument("--plan-sha")
    a = parser.parse_args(); a.out = a.out.resolve()
    if a.worker:
        if any(getattr(a, name) is None for name in
               ("instance", "seed", "method", "plan_sha")):
            parser.error("worker arguments incomplete")
        a.out.mkdir(parents=True, exist_ok=False)
        run_one(a); return
    a.out.mkdir(parents=True, exist_ok=True)
    with (a.out/".lock").open("w") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            execute(a)
        except Exception as exc:
            previous = (read_json(a.out/"STATUS.json")
                        if (a.out/"STATUS.json").exists() else {})
            write_json(a.out/"STATUS.json", {
                **previous, "stage": "stopped", "error": str(exc)})
            log(f"STOPPED {exc}")
            raise


if __name__ == "__main__":
    main()
