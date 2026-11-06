#!/usr/bin/env python3
"""Reviewer checks for the frozen K8 result: oracle, simple and OOS."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import csv
import fcntl
import hashlib
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

import baseline_uncertainty as base
from evaluate_m_directed_balanced_oos import (_evaluate_front, _normalizer,
    _uncertainty_profile, derive_common_oos_reference)
from m_operator_offline import digest, read_json, write_json
from m_prescreen_final_oracle import FinalOracleController, FinalOraclePolicy
from m_prescreen_scale_policy import METHODS as SOURCE_METHODS
from run_ev_ccp_oos_pilot import scenario_digest
from run_learning_components_deterministic import _normalizer as training_normalizer
from run_m_operator_closed_loop_v1 import load_run
from transformer_ppo.archive import exact_hv_3d
from transformer_ppo.episode import run_episode
from transformer_ppo.instance_catalog import canonical_digest, read_instance


INSTANCES = tuple(f"prescreen-search-S0-{index:03d}" for index in range(1, 4))
SEEDS = tuple(range(918001, 918011))
BALANCED, TRANSFORMER = SOURCE_METHODS
SIMPLE, ORACLE = "K8_SIMPLE", "K8_ORACLE"
METHODS = (BALANCED, SIMPLE, TRANSFORMER, ORACLE)
NEW_METHODS = (SIMPLE, ORACLE)
OOS_SCENARIO_COUNT = 1000
OOS_SEED = 97_700_000
MODEL_EXECUTION_COMMIT = "3813704cf8b52f5092e7b88e40d9216b04c448f4"


def log(message):
    print(f"[M-K8-REVIEW] {message}", flush=True)


def summary(values, instances):
    values = [float(value) for value in values]
    mean = statistics.fmean(values)
    deviation = statistics.stdev(values) if len(values) > 1 else 0.0
    error = deviation/math.sqrt(len(values))
    return {"runs": len(values), "mean": mean,
        "mean_95ci_low": mean-1.96*error,
        "mean_95ci_high": mean+1.96*error,
        "wins": sum(value > 1e-12 for value in values),
        "ties": sum(abs(value) <= 1e-12 for value in values),
        "losses": sum(value < -1e-12 for value in values),
        "per_instance": [{"instance": name,
            "runs": sum(instance == name for instance in instances),
            "mean": statistics.fmean(value for value, instance in
                                      zip(values, instances) if instance == name),
            "wins": sum(value > 1e-12 for value, instance in
                        zip(values, instances) if instance == name)}
            for name in sorted(set(instances))]}


def run_audit_worker(a):
    torch.set_num_threads(1)
    info, _ = read_instance(a.instance)
    policy = FinalOraclePolicy(info["instance_id"], a.seed, a.method)
    holders = []

    def factory(*args, **kwargs):
        controller = FinalOracleController(*args, **kwargs)
        holders.append(controller)
        return controller

    result = run_episode(a.data, a.instance, a.out, policy,
        training_normalizer(a.normalization), a.seed, evaluation_budget=400,
        pop_size=20, generations=1000, evaluation_mode="ccp", component="M",
        scenario_count=100, ccp_confidence=.9,
        uncertainty_profile="transport_border_saa_v1",
        inference_controller_factories={"M": factory})
    if len(holders) != 1:
        raise RuntimeError("review controller was not installed once")
    result.update(strategy=a.method, algorithm_seed=a.seed,
        plan_sha256=a.plan_sha, new_training_steps=0, new_ppo_updates=0,
        learning_enabled=False, role="K8 reviewer audit")
    validate_audit_run(result, a.method)
    write_json(a.out/"RUN_COMPLETE.json", result)


def validate_audit_run(row, method):
    audit = row.get("k8_final_hv_audit", {})
    extra = audit.get("oracle_additional_objective_evaluations", -1)
    if (method not in NEW_METHODS or row.get("strategy") != method
            or row.get("real_objective_evaluation_count") != 400
            or row.get("evaluation_budget") != 400
            or row.get("ppo_updates", 0)
            or row.get("training_feedback_evaluations", 0)
            or audit.get("method") != method
            or audit.get("pool_events", 0) < 1
            or not math.isfinite(row.get("final_hypervolume", math.nan))
            or row.get("final_hypervolume", 0) <= 0):
        raise ValueError("invalid K8 reviewer run")
    if method == SIMPLE and extra != 0:
        raise ValueError("simple scorer evaluated hidden candidates")
    if method == ORACLE and extra <= 0:
        raise ValueError("oracle did not report its diagnostic cost")
    if audit.get("total_physical_objective_evaluations") != 400+extra:
        raise ValueError("oracle accounting mismatch")


def run_path(root, instance, seed, method):
    if method in SOURCE_METHODS:
        return root/"source_search"/"runs"/instance/f"seed_{seed}"/method
    return root/"runs"/instance/f"seed_{seed}"/method


def load_all_runs(a, plan_sha):
    source_sha = digest(a.source_search/"PLAN.json")
    rows = []
    for instance in INSTANCES:
        for seed in SEEDS:
            for method in METHODS:
                folder = (a.source_search/"runs"/instance/f"seed_{seed}"/method
                          if method in SOURCE_METHODS else
                          a.out/"runs"/instance/f"seed_{seed}"/method)
                row = load_run(folder, instance,
                    source_sha if method in SOURCE_METHODS else plan_sha)
                if method in NEW_METHODS:
                    validate_audit_run(row, method)
                rows.append(row)
    return rows


def summarize_optimization(runs):
    keyed = {(row["instance"], row["algorithm_seed"], row["strategy"]): row
             for row in runs}
    table, paired, per_instance = [], [], []
    for name in INSTANCES:
        gains = {method: [] for method in METHODS}
        for seed in SEEDS:
            baseline = keyed[name, seed, BALANCED]
            for method in METHODS:
                row = keyed[name, seed, method]
                if row["pairing"] != baseline["pairing"]:
                    raise ValueError("review methods are not exactly paired")
                gain = 100*(row["final_hypervolume"] /
                            baseline["final_hypervolume"]-1)
                gains[method].append(gain)
                paired.append({"instance": name, "seed": seed,
                    "method": method, "paired_percent_vs_balanced": gain})
        method_rows = {}
        for method in METHODS:
            selected = [keyed[name, seed, method] for seed in SEEDS]
            item = {"instance": name, "method": method, "runs": len(SEEDS),
                "mean_hv": statistics.fmean(
                    row["final_hypervolume"] for row in selected),
                "mean_paired_percent_vs_balanced": statistics.fmean(gains[method]),
                "wins_vs_balanced": sum(value > 1e-12
                                         for value in gains[method]),
                "mean_seconds": statistics.fmean(
                    row["optimisation_seconds"] for row in selected),
                "mean_oracle_additional_evaluations": statistics.fmean(
                    row.get("k8_final_hv_audit", {}).get(
                        "oracle_additional_objective_evaluations", 0)
                    for row in selected)}
            table.append(item); method_rows[method] = item
        transformer = method_rows[TRANSFORMER]["mean_paired_percent_vs_balanced"]
        oracle = method_rows[ORACLE]["mean_paired_percent_vs_balanced"]
        simple = method_rows[SIMPLE]["mean_paired_percent_vs_balanced"]
        per_instance.append({"instance": name,
            "transformer_mean_percent": transformer,
            "simple_mean_percent": simple, "oracle_mean_percent": oracle,
            "transformer_minus_simple_points": transformer-simple,
            "transformer_exceeds_greedy_oracle": transformer > oracle+1e-12,
            "oracle_capture_fraction": (transformer/oracle
                                        if oracle > 0 else None)})
    transformer_means = [row["transformer_mean_percent"] for row in per_instance]
    oracle_means = [row["oracle_mean_percent"] for row in per_instance]
    simple_means = [row["simple_mean_percent"] for row in per_instance]
    return {"pairing_exact": True, "per_instance": per_instance,
        "equal_instance_mean_percent": {
            TRANSFORMER: statistics.fmean(transformer_means),
            SIMPLE: statistics.fmean(simple_means),
            ORACLE: statistics.fmean(oracle_means)},
        "transformer_minus_simple_points": statistics.fmean(
            t-s for t, s in zip(transformer_means, simple_means)),
        "transformer_exceeds_greedy_oracle_instances": [
            row["instance"] for row in per_instance
            if row["transformer_exceeds_greedy_oracle"]],
        "greedy_oracle_qualification": (
            "K8_ORACLE maximizes immediate current-population HV at each step; "
            "it is not a dynamic-programming upper bound on final search HV"),
        "table": table, "paired_runs": paired}


def evaluate_oos_instance(task):
    started = time.perf_counter()
    instance_path = Path(task["instance_path"])
    data = Path(task["data"]); normalizer_path = Path(task["normalization"])
    out = Path(task["out"]); instance_name = task["instance"]
    instance, batches = read_instance(instance_path)
    if hashlib.sha256(data.read_bytes()).hexdigest() != instance["source_data_sha256"]:
        raise ValueError("OOS workbook differs from instance")
    base.BORDER_EVENT_DEFINITIONS = base.load_border_event_definitions(
        base.DEFAULT_BORDER_EVENT_DATA_FILE)
    network = base.load_network_from_extended(str(data))
    (_nodes, _regions, _hold, _proc, _trans_cost, arcs, _timetables,
     _source_batches, _wait_cost, _wait_emission, _carbon,
     _emission_factors, _speeds, _trans_map, _border_delay, _theta, _) = network
    profile = _uncertainty_profile("transport_border_saa_v1",
                                   OOS_SCENARIO_COUNT)
    active_border_delay = base.load_border_delay_map_from_workbook(
        data, include_customs=profile["border_include_customs"])
    scenarios = base.build_scenario_set(arcs, active_border_delay,
        OOS_SCENARIO_COUNT, OOS_SEED, stochastic=True,
        border_event_definitions=base.BORDER_EVENT_DEFINITIONS,
        mode_time_cv=profile["mode_time_cv"],
        mode_time_max_factor=profile["mode_time_max_factor"],
        border_delay_cv=profile["border_delay_cv"],
        border_delay_max_factor=profile["border_delay_max_factor"],
        border_mean_source=profile["border_mean_source"])
    scenario_sha = scenario_digest(scenarios)
    base.ACTIVE_SCENARIO_SET = scenarios; base._PATH_SCENARIO_CACHE = {}
    base.RISK_METRIC = "ccp"
    base.CONFIDENCE_COST = base.CONFIDENCE_EMISSION = base.CONFIDENCE_TIME = .9
    divisor = (max(1., sum(float(batch.quantity) for batch in batches)),
        max(1., sum(float(batch.quantity) for batch in batches)),
        max(1., max(float(batch.LT) for batch in batches)-
            min(float(batch.ET) for batch in batches)))
    instance_normalizer = _normalizer(normalizer_path).with_divisor(divisor)
    rows = []
    for seed in SEEDS:
        target = out/"raw"/instance_name/f"seed_{seed}.json"
        if target.is_file():
            value = read_json(target)
            if (value.get("oos_scenario_digest") == scenario_sha
                    and set(value.get("methods", {})) == set(METHODS)):
                rows.append(value); continue
        methods = {}
        for method in METHODS:
            methods[method] = _evaluate_front(
                Path(task["run_paths"][str(seed)][method]), instance, batches,
                network, scenarios, active_border_delay, instance_normalizer,
                .9, profile)
        value = {"completed": True, "instance": instance_name,
            "algorithm_seed": seed, "oos_scenario_count": scenarios.size,
            "oos_seed": scenarios.seed, "oos_scenario_digest": scenario_sha,
            "uncertainty_profile": profile,
            "normalization_divisor": list(divisor), "methods": methods}
        write_json(target, value); rows.append(value)
    return {"instance": instance_name, "rows": rows,
        "scenario_digest": scenario_sha,
        "elapsed_seconds": time.perf_counter()-started}


def run_oos(a):
    tasks = []
    for name in INSTANCES:
        paths = {str(seed): {method: str(
            a.source_search/"runs"/name/f"seed_{seed}"/method
            if method in SOURCE_METHODS else
            a.out/"runs"/name/f"seed_{seed}"/method)
            for method in METHODS} for seed in SEEDS}
        tasks.append({"instance": name,
            "instance_path": str(a.search_instances/f"{name}.json"),
            "data": str(a.data), "normalization": str(a.normalization),
            "out": str(a.out/"oos"), "run_paths": paths})
    rows = []
    with ProcessPoolExecutor(max_workers=min(3, a.workers)) as pool:
        futures = [pool.submit(evaluate_oos_instance, task) for task in tasks]
        for index, future in enumerate(as_completed(futures), 1):
            result = future.result(); rows.extend(result["rows"])
            log(f"OOS {index}/3 {result['instance']} elapsed={result['elapsed_seconds']:.1f}s")
    digests = {row["oos_scenario_digest"] for row in rows}
    training = {method["training_scenario_digest"] for row in rows
                for method in row["methods"].values()}
    if len(rows) != 30 or len(digests) != 1 or digests & training:
        raise RuntimeError("OOS completeness or independence check failed")
    reference_info = derive_common_oos_reference(
        rows, _normalizer(a.normalization).reference)
    reference = reference_info["reference"]
    metrics = {method: [] for method in METHODS if method != BALANCED}
    method_runtime_instances = []
    candidate_scenarios = 0
    for row in rows:
        hvs = {method: exact_hv_3d(
            row["methods"][method]["oos_normalized_nondominated_points"],
            reference) for method in METHODS}
        if any(value <= 0 for value in hvs.values()):
            raise RuntimeError("nonpositive OOS HV")
        row["oos_hypervolumes"] = hvs
        for method in METHODS:
            candidate_scenarios += (row["methods"][method]["source_front_size"]
                                    * OOS_SCENARIO_COUNT)
        for method in metrics:
            metrics[method].append(100*(hvs[method]/hvs[BALANCED]-1))
            method_runtime_instances.append(row["instance"])
        write_json(a.out/"oos"/"runs"/row["instance"]/
                   f"seed_{row['algorithm_seed']}.json", row)
    instances = [row["instance"] for row in rows]
    result = {"completed": True, "scenario_count": OOS_SCENARIO_COUNT,
        "oos_seed": OOS_SEED, "scenario_digest": next(iter(digests)),
        "distinct_from_all_training_scenarios": not bool(digests & training),
        "one_common_four_method_reference": True,
        "hv_reference": reference_info,
        "candidate_scenario_evaluations": candidate_scenarios,
        "vs_balanced_percent": {method: summary(values, instances)
                                for method, values in metrics.items()}}
    transformer = metrics[TRANSFORMER]; simple = metrics[SIMPLE]
    oracle = metrics[ORACLE]
    result["transformer_minus_simple_points"] = summary(
        [t-s for t, s in zip(transformer, simple)], instances)
    result["transformer_minus_oracle_points"] = summary(
        [t-o for t, o in zip(transformer, oracle)], instances)
    transformer_per = result["vs_balanced_percent"][TRANSFORMER]["per_instance"]
    result["transformer_positive_all_instances"] = all(
        row["mean"] > 0 for row in transformer_per)
    write_json(a.out/"oos"/"QUICK_RESULT.json", result)
    return result


def freeze_manifest(a):
    bundle = read_json(a.bundle); model = bundle["models"]["200"]
    checkpoints = [{"path": path, "sha256": digest(path)}
                   for path in model["checkpoints"]]
    manifest = {"protocol": "m-prescreen-frozen-model-v1",
        "model_execution_commit": MODEL_EXECUTION_COMMIT,
        "model_bundle": str(a.bundle.resolve()),
        "model_bundle_sha256": digest(a.bundle),
        "model_size_training_instances": 200,
        "training_instances": model["training_instances"],
        "validation_instances": bundle["validation_instances"],
        "training_seeds": bundle["training_seeds"],
        "epochs": bundle["epochs"], "batch_size": bundle["batch_size"],
        "width": bundle["width"], "loss": bundle["loss"],
        "normalization": str(a.normalization.resolve()),
        "normalization_sha256": digest(a.normalization),
        "collection_plan_sha256": bundle["collection_plan_sha256"],
        "source_search_plan_sha256": digest(a.source_search/"PLAN.json"),
        "source_search_complete_sha256": digest(a.source_search/"COMPLETE.json"),
        "checkpoints": checkpoints,
        "configuration": {"evaluation_mode": "ccp", "scenario_count": 100,
            "confidence": .9, "population": 20, "budget": 400,
            "candidate_count": 8}}
    identity = dict(manifest)
    manifest["manifest_digest"] = canonical_digest(identity)
    target = a.out/"FROZEN_MODEL_MANIFEST.json"
    if target.exists() and read_json(target) != manifest:
        raise ValueError("frozen model manifest changed")
    write_json(target, manifest)
    return manifest


def frozen_plan(a):
    manifest = freeze_manifest(a)
    catalog = read_json(a.search_instances/"catalog.json")
    identity = dict(catalog); claimed = identity.pop("catalog_digest", None)
    if canonical_digest(identity) != claimed or catalog.get("count") != 3:
        raise ValueError("search catalog changed")
    inputs = [a.data, a.normalization, a.bundle,
        a.source_search/"PLAN.json", a.source_search/"COMPLETE.json",
        a.search_instances/"catalog.json", Path(__file__).resolve(),
        SOURCE/"m_prescreen_final_oracle.py",
        SOURCE/"evaluate_m_directed_balanced_oos.py",
        SOURCE/"experiments/m_prescreen_protocol_v1.json"]
    inputs.extend(a.search_instances/f"{name}.json" for name in INSTANCES)
    inputs.extend(Path(row["path"]) for row in manifest["checkpoints"])
    return {"protocol": "m-prescreen-review-v1",
        "instances": list(INSTANCES), "algorithm_seeds": list(SEEDS),
        "existing_methods": list(SOURCE_METHODS),
        "new_methods": list(NEW_METHODS), "all_methods": list(METHODS),
        "new_search_count": 60, "scenario_count": 100,
        "oos_scenario_count": OOS_SCENARIO_COUNT, "oos_seed": OOS_SEED,
        "population": 20, "budget": 400, "ccp_confidence": .9,
        "workers": a.workers, "threads_per_worker": 1,
        "frozen_model_manifest_digest": manifest["manifest_digest"],
        "opens_locked_final_test": False,
        "input_sha256": {str(Path(path).resolve()): digest(path)
                         for path in inputs}}


def execute(a):
    root = a.out.resolve(); plan = frozen_plan(a)
    if (root/"PLAN.json").exists() and read_json(root/"PLAN.json") != plan:
        raise ValueError("review plan changed")
    write_json(root/"PLAN.json", plan)
    if (root/"PIPELINE_COMPLETE").exists():
        log("already complete; see QUICK_RESULT.json"); return
    plan_sha = digest(root/"PLAN.json")
    (root/"logs").mkdir(exist_ok=True)
    started = time.monotonic(); lock = threading.Lock(); stop = threading.Event()
    status = {"stage": "optimization_audit", "completed_new_runs": 0,
              "total_new_runs": 60}
    env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
           "OPENBLAS_NUM_THREADS": "1", "PYTHONUNBUFFERED": "1"}

    def group(name, seed):
        rows = []
        for method in NEW_METHODS:
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
                    "--normalization", str(a.normalization.resolve()),
                    "--bundle", str(a.bundle.resolve()),
                    "--source-search", str(a.source_search.resolve()),
                    "--out", str(folder),
                    "--instance", str((a.search_instances/f"{name}.json").resolve()),
                    "--seed", str(seed), "--method", method,
                    "--plan-sha", plan_sha]
                with logfile.open("w") as handle:
                    result = subprocess.run(command, cwd=SOURCE, env=env,
                        stdout=handle, stderr=subprocess.STDOUT, timeout=300)
                if result.returncode:
                    tail = "\n".join(logfile.read_text(
                        errors="replace").splitlines()[-20:])
                    raise RuntimeError(f"worker failed: {logfile}\n{tail}")
            row = load_run(folder, name, plan_sha); validate_audit_run(row, method)
            rows.append(row)
            with lock:
                status["completed_new_runs"] += 1
                status["elapsed_seconds"] = time.monotonic()-started
                write_json(root/"STATUS.json", status)
                log(f"{status['completed_new_runs']}/60 {'reused' if reused else 'complete'} {name}/{seed}/{method}")
        return rows

    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        futures = [pool.submit(group, name, seed) for name in INSTANCES
                   for seed in SEEDS]
        try:
            for future in as_completed(futures):
                future.result()
        except BaseException:
            stop.set()
            for future in futures: future.cancel()
            raise
    all_runs = load_all_runs(a, plan_sha)
    optimization = summarize_optimization(all_runs)
    write_json(root/"OPTIMIZATION_RESULT.json", optimization)
    with (root/"RESULT_TABLE.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle,
            fieldnames=list(optimization["table"][0]))
        writer.writeheader(); writer.writerows(optimization["table"])
    write_json(root/"STATUS.json", {**status, "stage": "independent_oos"})
    oos = run_oos(a)
    report = {"completed": True,
        "frozen_model_manifest": read_json(root/"FROZEN_MODEL_MANIFEST.json"),
        "optimization": optimization, "independent_oos": oos,
        "new_searches": 60,
        "new_search_objective_evaluations": 24_000,
        "oracle_additional_objective_evaluations": sum(
            row["k8_final_hv_audit"]["oracle_additional_objective_evaluations"]
            for row in all_runs if row["strategy"] == ORACLE),
        "new_training_steps": 0, "new_ppo_updates": 0,
        "elapsed_seconds_this_invocation": time.monotonic()-started,
        "review_checks_pass": bool(
            optimization["transformer_minus_simple_points"] > 0
            and oos["transformer_positive_all_instances"]),
        "next_action": "review results before any PPO"}
    write_json(root/"QUICK_RESULT.json", report)
    write_json(root/"STATUS.json", {**status, "stage": "complete",
        "review_checks_pass": report["review_checks_pass"]})
    (root/"PIPELINE_COMPLETE").write_text(
        f"review_checks_pass={report['review_checks_pass']}\n")
    log(f"COMPLETE review_checks_pass={report['review_checks_pass']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path,
        default=SOURCE/"data/data_expanded.xlsx")
    for name in ("search-instances", "normalization", "bundle",
                 "source-search", "out"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--workers", choices=(1, 2, 3), type=int, default=3)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--instance", type=Path)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--method", choices=NEW_METHODS)
    parser.add_argument("--plan-sha")
    a = parser.parse_args(); a.out = a.out.resolve()
    if a.worker:
        if any(getattr(a, name) is None for name in
               ("instance", "seed", "method", "plan_sha")):
            parser.error("worker arguments incomplete")
        a.out.mkdir(parents=True, exist_ok=False)
        run_audit_worker(a); return
    a.out.mkdir(parents=True, exist_ok=True)
    with (a.out/".lock").open("w") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try: execute(a)
        except Exception as exc:
            previous = (read_json(a.out/"STATUS.json")
                        if (a.out/"STATUS.json").exists() else {})
            write_json(a.out/"STATUS.json", {
                **previous, "stage": "stopped", "error": str(exc)})
            log(f"STOPPED {exc}"); raise


if __name__ == "__main__":
    main()
