#!/usr/bin/env python3
"""Bounded K8 greedy-oracle final-HV audit; no model training or final test."""
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
from m_prescreen_final_oracle import (METHODS, FinalOracleController,
                                      FinalOraclePolicy)
from run_learning_components_deterministic import _normalizer
from run_m_operator_closed_loop_v1 import load_run
from transformer_ppo.episode import run_episode
from transformer_ppo.instance_catalog import canonical_digest, read_instance


INSTANCES = ("train-S0-021", "train-S0-022", "train-S0-023")
SEEDS = tuple(range(915001, 915011))
MINIMUM_ORACLE_MEAN_PERCENT = 1.0


def log(message):
    print(f"[M-K8-FINAL] {message}", flush=True)


def validate_run(row, method=None):
    audit = row.get("k8_final_hv_audit", {})
    if method is not None and row.get("strategy") != method:
        raise ValueError("K8 final-HV method identity mismatch")
    if (row.get("real_objective_evaluation_count") != 400
            or row.get("evaluation_budget") != 400
            or row.get("ppo_updates", 0)
            or row.get("training_feedback_evaluations", 0)
            or row.get("candidate_audit_evaluations", 0)
            or audit.get("method") != row.get("strategy")
            or audit.get("pool_events", 0) < 1
            or audit.get("pool_candidates", 0) < audit.get("pool_events", 0)
            or not math.isfinite(row.get("final_hypervolume", math.nan))
            or row.get("final_hypervolume", 0.0) <= 0.0):
        raise ValueError("invalid K8 final-HV audit run")
    extra = int(audit.get("oracle_additional_objective_evaluations", -1))
    pooled = int(audit.get("oracle_pool_objective_evaluations", -1))
    candidates = int(audit.get("pool_candidates", -1))
    expected_extra = candidates - int(audit["pool_events"])
    if row["strategy"] == "K8_ORACLE":
        if pooled != candidates or extra != expected_extra or extra <= 0:
            raise ValueError("oracle objective-evaluation accounting mismatch")
    elif pooled or extra:
        raise ValueError("deployable audit baseline evaluated hidden candidates")
    if audit.get("total_physical_objective_evaluations") != 400 + extra:
        raise ValueError("physical objective-evaluation total mismatch")


def run_one(a):
    torch.set_num_threads(1)
    info, _ = read_instance(a.instance)
    policy = FinalOraclePolicy(info["instance_id"], a.seed, a.method)
    holders = []

    def factory(*args, **kwargs):
        controller = FinalOracleController(*args, **kwargs)
        holders.append(controller)
        return controller

    result = run_episode(
        a.data, a.instance, a.out, policy, _normalizer(a.normalization),
        a.seed, evaluation_budget=400, pop_size=20, generations=1000,
        evaluation_mode="ccp", component="M", scenario_count=100,
        ccp_confidence=.9,
        uncertainty_profile="transport_border_saa_v1",
        inference_controller_factories={"M": factory})
    if len(holders) != 1:
        raise RuntimeError("K8 final-HV controller was not installed exactly once")
    result.update(
        strategy=a.method, algorithm_seed=a.seed,
        plan_sha256=a.plan_sha, new_training_steps=0,
        new_ppo_updates=0, learning_enabled=False,
        role=("diagnostic greedy one-step oracle upper-bound audit"
              if a.method == "K8_ORACLE" else
              "evaluation-free comparison policy"))
    validate_run(result, a.method)
    write_json(a.out / "RUN_COMPLETE.json", result)


def summarize(runs, instances=INSTANCES, seeds=SEEDS):
    keyed = {(row["instance"], row["algorithm_seed"], row["strategy"]): row
             for row in runs}
    expected = {(name, seed, method) for name in instances for seed in seeds
                for method in METHODS}
    if len(keyed) != len(runs) or set(keyed) != expected:
        raise ValueError("incomplete or duplicate K8 final-HV grid")
    paired, table, per_instance = [], [], []
    for name in instances:
        gains = {method: [] for method in METHODS}
        for seed in seeds:
            baseline = keyed[name, seed, "BALANCED"]
            for method in METHODS:
                row = keyed[name, seed, method]
                validate_run(row, method)
                if row["pairing"] != baseline["pairing"]:
                    raise ValueError("initial population/scenarios/B400 pairing mismatch")
                gain = 100.0 * (row["final_hypervolume"] /
                                baseline["final_hypervolume"] - 1.0)
                gains[method].append(gain)
                paired.append({"instance": name, "algorithm_seed": seed,
                    "method": method, "paired_percent_vs_balanced": gain,
                    "final_hypervolume": row["final_hypervolume"]})
        method_rows = {}
        for method in METHODS:
            selected = [keyed[name, seed, method] for seed in seeds]
            values = gains[method]
            item = {"instance": name, "method": method,
                "runs": len(selected),
                "mean_hv": statistics.fmean(
                    row["final_hypervolume"] for row in selected),
                "mean_paired_percent_vs_balanced": statistics.fmean(values),
                "wins_vs_balanced": sum(value > 1e-12 for value in values),
                "mean_seconds": statistics.fmean(
                    row["optimisation_seconds"] for row in selected),
                "mean_additional_objective_evaluations": statistics.fmean(
                    row["k8_final_hv_audit"][
                        "oracle_additional_objective_evaluations"]
                    for row in selected)}
            table.append(item)
            method_rows[method] = item
        per_instance.append({"instance": name,
            "simple_mean_percent": method_rows["K8_SIMPLE"][
                "mean_paired_percent_vs_balanced"],
            "oracle_mean_percent": method_rows["K8_ORACLE"][
                "mean_paired_percent_vs_balanced"],
            "oracle_wins": method_rows["K8_ORACLE"]["wins_vs_balanced"]})
    oracle_means = [row["oracle_mean_percent"] for row in per_instance]
    simple_means = [row["simple_mean_percent"] for row in per_instance]
    oracle_mean = statistics.fmean(oracle_means)
    simple_mean = statistics.fmean(simple_means)
    passed = (oracle_mean >= MINIMUM_ORACLE_MEAN_PERCENT
              and all(value >= 0.0 for value in oracle_means))
    return {"completed": True,
        "audit_role": ("greedy one-step true-CCP100 candidate oracle; "
                       "diagnostic and not deployable at equal evaluation cost"),
        "pairing_exact": True,
        "per_instance": per_instance,
        "pooled_equal_instance": {
            "simple_mean_paired_percent_vs_balanced": simple_mean,
            "oracle_mean_paired_percent_vs_balanced": oracle_mean,
            "minimum_oracle_mean_percent": MINIMUM_ORACLE_MEAN_PERCENT},
        "table": table, "paired_runs": paired,
        "oracle_final_hv_screen_pass": passed,
        "conclusion": ("EXPAND_K8_TRAINING_INSTANCES"
                       if passed else "STOP_K8_PRESCREEN_INSUFFICIENT_FINAL_HV_UPPER_BOUND"),
        "next_action": ("collect 25/50/100/200-instance learning curve with frozen small ranker"
                        if passed else
                        "do not expand K8 data or train another prescreen model")}


def frozen_plan(a):
    protocol = read_json(SOURCE / "experiments/m_prescreen_protocol_v1.json")
    audit = protocol["final_hv_oracle_audit"]
    if (audit["instances"] != list(INSTANCES)
            or audit["algorithm_seeds"] != list(SEEDS)
            or audit["methods"] != list(METHODS)
            or audit["pass"] !=
            "equal-instance oracle mean >=1 percent and every instance nonnegative"):
        raise ValueError("K8 final-HV audit protocol changed")
    inputs = [a.data.resolve(), a.normalization.resolve(),
        SOURCE / "m_prescreen_k8.py",
        SOURCE / "m_prescreen_final_oracle.py", Path(__file__).resolve(),
        SOURCE / "experiments/m_prescreen_protocol_v1.json",
        SOURCE / "data/border_crossing_events.csv"]
    seen = set()
    for name in INSTANCES:
        path = (a.instances / f"{name}.json").resolve()
        info, _ = read_instance(path)
        signature = canonical_digest(info["batches"])
        if (info["instance_id"] != name
                or info["source_data_sha256"] != digest(a.data)
                or info["configuration"] != "S0" or signature in seen):
            raise ValueError(f"invalid or duplicate development instance: {name}")
        seen.add(signature)
        inputs.append(path)
    return {"protocol": "m-prescreen-k8-final-hv-oracle-v1",
        "instances": list(INSTANCES), "algorithm_seeds": list(SEEDS),
        "methods": list(METHODS), "run_count": len(INSTANCES)*len(SEEDS)*len(METHODS),
        "scenario_count": 100, "ccp_confidence": .9,
        "population_size": 20, "search_evaluation_budget": 400,
        "workers": a.workers, "threads_per_worker": 1,
        "run_timeout_seconds": 300, "invocation_limit_seconds": 3600,
        "automatic_retries": 0,
        "simple_score": "negative mean pre-evaluation local nominal cost/emission/time deltas",
        "oracle_score": "exact one-step current-population HV contribution under frozen CCP100",
        "oracle_accounting": ("selected candidate is one B400 evaluation; every other pool "
                              "evaluation is separately reported diagnostic cost"),
        "pass_rule": audit["pass"],
        "development_only": True, "opens_locked_final_test": False,
        "data": str(a.data.resolve()), "normalization": str(a.normalization.resolve()),
        "instances_root": str(a.instances.resolve()),
        "inputs_sha256": {str(path): digest(path) for path in inputs},
        "torch_version": str(torch.__version__),
        "numpy_version": str(np.__version__)}


def execute(a):
    root = a.out.resolve()
    plan = frozen_plan(a)
    if (root / "PLAN.json").exists() and read_json(root / "PLAN.json") != plan:
        raise ValueError("frozen K8 final-HV inputs changed; retain old output")
    write_json(root / "PLAN.json", plan)
    if (root / "PIPELINE_COMPLETE").exists():
        log("already complete; see QUICK_RESULT.json and RESULT_TABLE.csv")
        return
    plan_sha = digest(root / "PLAN.json")
    (root / "logs").mkdir(exist_ok=True)
    started, deadline = time.monotonic(), time.monotonic()+3600
    status = {"stage": "search", "completed_runs": 0,
              "total_runs": plan["run_count"]}
    lock, stop = threading.Lock(), threading.Event()
    env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
           "OPENBLAS_NUM_THREADS": "1", "PYTHONUNBUFFERED": "1"}
    log(f"{plan['run_count']} searches: 3 development instances x 10 seeds x 3 methods; CCP100/pop20/B400")

    def group(name, seed):
        rows = []
        offset = SEEDS.index(seed) % len(METHODS)
        order = METHODS[offset:] + METHODS[:offset]
        for method in order:
            if stop.is_set() or time.monotonic() >= deadline:
                raise TimeoutError("one-hour invocation limit reached; completed runs retained")
            folder = root / "runs" / name / f"seed_{seed}" / method
            reused = (folder / "RUN_COMPLETE.json").exists()
            if not reused:
                if folder.exists():
                    quarantine = root / "interrupted_runs" / (
                        f"{name}_{seed}_{method}_{time.time_ns()}")
                    quarantine.parent.mkdir(parents=True, exist_ok=True)
                    folder.rename(quarantine)
                logfile = root / "logs" / f"{name}_{seed}_{method}_{time.time_ns()}.log"
                command = [sys.executable, "-u", str(Path(__file__).resolve()),
                    "--worker", "--data", str(a.data.resolve()),
                    "--instances", str(a.instances.resolve()),
                    "--normalization", str(a.normalization.resolve()),
                    "--instance", str((a.instances/f"{name}.json").resolve()),
                    "--out", str(folder), "--seed", str(seed),
                    "--method", method, "--plan-sha", plan_sha]
                with logfile.open("w") as handle, subprocess.Popen(
                        command, cwd=SOURCE, env=env, stdout=handle,
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
                write_json(root / "STATUS.json", status)
                audit = row["k8_final_hv_audit"]
                log(f"{status['completed_runs']}/{plan['run_count']} {'reused' if reused else 'complete'} {name}/{seed}/{method} HV={row['final_hypervolume']:.7g} extra={audit['oracle_additional_objective_evaluations']}")
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
    report.update(design=plan,
        new_searches=len(runs), new_training_steps=0, new_ppo_updates=0,
        search_objective_evaluations=sum(
            row["real_objective_evaluation_count"] for row in runs),
        oracle_additional_objective_evaluations=sum(
            row["k8_final_hv_audit"]["oracle_additional_objective_evaluations"]
            for row in runs),
        elapsed_seconds_this_invocation=time.monotonic()-started)
    write_json(root / "COMPLETE.json", {**report, "runs": runs})
    write_json(root / "QUICK_RESULT.json", report)
    with (root / "RESULT_TABLE.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(report["table"][0]))
        writer.writeheader(); writer.writerows(report["table"])
    write_json(root / "STATUS.json", {**status, "stage": "complete",
        "conclusion": report["conclusion"]})
    (root / "PIPELINE_COMPLETE").write_text(report["conclusion"]+"\n")
    print((root / "RESULT_TABLE.csv").read_text(), flush=True)
    log(f"COMPLETE oracle={report['pooled_equal_instance']['oracle_mean_paired_percent_vs_balanced']:+.3f}% {report['conclusion']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path,
        default=SOURCE / "data/data_expanded.xlsx")
    parser.add_argument("--instances", type=Path, required=True)
    parser.add_argument("--normalization", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", choices=(1, 2, 3), default=3, type=int)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--instance", type=Path)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--method", choices=METHODS)
    parser.add_argument("--plan-sha")
    a = parser.parse_args()
    a.out = a.out.resolve()
    if a.worker:
        if any(getattr(a, key) is None for key in
               ("instance", "seed", "method", "plan_sha")):
            parser.error("worker arguments incomplete")
        a.out.mkdir(parents=True, exist_ok=False)
        run_one(a)
        return
    a.out.mkdir(parents=True, exist_ok=True)
    with (a.out / ".lock").open("w") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            execute(a)
        except Exception as exc:
            previous = (read_json(a.out / "STATUS.json")
                        if (a.out / "STATUS.json").exists() else {})
            write_json(a.out / "STATUS.json", {
                **previous, "stage": "stopped", "error": str(exc)})
            log(f"STOPPED {exc}")
            raise


if __name__ == "__main__":
    main()
