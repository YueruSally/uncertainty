#!/usr/bin/env python3
"""Frozen C-K8 Transformer development search against Base and fixed controls."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import threading
import time

import numpy as np
import torch

SOURCE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE))

from c_k8_scale_policy import FrozenCCrossoverRanker
from m_operator_offline import digest, read_json, write_json
from run_learning_components_deterministic import _normalizer
from transformer_ppo.crossover_k8 import CrossoverK8Controller
from transformer_ppo.episode import run_episode
from transformer_ppo.instance_catalog import canonical_digest, read_instance


BASE = "BASE"; DIRECTED = "C_DIRECTED_EARLY04"
SIMPLE = "C_K8_SIMPLE"; TRANSFORMER = "C_K8_TRANSFORMER"
ORACLE = "C_K8_ORACLE"
METHODS = (BASE, DIRECTED, SIMPLE, TRANSFORMER, ORACLE)
SOURCE_METHODS = (BASE, DIRECTED, SIMPLE, ORACLE)
INSTANCES = tuple(f"prescreen-search-S0-{index:03d}" for index in range(1, 4))
SEEDS = tuple(range(919001, 919011))


def log(message):
    print(f"[C-K8-SEARCH] {message}", flush=True)


def worker(a):
    torch.set_num_threads(1)
    instance, _ = read_instance(a.instance)
    predictor = FrozenCCrossoverRanker(read_json(a.bundle), digest)
    holders = []

    def factory(*args, **kwargs):
        controller = CrossoverK8Controller(
            *args, selection_mode="transformer", pool_size=8,
            selection_policy=predictor, **kwargs)
        holders.append(controller); return controller

    result = run_episode(
        a.data, a.instance, a.out, None, _normalizer(a.normalization),
        algorithm_seed=a.seed, evaluation_budget=400, pop_size=20,
        generations=1000, evaluation_mode="ccp", component="BASE",
        scenario_count=100, ccp_confidence=.9,
        uncertainty_profile="transport_border_saa_v1",
        inference_controller_factories={"BASE": factory},
        method_override=TRANSFORMER)
    if len(holders) != 1:
        raise RuntimeError("C-K8 Transformer controller was not installed once")
    result.update({
        "strategy": TRANSFORMER, "instance": instance["instance_id"],
        "algorithm_seed": a.seed, "plan_sha256": a.plan_sha,
        "scenario_digest": read_json(a.out / "configuration.json")["scenario_digest"],
        "new_training_steps": 0, "new_ppo_updates": 0,
        "learning_enabled": False,
        "role": "bounded frozen C-K8 supervised Transformer development search",
    })
    validate_transformer_run(result, a.seed, a.plan_sha)
    write_json(a.out / "RUN_COMPLETE.json", result)


def validate_transformer_run(row, seed, plan_sha):
    eligible = int(row.get("crossover_k8_eligible_pool_events", 0))
    if (row.get("completed") is not True
            or row.get("strategy") != TRANSFORMER
            or row.get("algorithm_seed") != seed
            or row.get("plan_sha256") != plan_sha
            or row.get("real_objective_evaluation_count") != 400
            or row.get("evaluation_budget") != 400
            or row.get("crossover_k8_selection_mode") != "transformer"
            or eligible <= 0
            or row.get("crossover_k8_candidate_objective_evaluations")
            != 2 * eligible
            or row.get("crossover_k8_hidden_candidate_objective_evaluations") != 0
            or row.get("ppo_updates", 0)
            or not math.isfinite(float(row.get("final_hypervolume", math.nan)))
            or float(row.get("final_hypervolume", 0.0)) <= 0.0):
        raise ValueError("invalid frozen C-K8 Transformer run")


def interval(grouped, seed):
    names = sorted(grouped); matrix = np.asarray([grouped[name] for name in names])
    generator = np.random.default_rng(seed)
    outer = generator.integers(0, matrix.shape[0], (50_000, matrix.shape[0]))
    inner = generator.integers(
        0, matrix.shape[1], (50_000, matrix.shape[0], matrix.shape[1]))
    means = matrix[outer[:, :, None], inner].mean((1, 2))
    return float(np.quantile(means, .025)), float(np.quantile(means, .975))


def paired_summary(rows, key, seed):
    grouped = {}
    for row in rows: grouped.setdefault(row["instance"], []).append(float(row[key]))
    values = [value for name in sorted(grouped) for value in grouped[name]]
    low, high = interval(grouped, seed)
    return {"mean": statistics.fmean(statistics.fmean(grouped[name])
                                     for name in sorted(grouped)),
            "hierarchical_bootstrap_95ci_low": low,
            "hierarchical_bootstrap_95ci_high": high,
            "wins": sum(v > 1e-12 for v in values),
            "ties": sum(abs(v) <= 1e-12 for v in values),
            "losses": sum(v < -1e-12 for v in values),
            "per_instance": [{"instance": name,
                "mean": statistics.fmean(grouped[name]),
                "wins": sum(v > 1e-12 for v in grouped[name]),
                "ties": sum(abs(v) <= 1e-12 for v in grouped[name]),
                "losses": sum(v < -1e-12 for v in grouped[name])}
                for name in sorted(grouped)]}


def summarize(runs):
    keyed = {(r["instance"], r["algorithm_seed"], r["strategy"]): r for r in runs}
    expected = {(name, seed, method) for name in INSTANCES
                for seed in SEEDS for method in METHODS}
    if len(keyed) != len(runs) or set(keyed) != expected:
        raise ValueError("incomplete or duplicate C-K8 search grid")
    paired = []
    for name in INSTANCES:
        for seed in SEEDS:
            baseline = keyed[name, seed, BASE]
            simple = keyed[name, seed, SIMPLE]
            directed = keyed[name, seed, DIRECTED]
            pairing = baseline.get("pairing")
            for method in METHODS:
                row = keyed[name, seed, method]
                if (row["scenario_digest"] != baseline["scenario_digest"]
                        or row.get("pairing") != pairing):
                    raise ValueError("C-K8 pairing or scenarios changed")
                paired.append({"instance": name, "algorithm_seed": seed,
                    "strategy": method,
                    "percent_vs_base": 100 * (row["final_hypervolume"]
                        / baseline["final_hypervolume"] - 1),
                    "percent_vs_simple": 100 * (row["final_hypervolume"]
                        / simple["final_hypervolume"] - 1),
                    "percent_vs_directed": 100 * (row["final_hypervolume"]
                        / directed["final_hypervolume"] - 1)})
    methods = {}; table = []
    for index, method in enumerate(METHODS):
        selected = [r for r in paired if r["strategy"] == method]
        actual = [r for r in runs if r["strategy"] == method]
        methods[method] = {
            "vs_base": paired_summary(selected, "percent_vs_base", 925100+index),
            "vs_simple": paired_summary(selected, "percent_vs_simple", 925200+index),
            "vs_directed": paired_summary(selected, "percent_vs_directed", 925300+index),
            "mean_seconds": statistics.fmean(r["optimisation_seconds"] for r in actual),
            "logical_objective_evaluations": sum(
                r["real_objective_evaluation_count"] for r in actual),
            "hidden_candidate_objective_evaluations": sum(
                r.get("crossover_k8_hidden_candidate_objective_evaluations", 0)
                for r in actual),
        }
        table.append({"method": method,
            "mean_percent_vs_base": methods[method]["vs_base"]["mean"],
            "mean_percent_vs_simple": methods[method]["vs_simple"]["mean"],
            "mean_seconds": methods[method]["mean_seconds"],
            "logical_evaluations": methods[method]["logical_objective_evaluations"],
            "hidden_evaluations": methods[method]["hidden_candidate_objective_evaluations"]})
    transformer = methods[TRANSFORMER]
    positive_base = sum(r["mean"] > 0 for r in transformer["vs_base"]["per_instance"])
    positive_simple = sum(r["mean"] > 0 for r in transformer["vs_simple"]["per_instance"])
    passed = bool(transformer["vs_base"]["mean"] > 0
        and transformer["vs_base"]["hierarchical_bootstrap_95ci_low"] > 0
        and positive_base >= 2 and transformer["vs_simple"]["mean"] > 0
        and positive_simple >= 2)
    return {"completed": True, "reference_method": BASE,
        "methods": methods, "table": table, "paired_runs": paired,
        "transformer_positive_instances_vs_base": positive_base,
        "transformer_positive_instances_vs_simple": positive_simple,
        "development_search_pass": passed,
        "conclusion": ("ADMIT_C_K8_TRANSFORMER_TO_BOUNDED_PPO" if passed else
                       "C_K8_TRANSFORMER_DID_NOT_ESTABLISH_GAIN_OVER_BASE_AND_SIMPLE"),
        "next_action": ("retain supervised ablation and add bounded PPO; final test remains closed"
                        if passed else "stop before PPO and do not open final test")}


def source_runs(a):
    quick = read_json(a.oracle_root / "QUICK_RESULT.json")
    rows = read_json(a.oracle_root / "COMPLETE.json").get("runs", ())
    keyed = {(r["instance"], r["algorithm_seed"], r["strategy"]): r for r in rows}
    expected = {(name, seed, method) for name in INSTANCES
                for seed in SEEDS for method in SOURCE_METHODS}
    if quick.get("oracle_final_hv_screen_pass") is not True or set(keyed) != expected:
        raise ValueError("source C-K8 oracle result is incomplete")
    return list(rows)


def frozen_plan(a):
    source = source_runs(a); bundle = read_json(a.bundle)
    selected = bundle.get("models", {}).get("200", {})
    if (bundle.get("protocol") != "c-k8-scale-transformer-curve-v1"
            or bundle.get("selected_size") != 200
            or len(selected.get("checkpoints", ())) != 3
            or not (a.training_root / "TRAINING_COMPLETE").exists()):
        raise ValueError("C-K8 size-200 Transformer is not frozen")
    catalog = read_json(a.search_instances / "catalog.json")
    identity = dict(catalog); claimed = identity.pop("catalog_digest", None)
    if (canonical_digest(identity) != claimed or catalog.get("count") != 3
            or catalog.get("final_test_instances_used") is not False
            or [r["instance_id"] for r in catalog["instances"]] != list(INSTANCES)):
        raise ValueError("C-K8 search catalog changed")
    inputs = [a.data.resolve(), a.normalization.resolve(), a.bundle.resolve(),
        a.oracle_root / "PLAN.json", a.oracle_root / "QUICK_RESULT.json",
        a.oracle_root / "COMPLETE.json", a.search_instances / "catalog.json",
        Path(__file__).resolve(), SOURCE / "i_k8_surrogate.py",
        SOURCE / "c_k8_scale_policy.py", SOURCE / "transformer_ppo/crossover_k8.py",
        SOURCE / "data/border_crossing_events.csv"]
    inputs.extend(Path(path) for path in selected["checkpoints"])
    inputs.extend(a.search_instances / f"{name}.json" for name in INSTANCES)
    return {"protocol": "c-k8-transformer-bounded-search-v1",
        "reference_method": BASE, "instances": list(INSTANCES),
        "algorithm_seeds": list(SEEDS), "methods": list(METHODS),
        "new_method_runs": 30, "reused_source_runs": len(source),
        "scenario_count": 100, "ccp_confidence": .9,
        "population_size": 20, "evaluation_budget": 400,
        "candidate_pool": 8, "model_training_instances": 200,
        "model_selection": "fixed epoch 30 three-seed ensemble",
        "workers": a.workers, "threads_per_worker": 1,
        "run_timeout_seconds": 600, "invocation_limit_seconds": 3600,
        "hidden_candidate_evaluations_transformer": 0,
        "development_only": True, "opens_locked_final_test": False,
        "input_sha256": {str(p.resolve()): digest(p) for p in inputs},
        "torch_version": str(torch.__version__), "numpy_version": str(np.__version__)}


def execute(a):
    root = a.out.resolve(); plan = frozen_plan(a)
    plan_path = root / "PLAN.json"
    if plan_path.exists() and read_json(plan_path) != plan:
        raise ValueError("frozen C-K8 search inputs changed")
    write_json(plan_path, plan)
    if (root / "PIPELINE_COMPLETE").exists():
        log("already complete; see QUICK_RESULT.json"); return
    plan_sha = digest(plan_path); (root / "logs").mkdir(exist_ok=True)
    started = time.monotonic(); deadline = started + 3600
    status = {"stage": "search", "completed_runs": 0, "total_runs": 30}
    lock = threading.Lock(); stop = threading.Event()
    environment = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                   "OPENBLAS_NUM_THREADS": "1", "PYTHONUNBUFFERED": "1"}
    log("30 new Transformer searches; reuse 120 frozen oracle-screen runs")

    def task(name, seed):
        folder = root / "runs" / name / f"seed_{seed}" / TRANSFORMER
        reused = (folder / "RUN_COMPLETE.json").exists()
        if not reused:
            if folder.exists():
                quarantine = root / "interrupted_runs" / f"{name}_{seed}_{time.time_ns()}"
                quarantine.parent.mkdir(parents=True, exist_ok=True); shutil.move(folder, quarantine)
            logfile = root / "logs" / f"{name}_{seed}_{time.time_ns()}.log"
            command = [sys.executable, "-u", str(Path(__file__).resolve()),
                "--worker", "--data", str(a.data.resolve()),
                "--search-instances", str(a.search_instances.resolve()),
                "--training-root", str(a.training_root.resolve()),
                "--normalization", str(a.normalization.resolve()),
                "--bundle", str(a.bundle.resolve()),
                "--oracle-root", str(a.oracle_root.resolve()),
                "--out", str(folder),
                "--instance", str((a.search_instances / f"{name}.json").resolve()),
                "--seed", str(seed), "--plan-sha", plan_sha]
            with logfile.open("w") as handle, subprocess.Popen(
                    command, cwd=SOURCE, env=environment,
                    stdout=handle, stderr=subprocess.STDOUT) as process:
                begin = time.monotonic()
                while process.poll() is None:
                    now = time.monotonic()
                    if stop.is_set() or now >= deadline or now - begin >= 600:
                        process.kill(); process.wait()
                        raise TimeoutError(f"run stopped; see {logfile}")
                    time.sleep(.2)
                if process.returncode:
                    tail = "\n".join(logfile.read_text(errors="replace").splitlines()[-25:])
                    raise RuntimeError(f"worker failed: {logfile}\n{tail}")
        row = read_json(folder / "RUN_COMPLETE.json")
        validate_transformer_run(row, seed, plan_sha)
        with lock:
            status["completed_runs"] += 1; status["elapsed_seconds"] = time.monotonic()-started
            write_json(root / "STATUS.json", status)
            log(f"{status['completed_runs']}/30 {'reused' if reused else 'complete'} "
                f"{name}/{seed} HV={row['final_hypervolume']:.7g}")
        return row

    transformer_runs = []
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        futures = [pool.submit(task, name, seed) for name in INSTANCES for seed in SEEDS]
        try:
            for future in as_completed(futures): transformer_runs.append(future.result())
        except BaseException:
            stop.set()
            for future in futures: future.cancel()
            raise
    runs = source_runs(a) + transformer_runs; report = summarize(runs)
    report.update({"design": plan, "new_searches": len(transformer_runs),
        "reused_searches": len(runs)-len(transformer_runs),
        "new_training_steps": 0, "new_ppo_updates": 0,
        "new_search_objective_evaluations": 400*len(transformer_runs),
        "new_hidden_candidate_objective_evaluations": 0,
        "elapsed_seconds_this_invocation": time.monotonic()-started})
    write_json(root / "COMPLETE.json", {**report, "runs": runs})
    write_json(root / "QUICK_RESULT.json", {
        k: v for k, v in report.items() if k not in ("runs", "paired_runs")})
    with (root / "RESULT_TABLE.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(report["table"][0]))
        writer.writeheader(); writer.writerows(report["table"])
    write_json(root / "STATUS.json", {**status, "stage": "complete",
                                      "conclusion": report["conclusion"]})
    (root / "PIPELINE_COMPLETE").write_text(report["conclusion"] + "\n")
    print((root / "RESULT_TABLE.csv").read_text(), flush=True)
    log(f"COMPLETE {report['conclusion']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=SOURCE / "data/data_expanded.xlsx")
    for name in ("search-instances", "training-root", "normalization",
                 "bundle", "oracle-root", "out"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--workers", choices=(1,2,3,4), default=3, type=int)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--instance", type=Path); parser.add_argument("--seed", type=int)
    parser.add_argument("--plan-sha")
    a = parser.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
    if a.worker:
        if None in (a.instance, a.seed, a.plan_sha): parser.error("worker arguments incomplete")
        worker(a); return
    execute(a)


if __name__ == "__main__": main()
