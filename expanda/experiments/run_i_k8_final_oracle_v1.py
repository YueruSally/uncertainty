#!/usr/bin/env python3
"""Final-HV upper-bound screen for K8 complete-individual initialization."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import hashlib
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

SOURCE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE))

from multi_instance_logging import write_json
from run_learning_components_deterministic import _normalizer
from transformer_ppo.episode import run_episode
from transformer_ppo.initialization_k8 import InitializationK8Controller
from transformer_ppo.instance_catalog import canonical_digest


BASE = "BASE"
BALANCED = "I_DIRECTED_BALANCED"
SIMPLE = "I_K8_SIMPLE"
ORACLE = "I_K8_ORACLE"
METHODS = (BASE, BALANCED, SIMPLE, ORACLE)
DEFAULT_SEEDS = tuple(range(918001, 918011))


def log(message):
    print(f"[I-K8-ORACLE] {message}", flush=True)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def worker(a):
    factory = None; component = "BASE"; profile = "legacy"
    if a.method in (BALANCED, SIMPLE, ORACLE):
        mode = {
            BALANCED: "balanced", SIMPLE: "simple", ORACLE: "oracle",
        }[a.method]

        def factory(*args, **kwargs):
            return InitializationK8Controller(
                *args, selection_mode=mode, pool_size=8, **kwargs)

    result = run_episode(
        a.data, a.instance, a.out, None, _normalizer(a.normalization),
        algorithm_seed=a.seed, evaluation_budget=400, pop_size=20,
        generations=1000, evaluation_mode="ccp", component=component,
        scenario_count=100, ccp_confidence=.9,
        uncertainty_profile="transport_border_saa_v1",
        rule_initialization_profile=profile,
        inference_controller_factories=(
            {"BASE": factory} if factory is not None else None),
        method_override=(a.method if factory is not None else None))
    result.update({
        "strategy": a.method,
        "instance": read_json(a.instance)["instance_id"],
        "algorithm_seed": a.seed,
        "plan_sha256": a.plan_sha,
        "scenario_digest": read_json(a.out / "configuration.json")[
            "scenario_digest"],
        "logical_search_objective_evaluations": int(
            result["real_objective_evaluation_count"]),
        "hidden_candidate_objective_evaluations": int(result.get(
            "initialization_k8_hidden_candidate_objective_evaluations", 0)),
        "physical_objective_evaluations": int(
            result["real_objective_evaluation_count"] + result.get(
                "initialization_k8_hidden_candidate_objective_evaluations", 0)),
    })
    if a.method in (BALANCED, SIMPLE, ORACLE):
        expected_candidates = 160 if a.method == ORACLE else 20
        expected_hidden = 140 if a.method == ORACLE else 0
        if (result.get("initialization_k8_pool_events") != 20
                or result.get(
                    "initialization_k8_candidate_objective_evaluations")
                != expected_candidates
                or result["hidden_candidate_objective_evaluations"]
                != expected_hidden):
            raise RuntimeError("I-K8 did not control all 20 initialization slots")
    if (result["logical_search_objective_evaluations"] != 400
            or result.get(
                "initialization_k8_candidate_construction_objective_evaluations",
                0) != 0):
        raise RuntimeError("invalid I-K8 evaluation accounting")
    write_json(a.out / "RUN_COMPLETE.json", result)


def valid_run(path, method, seed, plan_sha):
    marker = path / "RUN_COMPLETE.json"
    if not marker.is_file():
        return False
    try:
        row = read_json(marker)
        k8_accounting = True
        if method in (BALANCED, SIMPLE, ORACLE):
            k8_accounting = bool(
                row.get("initialization_k8_pool_events") == 20
                and row.get(
                    "initialization_k8_candidate_objective_evaluations")
                == (160 if method == ORACLE else 20)
                and row.get("hidden_candidate_objective_evaluations")
                == (140 if method == ORACLE else 0))
        return bool(
            row.get("completed") is True
            and row.get("strategy") == method
            and row.get("algorithm_seed") == seed
            and row.get("plan_sha256") == plan_sha
            and row.get("evaluation_budget") == 400
            and row.get("logical_search_objective_evaluations") == 400
            and row.get("initialization_k8_candidate_construction_objective_evaluations", 0) == 0
            and k8_accounting
            and math.isfinite(float(row.get("final_hypervolume", math.nan)))
            and float(row["final_hypervolume"]) > 0.0)
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return False


def hierarchical_interval(grouped, seed):
    names = sorted(grouped); matrix = np.asarray([grouped[name] for name in names])
    if matrix.ndim != 2 or matrix.shape[0] < 2:
        raise ValueError("hierarchical interval requires a balanced instance grid")
    generator = np.random.default_rng(seed)
    outer = generator.integers(0, matrix.shape[0], (50_000, matrix.shape[0]))
    inner = generator.integers(
        0, matrix.shape[1], (50_000, matrix.shape[0], matrix.shape[1]))
    sampled = matrix[outer[:, :, None], inner]
    means = sampled.mean((1, 2))
    return [float(np.quantile(means, .025)),
            float(np.quantile(means, .975))]


def paired_summary(rows, key, seed):
    grouped = {}
    for row in rows:
        grouped.setdefault(row["instance"], []).append(float(row[key]))
    values = [value for name in sorted(grouped) for value in grouped[name]]
    interval = hierarchical_interval(grouped, seed)
    return {
        "mean": statistics.fmean(
            statistics.fmean(grouped[name]) for name in sorted(grouped)),
        "hierarchical_bootstrap_95ci_low": interval[0],
        "hierarchical_bootstrap_95ci_high": interval[1],
        "wins": sum(value > 1e-12 for value in values),
        "ties": sum(abs(value) <= 1e-12 for value in values),
        "losses": sum(value < -1e-12 for value in values),
        "per_instance": [{
            "instance": name,
            "mean": statistics.fmean(grouped[name]),
            "wins": sum(value > 1e-12 for value in grouped[name]),
            "ties": sum(abs(value) <= 1e-12 for value in grouped[name]),
            "losses": sum(value < -1e-12 for value in grouped[name]),
        } for name in sorted(grouped)],
    }


def summarize(rows, instances, seeds):
    keyed = {(row["instance"], row["algorithm_seed"], row["strategy"]): row
             for row in rows}
    expected = {(name, seed, method) for name in instances
                for seed in seeds for method in METHODS}
    if len(keyed) != len(rows) or set(keyed) != expected:
        raise ValueError("I-K8 result grid is incomplete or duplicated")
    scenario_failures = []
    paired = []
    for name in instances:
        for seed in seeds:
            base_row = keyed[name, seed, BASE]
            balanced = keyed[name, seed, BALANCED]
            for method in METHODS:
                row = keyed[name, seed, method]
                if row["scenario_digest"] != base_row["scenario_digest"]:
                    scenario_failures.append(f"{name}:{seed}:{method}")
                paired.append({
                    "instance": name, "algorithm_seed": seed,
                    "strategy": method,
                    "percent_vs_base": 100 * (
                        row["final_hypervolume"] / base_row["final_hypervolume"] - 1),
                    "percent_vs_balanced": 100 * (
                        row["final_hypervolume"] / balanced["final_hypervolume"] - 1),
                })
    summaries = {}
    for index, method in enumerate(METHODS):
        selected = [row for row in paired if row["strategy"] == method]
        method_rows = [row for row in rows if row["strategy"] == method]
        summaries[method] = {
            "runs": len(selected),
            "vs_base": paired_summary(selected, "percent_vs_base", 918100+index),
            "vs_balanced": paired_summary(
                selected, "percent_vs_balanced", 918200+index),
            "mean_seconds": statistics.fmean(
                row["optimisation_seconds"] for row in method_rows),
            "logical_objective_evaluations": sum(
                row["logical_search_objective_evaluations"] for row in method_rows),
            "hidden_candidate_objective_evaluations": sum(
                row["hidden_candidate_objective_evaluations"] for row in method_rows),
            "physical_objective_evaluations": sum(
                row["physical_objective_evaluations"] for row in method_rows),
        }
    oracle = summaries[ORACLE]["vs_balanced"]
    simple = summaries[SIMPLE]["vs_balanced"]
    positive_instances = sum(row["mean"] > 0.0 for row in oracle["per_instance"])
    passed = bool(
        not scenario_failures
        and oracle["mean"] >= 1.0
        and positive_instances >= 2
        and oracle["mean"] > simple["mean"] + 1e-12)
    return {
        "completed": True,
        "definition": (
            "K8 complete-individual initialization final-HV screen; Base and "
            "directed-Balanced candidates are always in the pool; candidate "
            "construction performs zero CCP evaluations; oracle hidden "
            "evaluations are reported but do not extend the B400 search"),
        "design": {"instances": len(instances), "seeds": len(seeds),
                   "methods": list(METHODS), "scenario_count": 100,
                   "ccp_confidence": .9, "population": 20,
                   "logical_evaluation_budget": 400, "candidate_pool": 8},
        "scenario_exact": not scenario_failures,
        "scenario_failures": scenario_failures,
        "methods": summaries,
        "oracle_positive_instances": positive_instances,
        "oracle_final_hv_screen_pass": passed,
        "admission": {"minimum_oracle_mean_percent_vs_balanced": 1.0,
                      "minimum_positive_instances": 2,
                      "oracle_must_exceed_simple": True},
        "conclusion": ("EXPAND_I_K8_TRAINING_INSTANCES" if passed else
                       "STOP_I_K8_OR_REDESIGN_CANDIDATE_POOL"),
        "next_action": (
            "collect 25/50/100/200-instance I-K8 labels with frozen simple scorer"
            if passed else
            "do not train I Transformer; inspect oracle space and candidate diversity"),
        "paired_runs": paired,
    }


def execute(a):
    root = a.out.resolve(); root.mkdir(parents=True, exist_ok=True)
    instance_paths = [a.instances / f"{name}.json" for name in a.instance_ids]
    missing = [str(path) for path in instance_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing I development instances: {missing}")
    plan = {
        "protocol": "i-k8-complete-individual-final-oracle-v1",
        "instances": list(a.instance_ids), "algorithm_seeds": list(a.seeds),
        "methods": list(METHODS), "candidate_pool": 8,
        "scenario_count": 100, "ccp_confidence": .9,
        "population": 20, "logical_evaluation_budget": 400,
        "candidate_construction_ccp_evaluations": 0,
        "input_sha256": {str(path.resolve()): digest(path) for path in (
            a.data, a.normalization, Path(__file__).resolve(),
            SOURCE / "transformer_ppo/initialization_k8.py",
            SOURCE / "transformer_ppo/episode.py", *instance_paths)},
    }
    plan["plan_digest"] = canonical_digest(plan)
    plan_path = root / "PLAN.json"
    if plan_path.exists() and read_json(plan_path) != plan:
        raise ValueError("I-K8 frozen plan changed")
    write_json(plan_path, plan); plan_sha = digest(plan_path)
    if (root / "PIPELINE_COMPLETE").is_file():
        log("already complete; see QUICK_RESULT.json")
        return
    (root / "logs").mkdir(exist_ok=True)
    started = time.monotonic(); lock = threading.Lock(); completed = 0
    environment = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                   "OPENBLAS_NUM_THREADS": "1", "PYTHONUNBUFFERED": "1"}

    def task(instance_path, seed, method):
        nonlocal completed
        folder = root / instance_path.stem / f"seed_{seed}" / method
        reused = valid_run(folder, method, seed, plan_sha)
        if not reused:
            if folder.exists():
                quarantine = root / "interrupted_runs" / (
                    f"{instance_path.stem}_{seed}_{method}_{time.time_ns()}")
                quarantine.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(folder, quarantine)
            logfile = root / "logs" / (
                f"{instance_path.stem}_{seed}_{method}_{time.time_ns()}.log")
            command = [sys.executable, "-u", str(Path(__file__).resolve()),
                "--worker", "--data", str(a.data.resolve()),
                "--normalization", str(a.normalization.resolve()),
                "--instance", str(instance_path.resolve()),
                "--out", str(folder), "--seed", str(seed),
                "--method", method, "--plan-sha", plan_sha]
            with logfile.open("w") as handle:
                result = subprocess.run(command, cwd=SOURCE, env=environment,
                    stdout=handle, stderr=subprocess.STDOUT, timeout=600)
            if result.returncode or not valid_run(folder, method, seed, plan_sha):
                tail = "\n".join(logfile.read_text(
                    errors="replace").splitlines()[-30:])
                raise RuntimeError(f"I-K8 worker failed: {logfile}\n{tail}")
        row = read_json(folder / "RUN_COMPLETE.json")
        with lock:
            completed += 1
            write_json(root / "STATUS.json", {
                "stage": "I_K8_FINAL_ORACLE", "completed_runs": completed,
                "total_runs": len(instance_paths) * len(a.seeds) * len(METHODS),
                "elapsed_seconds": time.monotonic() - started})
            log(f"{completed}/{len(instance_paths)*len(a.seeds)*len(METHODS)} "
                f"{'reused' if reused else 'complete'} "
                f"{instance_path.stem}/{seed}/{method}")
        return row

    rows = []
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        futures = [pool.submit(task, path, seed, method)
                   for path in instance_paths for seed in a.seeds
                   for method in METHODS]
        for future in as_completed(futures):
            rows.append(future.result())
    report = summarize(rows, list(a.instance_ids), list(a.seeds))
    report.update({"elapsed_seconds_this_invocation": time.monotonic() - started,
                   "new_training_steps": 0, "new_ppo_updates": 0})
    write_json(root / "COMPLETE.json", {**report, "runs": rows})
    quick = {key: value for key, value in report.items() if key != "paired_runs"}
    write_json(root / "QUICK_RESULT.json", quick)
    with (root / "RESULT_TABLE.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["method", "mean_percent_vs_base",
                         "mean_percent_vs_balanced", "mean_seconds",
                         "logical_evaluations", "hidden_evaluations"])
        for method in METHODS:
            value = report["methods"][method]
            writer.writerow([method, value["vs_base"]["mean"],
                value["vs_balanced"]["mean"], value["mean_seconds"],
                value["logical_objective_evaluations"],
                value["hidden_candidate_objective_evaluations"]])
    (root / "PIPELINE_COMPLETE").write_text(report["conclusion"] + "\n")
    log(f"COMPLETE {report['conclusion']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path,
                        default=SOURCE / "data/data_expanded.xlsx")
    parser.add_argument("--instances", type=Path)
    parser.add_argument("--normalization", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--instance-ids", nargs="+", default=[
        "prescreen-search-S0-001", "prescreen-search-S0-002",
        "prescreen-search-S0-003"])
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--instance", type=Path)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--method", choices=METHODS)
    parser.add_argument("--plan-sha")
    args = parser.parse_args()
    if args.worker:
        if None in (args.instance, args.seed, args.method, args.plan_sha):
            parser.error("worker arguments are incomplete")
        worker(args); return
    if args.instances is None:
        parser.error("--instances is required")
    if args.workers < 1:
        parser.error("--workers must be positive")
    execute(args)


if __name__ == "__main__":
    main()
