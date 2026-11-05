#!/usr/bin/env python3
"""Two fixed methods, two development instances, twenty paired seeds; no training."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

import numpy as np
import torch

from m_operator_fixed_mix import FixedMixPolicy, FixedMixController, INSTANCES, METHODS, SEEDS, summarize, validate_pair
from m_operator_offline import digest, read_json, write_json
from run_learning_components_deterministic import _normalizer
from run_m_operator_closed_loop_v1 import load_run
from transformer_ppo.episode import run_episode
from transformer_ppo.instance_catalog import read_instance


def log(message):
    print(f"[M-MIX] {message}", flush=True)


def run_one(data, instance, normalization, out, seed, method, plan_sha):
    torch.set_num_threads(1)
    identity, _ = read_instance(instance)
    probability = 0. if method == METHODS[0] else .2
    policy = FixedMixPolicy(identity["instance_id"], seed, probability)
    started = time.perf_counter()
    row = run_episode(data, instance, out, policy, _normalizer(normalization), seed,
        evaluation_budget=400, pop_size=20, generations=1000, evaluation_mode="ccp",
        scenario_count=100, ccp_confidence=.9, uncertainty_profile="transport_border_saa_v1", component="M",
        inference_controller_factories={"M": FixedMixController})
    if row["real_objective_evaluation_count"] != 400 or row["ppo_updates"] or row["counterfactual_evaluation_count"]:
        raise RuntimeError("fixed mixture requires exactly B400 and zero training/counterfactual evaluations")
    row.update(method=method, strategy=method, algorithm_seed=seed, plan_sha256=plan_sha,
        worker_seconds=time.perf_counter() - started, new_training_steps=0, learning_enabled=False)
    config = read_json(out / "configuration.json")
    config.update(method=method, learning_enabled=False, fixed_mixture={
        "mode_probability": probability, "minimum_generation": 10, "mode_illegal": "fallback Balanced",
        "selection_seed": policy.selection_seed, "selection_rng": "private; independent of target and NSGA-II RNG"})
    write_json(out / "configuration.json", config)
    write_json(out / "COMPLETE.json", row)
    write_json(out / "RUN_COMPLETE.json", row)
    return row


def frozen_plan(a):
    source = Path(__file__).resolve().parent
    inputs = [a.data.resolve(), a.normalization.resolve(), source / "data/border_crossing_events.csv"]
    for name in INSTANCES:
        path = (a.instances / (name + ".json")).resolve()
        record, _ = read_instance(path)
        if record["instance_id"] != name or record["source_data_sha256"] != digest(a.data):
            raise ValueError(f"instance/workbook identity mismatch: {name}")
        inputs.append(path)
    _normalizer(a.normalization)
    inputs += sorted(source.glob("*.py")) + sorted((source / "transformer_ppo").glob("*.py"))
    return {"protocol": "fixed-balanced80-mode20-v1", "instances": list(INSTANCES), "algorithm_seeds": list(SEEDS),
        "methods": list(METHODS), "run_count": 80, "paired_run_count": 40,
        "scenario_count": 100, "ccp_confidence": .9, "population_size": 20, "evaluation_budget_per_run": 400,
        "generation_cap": 1000, "minimum_control_generation": 10, "mode_probability": .2,
        "mix_definition": "independent Bernoulli draw at each post-G10 mutation decision; actual fraction reported; mode illegal -> Balanced; Balanced unavailable -> same no-op in both arms",
        "execution_contract": "operator_shared_v1", "no_checkpoint": True, "new_training_steps": 0,
        "new_ppo_updates": 0, "counterfactual_evaluations": 0,
        "workers": a.workers, "threads_per_worker": 1, "per_run_timeout_seconds": 300,
        "invocation_time_limit_seconds": 1800, "automatic_retry_count": 0,
        "torch": str(torch.__version__), "numpy": str(np.__version__),
        "source_sha256": {str(p): digest(p) for p in inputs},
        "primary_metric": "mean of 100*(mixture_HV/paired_heuristic_HV-1), separately per instance",
        "uncertainty": "10000 paired-seed percentile bootstrap samples, fixed seed 896900; interval 2.5%..97.5%; reused indices for instance contrast",
        "decision_rule": "positive if interval lower >0, negative if upper <0, otherwise inconclusive; no automatic model/PPO admission",
        "interpretation": "one fixed mixture on two already-inspected development instances; failure does not distinguish state dependence from noise or a poor mixture ratio",
        "timing_scope": "same worker/thread limits and alternating method order; concurrent exploratory timing, not isolated formal benchmarking"}


def execute(a):
    root, source = a.out.resolve(), Path(__file__).resolve().parent
    plan = frozen_plan(a)
    if (root / "PLAN.json").exists() and read_json(root / "PLAN.json") != plan:
        raise ValueError("inputs/code/design changed; retain this output and use a new root")
    write_json(root / "PLAN.json", plan)
    plan_sha = digest(root / "PLAN.json")
    if (root / "PIPELINE_COMPLETE").exists():
        log("already complete: read QUICK_RESULT.json and RESULT_TABLE.csv")
        return
    started = time.monotonic()
    deadline = started + 1800
    stop, status_lock = threading.Event(), threading.Lock()
    status = {"stage": "search", "completed_runs": 0, "total_runs": 80, "workers": a.workers}
    env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "PYTHONUNBUFFERED": "1"}
    (root / "logs").mkdir(exist_ok=True)
    log(f"80 searches / 40 pairs; CCP100/pop20/B400; workers={a.workers}; no model/training")

    def one_pair(instance, seed):
        pair = []
        for method in (METHODS if (seed - SEEDS[0]) % 2 == 0 else METHODS[::-1]):
            if stop.is_set():
                raise RuntimeError("stopped after another run failed")
            folder = root / instance / f"seed_{seed}" / method
            reused = (folder / "RUN_COMPLETE.json").exists()
            if not reused:
                if folder.exists() and any(folder.iterdir()):
                    destination = root / "interrupted_runs" / instance / f"seed_{seed}" / f"{method}_{time.time_ns()}"
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    folder.rename(destination)
                    log(f"preserved incomplete run: {destination}")
                filename = root / "logs" / f"{instance}_{seed}_{method}_{time.time_ns()}.log"
                cmd = [sys.executable, "-u", str(source / "run_m_fixed_mix_v1.py"), "--worker",
                    "--data", str(a.data.resolve()), "--instance", str((a.instances / (instance + ".json")).resolve()),
                    "--normalization", str(a.normalization.resolve()), "--out", str(folder),
                    "--seed", str(seed), "--method", method, "--plan-sha", plan_sha]
                log(f"start {instance} seed={seed} {method}")
                with filename.open("w") as handle:
                    with subprocess.Popen(cmd, cwd=source, env=env, stdout=handle, stderr=subprocess.STDOUT) as process:
                        run_started = last_log = time.monotonic()
                        try:
                            while process.poll() is None:
                                now = time.monotonic()
                                if stop.is_set() or now >= deadline or now - run_started >= 300:
                                    raise TimeoutError(f"stopped/time limit; log={filename}")
                                if now - last_log >= 20:
                                    log(f"running {instance}/{seed}/{method} pid={process.pid} elapsed={now-run_started:.0f}s")
                                    last_log = now
                                time.sleep(.2)
                        finally:
                            if process.poll() is None:
                                process.kill()
                                process.wait()
                        if process.returncode:
                            tail = "\n".join(filename.read_text(errors="replace").splitlines()[-12:])
                            raise RuntimeError(f"worker exit {process.returncode}; log={filename}\n{tail}")
            row = load_run(folder, instance, plan_sha)
            if row["strategy"] != method or row["algorithm_seed"] != seed:
                raise ValueError("saved run identity mismatch")
            pair.append(row)
            with status_lock:
                status["completed_runs"] += 1
                status["elapsed_seconds"] = time.monotonic() - started
                write_json(root / "STATUS.json", status)
                log(f"{status['completed_runs']}/80 {'reused' if reused else 'complete'} {instance}/{seed}/{method} HV={row['final_hypervolume']:.7g} time={row['optimisation_seconds']:.2f}s mode={row['fixed_mix']['mode_selected']}/{row['fixed_mix']['supported_decisions']}")
        paired = validate_pair(pair)
        log(f"pair verified {instance}/{seed} fixed-mix vs heuristic={paired['paired_percent_vs_heuristic']:+.3f}%")
        return pair

    runs = []
    # Seed-major ordering covers both instances throughout the fixed schedule.
    with ThreadPoolExecutor(max_workers=a.workers) as executor:
        futures = [executor.submit(one_pair, instance, seed) for seed in SEEDS for instance in INSTANCES]
        try:
            for future in as_completed(futures):
                runs.extend(future.result())
        except BaseException:
            stop.set()
            for future in futures:
                future.cancel()
            raise
    result = summarize(runs)
    pairs = result.pop("pairs")
    report = {"completed": True, "design": plan, "pairing_exact": True,
              "elapsed_seconds_this_invocation": time.monotonic() - started, **result}
    write_json(root / "COMPLETE.json", {**report, "runs": sorted(runs, key=lambda r: (r["instance"], r["algorithm_seed"], r["strategy"])), "pairs": pairs})
    write_json(root / "QUICK_RESULT.json", report)
    with (root / "RESULT_TABLE.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(report["per_instance"][0]))
        writer.writeheader()
        writer.writerows(report["per_instance"])
    write_json(root / "STATUS.json", {**status, "stage": "complete", "conclusion": report["conclusion"]})
    (root / "PIPELINE_COMPLETE").write_text("80 completed searches; see conclusion in QUICK_RESULT.json\n")
    for row in report["per_instance"]:
        log(f"RESULT {row['instance']}: paired={row['mean_paired_percent_vs_heuristic']:+.3f}% CI=[{row['paired_seed_bootstrap_95ci_low']:+.3f},{row['paired_seed_bootstrap_95ci_high']:+.3f}] H={row['mean_heuristic_seconds']:.2f}s MIX={row['mean_mixture_seconds']:.2f}s")
    log(f"COMPLETE {report['conclusion']}: {report['explanation']}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--worker", action="store_true")
    p.add_argument("--instances", type=Path)
    p.add_argument("--instance", type=Path)
    p.add_argument("--normalization", type=Path, required=True)
    p.add_argument("--data", type=Path, default=Path(__file__).resolve().parent / "data/data_expanded.xlsx")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--workers", type=int, choices=(1, 2), default=2)
    p.add_argument("--seed", type=int)
    p.add_argument("--method", choices=METHODS)
    p.add_argument("--plan-sha")
    a = p.parse_args()
    if a.worker:
        if a.instance is None or a.seed is None or a.method is None or not a.plan_sha:
            p.error("worker requires instance, seed, method, plan-sha")
        run_one(a.data, a.instance, a.normalization, a.out, a.seed, a.method, a.plan_sha)
        return
    if a.instances is None:
        p.error("coordinator requires --instances")
    a.out.mkdir(parents=True, exist_ok=True)
    with (a.out / ".lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            execute(a)
        except Exception as exc:
            status = read_json(a.out / "STATUS.json") if (a.out / "STATUS.json").exists() else {}
            write_json(a.out / "STATUS.json", {**status, "stage": "failed", "error": str(exc)})
            log(f"FAILED {exc}")
            raise


if __name__ == "__main__":
    main()
