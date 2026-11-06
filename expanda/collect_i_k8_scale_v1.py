#!/usr/bin/env python3
"""Collect exact K8 initialization labels on 200 train/20 validation instances."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

import torch

from m_operator_offline import digest, read_json, write_json
from run_learning_components_deterministic import _normalizer
from transformer_ppo.episode import run_episode
from transformer_ppo.initialization_k8 import InitializationK8Controller
from transformer_ppo.instance_catalog import canonical_digest, read_instance


def log(message):
    print(f"[I-K8-SCALE-DATA] {message}", flush=True)


def worker(a):
    torch.set_num_threads(1)
    instance, _ = read_instance(a.instance)
    if instance.get("algorithm_seed") != a.seed:
        raise ValueError("algorithm seed differs from generated instance manifest")
    holders = []

    def factory(*args, **kwargs):
        controller = InitializationK8Controller(
            *args, selection_mode="collect_simple", pool_size=8, **kwargs)
        holders.append(controller)
        return controller

    result = run_episode(
        a.data, a.instance, a.out, None, _normalizer(a.normalization),
        algorithm_seed=a.seed, evaluation_budget=400, pop_size=20,
        generations=1000, evaluation_mode="ccp", component="BASE",
        scenario_count=100, ccp_confidence=.9,
        uncertainty_profile="transport_border_saa_v1",
        inference_controller_factories={"BASE": factory},
        method_override="I_K8_COLLECT_SIMPLE")
    if len(holders) != 1:
        raise RuntimeError("I-K8 collection controller was not installed once")
    controller = holders[0]
    rows = controller.feedback_rows
    if (len(rows) != 20
            or result.get("real_objective_evaluation_count") != 400
            or result.get("initialization_k8_pool_events") != 20
            or result.get("initialization_k8_candidate_objective_evaluations") != 160
            or result.get("initialization_k8_hidden_candidate_objective_evaluations") != 140
            or any(len(row.get("labels", ())) != 8 for row in rows)
            or any(row.get("additional_objective_evaluations") != 7
                   for row in rows)):
        raise RuntimeError("incomplete exact I-K8 collection trajectory")
    feedback = {"rows": rows, "state_count": len(rows),
                "additional_objective_evaluations": 140}
    write_json(a.out / "FEEDBACK.json", feedback)
    result.update({
        "protocol": "i-k8-scale-data-v1",
        "instance_id": instance["instance_id"],
        "development_role": instance["development_role"],
        "algorithm_seed": a.seed,
        "plan_sha256": a.plan_sha,
        "feedback_sha256": digest(a.out / "FEEDBACK.json"),
        "feedback_state_count": len(rows),
        "total_physical_objective_evaluations": 540,
        "new_training_steps": 0,
        "new_ppo_updates": 0,
    })
    write_json(a.out / "RUN_COMPLETE.json", result)


def validate(folder, record, plan_sha):
    result = read_json(folder / "RUN_COMPLETE.json")
    feedback_path = folder / "FEEDBACK.json"
    feedback = read_json(feedback_path)
    config = read_json(folder / "configuration.json")
    rows = feedback.get("rows", ())
    if (result.get("protocol") != "i-k8-scale-data-v1"
            or result.get("instance_id") != record["instance_id"]
            or result.get("development_role") != record["development_role"]
            or result.get("algorithm_seed") != record["algorithm_seed"]
            or result.get("plan_sha256") != plan_sha
            or result.get("feedback_sha256") != digest(feedback_path)
            or result.get("real_objective_evaluation_count") != 400
            or result.get("initialization_k8_pool_events") != 20
            or result.get("initialization_k8_candidate_objective_evaluations") != 160
            or result.get("initialization_k8_hidden_candidate_objective_evaluations") != 140
            or result.get("feedback_state_count") != 20
            or len(rows) != 20
            or feedback.get("additional_objective_evaluations") != 140
            or any(len(row.get("labels", ())) != 8 for row in rows)
            or any(row.get("candidate_roles", [None])[0] != "base"
                   for row in rows)
            or config.get("training_size") != 100
            or config.get("population") != 20
            or config.get("evaluation_budget") != 400
            or config.get("alpha") != .9
            or config.get("uncertainty", {}).get("name") !=
                "transport_border_saa_v1"):
        raise ValueError(f"invalid I-K8 collection trajectory: {folder}")
    return result


def build_plan(a):
    catalog = read_json(a.generated_instances / "catalog.json")
    identity = dict(catalog)
    claimed = identity.pop("catalog_digest", None)
    if (canonical_digest(identity) != claimed
            or catalog.get("train_count") != 200
            or catalog.get("validation_count") != 20
            or catalog.get("final_test_instances_used") is not False):
        raise ValueError("generated scale catalog is invalid")
    records = catalog["instances"]
    if len(records) != 220:
        raise ValueError("I-K8 scale collection requires 220 instances")
    inputs = [Path(__file__).resolve(),
              Path(__file__).parent / "transformer_ppo/initialization_k8.py",
              a.data.resolve(), a.normalization.resolve(),
              a.generated_instances / "catalog.json",
              Path(__file__).parent / "data/border_crossing_events.csv"]
    for record in records:
        path = a.generated_instances / f"{record['instance_id']}.json"
        instance, _ = read_instance(path)
        if (instance.get("development_role") != record["development_role"]
                or instance.get("algorithm_seed") != record["algorithm_seed"]):
            raise ValueError("generated manifest differs from catalog")
        inputs.append(path)
    return {
        "protocol": "i-k8-scale-data-v1",
        "catalog_digest": claimed,
        "instances": records,
        "instance_count": 220,
        "train_count": 200,
        "validation_count": 20,
        "states_per_trajectory": 20,
        "candidate_count": 8,
        "selection_during_collection": "frozen deterministic simple scorer",
        "label_definition": "exact CCP100 HV contribution relative to the pre-slot population",
        "search_budget": 400,
        "scenario_count": 100,
        "population": 20,
        "ccp_confidence": .9,
        "expected_search_objective_evaluations": 88_000,
        "expected_label_objective_evaluations": 30_800,
        "expected_total_objective_evaluations": 118_800,
        "workers": a.workers,
        "threads_per_worker": 1,
        "invocation_limit_seconds": 7200,
        "run_limit_seconds": 300,
        "automatic_retries": 0,
        "final_test_instances_used": False,
        "data": str(a.data.resolve()),
        "normalization": str(a.normalization.resolve()),
        "generated_instances": str(a.generated_instances.resolve()),
        "input_sha256": {str(path.resolve()): digest(path) for path in inputs},
    }


def execute(a):
    root = a.out.resolve()
    plan = build_plan(a)
    if (root / "PLAN.json").exists() and read_json(root / "PLAN.json") != plan:
        raise ValueError("frozen I-K8 collection inputs changed")
    write_json(root / "PLAN.json", plan)
    if (root / "COLLECTION_COMPLETE").exists():
        log("already complete; see QUICK_RESULT.json")
        return
    plan_sha = digest(root / "PLAN.json")
    (root / "logs").mkdir(exist_ok=True)
    started = time.monotonic()
    deadline = started + 7200
    status = {"stage": "collection", "completed_runs": 0, "total_runs": 220}
    lock = threading.Lock()
    stop = threading.Event()
    environment = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                   "OPENBLAS_NUM_THREADS": "1", "PYTHONUNBUFFERED": "1"}
    log("220 trajectories x 20 exact-K8 initialization states; CCP100/pop20/B400")

    def task(record):
        name = record["instance_id"]
        seed = record["algorithm_seed"]
        folder = root / "runs" / name
        reused = (folder / "RUN_COMPLETE.json").exists()
        if not reused:
            if folder.exists():
                quarantine = root / "interrupted_runs" / f"{name}_{time.time_ns()}"
                quarantine.parent.mkdir(parents=True, exist_ok=True)
                folder.rename(quarantine)
            logfile = root / "logs" / f"{name}_{time.time_ns()}.log"
            command = [sys.executable, "-u", str(Path(__file__).resolve()),
                "--worker", "--data", str(a.data.resolve()),
                "--generated-instances", str(a.generated_instances.resolve()),
                "--normalization", str(a.normalization.resolve()),
                "--instance", str((a.generated_instances / f"{name}.json").resolve()),
                "--out", str(folder), "--seed", str(seed), "--plan-sha", plan_sha]
            with logfile.open("w") as handle, subprocess.Popen(
                    command, cwd=Path(__file__).parent, env=environment,
                    stdout=handle, stderr=subprocess.STDOUT) as process:
                begin = time.monotonic()
                while process.poll() is None:
                    now = time.monotonic()
                    if stop.is_set() or now >= deadline or now - begin >= 300:
                        process.kill()
                        process.wait()
                        raise TimeoutError(f"trajectory stopped; see {logfile}")
                    time.sleep(.2)
                if process.returncode:
                    tail = "\n".join(logfile.read_text(
                        errors="replace").splitlines()[-25:])
                    raise RuntimeError(f"worker failed: {logfile}\n{tail}")
        result = validate(folder, record, plan_sha)
        with lock:
            status["completed_runs"] += 1
            status["elapsed_seconds"] = time.monotonic() - started
            write_json(root / "STATUS.json", status)
            log(f"{status['completed_runs']}/220 "
                f"{'reused' if reused else 'complete'} {name}")
        return result

    runs = []
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        futures = [pool.submit(task, record) for record in plan["instances"]]
        try:
            for future in as_completed(futures):
                runs.append(future.result())
        except BaseException:
            stop.set()
            for future in futures:
                future.cancel()
            raise
    role_counts = Counter(row["development_role"] for row in runs)
    result = {
        "completed": True,
        "instance_count": len(runs),
        "role_counts": dict(role_counts),
        "state_count": sum(row["feedback_state_count"] for row in runs),
        "search_objective_evaluations": sum(
            row["real_objective_evaluation_count"] for row in runs),
        "label_objective_evaluations": sum(
            row["initialization_k8_hidden_candidate_objective_evaluations"]
            for row in runs),
        "total_objective_evaluations": sum(
            row["total_physical_objective_evaluations"] for row in runs),
        "new_training_steps": 0,
        "new_ppo_updates": 0,
        "final_test_instances_used": False,
        "elapsed_seconds_this_invocation": time.monotonic() - started,
        "next_action": "train frozen 25/50/100/200-instance I-K8 Transformer",
    }
    if (result["role_counts"] != {"train": 200, "validation": 20}
            or result["state_count"] != 4400
            or result["search_objective_evaluations"] != 88_000
            or result["label_objective_evaluations"] != 30_800
            or result["total_objective_evaluations"] != 118_800):
        raise RuntimeError("completed I-K8 collection totals mismatch")
    write_json(root / "COMPLETE.json", {**result, "runs": runs})
    write_json(root / "QUICK_RESULT.json", result)
    write_json(root / "STATUS.json", {**status, "stage": "complete"})
    (root / "COLLECTION_COMPLETE").write_text(
        digest(root / "QUICK_RESULT.json") + "\n")
    log(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path,
        default=Path(__file__).parent / "data/data_expanded.xlsx")
    parser.add_argument("--generated-instances", type=Path, required=True)
    parser.add_argument("--normalization", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", choices=range(1, 9), default=4, type=int)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--instance", type=Path)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--plan-sha")
    a = parser.parse_args()
    a.out = a.out.resolve()
    if a.worker:
        if any(getattr(a, name) is None for name in
               ("instance", "seed", "plan_sha")):
            parser.error("worker arguments incomplete")
        a.out.mkdir(parents=True, exist_ok=False)
        worker(a)
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
