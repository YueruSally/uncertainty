#!/usr/bin/env python3
"""Collect one temporally balanced K8 trajectory per generated S0 instance."""
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

from analyze_m_operator_oracle_audit import phase
from m_operator_offline import digest, read_json, write_json
from m_prescreen_k8 import CollectBalancedPolicy, CollectK8Controller
from run_learning_components_deterministic import _normalizer
from transformer_ppo.episode import run_episode
from transformer_ppo.instance_catalog import canonical_digest, read_instance


PHASES = ("early", "middle", "late")
MAX_STATES_PER_TRAJECTORY = 30
MIN_STATES_PER_TRAJECTORY = 12


def log(message):
    print(f"[M-K8-SCALE-DATA] {message}", flush=True)


class PhaseBalancedK8Controller(CollectK8Controller):
    def _choose_operator(self, *args, **kwargs):
        decision, candidates = super()._choose_operator(*args, **kwargs)
        row = self.feedback_pending
        if row is not None:
            if (len(self.proposals) != 8
                    or len(self.feedback_rows) >= MAX_STATES_PER_TRAJECTORY):
                self.feedback_pending = None
            else:
                row["sampling_phase"] = "pending_trajectory_tertile"
        return decision, candidates

    def summary(self):
        return {**super().summary(),
            "prescreen_scale_phase_assignment":
                "chronological tertiles assigned after trajectory completion"}


def worker(a):
    torch.set_num_threads(1)
    instance, _ = read_instance(a.instance)
    if instance.get("algorithm_seed") != a.seed:
        raise ValueError("algorithm seed differs from generated instance manifest")
    policy = CollectBalancedPolicy(instance["instance_id"], a.seed)
    holders = []

    def factory(*args, **kwargs):
        controller = PhaseBalancedK8Controller(*args, **kwargs)
        holders.append(controller)
        return controller

    result = run_episode(a.data, a.instance, a.out, policy,
        _normalizer(a.normalization), a.seed, evaluation_budget=400,
        pop_size=20, generations=1000, evaluation_mode="ccp", component="M",
        scenario_count=100, ccp_confidence=.9,
        uncertainty_profile="transport_border_saa_v1",
        inference_controller_factories={"M": factory})
    controller = holders[0]
    rows = controller.feedback_rows
    rows.sort(key=lambda row: (
        row["generation"], row["evaluation"], row["event"]))
    for index, row in enumerate(rows):
        row["sampling_phase"] = trajectory_phase(index, len(rows))
    counts = Counter(row["sampling_phase"] for row in rows)
    if (len(holders) != 1 or result["real_objective_evaluation_count"] != 400
            or not MIN_STATES_PER_TRAJECTORY <= len(rows) <= MAX_STATES_PER_TRAJECTORY
            or any(counts[name] < 4 for name in PHASES)
            or any(len(row["labels"]) != 8 for row in rows)
            or controller.feedback_evaluations != len(rows)*7
            or result.get("ppo_updates", 0)):
        raise RuntimeError(f"incomplete phase-balanced K8 trajectory: {counts}")
    write_json(a.out/"FEEDBACK.json", {"rows": rows,
        "phase_counts": dict(counts),
        "additional_objective_evaluations": controller.feedback_evaluations})
    result.update(protocol="m-prescreen-scale-data-v1",
        instance_id=instance["instance_id"],
        development_role=instance["development_role"],
        algorithm_seed=a.seed, plan_sha256=a.plan_sha,
        feedback_sha256=digest(a.out/"FEEDBACK.json"),
        feedback_state_count=len(rows), phase_counts=dict(counts),
        total_physical_objective_evaluations=(
            result["real_objective_evaluation_count"]+
            controller.feedback_evaluations))
    write_json(a.out/"RUN_COMPLETE.json", result)


def validate(folder, record, plan_sha):
    result = read_json(folder/"RUN_COMPLETE.json")
    feedback = read_json(folder/"FEEDBACK.json")
    config = read_json(folder/"configuration.json")
    counts = Counter(row["sampling_phase"] for row in feedback["rows"])
    ordered = sorted(feedback["rows"], key=lambda row: (
        row["generation"], row["evaluation"], row["event"]))
    phase_assignment_exact = all(
        row["sampling_phase"] == trajectory_phase(index, len(ordered))
        for index, row in enumerate(ordered))
    if (result.get("protocol") != "m-prescreen-scale-data-v1"
            or result.get("instance_id") != record["instance_id"]
            or result.get("development_role") != record["development_role"]
            or result.get("algorithm_seed") != record["algorithm_seed"]
            or result.get("plan_sha256") != plan_sha
            or result.get("feedback_sha256") != digest(folder/"FEEDBACK.json")
            or result.get("real_objective_evaluation_count") != 400
            or not MIN_STATES_PER_TRAJECTORY <= len(feedback["rows"]) <= MAX_STATES_PER_TRAJECTORY
            or result.get("feedback_state_count") != len(feedback["rows"])
            or feedback.get("additional_objective_evaluations") != len(feedback["rows"])*7
            or any(counts[name] < 4 for name in PHASES)
            or not phase_assignment_exact
            or any(len(row["labels"]) != 8 for row in feedback["rows"])
            or config.get("training_size") != 100
            or config.get("population") != 20
            or config.get("evaluation_budget") != 400
            or config.get("alpha") != .9
            or config.get("uncertainty", {}).get("name") !=
                "transport_border_saa_v1"):
        raise ValueError(f"invalid completed scale trajectory: {folder}")
    return result


def trajectory_phase(index, count):
    """Assign available post-G10 decisions to chronological tertiles."""
    if not 0 <= index < count:
        raise ValueError("trajectory phase index is out of range")
    return phase(index/count)


def build_plan(a):
    catalog = read_json(a.generated_instances/"catalog.json")
    identity = dict(catalog); claimed = identity.pop("catalog_digest", None)
    if (canonical_digest(identity) != claimed
            or catalog.get("train_count") != 200
            or catalog.get("validation_count") != 20
            or catalog.get("final_test_instances_used") is not False):
        raise ValueError("generated scale catalog is invalid")
    records = catalog["instances"]
    if len(records) != 220:
        raise ValueError("scale catalog must contain 220 instances")
    inputs = [Path(__file__).resolve(), Path(__file__).parent/"m_prescreen_k8.py",
        Path(__file__).parent/"generate_m_prescreen_scale_instances.py",
        a.data.resolve(), a.normalization.resolve(),
        a.generated_instances/"catalog.json",
        Path(__file__).parent/"data/border_crossing_events.csv"]
    for record in records:
        path = a.generated_instances/f"{record['instance_id']}.json"
        instance, _ = read_instance(path)
        if (instance.get("development_role") != record["development_role"]
                or instance.get("algorithm_seed") != record["algorithm_seed"]):
            raise ValueError("generated manifest differs from catalog")
        inputs.append(path)
    return {"protocol": "m-prescreen-scale-data-v1",
        "catalog_digest": claimed, "instances": records,
        "instance_count": 220, "train_count": 200,
        "validation_count": 20,
        "states_per_trajectory": {"minimum": MIN_STATES_PER_TRAJECTORY,
            "maximum": MAX_STATES_PER_TRAJECTORY,
            "phase_assignment": "chronological within-trajectory tertiles after collection"},
        "candidate_count": 8,
        "search_budget": 400, "scenario_count": 100,
        "population": 20, "ccp_confidence": .9,
        "expected_search_objective_evaluations": 88_000,
        "maximum_label_objective_evaluations": 46_200,
        "maximum_total_objective_evaluations": 134_200,
        "workers": a.workers, "threads_per_worker": 1,
        "invocation_limit_seconds": 7200, "run_limit_seconds": 300,
        "automatic_retries": 0,
        "data": str(a.data.resolve()),
        "normalization": str(a.normalization.resolve()),
        "generated_instances": str(a.generated_instances.resolve()),
        "input_sha256": {str(path.resolve()): digest(path) for path in inputs}}


def execute(a):
    root = a.out.resolve(); plan = build_plan(a)
    if (root/"PLAN.json").exists() and read_json(root/"PLAN.json") != plan:
        raise ValueError("frozen scale collection inputs changed")
    write_json(root/"PLAN.json", plan)
    if (root/"COLLECTION_COMPLETE").exists():
        log("already complete; see QUICK_RESULT.json")
        return
    plan_sha = digest(root/"PLAN.json")
    (root/"logs").mkdir(exist_ok=True)
    started = time.monotonic(); deadline = started+7200
    status = {"stage": "collection", "completed_runs": 0,
              "total_runs": 220}
    lock, stop = threading.Lock(), threading.Event()
    env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
           "OPENBLAS_NUM_THREADS": "1", "PYTHONUNBUFFERED": "1"}
    log("220 trajectories; 12-30 exact-K8 states each; chronological tertiles; CCP100/pop20/B400")

    def task(record):
        name, seed = record["instance_id"], record["algorithm_seed"]
        folder = root/"runs"/name
        reused = (folder/"RUN_COMPLETE.json").exists()
        if not reused:
            if folder.exists():
                quarantine = root/"interrupted_runs"/f"{name}_{time.time_ns()}"
                quarantine.parent.mkdir(parents=True, exist_ok=True)
                folder.rename(quarantine)
            logfile = root/"logs"/f"{name}_{time.time_ns()}.log"
            command = [sys.executable, "-u", str(Path(__file__).resolve()),
                "--worker", "--data", str(a.data.resolve()),
                "--generated-instances", str(a.generated_instances.resolve()),
                "--normalization", str(a.normalization.resolve()),
                "--instance", str((a.generated_instances/f"{name}.json").resolve()),
                "--out", str(folder), "--seed", str(seed),
                "--plan-sha", plan_sha]
            with logfile.open("w") as handle, subprocess.Popen(command,
                    cwd=Path(__file__).parent, env=env, stdout=handle,
                    stderr=subprocess.STDOUT) as process:
                begin = heartbeat = time.monotonic()
                while process.poll() is None:
                    now = time.monotonic()
                    if stop.is_set() or now >= deadline or now-begin >= 300:
                        process.kill(); process.wait()
                        raise TimeoutError(f"trajectory stopped; see {logfile}")
                    if now-heartbeat >= 30:
                        log(f"running {name} pid={process.pid} elapsed={now-begin:.0f}s")
                        heartbeat = now
                    time.sleep(.2)
                if process.returncode:
                    tail = "\n".join(logfile.read_text(
                        errors="replace").splitlines()[-20:])
                    raise RuntimeError(f"worker failed: {logfile}\n{tail}")
        result = validate(folder, record, plan_sha)
        with lock:
            status["completed_runs"] += 1
            status["elapsed_seconds"] = time.monotonic()-started
            write_json(root/"STATUS.json", status)
            log(f"{status['completed_runs']}/220 {'reused' if reused else 'complete'} {name}")
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
    result = {"completed": True, "instance_count": len(runs),
        "role_counts": dict(role_counts),
        "state_count": sum(row["feedback_state_count"] for row in runs),
        "phase_counts": {name: sum(row["phase_counts"][name] for row in runs)
                         for name in PHASES},
        "search_objective_evaluations": sum(
            row["real_objective_evaluation_count"] for row in runs),
        "label_objective_evaluations": sum(
            row["training_feedback_evaluations"] for row in runs),
        "total_objective_evaluations": sum(
            row["total_physical_objective_evaluations"] for row in runs),
        "new_training_steps": 0, "new_ppo_updates": 0,
        "final_test_instances_used": False,
        "elapsed_seconds_this_invocation": time.monotonic()-started,
        "next_action": "train frozen 25/50/100/200-instance nested learning curve"}
    if (result["role_counts"] != {"train": 200, "validation": 20}
            or not 220*MIN_STATES_PER_TRAJECTORY <= result["state_count"] <= 220*MAX_STATES_PER_TRAJECTORY
            or any(result["phase_counts"][name] < 220*4 for name in PHASES)
            or result["label_objective_evaluations"] != result["state_count"]*7
            or result["total_objective_evaluations"] != 88_000+result["label_objective_evaluations"]):
        raise RuntimeError("completed scale collection totals mismatch")
    write_json(root/"COMPLETE.json", {**result, "runs": runs})
    write_json(root/"QUICK_RESULT.json", result)
    write_json(root/"STATUS.json", {**status, "stage": "complete"})
    (root/"COLLECTION_COMPLETE").write_text(digest(root/"QUICK_RESULT.json")+"\n")
    log(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path,
        default=Path(__file__).parent/"data/data_expanded.xlsx")
    parser.add_argument("--generated-instances", type=Path, required=True)
    parser.add_argument("--normalization", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", choices=range(1, 9), default=4, type=int)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--instance", type=Path)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--plan-sha")
    a = parser.parse_args(); a.out = a.out.resolve()
    if a.worker:
        if any(getattr(a, name) is None for name in
               ("instance", "seed", "plan_sha")):
            parser.error("worker arguments incomplete")
        a.out.mkdir(parents=True, exist_ok=False)
        worker(a); return
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
