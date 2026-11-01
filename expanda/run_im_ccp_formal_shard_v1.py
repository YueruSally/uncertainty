#!/usr/bin/env python3
"""Execute one resumable shard of the frozen formal CCP Base/I/M test."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from im_ccp_formal_plan import (
    CCP_CONFIDENCE,
    EVALUATION_BUDGET,
    GENERATIONS,
    I_PROFILE,
    METHODS,
    M_MINIMUM_GENERATION,
    M_OPERATOR,
    M_PROBABILITY,
    M_SELECTOR,
    POPULATION,
    SHARD_COUNT,
    TRAINING_SCENARIO_COUNT,
    UNCERTAINTY_PROFILE,
    canonical_digest,
    expected_rule_i_attempts,
)


ROOT = Path(__file__).resolve().parent
RUNNER = ROOT / "run_learning_components_deterministic.py"
EPSILON = 1e-12


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n",
        encoding="utf-8")
    temporary.replace(path)


def _verify_plan(path):
    plan = _read(path)
    payload = dict(plan)
    digest = payload.pop("plan_digest", None)
    if not digest or canonical_digest(payload) != digest:
        raise ValueError("formal plan digest mismatch")
    schedule = plan.get("schedule", [])
    if (plan.get("formal") is not True
            or tuple(plan.get("methods", ())) != METHODS
            or plan.get("population") != POPULATION
            or plan.get("generations") != GENERATIONS
            or plan.get("evaluation_budget") != EVALUATION_BUDGET
            or plan.get("training_scenario_count")
            != TRAINING_SCENARIO_COUNT
            or len(schedule) != 90
            or canonical_digest(schedule) != plan.get("schedule_digest")
            or plan.get("execution", {}).get("shard_count") != SHARD_COUNT):
        raise ValueError("plan is not the frozen 90-run formal CCP design")
    for relative, expected in plan["implementation_sha256"].items():
        target = ROOT / relative
        if not target.is_file() or _sha256(target) != expected:
            raise ValueError(f"formal implementation changed: {target}")
    artifacts = (
        (Path(plan["network"]["path"]), plan["network"]["sha256"]),
        (Path(plan["border_event_data"]["path"]),
         plan["border_event_data"]["sha256"]),
        (Path(plan["normalization"]["path"]),
         plan["normalization"]["sha256"]),
        (Path(plan["m_infrastructure_checkpoint"]["path"]),
         plan["m_infrastructure_checkpoint"]["sha256"]),
        (Path(plan["final_test_instances"]["catalog_path"]),
         plan["final_test_instances"]["catalog_sha256"]),
    )
    for target, expected in artifacts:
        if not target.is_file() or _sha256(target) != expected:
            raise ValueError(f"frozen artifact is missing or changed: {target}")
    return plan


def _component(method):
    return {
        "BASE": "BASE",
        "I_DIRECTED_BALANCED": "RULE-I",
        "M_DIRECTED_BALANCED": "M",
    }[method]


def _validate_result(run, task, plan):
    complete_path = run / "COMPLETE.json"
    config_path = run / "configuration.json"
    front_path = run / "final_feasible_nondominated.json"
    if not complete_path.is_file():
        return None
    if not config_path.is_file() or not front_path.is_file():
        raise ValueError(f"incomplete result files: {task['task_id']}")
    complete = _read(complete_path)
    config = _read(config_path)
    front = _read(front_path)
    component = _component(task["method"])
    instance = next(
        row for row in plan["final_test_instances"]["instances"]
        if row["instance_id"] == task["instance_id"])
    hv = complete.get("final_hypervolume")
    if (complete.get("completed") is not True
            or complete.get("component") != component
            or complete.get("real_objective_evaluation_count")
            != EVALUATION_BUDGET
            or complete.get("evaluation_budget") != EVALUATION_BUDGET
            or complete.get("generations_completed") != GENERATIONS
            or complete.get("stop_reason") != "evaluation_budget"
            or not isinstance(hv, (int, float)) or not math.isfinite(hv)
            or complete.get("final_front_size") != len(front)
            or not front
            or complete.get("ppo_updates", 0) != 0
            or config.get("component") != component
            or config.get("evaluation_mode") != "ccp"
            or config.get("instance_id") != task["instance_id"]
            or config.get("instance_digest") != instance["instance_digest"]
            or config.get("population") != POPULATION
            or config.get("generations") != GENERATIONS
            or config.get("evaluation_budget") != EVALUATION_BUDGET
            or config.get("training_size") != TRAINING_SCENARIO_COUNT
            or abs(float(config.get("alpha", -1.0))
                   - CCP_CONFIDENCE) > EPSILON
            or config.get("uncertainty", {}).get("name")
            != UNCERTAINTY_PROFILE
            or config.get("seeds", {}).get("algorithm")
            != task["algorithm_seed"]
            or config.get("seeds", {}).get("training")
            != instance["ccp100_seed"]
            or config.get("policy_training") is not False):
        raise ValueError(f"invalid frozen result: {task['task_id']}")
    if task["method"] == "BASE":
        if config.get("policy_checkpoint") is not None:
            raise ValueError(f"Base unexpectedly loaded a policy: {task['task_id']}")
    elif task["method"] == "I_DIRECTED_BALANCED":
        if (config.get("initialization_rule_profile") != I_PROFILE
                or config.get("policy_checkpoint") is not None
                or complete.get("initialization_attempts")
                != expected_rule_i_attempts()):
            raise ValueError(f"invalid directed I control: {task['task_id']}")
    else:
        control = config.get("learning_control", {})
        provenance = config.get("policy_checkpoint") or {}
        if (control.get("model_choice_probabilities", {}).get("M") != 0.0
                or control.get("m_forced_operator") != M_OPERATOR
                or control.get("m_forced_operator_min_generation")
                != M_MINIMUM_GENERATION
                or abs(float(control.get("m_forced_operator_probability", -1))
                       - M_PROBABILITY) > EPSILON
                or control.get("m_forced_replace_selector") != M_SELECTOR
                or int(control.get("m_forced_operator_best_of", 0)) != 1
                or provenance.get("checkpoint_sha256")
                != plan["m_infrastructure_checkpoint"]["sha256"]):
            raise ValueError(f"invalid directed M control: {task['task_id']}")
    return {
        "task_id": task["task_id"],
        "execution_shard": task["execution_shard"],
        "instance_id": task["instance_id"],
        "instance_digest": instance["instance_digest"],
        "algorithm_repeat": task["algorithm_repeat"],
        "algorithm_seed": task["algorithm_seed"],
        "method": task["method"],
        "component": component,
        "training_scenario_digest": config["scenario_digest"],
        "final_hypervolume": float(hv),
        "final_front_size": int(complete["final_front_size"]),
        "real_objective_evaluation_count": int(
            complete["real_objective_evaluation_count"]),
        "optimisation_seconds": float(complete["optimisation_seconds"]),
        "objective_evaluation_seconds": float(
            complete["objective_evaluation_seconds"]),
        "inference_seconds": float(complete["inference_seconds"]),
        "run_path": str(run),
        "complete_sha256": _sha256(complete_path),
        "configuration_sha256": _sha256(config_path),
        "front_sha256": _sha256(front_path),
    }


def _command(task, plan, run, device):
    instance = (
        Path(plan["final_test_instances"]["directory"])
        / f"{task['instance_id']}.json")
    command = [
        sys.executable, "-u", str(RUNNER),
        "--data", plan["network"]["path"],
        "--instance", str(instance),
        "--normalization", plan["normalization"]["path"],
        "--out", str(run),
        "--component", _component(task["method"]),
        "--algorithm-seed", str(task["algorithm_seed"]),
        "--evaluation-mode", "ccp",
        "--uncertainty-profile", UNCERTAINTY_PROFILE,
        "--scenario-count", str(TRAINING_SCENARIO_COUNT),
        "--ccp-confidence", str(CCP_CONFIDENCE),
        "--evaluation-budget", str(EVALUATION_BUDGET),
        "--pop", str(POPULATION),
        "--gens", str(GENERATIONS),
        "--device", device,
    ]
    if task["method"] == "I_DIRECTED_BALANCED":
        command.extend(["--initialization-rule-profile", I_PROFILE])
    elif task["method"] == "M_DIRECTED_BALANCED":
        command.extend([
            "--checkpoint", plan["m_infrastructure_checkpoint"]["path"],
            "--inference-action-mode", "argmax",
            "--inference-policy-seed", "45000",
            "--model-choice-probability", "0",
            "--m-forced-operator", M_OPERATOR,
            "--m-forced-operator-min-generation",
            str(M_MINIMUM_GENERATION),
            "--m-forced-operator-probability", str(M_PROBABILITY),
            "--m-forced-replace-selector", M_SELECTOR,
        ])
    return command


def _quarantine(run, out):
    if not run.exists() or not any(run.iterdir()):
        return
    root = out / "interrupted_runs"
    root.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    destination = root / f"{run.name}_{stamp}"
    suffix = 1
    while destination.exists():
        destination = root / f"{run.name}_{stamp}_{suffix}"
        suffix += 1
    shutil.move(str(run), str(destination))


def _execute(task, plan, out, device):
    run = out / "runs" / task["task_id"]
    result = _validate_result(run, task, plan)
    if result is not None:
        return result, "existing"
    _quarantine(run, out)
    run.parent.mkdir(parents=True, exist_ok=True)
    log = out / "logs" / f"{task['task_id']}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["OMP_NUM_THREADS"] = "1"
    environment["MKL_NUM_THREADS"] = "1"
    with log.open("w", encoding="utf-8") as handle:
        process = subprocess.run(
            _command(task, plan, run, device), cwd=ROOT,
            stdout=handle, stderr=subprocess.STDOUT,
            env=environment, check=False)
    if process.returncode:
        raise RuntimeError(
            f"{task['task_id']} exited {process.returncode}; inspect {log}")
    result = _validate_result(run, task, plan)
    if result is None:
        raise RuntimeError(f"task exited without completion: {task['task_id']}")
    return result, "complete"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--shard", type=int, choices=range(SHARD_COUNT),
                        required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error("workers must be positive")
    args.plan = args.plan.resolve()
    args.out = args.out.resolve()
    try:
        plan = _verify_plan(args.plan)
        tasks = [
            task for task in plan["schedule"]
            if task["execution_shard"] == args.shard]
        if len(tasks) != 30:
            raise ValueError("each formal shard must contain exactly 30 tasks")
    except (KeyError, OSError, TypeError, ValueError,
            json.JSONDecodeError) as error:
        parser.error(str(error))
    config = {
        "schema_version": 1,
        "plan_path": str(args.plan),
        "plan_digest": plan["plan_digest"],
        "schedule_digest": plan["schedule_digest"],
        "shard": args.shard,
        "shard_count": SHARD_COUNT,
        "task_count": len(tasks),
        "task_ids": [task["task_id"] for task in tasks],
        "device": args.device,
    }
    if args.preflight_only:
        print(json.dumps({"preflight": "passed", **config}, indent=2))
        return
    config_path = args.out / "shard_configuration.json"
    if config_path.is_file() and _read(config_path) != config:
        parser.error("existing shard configuration differs; use a new output")
    if not config_path.is_file():
        if args.out.exists() and any(args.out.iterdir()):
            parser.error("non-empty shard output lacks matching configuration")
        _atomic_json(config_path, config)

    completed = {}
    for task in tasks:
        value = _validate_result(args.out / "runs" / task["task_id"], task, plan)
        if value is not None:
            completed[task["task_id"]] = value
    pending = [task for task in tasks if task["task_id"] not in completed]
    initial_completed = len(completed)
    started = time.perf_counter()
    statuses = {"existing": initial_completed, "complete": 0}
    print(
        f"[FORMAL SHARD {args.shard}] tasks={len(tasks)} "
        f"existing={initial_completed} pending={len(pending)} "
        f"workers={args.workers}", flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                _execute, task, plan, args.out, args.device): task
            for task in pending
        }
        for future in as_completed(futures):
            task = futures[future]
            result, status = future.result()
            completed[task["task_id"]] = result
            statuses[status] = statuses.get(status, 0) + 1
            elapsed = time.perf_counter() - started
            done_now = len(completed) - initial_completed
            eta_hours = (
                elapsed / done_now * (len(tasks) - len(completed)) / 3600.0
                if done_now else 0.0)
            ordered = [completed[key] for key in sorted(completed)]
            _atomic_json(args.out / "shard_progress.json", {
                "completed": len(completed) == len(tasks),
                "shard": args.shard,
                "completed_task_count": len(completed),
                "task_count": len(tasks),
                "estimated_remaining_hours": eta_hours,
                "last_task": task["task_id"],
                "results": ordered,
            })
            print(
                f"[FORMAL SHARD {args.shard}] {len(completed)}/{len(tasks)} "
                f"{task['task_id']} eta_hours={eta_hours:.2f}", flush=True)

    results = [completed[key] for key in sorted(completed)]
    if len(results) != len(tasks):
        raise RuntimeError("formal shard ended without complete task coverage")
    results_path = args.out / "results.json"
    _atomic_json(results_path, results)
    complete = {
        "completed": True,
        "shard": args.shard,
        "task_count": len(tasks),
        "initial_completed_task_count": initial_completed,
        "newly_completed_task_count": len(tasks) - initial_completed,
        "status_counts": statuses,
        "wall_seconds_this_invocation": time.perf_counter() - started,
        "total_optimisation_seconds": sum(
            row["optimisation_seconds"] for row in results),
        "results_sha256": _sha256(results_path),
        **config,
    }
    _atomic_json(args.out / "SHARD_COMPLETE.json", complete)
    print(json.dumps({
        "completed": True,
        "shard": args.shard,
        "task_count": len(results),
        "wall_hours": complete["wall_seconds_this_invocation"] / 3600.0,
    }, indent=2))


if __name__ == "__main__":
    main()
