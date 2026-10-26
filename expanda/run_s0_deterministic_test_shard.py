#!/usr/bin/env python3
"""Execute one resumable shard of the frozen S0 deterministic final test."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import time

from s0_deterministic_test_plan import canonical_digest


ROOT = Path(__file__).resolve().parent
SINGLE_RUNNER = ROOT / "run_learning_components_deterministic.py"


def _read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _atomic_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _resolve(value):
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _verify_plan(path: Path):
    plan = _read(path)
    expected = plan.get("plan_digest")
    payload = dict(plan)
    payload.pop("plan_digest", None)
    if not expected or canonical_digest(payload) != expected:
        raise ValueError("final-test plan digest mismatch")
    schedule = plan.get("schedule", [])
    if (plan.get("formal") is not True
            or plan.get("population") != 250
            or plan.get("evaluation_budget") != 30_000
            or plan.get("generation_cap") != 250
            or len(schedule) != 420
            or canonical_digest(schedule) != plan.get("schedule_digest")
            or plan.get("execution", {}).get("shard_count") != 2):
        raise ValueError("final-test plan is not the frozen 420-run design")
    if _sha256(SINGLE_RUNNER) != plan["implementation_sha256"][
            "run_learning_components_deterministic.py"]:
        raise ValueError("scientific single-run implementation changed after freezing")
    artifacts = [
        ("network workbook", ROOT / "data/data_expanded.xlsx",
         plan["source_data_sha256"]),
        ("objective normalization", _resolve(plan["normalization"]["path"]),
         plan["normalization"]["sha256"]),
        ("Rule-I normalization",
         _resolve(plan["rule_initialization_normalization"]["path"]),
         plan["rule_initialization_normalization"]["sha256"]),
        ("final-test catalog",
         _resolve(plan["final_test_instances"]["directory"]) / "catalog.json",
         plan["final_test_instances"]["catalog_sha256"]),
    ]
    for component in ("I", "C", "M"):
        for seed in plan["policy_seeds"]:
            record = plan["checkpoints"][component][str(seed)]
            artifacts.append((
                f"{component}{seed} checkpoint", _resolve(record["path"]),
                record["sha256"]))
    for label, artifact, digest in artifacts:
        if not artifact.is_file() or _sha256(artifact) != digest:
            raise ValueError(f"{label} is missing or changed: {artifact}")
    return plan


def _frozen_updates_are_zero(method, summary):
    value = summary.get("ppo_updates")
    if method == "ICM":
        return value == {"I": 0, "C": 0, "M": 0}
    return value == 0


def _validate_result(run_root: Path, task: dict, plan: dict):
    complete_path = run_root / "COMPLETE.json"
    config_path = run_root / "configuration.json"
    front_path = run_root / "final_feasible_nondominated.json"
    if not complete_path.is_file():
        return None
    if not config_path.is_file() or not front_path.is_file():
        raise ValueError(f"incomplete result files for {task['task_id']}")
    summary = _read(complete_path)
    config = _read(config_path)
    front = _read(front_path)
    instance = next(
        row for row in plan["final_test_instances"]["instances"]
        if row["instance_id"] == task["instance_id"])
    hv = summary.get("final_hypervolume")
    normal = config.get("normalization", {})
    if (summary.get("completed") is not True
            or summary.get("component") != task["method"]
            or summary.get("real_objective_evaluation_count") != 30_000
            or summary.get("evaluation_budget") != 30_000
            or summary.get("stop_reason") != "evaluation_budget"
            or not isinstance(hv, (int, float)) or not math.isfinite(hv)
            or summary.get("final_front_size") != len(front)
            or len(front) < 1
            or not _frozen_updates_are_zero(task["method"], summary)
            or config.get("component") != task["method"]
            or config.get("evaluation_mode") != "deterministic"
            or config.get("instance_id") != task["instance_id"]
            or config.get("instance_digest") != instance["instance_digest"]
            or config.get("population") != 250
            or config.get("generations") != 250
            or config.get("evaluation_budget") != 30_000
            or config.get("seeds", {}).get("algorithm")
            != task["algorithm_seed"]
            or config.get("policy_training") is not False
            or tuple(normal.get("minimum", ()))
            != tuple(_read(_resolve(plan["normalization"]["path"]))["minimum"])
            or tuple(normal.get("maximum", ()))
            != tuple(_read(_resolve(plan["normalization"]["path"]))["maximum"])
            or tuple(normal.get("reference", ()))
            != tuple(plan["normalization"]["reference"])):
        raise ValueError(f"invalid frozen result for {task['task_id']}")
    provenance = config.get("policy_checkpoint")
    if task["method"] in ("BASE", "RULE-I"):
        if provenance is not None:
            raise ValueError(f"unexpected checkpoint for {task['task_id']}")
        if (task["method"] == "RULE-I"
                and config.get("initialization_rule", {}).get(
                    "normalization_digest")
                != plan["rule_initialization_normalization"][
                    "normalization_digest"]):
            raise ValueError("wrong Rule-I normalization in completed result")
    elif task["method"] == "ICM":
        if not isinstance(provenance, dict) or set(provenance) != {"I", "C", "M"}:
            raise ValueError(f"invalid ICM checkpoint provenance: {task['task_id']}")
        for component in ("I", "C", "M"):
            expected = plan["checkpoints"][component][str(task["policy_seed"])]
            if provenance[component].get("checkpoint_sha256") != expected["sha256"]:
                raise ValueError(f"wrong {component} checkpoint in {task['task_id']}")
    else:
        expected = plan["checkpoints"][task["method"]][str(task["policy_seed"])]
        if (not isinstance(provenance, dict)
                or provenance.get("checkpoint_sha256") != expected["sha256"]
                or provenance.get("policy_seed") != task["policy_seed"]):
            raise ValueError(f"wrong checkpoint in {task['task_id']}")
    return {
        "task_id": task["task_id"],
        "execution_shard": task["execution_shard"],
        "instance_id": task["instance_id"],
        "instance_digest": instance["instance_digest"],
        "algorithm_repeat": task["algorithm_repeat"],
        "algorithm_seed": task["algorithm_seed"],
        "method": task["method"],
        "policy_seed": task["policy_seed"],
        "final_hypervolume": float(hv),
        "final_front_size": int(summary["final_front_size"]),
        "real_objective_evaluation_count": int(
            summary["real_objective_evaluation_count"]),
        "optimisation_seconds": float(summary["optimisation_seconds"]),
        "objective_evaluation_seconds": float(
            summary["objective_evaluation_seconds"]),
        "feature_construction_seconds": float(
            summary["feature_construction_seconds"]),
        "inference_seconds": float(summary["inference_seconds"]),
        "ppo_update_seconds": float(summary["ppo_update_seconds"]),
        "other_optimisation_seconds": float(
            summary["other_optimisation_seconds"]),
        "decision_counts": summary.get("decision_counts", {}),
        "complete_sha256": _sha256(complete_path),
        "configuration_sha256": _sha256(config_path),
        "front_sha256": _sha256(front_path),
    }


def _quarantine(path: Path, out: Path):
    if not path.exists() or not any(path.iterdir()):
        return
    destination_root = out / "interrupted_runs"
    destination_root.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    destination = destination_root / f"{path.name}_{stamp}"
    suffix = 1
    while destination.exists():
        destination = destination_root / f"{path.name}_{stamp}_{suffix}"
        suffix += 1
    shutil.move(str(path), str(destination))
    print(f"[RESUME] quarantined incomplete run at {destination}", flush=True)


def _command(task, plan, run_root, device):
    instance = (
        _resolve(plan["final_test_instances"]["directory"])
        / f"{task['instance_id']}.json")
    command = [
        sys.executable, "-u", str(SINGLE_RUNNER),
        "--data", str(ROOT / "data/data_expanded.xlsx"),
        "--instance", str(instance),
        "--normalization", str(_resolve(plan["normalization"]["path"])),
        "--component", task["method"],
        "--algorithm-seed", str(task["algorithm_seed"]),
        "--evaluation-budget", "30000",
        "--pop", "250", "--gens", "250",
        "--out", str(run_root),
    ]
    if task["method"] == "RULE-I":
        command.extend([
            "--initialization-rule-normalization",
            str(_resolve(plan["rule_initialization_normalization"]["path"])),
        ])
    elif task["method"] in ("I", "C", "M"):
        command.extend([
            "--checkpoint",
            str(_resolve(plan["checkpoints"][task["method"]][
                str(task["policy_seed"])]["path"])),
        ])
    elif task["method"] == "ICM":
        for component in ("I", "C", "M"):
            command.extend([
                f"--{component.lower()}-checkpoint",
                str(_resolve(plan["checkpoints"][component][
                    str(task["policy_seed"])]["path"])),
            ])
    if device:
        command.extend(["--device", device])
    return command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--shard", type=int, choices=(0, 1), required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    args.plan = args.plan.resolve()
    args.out = args.out.resolve()
    try:
        plan = _verify_plan(args.plan)
        tasks = [
            row for row in plan["schedule"]
            if row["execution_shard"] == args.shard
        ]
        if len(tasks) != 210:
            raise ValueError("each frozen execution shard must contain 210 tasks")
    except (KeyError, OSError, TypeError, ValueError,
            json.JSONDecodeError) as error:
        parser.error(str(error))
    config = {
        "schema_version": 1,
        "stage": "S0 deterministic final-test execution shard",
        "plan_sha256": _sha256(args.plan),
        "plan_digest": plan["plan_digest"],
        "schedule_digest": plan["schedule_digest"],
        "shard": args.shard,
        "shard_count": 2,
        "task_count": len(tasks),
        "task_ids": [row["task_id"] for row in tasks],
        "device": args.device,
        "python_executable": sys.executable,
    }
    if args.preflight_only:
        print(json.dumps({
            "preflight": "passed", **config,
        }, indent=2, allow_nan=False))
        return
    config_path = args.out / "shard_configuration.json"
    if config_path.exists():
        if _read(config_path) != config:
            parser.error("existing shard configuration differs; use a new output")
    elif args.out.exists() and any(args.out.iterdir()):
        parser.error("non-empty output has no matching shard configuration")
    else:
        args.out.mkdir(parents=True, exist_ok=True)
        _atomic_json(config_path, config)

    results = []
    for task in tasks:
        result = _validate_result(
            args.out / "runs" / task["task_id"], task, plan)
        if result is not None:
            results.append(result)
    initial_completed = len(results)
    started = time.perf_counter()
    for index, task in enumerate(tasks, start=1):
        run_root = args.out / "runs" / task["task_id"]
        result = _validate_result(run_root, task, plan)
        if result is not None:
            continue
        _quarantine(run_root, args.out)
        run_root.parent.mkdir(parents=True, exist_ok=True)
        log_path = args.out / "logs" / f"{task['task_id']}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        print(
            f"[SHARD {args.shard}] {len(results) + 1}/{len(tasks)} "
            f"{task['task_id']} seed={task['algorithm_seed']}", flush=True)
        one_started = time.perf_counter()
        with log_path.open("w", encoding="utf-8") as handle:
            process = subprocess.run(
                _command(task, plan, run_root, args.device), cwd=ROOT,
                stdout=handle, stderr=subprocess.STDOUT, check=False)
        if process.returncode:
            raise RuntimeError(
                f"task failed with exit {process.returncode}; inspect {log_path}")
        result = _validate_result(run_root, task, plan)
        if result is None:
            raise RuntimeError(f"task exited without completion: {task['task_id']}")
        results.append(result)
        results.sort(key=lambda row: row["task_id"])
        completed_weight = sum(
            row["estimated_runtime_weight"] for row in tasks
            if any(item["task_id"] == row["task_id"] for item in results))
        total_weight = sum(row["estimated_runtime_weight"] for row in tasks)
        total_seconds = sum(row["optimisation_seconds"] for row in results)
        remaining_hours = (
            total_seconds / completed_weight * (total_weight - completed_weight)
            / 3600.0)
        progress = {
            "completed": len(results) == len(tasks),
            "shard": args.shard,
            "completed_task_count": len(results),
            "task_count": len(tasks),
            "completed_weight": completed_weight,
            "total_weight": total_weight,
            "estimated_remaining_hours": remaining_hours,
            "elapsed_seconds_this_invocation": time.perf_counter() - started,
            "total_optimisation_seconds": total_seconds,
            "last_task": task["task_id"],
            "results": results,
        }
        _atomic_json(args.out / "shard_progress.json", progress)
        print(
            f"[SHARD {args.shard}] completed {len(results)}/{len(tasks)} "
            f"seconds={time.perf_counter() - one_started:.1f} "
            f"eta_hours={remaining_hours:.2f}", flush=True)

    results.sort(key=lambda row: row["task_id"])
    results_path = args.out / "results.json"
    _atomic_json(results_path, results)
    complete = {
        "completed": True,
        "shard": args.shard,
        "task_count": len(tasks),
        "initial_completed_task_count": initial_completed,
        "newly_completed_task_count": len(tasks) - initial_completed,
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
        "task_count": len(tasks),
        "wall_hours_this_invocation": complete["wall_seconds_this_invocation"] / 3600,
        "results": str(results_path),
    }, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

