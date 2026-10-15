#!/usr/bin/env python3
"""Run the training-only RF-Reward C+ evidence gate reproducibly."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from multi_instance_logging import write_json


ROOT = Path(__file__).resolve().parent
COLLECTOR = ROOT / "run_rf_reward_collection_ccp100.py"
TRAINER = ROOT / "train_rf_reward_ccp100.py"
COLLECTION_METHOD = "rf-reward-exploration-collector"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _digest(value) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _select_records(catalog: dict, per_config: int, selection_seed: int) -> list[dict]:
    records = [row for row in catalog["instances"] if row["split"] == "train"]
    counts = Counter(row["configuration"] for row in records)
    expected = Counter({f"S{i}": 50 for i in range(9)})
    if len(records) != 450 or counts != expected:
        raise ValueError(
            f"expected 450 training manifests (50 per S0-S8), found {counts}")
    if per_config < 5 or per_config > 50:
        raise ValueError("instances-per-config must be between 5 and 50")
    grouped = defaultdict(list)
    for record in records:
        grouped[record["configuration"]].append(record)
    selected = []
    for configuration in (f"S{i}" for i in range(9)):
        ordered = sorted(grouped[configuration], key=lambda row: (
            hashlib.sha256(
                f"{selection_seed}:{row['instance_id']}".encode("utf-8")
            ).hexdigest(),
            row["instance_id"],
        ))
        selected.extend(ordered[:per_config])
    return selected


def _seeds(args, trajectory: int, instance_index: int) -> tuple[int, int]:
    offset = trajectory * 1_000_000 + instance_index
    return args.algorithm_seed_base + offset, args.policy_seed_base + offset


def _completed(run_dir: Path, record: dict, args, trajectory: int,
               instance_index: int) -> bool:
    complete_path = run_dir / "COMPLETE.json"
    config_path = run_dir / "configuration.json"
    collection_path = run_dir / "rf_collection_configuration.json"
    paths = (complete_path, config_path, collection_path)
    if not all(path.exists() for path in paths):
        return False
    complete = json.loads(complete_path.read_text(encoding="utf-8"))
    config = json.loads(config_path.read_text(encoding="utf-8"))
    collection = json.loads(collection_path.read_text(encoding="utf-8"))
    algorithm_seed, policy_seed = _seeds(args, trajectory, instance_index)
    expected = {
        "method": COLLECTION_METHOD,
        "evaluation_budget": args.evaluation_budget,
        "population": args.pop,
        "generations": args.gens,
        "instance_id": record["instance_id"],
        "instance_digest": record["instance_digest"],
        "algorithm_seed": algorithm_seed,
        "policy_seed": policy_seed,
        "uniform_mixture": args.uniform_mixture,
        "normalization_sha256": args.normalization_digest,
    }
    observed = {
        "method": complete.get("method"),
        "evaluation_budget": complete.get("evaluation_budget"),
        "population": config.get("population"),
        "generations": config.get("generations"),
        "instance_id": config.get("instance_id"),
        "instance_digest": config.get("instance_digest"),
        "algorithm_seed": collection.get("algorithm_seed"),
        "policy_seed": collection.get("policy_seed"),
        "uniform_mixture": collection.get("uniform_mixture"),
        "normalization_sha256": collection.get("normalization_sha256"),
    }
    if complete.get("completed") is not True or observed != expected:
        raise ValueError(
            f"completed collection metadata mismatch at {run_dir}: "
            f"expected={expected}, observed={observed}")
    if int(complete.get("ccp_evaluation_count", args.evaluation_budget + 1)) > args.evaluation_budget:
        raise ValueError(f"collection exceeded evaluation budget: {run_dir}")
    if not (run_dir / "final_feasible_nondominated.json").exists():
        raise ValueError(f"completed collection is missing its final front: {run_dir}")
    return True


def _run_one(task, args, selected: list[dict]):
    trajectory, instance_index = task
    record = selected[instance_index]
    trajectory_dir = args.out / f"trajectory_{trajectory + 1}"
    run_dir = trajectory_dir / "runs" / record["instance_id"]
    if _completed(run_dir, record, args, trajectory, instance_index):
        return trajectory, record["instance_id"], "skipped", 0.0
    algorithm_seed, policy_seed = _seeds(args, trajectory, instance_index)
    command = [
        sys.executable, "-u", str(COLLECTOR),
        "--data", str(args.data),
        "--instance", str(args.instances / f"{record['instance_id']}.json"),
        "--normalization", str(args.normalization),
        "--out", str(run_dir),
        "--algorithm-seed", str(algorithm_seed),
        "--policy-seed", str(policy_seed),
        "--uniform-mixture", str(args.uniform_mixture),
        "--evaluation-budget", str(args.evaluation_budget),
        "--pop", str(args.pop), "--gens", str(args.gens),
    ]
    environment = os.environ.copy()
    environment.update({
        "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1",
        "PYTHONUNBUFFERED": "1",
    })
    log_path = trajectory_dir / "logs" / f"{record['instance_id']}.log"
    started = time.perf_counter()
    with log_path.open("x", encoding="utf-8") as handle:
        result = subprocess.run(
            command, stdout=handle, stderr=subprocess.STDOUT,
            cwd=ROOT.parent, env=environment, check=False)
    if result.returncode != 0:
        raise RuntimeError(
            f"trajectory {trajectory + 1} {record['instance_id']} failed with "
            f"exit code {result.returncode}; see {log_path}")
    if not _completed(run_dir, record, args, trajectory, instance_index):
        raise RuntimeError(f"collection exited without valid completion: {run_dir}")
    return (trajectory, record["instance_id"], "completed",
            time.perf_counter() - started)


def _model_summary(trajectory_dir: Path, expected_instances: list[str],
                   rf_seed: int, cv_seed: int) -> dict | None:
    summary_path = trajectory_dir / "model" / "training_summary.json"
    model_path = trajectory_dir / "model" / "rf_reward.joblib"
    if not summary_path.exists() or not model_path.exists():
        return None
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if (summary.get("instance_count") != len(expected_instances)
            or summary.get("rf_seed") != rf_seed
            or summary.get("cv_seed") != cv_seed
            or summary.get("grouped_validation_folds") != 5
            or summary.get("configuration_balanced_folds") is not True
            or sorted(summary["best"]["validation"]["per_instance"][i]["instance_id"]
                      for i in range(len(expected_instances))) != sorted(expected_instances)
            or summary.get("model_sha256") != _sha256(model_path)):
        raise ValueError(f"model provenance mismatch: {trajectory_dir / 'model'}")
    return summary


def _train_one(trajectory: int, args, selected: list[dict]) -> dict:
    trajectory_dir = args.out / f"trajectory_{trajectory + 1}"
    expected_instances = [row["instance_id"] for row in selected]
    rf_seed = args.rf_seed_base + trajectory
    summary = _model_summary(trajectory_dir, expected_instances, rf_seed, args.cv_seed)
    if summary is None:
        model_dir = trajectory_dir / "model"
        if model_dir.exists() and any(model_dir.iterdir()):
            raise RuntimeError(
                f"incomplete non-empty model directory requires inspection: {model_dir}")
        command = [
            sys.executable, "-u", str(TRAINER),
            "--runs", str(trajectory_dir / "runs"),
            "--out", str(model_dir),
            "--rf-seed", str(rf_seed), "--jobs", str(args.rf_jobs),
            "--folds", "5", "--configuration-balanced-folds",
            "--cv-seed", str(args.cv_seed),
        ]
        log_path = trajectory_dir / "model_training.log"
        environment = os.environ.copy()
        environment["PYTHONUNBUFFERED"] = "1"
        with log_path.open("x", encoding="utf-8") as handle:
            result = subprocess.run(
                command, stdout=handle, stderr=subprocess.STDOUT,
                cwd=ROOT.parent, env=environment, check=False)
        if result.returncode != 0:
            raise RuntimeError(
                f"trajectory {trajectory + 1} model training failed; see {log_path}")
        summary = _model_summary(
            trajectory_dir, expected_instances, rf_seed, args.cv_seed)
        if summary is None:
            raise RuntimeError(
                f"trajectory {trajectory + 1} model exited without valid artifacts")
    validation = summary["best"]["validation"]
    baseline = summary["training_fold_mean_baseline"]
    result = {
        "trajectory": trajectory + 1,
        "rf_seed": rf_seed,
        "sample_count": summary["sample_count"],
        "model_sha256": summary["model_sha256"],
        "training_data_digest": summary["training_data_digest"],
        "rf_mean_instance_mae": validation["mean_instance_mae"],
        "rf_mean_instance_rmse": validation["mean_instance_rmse"],
        "baseline_mean_instance_mae": baseline["mean_instance_mae"],
        "baseline_mean_instance_rmse": baseline["mean_instance_rmse"],
        "rf_beats_baseline_mae": (
            validation["mean_instance_mae"] < baseline["mean_instance_mae"]),
        "rf_beats_baseline_rmse": (
            validation["mean_instance_rmse"] < baseline["mean_instance_rmse"]),
        "best_parameters": summary["best"]["parameters"],
    }
    write_json(trajectory_dir / "TRAJECTORY_COMPLETE.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path,
                        default=ROOT / "data/data_expanded.xlsx")
    parser.add_argument("--instances", type=Path, required=True)
    parser.add_argument("--normalization", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--catalog-digest", required=True)
    parser.add_argument("--normalization-digest", required=True)
    parser.add_argument("--instances-per-config", type=int, default=5)
    parser.add_argument("--trajectories", type=int, default=3)
    parser.add_argument("--evaluation-budget", type=int, default=1_000)
    parser.add_argument("--uniform-mixture", type=float, default=0.5)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--rf-jobs", type=int, default=4)
    parser.add_argument("--pop", type=int, default=100)
    parser.add_argument("--gens", type=int, default=1_000)
    parser.add_argument("--selection-seed", type=int, default=82_000_000)
    parser.add_argument("--algorithm-seed-base", type=int, default=83_000_000)
    parser.add_argument("--policy-seed-base", type=int, default=93_000_000)
    parser.add_argument("--rf-seed-base", type=int, default=103_000_000)
    parser.add_argument("--cv-seed", type=int, default=104_000_000)
    args = parser.parse_args()
    args.data = args.data.resolve()
    args.instances = args.instances.resolve()
    args.normalization = args.normalization.resolve()
    args.out = args.out.resolve()
    if args.workers < 1 or args.workers > 16:
        parser.error("workers must be between 1 and 16")
    if args.rf_jobs < 1 or args.rf_jobs > 16:
        parser.error("rf-jobs must be between 1 and 16")
    if args.trajectories < 1 or args.trajectories > 9:
        parser.error("trajectories must be between 1 and 9")
    if not 0.0 < args.uniform_mixture <= 1.0:
        parser.error("uniform-mixture must be in (0, 1]")
    if _sha256(args.normalization) != args.normalization_digest:
        parser.error("normalization digest does not match the frozen command")
    normalization = json.loads(args.normalization.read_text(encoding="utf-8"))
    if normalization.get("source") != "training-only":
        parser.error("normalization must declare source=training-only")
    catalog_path = args.instances / "catalog.json"
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    if catalog.get("catalog_digest") != args.catalog_digest:
        parser.error("catalog digest does not match the frozen command")
    source_data_sha256 = _sha256(args.data)
    if catalog.get("source_data_sha256") != source_data_sha256:
        parser.error("catalog source-data digest does not match --data")
    try:
        selected = _select_records(
            catalog, args.instances_per_config, args.selection_seed)
    except ValueError as error:
        parser.error(str(error))
    selection = [{key: row[key] for key in (
        "instance_id", "configuration", "instance_digest")}
        for row in selected]
    selection_digest = _digest(selection)
    for record in selected:
        manifest_path = args.instances / f"{record['instance_id']}.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (manifest.get("split") != "train"
                or manifest.get("instance_id") != record["instance_id"]
                or manifest.get("configuration") != record["configuration"]
                or manifest.get("instance_digest") != record["instance_digest"]):
            parser.error(f"selected manifest disagrees with catalog: {manifest_path}")
    batch_config = {
        "schema_version": 1,
        "stage": "RF-Reward C+ training-only evidence gate",
        "catalog_digest": args.catalog_digest,
        "normalization_digest": args.normalization_digest,
        "source_data_sha256": source_data_sha256,
        "instances_per_configuration": args.instances_per_config,
        "configuration_count": 9,
        "instance_count_per_trajectory": len(selected),
        "trajectories": args.trajectories,
        "evaluation_budget_per_instance": args.evaluation_budget,
        "maximum_ccp100_evaluations": (
            len(selected) * args.trajectories * args.evaluation_budget),
        "uniform_mixture": args.uniform_mixture,
        "primary_sample_weighting": "none",
        "workers": args.workers,
        "rf_jobs": args.rf_jobs,
        "population": args.pop,
        "generations": args.gens,
        "selection_seed": args.selection_seed,
        "selection_digest": selection_digest,
        "algorithm_seed_base": args.algorithm_seed_base,
        "policy_seed_base": args.policy_seed_base,
        "rf_seed_base": args.rf_seed_base,
        "cv_seed": args.cv_seed,
        "thread_limits_per_collection_worker": 1,
        "selected_instances": selection,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    config_path = args.out / "batch_configuration.json"
    if config_path.exists():
        existing = json.loads(config_path.read_text(encoding="utf-8"))
        if existing != batch_config:
            parser.error("existing batch configuration differs; use a new output directory")
    else:
        write_json(config_path, batch_config)
    for trajectory in range(args.trajectories):
        trajectory_dir = args.out / f"trajectory_{trajectory + 1}"
        (trajectory_dir / "runs").mkdir(parents=True, exist_ok=True)
        (trajectory_dir / "logs").mkdir(exist_ok=True)
    tasks = [(trajectory, index)
             for trajectory in range(args.trajectories)
             for index in range(len(selected))]
    incomplete = []
    already_complete = 0
    for trajectory, index in tasks:
        trajectory_dir = args.out / f"trajectory_{trajectory + 1}"
        run_dir = trajectory_dir / "runs" / selected[index]["instance_id"]
        if _completed(run_dir, selected[index], args, trajectory, index):
            already_complete += 1
        elif run_dir.exists() and any(run_dir.iterdir()):
            incomplete.append(str(run_dir))
        log_path = trajectory_dir / "logs" / f"{selected[index]['instance_id']}.log"
        if not _completed(run_dir, selected[index], args, trajectory, index) and log_path.exists():
            incomplete.append(str(log_path))
    if incomplete:
        raise RuntimeError(
            "incomplete outputs require inspection before resume:\n"
            + "\n".join(sorted(set(incomplete))))
    total = len(tasks)
    print(
        f"[RF-CPLUS] total={total} already_complete={already_complete} "
        f"workers={args.workers} budget={args.evaluation_budget} "
        f"selection_digest={selection_digest}", flush=True)
    started = time.perf_counter()
    completed = already_complete
    durations = []
    pending = [task for task in tasks if not _completed(
        args.out / f"trajectory_{task[0] + 1}" / "runs"
        / selected[task[1]]["instance_id"],
        selected[task[1]], args, task[0], task[1])]
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(_run_one, task, args, selected): task
                   for task in pending}
        for future in as_completed(futures):
            trajectory, instance_id, status, duration = future.result()
            completed += 1
            if status == "completed":
                durations.append(duration)
            elapsed = time.perf_counter() - started
            new_completed = max(1, completed - already_complete)
            remaining_h = max(
                0.0, elapsed / new_completed * (total - completed) / 3600)
            print(
                f"[RF-CPLUS] {completed:03d}/{total} trajectory={trajectory + 1} "
                f"{instance_id} {status} run_seconds={duration:.1f} "
                f"eta_hours={remaining_h:.2f}", flush=True)
    trajectory_results = [
        _train_one(trajectory, args, selected)
        for trajectory in range(args.trajectories)
    ]
    total_evaluations = 0
    for trajectory, index in tasks:
        complete = json.loads((
            args.out / f"trajectory_{trajectory + 1}" / "runs"
            / selected[index]["instance_id"] / "COMPLETE.json"
        ).read_text(encoding="utf-8"))
        total_evaluations += int(complete["ccp_evaluation_count"])
    summary = {
        "completed": True,
        "wall_seconds": time.perf_counter() - started,
        "resumed_collection_count": already_complete,
        "new_collection_count": len(durations),
        "mean_new_collection_seconds": (
            sum(durations) / len(durations) if durations else 0.0),
        "actual_ccp100_evaluations": total_evaluations,
        "trajectory_results": trajectory_results,
        "rf_beats_baseline_rmse_count": sum(
            row["rf_beats_baseline_rmse"] for row in trajectory_results),
        "rf_beats_baseline_mae_count": sum(
            row["rf_beats_baseline_mae"] for row in trajectory_results),
        "all_trajectories_beat_baseline_rmse": all(
            row["rf_beats_baseline_rmse"] for row in trajectory_results),
        "interpretation": (
            "training-only C+ evidence gate; not formal test or OOS-HV evidence"),
        **batch_config,
    }
    write_json(args.out / "BATCH_COMPLETE.json", summary)
    print(json.dumps({key: summary[key] for key in (
        "completed", "actual_ccp100_evaluations",
        "rf_beats_baseline_rmse_count", "rf_beats_baseline_mae_count",
        "all_trajectories_beat_baseline_rmse")}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
