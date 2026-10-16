#!/usr/bin/env python3
"""Run frozen, resumable stage-D RF-Reward collection and model fitting."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from multi_instance_logging import write_json
from run_rf_reward_cplus_batch import (
    _completed, _digest, _run_one, _select_records, _sha256,
)
from transformer_ppo.rf_reward import FEATURE_NAMES


ROOT = Path(__file__).resolve().parent
TRAINER = ROOT / "train_rf_reward_ccp100.py"

INSTANCES_PER_CONFIGURATION = 50
TRAJECTORIES = 3
EVALUATION_BUDGET = 5_000
POPULATION = 100
GENERATIONS = 1_000
UNIFORM_MIXTURE = 0.5
FOLDS = 5
SELECTION_SEED = 141_000_000
ALGORITHM_SEED_BASE = 150_000_000
POLICY_SEED_BASE = 160_000_000
RF_SEED_BASE = 170_000_000
CV_SEED = 180_000_000
FIXED_PARAMETERS = {
    "n_estimators": 200,
    "max_depth": 8,
    "min_samples_leaf": 5,
    "max_features": 0.5,
}
EXPECTED_NORMALIZATION_SHA256 = (
    "de73404449c2b612609ad5248f815793f99b676e4260e2068ae403f043addbcd"
)


def _validate_inputs(args):
    if args.normalization_digest != EXPECTED_NORMALIZATION_SHA256:
        raise ValueError(
            "formal stage D requires the predeclared normalization digest "
            f"{EXPECTED_NORMALIZATION_SHA256}")
    if _sha256(args.normalization) != args.normalization_digest:
        raise ValueError("normalization digest does not match the frozen command")
    normalization = json.loads(args.normalization.read_text(encoding="utf-8"))
    if normalization.get("source") != "training-only":
        raise ValueError("normalization must declare source=training-only")
    catalog_path = args.instances / "catalog.json"
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    if catalog.get("catalog_digest") != args.catalog_digest:
        raise ValueError("catalog digest does not match the frozen command")
    source_data_sha256 = _sha256(args.data)
    if catalog.get("source_data_sha256") != source_data_sha256:
        raise ValueError("catalog source-data digest does not match --data")
    selected = _select_records(
        catalog, INSTANCES_PER_CONFIGURATION, SELECTION_SEED)
    if len(selected) != 450:
        raise ValueError("formal stage D requires exactly 450 training instances")
    for record in selected:
        manifest_path = args.instances / f"{record['instance_id']}.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (manifest.get("split") != "train"
                or manifest.get("instance_id") != record["instance_id"]
                or manifest.get("configuration") != record["configuration"]
                or manifest.get("instance_digest") != record["instance_digest"]):
            raise ValueError(f"selected manifest disagrees with catalog: {manifest_path}")
    return catalog, selected, source_data_sha256


def _batch_configuration(args, selected, source_data_sha256):
    selection = [{key: row[key] for key in (
        "instance_id", "configuration", "instance_digest")}
        for row in selected]
    return {
        "schema_version": 1,
        "stage": "RF-Reward formal training D; training instances only",
        "catalog_digest": args.catalog_digest,
        "normalization_digest": args.normalization_digest,
        "source_data_sha256": source_data_sha256,
        "instances_per_configuration": INSTANCES_PER_CONFIGURATION,
        "configuration_count": 9,
        "instance_count_per_trajectory": len(selected),
        "trajectories": TRAJECTORIES,
        "collection_count": len(selected) * TRAJECTORIES,
        "evaluation_budget_per_instance": EVALUATION_BUDGET,
        "maximum_ccp100_evaluations": (
            len(selected) * TRAJECTORIES * EVALUATION_BUDGET),
        "population": POPULATION,
        "generations": GENERATIONS,
        "uniform_mixture": UNIFORM_MIXTURE,
        "collection_policy": "(1-rho)*p_rule+rho*uniform_eligible",
        "primary_sample_weighting": "none",
        "fixed_rf_parameters": FIXED_PARAMETERS,
        "hyperparameter_protocol": (
            "frozen from C+ before stage D; no stage-D model selection"),
        "diagnostic_validation": {
            "folds": FOLDS,
            "configuration_balanced": True,
            "instance_grouped": True,
            "cv_seed": CV_SEED,
            "same_folds_across_trajectories": True,
        },
        "workers": args.workers,
        "rf_jobs": args.rf_jobs,
        "thread_limits_per_collection_worker": 1,
        "selection_seed": SELECTION_SEED,
        "selection_digest": _digest(selection),
        "algorithm_seed_base": ALGORITHM_SEED_BASE,
        "policy_seed_base": POLICY_SEED_BASE,
        "rf_seed_base": RF_SEED_BASE,
        "feature_schema_digest": _digest(list(FEATURE_NAMES)),
        "batch_runner_sha256": _sha256(Path(__file__)),
        "collector_sha256": _sha256(ROOT / "run_rf_reward_collection_ccp100.py"),
        "trainer_sha256": _sha256(TRAINER),
        "selected_instances": selection,
    }


def _apply_frozen_args(args):
    args.instances_per_config = INSTANCES_PER_CONFIGURATION
    args.trajectories = TRAJECTORIES
    args.evaluation_budget = EVALUATION_BUDGET
    args.pop = POPULATION
    args.gens = GENERATIONS
    args.uniform_mixture = UNIFORM_MIXTURE
    args.selection_seed = SELECTION_SEED
    args.algorithm_seed_base = ALGORITHM_SEED_BASE
    args.policy_seed_base = POLICY_SEED_BASE
    args.rf_seed_base = RF_SEED_BASE
    args.cv_seed = CV_SEED
    return args


def _formal_model_summary(trajectory_dir: Path, expected_instances: list[str],
                          rf_seed: int):
    summary_path = trajectory_dir / "model" / "training_summary.json"
    model_path = trajectory_dir / "model" / "rf_reward.joblib"
    if not summary_path.exists() or not model_path.exists():
        return None
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    validation_instances = sorted(
        row["instance_id"] for row in summary["best"]["validation"]["per_instance"])
    expected = sorted(expected_instances)
    checks = {
        "instance_count": summary.get("instance_count") == len(expected),
        "instances": validation_instances == expected,
        "rf_seed": summary.get("rf_seed") == rf_seed,
        "cv_seed": summary.get("cv_seed") == CV_SEED,
        "folds": summary.get("grouped_validation_folds") == FOLDS,
        "balanced": summary.get("configuration_balanced_folds") is True,
        "fixed_parameters": summary.get("fixed_parameters") == FIXED_PARAMETERS,
        "parameter_protocol": summary.get("parameter_protocol")
            == "predeclared fixed parameters; CV is diagnostic only",
        "model_hash": summary.get("model_sha256") == _sha256(model_path),
        "sample_weighting": summary.get("sample_weighting") == "none",
        "archive_reward_consistency": summary.get("maximum_archive_delta_error") == 0.0,
    }
    if not all(checks.values()):
        raise ValueError(
            f"formal model provenance mismatch at {trajectory_dir / 'model'}: {checks}")
    return summary


def _train_trajectory(trajectory: int, args, selected: list[dict]):
    trajectory_dir = args.out / f"trajectory_{trajectory + 1}"
    expected_instances = [row["instance_id"] for row in selected]
    rf_seed = RF_SEED_BASE + trajectory
    summary = _formal_model_summary(trajectory_dir, expected_instances, rf_seed)
    if summary is None:
        model_dir = trajectory_dir / "model"
        if model_dir.exists() and any(model_dir.iterdir()):
            raise RuntimeError(
                f"incomplete non-empty model directory requires inspection: {model_dir}")
        command = [
            sys.executable, "-u", str(TRAINER),
            "--runs", str(trajectory_dir / "runs"),
            "--out", str(model_dir),
            "--rf-seed", str(rf_seed),
            "--jobs", str(args.rf_jobs),
            "--folds", str(FOLDS),
            "--configuration-balanced-folds",
            "--cv-seed", str(CV_SEED),
            "--fixed-parameters-json", json.dumps(
                FIXED_PARAMETERS, sort_keys=True, separators=(",", ":")),
        ]
        log_path = trajectory_dir / "model_training.log"
        environment = os.environ.copy()
        environment.update({"PYTHONUNBUFFERED": "1"})
        with log_path.open("x", encoding="utf-8") as handle:
            result = subprocess.run(
                command, stdout=handle, stderr=subprocess.STDOUT,
                cwd=ROOT.parent, env=environment, check=False)
        if result.returncode != 0:
            raise RuntimeError(
                f"trajectory {trajectory + 1} model training failed; see {log_path}")
        summary = _formal_model_summary(trajectory_dir, expected_instances, rf_seed)
        if summary is None:
            raise RuntimeError(
                f"trajectory {trajectory + 1} model exited without valid artifacts")
    validation = summary["best"]["validation"]
    baseline = summary["training_fold_mean_baseline"]
    result = {
        "completed": True,
        "trajectory": trajectory + 1,
        "rf_seed": rf_seed,
        "sample_count": summary["sample_count"],
        "model_sha256": summary["model_sha256"],
        "training_data_digest": summary["training_data_digest"],
        "fixed_parameters": summary["fixed_parameters"],
        "outcome_counts": summary["outcome_counts"],
        "logging_propensity": summary["logging_propensity"],
        "configuration_reward": summary["configuration_reward"],
        "rf_mean_instance_mae": validation["mean_instance_mae"],
        "rf_mean_instance_rmse": validation["mean_instance_rmse"],
        "baseline_mean_instance_mae": baseline["mean_instance_mae"],
        "baseline_mean_instance_rmse": baseline["mean_instance_rmse"],
        "rf_beats_baseline_mae": (
            validation["mean_instance_mae"] < baseline["mean_instance_mae"]),
        "rf_beats_baseline_rmse": (
            validation["mean_instance_rmse"] < baseline["mean_instance_rmse"]),
        "maximum_archive_delta_error": summary["maximum_archive_delta_error"],
    }
    write_json(trajectory_dir / "FORMAL_TRAJECTORY_COMPLETE.json", result)
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
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--rf-jobs", type=int, default=4)
    parser.add_argument("--preflight-only", action="store_true")
    args = _apply_frozen_args(parser.parse_args())
    args.data = args.data.resolve()
    args.instances = args.instances.resolve()
    args.normalization = args.normalization.resolve()
    args.out = args.out.resolve()
    if args.workers < 1 or args.workers > 16:
        parser.error("workers must be between 1 and 16")
    if args.rf_jobs < 1 or args.rf_jobs > 16:
        parser.error("rf-jobs must be between 1 and 16")
    try:
        _, selected, source_data_sha256 = _validate_inputs(args)
    except (KeyError, OSError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))
    batch_config = _batch_configuration(args, selected, source_data_sha256)
    if args.preflight_only:
        print(json.dumps({key: batch_config[key] for key in (
            "stage", "catalog_digest", "normalization_digest",
            "instance_count_per_trajectory", "trajectories", "collection_count",
            "evaluation_budget_per_instance", "maximum_ccp100_evaluations",
            "selection_digest", "fixed_rf_parameters")}, sort_keys=True))
        return
    args.out.mkdir(parents=True, exist_ok=True)
    config_path = args.out / "formal_training_configuration.json"
    if config_path.exists():
        existing = json.loads(config_path.read_text(encoding="utf-8"))
        if existing != batch_config:
            parser.error("existing formal configuration differs; use a new output directory")
    else:
        write_json(config_path, batch_config)
    for trajectory in range(TRAJECTORIES):
        trajectory_dir = args.out / f"trajectory_{trajectory + 1}"
        (trajectory_dir / "runs").mkdir(parents=True, exist_ok=True)
        (trajectory_dir / "logs").mkdir(exist_ok=True)
    tasks = [(trajectory, index)
             for trajectory in range(TRAJECTORIES)
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
        if (not _completed(run_dir, selected[index], args, trajectory, index)
                and log_path.exists()):
            incomplete.append(str(log_path))
    if incomplete:
        raise RuntimeError(
            "incomplete outputs require inspection before resume:\n"
            + "\n".join(sorted(set(incomplete))))
    total = len(tasks)
    print(
        f"[RF-FORMAL-D] total={total} already_complete={already_complete} "
        f"workers={args.workers} budget={EVALUATION_BUDGET} "
        f"selection_digest={batch_config['selection_digest']}", flush=True)
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
        try:
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
                    f"[RF-FORMAL-D] {completed:04d}/{total} "
                    f"trajectory={trajectory + 1} {instance_id} {status} "
                    f"run_seconds={duration:.1f} eta_hours={remaining_h:.2f}",
                    flush=True)
        except BaseException:
            for pending_future in futures:
                pending_future.cancel()
            raise
    trajectory_results = [
        _train_trajectory(trajectory, args, selected)
        for trajectory in range(TRAJECTORIES)
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
        "stage": "RF-Reward formal training D",
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
        "formal_test_or_oos_used": False,
        **batch_config,
    }
    write_json(args.out / "FORMAL_TRAINING_COMPLETE.json", summary)
    print(json.dumps({key: summary[key] for key in (
        "completed", "actual_ccp100_evaluations",
        "rf_beats_baseline_rmse_count", "rf_beats_baseline_mae_count",
        "formal_test_or_oos_used")}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
