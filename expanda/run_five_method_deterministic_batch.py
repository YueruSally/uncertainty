#!/usr/bin/env python3
"""Run the frozen Base/I/C/M/I+C+M comparison on all ten test-S0 instances."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parent
SINGLE_RUNNER = ROOT / "run_five_method_deterministic.py"
METHODS = ("BASE", "I", "C", "M", "ICM")


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


def _git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
            stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _test_records(instances: Path, configuration: str):
    catalog_path = instances / "catalog.json"
    if not catalog_path.is_file():
        raise ValueError(f"catalog is missing: {catalog_path}")
    catalog = _read(catalog_path)
    records = [
        row for row in catalog.get("instances", [])
        if row.get("split") == "test"
        and row.get("configuration") == configuration
    ]
    records.sort(key=lambda row: row["instance_id"])
    if len(records) != 10:
        raise ValueError(
            f"formal {configuration} test requires exactly 10 test instances, "
            f"found {len(records)}")
    for record in records:
        path = instances / f"{record['instance_id']}.json"
        if not path.is_file():
            raise ValueError(f"test manifest is missing: {path}")
        manifest = _read(path)
        if (manifest.get("instance_digest") != record.get("instance_digest")
                or manifest.get("split") != "test"
                or manifest.get("configuration") != configuration):
            raise ValueError(f"catalog/manifest mismatch: {path}")
    return catalog, records


def _checkpoint_info(path: Path, component: str, args, normalization: dict,
                     catalog_digest):
    path = path.resolve()
    root = path.parent
    training_config_path = root / "training_configuration.json"
    training_summary_path = root / "training_summary.json"
    if not path.is_file() or not training_config_path.is_file() \
            or not training_summary_path.is_file():
        raise ValueError(f"incomplete {component} checkpoint provenance: {path}")
    config = _read(training_config_path)
    summary = _read(training_summary_path)
    checkpoint_sha = _sha256(path)
    expected_path = (root / summary.get("checkpoint", "")).resolve()
    episodes = summary.get("episodes", [])
    ordered_instances = config.get("ordered_instances", [])
    ordered_ids = [row.get("instance_id") for row in ordered_instances]
    if (expected_path != path
            or summary.get("completed") is not True
            or summary.get("component") != component
            or len(episodes) != 50
            or summary.get("total_actual_evaluations") != 50 * args.evaluation_budget
            or summary.get("checkpoint_sha256") != checkpoint_sha
            or config.get("component") != component
            or config.get("architecture") != "transformer"
            or config.get("policy_seed") != args.policy_seed
            or config.get("episode_count") != 50
            or config.get("evaluation_mode") != "deterministic"
            or config.get("configuration") != args.configuration
            or config.get("evaluation_budget_per_episode") != args.evaluation_budget
            or config.get("population") != args.pop
            or config.get("generations") != args.gens
            or config.get("normalization_digest")
            != normalization.get("normalization_digest")
            or config.get("normalization_sha256") != _sha256(args.normalization)
            or config.get("catalog_digest") != catalog_digest
            or config.get("source_data_sha256") != _sha256(args.data)
            or config.get("oos_used_for_training") is not False):
        raise ValueError(f"invalid frozen {component} checkpoint provenance: {path}")
    if (len(ordered_ids) != 50
            or any(not str(instance_id).startswith(
                f"train-{args.configuration}-") for instance_id in ordered_ids)
            or [row.get("instance") for row in episodes] != ordered_ids):
        raise ValueError(f"invalid training-instance order in {component}: {path}")
    if any(
            row.get("component") != component
            or row.get("real_objective_evaluation_count") != args.evaluation_budget
            or row.get("stop_reason") != "evaluation_budget"
            for row in episodes):
        raise ValueError(f"invalid episode summary in {component} training: {path}")
    return {
        "path": str(path),
        "sha256": checkpoint_sha,
        "component": component,
        "architecture": "transformer",
        "policy_seed": args.policy_seed,
        "training_episode_count": len(episodes),
        "ppo_updates": episodes[-1]["ppo_updates_total"],
        "ordered_instances_digest": hashlib.sha256(json.dumps(
            ordered_instances, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "training_configuration_sha256": _sha256(training_config_path),
        "training_summary_sha256": _sha256(training_summary_path),
    }


def _frozen_updates_are_zero(component, summary):
    updates = summary.get("ppo_updates")
    if component == "ICM":
        return updates == {"I": 0, "C": 0, "M": 0}
    return updates == 0


def _completed_instance(out: Path, record: dict, seed: int, args, checkpoints,
                        normalization):
    complete_path = out / "COMPLETE.json"
    if not complete_path.is_file():
        return False
    complete = _read(complete_path)
    summaries = complete.get("summaries", {})
    if (complete.get("completed") is not True
            or complete.get("methods") != list(METHODS)
            or complete.get("algorithm_seed") != seed
            or complete.get("evaluation_budget") != args.evaluation_budget
            or set(summaries) != set(METHODS)):
        raise ValueError(f"invalid five-method completion marker: {out}")
    for component in METHODS:
        summary = summaries[component]
        config_path = out / component / "configuration.json"
        front_path = out / component / "final_feasible_nondominated.json"
        if not config_path.is_file() or not front_path.is_file():
            raise ValueError(f"missing {component} result files: {out}")
        config = _read(config_path)
        hv = summary.get("final_hypervolume")
        if (summary.get("completed") is not True
                or summary.get("component") != component
                or summary.get("stop_reason") != "evaluation_budget"
                or summary.get("real_objective_evaluation_count")
                != args.evaluation_budget
                or summary.get("evaluation_budget") != args.evaluation_budget
                or not isinstance(hv, (int, float)) or not math.isfinite(hv)
                or summary.get("final_front_size", 0) < 1
                or not _frozen_updates_are_zero(component, summary)
                or config.get("evaluation_mode") != "deterministic"
                or config.get("instance_id") != record["instance_id"]
                or config.get("instance_digest") != record["instance_digest"]
                or config.get("population") != args.pop
                or config.get("generations") != args.gens
                or config.get("seeds", {}).get("algorithm") != seed
                or tuple(config.get("normalization", {}).get("minimum", ()))
                != tuple(normalization["minimum"])
                or tuple(config.get("normalization", {}).get("maximum", ()))
                != tuple(normalization["maximum"])
                or tuple(config.get("normalization", {}).get("reference", ()))
                != tuple(normalization["reference"])):
            raise ValueError(f"invalid {component} frozen result: {out}")
        if component in ("I", "C", "M"):
            provenance = config.get("policy_checkpoint") or {}
            if (config.get("policy_training") is not False
                    or provenance.get("checkpoint_sha256")
                    != checkpoints[component]["sha256"]):
                raise ValueError(f"wrong {component} checkpoint provenance: {out}")
        elif component == "ICM":
            provenance = config.get("policy_checkpoint") or {}
            if (config.get("policy_training") is not False
                    or set(provenance) != {"I", "C", "M"}
                    or any(provenance[name].get("checkpoint_sha256")
                           != checkpoints[name]["sha256"]
                           for name in ("I", "C", "M"))):
                raise ValueError(f"wrong ICM checkpoint provenance: {out}")
        elif config.get("policy_checkpoint") is not None:
            raise ValueError(f"Base unexpectedly records a checkpoint: {out}")
    return True


def _quarantine(path: Path, root: Path):
    if not path.exists() or not any(path.iterdir()):
        return
    destination_root = root / "interrupted_instances"
    destination_root.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    destination = destination_root / f"{path.name}_{stamp}"
    suffix = 1
    while destination.exists():
        destination = destination_root / f"{path.name}_{stamp}_{suffix}"
        suffix += 1
    shutil.move(str(path), str(destination))
    print(f"[RESUME] moved incomplete output to {destination}", flush=True)


def _run_one(instance_path: Path, out: Path, log: Path, seed: int, args):
    command = [
        sys.executable, "-u", str(SINGLE_RUNNER),
        "--data", str(args.data),
        "--instance", str(instance_path),
        "--normalization", str(args.normalization),
        "--i-checkpoint", str(args.i_checkpoint),
        "--c-checkpoint", str(args.c_checkpoint),
        "--m-checkpoint", str(args.m_checkpoint),
        "--out", str(out),
        "--algorithm-seed", str(seed),
        "--evaluation-budget", str(args.evaluation_budget),
        "--pop", str(args.pop), "--gens", str(args.gens),
    ]
    if args.device:
        command.extend(["--device", args.device])
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as handle:
        result = subprocess.run(
            command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT,
            check=False)
    if result.returncode:
        raise RuntimeError(
            f"five-method run failed with exit {result.returncode}; inspect {log}")


def _bootstrap_ci(values, seed, samples=20_000):
    data = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(data), size=(samples, len(data)))
    means = data[indices].mean(axis=1)
    return [float(x) for x in np.quantile(means, (0.025, 0.975))]


def _exact_sign_flip_p_value(values):
    data = np.asarray(values, dtype=float)
    observed = abs(float(data.sum()))
    tolerance = max(1e-15, observed * 1e-12)
    extreme = 0
    for signs in itertools.product((-1.0, 1.0), repeat=len(data)):
        if abs(float(np.dot(data, signs))) >= observed - tolerance:
            extreme += 1
    return extreme / (2 ** len(data))


def _holm(p_values):
    order = sorted(range(len(p_values)), key=lambda index: p_values[index])
    adjusted = [1.0] * len(p_values)
    running = 0.0
    count = len(p_values)
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (count - rank) * p_values[index]))
        adjusted[index] = running
    return adjusted


def _analysis(rows):
    by_method = {
        method: [row for row in rows if row["method"] == method]
        for method in METHODS
    }
    identifiers = [row["instance_id"] for row in by_method["BASE"]]
    if (len(identifiers) != 10 or len(set(identifiers)) != 10
            or any([row["instance_id"] for row in by_method[method]] != identifiers
                   for method in METHODS)):
        raise ValueError("analysis requires the same ordered ten instances per method")
    descriptive = {}
    for method, method_rows in by_method.items():
        values = np.asarray([row["final_hypervolume"] for row in method_rows])
        descriptive[method] = {
            "count": len(values), "mean_hv": float(values.mean()),
            "sample_sd_hv": float(values.std(ddof=1)),
            "median_hv": float(np.median(values)),
            "minimum_hv": float(values.min()), "maximum_hv": float(values.max()),
            "mean_front_size": float(np.mean(
                [row["final_front_size"] for row in method_rows])),
            "total_optimisation_seconds": float(sum(
                row["optimisation_seconds"] for row in method_rows)),
        }
    base = np.asarray([
        row["final_hypervolume"] for row in by_method["BASE"]], dtype=float)
    comparisons = []
    for index, method in enumerate(METHODS[1:]):
        values = np.asarray([
            row["final_hypervolume"] for row in by_method[method]], dtype=float)
        differences = values - base
        outcomes = Counter(
            "win" if value > 1e-12 else "loss" if value < -1e-12 else "tie"
            for value in differences)
        comparisons.append({
            "method": method, "control": "BASE",
            "instance_ids": identifiers,
            "paired_hv_differences": differences.tolist(),
            "mean_hv_difference": float(differences.mean()),
            "median_hv_difference": float(np.median(differences)),
            "mean_hv_difference_bootstrap_95ci": _bootstrap_ci(
                differences, 81_500_000 + index),
            "exact_paired_sign_flip_p_value": _exact_sign_flip_p_value(differences),
            "wins": outcomes["win"], "ties": outcomes["tie"],
            "losses": outcomes["loss"],
        })
    adjusted = _holm([
        row["exact_paired_sign_flip_p_value"] for row in comparisons])
    for row, value in zip(comparisons, adjusted):
        row["holm_adjusted_p_value"] = value
    return {
        "schema_version": 1,
        "pairing_unit": "held-out S0 test instance with a predeclared paired seed",
        "descriptive": descriptive,
        "paired_vs_base": comparisons,
        "interpretation_rule": (
            "Final conclusions use final_hypervolume, not reward_archive_hv. "
            "All four paired comparisons form one Holm-corrected family."),
    }


def _write_csv(path: Path, rows):
    fields = [
        "instance_id", "instance_digest", "algorithm_seed", "method",
        "final_hypervolume", "final_front_size", "delta_vs_base",
        "percent_delta_vs_base", "real_objective_evaluation_count",
        "optimisation_seconds", "inference_seconds",
    ]
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path,
                        default=ROOT / "data/data_expanded.xlsx")
    parser.add_argument("--instances", type=Path, required=True)
    parser.add_argument("--normalization", type=Path, required=True)
    parser.add_argument("--i-checkpoint", type=Path, required=True)
    parser.add_argument("--c-checkpoint", type=Path, required=True)
    parser.add_argument("--m-checkpoint", type=Path, required=True)
    parser.add_argument("--policy-seed", type=int, required=True)
    parser.add_argument("--configuration", choices=("S0",), default="S0")
    parser.add_argument("--algorithm-seed-base", type=int, default=81_000_000)
    parser.add_argument("--evaluation-budget", type=int, required=True)
    parser.add_argument("--pop", type=int, required=True)
    parser.add_argument("--gens", type=int, required=True)
    parser.add_argument("--device")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true",
                        help="Validate all frozen inputs without creating outputs.")
    args = parser.parse_args()
    args.data = args.data.resolve()
    args.instances = args.instances.resolve()
    args.normalization = args.normalization.resolve()
    args.i_checkpoint = args.i_checkpoint.resolve()
    args.c_checkpoint = args.c_checkpoint.resolve()
    args.m_checkpoint = args.m_checkpoint.resolve()
    args.out = args.out.resolve()
    for label, path in (
            ("network workbook", args.data),
            ("normalization", args.normalization),
            ("I checkpoint", args.i_checkpoint),
            ("C checkpoint", args.c_checkpoint),
            ("M checkpoint", args.m_checkpoint)):
        if not path.is_file():
            parser.error(f"{label} is missing: {path}")
    try:
        catalog, records = _test_records(args.instances, args.configuration)
        normalization = _read(args.normalization)
        if (normalization.get("source") != "training-only"
                or normalization.get("source_method")
                != "base-nsga2-deterministic"
                or normalization.get("configuration") != args.configuration
                or normalization.get("instance_count") != 50
                or normalization.get("test_instances_used") is not False
                or normalization.get("validation_instances_used") is not False):
            raise ValueError("normalization is not the frozen S0 training-only artifact")
        catalog_digest = catalog.get("catalog_digest")
        checkpoints = {
            "I": _checkpoint_info(
                args.i_checkpoint, "I", args, normalization, catalog_digest),
            "C": _checkpoint_info(
                args.c_checkpoint, "C", args, normalization, catalog_digest),
            "M": _checkpoint_info(
                args.m_checkpoint, "M", args, normalization, catalog_digest),
        }
        if len({row["ordered_instances_digest"]
                for row in checkpoints.values()}) != 1:
            raise ValueError("I, C, and M did not use the same training-instance order")
    except (KeyError, OSError, TypeError, ValueError,
            json.JSONDecodeError) as error:
        parser.error(str(error))

    batch_config = {
        "schema_version": 1,
        "stage": "paired five-method single-scene deterministic test",
        "formal": True,
        "configuration": args.configuration,
        "test_instance_count": len(records),
        "test_instances": [row["instance_id"] for row in records],
        "test_instances_used_for_training": False,
        "validation_set_used": False,
        "methods": list(METHODS),
        "evaluation_budget": args.evaluation_budget,
        "population": args.pop, "generations": args.gens,
        "algorithm_seed_base": args.algorithm_seed_base,
        "seed_formula": "algorithm_seed_base + sorted_test_index",
        "policy_seed": args.policy_seed,
        "catalog_digest": catalog_digest,
        "normalization_digest": normalization["normalization_digest"],
        "normalization_sha256": _sha256(args.normalization),
        "source_data_sha256": _sha256(args.data),
        "checkpoints": checkpoints,
        "git_commit": _git_commit(),
        "implementation_sha256": {
            path.name: _sha256(path) for path in (
                Path(__file__), SINGLE_RUNNER,
                ROOT / "run_learning_components_deterministic.py",
                ROOT / "transformer_ppo/episode.py",
                ROOT / "transformer_ppo/comparison_controllers.py",
            )
        },
    }
    if args.preflight_only:
        print(json.dumps({
            "preflight": "passed",
            "configuration": args.configuration,
            "test_instance_count": len(records),
            "methods": list(METHODS),
            "normalization_digest": normalization["normalization_digest"],
            "checkpoint_sha256": {
                name: row["sha256"] for name, row in checkpoints.items()
            },
            "common_training_order_digest": next(iter(
                checkpoints.values()))["ordered_instances_digest"],
        }, indent=2, allow_nan=False))
        return
    config_path = args.out / "batch_configuration.json"
    if config_path.exists():
        if _read(config_path) != batch_config:
            parser.error("existing batch configuration differs; use a new output directory")
    elif args.out.exists() and any(args.out.iterdir()):
        parser.error("non-empty output has no batch_configuration.json")
    else:
        args.out.mkdir(parents=True, exist_ok=True)
        _atomic_json(config_path, batch_config)

    started = time.perf_counter()
    already_complete = 0
    for index, record in enumerate(records):
        seed = args.algorithm_seed_base + index
        instance_out = args.out / "instances" / record["instance_id"]
        if _completed_instance(
                instance_out, record, seed, args, checkpoints, normalization):
            already_complete += 1
            print(
                f"[RESUME] {index + 1}/10 {record['instance_id']} already complete",
                flush=True)
            continue
        _quarantine(instance_out, args.out)
        print(f"[TEST] {index + 1}/10 {record['instance_id']} seed={seed}", flush=True)
        one_started = time.perf_counter()
        _run_one(
            args.instances / f"{record['instance_id']}.json", instance_out,
            args.out / "logs" / f"{record['instance_id']}.log", seed, args)
        if not _completed_instance(
                instance_out, record, seed, args, checkpoints, normalization):
            raise RuntimeError(f"instance exited without valid completion: {instance_out}")
        completed = index + 1
        elapsed = time.perf_counter() - started
        newly_completed = completed - already_complete
        eta_minutes = elapsed / max(1, newly_completed) * (10 - completed) / 60.0
        print(
            f"[TEST] {completed}/10 {record['instance_id']} "
            f"seconds={time.perf_counter() - one_started:.1f} "
            f"eta_minutes={eta_minutes:.1f}", flush=True)

    rows = []
    for index, record in enumerate(records):
        instance_out = args.out / "instances" / record["instance_id"]
        complete = _read(instance_out / "COMPLETE.json")
        base_hv = float(complete["summaries"]["BASE"]["final_hypervolume"])
        for method in METHODS:
            summary = complete["summaries"][method]
            hv = float(summary["final_hypervolume"])
            delta = hv - base_hv
            rows.append({
                "instance_id": record["instance_id"],
                "instance_digest": record["instance_digest"],
                "algorithm_seed": args.algorithm_seed_base + index,
                "method": method, "final_hypervolume": hv,
                "final_front_size": int(summary["final_front_size"]),
                "delta_vs_base": delta,
                "percent_delta_vs_base": (
                    100.0 * delta / base_hv if base_hv > 0.0 else None),
                "real_objective_evaluation_count": int(
                    summary["real_objective_evaluation_count"]),
                "optimisation_seconds": float(summary["optimisation_seconds"]),
                "inference_seconds": float(summary["inference_seconds"]),
            })
    rows_path = args.out / "results.json"
    csv_path = args.out / "results.csv"
    analysis_path = args.out / "analysis.json"
    _atomic_json(rows_path, rows)
    _write_csv(csv_path, rows)
    _atomic_json(analysis_path, _analysis(rows))
    complete = {
        "completed": True,
        "completed_instance_count": len(records),
        "method_instance_count": len(rows),
        "resumed_instance_count": already_complete,
        "new_instance_count": len(records) - already_complete,
        "wall_seconds": time.perf_counter() - started,
        "results_sha256": _sha256(rows_path),
        "results_csv_sha256": _sha256(csv_path),
        "analysis_sha256": _sha256(analysis_path),
        **batch_config,
    }
    _atomic_json(args.out / "BATCH_COMPLETE.json", complete)
    print(json.dumps({
        "completed": True, "completed_instance_count": len(records),
        "method_instance_count": len(rows),
        "wall_seconds": complete["wall_seconds"],
        "analysis": str(analysis_path),
    }, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
