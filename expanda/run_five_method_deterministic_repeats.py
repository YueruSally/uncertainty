#!/usr/bin/env python3
"""Complete and analyse ten frozen-policy deterministic S0 test repeats."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from run_five_method_deterministic_batch import (
    METHODS,
    _bootstrap_ci,
    _exact_sign_flip_p_value,
    _holm,
)


ROOT = Path(__file__).resolve().parent
BATCH_RUNNER = ROOT / "run_five_method_deterministic_batch.py"


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


def _write_csv(path: Path, rows: list[dict]):
    fields = [
        "repeat", "instance_id", "instance_digest", "algorithm_seed", "method",
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


def _seed_base(first_seed_base: int, repeat_index: int, stride: int):
    return first_seed_base + repeat_index * stride


def _validate_batch(path: Path, expected_seed_base: int, expected_repeat: int,
                    args, first_config: dict):
    complete_path = path / "BATCH_COMPLETE.json"
    results_path = path / "results.json"
    if not complete_path.is_file() or not results_path.is_file():
        return None
    complete = _read(complete_path)
    rows = _read(results_path)
    expected_pairs = {
        (f"test-{args.configuration}-{index:03d}", method)
        for index in range(1, 11) for method in METHODS
    }
    actual_pairs = {(row.get("instance_id"), row.get("method")) for row in rows}
    invariant_fields = (
        "configuration", "test_instances", "methods", "evaluation_budget",
        "population", "generations", "policy_seed", "catalog_digest",
        "normalization_digest", "normalization_sha256", "source_data_sha256",
        "checkpoints",
    )
    if (complete.get("completed") is not True
            or complete.get("completed_instance_count") != 10
            or complete.get("method_instance_count") != 50
            or complete.get("algorithm_seed_base") != expected_seed_base
            or len(rows) != 50 or actual_pairs != expected_pairs
            or _sha256(results_path) != complete.get("results_sha256")
            or any(complete.get(field) != first_config.get(field)
                   for field in invariant_fields)):
        raise ValueError(
            f"invalid repeat {expected_repeat} batch or frozen-input mismatch: {path}")
    for row in rows:
        instance_index = int(row["instance_id"].rsplit("-", 1)[1]) - 1
        if (row.get("algorithm_seed") != expected_seed_base + instance_index
                or row.get("real_objective_evaluation_count")
                != args.evaluation_budget
                or not isinstance(row.get("final_hypervolume"), (int, float))
                or not math.isfinite(row["final_hypervolume"])):
            raise ValueError(f"invalid result row in repeat {expected_repeat}: {row}")
    return [{"repeat": expected_repeat, **row} for row in rows]


def _run_repeat(repeat: int, seed_base: int, output: Path, log: Path, args):
    command = [
        sys.executable, "-u", str(BATCH_RUNNER),
        "--data", str(args.data),
        "--instances", str(args.instances),
        "--normalization", str(args.normalization),
        "--i-checkpoint", str(args.i_checkpoint),
        "--c-checkpoint", str(args.c_checkpoint),
        "--m-checkpoint", str(args.m_checkpoint),
        "--policy-seed", str(args.policy_seed),
        "--configuration", args.configuration,
        "--algorithm-seed-base", str(seed_base),
        "--evaluation-budget", str(args.evaluation_budget),
        "--pop", str(args.pop), "--gens", str(args.gens),
        "--out", str(output),
    ]
    if args.device:
        command.extend(["--device", args.device])
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as handle:
        process = subprocess.Popen(
            command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1)
        assert process.stdout is not None
        for line in process.stdout:
            handle.write(line)
            handle.flush()
            print(line, end="", flush=True)
        return_code = process.wait()
    if return_code:
        raise RuntimeError(
            f"repeat {repeat} failed with exit {return_code}; inspect {log}")


def _analysis(rows: list[dict], repeat_count: int):
    instance_ids = sorted({row["instance_id"] for row in rows})
    repeats = sorted({int(row["repeat"]) for row in rows})
    if (len(instance_ids) != 10 or repeats != list(range(1, repeat_count + 1))
            or len(rows) != 10 * repeat_count * len(METHODS)):
        raise ValueError("analysis requires a complete 10-instance repeated design")
    cells = {}
    for row in rows:
        key = (row["instance_id"], int(row["repeat"]), row["method"])
        if key in cells:
            raise ValueError(f"duplicate repeated result cell: {key}")
        cells[key] = row
    expected = {
        (instance_id, repeat, method)
        for instance_id in instance_ids
        for repeat in repeats
        for method in METHODS
    }
    if set(cells) != expected:
        raise ValueError("missing repeated result cells")

    instance_method = []
    for instance_id in instance_ids:
        for method in METHODS:
            method_rows = [cells[(instance_id, repeat, method)] for repeat in repeats]
            values = np.asarray(
                [row["final_hypervolume"] for row in method_rows], dtype=float)
            instance_method.append({
                "instance_id": instance_id,
                "method": method,
                "repeat_count": repeat_count,
                "mean_hv": float(values.mean()),
                "sample_sd_hv": float(values.std(ddof=1)),
                "median_hv": float(np.median(values)),
                "minimum_hv": float(values.min()),
                "maximum_hv": float(values.max()),
                "mean_front_size": float(np.mean(
                    [row["final_front_size"] for row in method_rows])),
            })

    descriptive = {}
    for method in METHODS:
        method_rows = [row for row in instance_method if row["method"] == method]
        values = np.asarray([row["mean_hv"] for row in method_rows], dtype=float)
        raw_rows = [row for row in rows if row["method"] == method]
        descriptive[method] = {
            "independent_instance_count": 10,
            "algorithm_repeats_per_instance": repeat_count,
            "run_count": len(raw_rows),
            "mean_of_instance_mean_hv": float(values.mean()),
            "sample_sd_of_instance_mean_hv": float(values.std(ddof=1)),
            "median_of_instance_mean_hv": float(np.median(values)),
            "minimum_instance_mean_hv": float(values.min()),
            "maximum_instance_mean_hv": float(values.max()),
            "mean_front_size_over_all_runs": float(np.mean(
                [row["final_front_size"] for row in raw_rows])),
            "total_optimisation_seconds": float(sum(
                row["optimisation_seconds"] for row in raw_rows)),
        }

    look_up = {
        (row["instance_id"], row["method"]): row for row in instance_method
    }
    comparisons = []
    for method_index, method in enumerate(METHODS[1:]):
        per_instance = []
        all_paired = []
        for instance_id in instance_ids:
            difference = (
                look_up[(instance_id, method)]["mean_hv"]
                - look_up[(instance_id, "BASE")]["mean_hv"])
            per_instance.append(difference)
            for repeat in repeats:
                all_paired.append(
                    cells[(instance_id, repeat, method)]["final_hypervolume"]
                    - cells[(instance_id, repeat, "BASE")]["final_hypervolume"])
        differences = np.asarray(per_instance, dtype=float)
        all_differences = np.asarray(all_paired, dtype=float)
        outcomes = Counter(
            "win" if value > 1e-12 else "loss" if value < -1e-12 else "tie"
            for value in differences)
        comparisons.append({
            "method": method,
            "control": "BASE",
            "independent_pairing_unit": (
                "held-out S0 test instance after averaging algorithm repeats"),
            "instance_ids": instance_ids,
            "paired_instance_mean_hv_differences": differences.tolist(),
            "mean_instance_hv_difference": float(differences.mean()),
            "median_instance_hv_difference": float(np.median(differences)),
            "mean_instance_hv_difference_bootstrap_95ci": _bootstrap_ci(
                differences, 82_500_000 + method_index),
            "exact_paired_sign_flip_p_value": _exact_sign_flip_p_value(differences),
            "wins": outcomes["win"], "ties": outcomes["tie"],
            "losses": outcomes["loss"],
            "descriptive_all_seed_instance_pair_mean_difference": float(
                all_differences.mean()),
            "descriptive_all_seed_instance_pair_sample_sd": float(
                all_differences.std(ddof=1)),
            "descriptive_seed_instance_pair_count": len(all_differences),
        })
    adjusted = _holm([
        row["exact_paired_sign_flip_p_value"] for row in comparisons])
    for row, value in zip(comparisons, adjusted):
        row["holm_adjusted_p_value"] = value
    return {
        "schema_version": 1,
        "independent_instance_count": 10,
        "algorithm_repeats_per_instance": repeat_count,
        "method_run_count": 10 * repeat_count,
        "descriptive": descriptive,
        "instance_method_summary": instance_method,
        "paired_vs_base": comparisons,
        "interpretation_rule": (
            "Algorithm repeats are averaged within each held-out instance. "
            "Inference uses ten independent instance-level paired differences; "
            "the 100 seed-instance cells are descriptive and are not treated as "
            "100 independent samples. All four comparisons form one "
            "Holm-corrected family."),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data/data_expanded.xlsx")
    parser.add_argument("--instances", type=Path, required=True)
    parser.add_argument("--normalization", type=Path, required=True)
    parser.add_argument("--i-checkpoint", type=Path, required=True)
    parser.add_argument("--c-checkpoint", type=Path, required=True)
    parser.add_argument("--m-checkpoint", type=Path, required=True)
    parser.add_argument("--policy-seed", type=int, required=True)
    parser.add_argument("--configuration", choices=("S0",), default="S0")
    parser.add_argument("--first-batch", type=Path, required=True)
    parser.add_argument("--first-algorithm-seed-base", type=int, default=81_000_000)
    parser.add_argument("--repeat-count", type=int, default=10)
    parser.add_argument("--seed-stride", type=int, default=1_000)
    parser.add_argument("--evaluation-budget", type=int, required=True)
    parser.add_argument("--pop", type=int, required=True)
    parser.add_argument("--gens", type=int, required=True)
    parser.add_argument("--device")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    for name in (
            "data", "instances", "normalization", "i_checkpoint", "c_checkpoint",
            "m_checkpoint", "first_batch", "out"):
        setattr(args, name, getattr(args, name).resolve())
    if args.repeat_count != 10:
        parser.error("the frozen formal design requires exactly 10 repeats")
    if args.seed_stride < 10:
        parser.error("seed stride must be at least 10 to prevent overlap")
    if args.out == args.first_batch or args.first_batch in args.out.parents:
        parser.error("multi-repeat output must not contain or overwrite first batch")
    for label, path in (
            ("network workbook", args.data), ("instances", args.instances),
            ("normalization", args.normalization), ("I checkpoint", args.i_checkpoint),
            ("C checkpoint", args.c_checkpoint), ("M checkpoint", args.m_checkpoint),
            ("first batch", args.first_batch)):
        if not path.exists():
            parser.error(f"{label} is missing: {path}")

    try:
        first_complete = _read(args.first_batch / "BATCH_COMPLETE.json")
        first_rows = _validate_batch(
            args.first_batch, args.first_algorithm_seed_base, 1, args,
            first_complete)
        if first_rows is None:
            raise ValueError("first batch is incomplete")
        catalog = _read(args.instances / "catalog.json")
        expected_checkpoints = first_complete["checkpoints"]
        if (first_complete.get("configuration") != args.configuration
                or first_complete.get("evaluation_budget") != args.evaluation_budget
                or first_complete.get("population") != args.pop
                or first_complete.get("generations") != args.gens
                or first_complete.get("policy_seed") != args.policy_seed
                or _sha256(args.data) != first_complete.get("source_data_sha256")
                or _sha256(args.normalization)
                != first_complete.get("normalization_sha256")
                or catalog.get("catalog_digest")
                != first_complete.get("catalog_digest")
                or _sha256(args.i_checkpoint)
                != expected_checkpoints["I"]["sha256"]
                or _sha256(args.c_checkpoint)
                != expected_checkpoints["C"]["sha256"]
                or _sha256(args.m_checkpoint)
                != expected_checkpoints["M"]["sha256"]):
            raise ValueError("command inputs do not match the frozen first batch")
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))

    config = {
        "schema_version": 1,
        "stage": "frozen five-method deterministic S0 algorithm-seed replication",
        "formal": True,
        "configuration": args.configuration,
        "independent_test_instance_count": 10,
        "algorithm_repeats_per_instance": args.repeat_count,
        "methods": list(METHODS),
        "evaluation_budget": args.evaluation_budget,
        "population": args.pop,
        "generations": args.gens,
        "policy_seed": args.policy_seed,
        "first_algorithm_seed_base": args.first_algorithm_seed_base,
        "seed_stride": args.seed_stride,
        "seed_base_formula": "first_algorithm_seed_base + (repeat-1) * seed_stride",
        "first_batch": str(args.first_batch),
        "first_batch_complete_sha256": _sha256(
            args.first_batch / "BATCH_COMPLETE.json"),
        "first_batch_results_sha256": _sha256(args.first_batch / "results.json"),
        "frozen_checkpoint_sha256": {
            name: first_complete["checkpoints"][name]["sha256"]
            for name in ("I", "C", "M")
        },
        "catalog_digest": first_complete["catalog_digest"],
        "normalization_digest": first_complete["normalization_digest"],
        "git_commit": _git_commit(),
        "implementation_sha256": {
            Path(__file__).name: _sha256(Path(__file__)),
            BATCH_RUNNER.name: _sha256(BATCH_RUNNER),
        },
        "inference_unit": (
            "ten held-out instances after averaging ten algorithm repeats"),
    }
    if args.preflight_only:
        print(json.dumps({
            "preflight": "passed",
            "completed_repeat_count": 1,
            "remaining_repeat_count": args.repeat_count - 1,
            "independent_test_instance_count": 10,
            "method_run_count_after_completion": (
                10 * args.repeat_count * len(METHODS)),
            "first_algorithm_seed_base": args.first_algorithm_seed_base,
            "last_algorithm_seed_base": _seed_base(
                args.first_algorithm_seed_base, args.repeat_count - 1,
                args.seed_stride),
            "frozen_checkpoint_sha256": config["frozen_checkpoint_sha256"],
        }, indent=2, allow_nan=False))
        return
    config_path = args.out / "replication_configuration.json"
    if config_path.exists():
        if _read(config_path) != config:
            parser.error("existing replication configuration differs; use a new output")
    elif args.out.exists() and any(args.out.iterdir()):
        parser.error("non-empty output has no replication_configuration.json")
    else:
        args.out.mkdir(parents=True, exist_ok=True)
        _atomic_json(config_path, config)

    all_rows = list(first_rows)
    started = time.perf_counter()
    newly_completed = 0
    for repeat_index in range(1, args.repeat_count):
        repeat = repeat_index + 1
        seed_base = _seed_base(
            args.first_algorithm_seed_base, repeat_index, args.seed_stride)
        repeat_out = args.out / "repeats" / f"repeat_{repeat:02d}"
        try:
            rows = _validate_batch(
                repeat_out, seed_base, repeat, args, first_complete)
        except (KeyError, OSError, TypeError, ValueError,
                json.JSONDecodeError) as error:
            parser.error(str(error))
        if rows is not None:
            print(f"[REPEAT] {repeat}/10 already complete", flush=True)
            all_rows.extend(rows)
            continue
        print(f"[REPEAT] {repeat}/10 seed_base={seed_base}", flush=True)
        one_started = time.perf_counter()
        _run_repeat(
            repeat, seed_base, repeat_out,
            args.out / "logs" / f"repeat_{repeat:02d}.log", args)
        rows = _validate_batch(
            repeat_out, seed_base, repeat, args, first_complete)
        if rows is None:
            raise RuntimeError(f"repeat {repeat} exited without valid completion")
        all_rows.extend(rows)
        newly_completed += 1
        elapsed = time.perf_counter() - started
        remaining = args.repeat_count - repeat
        eta_hours = elapsed / newly_completed * remaining / 3600.0
        print(
            f"[REPEAT] {repeat}/10 seconds={time.perf_counter() - one_started:.1f} "
            f"eta_hours={eta_hours:.2f}", flush=True)

    all_rows.sort(key=lambda row: (
        int(row["repeat"]), row["instance_id"], METHODS.index(row["method"])))
    results_path = args.out / "results.json"
    csv_path = args.out / "results.csv"
    analysis_path = args.out / "analysis.json"
    _atomic_json(results_path, all_rows)
    _write_csv(csv_path, all_rows)
    _atomic_json(analysis_path, _analysis(all_rows, args.repeat_count))
    complete = {
        "completed": True,
        "completed_repeat_count": args.repeat_count,
        "independent_instance_count": 10,
        "method_run_count": 10 * args.repeat_count * len(METHODS),
        "new_repeat_count": args.repeat_count - 1,
        "new_wall_seconds": time.perf_counter() - started,
        "results_sha256": _sha256(results_path),
        "results_csv_sha256": _sha256(csv_path),
        "analysis_sha256": _sha256(analysis_path),
        **config,
    }
    _atomic_json(args.out / "MULTISEED_COMPLETE.json", complete)
    print(json.dumps({
        "completed": True,
        "completed_repeat_count": args.repeat_count,
        "method_run_count": complete["method_run_count"],
        "new_wall_seconds": complete["new_wall_seconds"],
        "analysis": str(analysis_path),
    }, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
