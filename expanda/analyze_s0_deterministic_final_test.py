#!/usr/bin/env python3
"""Analyze the complete frozen S0 deterministic final-test design."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
import random
import statistics

from s0_deterministic_test_plan import canonical_digest


METHODS = ("BASE", "RULE-I", "I", "C", "M", "ICM")
PRIMARY = ("I", "C", "M", "ICM")


def _read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _mean(values):
    return float(statistics.fmean(values))


def _quantile(values, probability):
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return float(ordered[lower])
    fraction = position - lower
    return float(ordered[lower] * (1 - fraction) + ordered[upper] * fraction)


def _bootstrap_ci(values, seed, samples=20_000):
    rng = random.Random(seed)
    means = [
        _mean([values[rng.randrange(len(values))] for _ in values])
        for _ in range(samples)
    ]
    return [_quantile(means, 0.025), _quantile(means, 0.975)]


def _sign_flip(values):
    observed = abs(sum(values))
    tolerance = max(1e-15, observed * 1e-12)
    extreme = sum(
        abs(sum(value * sign for value, sign in zip(values, signs)))
        >= observed - tolerance
        for signs in itertools.product((-1.0, 1.0), repeat=len(values))
    )
    return extreme / (2 ** len(values))


def _holm(values):
    order = sorted(range(len(values)), key=lambda index: values[index])
    result = [1.0] * len(values)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (len(values) - rank) * values[index]))
        result[index] = running
    return result


def _load_shard(root, shard, plan):
    complete_path = root / "SHARD_COMPLETE.json"
    results_path = root / "results.json"
    complete = _read(complete_path)
    rows = _read(results_path)
    if (complete.get("completed") is not True
            or complete.get("shard") != shard
            or complete.get("task_count") != 210
            or complete.get("plan_digest") != plan["plan_digest"]
            or complete.get("schedule_digest") != plan["schedule_digest"]
            or complete.get("results_sha256") != _sha256(results_path)
            or len(rows) != 210
            or any(row.get("execution_shard") != shard for row in rows)):
        raise ValueError(f"invalid or incomplete execution shard {shard}")
    return rows


def analyze(rows, plan):
    schedule = {row["task_id"]: row for row in plan["schedule"]}
    actual = {row["task_id"]: row for row in rows}
    if len(actual) != 420 or set(actual) != set(schedule):
        raise ValueError("results do not exactly cover the frozen schedule")
    for task_id, row in actual.items():
        task = schedule[task_id]
        if (row["instance_id"] != task["instance_id"]
                or row["algorithm_repeat"] != task["algorithm_repeat"]
                or row["algorithm_seed"] != task["algorithm_seed"]
                or row["method"] != task["method"]
                or row["policy_seed"] != task["policy_seed"]
                or row["real_objective_evaluation_count"] != 30_000
                or not math.isfinite(row["final_hypervolume"])):
            raise ValueError(f"result differs from frozen task: {task_id}")

    instance_ids = [
        row["instance_id"]
        for row in plan["final_test_instances"]["instances"]
    ]
    instance_method = []
    for instance_id in instance_ids:
        for method in METHODS:
            subset = [
                row for row in rows
                if row["instance_id"] == instance_id and row["method"] == method
            ]
            expected = 3 if method in ("BASE", "RULE-I") else 9
            if len(subset) != expected:
                raise ValueError(f"wrong replication count: {instance_id} {method}")
            values = [row["final_hypervolume"] for row in subset]
            instance_method.append({
                "instance_id": instance_id,
                "method": method,
                "run_count": len(subset),
                "mean_hv": _mean(values),
                "sample_sd_hv": float(statistics.stdev(values)),
                "minimum_hv": min(values),
                "maximum_hv": max(values),
                "mean_front_size": _mean(
                    [row["final_front_size"] for row in subset]),
            })
    lookup = {
        (row["instance_id"], row["method"]): row
        for row in instance_method
    }
    descriptive = {}
    for method in METHODS:
        aggregate = [row for row in instance_method if row["method"] == method]
        values = [row["mean_hv"] for row in aggregate]
        raw = [row for row in rows if row["method"] == method]
        descriptive[method] = {
            "independent_instance_count": 10,
            "run_count": len(raw),
            "mean_of_instance_mean_hv": _mean(values),
            "sample_sd_of_instance_mean_hv": float(statistics.stdev(values)),
            "median_of_instance_mean_hv": float(statistics.median(values)),
            "minimum_instance_mean_hv": min(values),
            "maximum_instance_mean_hv": max(values),
            "total_optimisation_seconds": sum(
                row["optimisation_seconds"] for row in raw),
            "mean_optimisation_seconds": _mean(
                [row["optimisation_seconds"] for row in raw]),
        }

    comparisons = []
    for index, method in enumerate(PRIMARY):
        differences = [
            lookup[(instance, method)]["mean_hv"]
            - lookup[(instance, "BASE")]["mean_hv"]
            for instance in instance_ids
        ]
        outcomes = Counter(
            "win" if value > 1e-12 else "loss" if value < -1e-12 else "tie"
            for value in differences)
        comparisons.append({
            "method": method,
            "control": "BASE",
            "instance_ids": instance_ids,
            "paired_instance_mean_hv_differences": differences,
            "mean_hv_difference": _mean(differences),
            "mean_percent_difference": _mean([
                100.0 * difference / lookup[(instance, "BASE")]["mean_hv"]
                for instance, difference in zip(instance_ids, differences)
            ]),
            "bootstrap_95ci": _bootstrap_ci(
                differences, 94_000_000 + index),
            "exact_sign_flip_p_value": _sign_flip(differences),
            "wins": outcomes["win"],
            "ties": outcomes["tie"],
            "losses": outcomes["loss"],
        })
    adjusted = _holm([row["exact_sign_flip_p_value"] for row in comparisons])
    for row, value in zip(comparisons, adjusted):
        row["holm_adjusted_p_value"] = value

    rule_differences = [
        lookup[(instance, "I")]["mean_hv"]
        - lookup[(instance, "RULE-I")]["mean_hv"]
        for instance in instance_ids
    ]
    rule_outcomes = Counter(
        "win" if value > 1e-12 else "loss" if value < -1e-12 else "tie"
        for value in rule_differences)
    secondary = {
        "method": "I",
        "control": "RULE-I",
        "paired_instance_mean_hv_differences": rule_differences,
        "mean_hv_difference": _mean(rule_differences),
        "bootstrap_95ci": _bootstrap_ci(rule_differences, 94_000_100),
        "exact_sign_flip_p_value": _sign_flip(rule_differences),
        "wins": rule_outcomes["win"],
        "ties": rule_outcomes["tie"],
        "losses": rule_outcomes["loss"],
        "multiplicity_role": "predeclared secondary contrast; not in primary family",
    }
    return {
        "schema_version": 1,
        "plan_digest": plan["plan_digest"],
        "independent_unit": "final-test instance",
        "independent_instance_count": 10,
        "raw_run_count": 420,
        "descriptive": descriptive,
        "instance_method_summary": instance_method,
        "primary_paired_vs_base": comparisons,
        "secondary_i_vs_rule_i": secondary,
        "interpretation_rule": (
            "Average policy and algorithm seeds within each final-test instance; "
            "perform inference on ten paired instance-level differences only."),
    }


def _write_csv(path, rows):
    fields = [
        "task_id", "execution_shard", "instance_id", "instance_digest",
        "algorithm_repeat", "algorithm_seed", "method", "policy_seed",
        "final_hypervolume", "final_front_size",
        "real_objective_evaluation_count", "optimisation_seconds",
        "objective_evaluation_seconds", "feature_construction_seconds",
        "inference_seconds", "ppo_update_seconds", "other_optimisation_seconds",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _report(analysis):
    lines = [
        "# S0 deterministic final-test analysis",
        "",
        "Inference uses ten previously unseen final-test instances. Policy and "
        "algorithm seeds are averaged within each instance.",
        "",
        "## Descriptive results",
        "",
        "| Method | Runs | Mean instance HV | SD | Mean runtime (min) |",
        "|---|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        row = analysis["descriptive"][method]
        lines.append(
            f"| {method} | {row['run_count']} | "
            f"{row['mean_of_instance_mean_hv']:.6f} | "
            f"{row['sample_sd_of_instance_mean_hv']:.6f} | "
            f"{row['mean_optimisation_seconds'] / 60:.2f} |")
    lines.extend([
        "", "## Primary comparisons with Base", "",
        "| Method | Mean delta | Mean delta % | 95% bootstrap CI | W-T-L | "
        "Raw p | Holm p |",
        "|---|---:|---:|---:|:---:|---:|---:|",
    ])
    for row in analysis["primary_paired_vs_base"]:
        low, high = row["bootstrap_95ci"]
        lines.append(
            f"| {row['method']} | {row['mean_hv_difference']:+.6f} | "
            f"{row['mean_percent_difference']:+.3f}% | "
            f"[{low:+.6f}, {high:+.6f}] | "
            f"{row['wins']}-{row['ties']}-{row['losses']} | "
            f"{row['exact_sign_flip_p_value']:.6f} | "
            f"{row['holm_adjusted_p_value']:.6f} |")
    row = analysis["secondary_i_vs_rule_i"]
    low, high = row["bootstrap_95ci"]
    lines.extend([
        "", "## Predeclared secondary comparison", "",
        "| Contrast | Mean delta | 95% bootstrap CI | W-T-L | Raw p |",
        "|---|---:|---:|:---:|---:|",
        f"| I vs RULE-I | {row['mean_hv_difference']:+.6f} | "
        f"[{low:+.6f}, {high:+.6f}] | "
        f"{row['wins']}-{row['ties']}-{row['losses']} | "
        f"{row['exact_sign_flip_p_value']:.6f} |",
        "", "The secondary contrast is not part of the four-comparison Holm family.",
        "",
    ])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--shard-0", type=Path, required=True)
    parser.add_argument("--shard-1", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        parser.error(f"output directory must be empty: {args.out}")
    try:
        plan = _read(args.plan)
        payload = dict(plan)
        digest = payload.pop("plan_digest", None)
        if canonical_digest(payload) != digest:
            raise ValueError("final-test plan digest mismatch")
        rows = _load_shard(args.shard_0, 0, plan)
        rows.extend(_load_shard(args.shard_1, 1, plan))
        result = analyze(rows, plan)
    except (KeyError, OSError, TypeError, ValueError,
            json.JSONDecodeError) as error:
        parser.error(str(error))
    args.out.mkdir(parents=True, exist_ok=True)
    rows.sort(key=lambda row: row["task_id"])
    _atomic_json(args.out / "raw_results.json", rows)
    _write_csv(args.out / "raw_results.csv", rows)
    _atomic_json(args.out / "analysis.json", result)
    (args.out / "analysis.md").write_text(_report(result), encoding="utf-8")
    complete = {
        "completed": True,
        "plan_digest": plan["plan_digest"],
        "raw_run_count": 420,
        "raw_results_sha256": _sha256(args.out / "raw_results.json"),
        "raw_results_csv_sha256": _sha256(args.out / "raw_results.csv"),
        "analysis_sha256": _sha256(args.out / "analysis.json"),
        "report_sha256": _sha256(args.out / "analysis.md"),
    }
    _atomic_json(args.out / "COMPLETE.json", complete)
    print(json.dumps({
        **complete,
        "report": str(args.out / "analysis.md"),
    }, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

