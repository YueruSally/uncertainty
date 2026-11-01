#!/usr/bin/env python3
"""Validate formal shards and independently re-evaluate Base/I/M fronts."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import itertools
import json
import math
from pathlib import Path
import random
import statistics
import time

import baseline_uncertainty as base
from evaluate_m_directed_balanced_oos import (
    _evaluate_front,
    _normalizer,
    _uncertainty_profile,
    derive_common_oos_reference,
)
from im_ccp_formal_plan import (
    ALGORITHM_SEEDS,
    CCP_CONFIDENCE,
    EVALUATION_BUDGET,
    FINAL_INSTANCE_COUNT,
    METHODS,
    OOS_SCENARIO_COUNT,
    SHARD_COUNT,
    TASKS_PER_SHARD,
    TOTAL_TASK_COUNT,
    TRAINING_SCENARIO_COUNT,
    UNCERTAINTY_PROFILE,
    canonical_digest,
)
from run_ev_ccp_oos_pilot import scenario_digest
from transformer_ppo.archive import exact_hv_3d
from transformer_ppo.instance_catalog import read_instance


ROOT = Path(__file__).resolve().parent
EPSILON = 1e-12
COMPARISONS = ("I_DIRECTED_BALANCED", "M_DIRECTED_BALANCED")


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
    if (plan.get("formal") is not True
            or tuple(plan.get("methods", ())) != METHODS
            or plan.get("algorithm_seeds") != list(ALGORITHM_SEEDS)
            or plan.get("final_test_instances", {}).get("instance_count")
            != FINAL_INSTANCE_COUNT
            or len(plan.get("schedule", ())) != TOTAL_TASK_COUNT
            or plan.get("evaluation_budget") != EVALUATION_BUDGET
            or plan.get("training_scenario_count")
            != TRAINING_SCENARIO_COUNT
            or plan.get("oos_scenario_count") != OOS_SCENARIO_COUNT
            or plan.get("execution", {}).get("shard_count") != SHARD_COUNT
            or plan.get("execution", {}).get("tasks_per_shard")
            != TASKS_PER_SHARD):
        raise ValueError("plan is not the frozen formal CCP I/M design")
    for relative, expected in plan["implementation_sha256"].items():
        target = ROOT / relative
        if not target.is_file() or _sha256(target) != expected:
            raise ValueError(f"formal implementation changed: {target}")
    for target, expected in (
            (Path(plan["network"]["path"]), plan["network"]["sha256"]),
            (Path(plan["border_event_data"]["path"]),
             plan["border_event_data"]["sha256"]),
            (Path(plan["normalization"]["path"]),
             plan["normalization"]["sha256"]),
            (Path(plan["m_infrastructure_checkpoint"]["path"]),
             plan["m_infrastructure_checkpoint"]["sha256"]),
            (Path(plan["final_test_instances"]["catalog_path"]),
             plan["final_test_instances"]["catalog_sha256"])):
        if not target.is_file() or _sha256(target) != expected:
            raise ValueError(f"frozen artifact is missing or changed: {target}")
    return plan


def _load_results(formal_root, plan):
    rows = []
    for shard in range(SHARD_COUNT):
        root = formal_root / f"shard_{shard}"
        complete_path = root / "SHARD_COMPLETE.json"
        results_path = root / "results.json"
        if not complete_path.is_file() or not results_path.is_file():
            raise ValueError(f"formal shard {shard} is incomplete")
        complete = _read(complete_path)
        shard_rows = _read(results_path)
        if (complete.get("completed") is not True
                or complete.get("shard") != shard
                or complete.get("plan_digest") != plan["plan_digest"]
                or complete.get("schedule_digest") != plan["schedule_digest"]
                or complete.get("task_count") != TASKS_PER_SHARD
                or complete.get("results_sha256") != _sha256(results_path)
                or len(shard_rows) != TASKS_PER_SHARD):
            raise ValueError(f"invalid formal shard {shard}")
        rows.extend(shard_rows)
    schedule = {task["task_id"]: task for task in plan["schedule"]}
    actual = {row["task_id"]: row for row in rows}
    if len(actual) != TOTAL_TASK_COUNT or set(actual) != set(schedule):
        raise ValueError("formal results do not exactly cover the frozen schedule")
    for task_id, row in actual.items():
        task = schedule[task_id]
        if (row["instance_id"] != task["instance_id"]
                or row["algorithm_seed"] != task["algorithm_seed"]
                or row["method"] != task["method"]
                or row["real_objective_evaluation_count"]
                != EVALUATION_BUDGET):
            raise ValueError(f"formal result differs from plan: {task_id}")
    for instance in {
            row["instance_id"] for row in rows}:
        for seed in ALGORITHM_SEEDS:
            subset = [
                row for row in rows
                if row["instance_id"] == instance
                and row["algorithm_seed"] == seed]
            if (len(subset) != len(METHODS)
                    or len({row["training_scenario_digest"]
                            for row in subset}) != 1):
                raise ValueError(
                    f"training scenarios are not exactly paired: "
                    f"{instance}:seed_{seed}")
    return sorted(
        rows, key=lambda row: (
            row["instance_id"], row["algorithm_seed"], row["method"]))


def _evaluate_instance(task):
    started = time.perf_counter()
    instance_path = Path(task["instance_path"])
    data = Path(task["data"])
    target = Path(task["target"])
    instance, batches = read_instance(instance_path)
    if hashlib.sha256(data.read_bytes()).hexdigest() != instance[
            "source_data_sha256"]:
        raise ValueError("network workbook changed after instance generation")
    base.BORDER_EVENT_DEFINITIONS = base.load_border_event_definitions(
        base.DEFAULT_BORDER_EVENT_DATA_FILE)
    network = base.load_network_from_extended(str(data))
    (_nodes, _regions, _hold, _proc, _trans_cost, arcs, _timetables,
     _source_batches, _wait_cost, _wait_emission, _carbon,
     _emission_factors, _speeds, _trans_map, _border_delay, _theta, _) = network
    profile = _uncertainty_profile(
        UNCERTAINTY_PROFILE, OOS_SCENARIO_COUNT)
    active_border_delay = base.load_border_delay_map_from_workbook(
        data, include_customs=profile["border_include_customs"])
    oos_seed = int(instance["oos5000_seed"])
    scenarios = base.build_scenario_set(
        arcs, active_border_delay, OOS_SCENARIO_COUNT, oos_seed,
        stochastic=True,
        border_event_definitions=base.BORDER_EVENT_DEFINITIONS,
        mode_time_cv=profile["mode_time_cv"],
        mode_time_max_factor=profile["mode_time_max_factor"],
        border_delay_cv=profile["border_delay_cv"],
        border_delay_max_factor=profile["border_delay_max_factor"],
        border_mean_source=profile["border_mean_source"])
    digest = scenario_digest(scenarios)
    if target.is_file():
        cached = _read(target)
        if (cached.get("completed") is True
                and cached.get("instance") == instance["instance_id"]
                and cached.get("oos_scenario_count") == OOS_SCENARIO_COUNT
                and cached.get("oos_seed") == oos_seed
                and cached.get("oos_scenario_digest") == digest
                and len(cached.get("runs", ()))
                == len(METHODS) * len(ALGORITHM_SEEDS)):
            return cached
        raise ValueError(f"incompatible OOS cache exists: {target}")

    base.ACTIVE_SCENARIO_SET = scenarios
    base._PATH_SCENARIO_CACHE = {}
    base.RISK_METRIC = "ccp"
    base.CONFIDENCE_COST = base.CONFIDENCE_EMISSION = base.CONFIDENCE_TIME = \
        CCP_CONFIDENCE
    objective_divisor = (
        max(1.0, sum(float(batch.quantity) for batch in batches)),
        max(1.0, sum(float(batch.quantity) for batch in batches)),
        max(1.0, max(float(batch.LT) for batch in batches)
            - min(float(batch.ET) for batch in batches)),
    )
    normalizer = _normalizer(Path(task["normalization"])).with_divisor(
        objective_divisor)
    runs = []
    for source in task["runs"]:
        method = _evaluate_front(
            Path(source["run_path"]), instance, batches, network, scenarios,
            active_border_delay, normalizer, CCP_CONFIDENCE, profile)
        runs.append({
            "instance": instance["instance_id"],
            "algorithm_seed": source["algorithm_seed"],
            "method": source["method"],
            "source_run_path": source["run_path"],
            "training_scenario_digest": source["training_scenario_digest"],
            "oos": method,
        })
    result = {
        "completed": True,
        "instance": instance["instance_id"],
        "instance_digest": instance["instance_digest"],
        "oos_scenario_count": OOS_SCENARIO_COUNT,
        "oos_seed": oos_seed,
        "oos_scenario_digest": digest,
        "uncertainty_profile": profile,
        "alpha": CCP_CONFIDENCE,
        "normalization_divisor": list(objective_divisor),
        "elapsed_seconds": time.perf_counter() - started,
        "runs": runs,
    }
    _atomic_json(target, result)
    return result


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
    return float(ordered[lower] * (1 - fraction)
                 + ordered[upper] * fraction)


def _bootstrap_ci(values, seed, samples=20_000):
    rng = random.Random(seed)
    means = [
        _mean([values[rng.randrange(len(values))] for _ in values])
        for _ in range(samples)]
    return [_quantile(means, 0.025), _quantile(means, 0.975)]


def _sign_flip(values):
    observed = abs(sum(values))
    tolerance = max(1e-15, observed * 1e-12)
    extreme = sum(
        abs(sum(value * sign for value, sign in zip(values, signs)))
        >= observed - tolerance
        for signs in itertools.product((-1.0, 1.0), repeat=len(values)))
    return extreme / (2 ** len(values))


def _holm(values):
    order = sorted(range(len(values)), key=lambda index: values[index])
    adjusted = [1.0] * len(values)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(
            running, min(1.0, (len(values) - rank) * values[index]))
        adjusted[index] = running
    return adjusted


def _analyze_metric(rows, metric, instance_ids, bootstrap_seed):
    lookup = {}
    summaries = []
    for instance in instance_ids:
        for method in METHODS:
            values = [
                float(row[metric]) for row in rows
                if row["instance_id"] == instance
                and row["method"] == method]
            if len(values) != len(ALGORITHM_SEEDS):
                raise ValueError(
                    f"wrong replication count: {instance}:{method}:{metric}")
            value = _mean(values)
            lookup[(instance, method)] = value
            summaries.append({
                "instance": instance,
                "method": method,
                "seed_values": values,
                "mean_hv": value,
                "sample_sd_hv": float(statistics.stdev(values)),
            })
    comparisons = []
    for index, method in enumerate(COMPARISONS):
        differences = [
            lookup[(instance, method)] - lookup[(instance, "BASE")]
            for instance in instance_ids]
        percentages = [
            100.0 * difference / lookup[(instance, "BASE")]
            for instance, difference in zip(instance_ids, differences)]
        outcomes = Counter(
            "win" if value > EPSILON else
            "loss" if value < -EPSILON else "tie"
            for value in differences)
        comparisons.append({
            "method": method,
            "control": "BASE",
            "independent_instance_count": len(instance_ids),
            "paired_instance_mean_hv_differences": differences,
            "paired_instance_mean_percent_differences": percentages,
            "mean_hv_difference": _mean(differences),
            "mean_percent_difference": _mean(percentages),
            "median_percent_difference": float(statistics.median(percentages)),
            "minimum_percent_difference": min(percentages),
            "maximum_percent_difference": max(percentages),
            "bootstrap_95ci_hv_difference": _bootstrap_ci(
                differences, bootstrap_seed + index),
            "bootstrap_95ci_percent_difference": _bootstrap_ci(
                percentages, bootstrap_seed + 100 + index),
            "exact_sign_flip_p_value": _sign_flip(differences),
            "wins": outcomes["win"],
            "ties": outcomes["tie"],
            "losses": outcomes["loss"],
        })
    adjusted = _holm([
        row["exact_sign_flip_p_value"] for row in comparisons])
    for row, value in zip(comparisons, adjusted):
        row["holm_adjusted_p_value"] = value
    descriptive = {}
    for method in METHODS:
        values = [lookup[(instance, method)] for instance in instance_ids]
        descriptive[method] = {
            "mean_of_instance_mean_hv": _mean(values),
            "sample_sd_of_instance_mean_hv": float(statistics.stdev(values)),
            "median_of_instance_mean_hv": float(statistics.median(values)),
            "minimum_instance_mean_hv": min(values),
            "maximum_instance_mean_hv": max(values),
        }
    return {
        "metric": metric,
        "independent_unit": "previously unseen final-test instance",
        "descriptive": descriptive,
        "instance_method_summary": summaries,
        "paired_vs_base": comparisons,
    }


def _report(result):
    lines = [
        "# Formal frozen-CCP Base/I/M result",
        "",
        "The independent unit is one previously unseen final-test instance. "
        f"{len(ALGORITHM_SEEDS)} algorithm seeds are averaged within each "
        "instance before "
        "paired inference.",
        "",
        "## Protocol",
        "",
        f"- Population: {result['protocol']['population']}",
        f"- Generations: {result['protocol']['generations']}",
        f"- Real evaluations per run: "
        f"{result['protocol']['evaluation_budget']}",
        f"- Optimization scenarios: "
        f"{result['protocol']['training_scenario_count']}",
        f"- Independent OOS scenarios per instance: "
        f"{result['protocol']['oos_scenario_count']}",
        "",
    ]
    for label, key in (("Optimization", "optimization"),
                       ("Independent OOS", "oos")):
        lines.extend([
            f"## {label}", "",
            "| Method | Mean HV change | Median | 95% bootstrap CI | "
            "W-T-L | Holm p |",
            "|---|---:|---:|---:|:---:|---:|",
        ])
        for row in result[key]["paired_vs_base"]:
            low, high = row["bootstrap_95ci_percent_difference"]
            lines.append(
                f"| {row['method']} | "
                f"{row['mean_percent_difference']:+.3f}% | "
                f"{row['median_percent_difference']:+.3f}% | "
                f"[{low:+.3f}%, {high:+.3f}%] | "
                f"{row['wins']}-{row['ties']}-{row['losses']} | "
                f"{row['holm_adjusted_p_value']:.6f} |")
        lines.append("")
    lines.extend([
        "## Interpretation boundary", "",
        "These are deterministic directed I/M teacher rules, not evidence "
        "that PPO learned the same improvements.", "",
    ])
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--formal-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-workers", type=int, default=4)
    args = parser.parse_args(argv)
    if args.max_workers < 1:
        parser.error("max-workers must be positive")
    args.plan = args.plan.resolve()
    args.formal_root = args.formal_root.resolve()
    args.out = args.out.resolve()
    try:
        plan = _verify_plan(args.plan)
        optimization_rows = _load_results(args.formal_root, plan)
    except (KeyError, OSError, TypeError, ValueError,
            json.JSONDecodeError) as error:
        parser.error(str(error))
    args.out.mkdir(parents=True, exist_ok=True)
    instance_records = plan["final_test_instances"]["instances"]
    tasks = []
    for record in instance_records:
        instance_id = record["instance_id"]
        tasks.append({
            "instance_path": str(
                Path(plan["final_test_instances"]["directory"])
                / f"{instance_id}.json"),
            "data": plan["network"]["path"],
            "normalization": plan["normalization"]["path"],
            "target": str(args.out / "oos_raw" / f"{instance_id}.json"),
            "runs": [
                row for row in optimization_rows
                if row["instance_id"] == instance_id],
        })
    evaluated = []
    print(
        f"[FORMAL OOS] instances={len(tasks)} runs={len(optimization_rows)} "
        f"scenarios={OOS_SCENARIO_COUNT} workers={args.max_workers}",
        flush=True)
    with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        futures = [executor.submit(_evaluate_instance, task) for task in tasks]
        for index, future in enumerate(as_completed(futures), 1):
            value = future.result()
            evaluated.append(value)
            print(
                f"[FORMAL OOS] {index}/{len(tasks)} "
                f"{value['instance']} complete", flush=True)
    if len(evaluated) != FINAL_INSTANCE_COUNT:
        raise RuntimeError("independent OOS evaluation is incomplete")
    oos_runs = []
    for value in evaluated:
        if value["oos_scenario_count"] != OOS_SCENARIO_COUNT:
            raise ValueError("wrong independent scenario count")
        training = {
            row["training_scenario_digest"] for row in value["runs"]}
        if value["oos_scenario_digest"] in training:
            raise ValueError("OOS scenario set reuses a training scenario set")
        for row in value["runs"]:
            oos_runs.append({
                "instance_id": row["instance"],
                "algorithm_seed": row["algorithm_seed"],
                "method": row["method"],
                "methods": {row["method"]: row["oos"]},
                "oos_seed": value["oos_seed"],
                "oos_scenario_digest": value["oos_scenario_digest"],
            })
    reference_record = derive_common_oos_reference(
        oos_runs, _normalizer(Path(plan["normalization"]["path"])).reference)
    reference = reference_record["reference"]
    for row in oos_runs:
        points = row["methods"][row["method"]][
            "oos_normalized_nondominated_points"]
        row["oos_hv"] = exact_hv_3d(points, reference)
        if not math.isfinite(row["oos_hv"]):
            raise ValueError("non-finite OOS hypervolume")
    optimization_metric_rows = [
        {**row, "optimization_hv": row["final_hypervolume"]}
        for row in optimization_rows]
    instance_ids = [record["instance_id"] for record in instance_records]
    optimization = _analyze_metric(
        optimization_metric_rows, "optimization_hv", instance_ids,
        95_000_000)
    oos = _analyze_metric(oos_runs, "oos_hv", instance_ids, 96_000_000)
    runtime = {}
    for method in METHODS:
        values = [
            row["optimisation_seconds"] for row in optimization_rows
            if row["method"] == method]
        runtime[method] = {
            "run_count": len(values),
            "mean_seconds": _mean(values),
            "median_seconds": float(statistics.median(values)),
            "total_seconds": sum(values),
        }
    result = {
        "schema_version": 1,
        "completed": True,
        "plan_digest": plan["plan_digest"],
        "schedule_digest": plan["schedule_digest"],
        "claim_boundary": plan["claim_boundary"],
        "protocol": {
            "instance_count": FINAL_INSTANCE_COUNT,
            "algorithm_seed_count": len(ALGORITHM_SEEDS),
            "method_count": len(METHODS),
            "optimization_run_count": len(optimization_rows),
            "population": plan["population"],
            "generations": plan["generations"],
            "evaluation_budget": plan["evaluation_budget"],
            "training_scenario_count": TRAINING_SCENARIO_COUNT,
            "oos_scenario_count": OOS_SCENARIO_COUNT,
            "ccp_confidence": CCP_CONFIDENCE,
            "uncertainty_profile": UNCERTAINTY_PROFILE,
            "independent_unit": "previously unseen final-test instance",
            "seed_averaging": (
                f"{len(ALGORITHM_SEEDS)} algorithm seeds averaged within "
                "each instance"),
            "oos_pairing": (
                "one independent 5000-scenario set per instance, common to "
                f"Base, I, M and all {len(ALGORITHM_SEEDS)} algorithm seeds"),
        },
        "scenario_validation": {
            "training_exactly_paired_within_instance_seed": True,
            "oos_exactly_paired_within_instance": True,
            "oos_distinct_from_training": True,
            "oos_instance_seeds": {
                value["instance"]: value["oos_seed"] for value in evaluated},
            "oos_instance_digests": {
                value["instance"]: value["oos_scenario_digest"]
                for value in evaluated},
        },
        "oos_hv_reference": reference_record,
        "runtime": runtime,
        "optimization": optimization,
        "oos": oos,
    }
    _atomic_json(args.out / "FORMAL_RESULT.json", result)
    quick = {
        "completed": True,
        "plan_digest": result["plan_digest"],
        "claim_boundary": result["claim_boundary"],
        "protocol": result["protocol"],
        "runtime": runtime,
        "optimization_paired_vs_base": optimization["paired_vs_base"],
        "oos_paired_vs_base": oos["paired_vs_base"],
        "scenario_validation": result["scenario_validation"],
    }
    _atomic_json(args.out / "QUICK_RESULT.json", quick)
    optimization_by_method = {
        row["method"]: row for row in optimization["paired_vs_base"]}
    oos_by_method = {
        row["method"]: row for row in oos["paired_vs_base"]}
    average_result = {
        "completed": True,
        "definition": (
            "formal average HV changes versus exactly paired Base and "
            "average optimization runtimes"),
        "protocol": result["protocol"],
        "methods": [
            {
                "method": method,
                "mean_optimization_hv_change_percent": (
                    0.0 if method == "BASE" else
                    optimization_by_method[method][
                        "mean_percent_difference"]),
                "mean_independent_oos_hv_change_percent": (
                    0.0 if method == "BASE" else
                    oos_by_method[method]["mean_percent_difference"]),
                "mean_optimisation_seconds": runtime[method]["mean_seconds"],
            }
            for method in METHODS
        ],
        "scenario_validation": result["scenario_validation"],
    }
    _atomic_json(args.out / "AVERAGE_RESULT.json", average_result)
    (args.out / "REPORT.md").write_text(
        _report(result), encoding="utf-8")
    (args.out / "PIPELINE_COMPLETE").touch()
    print(json.dumps(quick, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
