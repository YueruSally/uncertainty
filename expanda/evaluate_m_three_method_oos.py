#!/usr/bin/env python3
"""Joint OOS re-evaluation of Base, heuristic M, and Transformer-PPO M."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import math
from pathlib import Path
import statistics
import time

import numpy as np

import baseline_uncertainty as base
from evaluate_m_directed_balanced_oos import (
    _evaluate_front,
    _normalizer,
    _read,
    _uncertainty_profile,
    _write,
    derive_common_oos_reference,
    discover_run_roots,
    resolve_run,
)
from run_ev_ccp_oos_pilot import scenario_digest
from transformer_ppo.archive import exact_hv_3d
from transformer_ppo.instance_catalog import read_instance


EPSILON = 1e-12
BASE = "BASE"
HEURISTIC = "HEURISTIC_BALANCED"
TRANSFORMER = "TRANSFORMER_PPO"
STRATEGIES = (BASE, HEURISTIC, TRANSFORMER)


def _summary(rows, key):
    values = [float(row[key]) for row in rows]
    if not values:
        raise ValueError(f"cannot summarize empty metric: {key}")
    mean = statistics.fmean(values)
    standard_deviation = statistics.stdev(values) if len(values) >= 2 else 0.0
    standard_error = standard_deviation / math.sqrt(len(values))
    per_instance = []
    for instance in sorted({row["instance"] for row in rows}):
        selected = [
            float(row[key]) for row in rows if row["instance"] == instance]
        per_instance.append({
            "instance": instance,
            "runs": len(selected),
            "mean": statistics.fmean(selected),
            "minimum": min(selected),
            "maximum": max(selected),
            "wins": sum(value > EPSILON for value in selected),
            "ties": sum(abs(value) <= EPSILON for value in selected),
            "losses": sum(value < -EPSILON for value in selected),
        })
    return {
        "metric": key,
        "run_count": len(values),
        "mean": mean,
        "standard_deviation": standard_deviation,
        "standard_error": standard_error,
        "mean_95ci_low": mean - 1.96 * standard_error,
        "mean_95ci_high": mean + 1.96 * standard_error,
        "median": statistics.median(values),
        "minimum": min(values),
        "maximum": max(values),
        "wins": sum(value > EPSILON for value in values),
        "ties": sum(abs(value) <= EPSILON for value in values),
        "losses": sum(value < -EPSILON for value in values),
        "per_instance": per_instance,
    }


def _evaluate_instance(task):
    started = time.perf_counter()
    instance_name = task["instance_name"]
    instance_path = Path(task["instance_path"])
    data = Path(task["data"])
    out = Path(task["out"])
    roots = [Path(path) for path in task["roots"]]
    normalizer = _normalizer(Path(task["normalization"]))
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
        task["uncertainty_profile"], task["scenario_count"])
    active_border_delay = base.load_border_delay_map_from_workbook(
        data, include_customs=profile["border_include_customs"])
    scenarios = base.build_scenario_set(
        arcs, active_border_delay, profile["scenario_count"],
        task["oos_seed"], stochastic=True,
        border_event_definitions=base.BORDER_EVENT_DEFINITIONS,
        mode_time_cv=profile["mode_time_cv"],
        mode_time_max_factor=profile["mode_time_max_factor"],
        border_delay_cv=profile["border_delay_cv"],
        border_delay_max_factor=profile["border_delay_max_factor"],
        border_mean_source=profile["border_mean_source"])
    digest = scenario_digest(scenarios)
    base.ACTIVE_SCENARIO_SET = scenarios
    base._PATH_SCENARIO_CACHE = {}
    base.RISK_METRIC = "ccp"
    base.CONFIDENCE_COST = base.CONFIDENCE_EMISSION = base.CONFIDENCE_TIME = \
        task["alpha"]

    objective_divisor = (
        max(1.0, sum(float(batch.quantity) for batch in batches)),
        max(1.0, sum(float(batch.quantity) for batch in batches)),
        max(1.0, max(float(batch.LT) for batch in batches)
            - min(float(batch.ET) for batch in batches)),
    )
    instance_normalizer = normalizer.with_divisor(objective_divisor)
    rows = []
    for seed in task["algorithm_seeds"]:
        target = out / "raw" / instance_name / f"seed_{seed}.json"
        if target.is_file():
            value = _read(target)
            if (value.get("oos_metric_schema") == 3
                    and value.get("completed") is True
                    and value.get("oos_scenario_digest") == digest
                    and set(value.get("methods", ())) == set(STRATEGIES)):
                rows.append(value)
                continue
        methods = {}
        for strategy in STRATEGIES:
            run = resolve_run(roots, instance_name, seed, strategy)
            methods[strategy] = _evaluate_front(
                run, instance, batches, network, scenarios,
                active_border_delay, instance_normalizer, task["alpha"],
                profile)
        result = {
            "oos_metric_schema": 3,
            "completed": True,
            "instance": instance_name,
            "algorithm_seed": seed,
            "oos_scenario_count": scenarios.size,
            "oos_seed": scenarios.seed,
            "oos_scenario_digest": digest,
            "uncertainty_profile": profile,
            "alpha": task["alpha"],
            "normalization_divisor": list(objective_divisor),
            "methods": methods,
        }
        _write(target, result)
        rows.append(result)
    return {
        "instance": instance_name,
        "rows": rows,
        "scenario_digest": digest,
        "elapsed_seconds": time.perf_counter() - started,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data", type=Path,
        default=Path(__file__).resolve().parent / "data/data_expanded.xlsx")
    parser.add_argument("--instances", type=Path, required=True)
    parser.add_argument("--normalization", type=Path, required=True)
    parser.add_argument("--validation-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--instance-ids", nargs="+", required=True)
    parser.add_argument("--algorithm-seeds", nargs="+", type=int, required=True)
    parser.add_argument("--scenario-count", type=int, default=1000)
    parser.add_argument("--oos-seed", type=int, default=77000000)
    parser.add_argument("--alpha", type=float, default=0.90)
    parser.add_argument(
        "--uncertainty-profile", default="transport_border_saa_v1")
    parser.add_argument("--max-workers", type=int, default=2)
    args = parser.parse_args(argv)
    if not 0.0 < args.alpha <= 1.0:
        parser.error("alpha must be in (0, 1]")
    if args.scenario_count < 2 or args.max_workers < 1:
        parser.error("scenario-count must be >=2 and max-workers positive")
    if not args.validation_root.joinpath("VALIDATION_PASS").is_file():
        parser.error("independent validation did not pass")
    for path, name in (
            (args.data, "network data"),
            (args.normalization, "normalization")):
        if not path.is_file():
            parser.error(f"missing {name}: {path}")
    if len(set(args.algorithm_seeds)) != len(args.algorithm_seeds):
        parser.error("algorithm seeds must be unique")

    roots = discover_run_roots(args.validation_root)
    instance_paths = {}
    for instance_id in args.instance_ids:
        path = args.instances / f"{instance_id}.json"
        if not path.is_file():
            parser.error(f"missing instance: {path}")
        normalized = instance_id.replace("train-", "").replace("-", "_")
        instance_paths[normalized] = path
    for instance in instance_paths:
        for seed in args.algorithm_seeds:
            for strategy in STRATEGIES:
                resolve_run(roots, instance, seed, strategy)

    args.out.mkdir(parents=True, exist_ok=True)
    tasks = [{
        "instance_name": instance,
        "instance_path": str(path),
        "data": str(args.data),
        "normalization": str(args.normalization),
        "roots": [str(root) for root in roots],
        "out": str(args.out),
        "algorithm_seeds": args.algorithm_seeds,
        "scenario_count": args.scenario_count,
        "oos_seed": args.oos_seed,
        "alpha": args.alpha,
        "uncertainty_profile": args.uncertainty_profile,
    } for instance, path in sorted(instance_paths.items())]

    results = []
    print(
        f"[M-THREE-OOS] instances={len(tasks)} triples="
        f"{len(tasks) * len(args.algorithm_seeds)} "
        f"scenarios={args.scenario_count} workers={args.max_workers}",
        flush=True)
    with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        futures = [executor.submit(_evaluate_instance, task) for task in tasks]
        for index, future in enumerate(as_completed(futures), 1):
            result = future.result()
            results.extend(result["rows"])
            print(
                f"[M-THREE-OOS] {index}/{len(tasks)} instance complete: "
                f"{result['instance']} elapsed={result['elapsed_seconds']:.1f}s",
                flush=True)

    expected = len(instance_paths) * len(args.algorithm_seeds)
    if len(results) != expected:
        raise RuntimeError(f"expected {expected} triples, got {len(results)}")
    digests = {row["oos_scenario_digest"] for row in results}
    if len(digests) != 1:
        raise RuntimeError(f"independent scenario digests differ: {digests}")
    training_digests = {
        method["training_scenario_digest"]
        for row in results for method in row["methods"].values()}
    if digests & training_digests:
        raise RuntimeError("independent scenario set reuses a training digest")

    original_reference = _normalizer(args.normalization).reference
    hv_reference = derive_common_oos_reference(results, original_reference)
    reference = hv_reference["reference"]
    for row in results:
        hypervolumes = {
            strategy: exact_hv_3d(
                row["methods"][strategy][
                    "oos_normalized_nondominated_points"],
                reference)
            for strategy in STRATEGIES
        }
        if hypervolumes[BASE] <= 0.0 or hypervolumes[HEURISTIC] <= 0.0:
            raise RuntimeError(
                "pooled OOS reference produced a nonpositive comparison HV "
                f"for {row['instance']}:seed_{row['algorithm_seed']}")
        row["hv_reference"] = reference
        row["oos_hypervolumes"] = hypervolumes
        row["heuristic_vs_base_percent"] = 100.0 * (
            hypervolumes[HEURISTIC] - hypervolumes[BASE]) / hypervolumes[BASE]
        row["transformer_vs_base_percent"] = 100.0 * (
            hypervolumes[TRANSFORMER] - hypervolumes[BASE]) / hypervolumes[BASE]
        row["transformer_minus_heuristic_percentage_points"] = (
            row["transformer_vs_base_percent"]
            - row["heuristic_vs_base_percent"])
        row["transformer_vs_heuristic_percent"] = 100.0 * (
            hypervolumes[TRANSFORMER] - hypervolumes[HEURISTIC]
        ) / hypervolumes[HEURISTIC]
        _write(
            args.out / "runs" / row["instance"]
            / f"seed_{row['algorithm_seed']}.json",
            row)

    result = {
        "completed": True,
        "definition": (
            "frozen Base, heuristic, and Transformer-PPO final decisions "
            "jointly re-evaluated without re-optimisation on one independent "
            "CCP scenario set and one pooled three-method HV reference"),
        "instance_count": len(instance_paths),
        "algorithm_seed_count": len(args.algorithm_seeds),
        "paired_run_count": len(results),
        "oos_scenario_count": args.scenario_count,
        "oos_seed": args.oos_seed,
        "oos_scenario_digest": next(iter(digests)),
        "training_scenario_digests": sorted(training_digests),
        "oos_distinct_from_training": not bool(digests & training_digests),
        "one_common_oos_scenario_set": len(digests) == 1,
        "one_common_three_method_hv_reference": True,
        "uncertainty_profile": args.uncertainty_profile,
        "alpha": args.alpha,
        "hv_reference": hv_reference,
        "heuristic_vs_base": _summary(results, "heuristic_vs_base_percent"),
        "transformer_vs_base": _summary(
            results, "transformer_vs_base_percent"),
        "transformer_minus_heuristic_percentage_points": _summary(
            results, "transformer_minus_heuristic_percentage_points"),
        "transformer_vs_heuristic_percent": _summary(
            results, "transformer_vs_heuristic_percent"),
        "runs": sorted(
            results, key=lambda row: (row["instance"], row["algorithm_seed"])),
    }
    model = result["transformer_vs_base"]
    direct = result["transformer_minus_heuristic_percentage_points"]
    result["admission_thresholds"] = {
        "minimum_transformer_vs_base_95ci_low": 0.0,
        "minimum_transformer_minus_heuristic_mean_points": 0.0,
        "minimum_transformer_minus_heuristic_wins": math.ceil(
            0.5 * len(results)),
    }
    result["admission_pass"] = bool(
        model["mean_95ci_low"] > 0.0
        and direct["mean"] > 0.0
        and direct["wins"] >= math.ceil(0.5 * len(results)))
    _write(args.out / "OOS_COMPLETE.json", result)
    quick_keys = (
        "completed", "definition", "instance_count", "algorithm_seed_count",
        "paired_run_count", "oos_scenario_count", "oos_seed",
        "oos_scenario_digest", "oos_distinct_from_training",
        "one_common_oos_scenario_set",
        "one_common_three_method_hv_reference", "uncertainty_profile", "alpha",
        "hv_reference", "heuristic_vs_base", "transformer_vs_base",
        "transformer_minus_heuristic_percentage_points",
        "transformer_vs_heuristic_percent", "admission_thresholds",
        "admission_pass")
    quick = {key: result[key] for key in quick_keys}
    _write(args.out / "QUICK_RESULT.json", quick)
    print(json.dumps(quick, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
