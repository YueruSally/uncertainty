#!/usr/bin/env python3
"""Re-evaluate frozen Base and one comparison front on independent CCP scenarios."""
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
from run_ccp_candidate_pool import decision_signature
from run_ev_ccp_oos_pilot import scenario_digest
from run_formal_ev_vs_ccp_s30_30runs import restore_individual
from transformer_ppo.archive import ObjectiveNormalizer, exact_hv_3d
from transformer_ppo.instance_catalog import read_instance


EPSILON = 1e-12
BASE_STRATEGY = "BASE"
BALANCED_STRATEGY = "REPLACE_G10_JOINT_BALANCED"
OBJECTIVES = ("cost", "emission", "makespan")
UNCERTAINTY_PROFILES = {
    "transport_border_saa_v1": {
        "mode_time_cv": {"road": 0.15, "rail": 0.10, "water": 0.20},
        "mode_time_max_factor": {
            "road": None, "rail": None, "water": None},
        "border_delay_cv": {"road": 0.50, "rail": 0.50, "water": 0.50},
        "border_delay_max_factor": 3.5,
        "border_mean_source": "node_border",
        "border_include_customs": False,
    },
}


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n",
        encoding="utf-8")


def _normalizer(path):
    value = _read(path)
    if value.get("source") != "training-only":
        raise ValueError("normalization must declare source=training-only")
    return ObjectiveNormalizer(
        tuple(value["minimum"]), tuple(value["maximum"]),
        tuple(value.get("reference", (1.1, 1.1, 1.1))))


def _uncertainty_profile(name, scenario_count):
    try:
        profile = dict(UNCERTAINTY_PROFILES[name])
    except KeyError as error:
        raise ValueError(f"unsupported uncertainty profile: {name}") from error
    profile["name"] = name
    profile["scenario_count"] = int(scenario_count)
    return profile


def discover_run_roots(primary_root):
    """Discover compatible roots recursively from aggregate result metadata."""
    queue = [Path(primary_root)]
    roots = []
    seen = set()
    while queue:
        root = queue.pop(0)
        key = str(root)
        if key in seen:
            continue
        seen.add(key)
        roots.append(root)
        aggregate = root / "COMPLETE.json"
        if not aggregate.is_file():
            continue
        try:
            value = _read(aggregate)
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            continue
        references = []
        references.extend(value.get("reuse_roots") or [])
        for field in (
                "reuse_root", "reuse_base_root", "positive_probe_root",
                "reversible_probe_root"):
            if value.get(field):
                references.append(value[field])
        queue.extend(Path(item) for item in references)
    return roots


def resolve_run(roots, instance, seed, strategy):
    # Aggregate validation files record the exact run used for every paired
    # comparison.  Prefer that provenance over directory discovery: an older
    # diagnostic root may contain another valid BASE run for the same
    # instance/seed but with a different final archive.
    for root in roots:
        aggregate = root / "COMPLETE.json"
        if not aggregate.is_file():
            continue
        try:
            value = _read(aggregate)
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            continue
        recorded = [
            Path(row["run_path"])
            for row in value.get("runs", ())
            if row.get("instance") == instance
            and int(row.get("algorithm_seed", -1)) == int(seed)
            and row.get("strategy") == strategy
            and row.get("run_path")
        ]
        valid_recorded = [
            run for run in recorded
            if ((run / "COMPLETE.json").is_file()
                and (run / "configuration.json").is_file()
                and (run / "final_feasible_nondominated.json").is_file())
        ]
        if valid_recorded:
            fingerprints = {
                hashlib.sha256(
                    (run / "final_feasible_nondominated.json").read_bytes()
                ).hexdigest()
                for run in valid_recorded
            }
            if len(fingerprints) != 1:
                raise RuntimeError(
                    f"conflicting recorded {strategy} runs for "
                    f"{instance}:seed_{seed}: {valid_recorded}")
            return valid_recorded[0]

    matches = []
    for root in roots:
        candidate = root / instance / f"seed_{seed}" / strategy
        if ((candidate / "COMPLETE.json").is_file()
                and (candidate / "configuration.json").is_file()
                and (candidate / "final_feasible_nondominated.json").is_file()):
            matches.append(candidate)
    if not matches:
        raise FileNotFoundError(
            f"missing {strategy} run for {instance}:seed_{seed}; "
            f"searched={[str(root) for root in roots]}")
    fingerprints = {
        hashlib.sha256(
            (run / "final_feasible_nondominated.json").read_bytes()
        ).hexdigest()
        for run in matches
    }
    if len(fingerprints) != 1:
        raise RuntimeError(
            f"conflicting reusable {strategy} runs for "
            f"{instance}:seed_{seed}: {matches}")
    return matches[0]


def _nondominated(points):
    points = [tuple(float(value) for value in point) for point in points]
    unique = sorted(set(points))
    return [
        point for point in unique
        if not any(
            all(other[i] <= point[i] for i in range(3))
            and any(other[i] < point[i] for i in range(3))
            for other in unique if other != point)
    ]


def derive_common_oos_reference(rows, original_reference):
    """Return one pooled OOS nadir-plus-margin reference for every pair."""
    points = []
    for row in rows:
        for method in row["methods"].values():
            points.extend(method["oos_normalized_nondominated_points"])
    values = np.asarray(points, dtype=float)
    if (values.ndim != 2 or values.shape[1] != 3
            or not np.all(np.isfinite(values))):
        raise ValueError("invalid pooled normalized OOS points")
    minima = values.min(axis=0)
    maxima = values.max(axis=0)
    margins = np.maximum(0.05, 0.10 * (maxima - minima))
    reference = np.maximum(
        np.asarray(original_reference, dtype=float), maxima + margins)
    if not np.all(values < reference):
        raise RuntimeError("derived OOS reference does not dominate all points")
    return {
        "rule": (
            "componentwise maximum of the original reference and the pooled "
            "Base+Balanced OOS nadir plus max(0.05, 10% pooled axis range)"),
        "original_reference": [float(value) for value in original_reference],
        "pooled_minimum": minima.tolist(),
        "pooled_maximum": maxima.tolist(),
        "margin": margins.tolist(),
        "reference": reference.tolist(),
        "pooled_point_count": int(len(values)),
    }


def _evaluate_front(source_path, instance, batches, network, scenarios,
                    active_border_delay, normalizer, alpha,
                    expected_uncertainty):
    (nodes, regions, hold, proc, trans_cost, arcs, timetables, _source_batches,
     wait_cost, wait_emission, carbon, emission_factors, speeds, trans_map,
     _border_delay, theta, _) = network
    del nodes, regions, emission_factors, speeds
    source = _read(source_path / "final_feasible_nondominated.json")
    config = _read(source_path / "configuration.json")
    if config["instance_digest"] != instance["instance_digest"]:
        raise ValueError(f"instance mismatch for {source_path}")
    source_uncertainty = config.get("uncertainty") or {}
    expected_fields = (
        "name", "mode_time_cv", "mode_time_max_factor",
        "border_delay_cv", "border_delay_max_factor",
        "border_mean_source", "border_include_customs")
    if (config.get("evaluation_mode") != "ccp"
            or abs(float(config.get("alpha", -1.0)) - alpha) > 1e-12
            or any(
                source_uncertainty.get(field)
                != expected_uncertainty.get(field)
                for field in expected_fields)):
        raise ValueError(
            f"source CCP configuration does not match OOS profile: "
            f"{source_path}")
    if not source:
        raise ValueError(f"empty final front: {source_path}")
    training_digest = config["scenario_digest"]
    if training_digest == scenario_digest(scenarios):
        raise ValueError("independent scenario digest equals training digest")

    tt = base.build_timetable_dict(timetables)
    lookup = base.build_arc_lookup(arcs)
    q90_points = []
    candidate_rows = []
    unchanged = 0
    for row in source:
        individual = restore_individual(row, {}, tt, lookup)
        before = decision_signature(individual)
        base.evaluate_individual(
            individual, batches, arcs, tt, wait_cost, wait_emission,
            node_hold_cost=hold, node_proc_cost=proc,
            carbon_tax_map=carbon, trans_map=trans_map,
            border_delay_map=active_border_delay,
            theta_rm=theta, node_trans_cost=trans_cost)
        after = decision_signature(individual)
        if before != after or before != row["decision_fingerprint"]:
            raise RuntimeError("independent evaluation changed a fixed decision")
        unchanged += 1
        arrays = (
            np.asarray(individual.cost_s, dtype=float),
            np.asarray(individual.emission_s, dtype=float),
            np.asarray(individual.makespan_s, dtype=float),
        )
        if any(
                values.shape != (scenarios.size,)
                or not np.all(np.isfinite(values))
                for values in arrays):
            raise ValueError("invalid independent objective array")
        point = tuple(
            float(base.empirical_ccp_quantile(values, alpha))
            for values in arrays)
        q90_points.append(point)
        candidate_rows.append({
            "decision_fingerprint": before,
            "q90": dict(zip(OBJECTIVES, point)),
        })

    front = _nondominated(q90_points)
    normalized = [normalizer.transform(point) for point in front]
    return {
        "run": str(source_path),
        "training_scenario_digest": training_digest,
        "source_front_size": len(source),
        "oos_nondominated_size": len(front),
        "unchanged_decisions": unchanged,
        "oos_q90_nondominated_points": [list(point) for point in front],
        "oos_normalized_nondominated_points": [
            list(point) for point in normalized],
        "candidates": candidate_rows,
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
     _emission_factors, _speeds, _trans_map, border_delay, _theta, _) = network
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
    comparison_strategy = task["comparison_strategy"]
    for seed in task["algorithm_seeds"]:
        target = out / instance_name / f"seed_{seed}.json"
        if target.is_file():
            value = _read(target)
            if (value.get("oos_metric_schema") == 2
                    and value.get("completed") is True
                    and value.get("oos_scenario_digest") == digest
                    and set(value.get("methods", ()))
                    == {BASE_STRATEGY, comparison_strategy}):
                rows.append(value)
                continue
        methods = {}
        for strategy in (BASE_STRATEGY, comparison_strategy):
            run = resolve_run(roots, instance_name, seed, strategy)
            methods[strategy] = _evaluate_front(
                run, instance, batches, network, scenarios,
                active_border_delay, instance_normalizer, task["alpha"],
                profile)
        result = {
            "oos_metric_schema": 2,
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


def _aggregate(rows, seeds):
    selected = [row for row in rows if row["algorithm_seed"] in seeds]
    values = [row["paired_oos_percent"] for row in selected]
    if not values:
        raise ValueError("cannot aggregate an empty OOS subset")
    mean = statistics.fmean(values)
    standard_deviation = statistics.stdev(values) if len(values) >= 2 else 0.0
    standard_error = standard_deviation / math.sqrt(len(values))
    per_instance = []
    for instance in sorted({row["instance"] for row in selected}):
        instance_values = [
            row["paired_oos_percent"] for row in selected
            if row["instance"] == instance]
        per_instance.append({
            "instance": instance,
            "runs": len(instance_values),
            "mean_paired_oos_percent": statistics.fmean(instance_values),
            "median_paired_oos_percent": statistics.median(instance_values),
            "minimum_paired_oos_percent": min(instance_values),
            "maximum_paired_oos_percent": max(instance_values),
            "wins": sum(value > EPSILON for value in instance_values),
            "ties": sum(abs(value) <= EPSILON for value in instance_values),
            "losses": sum(value < -EPSILON for value in instance_values),
        })
    return {
        "algorithm_seeds": sorted(seeds),
        "run_count": len(values),
        "mean_paired_oos_percent": mean,
        "standard_deviation_paired_oos_percent": standard_deviation,
        "standard_error_paired_oos_percent": standard_error,
        "mean_95ci_low": mean - 1.96 * standard_error,
        "mean_95ci_high": mean + 1.96 * standard_error,
        "median_paired_oos_percent": statistics.median(values),
        "minimum_paired_oos_percent": min(values),
        "maximum_paired_oos_percent": max(values),
        "wins": sum(value > EPSILON for value in values),
        "ties": sum(abs(value) <= EPSILON for value in values),
        "losses": sum(value < -EPSILON for value in values),
        "runs_above_5_percent": sum(value >= 5.0 for value in values),
        "runs_above_10_percent": sum(value >= 10.0 for value in values),
        "per_instance": per_instance,
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
    parser.add_argument("--holdout-seeds", nargs="+", type=int)
    parser.add_argument(
        "--comparison-strategy", default=BALANCED_STRATEGY,
        help="Run-directory strategy compared with BASE.")
    parser.add_argument("--scenario-count", type=int, default=1000)
    parser.add_argument("--oos-seed", type=int, default=880001)
    parser.add_argument("--alpha", type=float, default=0.90)
    parser.add_argument(
        "--uncertainty-profile", default="transport_border_saa_v1")
    parser.add_argument("--max-workers", type=int, default=5)
    args = parser.parse_args(argv)
    if not 0.0 < args.alpha <= 1.0:
        parser.error("alpha must be in (0, 1]")
    if args.scenario_count < 2 or args.max_workers < 1:
        parser.error("scenario-count must be >=2 and max-workers positive")
    if not args.validation_root.joinpath("COMPLETE.json").is_file():
        parser.error("validation root is incomplete")
    for path, name in (
            (args.data, "network data"),
            (args.normalization, "normalization")):
        if not path.is_file():
            parser.error(f"missing {name}: {path}")
    if len(set(args.algorithm_seeds)) != len(args.algorithm_seeds):
        parser.error("algorithm seeds must be unique")
    holdout_seeds = args.holdout_seeds or args.algorithm_seeds
    if not set(holdout_seeds).issubset(args.algorithm_seeds):
        parser.error("holdout seeds must be a subset of algorithm seeds")

    roots = discover_run_roots(args.validation_root)
    instance_paths = {}
    for instance_id in args.instance_ids:
        path = args.instances / f"{instance_id}.json"
        if not path.is_file():
            parser.error(f"missing instance: {path}")
        instance_paths[instance_id.replace("train-", "").replace("-", "_")] = path
    for instance in instance_paths:
        for seed in args.algorithm_seeds:
            for strategy in (BASE_STRATEGY, args.comparison_strategy):
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
        "comparison_strategy": args.comparison_strategy,
    } for instance, path in sorted(instance_paths.items())]

    results = []
    print(
        f"[OOS] instances={len(tasks)} pairs="
        f"{len(tasks) * len(args.algorithm_seeds)} "
        f"scenarios={args.scenario_count} workers={args.max_workers}",
        flush=True)
    with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        futures = [executor.submit(_evaluate_instance, task) for task in tasks]
        for index, future in enumerate(as_completed(futures), 1):
            result = future.result()
            results.extend(result["rows"])
            print(
                f"[OOS] {index}/{len(tasks)} instance complete: "
                f"{result['instance']} elapsed={result['elapsed_seconds']:.1f}s",
                flush=True)

    expected = len(instance_paths) * len(args.algorithm_seeds)
    if len(results) != expected:
        raise RuntimeError(f"expected {expected} paired results, got {len(results)}")
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
        base_hv = exact_hv_3d(
            row["methods"][BASE_STRATEGY][
                "oos_normalized_nondominated_points"],
            reference)
        comparison_hv = exact_hv_3d(
            row["methods"][args.comparison_strategy][
                "oos_normalized_nondominated_points"],
            reference)
        if base_hv <= 0.0:
            raise RuntimeError(
                "pooled OOS reference produced a nonpositive Base HV for "
                f"{row['instance']}:seed_{row['algorithm_seed']}")
        row["hv_reference"] = reference
        row["base_oos_hv"] = base_hv
        row["comparison_strategy"] = args.comparison_strategy
        row["comparison_oos_hv"] = comparison_hv
        # Retain the historical field for the frozen M release and old
        # downstream readers.  New component-specific runs should consume
        # comparison_oos_hv together with comparison_strategy.
        if args.comparison_strategy == BALANCED_STRATEGY:
            row["balanced_oos_hv"] = comparison_hv
        row["paired_oos_percent"] = 100.0 * (
            comparison_hv - base_hv) / base_hv
        _write(
            args.out / row["instance"]
            / f"seed_{row['algorithm_seed']}.json",
            row)

    result = {
        "completed": True,
        "definition": (
            "frozen Base and comparison final decisions re-evaluated "
            "without re-optimisation on one common independent CCP scenario set"),
        "comparison_strategy": args.comparison_strategy,
        "instance_count": len(instance_paths),
        "algorithm_seed_count": len(args.algorithm_seeds),
        "paired_run_count": len(results),
        "oos_scenario_count": args.scenario_count,
        "oos_seed": args.oos_seed,
        "oos_scenario_digest": next(iter(digests)),
        "training_scenario_digests": sorted(training_digests),
        "oos_distinct_from_training": not bool(digests & training_digests),
        "one_common_oos_scenario_set": len(digests) == 1,
        "uncertainty_profile": args.uncertainty_profile,
        "alpha": args.alpha,
        "hv_reference": hv_reference,
        "all_seeds": _aggregate(results, set(args.algorithm_seeds)),
        "held_out_seeds": _aggregate(results, set(holdout_seeds)),
        "runs": sorted(
            results, key=lambda row: (row["instance"], row["algorithm_seed"])),
    }
    held_out = result["held_out_seeds"]
    result["admission_thresholds"] = {
        "minimum_mean_paired_oos_percent": 10.0,
        "minimum_95ci_low": 0.0,
        "minimum_win_fraction": 0.80,
        "maximum_loss_fraction": 0.20,
    }
    thresholds = result["admission_thresholds"]
    result["admission_pass"] = bool(
        held_out["mean_paired_oos_percent"] >=
        thresholds["minimum_mean_paired_oos_percent"]
        and held_out["mean_95ci_low"] >= thresholds["minimum_95ci_low"]
        and held_out["wins"] / held_out["run_count"] >=
        thresholds["minimum_win_fraction"]
        and held_out["losses"] / held_out["run_count"] <=
        thresholds["maximum_loss_fraction"])
    _write(args.out / "OOS_COMPLETE.json", result)
    quick = {key: result[key] for key in (
        "completed", "definition", "instance_count", "algorithm_seed_count",
        "paired_run_count", "oos_scenario_count", "oos_seed",
        "oos_scenario_digest", "oos_distinct_from_training",
        "one_common_oos_scenario_set", "uncertainty_profile", "alpha",
        "comparison_strategy",
        "hv_reference", "all_seeds", "held_out_seeds", "admission_thresholds",
        "admission_pass")}
    _write(args.out / "QUICK_RESULT.json", quick)
    print(json.dumps(quick, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
