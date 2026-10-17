#!/usr/bin/env python3
"""Additive five-method OOS-HV analysis after all three RF blocks complete."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from run_rf_reward_formal_test_batch import (
    BLOCKS, EXPECTED_CATALOG_DIGEST, EXPECTED_NORMALIZATION_SHA256,
    _atomic_json, _digest, _exact_block_mapping, _parse_block_path, _read,
    _rf_paths, _sha256, audit_inputs, audit_rf_test_outputs,
)
from transformer_ppo.archive import exact_hv_3d, nondominated
from transformer_ppo.statistics import holm


METHODS = ("random", "rule", "rf", "transformer", "mlp")
OBJECTIVES = ("cost", "emission", "makespan")
MAIN_REFERENCE = (1.1, 1.1, 1.1)
SENSITIVITY_REFERENCES = ((1.05, 1.05, 1.05), (1.2, 1.2, 1.2))
COMPARISONS = (
    ("rf_vs_rule", "rf", "rule"),
    ("transformer_vs_rule", "transformer", "rule"),
    ("transformer_vs_rf", "transformer", "rf"),
)
STRATA = {
    "S0-S8": tuple(f"S{index}" for index in range(9)),
    "S9-S11": tuple(f"S{index}" for index in range(9, 12)),
}
BOOTSTRAP_SAMPLES = 20_000
SIGN_FLIP_SAMPLES = 100_000
STATISTICS_SEED = 191_000_000
TIE_TOLERANCE = 1e-12


def _reference_label(reference) -> str:
    return "_".join(str(value).replace(".", "p") for value in reference)


def _normalise(point, minimum, scale):
    return tuple((float(value) - minimum[index]) / scale[index]
                 for index, value in enumerate(point))


def pooled_bounds(fronts: list[list[tuple[float, float, float]]]) -> dict:
    pooled = nondominated(point for front in fronts for point in front)
    if not pooled:
        raise ValueError("cannot define OOS bounds from an empty pooled front")
    values = np.asarray(pooled, dtype=float)
    minimum = values.min(axis=0)
    maximum = values.max(axis=0)
    zero_span = maximum == minimum
    scale = np.where(zero_span, 1.0, maximum - minimum)
    return {
        "minimum": minimum.tolist(),
        "maximum": maximum.tolist(),
        "scale": scale.tolist(),
        "zero_span": zero_span.tolist(),
        "pooled_nondominated_size": len(pooled),
        "pooled_nondominated_points": [list(point) for point in pooled],
    }


def stratified_bootstrap_ci(rows: list[dict], seed: int) -> list[float]:
    by_configuration = {}
    for row in rows:
        by_configuration.setdefault(row["configuration"], []).append(
            float(row["mean_block_difference"]))
    if not by_configuration:
        raise ValueError("stratified bootstrap requires paired differences")
    rng = np.random.default_rng(seed)
    totals = np.zeros(BOOTSTRAP_SAMPLES, dtype=float)
    count = 0
    for configuration in sorted(by_configuration):
        values = np.asarray(by_configuration[configuration], dtype=float)
        indices = rng.integers(0, len(values), size=(BOOTSTRAP_SAMPLES, len(values)))
        totals += values[indices].sum(axis=1)
        count += len(values)
    return [float(value) for value in np.quantile(totals / count, (0.025, 0.975))]


def sign_flip_mean_p_value(values, seed: int) -> float:
    differences = np.asarray(values, dtype=float)
    if not len(differences) or not np.all(np.isfinite(differences)):
        raise ValueError("finite paired differences required")
    observed = abs(float(differences.mean()))
    rng = np.random.default_rng(seed)
    extreme = 0
    generated = 0
    chunk_size = 5_000
    while generated < SIGN_FLIP_SAMPLES:
        count = min(chunk_size, SIGN_FLIP_SAMPLES - generated)
        signs = rng.choice((-1.0, 1.0), size=(count, len(differences)))
        simulated = np.abs((signs * differences).mean(axis=1))
        extreme += int(np.count_nonzero(simulated >= observed - 1e-15))
        generated += count
    return float((extreme + 1) / (SIGN_FLIP_SAMPLES + 1))


def _oos_path(method: str, block: int, instance_id: str,
              existing_blocks: dict[int, Path], rf_root: Path) -> Path:
    if method == "rf":
        _, oos_dir = _rf_paths(rf_root, block, instance_id)
    else:
        oos_dir = existing_blocks[block] / "instances" / instance_id / method / "oos"
    return oos_dir / "oos_5000.json"


def _run_path(method: str, block: int, instance_id: str,
              existing_blocks: dict[int, Path], rf_root: Path) -> Path:
    if method == "rf":
        run_dir, _ = _rf_paths(rf_root, block, instance_id)
        return run_dir
    return existing_blocks[block] / "instances" / instance_id / method / "run"


def _load_points(path: Path) -> tuple[dict, list[tuple[float, float, float]]]:
    value = _read(path)
    points = []
    for candidate in value.get("candidates", []):
        point = tuple(float(candidate[name]["q90"]) for name in OBJECTIVES)
        if len(point) != 3 or not all(math.isfinite(number) for number in point):
            raise ValueError(f"nonfinite OOS q90 objective at {path}")
        points.append(point)
    return value, nondominated(points)


def _metric_rows(records, existing_blocks, rf_root):
    metrics, bounds_rows, sources = [], [], []
    for record in records:
        instance_id = record["instance_id"]
        fronts = {}
        master_digests = set()
        for block in BLOCKS:
            for method in METHODS:
                path = _oos_path(method, block, instance_id, existing_blocks, rf_root)
                value, front = _load_points(path)
                if (value.get("instance_id") != instance_id
                        or value.get("instance_digest") != record["instance_digest"]
                        or value.get("size") != 5000
                        or value.get("oos_seed") != record["oos5000_seed"]):
                    raise ValueError(f"OOS source disagrees with manifest: {path}")
                master_digests.add(value["oos_master_digest"])
                fronts[(block, method)] = front
                sources.append({
                    "block": block, "configuration": record["configuration"],
                    "instance_id": instance_id, "method": method,
                    "path": str(path), "sha256": _sha256(path),
                    "npz_path": str(path.with_suffix(".npz")),
                    "npz_sha256": _sha256(path.with_suffix(".npz")),
                    "candidate_count": len(value.get("candidates", [])),
                    "local_nondominated_size": len(front),
                    "oos_master_digest": value["oos_master_digest"],
                })
        if len(master_digests) != 1:
            raise ValueError(f"five methods/three blocks lack common OOS master: {instance_id}")
        bounds = pooled_bounds(list(fronts.values()))
        minimum = bounds["minimum"]
        scale = bounds["scale"]
        bounds.update({
            "configuration": record["configuration"],
            "instance_id": instance_id,
            "source": "pooled global nondominated OOS-q90 set across 5 methods x 3 blocks",
            "no_clipping": True,
            "zero_span_rule": "scale=1 and normalized coordinate=0",
            "oos_master_digest": next(iter(master_digests)),
        })
        bounds["bounds_digest"] = _digest({
            key: bounds[key] for key in (
                "instance_id", "minimum", "maximum", "scale", "zero_span",
                "pooled_nondominated_points")
        })
        bounds_rows.append(bounds)
        for (block, method), front in sorted(fronts.items()):
            normalised = [_normalise(point, minimum, scale) for point in front]
            row = {
                "block": block, "configuration": record["configuration"],
                "instance_id": instance_id, "method": method,
                "bounds_digest": bounds["bounds_digest"],
                "oos_master_digest": next(iter(master_digests)),
                "oos_front_size": len(front),
                "oos_hv": exact_hv_3d(normalised, MAIN_REFERENCE),
                "main_reference": list(MAIN_REFERENCE),
                "points_outside_main_reference": sum(
                    any(point[index] >= MAIN_REFERENCE[index] for index in range(3))
                    for point in normalised),
            }
            for reference in SENSITIVITY_REFERENCES:
                row[f"oos_hv_reference_{_reference_label(reference)}"] = exact_hv_3d(
                    normalised, reference)
            metrics.append(row)
    return metrics, bounds_rows, sources


def _paired_statistics(metrics: list[dict], audit: dict) -> dict:
    lookup = {
        (row["instance_id"], int(row["block"]), row["method"]): float(row["oos_hv"])
        for row in metrics
    }
    if len(lookup) != len(metrics):
        raise ValueError("duplicate block-instance-method metric rows")
    instance_metadata = {}
    for row in metrics:
        instance_metadata[row["instance_id"]] = row["configuration"]
    comparison_entries = []
    family_index = 0
    for stratum, configurations in STRATA.items():
        instance_ids = sorted(
            instance_id for instance_id, configuration in instance_metadata.items()
            if configuration in configurations)
        expected_count = 90 if stratum == "S0-S8" else 30
        if len(instance_ids) != expected_count:
            raise ValueError(f"{stratum} requires {expected_count} instances")
        for comparison, left, right in COMPARISONS:
            paired_rows = []
            for instance_id in instance_ids:
                block_differences = [
                    lookup[(instance_id, block, left)]
                    - lookup[(instance_id, block, right)]
                    for block in BLOCKS
                ]
                paired_rows.append({
                    "instance_id": instance_id,
                    "configuration": instance_metadata[instance_id],
                    "block_differences": block_differences,
                    "mean_block_difference": float(np.mean(block_differences)),
                })
            differences = [row["mean_block_difference"] for row in paired_rows]
            outcome = Counter(
                "win" if value > TIE_TOLERANCE
                else "loss" if value < -TIE_TOLERANCE else "tie"
                for value in differences)
            entry = {
                "stratum": stratum, "comparison": comparison,
                "left": left, "right": right,
                "instance_count": len(paired_rows),
                "pairing": "mean of three within-block paired HV differences per instance",
                "mean_difference": float(np.mean(differences)),
                "median_difference": float(np.median(differences)),
                "stratified_instance_bootstrap_95ci": stratified_bootstrap_ci(
                    paired_rows, STATISTICS_SEED + family_index * 10 + 1),
                "two_sided_sign_flip_mean_p_value": sign_flip_mean_p_value(
                    differences, STATISTICS_SEED + family_index * 10 + 2),
                "wins": outcome["win"], "ties": outcome["tie"],
                "losses": outcome["loss"],
                "tie_tolerance": TIE_TOLERANCE,
                "per_configuration": {
                    configuration: {
                        "instance_count": sum(
                            row["configuration"] == configuration for row in paired_rows),
                        "mean_difference": float(np.mean([
                            row["mean_block_difference"] for row in paired_rows
                            if row["configuration"] == configuration
                        ])),
                    }
                    for configuration in configurations
                },
                "per_block_mean_difference": {
                    str(block): float(np.mean([
                        row["block_differences"][block - 1] for row in paired_rows
                    ]))
                    for block in BLOCKS
                },
                "paired_instances": paired_rows,
            }
            comparison_entries.append(entry)
            family_index += 1
    adjusted = holm([
        entry["two_sided_sign_flip_mean_p_value"] for entry in comparison_entries
    ])
    for entry, value in zip(comparison_entries, adjusted):
        entry["holm_adjusted_p_value_six_test_family"] = value
    descriptive = []
    seed_performance = []
    for stratum, configurations in STRATA.items():
        subset = [row for row in metrics if row["configuration"] in configurations]
        for method in METHODS:
            values = [float(row["oos_hv"]) for row in subset if row["method"] == method]
            descriptive.append({
                "stratum": stratum, "method": method,
                "block_instance_count": len(values),
                "mean_oos_hv": float(np.mean(values)),
                "median_oos_hv": float(np.median(values)),
            })
            for block in BLOCKS:
                block_values = [
                    float(row["oos_hv"]) for row in subset
                    if row["method"] == method and int(row["block"]) == block
                ]
                model_sha = None
                if method == "rf":
                    model_sha = audit["rf_models"][str(block)]["sha256"]
                elif method in ("transformer", "mlp"):
                    model_sha = audit["blocks"][str(block)]["checkpoints"][method]["sha256"]
                seed_performance.append({
                    "stratum": stratum, "method": method, "block": block,
                    "model_sha256": model_sha,
                    "instance_count": len(block_values),
                    "mean_oos_hv": float(np.mean(block_values)),
                    "median_oos_hv": float(np.median(block_values)),
                })
    return {
        "schema_version": 1,
        "primary_metric": "OOS-5000 empirical-q90 hypervolume; higher is better",
        "main_reference": list(MAIN_REFERENCE),
        "primary_unit": "test instance",
        "block_aggregation": "mean of three paired block differences",
        "confidence_interval": (
            "20,000-replicate percentile bootstrap, resampling instances within "
            "each fixed configuration"),
        "hypothesis_test": (
            "two-sided 100,000-draw sign-flip randomization test of the mean "
            "instance-level paired difference"),
        "multiplicity": "Holm adjustment over 3 comparisons x 2 strata = 6 tests",
        "comparisons": comparison_entries,
        "method_descriptive": descriptive,
        "model_seed_performance": seed_performance,
    }


def _read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _auxiliary_metrics(records, existing_blocks, rf_root, rf_training_root):
    run_rows = []
    for record in records:
        for block in BLOCKS:
            for method in METHODS:
                run_dir = _run_path(
                    method, block, record["instance_id"], existing_blocks, rf_root)
                complete_path = run_dir / "COMPLETE.json"
                outcomes_path = run_dir / "mutation_outcomes.jsonl"
                skips_path = run_dir / "mutation_skips.jsonl"
                complete = _read(complete_path)
                outcomes = _read_jsonl(outcomes_path)
                skips = _read_jsonl(skips_path)
                attempts = int(complete["mutation_attempts"])
                if (attempts != len(outcomes) + len(skips)
                        or int(complete["no_eligible_candidate"]) != len(skips)
                        or int(complete["effective_modifications"])
                        != sum(bool(row["effective_mutation"]) for row in outcomes)):
                    raise ValueError(f"mutation process counts disagree: {run_dir}")
                run_rows.append({
                    "block": block, "configuration": record["configuration"],
                    "instance_id": record["instance_id"], "method": method,
                    "complete_sha256": _sha256(complete_path),
                    "mutation_outcomes_sha256": _sha256(outcomes_path),
                    "mutation_skips_sha256": _sha256(skips_path),
                    "attempt_denominator": attempts,
                    "effective_modifications": int(complete["effective_modifications"]),
                    "no_eligible_candidate": len(skips),
                    "no_change": sum(not bool(row["decision_changed"]) for row in outcomes),
                    "mutation_failure": sum(
                        not bool(row["mutation_success"]) for row in outcomes),
                    "repair_failure": sum(
                        not bool(row["repair_success"]) for row in outcomes),
                    "mutation_reverted": sum(
                        bool(row["mutation_reverted"]) for row in outcomes),
                    "optimisation_seconds": float(complete["optimisation_seconds"]),
                    "inference_seconds": float(complete["inference_seconds"]),
                    "ccp_evaluation_count": int(complete["ccp_evaluation_count"]),
                })
    aggregate_rows = []
    for stratum, configurations in STRATA.items():
        for method in METHODS:
            subset = [row for row in run_rows
                      if row["configuration"] in configurations
                      and row["method"] == method]
            attempts = sum(row["attempt_denominator"] for row in subset)
            if attempts <= 0:
                raise ValueError(f"zero mutation-attempt denominator: {stratum}/{method}")
            aggregate = {
                "stratum": stratum, "method": method,
                "run_count": len(subset), "attempt_denominator": attempts,
                "denominator_definition": "all mutation attempts, including no-candidate skips",
                "category_overlap": True,
                "total_optimisation_seconds": sum(
                    row["optimisation_seconds"] for row in subset),
                "total_inference_seconds": sum(row["inference_seconds"] for row in subset),
                "total_ccp_evaluations": sum(
                    row["ccp_evaluation_count"] for row in subset),
            }
            for field in (
                    "effective_modifications", "no_eligible_candidate", "no_change",
                    "mutation_failure", "repair_failure", "mutation_reverted"):
                count = sum(row[field] for row in subset)
                aggregate[field] = count
                aggregate[f"{field}_rate"] = float(count / attempts)
            aggregate_rows.append(aggregate)
    training_path = rf_training_root / "FORMAL_TRAINING_COMPLETE.json"
    training = _read(training_path)
    training_cost = {
        "source": str(training_path),
        "source_sha256": _sha256(training_path),
        "actual_ccp100_evaluations": training["actual_ccp100_evaluations"],
        "wall_seconds": training["wall_seconds"],
        "trajectory_sample_counts": {
            str(row["trajectory"]): row["sample_count"]
            for row in training["trajectory_results"]
        },
    }
    return {
        "schema_version": 1,
        "purpose": "supporting process metrics; never a substitute for OOS-HV",
        "rate_denominator": "all mutation attempts",
        "overlapping_categories": (
            "no-change, mutation-failure, repair-failure and reverted are not "
            "mutually exclusive"),
        "run_rows": run_rows,
        "aggregates": aggregate_rows,
        "rf_training_cost": training_cost,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path,
                        default=Path(__file__).parent / "data/data_expanded.xlsx")
    parser.add_argument("--instances", type=Path, required=True)
    parser.add_argument("--normalization", type=Path, required=True)
    parser.add_argument("--rf-training-root", type=Path, required=True)
    parser.add_argument("--existing-block", type=_parse_block_path,
                        action="append", required=True)
    parser.add_argument("--rf-formal-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    try:
        existing_blocks = _exact_block_mapping(args.existing_block)
    except ValueError as error:
        parser.error(str(error))
    for name in (
            "data", "instances", "normalization", "rf_training_root",
            "rf_formal_root", "out"):
        setattr(args, name, getattr(args, name).resolve())
    if args.out.exists() and any(args.out.iterdir()):
        parser.error("analysis output directory must be empty")
    try:
        records, models, audit = audit_inputs(
            args.data, args.instances, args.normalization,
            args.rf_training_root, existing_blocks)
        rf_complete = audit_rf_test_outputs(
            args.rf_formal_root, records, models, audit)
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))
    protocol = {
        "schema_version": 1,
        "frozen_before_five_method_HV_read": True,
        "catalog_digest": EXPECTED_CATALOG_DIGEST,
        "normalization_file_sha256": EXPECTED_NORMALIZATION_SHA256,
        "methods": list(METHODS),
        "blocks": list(BLOCKS),
        "test_instances": 120,
        "primary_oos_statistic": "empirical q90 for cost, emission and makespan",
        "bounds_unit": "one common set per test instance across 5 methods x 3 blocks",
        "bounds_source": "componentwise min/max of pooled global nondominated OOS-q90 set",
        "normalization_formula": "(objective-minimum)/scale",
        "zero_span_rule": "scale=1 and normalized coordinate=0",
        "clipping": False,
        "main_reference": list(MAIN_REFERENCE),
        "sensitivity_references": [list(value) for value in SENSITIVITY_REFERENCES],
        "primary_comparisons": [row[0] for row in COMPARISONS],
        "strata": {key: list(value) for key, value in STRATA.items()},
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "sign_flip_samples": SIGN_FLIP_SAMPLES,
        "statistics_seed": STATISTICS_SEED,
        "multiple_testing": "Holm over six primary tests",
        "existing_source_audit_digest": _digest(audit),
        "rf_formal_complete_sha256": _sha256(
            args.rf_formal_root / "RF_FORMAL_COMPLETE.json"),
        "rf_formal_source_manifest_sha256": rf_complete[
            "rf_formal_source_manifest_sha256"],
        "legacy_hv_values_used": False,
        "outputs_are_additive": True,
        "implementation_sha256": _sha256(Path(__file__)),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    protocol_path = args.out / "analysis_protocol.json"
    _atomic_json(protocol_path, protocol)
    metrics, bounds, sources = _metric_rows(records, existing_blocks, args.rf_formal_root)
    if len(metrics) != 120 * 3 * 5 or len(bounds) != 120 or len(sources) != 120 * 3 * 5:
        raise RuntimeError("five-method formal analysis has unexpected row counts")
    statistics = _paired_statistics(metrics, audit)
    auxiliary = _auxiliary_metrics(
        records, existing_blocks, args.rf_formal_root, args.rf_training_root)
    metrics_path = args.out / "five_method_oos_hv_rows.json"
    bounds_path = args.out / "per_instance_oos_bounds.json"
    sources_path = args.out / "raw_oos_source_manifest.json"
    statistics_path = args.out / "formal_statistics.json"
    auxiliary_path = args.out / "auxiliary_process_metrics.json"
    _atomic_json(metrics_path, metrics)
    _atomic_json(bounds_path, bounds)
    _atomic_json(sources_path, sources)
    _atomic_json(statistics_path, statistics)
    _atomic_json(auxiliary_path, auxiliary)
    completion = {
        "completed": True,
        "metric_row_count": len(metrics),
        "bounds_count": len(bounds),
        "raw_oos_source_count": len(sources),
        "primary_test_count": len(statistics["comparisons"]),
        "analysis_protocol_sha256": _sha256(protocol_path),
        "five_method_oos_hv_rows_sha256": _sha256(metrics_path),
        "per_instance_oos_bounds_sha256": _sha256(bounds_path),
        "raw_oos_source_manifest_sha256": _sha256(sources_path),
        "formal_statistics_sha256": _sha256(statistics_path),
        "auxiliary_process_metrics_sha256": _sha256(auxiliary_path),
        "legacy_hv_values_used": False,
        "source_files_modified": False,
    }
    _atomic_json(args.out / "ANALYSIS_COMPLETE.json", completion)
    print(json.dumps(completion, sort_keys=True))


if __name__ == "__main__":
    main()
