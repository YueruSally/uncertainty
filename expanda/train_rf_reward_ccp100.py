#!/usr/bin/env python3
"""Fit RF-Reward with instance-grouped validation on exact executed-action labels."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold

from multi_instance_logging import write_json
from transformer_ppo.rf_reward import FEATURE_NAMES


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(roots):
    samples, sources = [], []
    for root in roots:
        for path in sorted(root.rglob("rf_reward_samples.jsonl")):
            config_path = path.parent / "configuration.json"
            config = json.loads(config_path.read_text(encoding="utf-8"))
            if config.get("instance_split") != "train":
                raise ValueError(f"RF training source is not a training instance: {path}")
            count = 0
            with path.open(encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, 1):
                    row = json.loads(line)
                    if tuple(row.get("feature_names", ())) != FEATURE_NAMES:
                        raise ValueError(f"feature schema mismatch at {path}:{line_number}")
                    values = np.asarray(row["feature_values"], dtype=float)
                    if values.shape != (len(FEATURE_NAMES),) or not np.all(np.isfinite(values)):
                        raise ValueError(f"invalid features at {path}:{line_number}")
                    if not np.isfinite(float(row["reward"])):
                        raise ValueError(f"nonfinite reward at {path}:{line_number}")
                    if not 0.0 < float(row["logging_propensity"]) <= 1.0:
                        raise ValueError(f"invalid logging propensity at {path}:{line_number}")
                    delta = float(row["reward_archive_hv_after"]) - float(
                        row["reward_archive_hv_before"])
                    if abs(delta - float(row["reward_hv_delta"])) > 1e-10:
                        raise ValueError(f"archive reward inconsistency at {path}:{line_number}")
                    row["source_path"] = str(path)
                    samples.append(row)
                    count += 1
            sources.append({"path": str(path), "sha256": _sha256(path), "samples": count,
                            "instance_id": config["instance_id"]})
    if not samples:
        raise ValueError("no RF reward samples found")
    digest = hashlib.sha256(json.dumps(
        sources, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return samples, sources, digest


def _group_metrics(y_true, y_pred, groups):
    rows = []
    for group in sorted(set(groups)):
        mask = groups == group
        error = y_pred[mask] - y_true[mask]
        rows.append({
            "instance_id": group,
            "samples": int(mask.sum()),
            "mae": float(np.mean(np.abs(error))),
            "rmse": float(np.sqrt(np.mean(error ** 2))),
        })
    return {
        "per_instance": rows,
        "mean_instance_mae": float(np.mean([row["mae"] for row in rows])),
        "mean_instance_rmse": float(np.mean([row["rmse"] for row in rows])),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--rf-seed", type=int, required=True)
    parser.add_argument("--jobs", type=int, default=1)
    args = parser.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        parser.error("output directory must be empty")
    args.out.mkdir(parents=True, exist_ok=True)
    samples, sources, data_digest = _load(args.runs)
    X = np.asarray([row["feature_values"] for row in samples], dtype=float)
    y = np.asarray([row["reward"] for row in samples], dtype=float)
    groups = np.asarray([row["instance_id"] for row in samples], dtype=object)
    unique_groups = sorted(set(groups))
    if len(unique_groups) < 2:
        raise ValueError("instance-grouped validation requires at least two training instances")
    folds = min(3, len(unique_groups))
    grid = [dict(n_estimators=200, max_depth=depth,
                 min_samples_leaf=leaf, max_features=features)
            for depth, leaf, features in itertools.product(
                (8, None), (2, 5), (0.5, 1.0))]
    candidates = []
    splitter = GroupKFold(n_splits=folds)
    baseline_predictions = np.full(len(y), np.nan)
    for train, valid in splitter.split(X, y, groups):
        baseline_predictions[valid] = float(np.mean(y[train]))
    baseline_validation = _group_metrics(y, baseline_predictions, groups)
    for index, params in enumerate(grid):
        predictions = np.full(len(y), np.nan)
        fold_groups = []
        for train, valid in splitter.split(X, y, groups):
            model = RandomForestRegressor(
                **params, random_state=args.rf_seed + index, n_jobs=args.jobs)
            model.fit(X[train], y[train])
            predictions[valid] = model.predict(X[valid])
            fold_groups.append(sorted(set(groups[valid])))
        metrics = _group_metrics(y, predictions, groups)
        candidates.append({"parameters": params, "validation": metrics,
                           "fold_validation_instances": fold_groups})
    best = min(candidates, key=lambda row: (
        row["validation"]["mean_instance_rmse"],
        row["validation"]["mean_instance_mae"],
        json.dumps(row["parameters"], sort_keys=True)))
    model = RandomForestRegressor(
        **best["parameters"], random_state=args.rf_seed, n_jobs=args.jobs)
    model.fit(X, y)
    propensities = np.asarray([row["logging_propensity"] for row in samples], dtype=float)
    artifact = {
        "schema_version": 1,
        "target": "ppo_immediate_mutation_reward",
        "feature_names": FEATURE_NAMES,
        "model": model,
        "rf_seed": args.rf_seed,
        "training_data_digest": data_digest,
        "training_instance_ids": unique_groups,
        "hyperparameters": best["parameters"],
        "sample_weighting": "none",
        "sample_weighting_reason": (
            "estimate conditional mean reward directly; logging propensities are retained "
            "for overlap checks and a separately predeclared sensitivity analysis"),
    }
    model_path = args.out / "rf_reward.joblib"
    joblib.dump(artifact, model_path)
    summary = {
        "schema_version": 1,
        "model": str(model_path),
        "model_sha256": _sha256(model_path),
        "training_data_digest": data_digest,
        "sources": sources,
        "sample_count": len(samples),
        "instance_count": len(unique_groups),
        "operator_counts": {operator: sum(row["operator"] == operator for row in samples)
                            for operator in sorted({row["operator"] for row in samples})},
        "reward": {"mean": float(y.mean()), "std": float(y.std()),
                   "minimum": float(y.min()), "maximum": float(y.max()),
                   "positive_fraction": float(np.mean(y > 0)),
                   "negative_fraction": float(np.mean(y < 0))},
        "logging_propensity": {
            "minimum": float(propensities.min()),
            "median": float(np.median(propensities)),
            "maximum": float(propensities.max()),
            "unclipped_ips_effective_sample_size": float(
                (np.sum(1.0 / propensities) ** 2)
                / np.sum((1.0 / propensities) ** 2)),
        },
        "grouped_validation_folds": folds,
        "training_fold_mean_baseline": baseline_validation,
        "best": best,
        "all_candidates": candidates,
        "sample_weighting": "none",
    }
    write_json(args.out / "training_summary.json", summary)
    print({key: summary[key] for key in (
        "sample_count", "instance_count", "model_sha256", "best")})


if __name__ == "__main__":
    main()
