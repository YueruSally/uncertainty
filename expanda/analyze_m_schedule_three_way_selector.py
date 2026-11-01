#!/usr/bin/env python3
"""Offline Base/G10/G12 selector from pre-intervention search state."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from analyze_m_schedule_state_gate import _find_run, _jsonl, _read
from m_schedule_state_features import (
    EPSILON,
    assert_progress_replay,
    pre_intervention_progress,
    progress_features,
    short_instance_id,
)


MODELS = ("ridge", "extra_trees_regressor")
MARGINS = (0.0, 0.5, 1.0, 2.0)
STRATEGIES = ("REPLACE_G10_P100", "REPLACE_G12_P100")


def _paired_percent(base_hv, candidate_hv):
    if base_hv <= EPSILON:
        raise ValueError(f"invalid Base HV: {base_hv}")
    return 100.0 * (candidate_hv - base_hv) / base_hv


def _build_rows(roots, instances, seeds, evaluation_budget):
    rows = []
    feature_names = None
    for instance in instances:
        for seed in seeds:
            runs = {
                strategy: _find_run(roots, instance, seed, strategy)
                for strategy in ("BASE", *STRATEGIES)
            }
            configs = {
                strategy: _read(run / "configuration.json")
                for strategy, run in runs.items()
            }
            digests = {
                config.get("scenario_digest") for config in configs.values()}
            if len(digests) != 1:
                raise ValueError(
                    f"scenario mismatch for {instance}:seed_{seed}")

            base_progress = pre_intervention_progress(
                _jsonl(runs["BASE"] / "search_progress.jsonl"),
                10, evaluation_budget)
            for strategy, minimum_generation in (
                    ("REPLACE_G10_P100", 10),
                    ("REPLACE_G12_P100", 12)):
                scheduled_progress = pre_intervention_progress(
                    _jsonl(runs[strategy] / "search_progress.jsonl"),
                    10, evaluation_budget)
                try:
                    assert_progress_replay(base_progress, scheduled_progress)
                except ValueError as exc:
                    raise ValueError(
                        f"{instance}:seed_{seed}:{strategy} replay: {exc}") \
                        from exc

            names, features = progress_features(base_progress)
            if feature_names is None:
                feature_names = names
            elif feature_names != names:
                raise ValueError("three-way feature schema changed")
            completes = {
                strategy: _read(run / "COMPLETE.json")
                for strategy, run in runs.items()
            }
            base_hv = float(completes["BASE"]["final_hypervolume"])
            gains = {
                strategy: _paired_percent(
                    base_hv,
                    float(completes[strategy]["final_hypervolume"]))
                for strategy in STRATEGIES
            }
            oracle_strategy = max(
                ("BASE", *STRATEGIES),
                key=lambda strategy: 0.0 if strategy == "BASE"
                else gains[strategy])
            rows.append({
                "key": f"{instance}:{seed}",
                "instance": instance,
                "algorithm_seed": int(seed),
                "features": features,
                "gains": gains,
                "oracle_strategy": oracle_strategy,
                "oracle_gain": max(0.0, *gains.values()),
            })
    return rows, feature_names


def _model(name, seed):
    if name == "ridge":
        return make_pipeline(StandardScaler(), Ridge(alpha=10.0))
    if name == "extra_trees_regressor":
        return ExtraTreesRegressor(
            n_estimators=500,
            max_depth=5,
            min_samples_leaf=3,
            max_features="sqrt",
            random_state=seed,
            n_jobs=-1,
        )
    raise ValueError(f"unknown model: {name}")


def _fit_predict(training, validation, model_name, seed):
    model = _model(model_name, seed)
    model.fit(
        np.asarray([row["features"] for row in training], dtype=float),
        np.asarray([
            [row["gains"][strategy] for strategy in STRATEGIES]
            for row in training
        ], dtype=float),
    )
    predictions = np.asarray(model.predict(np.asarray(
        [row["features"] for row in validation], dtype=float)), dtype=float)
    if predictions.shape != (len(validation), len(STRATEGIES)):
        raise ValueError(f"unexpected prediction shape: {predictions.shape}")
    if not np.isfinite(predictions).all():
        raise ValueError("non-finite schedule prediction")
    return predictions


def _select(prediction, margin):
    index = int(np.argmax(prediction))
    if float(prediction[index]) < margin:
        return "BASE"
    return STRATEGIES[index]


def _deployment(rows, predictions, margin):
    selected = [_select(prediction, margin) for prediction in predictions]
    realized = [
        0.0 if strategy == "BASE" else row["gains"][strategy]
        for row, strategy in zip(rows, selected)
    ]
    active = [
        value for value, strategy in zip(realized, selected)
        if strategy != "BASE"]
    return {
        "predicted_gain_margin": margin,
        "evaluation_runs": len(rows),
        "action_counts": {
            strategy: selected.count(strategy)
            for strategy in ("BASE", *STRATEGIES)
        },
        "active_runs": len(active),
        "wins": sum(value > EPSILON for value in realized),
        "ties_or_fallbacks": sum(abs(value) <= EPSILON for value in realized),
        "losses": sum(value < -EPSILON for value in realized),
        "active_precision": (
            sum(value > EPSILON for value in active) / len(active)
            if active else 0.0),
        "active_loss_fraction": (
            sum(value < -EPSILON for value in active) / len(active)
            if active else 0.0),
        "mean_deployed_percent": statistics.fmean(realized),
        "median_deployed_percent": statistics.median(realized),
        "minimum_deployed_percent": min(realized),
        "maximum_deployed_percent": max(realized),
        "oracle_action_accuracy": sum(
            strategy == row["oracle_strategy"]
            for row, strategy in zip(rows, selected)) / len(rows),
        "selected": [
            {
                "key": row["key"],
                "strategy": strategy,
                "realized_percent": value,
                "predicted_g10_percent": float(prediction[0]),
                "predicted_g12_percent": float(prediction[1]),
            }
            for row, prediction, strategy, value in zip(
                rows, predictions, selected, realized)
        ],
    }


def _result(rows, predictions, protocol, model_name):
    return {
        "protocol": protocol,
        "model": model_name,
        "run_count": len(rows),
        "deployments": [
            _deployment(rows, predictions, margin) for margin in MARGINS],
    }


def _group_cv(rows, model_name, group_field, seed):
    by_key = {}
    for fold_index, heldout in enumerate(sorted({
            row[group_field] for row in rows})):
        training = [row for row in rows if row[group_field] != heldout]
        validation = [row for row in rows if row[group_field] == heldout]
        predictions = _fit_predict(
            training, validation, model_name, seed + fold_index)
        by_key.update({
            row["key"]: prediction
            for row, prediction in zip(validation, predictions)
        })
    ordered = np.asarray([by_key[row["key"]] for row in rows])
    return _result(
        rows, ordered, f"leave_one_{group_field}_out", model_name)


def _seed_holdout(rows, model_name, training_seeds, seed):
    training = [
        row for row in rows if row["algorithm_seed"] in training_seeds]
    validation = [
        row for row in rows if row["algorithm_seed"] not in training_seeds]
    predictions = _fit_predict(training, validation, model_name, seed)
    result = _result(
        validation, predictions,
        "train_first_3_seeds_test_last_7_seeds", model_name)
    result["training_run_count"] = len(training)
    result["training_seeds"] = sorted(training_seeds)
    result["test_seeds"] = sorted({
        row["algorithm_seed"] for row in validation})
    return result


def _baseline(rows, strategy):
    values = [
        0.0 if strategy == "BASE" else row["gains"][strategy]
        for row in rows]
    return {
        "strategy": strategy,
        "runs": len(rows),
        "mean_paired_percent": statistics.fmean(values),
        "median_paired_percent": statistics.median(values),
        "minimum_paired_percent": min(values),
        "maximum_paired_percent": max(values),
        "wins": sum(value > EPSILON for value in values),
        "losses": sum(value < -EPSILON for value in values),
    }


def _compact(result):
    return {
        key: value for key, value in result.items()
        if key != "deployments"
    } | {
        "deployments": [
            {key: value for key, value in deployment.items()
             if key != "selected"}
            for deployment in result["deployments"]
        ]
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roots", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--instance-ids", nargs="+", required=True)
    parser.add_argument("--algorithm-seeds", nargs="+", type=int,
                        required=True)
    parser.add_argument("--training-seeds", nargs="+", type=int,
                        default=(777001, 777002, 777003))
    parser.add_argument("--evaluation-budget", type=int, default=400)
    parser.add_argument("--seed", type=int, default=13001)
    args = parser.parse_args(argv)

    instances = [short_instance_id(value) for value in args.instance_ids]
    rows, feature_names = _build_rows(
        args.roots, instances, args.algorithm_seeds,
        args.evaluation_budget)
    training_seeds = set(args.training_seeds)
    test_rows = [
        row for row in rows if row["algorithm_seed"] not in training_seeds]

    results = []
    for offset, model_name in enumerate(MODELS):
        model_seed = args.seed + offset * 1_000
        results.extend((
            _group_cv(rows, model_name, "instance", model_seed),
            _group_cv(
                rows, model_name, "algorithm_seed", model_seed + 100),
            _seed_holdout(
                rows, model_name, training_seeds, model_seed + 200),
        ))

    report = {
        "completed": True,
        "definition": (
            "pre-generation-10 regression selector among exact Base, "
            "REPLACE_G10_P100, and REPLACE_G12_P100"),
        "scenario_exact": True,
        "pre_intervention_replay_exact": True,
        "run_count": len(rows),
        "feature_dimension": len(feature_names),
        "feature_names": feature_names,
        "all_run_baselines": [
            _baseline(rows, strategy)
            for strategy in ("BASE", *STRATEGIES)],
        "seed_holdout_baselines": [
            _baseline(test_rows, strategy)
            for strategy in ("BASE", *STRATEGIES)],
        "positive_oracle": {
            "all_runs_mean_percent": statistics.fmean(
                row["oracle_gain"] for row in rows),
            "seed_holdout_mean_percent": statistics.fmean(
                row["oracle_gain"] for row in test_rows),
            "all_runs_action_counts": {
                strategy: sum(
                    row["oracle_strategy"] == strategy for row in rows)
                for strategy in ("BASE", *STRATEGIES)
            },
        },
        "results": results,
        "admission_thresholds": {
            "primary_protocol": "train_first_3_seeds_test_last_7_seeds",
            "minimum_active_runs": 14,
            "minimum_mean_deployed_percent": 2.0,
            "maximum_active_loss_fraction": 0.20,
            "minimum_deployed_percent": -5.0,
        },
    }
    threshold = report["admission_thresholds"]
    admitted = []
    for result in results:
        if result["protocol"] != threshold["primary_protocol"]:
            continue
        for deployment in result["deployments"]:
            if (
                deployment["active_runs"] >= threshold["minimum_active_runs"]
                and deployment["mean_deployed_percent"]
                    >= threshold["minimum_mean_deployed_percent"]
                and deployment["active_loss_fraction"]
                    <= threshold["maximum_active_loss_fraction"]
                and deployment["minimum_deployed_percent"]
                    >= threshold["minimum_deployed_percent"]
            ):
                admitted.append({
                    "model": result["model"],
                    **{key: value for key, value in deployment.items()
                       if key != "selected"},
                })
    report["admission_pass"] = bool(admitted)
    report["admitted_settings"] = admitted
    report["rows"] = rows

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "FINAL_REPORT.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n",
        encoding="utf-8")
    quick = {
        "completed": True,
        "definition": report["definition"],
        "scenario_exact": True,
        "pre_intervention_replay_exact": True,
        "run_count": len(rows),
        "feature_dimension": len(feature_names),
        "all_run_baselines": report["all_run_baselines"],
        "seed_holdout_baselines": report["seed_holdout_baselines"],
        "positive_oracle": report["positive_oracle"],
        "seed_holdout_results": [
            _compact(result) for result in results
            if result["protocol"]
            == "train_first_3_seeds_test_last_7_seeds"],
        "loo_results": [
            _compact(result) for result in results
            if result["protocol"]
            != "train_first_3_seeds_test_last_7_seeds"],
        "admission_thresholds": threshold,
        "admission_pass": report["admission_pass"],
        "admitted_settings": admitted,
    }
    (args.out / "QUICK_RESULT.json").write_text(
        json.dumps(quick, indent=2, allow_nan=False) + "\n",
        encoding="utf-8")
    print(json.dumps(quick, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
