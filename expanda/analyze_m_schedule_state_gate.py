#!/usr/bin/env python3
"""Predict whether a generation-10 forced-replace M schedule should run."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from m_schedule_state_features import (
    EPSILON,
    assert_progress_replay,
    pre_intervention_progress,
    progress_features,
    short_instance_id,
)


MODELS = ("balanced_logistic", "extra_trees")
THRESHOLDS = (0.50, 0.60, 0.70, 0.80)


def _read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl(path: Path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _find_run(roots, instance, seed, strategy):
    candidates = [
        root / instance / f"seed_{seed}" / strategy for root in roots]
    matches = [
        path for path in candidates
        if (path / "COMPLETE.json").is_file()
        and (path / "configuration.json").is_file()
        and (path / "search_progress.jsonl").is_file()
    ]
    if not matches:
        raise FileNotFoundError(
            f"missing {strategy} run for {instance}:seed_{seed}; "
            f"searched={[str(path) for path in candidates]}")
    return matches[0]


def _build_rows(roots, instances, seeds, minimum_generation,
                evaluation_budget):
    rows = []
    feature_names = None
    scenario_failures = []
    replay_failures = []
    for instance in instances:
        for seed in seeds:
            base = _find_run(roots, instance, seed, "BASE")
            scheduled = _find_run(
                roots, instance, seed, "REPLACE_G10_P100")
            base_config = _read(base / "configuration.json")
            scheduled_config = _read(scheduled / "configuration.json")
            if (base_config.get("scenario_digest")
                    != scheduled_config.get("scenario_digest")):
                scenario_failures.append(f"{instance}:seed_{seed}")
            base_progress = pre_intervention_progress(
                _jsonl(base / "search_progress.jsonl"),
                minimum_generation, evaluation_budget)
            scheduled_progress = pre_intervention_progress(
                _jsonl(scheduled / "search_progress.jsonl"),
                minimum_generation, evaluation_budget)
            try:
                assert_progress_replay(base_progress, scheduled_progress)
            except ValueError as exc:
                replay_failures.append(
                    f"{instance}:seed_{seed}: {exc}")
            names, features = progress_features(base_progress)
            if feature_names is None:
                feature_names = names
            elif names != feature_names:
                raise ValueError("pre-intervention feature schema changed")
            base_hv = float(_read(base / "COMPLETE.json")[
                "final_hypervolume"])
            scheduled_hv = float(_read(scheduled / "COMPLETE.json")[
                "final_hypervolume"])
            if base_hv <= EPSILON:
                raise ValueError(f"invalid Base HV: {base}")
            gain = 100.0 * (scheduled_hv - base_hv) / base_hv
            rows.append({
                "key": f"{instance}:{seed}",
                "instance": instance,
                "algorithm_seed": int(seed),
                "base_run": str(base),
                "scheduled_run": str(scheduled),
                "base_hv": base_hv,
                "scheduled_hv": scheduled_hv,
                "paired_percent": gain,
                "positive": bool(gain > EPSILON),
                "features": features,
            })
    if scenario_failures:
        raise ValueError(f"scenario mismatch: {scenario_failures}")
    if replay_failures:
        raise ValueError(
            "pre-intervention state replay mismatch: "
            + " | ".join(replay_failures[:5]))
    return rows, feature_names


def _model(name, seed):
    if name == "balanced_logistic":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=0.1, class_weight="balanced", max_iter=5_000,
                random_state=seed),
        )
    if name == "extra_trees":
        return ExtraTreesClassifier(
            n_estimators=500,
            max_depth=5,
            min_samples_leaf=3,
            max_features="sqrt",
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        )
    raise ValueError(f"unknown model: {name}")


def _fit_predict(training, validation, model_name, seed):
    labels = np.asarray([row["positive"] for row in training], dtype=int)
    if len(np.unique(labels)) != 2:
        raise ValueError("training split lacks positive and negative labels")
    model = _model(model_name, seed)
    model.fit(
        np.asarray([row["features"] for row in training], dtype=float),
        labels,
    )
    return model.predict_proba(np.asarray(
        [row["features"] for row in validation], dtype=float))[:, 1]


def _deployment(rows, probabilities, threshold):
    selected = [
        row for row, probability in zip(rows, probabilities)
        if probability >= threshold
    ]
    gains = [row["paired_percent"] for row in selected]
    all_gains = [row["paired_percent"] for row in rows]
    count = len(selected)
    return {
        "probability_threshold": threshold,
        "evaluation_runs": len(rows),
        "selected_runs": count,
        "fallback_runs": len(rows) - count,
        "selected_fraction": count / len(rows),
        "selection_precision": (
            sum(value > EPSILON for value in gains) / count
            if count else 0.0),
        "selected_loss_fraction": (
            sum(value < -EPSILON for value in gains) / count
            if count else 0.0),
        "mean_selected_percent": (
            statistics.fmean(gains) if count else 0.0),
        "median_selected_percent": (
            statistics.median(gains) if count else 0.0),
        "minimum_selected_percent": min(gains) if count else None,
        "maximum_selected_percent": max(gains) if count else None,
        "mean_deployed_percent": sum(gains) / len(rows),
        "always_g10_mean_percent": statistics.fmean(all_gains),
        "retained_positive_gain_fraction": (
            sum(max(0.0, value) for value in gains)
            / max(EPSILON, sum(max(0.0, value) for value in all_gains))),
        "selected_keys": [row["key"] for row in selected],
    }


def _prediction_metrics(rows, probabilities):
    labels = np.asarray([row["positive"] for row in rows], dtype=int)
    result = {
        "run_count": len(rows),
        "positive_run_count": int(labels.sum()),
        "positive_run_fraction": float(labels.mean()),
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "average_precision": float(
            average_precision_score(labels, probabilities)),
        "thresholds": [
            _deployment(rows, probabilities, threshold)
            for threshold in THRESHOLDS
        ],
        "predictions": [
            {
                "key": row["key"],
                "instance": row["instance"],
                "algorithm_seed": row["algorithm_seed"],
                "paired_percent": row["paired_percent"],
                "positive": row["positive"],
                "probability": float(probability),
            }
            for row, probability in zip(rows, probabilities)
        ],
    }
    return result


def _group_cv(rows, model_name, group_field, seed):
    probabilities = {}
    groups = sorted({row[group_field] for row in rows})
    for fold_index, heldout in enumerate(groups):
        training = [row for row in rows if row[group_field] != heldout]
        validation = [row for row in rows if row[group_field] == heldout]
        predicted = _fit_predict(
            training, validation, model_name, seed + fold_index)
        probabilities.update({
            row["key"]: float(probability)
            for row, probability in zip(validation, predicted)
        })
    ordered = np.asarray([probabilities[row["key"]] for row in rows])
    result = _prediction_metrics(rows, ordered)
    result["protocol"] = f"leave_one_{group_field}_out"
    result["model"] = model_name
    return result


def _seed_holdout(rows, model_name, training_seeds, seed):
    training = [
        row for row in rows if row["algorithm_seed"] in training_seeds]
    validation = [
        row for row in rows if row["algorithm_seed"] not in training_seeds]
    probabilities = _fit_predict(training, validation, model_name, seed)
    result = _prediction_metrics(validation, probabilities)
    result.update({
        "protocol": "train_first_3_seeds_test_last_7_seeds",
        "model": model_name,
        "training_run_count": len(training),
        "training_seeds": sorted(training_seeds),
        "test_seeds": sorted({
            row["algorithm_seed"] for row in validation}),
    })
    return result


def _consensus(left, right):
    if left["protocol"] != right["protocol"]:
        raise ValueError("consensus protocols differ")
    left_predictions = {row["key"]: row for row in left["predictions"]}
    right_predictions = {row["key"]: row for row in right["predictions"]}
    if set(left_predictions) != set(right_predictions):
        raise ValueError("consensus prediction keys differ")
    rows = []
    left_probabilities = []
    right_probabilities = []
    for key in left_predictions:
        lrow = left_predictions[key]
        rrow = right_predictions[key]
        rows.append({
            "key": key,
            "instance": lrow["instance"],
            "algorithm_seed": lrow["algorithm_seed"],
            "paired_percent": lrow["paired_percent"],
            "positive": lrow["positive"],
        })
        left_probabilities.append(lrow["probability"])
        right_probabilities.append(rrow["probability"])
    thresholds = []
    for threshold in THRESHOLDS:
        selected_probabilities = np.asarray([
            min(left_probability, right_probability)
            for left_probability, right_probability in zip(
                left_probabilities, right_probabilities)
        ])
        thresholds.append(_deployment(rows, selected_probabilities, threshold))
    return {
        "protocol": left["protocol"],
        "model": "logistic_extra_trees_consensus",
        "run_count": len(rows),
        "thresholds": thresholds,
    }


def _baseline(rows):
    gains = [row["paired_percent"] for row in rows]
    mean = statistics.fmean(gains)
    standard_deviation = statistics.stdev(gains)
    standard_error = standard_deviation / math.sqrt(len(gains))
    return {
        "runs": len(rows),
        "mean_paired_percent": mean,
        "mean_95ci_low": mean - 1.96 * standard_error,
        "mean_95ci_high": mean + 1.96 * standard_error,
        "median_paired_percent": statistics.median(gains),
        "minimum_paired_percent": min(gains),
        "maximum_paired_percent": max(gains),
        "wins": sum(value > EPSILON for value in gains),
        "losses": sum(value < -EPSILON for value in gains),
    }


def _compact(protocol):
    return {
        "protocol": protocol["protocol"],
        "model": protocol["model"],
        "roc_auc": protocol.get("roc_auc"),
        "average_precision": protocol.get("average_precision"),
        "thresholds": [
            {key: value for key, value in deployment.items()
             if key != "selected_keys"}
            for deployment in protocol["thresholds"]
        ],
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
    parser.add_argument("--minimum-generation", type=int, default=10)
    parser.add_argument("--evaluation-budget", type=int, default=400)
    parser.add_argument("--seed", type=int, default=12001)
    args = parser.parse_args(argv)

    instances = [short_instance_id(value) for value in args.instance_ids]
    rows, feature_names = _build_rows(
        args.roots, instances, args.algorithm_seeds,
        args.minimum_generation, args.evaluation_budget)
    if len(rows) != len(instances) * len(args.algorithm_seeds):
        raise ValueError("incomplete M schedule gate dataset")
    training_seeds = set(args.training_seeds)

    protocols = []
    for offset, model_name in enumerate(MODELS):
        model_seed = args.seed + offset * 1_000
        protocols.extend((
            _group_cv(rows, model_name, "instance", model_seed),
            _group_cv(rows, model_name, "algorithm_seed", model_seed + 100),
            _seed_holdout(rows, model_name, training_seeds, model_seed + 200),
        ))
    consensus = []
    for protocol_name in (
            "leave_one_instance_out", "leave_one_algorithm_seed_out",
            "train_first_3_seeds_test_last_7_seeds"):
        pair = [
            protocol for protocol in protocols
            if protocol["protocol"] == protocol_name]
        consensus.append(_consensus(pair[0], pair[1]))

    report = {
        "completed": True,
        "definition": (
            "pre-generation-10 Base search-state gate for deciding whether "
            "to deploy REPLACE_G10_P100 or fall back to exact Base"),
        "scenario_exact": True,
        "pre_intervention_replay_exact": True,
        "instance_count": len(instances),
        "algorithm_seed_count": len(args.algorithm_seeds),
        "run_count": len(rows),
        "feature_dimension": len(feature_names),
        "feature_names": feature_names,
        "always_g10_baseline": _baseline(rows),
        "positive_oracle": {
            "mean_deployed_percent": (
                sum(max(0.0, row["paired_percent"]) for row in rows)
                / len(rows)),
            "selected_runs": sum(row["positive"] for row in rows),
        },
        "protocols": protocols,
        "consensus_protocols": consensus,
        "admission_thresholds": {
            "primary_protocol": "train_first_3_seeds_test_last_7_seeds",
            "minimum_selected_runs": 14,
            "minimum_selection_precision": 0.70,
            "minimum_mean_deployed_percent": 2.0,
            "maximum_selected_loss_fraction": 0.20,
            "minimum_selected_percent": -5.0,
        },
    }
    admission = report["admission_thresholds"]
    primary = [
        protocol for protocol in protocols + consensus
        if protocol["protocol"] == admission["primary_protocol"]]
    admitted = []
    for protocol in primary:
        for deployment in protocol["thresholds"]:
            minimum = deployment["minimum_selected_percent"]
            if (
                deployment["selected_runs"]
                    >= admission["minimum_selected_runs"]
                and deployment["selection_precision"]
                    >= admission["minimum_selection_precision"]
                and deployment["mean_deployed_percent"]
                    >= admission["minimum_mean_deployed_percent"]
                and deployment["selected_loss_fraction"]
                    <= admission["maximum_selected_loss_fraction"]
                and minimum is not None
                and minimum >= admission["minimum_selected_percent"]
            ):
                admitted.append({
                    "model": protocol["model"],
                    **{key: value for key, value in deployment.items()
                       if key != "selected_keys"},
                })
    report["admission_pass"] = bool(admitted)
    report["admitted_settings"] = admitted
    report["rows"] = rows

    args.out.mkdir(parents=True, exist_ok=True)
    report_path = args.out / "FINAL_REPORT.json"
    quick_path = args.out / "QUICK_RESULT.json"
    report_path.write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n",
        encoding="utf-8")
    quick = {
        "completed": True,
        "definition": report["definition"],
        "scenario_exact": True,
        "pre_intervention_replay_exact": True,
        "run_count": len(rows),
        "feature_dimension": len(feature_names),
        "always_g10_baseline": report["always_g10_baseline"],
        "positive_oracle": report["positive_oracle"],
        "seed_holdout_results": [
            _compact(protocol) for protocol in protocols + consensus
            if protocol["protocol"]
            == "train_first_3_seeds_test_last_7_seeds"
        ],
        "loo_results": [
            _compact(protocol) for protocol in protocols + consensus
            if protocol["protocol"] !=
            "train_first_3_seeds_test_last_7_seeds"
        ],
        "admission_thresholds": admission,
        "admission_pass": report["admission_pass"],
        "admitted_settings": admitted,
    }
    quick_path.write_text(
        json.dumps(quick, indent=2, allow_nan=False) + "\n",
        encoding="utf-8")
    print(json.dumps(quick, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
