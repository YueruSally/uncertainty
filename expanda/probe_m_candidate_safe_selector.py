#!/usr/bin/env python3
"""LOO candidate-level safe selector for downstream M interventions."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


EPSILON = 1e-12


def _jsonl(path: Path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _load(path: Path):
    rows = _jsonl(path)
    if not rows:
        raise ValueError(f"empty candidate dataset: {path}")
    dimensions = {len(row["features"]) for row in rows}
    if len(dimensions) != 1:
        raise ValueError(f"inconsistent feature dimensions: {dimensions}")
    keys = [(row["state_key"], int(row["candidate_id"])) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("candidate keys are not unique")
    for row in rows:
        row["candidate_id"] = int(row["candidate_id"])
        row["gain"] = float(row["final_percent"])
        row["positive"] = bool(row["gain"] > EPSILON)
    rows.sort(key=lambda row: (row["state_key"], row["candidate_id"]))
    return rows, dimensions.pop()


def _classifier(name, seed):
    if name == "balanced_logistic":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=0.1, class_weight="balanced", max_iter=5_000,
                random_state=seed))
    if name in ("extra_trees", "two_stage_extra_trees"):
        return ExtraTreesClassifier(
            n_estimators=500, max_depth=8, min_samples_leaf=3,
            max_features="sqrt", class_weight="balanced",
            random_state=seed, n_jobs=-1)
    raise ValueError(f"unknown selector: {name}")


def _regressor(seed):
    return ExtraTreesRegressor(
        n_estimators=500, max_depth=8, min_samples_leaf=3,
        max_features="sqrt", random_state=seed, n_jobs=-1)


def _selected_states(rows, probability_by_key, score_by_key):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["state_key"]].append(row)
    selected = []
    for state_key in sorted(grouped):
        candidates = grouped[state_key]
        chosen = max(
            candidates,
            key=lambda row: (
                score_by_key[(state_key, row["candidate_id"])],
                probability_by_key[(state_key, row["candidate_id"])],
                -row["candidate_id"]))
        key = (state_key, chosen["candidate_id"])
        selected.append({
            "state_key": state_key,
            "instance": chosen["instance"],
            "candidate_id": chosen["candidate_id"],
            "gain_percent": chosen["gain"],
            "positive_probability": probability_by_key[key],
            "ranking_score": score_by_key[key],
            "oracle_percent": max(row["gain"] for row in candidates),
        })
    return selected


def _deployment(selected, threshold):
    deployed = [
        row for row in selected
        if row["positive_probability"] >= threshold]
    gains = [row["gain_percent"] for row in deployed]
    per_instance = {instance: 0.0 for instance in {
        row["instance"] for row in selected}}
    for row in deployed:
        per_instance[row["instance"]] += row["gain_percent"]
    return {
        "positive_probability_threshold": threshold,
        "selected_states": len(deployed),
        "selected_fraction": len(deployed) / len(selected),
        "selected_actions_above_base_fraction": (
            sum(gain > EPSILON for gain in gains) / len(gains)
            if gains else 0.0),
        "selected_actions_below_base_fraction": (
            sum(gain < -EPSILON for gain in gains) / len(gains)
            if gains else 0.0),
        "mean_selected_gain_percent": (
            sum(gains) / len(gains) if gains else 0.0),
        "mean_deployed_gain_percent": (
            sum(gains) / len(selected) if gains else 0.0),
        "positive_folds": sum(value > EPSILON for value in per_instance.values()),
        "minimum_selected_gain_percent": min(gains) if gains else None,
        "maximum_selected_gain_percent": max(gains) if gains else None,
        "selected_state_keys": [row["state_key"] for row in deployed],
    }


def _evaluate(rows, name, seed):
    instances = sorted({row["instance"] for row in rows})
    probability_by_key = {}
    score_by_key = {}
    fold_rows = []
    for fold_index, heldout in enumerate(instances):
        training = [row for row in rows if row["instance"] != heldout]
        validation = [row for row in rows if row["instance"] == heldout]
        train_x = np.asarray([row["features"] for row in training], dtype=float)
        train_y = np.asarray([row["positive"] for row in training], dtype=int)
        validation_x = np.asarray(
            [row["features"] for row in validation], dtype=float)
        if len(set(train_y.tolist())) != 2:
            raise ValueError(f"training fold {heldout} lacks both classes")
        classifier = _classifier(name, seed + fold_index)
        classifier.fit(train_x, train_y)
        probabilities = classifier.predict_proba(validation_x)[:, 1]
        if name == "two_stage_extra_trees":
            positive_training = [row for row in training if row["positive"]]
            regressor = _regressor(seed + 10_000 + fold_index)
            regressor.fit(
                np.asarray([
                    row["features"] for row in positive_training], dtype=float),
                np.asarray([
                    row["gain"] for row in positive_training], dtype=float))
            magnitudes = np.clip(regressor.predict(validation_x), 0.0, None)
            scores = probabilities * magnitudes
        else:
            scores = probabilities
        for row, probability, score in zip(
                validation, probabilities, scores):
            key = (row["state_key"], row["candidate_id"])
            probability_by_key[key] = float(probability)
            score_by_key[key] = float(score)
        fold_rows.append({
            "heldout_instance": heldout,
            "candidate_count": len(validation),
            "positive_candidates": sum(row["positive"] for row in validation),
        })
    ordered_probabilities = np.asarray([
        probability_by_key[(row["state_key"], row["candidate_id"])]
        for row in rows], dtype=float)
    labels = np.asarray([row["positive"] for row in rows], dtype=bool)
    selected = _selected_states(rows, probability_by_key, score_by_key)
    ungated_gains = [row["gain_percent"] for row in selected]
    return {
        "model": name,
        "candidate_loo_roc_auc": float(
            roc_auc_score(labels, ordered_probabilities)),
        "candidate_loo_average_precision": float(
            average_precision_score(labels, ordered_probabilities)),
        "ungated_selector": {
            "state_count": len(selected),
            "top_1_accuracy": (
                sum(row["gain_percent"] >= row["oracle_percent"] - EPSILON
                    for row in selected) / len(selected)),
            "selected_actions_above_base_fraction": (
                sum(gain > EPSILON for gain in ungated_gains) / len(selected)),
            "mean_selected_gain_percent": sum(ungated_gains) / len(selected),
            "minimum_selected_gain_percent": min(ungated_gains),
            "maximum_selected_gain_percent": max(ungated_gains),
        },
        "deployments": [
            _deployment(selected, threshold)
            for threshold in (0.50, 0.60, 0.70, 0.80)
        ],
        "folds": fold_rows,
        "selected_actions": selected,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=10021)
    args = parser.parse_args(argv)
    rows, feature_dimension = _load(args.dataset)
    report = {
        "completed": True,
        "definition": (
            "leave-one-instance-out candidate-level selector trained to "
            "choose a profitable M action or retain exact Base"),
        "candidate_count": len(rows),
        "state_count": len({row["state_key"] for row in rows}),
        "instance_count": len({row["instance"] for row in rows}),
        "feature_dimension": feature_dimension,
        "positive_candidate_count": sum(row["positive"] for row in rows),
        "models": [
            _evaluate(rows, name, args.seed + offset * 100)
            for offset, name in enumerate((
                "balanced_logistic", "extra_trees",
                "two_stage_extra_trees"))
        ],
        "admission_thresholds": {
            "deployment_probability_threshold": 0.60,
            "selected_states_minimum": 12,
            "selected_actions_above_base_fraction": 0.60,
            "mean_deployed_gain_percent": 1.0,
            "positive_folds": 7,
            "minimum_selected_gain_percent": -5.0,
        },
    }
    thresholds = report["admission_thresholds"]
    admitted = []
    for model in report["models"]:
        deployment = next(
            row for row in model["deployments"]
            if row["positive_probability_threshold"]
            == thresholds["deployment_probability_threshold"])
        if (
            deployment["selected_states"] >= thresholds["selected_states_minimum"]
            and deployment["selected_actions_above_base_fraction"]
            >= thresholds["selected_actions_above_base_fraction"]
            and deployment["mean_deployed_gain_percent"]
            >= thresholds["mean_deployed_gain_percent"]
            and deployment["positive_folds"] >= thresholds["positive_folds"]
            and deployment["minimum_selected_gain_percent"] is not None
            and deployment["minimum_selected_gain_percent"]
            >= thresholds["minimum_selected_gain_percent"]
        ):
            admitted.append(model["model"])
    report["admission_pass"] = bool(admitted)
    report["admitted_models"] = admitted
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n",
        encoding="utf-8")
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
