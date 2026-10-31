#!/usr/bin/env python3
"""LOO safety gate for the action selected by the downstream M Transformer."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


EPSILON = 1e-12


def _json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl(path: Path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _load_rows(state_dataset: Path, candidate_dataset: Path,
               transformer_root: Path):
    states = {row["state_key"]: row for row in _jsonl(state_dataset)}
    candidates = {
        (row["state_key"], int(row["candidate_id"])): row
        for row in _jsonl(candidate_dataset)
    }
    reports = sorted(transformer_root.glob("heldout_S0_*/report.json"))
    if len(reports) != 10:
        raise ValueError(
            f"expected 10 Transformer fold reports, found {len(reports)}")
    rows = []
    seen = set()
    for path in reports:
        report = _json(path)
        heldout = report["heldout_instance"]
        if report.get("training_objective") != "regression_sign":
            raise ValueError(
                f"{path} is not the retained V2 regression_sign selector")
        for prediction in report["validation"]:
            key = prediction["state_key"]
            if key in seen:
                raise ValueError(f"duplicate Transformer state: {key}")
            seen.add(key)
            if prediction["instance"] != heldout:
                raise ValueError(f"validation fold mismatch for {key}")
            if key not in states:
                raise ValueError(f"state gate features missing for {key}")
            candidate_id = int(prediction["selected_candidate_index"])
            candidate_key = (key, candidate_id)
            if candidate_key not in candidates:
                raise ValueError(
                    f"selected candidate features missing for {candidate_key}")
            scores = np.asarray(prediction["scores"], dtype=float)
            if not np.isfinite(scores).all() or len(scores) < 2:
                raise ValueError(f"invalid Transformer scores for {key}")
            ordered = np.sort(scores)
            maximum = float(ordered[-1])
            second = float(ordered[-2])
            state = states[key]
            candidate = candidates[candidate_key]
            gain = float(prediction["selected_gain_percent"])
            engineered = [
                float(prediction["generation"]),
                math.log1p(len(scores)),
                maximum,
                second,
                maximum - second,
                float(scores.mean()),
                float(scores.std()),
                float(scores.min()),
                float(scores.max() - scores.min()),
                float(prediction["gate_probability"]),
                float(candidate.get("base_probability", 0.0)),
                float(candidate.get("immediate_archive_delta", 0.0)),
            ]
            features = (
                [float(value) for value in state["features"]]
                + [float(value) for value in candidate["features"]]
                + engineered
            )
            if not np.isfinite(np.asarray(features, dtype=float)).all():
                raise ValueError(f"non-finite selected-action features: {key}")
            rows.append({
                "state_key": key,
                "instance": prediction["instance"],
                "generation": int(prediction["generation"]),
                "selected_candidate_index": candidate_id,
                "selected_gain_percent": gain,
                "positive": bool(gain > EPSILON),
                "features": features,
            })
    if set(states) != seen:
        raise ValueError(
            "state-key mismatch: "
            f"missing_transformer={sorted(set(states) - seen)[:5]} "
            f"missing_state={sorted(seen - set(states))[:5]}")
    rows.sort(key=lambda row: row["state_key"])
    return rows


def _make_model(name, seed):
    if name == "balanced_logistic":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=0.1, class_weight="balanced", max_iter=5_000,
                random_state=seed))
    if name == "extra_trees":
        return ExtraTreesClassifier(
            n_estimators=500, max_depth=6, min_samples_leaf=3,
            max_features="sqrt", class_weight="balanced",
            random_state=seed, n_jobs=-1)
    raise ValueError(f"unknown model: {name}")


def _deployment(rows, probabilities, selected_indices):
    selected = [rows[index] for index in selected_indices]
    selected_gains = [row["selected_gain_percent"] for row in selected]
    per_instance = {instance: 0.0 for instance in {
        row["instance"] for row in rows}}
    for row in selected:
        per_instance[row["instance"]] += row["selected_gain_percent"]
    count = len(selected)
    return {
        "selected_states": count,
        "selected_fraction": count / len(rows),
        "selected_actions_above_base_fraction": (
            sum(value > EPSILON for value in selected_gains) / count
            if count else 0.0),
        "selected_actions_below_base_fraction": (
            sum(value < -EPSILON for value in selected_gains) / count
            if count else 0.0),
        "mean_selected_gain_percent": (
            sum(selected_gains) / count if count else 0.0),
        "mean_deployed_gain_percent": (
            sum(selected_gains) / len(rows) if count else 0.0),
        "positive_folds": sum(value > EPSILON for value in per_instance.values()),
        "minimum_selected_gain_percent": (
            min(selected_gains) if count else None),
        "maximum_selected_gain_percent": (
            max(selected_gains) if count else None),
        "selected_state_keys": [row["state_key"] for row in selected],
        "selected_probabilities": [
            float(probabilities[index]) for index in selected_indices],
    }


def _evaluate(rows, name, seed):
    instances = sorted({row["instance"] for row in rows})
    scores_by_key = {}
    folds = []
    for fold_index, heldout in enumerate(instances):
        training = [row for row in rows if row["instance"] != heldout]
        validation = [row for row in rows if row["instance"] == heldout]
        train_y = np.asarray([row["positive"] for row in training], dtype=int)
        if len(set(train_y.tolist())) != 2:
            raise ValueError(f"training fold {heldout} lacks both classes")
        model = _make_model(name, seed + fold_index)
        model.fit(
            np.asarray([row["features"] for row in training], dtype=float),
            train_y)
        probabilities = model.predict_proba(np.asarray(
            [row["features"] for row in validation], dtype=float))[:, 1]
        for row, probability in zip(validation, probabilities):
            scores_by_key[row["state_key"]] = float(probability)
        folds.append({
            "heldout_instance": heldout,
            "state_count": len(validation),
            "positive_actions": sum(row["positive"] for row in validation),
        })
    probabilities = np.asarray([
        scores_by_key[row["state_key"]] for row in rows], dtype=float)
    labels = np.asarray([row["positive"] for row in rows], dtype=bool)
    threshold_indices = np.flatnonzero(probabilities >= 0.5).tolist()
    top_fraction = []
    order = np.argsort(-probabilities, kind="stable")
    for fraction in (0.05, 0.10, 0.15, 0.20, 0.25, 0.30):
        count = max(1, math.ceil(len(rows) * fraction))
        metrics = _deployment(rows, probabilities, order[:count].tolist())
        metrics["fraction"] = fraction
        top_fraction.append(metrics)
    threshold = _deployment(rows, probabilities, threshold_indices)
    threshold["probability_threshold"] = 0.5
    return {
        "model": name,
        "loo_roc_auc": float(roc_auc_score(labels, probabilities)),
        "loo_average_precision": float(
            average_precision_score(labels, probabilities)),
        "threshold_0_5": threshold,
        "top_fraction": top_fraction,
        "folds": folds,
        "predictions": [{
            "state_key": row["state_key"],
            "instance": row["instance"],
            "selected_gain_percent": row["selected_gain_percent"],
            "positive": row["positive"],
            "probability": float(probability),
        } for row, probability in zip(rows, probabilities)],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-dataset", type=Path, required=True)
    parser.add_argument("--candidate-dataset", type=Path, required=True)
    parser.add_argument("--transformer-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=10001)
    args = parser.parse_args(argv)
    rows = _load_rows(
        args.state_dataset, args.candidate_dataset, args.transformer_root)
    positive = sum(row["positive"] for row in rows)
    report = {
        "completed": True,
        "definition": (
            "leave-one-instance-out safety gate for the actual action selected "
            "by the retained V2 M downstream Transformer"),
        "state_count": len(rows),
        "instance_count": len({row["instance"] for row in rows}),
        "positive_selected_actions": positive,
        "positive_selected_action_fraction": positive / len(rows),
        "feature_dimension": len(rows[0]["features"]),
        "models": [
            _evaluate(rows, name, args.seed + offset * 100)
            for offset, name in enumerate(
                ("balanced_logistic", "extra_trees"))
        ],
        "admission_thresholds": {
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
        deployment = model["threshold_0_5"]
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
