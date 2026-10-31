#!/usr/bin/env python3
"""LOO consensus gate for late-stage immediate-positive M candidates."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


EPSILON = 1e-12


def _jsonl(path: Path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _mean(values):
    return float(np.mean(values)) if values else None


def _positive_probability(model, matrix):
    classes = list(model.classes_)
    if 1 not in classes:
        return np.zeros(len(matrix), dtype=float)
    return model.predict_proba(matrix)[:, classes.index(1)]


def _fit(training, seed):
    matrix = np.asarray([row["features"] for row in training], dtype=float)
    labels = np.asarray([
        float(row["immediate_archive_delta"]) > EPSILON
        for row in training
    ], dtype=int)
    if len(set(labels.tolist())) != 2:
        raise ValueError("consensus training data require both immediate classes")
    linear = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=0.1, class_weight="balanced", max_iter=5_000,
            random_state=seed),
    ).fit(matrix, labels)
    trees = ExtraTreesClassifier(
        n_estimators=500, min_samples_leaf=2, max_features="sqrt",
        class_weight="balanced", n_jobs=-1, random_state=seed,
    ).fit(matrix, labels)
    return linear, trees


def _fold_summary(rows, heldout):
    selected = [row for row in rows if row["intervene"]]
    return {
        "heldout_instance": heldout,
        "state_count": len(rows),
        "selected_state_count": len(selected),
        "mean_deployed_final_percent": _mean([
            row["deployed_final_percent"] for row in rows]),
        "mean_selected_final_percent": _mean([
            row["selected_final_percent"] for row in selected]),
        "selected_final_positive_fraction": _mean([
            row["selected_final_percent"] > EPSILON for row in selected]),
        "selected_final_negative_fraction": _mean([
            row["selected_final_percent"] < -EPSILON for row in selected]),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--probability-threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=9997)
    args = parser.parse_args(argv)
    if not 0.0 <= args.probability_threshold <= 1.0:
        parser.error("probability threshold must be in [0, 1]")

    rows = _jsonl(args.dataset)
    if any(int(row["generation"]) < 10 for row in rows):
        parser.error("consensus probe accepts only generation >= 10")
    instances = sorted({row["instance"] for row in rows})
    if len(instances) != 10:
        parser.error(f"expected 10 instances, found {len(instances)}")
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["state_key"]].append(row)
    if len(grouped) != 30 or any(len(value) != 4 for value in grouped.values()):
        parser.error("expected 30 late states with four candidates each")

    predictions = []
    folds = []
    for fold_index, heldout in enumerate(instances):
        training = [row for row in rows if row["instance"] != heldout]
        validation_states = {
            key: sorted(value, key=lambda row: int(row["candidate_id"]))
            for key, value in grouped.items()
            if value[0]["instance"] == heldout
        }
        linear, trees = _fit(training, args.seed + fold_index)
        fold_rows = []
        for state_key, state_rows in sorted(validation_states.items()):
            matrix = np.asarray([
                row["features"] for row in state_rows], dtype=float)
            linear_probability = _positive_probability(linear, matrix)
            tree_probability = _positive_probability(trees, matrix)
            linear_position = int(np.argmax(linear_probability))
            tree_position = int(np.argmax(tree_probability))
            agree = linear_position == tree_position
            selected_position = linear_position if agree else None
            confidence_pass = bool(
                agree
                and linear_probability[selected_position]
                >= args.probability_threshold
                and tree_probability[selected_position]
                >= args.probability_threshold
            )
            selected = (
                state_rows[selected_position] if confidence_pass else None)
            final_percent = (
                float(selected["continuation_mean_final_percent"])
                if selected is not None else 0.0)
            record = {
                "state_key": state_key,
                "instance": heldout,
                "generation": int(state_rows[0]["generation"]),
                "linear_candidate_id": int(
                    state_rows[linear_position]["candidate_id"]),
                "tree_candidate_id": int(
                    state_rows[tree_position]["candidate_id"]),
                "models_agree": agree,
                "linear_probability": float(
                    linear_probability[linear_position]),
                "tree_probability": float(tree_probability[tree_position]),
                "intervene": confidence_pass,
                "selected_candidate_id": (
                    int(selected["candidate_id"])
                    if selected is not None else None),
                "selected_immediate_positive": (
                    float(selected["immediate_archive_delta"]) > EPSILON
                    if selected is not None else None),
                "selected_final_percent": (
                    final_percent if selected is not None else None),
                "selected_safe_score": (
                    float(selected[
                        "continuation_risk_adjusted_final_percent"])
                    if selected is not None else None),
                "deployed_final_percent": final_percent,
            }
            predictions.append(record)
            fold_rows.append(record)
        folds.append(_fold_summary(fold_rows, heldout))

    selected = [row for row in predictions if row["intervene"]]
    selected_final = [row["selected_final_percent"] for row in selected]
    selected_safe = [row["selected_safe_score"] for row in selected]
    report = {
        "completed": True,
        "definition": (
            "late-stage LOO consensus of balanced logistic and Extra Trees "
            "immediate-positive classifiers; disagreement or confidence "
            "below threshold falls back to exact Base"),
        "dataset": str(args.dataset.resolve()),
        "probability_threshold": args.probability_threshold,
        "state_count": len(predictions),
        "selected_state_count": len(selected),
        "selected_state_fraction": len(selected) / len(predictions),
        "model_agreement_fraction": _mean([
            row["models_agree"] for row in predictions]),
        "selected_immediate_precision": _mean([
            row["selected_immediate_positive"] for row in selected]),
        "mean_selected_final_percent": _mean(selected_final),
        "median_selected_final_percent": (
            float(np.median(selected_final)) if selected_final else None),
        "minimum_selected_final_percent": (
            min(selected_final) if selected_final else None),
        "maximum_selected_final_percent": (
            max(selected_final) if selected_final else None),
        "selected_final_positive_fraction": _mean([
            value > EPSILON for value in selected_final]),
        "selected_final_negative_fraction": _mean([
            value < -EPSILON for value in selected_final]),
        "selected_safe_positive_fraction": _mean([
            value > EPSILON for value in selected_safe]),
        "mean_deployed_final_percent": _mean([
            row["deployed_final_percent"] for row in predictions]),
        "positive_deployed_folds": sum(
            (fold["mean_deployed_final_percent"] or 0.0) > 0.0
            for fold in folds),
        "folds": folds,
        "predictions": predictions,
        "admission_thresholds": {
            "minimum_selected_states": 6,
            "selected_immediate_precision": 0.70,
            "mean_selected_final_percent": 1.0,
            "selected_final_positive_fraction": 0.60,
            "selected_final_negative_fraction": 0.20,
            "positive_deployed_folds": 7,
        },
    }
    thresholds = report["admission_thresholds"]
    report["admission_pass"] = bool(
        report["selected_state_count"] >= thresholds["minimum_selected_states"]
        and report["selected_immediate_precision"] is not None
        and report["selected_immediate_precision"] >=
        thresholds["selected_immediate_precision"]
        and report["mean_selected_final_percent"] is not None
        and report["mean_selected_final_percent"] >=
        thresholds["mean_selected_final_percent"]
        and report["selected_final_positive_fraction"] is not None
        and report["selected_final_positive_fraction"] >=
        thresholds["selected_final_positive_fraction"]
        and report["selected_final_negative_fraction"] is not None
        and report["selected_final_negative_fraction"] <=
        thresholds["selected_final_negative_fraction"]
        and report["positive_deployed_folds"] >=
        thresholds["positive_deployed_folds"]
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n",
        encoding="utf-8")
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
