#!/usr/bin/env python3
"""Leave-one-instance-out state gate probe for downstream C interventions."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def _jsonl(path: Path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _selection_metrics(rows, scores, labels, fraction):
    count = max(1, math.ceil(len(rows) * fraction))
    selected = np.argsort(-scores, kind="stable")[:count]
    selected_labels = labels[selected]
    oracle = np.asarray([
        row["audited_oracle_percent"] for row in rows], dtype=float)[selected]
    positives = int(labels.sum())
    return {
        "fraction": fraction,
        "selected_states": int(count),
        "precision": float(selected_labels.mean()),
        "recall": float(selected_labels.sum() / positives) if positives else None,
        "mean_selected_oracle_percent": float(oracle.mean()),
        "median_selected_oracle_percent": float(np.median(oracle)),
        "maximum_selected_oracle_percent": float(oracle.max()),
    }


def _evaluate_label(rows, label, seed):
    instances = sorted({row["instance"] for row in rows})
    scores_by_key = {}
    folds = []
    for fold_index, heldout in enumerate(instances):
        training = [row for row in rows if row["instance"] != heldout]
        validation = [row for row in rows if row["instance"] == heldout]
        train_y = np.asarray([row[label] for row in training], dtype=int)
        if len(set(train_y.tolist())) == 2:
            model = make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    C=0.1, class_weight="balanced", max_iter=5_000,
                    random_state=seed + fold_index),
            ).fit(
                np.asarray([row["features"] for row in training], dtype=float),
                train_y)
            probabilities = model.predict_proba(np.asarray(
                [row["features"] for row in validation], dtype=float))[:, 1]
            fold_mode = "logistic"
        else:
            probabilities = np.full(
                len(validation), float(train_y[0]), dtype=float)
            fold_mode = "constant_single_class_training"
        validation_y = np.asarray([
            row[label] for row in validation], dtype=int)
        for row, score in zip(validation, probabilities):
            scores_by_key[row["state_key"]] = float(score)
        folds.append({
            "heldout_instance": heldout,
            "state_count": len(validation),
            "positive_states": int(validation_y.sum()),
            "mean_probability": float(probabilities.mean()),
            "mode": fold_mode,
        })

    ordered_scores = np.asarray([
        scores_by_key[row["state_key"]] for row in rows], dtype=float)
    labels = np.asarray([row[label] for row in rows], dtype=int)
    predictions = ordered_scores >= 0.5
    true_positive = int(np.logical_and(predictions, labels == 1).sum())
    selected = int(predictions.sum())
    positives = int(labels.sum())
    return {
        "label": label,
        "state_count": len(rows),
        "positive_states": positives,
        "prevalence": float(labels.mean()),
        "loo_roc_auc": (
            float(roc_auc_score(labels, ordered_scores))
            if len(set(labels.tolist())) == 2 else None),
        "loo_average_precision": (
            float(average_precision_score(labels, ordered_scores))
            if labels.sum() > 0 else None),
        "threshold_0_5": {
            "selected_states": selected,
            "precision": true_positive / selected if selected else None,
            "recall": true_positive / positives if positives else None,
        },
        "top_fraction": [
            _selection_metrics(rows, ordered_scores, labels, fraction)
            for fraction in (0.10, 0.20, 0.30)
        ],
        "folds": folds,
        "predictions": [{
            "state_key": row["state_key"],
            "instance": row["instance"],
            "generation": row["generation"],
            "oracle_percent": row["audited_oracle_percent"],
            "label": bool(row[label]),
            "probability": float(score),
        } for row, score in zip(rows, ordered_scores)],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=9941)
    parser.add_argument(
        "--labels", nargs="+",
        choices=("positive_any", "positive_5", "positive_10"),
        default=("positive_any", "positive_5", "positive_10"))
    args = parser.parse_args(argv)
    rows = _jsonl(args.dataset)
    if len({row["instance"] for row in rows}) < 2:
        parser.error("at least two instances are required")
    dimensions = {len(row["features"]) for row in rows}
    if len(dimensions) != 1:
        parser.error(f"inconsistent feature dimensions: {dimensions}")
    report = {
        "completed": True,
        "dataset": str(args.dataset.resolve()),
        "model": "balanced_logistic_state_gate",
        "validation": "leave-one-instance-out",
        "feature_dimension": dimensions.pop(),
        "labels": {
            label: _evaluate_label(rows, label, args.seed + offset * 100)
            for offset, label in enumerate(args.labels)
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n",
        encoding="utf-8")
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
