#!/usr/bin/env python3
"""Aggregate out-of-instance M downstream Transformer predictions."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


EPSILON = 1e-12


def _average_ranks(values):
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def _auc(labels, scores):
    labels = np.asarray(labels, dtype=bool)
    positive, negative = labels.sum(), (~labels).sum()
    if positive == 0 or negative == 0:
        return None
    ranks = _average_ranks(scores)
    return float((ranks[labels].sum() - positive * (positive + 1) / 2)
                 / (positive * negative))


def _average_precision(labels, scores):
    labels = np.asarray(labels, dtype=bool)
    if labels.sum() == 0:
        return None
    order = np.argsort(-np.asarray(scores, dtype=float), kind="mergesort")
    ordered = labels[order]
    precision = np.cumsum(ordered) / np.arange(1, len(ordered) + 1)
    return float(precision[ordered].mean())


def _pair_metrics(rows):
    correct = total = 0
    correlations = []
    top_matches = 0
    for row in rows:
        labels = np.asarray(row["labels"], dtype=float)
        scores = np.asarray(row["scores"], dtype=float)
        for left in range(len(labels)):
            for right in range(left + 1, len(labels)):
                label_difference = labels[left] - labels[right]
                if abs(label_difference) <= EPSILON:
                    continue
                score_difference = scores[left] - scores[right]
                correct += int(score_difference * label_difference > 0)
                total += 1
        label_rank = _average_ranks(labels)
        score_rank = _average_ranks(scores)
        if np.std(label_rank) > 0 and np.std(score_rank) > 0:
            correlations.append(float(np.corrcoef(label_rank, score_rank)[0, 1]))
        selected = int(row["selected_position"])
        top_matches += int(labels[selected] >= labels.max() - EPSILON)
    return {
        "pairwise_accuracy": correct / total if total else None,
        "mean_state_spearman": (
            float(np.mean(correlations)) if correlations else None),
        "top_1_accuracy": top_matches / len(rows),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    reports = sorted(args.root.glob("heldout_S0_*/report.json"))
    if len(reports) != 10:
        parser.error(f"expected 10 completed folds, found {len(reports)}")
    rows = []
    per_fold = []
    training_objectives = set()
    for path in reports:
        report = json.loads(path.read_text(encoding="utf-8"))
        training_objectives.add(report.get(
            "training_objective", "pairwise_gate"))
        fold_rows = report["validation"]
        metrics = _pair_metrics(fold_rows)
        metrics.update({
            "fold": report["heldout_instance"],
            "state_count": len(fold_rows),
            "mean_selected_gain_percent": float(np.mean([
                row["selected_gain_percent"] for row in fold_rows])),
            "mean_deployed_gain_percent": float(np.mean([
                row["deployed_gain_percent"] for row in fold_rows])),
            "gate_selected_states": sum(
                row["gate_selected"] for row in fold_rows),
        })
        per_fold.append(metrics)
        rows.extend(fold_rows)
    if len(training_objectives) != 1:
        parser.error(
            f"folds use different training objectives: {training_objectives}")
    training_objective = training_objectives.pop()
    pair = _pair_metrics(rows)
    gate_labels = [row["gate_target"] for row in rows]
    gate_scores = [row["gate_probability"] for row in rows]
    top_count = max(1, math.ceil(0.20 * len(rows)))
    top_rows = sorted(
        rows, key=lambda row: row["gate_probability"], reverse=True)[:top_count]
    deployed = [row for row in rows if row["gate_selected"]]
    positive_folds = sum(
        row["mean_deployed_gain_percent"] > 0 for row in per_fold)
    selected_positive_fraction = (
        sum(row["selected_gain_percent"] > 0 for row in deployed) / len(deployed)
        if deployed else 0.0)
    report = {
        "completed": True,
        "definition": (
            "leave-one-instance-out actual path-v4 Transformer trained on "
            "single-intervention final CCP HV percent labels"),
        "training_objective": training_objective,
        "state_count": len(rows),
        "fold_count": len(per_fold),
        "candidate_ranker": {
            **pair,
            "mean_selected_gain_percent": float(np.mean([
                row["selected_gain_percent"] for row in rows])),
        },
        "gate_positive_10": {
            "roc_auc": _auc(gate_labels, gate_scores),
            "average_precision": _average_precision(gate_labels, gate_scores),
            "top_20": {
                "selected_states": len(top_rows),
                "precision": sum(row["gate_target"] for row in top_rows) / len(top_rows),
                "mean_selected_oracle_percent": float(np.mean([
                    row["oracle_percent"] for row in top_rows])),
                "median_selected_oracle_percent": float(np.median([
                    row["oracle_percent"] for row in top_rows])),
                "maximum_selected_oracle_percent": max(
                    row["oracle_percent"] for row in top_rows),
            },
        },
        "conservative_policy": {
            "selected_state_count": len(deployed),
            "selected_state_fraction": len(deployed) / len(rows),
            "selected_actions_above_base_fraction": selected_positive_fraction,
            "mean_deployed_gain_percent": float(np.mean([
                row["deployed_gain_percent"] for row in rows])),
            "median_deployed_gain_percent": float(np.median([
                row["deployed_gain_percent"] for row in rows])),
            "positive_folds": positive_folds,
        },
        "per_fold": per_fold,
    }
    report["admission_thresholds"] = {
        "gate_top_20_precision": 0.60,
        "gate_top_20_mean_oracle_percent": 10.0,
        "candidate_pairwise_accuracy": 0.60,
        "selected_actions_above_base_fraction": 0.60,
        "positive_folds": 7,
        "mean_deployed_gain_percent": 1.0,
    }
    thresholds = report["admission_thresholds"]
    report["admission_pass"] = bool(
        report["gate_positive_10"]["top_20"]["precision"]
        >= thresholds["gate_top_20_precision"]
        and report["gate_positive_10"]["top_20"]["mean_selected_oracle_percent"]
        >= thresholds["gate_top_20_mean_oracle_percent"]
        and report["candidate_ranker"]["pairwise_accuracy"]
        >= thresholds["candidate_pairwise_accuracy"]
        and selected_positive_fraction
        >= thresholds["selected_actions_above_base_fraction"]
        and positive_folds >= thresholds["positive_folds"]
        and report["conservative_policy"]["mean_deployed_gain_percent"]
        >= thresholds["mean_deployed_gain_percent"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n",
        encoding="utf-8")
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
