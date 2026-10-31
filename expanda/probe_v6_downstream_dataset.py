#!/usr/bin/env python3
"""Leave-one-instance-out probe for configurable mutation advantages."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


EPSILON = 1e-12
TARGET = "plus_1_generation_archive_advantage"
FINAL = "final_archive_advantage"


def _jsonl(path: Path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _mean(values):
    return float(np.mean(values)) if values else None


def _average_ranks(values):
    values = np.asarray(values, dtype=float)
    result = np.zeros(len(values), dtype=float)
    order = np.argsort(values, kind="stable")
    start = 0
    while start < len(order):
        end = start + 1
        while (end < len(order)
               and abs(values[order[end]] - values[order[start]]) <= EPSILON):
            end += 1
        result[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return result


def _correlation(left, right):
    if len(left) < 2:
        return None
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    left -= left.mean()
    right -= right.mean()
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= EPSILON:
        return None
    return float(left.dot(right) / denominator)


def _conditional(values):
    values = np.maximum(0.0, np.asarray(values, dtype=float))
    mass = float(values.sum())
    return (values / mass if mass > 0.0
            else np.full(len(values), 1.0 / len(values)))


def _model_probabilities(scores, base, alpha, temperature):
    ranks = _average_ranks(scores)
    if len(ranks) > 1:
        ranks /= len(ranks) - 1
    logits = ranks / temperature
    logits -= logits.max()
    model = np.exp(logits)
    model /= model.sum()
    return (1.0 - alpha) * _conditional(base) + alpha * model


def _pairwise_training(rows):
    by_state = defaultdict(list)
    for row in rows:
        by_state[row["state_key"]].append(row)
    differences, labels = [], []
    informative_states = 0
    for state_rows in by_state.values():
        values = [float(row[TARGET]) for row in state_rows]
        if max(values) - min(values) <= EPSILON:
            continue
        informative_states += 1
        for left_index, left in enumerate(state_rows):
            for right in state_rows[left_index + 1:]:
                difference = float(left[TARGET]) - float(right[TARGET])
                if abs(difference) <= EPSILON:
                    continue
                feature_difference = (
                    np.asarray(left["features"], dtype=float)
                    - np.asarray(right["features"], dtype=float))
                label = int(difference > 0.0)
                differences.extend([feature_difference, -feature_difference])
                labels.extend([label, 1 - label])
    if not differences:
        raise ValueError("no unequal training candidate pairs")
    return (
        np.stack(differences), np.asarray(labels, dtype=int),
        informative_states,
    )


def _fit(rows, seed):
    pair_x, pair_y, informative_states = _pairwise_training(rows)
    scaler = StandardScaler().fit(pair_x)
    model = LogisticRegression(
        C=0.1, max_iter=2_000, random_state=seed).fit(
            scaler.transform(pair_x), pair_y)
    weights = model.coef_[0] / scaler.scale_
    return weights, {
        "training_candidate_count": len(rows),
        "training_state_count": len({row["state_key"] for row in rows}),
        "informative_training_state_count": informative_states,
        "training_pair_count": int(len(pair_y) // 2),
    }


def _fit_mlp(rows, seed):
    by_state = defaultdict(list)
    for row in rows:
        by_state[row["state_key"]].append(row)
    selected = []
    targets = []
    informative_states = 0
    pair_count = 0
    for state_rows in by_state.values():
        values = np.asarray([row[TARGET] for row in state_rows], dtype=float)
        span = float(values.max() - values.min())
        if span <= EPSILON:
            continue
        informative_states += 1
        pair_count += sum(
            abs(values[left] - values[right]) > EPSILON
            for left in range(len(values))
            for right in range(left + 1, len(values)))
        # A within-state percentile target avoids letting large-HV instances
        # dominate while retaining the state features needed for interactions.
        ranks = _average_ranks(values)
        ranks /= max(1, len(values) - 1)
        selected.extend(state_rows)
        targets.extend(ranks.tolist())
    if len(selected) < 8:
        raise ValueError("too few informative candidates for MLP probe")
    matrix = np.stack([
        np.asarray(row["features"], dtype=float) for row in selected])
    model = make_pipeline(
        StandardScaler(),
        MLPRegressor(
            hidden_layer_sizes=(64, 32), activation="relu",
            alpha=0.01, learning_rate_init=0.001,
            max_iter=2_000, early_stopping=len(selected) >= 40,
            validation_fraction=0.15, n_iter_no_change=30,
            random_state=seed),
    ).fit(matrix, np.asarray(targets, dtype=float))
    return model, {
        "training_candidate_count": len(rows),
        "informative_training_candidate_count": len(selected),
        "training_state_count": len(by_state),
        "informative_training_state_count": informative_states,
        "training_pair_count": int(pair_count),
    }


def _fit_extra_trees(rows, seed):
    by_state = defaultdict(list)
    for row in rows:
        by_state[row["state_key"]].append(row)
    selected = []
    targets = []
    informative_states = 0
    pair_count = 0
    for state_rows in by_state.values():
        values = np.asarray([row[TARGET] for row in state_rows], dtype=float)
        if float(values.max() - values.min()) <= EPSILON:
            continue
        informative_states += 1
        pair_count += sum(
            abs(values[left] - values[right]) > EPSILON
            for left in range(len(values))
            for right in range(left + 1, len(values)))
        ranks = _average_ranks(values)
        ranks /= max(1, len(values) - 1)
        selected.extend(state_rows)
        targets.extend(ranks.tolist())
    if len(selected) < 8:
        raise ValueError("too few informative candidates for Extra Trees probe")
    matrix = np.stack([
        np.asarray(row["features"], dtype=float) for row in selected])
    model = ExtraTreesRegressor(
        n_estimators=500, min_samples_leaf=2, max_features="sqrt",
        n_jobs=-1, random_state=seed).fit(
            matrix, np.asarray(targets, dtype=float))
    return model, {
        "training_candidate_count": len(rows),
        "informative_training_candidate_count": len(selected),
        "training_state_count": len(by_state),
        "informative_training_state_count": informative_states,
        "training_pair_count": int(pair_count),
    }


def _evaluate(rows, scorer, alpha, temperature):
    by_state = defaultdict(list)
    for row in rows:
        by_state[row["state_key"]].append(row)
    state_results = []
    total_pairs = correct_pairs = 0
    for state_key, state_rows in sorted(by_state.items()):
        matrix = np.stack([
            np.asarray(row["features"], dtype=float) for row in state_rows])
        scores = (
            scorer.predict(matrix)
            if hasattr(scorer, "predict") else matrix.dot(scorer))
        targets = np.asarray([row[TARGET] for row in state_rows], dtype=float)
        finals = np.asarray([row[FINAL] for row in state_rows], dtype=float)
        base = _conditional([row["base_probability"] for row in state_rows])
        proposal = _conditional([
            row["proposal_probability"] for row in state_rows])
        model = _model_probabilities(scores, base, alpha, temperature)
        informative = float(targets.max() - targets.min()) > EPSILON
        final_informative = float(finals.max() - finals.min()) > EPSILON
        pairs = correct = 0
        for left in range(len(state_rows)):
            for right in range(left + 1, len(state_rows)):
                target_difference = targets[left] - targets[right]
                if abs(target_difference) <= EPSILON:
                    continue
                pairs += 1
                score_difference = scores[left] - scores[right]
                if score_difference * target_difference > 0.0:
                    correct += 1
                elif abs(score_difference) <= EPSILON:
                    correct += 0.5
        total_pairs += pairs
        correct_pairs += correct
        model_top = np.flatnonzero(
            np.abs(model - model.max()) <= EPSILON).tolist()
        target_top = set(np.flatnonzero(
            np.abs(targets - targets.max()) <= EPSILON).tolist())
        final_top = set(np.flatnonzero(
            np.abs(finals - finals.max()) <= EPSILON).tolist())
        base_target = float(base.dot(targets))
        proposal_target = float(proposal.dot(targets))
        model_target = float(model.dot(targets))
        base_final = float(base.dot(finals))
        proposal_final = float(proposal.dot(finals))
        model_final = float(model.dot(finals))
        model_top_target = float(np.mean(targets[model_top]))
        model_top_final = float(np.mean(finals[model_top]))
        oracle_gap = float(targets.max()) - base_target
        state_results.append({
            "state_key": state_key,
            "instance": state_rows[0]["instance"],
            "candidate_count": len(state_rows),
            "informative": informative,
            "final_informative": final_informative,
            "pair_count": pairs,
            "pairwise_accuracy": correct / pairs if pairs else None,
            "spearman": _correlation(_average_ranks(scores),
                                      _average_ranks(targets)),
            "model_top_matches_target": any(
                index in target_top for index in model_top),
            "model_top_matches_final": any(
                index in final_top for index in model_top),
            "model_top_target": model_top_target,
            "model_top_final": model_top_final,
            "model_top_target_positive": model_top_target > EPSILON,
            "model_top_final_positive": model_top_final > EPSILON,
            "base_expected_target": base_target,
            "proposal_expected_target": proposal_target,
            "model_expected_target": model_target,
            "model_minus_base_expected_target": model_target - base_target,
            "proposal_minus_base_expected_target": proposal_target - base_target,
            "base_expected_final": base_final,
            "proposal_expected_final": proposal_final,
            "model_expected_final": model_final,
            "model_minus_base_expected_final": model_final - base_final,
            "proposal_minus_base_expected_final": proposal_final - base_final,
            "oracle_gap_fraction_captured": (
                (model_target - base_target) / oracle_gap
                if oracle_gap > EPSILON else None),
        })
    informative = [row for row in state_results if row["informative"]]
    final_informative = [
        row for row in state_results if row["final_informative"]]
    return {
        "state_count": len(state_results),
        "informative_state_count": len(informative),
        "final_informative_state_count": len(final_informative),
        "candidate_count": len(rows),
        "pair_count": total_pairs,
        "pairwise_accuracy": (
            correct_pairs / total_pairs if total_pairs else None),
        "mean_state_spearman": _mean([
            row["spearman"] for row in informative
            if row["spearman"] is not None]),
        "target_top_1_accuracy": _mean([
            row["model_top_matches_target"] for row in informative]),
        "final_top_1_accuracy": _mean([
            row["model_top_matches_final"] for row in final_informative]),
        "mean_model_top_target": _mean([
            row["model_top_target"] for row in state_results]),
        "mean_model_top_final": _mean([
            row["model_top_final"] for row in state_results]),
        "model_top_target_positive_fraction": _mean([
            row["model_top_target_positive"] for row in state_results]),
        "model_top_final_positive_fraction": _mean([
            row["model_top_final_positive"] for row in state_results]),
        "mean_model_minus_base_expected_target": _mean([
            row["model_minus_base_expected_target"] for row in informative]),
        "mean_proposal_minus_base_expected_target": _mean([
            row["proposal_minus_base_expected_target"] for row in informative]),
        "states_model_above_base_target_fraction": _mean([
            row["model_minus_base_expected_target"] > EPSILON
            for row in informative]),
        "mean_model_minus_base_expected_final": _mean([
            row["model_minus_base_expected_final"] for row in final_informative]),
        "mean_proposal_minus_base_expected_final": _mean([
            row["proposal_minus_base_expected_final"] for row in final_informative]),
        "states_model_above_base_final_fraction": _mean([
            row["model_minus_base_expected_final"] > EPSILON
            for row in final_informative]),
        "mean_oracle_gap_fraction_captured": _mean([
            row["oracle_gap_fraction_captured"] for row in informative
            if row["oracle_gap_fraction_captured"] is not None]),
        "per_state": state_results,
    }


def _weighted(folds, field, weight="informative_state_count"):
    selected = [
        (fold["validation"].get(field), fold["validation"].get(weight, 0))
        for fold in folds
    ]
    selected = [(value, count) for value, count in selected
                if value is not None and count > 0]
    total = sum(count for _, count in selected)
    return (sum(value * count for value, count in selected) / total
            if total else None)


def main(argv=None):
    global TARGET, FINAL
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--temperature", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=9701)
    parser.add_argument(
        "--target-field", default=TARGET,
        help="Numeric candidate field used for training and target metrics.")
    parser.add_argument(
        "--final-field", default=FINAL,
        help="Numeric candidate field reserved for independent final metrics.")
    parser.add_argument(
        "--model", choices=("pairwise_linear", "small_mlp", "extra_trees"),
        default="pairwise_linear")
    args = parser.parse_args(argv)
    TARGET = args.target_field
    FINAL = args.final_field
    if not 0.0 <= args.alpha <= 1.0 or args.temperature <= 0.0:
        parser.error("invalid alpha or temperature")
    rows = _jsonl(args.dataset)
    missing_fields = [
        row.get("state_key", "<unknown>") for row in rows
        if row.get(TARGET) is None or row.get(FINAL) is None
    ]
    if missing_fields:
        parser.error(
            f"target/final fields are missing in {len(missing_fields)} rows; "
            f"first={missing_fields[0]}")
    instances = sorted({row["instance"] for row in rows})
    if len(instances) < 2:
        parser.error("at least two instances are required")
    dimensions = {len(row["features"]) for row in rows}
    if len(dimensions) != 1:
        parser.error(f"inconsistent feature dimensions: {dimensions}")

    folds = []
    for fold_index, heldout in enumerate(instances):
        training = [row for row in rows if row["instance"] != heldout]
        validation = [row for row in rows if row["instance"] == heldout]
        if args.model == "small_mlp":
            scorer, training_summary = _fit_mlp(
                training, args.seed + fold_index)
        elif args.model == "extra_trees":
            scorer, training_summary = _fit_extra_trees(
                training, args.seed + fold_index)
        else:
            scorer, training_summary = _fit(
                training, args.seed + fold_index)
        folds.append({
            "heldout_instance": heldout,
            **training_summary,
            "validation": _evaluate(
                validation, scorer, args.alpha, args.temperature),
        })
    report = {
        "completed": True,
        "dataset": str(args.dataset.resolve()),
        "target": TARGET,
        "final_field": FINAL,
        "model": (
            "small_mlp_state_candidate_score"
            if args.model == "small_mlp"
            else "extra_trees_state_candidate_score"
            if args.model == "extra_trees"
            else "pairwise_logistic_linear_score"),
        "feature_mode": "state_relative",
        "feature_dimension": dimensions.pop(),
        "alpha": args.alpha,
        "temperature": args.temperature,
        "fold_count": len(folds),
        "aggregate": {
            "informative_state_count": sum(
                fold["validation"]["informative_state_count"]
                for fold in folds),
            "positive_expected_target_folds": sum(
                (fold["validation"][
                    "mean_model_minus_base_expected_target"] or 0.0) > 0.0
                for fold in folds),
            "positive_expected_final_folds": sum(
                (fold["validation"][
                    "mean_model_minus_base_expected_final"] or 0.0) > 0.0
                for fold in folds),
            "positive_model_top_target_folds": sum(
                (fold["validation"]["mean_model_top_target"] or 0.0) > 0.0
                for fold in folds),
            "positive_model_top_final_folds": sum(
                (fold["validation"]["mean_model_top_final"] or 0.0) > 0.0
                for fold in folds),
            "weighted_pairwise_accuracy": _weighted(
                folds, "pairwise_accuracy", "pair_count"),
            "weighted_mean_state_spearman": _weighted(
                folds, "mean_state_spearman"),
            "weighted_target_top_1_accuracy": _weighted(
                folds, "target_top_1_accuracy"),
            "weighted_final_top_1_accuracy": _weighted(
                folds, "final_top_1_accuracy", "final_informative_state_count"),
            "weighted_mean_model_top_target": _weighted(
                folds, "mean_model_top_target", "state_count"),
            "weighted_mean_model_top_final": _weighted(
                folds, "mean_model_top_final", "state_count"),
            "weighted_model_top_target_positive_fraction": _weighted(
                folds, "model_top_target_positive_fraction", "state_count"),
            "weighted_model_top_final_positive_fraction": _weighted(
                folds, "model_top_final_positive_fraction", "state_count"),
            "weighted_model_minus_base_expected_target": _weighted(
                folds, "mean_model_minus_base_expected_target"),
            "weighted_proposal_minus_base_expected_target": _weighted(
                folds, "mean_proposal_minus_base_expected_target"),
            "weighted_states_model_above_base_target_fraction": _weighted(
                folds, "states_model_above_base_target_fraction"),
            "weighted_model_minus_base_expected_final": _weighted(
                folds, "mean_model_minus_base_expected_final",
                "final_informative_state_count"),
            "weighted_proposal_minus_base_expected_final": _weighted(
                folds, "mean_proposal_minus_base_expected_final",
                "final_informative_state_count"),
            "weighted_states_model_above_base_final_fraction": _weighted(
                folds, "states_model_above_base_final_fraction",
                "final_informative_state_count"),
            "weighted_oracle_gap_fraction_captured": _weighted(
                folds, "mean_oracle_gap_fraction_captured"),
        },
        "folds": folds,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n",
        encoding="utf-8")
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
