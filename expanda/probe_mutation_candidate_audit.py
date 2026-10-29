#!/usr/bin/env python3
"""Probe whether logged M candidate features predict held-out local rankings."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import statistics

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


OPERATORS = ("add", "del", "mod", "mode", "replace")


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON in {path}:{line_number}: {error}") from error


def mean(values):
    return float(statistics.fmean(values)) if values else None


def average_ranks(values):
    result = [0.0] * len(values)
    ordered = sorted(range(len(values)), key=lambda index: values[index])
    offset = 0
    while offset < len(ordered):
        end = offset + 1
        while end < len(ordered) and values[ordered[end]] == values[ordered[offset]]:
            end += 1
        rank = (offset + end - 1) / 2.0
        for position in range(offset, end):
            result[ordered[position]] = rank
        offset = end
    return result


def correlation(left, right):
    if len(left) < 2 or len(set(left)) < 2 or len(set(right)) < 2:
        return None
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    left -= left.mean()
    right -= right.mean()
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return None if denominator <= 0.0 else float(left.dot(right) / denominator)


def feature_vector(row):
    parent = np.asarray(row["parent_tokens"], dtype=float)
    if parent.ndim != 2 or not len(parent):
        raise ValueError("parent tokens must be a non-empty matrix")
    graph_tokens = row.get("candidate_graph_tokens") or []
    graph_summary = [
        float(len(graph_tokens)),
        mean([float(token[0]) for token in graph_tokens]) or 0.0,
        mean([float(token[2]) for token in graph_tokens]) or 0.0,
        mean([float(token[3]) for token in graph_tokens]) or 0.0,
    ]
    operator = str(row["operator"])
    if operator not in OPERATORS:
        raise ValueError(f"unknown mutation operator: {operator}")
    return np.asarray([
        *[float(value) for value in row["candidate_features"]],
        *[float(value) for value in row["global_features"]],
        *parent.mean(axis=0).tolist(),
        *parent.std(axis=0).tolist(),
        *graph_summary,
        *[float(operator == name) for name in OPERATORS],
    ], dtype=float)


def raw_utility(row):
    if row["feasible"]:
        return float(row["single_point_hv"])
    violation = max(0.0, float(row["normalized_violation"]))
    return -1.0 - math.log1p(violation)


def load_runs(paths):
    candidates = []
    feature_dimension = None
    for run_index, run in enumerate(paths):
        path = run / "mutation_candidate_audit_outcomes.jsonl"
        if not path.is_file():
            raise ValueError(f"candidate audit outcomes are missing: {path}")
        grouped = defaultdict(list)
        for row in read_jsonl(path):
            grouped[(int(row["state_id"]), int(row["candidate_id"]))].append(row)
        state_candidates = defaultdict(list)
        for (state_id, candidate_id), rows in grouped.items():
            vector = feature_vector(rows[0])
            if feature_dimension is None:
                feature_dimension = len(vector)
            elif len(vector) != feature_dimension:
                raise ValueError("candidate feature dimensions differ between runs")
            if any(not np.array_equal(feature_vector(row), vector) for row in rows[1:]):
                raise ValueError("replicate features differ for one candidate")
            candidate = {
                "state_key": (run_index, state_id),
                "run": str(run.resolve()),
                "state_id": state_id,
                "candidate_id": candidate_id,
                "features": vector,
                "raw_utility": mean([raw_utility(row) for row in rows]),
                "archive_delta_hv": mean([
                    float(row["archive_delta_hv"]) for row in rows]),
                "baseline_probability": float(rows[0]["baseline_probability"]),
                "trained_probability": float(rows[0]["trained_probability"]),
                "replicates": len(rows),
            }
            state_candidates[candidate["state_key"]].append(candidate)
        for state_key, rows in state_candidates.items():
            minimum = min(row["raw_utility"] for row in rows)
            maximum = max(row["raw_utility"] for row in rows)
            span = maximum - minimum
            for row in rows:
                row["target"] = (
                    (row["raw_utility"] - minimum) / span
                    if span > 1e-12 else 0.5)
                row["state_has_quality_gap"] = span > 1e-12
                candidates.append(row)
    if not candidates:
        raise ValueError("candidate audit dataset is empty")
    return candidates, int(feature_dimension)


def conditional_probabilities(rows, name):
    values = np.asarray([max(0.0, float(row[name])) for row in rows])
    total = float(values.sum())
    if total <= 0.0:
        return np.full(len(rows), 1.0 / len(rows))
    return values / total


def pairwise_accuracy(scores, targets):
    correct = total = 0.0
    for left in range(len(targets)):
        for right in range(left + 1, len(targets)):
            difference = targets[left] - targets[right]
            if abs(difference) <= 1e-12:
                continue
            score_difference = scores[left] - scores[right]
            total += 1.0
            if score_difference * difference > 0.0:
                correct += 1.0
            elif abs(score_difference) <= 1e-15:
                correct += 0.5
    return None if total == 0.0 else correct / total


def evaluate(rows, predictions):
    by_state = defaultdict(list)
    for row, prediction in zip(rows, predictions):
        by_state[row["state_key"]].append((row, float(prediction)))
    summaries = []
    for state_rows in by_state.values():
        candidates = [item[0] for item in state_rows]
        scores = [item[1] for item in state_rows]
        targets = [float(row["target"]) for row in candidates]
        prior = conditional_probabilities(candidates, "baseline_probability")
        trained = conditional_probabilities(candidates, "trained_probability")
        oracle = max(targets)
        predicted_top = int(np.argmax(scores))
        trained_top = int(np.argmax(trained))
        prior_top = int(np.argmax(prior))
        prior_expected = float(prior.dot(targets))
        trained_expected = float(trained.dot(targets))
        gap = oracle - prior_expected
        summaries.append({
            "quality_gap": bool(max(targets) - min(targets) > 1e-12),
            "oracle": oracle,
            "prior_expected": prior_expected,
            "trained_expected": trained_expected,
            "prior_argmax": targets[prior_top],
            "trained_argmax": targets[trained_top],
            "model_argmax": targets[predicted_top],
            "model_oracle_hit": float(
                math.isclose(targets[predicted_top], oracle,
                             rel_tol=0.0, abs_tol=1e-12)),
            "model_above_prior_expected": float(
                targets[predicted_top] > prior_expected + 1e-12),
            "oracle_gap_fraction_captured": (
                (targets[predicted_top] - prior_expected) / gap
                if gap > 1e-12 else None),
            "pairwise_accuracy": pairwise_accuracy(scores, targets),
            "spearman": correlation(
                average_ranks(scores), average_ranks(targets)),
        })
    informative = [row for row in summaries if row["quality_gap"]]

    def values(name, rows=informative):
        return [row[name] for row in rows if row[name] is not None]

    return {
        "state_count": len(summaries),
        "informative_state_count": len(informative),
        "mean_oracle": mean(values("oracle")),
        "mean_prior_expected": mean(values("prior_expected")),
        "mean_trained_expected": mean(values("trained_expected")),
        "mean_prior_argmax": mean(values("prior_argmax")),
        "mean_trained_argmax": mean(values("trained_argmax")),
        "mean_model_argmax": mean(values("model_argmax")),
        "model_minus_prior_expected": (
            mean(values("model_argmax")) - mean(values("prior_expected"))
            if informative else None),
        "model_minus_trained_expected": (
            mean(values("model_argmax")) - mean(values("trained_expected"))
            if informative else None),
        "oracle_hit_fraction": mean(values("model_oracle_hit")),
        "states_above_prior_expected_fraction": mean(values(
            "model_above_prior_expected")),
        "mean_oracle_gap_fraction_captured": mean(values(
            "oracle_gap_fraction_captured")),
        "mean_pairwise_accuracy": mean(values("pairwise_accuracy")),
        "mean_spearman": mean(values("spearman")),
    }


def probe(train_paths, validation_paths, seed):
    train, dimension = load_runs(train_paths)
    validation, validation_dimension = load_runs(validation_paths)
    if dimension != validation_dimension:
        raise ValueError("training and validation feature dimensions differ")
    train_informative = [row for row in train if row["state_has_quality_gap"]]
    if len(train_informative) < 4:
        raise ValueError("too few informative training candidates")
    train_x = np.stack([row["features"] for row in train_informative])
    train_y = np.asarray([row["target"] for row in train_informative])
    validation_x = np.stack([row["features"] for row in validation])
    models = {
        "ridge": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "small_mlp": make_pipeline(
            StandardScaler(),
            MLPRegressor(
                hidden_layer_sizes=(64, 32), activation="tanh",
                solver="lbfgs", alpha=1e-3, max_iter=2_000,
                random_state=int(seed))),
    }
    result = {
        "schema_version": 1,
        "label": (
            "within-state min-max utility; feasible uses single-point HV, "
            "infeasible uses -1-log1p(normalized violation)"),
        "feature_definition": (
            "candidate + global + parent mean/std + ordered-path summary + operator"),
        "train_runs": [str(path.resolve()) for path in train_paths],
        "validation_runs": [str(path.resolve()) for path in validation_paths],
        "feature_dimension": dimension,
        "training_candidate_count": len(train),
        "informative_training_candidate_count": len(train_informative),
        "validation_candidate_count": len(validation),
        "models": {},
    }
    all_train_x = np.stack([row["features"] for row in train])
    for name, model in models.items():
        model.fit(train_x, train_y)
        result["models"][name] = {
            "training": evaluate(train, model.predict(all_train_x)),
            "validation": evaluate(validation, model.predict(validation_x)),
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, nargs="+", required=True)
    parser.add_argument("--validation", type=Path, nargs="+", required=True)
    parser.add_argument("--seed", type=int, default=9501)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if set(map(Path.resolve, args.train)) & set(map(Path.resolve, args.validation)):
        parser.error("training and validation runs must be disjoint")
    try:
        result = probe(args.train, args.validation, args.seed)
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))
    output = json.dumps(result, indent=2, allow_nan=False) + "\n"
    args.out.write_text(output, encoding="utf-8")
    print(output, end="")


if __name__ == "__main__":
    main()
