#!/usr/bin/env python3
"""Fit a portable pairwise-linear prior from mutation audit runs."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from probe_mutation_candidate_audit import (
    design_matrix, evaluate, load_runs, pairwise_training_data,
)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fit(runs, alpha, temperature, seed):
    rows, base_dimension = load_runs(
        runs, target_metric="archive_delta_hv", selected_operator="replace")
    informative = [row for row in rows if row["state_has_target_gap"]]
    if len(informative) < 4:
        raise ValueError("too few informative candidates")
    matrix = design_matrix(rows, "state_relative")
    indices = [index for index, row in enumerate(rows)
               if row["state_has_target_gap"]]
    informative_matrix = matrix[indices]
    pair_x, pair_y = pairwise_training_data(informative, informative_matrix)
    scaler = StandardScaler().fit(pair_x)
    scaled = scaler.transform(pair_x)
    model = LogisticRegression(
        C=0.1, max_iter=2_000, random_state=int(seed)).fit(scaled, pair_y)
    effective_weights = model.coef_[0] / scaler.scale_
    predictions = matrix.dot(effective_weights)
    return {
        "schema_version": 1,
        "model": "pairwise_logistic_linear_score",
        "target_metric": "archive_delta_hv",
        "selected_operator": "replace",
        "feature_mode": "state_relative",
        "base_feature_dimension": int(base_dimension),
        "feature_dimension": int(matrix.shape[1]),
        "effective_weights": effective_weights.tolist(),
        "alpha": float(alpha),
        "temperature": float(temperature),
        "seed": int(seed),
        "training_run_count": len(runs),
        "training_candidate_count": len(rows),
        "informative_training_candidate_count": len(informative),
        "training_pair_count": int(len(pair_y) // 2),
        "training_metrics": evaluate(rows, predictions),
        "source_runs": [{
            "path": str(run.resolve()),
            "outcomes_sha256": sha256(
                run / "mutation_candidate_audit_outcomes.jsonl"),
        } for run in runs],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--temperature", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=9601)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if not 0.0 <= args.alpha <= 1.0 or args.temperature <= 0.0:
        parser.error("invalid mixture alpha or temperature")
    try:
        artifact = fit(args.runs, args.alpha, args.temperature, args.seed)
    except (KeyError, OSError, TypeError, ValueError) as error:
        parser.error(str(error))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(artifact, indent=2, allow_nan=False) + "\n",
        encoding="utf-8")
    print(json.dumps({
        "out": str(args.out),
        "feature_dimension": artifact["feature_dimension"],
        "training_pair_count": artifact["training_pair_count"],
        "training_metrics": artifact["training_metrics"],
    }, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
