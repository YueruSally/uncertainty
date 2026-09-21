#!/usr/bin/env python3
"""Train a run-split model for Pareto-promising CCP100 mutation locations."""
import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (average_precision_score, balanced_accuracy_score,
                             brier_score_loss, log_loss, roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from learning_features import CATEGORICAL_FEATURES, FEATURES, NUMERIC_FEATURES
from mutation_logging import write_json


def metrics(y, probability):
    prediction = probability >= 0.5
    result = {
        "n": int(len(y)), "positive": int(np.sum(y)),
        "prevalence": float(np.mean(y)),
        "balanced_accuracy": float(balanced_accuracy_score(y, prediction)),
        "brier": float(brier_score_loss(y, probability)),
        "log_loss": float(log_loss(y, probability, labels=[0, 1])),
    }
    if len(np.unique(y)) == 2:
        result["roc_auc"] = float(roc_auc_score(y, probability))
        result["average_precision"] = float(average_precision_score(y, probability))
    else:
        result["roc_auc"] = result["average_precision"] = None
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=983001)
    parser.add_argument("--trees", type=int, default=400)
    args = parser.parse_args(argv)
    if args.out.exists() and any(args.out.iterdir()):
        parser.error("output directory must be empty")
    args.out.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(args.dataset)
    manifest = json.loads(args.manifest.read_text())
    usable = frame[frame["model_row_eligible"].astype(bool)].copy()
    train = usable[usable["split"] == "train"].copy()
    validation = usable[usable["split"] == "validation"].copy()
    if train.empty or validation.empty:
        raise ValueError("Both train and validation rows are required")
    y_train = train["pareto_promising"].astype(int).to_numpy()
    y_validation = validation["pareto_promising"].astype(int).to_numpy()
    if len(np.unique(y_train)) != 2:
        raise ValueError("Training target must contain both classes")
    numeric = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocess = ColumnTransformer([
        ("numeric", numeric, NUMERIC_FEATURES),
        ("categorical", categorical, CATEGORICAL_FEATURES),
    ])
    classifier = RandomForestClassifier(
        n_estimators=args.trees, min_samples_leaf=5, max_features="sqrt",
        class_weight="balanced_subsample", n_jobs=-1, random_state=args.seed)
    pipeline = Pipeline([("preprocess", preprocess), ("classifier", classifier)])
    propensity = train["baseline_selection_probability"].astype(float).to_numpy()
    inverse = 1.0 / np.maximum(propensity, 1e-12)
    clip = float(np.quantile(inverse, .99))
    weights = np.minimum(inverse, clip)
    weights /= np.mean(weights)
    pipeline.fit(train[FEATURES], y_train, classifier__sample_weight=weights)
    train_probability = pipeline.predict_proba(train[FEATURES])[:, 1]
    validation_probability = pipeline.predict_proba(validation[FEATURES])[:, 1]
    report = {
        "schema_version": 1, "target": manifest["target"],
        "target_definition": manifest["target_definition"],
        "random_seed": args.seed, "trees": args.trees,
        "inverse_propensity_clip": clip, "features": FEATURES,
        "train": metrics(y_train, train_probability),
        "validation": metrics(y_validation, validation_probability),
        "validation_by_operator": {}, "train_runs": manifest["train_runs"],
        "validation_runs": manifest["validation_runs"], "oos_used": False,
    }
    for operator, part in validation.assign(probability=validation_probability).groupby("operator"):
        report["validation_by_operator"][operator] = metrics(
            part["pareto_promising"].astype(int).to_numpy(),
            part["probability"].to_numpy())
    artifact = {
        "schema_version": 1, "pipeline": pipeline, "features": FEATURES,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "target": manifest["target"], "target_definition": manifest["target_definition"],
        "train_runs": manifest["train_runs"], "validation_runs": manifest["validation_runs"],
    }
    model_path = args.out / "location_model.joblib"
    joblib.dump(artifact, model_path)
    report["model_sha256"] = hashlib.sha256(model_path.read_bytes()).hexdigest()
    write_json(args.out / "training_metrics.json", report)
    predictions = validation[["run_number", "event_id", "operator"]].copy()
    predictions["target"] = y_validation
    predictions["probability"] = validation_probability
    predictions.to_csv(args.out / "validation_predictions.csv", index=False)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
