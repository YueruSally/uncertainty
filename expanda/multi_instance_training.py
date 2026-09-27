#!/usr/bin/env python3
"""Build Rule-logged instance splits and fit three post-V1 location labels."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (average_precision_score, balanced_accuracy_score,
                             brier_score_loss, f1_score, log_loss, roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from multi_instance_catalog import VALIDATION
from multi_instance_features import CATEGORICAL, FEATURES, NUMERIC
from multi_instance_logging import write_json

TARGETS = ("selection_survivor", "first_front_member", "parent_relation")
PARENT_RELATION_WEIGHT = .5  # Theory: equal ordinal spacing between loss, tie, win.


def _read_jsonl(path):
    with path.open(encoding="utf-8") as source:
        for line in source:
            yield json.loads(line)


def build(runs_root, out):
    if out.exists() and any(out.iterdir()):
        raise ValueError("dataset directory must be empty")
    out.mkdir(parents=True, exist_ok=True)
    rows, provenance, excluded = [], [], Counter()
    runs = sorted(p for p in runs_root.rglob("rule/run*/COMPLETE.json")
                  if p.parts[-5] in ("train", "validation_rule"))
    if not runs:
        raise ValueError("no Rule logging runs at <root>/<instance>/rule/run*/")
    for complete_file in runs:
        run = complete_file.parent
        done = json.loads(complete_file.read_text())
        config = json.loads((run / "configuration.json").read_text())
        instance = config["instance_id"]
        if (not done["completed"] or config["method"] != "Rule-CCP100"
                or config["instance_split"] == "test"):
            raise ValueError(f"invalid Rule training/validation run {run}")
        if config["instance_split"] != ("validation" if instance in VALIDATION else "train"):
            raise ValueError(f"instance split mismatch: {instance}")
        by_event = {}
        probabilities = Counter()
        for candidate in _read_jsonl(run / "mutation_candidates.jsonl"):
            probabilities[candidate["event_id"]] += candidate["p_rule"]
            if candidate["chosen"]:
                if candidate["event_id"] in by_event:
                    raise ValueError("more than one chosen candidate")
                by_event[candidate["event_id"]] = candidate
        if any(abs(p-1) > 1e-9 and abs(p) > 1e-9 for p in probabilities.values()):
            raise ValueError(f"invalid Rule probability mass in {run}")
        n = 0
        for outcome in _read_jsonl(run / "mutation_outcomes.jsonl"):
            event_id = outcome["event_id"]
            candidate = by_event.get(event_id)
            if candidate is None:
                excluded["missing_chosen_candidate"] += 1
                continue
            if not candidate["eligible"]:
                raise ValueError("Rule selected an ineligible location")
            if not outcome["outcome_attributable_to_selected_target"]:
                excluded["unattributable"] += 1
                continue
            if not outcome["finite_objective_label"]:
                excluded["missing_or_nonfinite_objective"] += 1
                continue
            if outcome["phase"] != "offspring":
                excluded["boost_phase"] += 1
                continue
            if outcome["survived_environmental_selection"] is None or outcome[
                    "first_front_member_label"] is None:
                excluded["missing_selection_or_rank"] += 1
                continue
            features = candidate["pre_mutation_features"]
            if set(features) != set(FEATURES):
                raise ValueError("feature schema mismatch")
            row = dict(features, instance_id=instance, run_id=outcome["run_id"],
                       event_id=event_id, split=config["instance_split"],
                       logging_policy_probability=candidate["logging_policy_probability"],
                       selection_survivor=int(outcome["selection_survivor"]),
                       first_front_member=int(outcome["first_front_member_label"]),
                       parent_relation=outcome["parent_relation"],
                       effective_mutation=bool(outcome["effective_mutation"]),
                       feasible_after=bool(outcome["feasible_after"]))
            rows.append(row)
            n += 1
        provenance.append(dict(run=str(run), instance_id=instance,
                               instance_digest=config["instance_digest"],
                               scenario_digest=config["scenario_digest"], rows=n))
    frame = pd.DataFrame(rows)
    if frame.empty or set(frame.split) != {"train", "validation"}:
        raise ValueError("both complete-instance train and validation data are required")
    frame.to_csv(out / "selected_outcomes.csv", index=False)
    manifest = dict(schema_version=3, targets=TARGETS,
                    train_instances=sorted(set(frame[frame.split == "train"].instance_id)),
                    validation_instances=sorted(set(frame[frame.split == "validation"].instance_id)),
                    parent_relation_weight=PARENT_RELATION_WEIGHT,
                    parent_relation_weight_rationale="Ordinal midpoint: dominates=1, incomparable=0.5, dominated=0; invalid excluded",
                    features=FEATURES, rows=len(frame), excluded=dict(excluded), runs=provenance,
                    by_operator={str(op): int(n) for op, n in
                                 frame.operator.value_counts().items()},
                    target_class_counts={target: {str(k): int(v) for k, v in
                        frame[target].value_counts().items()} for target in TARGETS},
                    oos_used_for_training=False)
    write_json(out / "dataset_manifest.json", manifest)
    return manifest


def _metrics(y, proba, classes, multiclass):
    pred = np.asarray(classes)[np.argmax(proba, axis=1)]
    result = dict(n=len(y), class_counts=dict(Counter(str(v) for v in y)),
                  log_loss=float(log_loss(y, proba, labels=classes)))
    if multiclass:
        result["macro_f1"] = float(f1_score(y, pred, labels=classes, average="macro",
                                             zero_division=0))
        result["balanced_accuracy"] = float(balanced_accuracy_score(y, pred))
    else:
        positive = proba[:, list(classes).index(1)]
        result["brier"] = float(brier_score_loss(y, positive))
        result["roc_auc"] = float(roc_auc_score(y, positive)) if len(set(y)) == 2 else None
        result["pr_auc"] = (float(average_precision_score(y, positive))
                            if len(set(y)) == 2 else None)
    return result


def train(dataset_dir, out, seed=983001, trees=400):
    started = time.perf_counter()
    if out.exists() and any(out.iterdir()):
        raise ValueError("model directory must be empty")
    out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((dataset_dir / "dataset_manifest.json").read_text())
    frame = pd.read_csv(dataset_dir / "selected_outcomes.csv")
    if set(frame[frame.split == "train"].instance_id) & set(
            frame[frame.split == "validation"].instance_id):
        raise ValueError("instance leakage across train and validation")
    report = dict(schema_version=3, train_instances=manifest["train_instances"],
                  validation_instances=manifest["validation_instances"], targets={})
    for target in TARGETS:
        usable = frame[frame.parent_relation != "invalid"] if target == "parent_relation" else frame
        train_rows, val_rows = usable[usable.split == "train"], usable[usable.split == "validation"]
        y, y_val = train_rows[target].to_numpy(), val_rows[target].to_numpy()
        required = {"dominates", "incomparable", "dominated"} if target == "parent_relation" else {0, 1}
        if not required.issubset(set(y)):
            raise ValueError(f"{target}: absent training classes {required - set(y)}")
        preprocess = ColumnTransformer([
            ("numeric", SimpleImputer(strategy="median"), NUMERIC),
            ("categorical", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                                      ("onehot", OneHotEncoder(handle_unknown="ignore"))]), CATEGORICAL),
        ])
        classifier = RandomForestClassifier(n_estimators=trees, min_samples_leaf=5,
            max_features="sqrt", class_weight="balanced_subsample", n_jobs=-1,
            random_state=seed)
        pipeline = Pipeline([("preprocess", preprocess), ("classifier", classifier)])
        propensity = train_rows.logging_policy_probability.to_numpy(dtype=float)
        if not np.all(np.isfinite(propensity)) or np.any(propensity <= 0):
            raise ValueError("invalid logged Rule propensity")
        inverse = 1 / propensity
        clip = float(np.quantile(inverse, .99))
        weights = np.minimum(inverse, clip)
        weights /= weights.mean()
        pipeline.fit(train_rows[FEATURES], y, classifier__sample_weight=weights)
        classes = list(pipeline.classes_)
        binary = target != "parent_relation"
        tr = pipeline.predict_proba(train_rows[FEATURES])
        va = pipeline.predict_proba(val_rows[FEATURES])
        artifact = dict(schema_version=3, target=target, pipeline=pipeline,
                        features=FEATURES, train_instances=manifest["train_instances"],
                        validation_instances=manifest["validation_instances"],
                        parent_relation_weight=PARENT_RELATION_WEIGHT,
                        oos_used_for_training=False)
        model_path = out / f"{target}.joblib"
        joblib.dump(artifact, model_path)
        report["targets"][target] = dict(
            train=_metrics(y, tr, classes, not binary),
            validation=_metrics(y_val, va, classes, not binary),
            validation_score_summary=dict(minimum=float(va.min()),
                maximum=float(va.max()), standard_deviation=float(va.std()),
                distinct_rounded_scores=int(np.unique(np.round(va, 4)).size)),
            validation_operator_counts={str(k): int(v) for k, v in
                val_rows.operator.value_counts().items()},
            inverse_propensity_clip=clip,
            model_sha256=hashlib.sha256(model_path.read_bytes()).hexdigest())
    report["training_seconds_total"] = time.perf_counter() - started
    write_json(out / "offline_checks.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build")
    b.add_argument("--runs-root", type=Path, required=True)
    b.add_argument("--out", type=Path, required=True)
    t = sub.add_parser("train")
    t.add_argument("--dataset", type=Path, required=True)
    t.add_argument("--out", type=Path, required=True)
    t.add_argument("--seed", type=int, default=983001)
    t.add_argument("--trees", type=int, default=400)
    args = parser.parse_args()
    print(build(args.runs_root, args.out) if args.command == "build" else
          train(args.dataset, args.out, args.seed, args.trees))
