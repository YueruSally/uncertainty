#!/usr/bin/env python3
"""Summarise counterfactual position diagnostics without using formal tests."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold

from multi_instance_logging import write_json
from transformer_ppo.rf_reward import FEATURE_NAMES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--rf-seed", type=int, default=64_000_001)
    args = parser.parse_args()
    rows = []
    for root in args.runs:
        for path in sorted(root.rglob("position_diagnostics.jsonl")):
            config = json.loads((path.parent / "configuration.json").read_text())
            if config.get("instance_split") != "train":
                raise ValueError(f"diagnostic source is not training-only: {path}")
            with path.open(encoding="utf-8") as handle:
                rows.extend(json.loads(line) for line in handle)
    if not rows:
        raise ValueError("no position diagnostic rows found")
    positions = defaultdict(list)
    for row in rows:
        positions[(row["instance_id"], row["event_id"], row["candidate_index"])].append(row)
    aggregated = []
    for key, repeated in positions.items():
        aggregated.append({
            "instance_id": key[0], "event_id": key[1], "candidate_index": key[2],
            "operator": repeated[0]["operator"],
            "feature_values": repeated[0]["feature_values"],
            "mean_reward": float(np.mean([row["reward"] for row in repeated])),
            "reward_std": float(np.std([row["reward"] for row in repeated])),
            "mean_hv_delta": float(np.mean([row["reward_hv_delta"] for row in repeated])),
            "effective_rate": float(np.mean([row["effective_mutation"] for row in repeated])),
        })
    by_event = defaultdict(list)
    for row in aggregated:
        by_event[(row["instance_id"], row["event_id"])].append(row)
    event_ranges = []
    for key, values in by_event.items():
        rewards = np.asarray([row["mean_reward"] for row in values])
        event_ranges.append({"instance_id": key[0], "event_id": key[1],
                             "operator": values[0]["operator"],
                             "positions": len(values),
                             "reward_range": float(rewards.max() - rewards.min()),
                             "reward_std_across_positions": float(rewards.std())})
        center = float(rewards.mean())
        for row in values:
            row["event_centered_reward"] = row["mean_reward"] - center
    action_indices = [i for i, name in enumerate(FEATURE_NAMES)
                      if name.startswith("action_") or name.startswith("operator_")]
    X = np.asarray([[row["feature_values"][i] for i in action_indices]
                    for row in aggregated], dtype=float)
    y = np.asarray([row["event_centered_reward"] for row in aggregated], dtype=float)
    groups = np.asarray([f"{row['instance_id']}::{row['event_id']}" for row in aggregated])
    relationship = {"status": "insufficient_events"}
    if len(set(groups)) >= 3 and np.sum(y ** 2) > 0:
        predictions = np.full(len(y), np.nan)
        folds = min(5, len(set(groups)))
        for fold, (train, valid) in enumerate(
                GroupKFold(folds).split(X, y, groups)):
            model = RandomForestRegressor(
                n_estimators=300, max_depth=6, min_samples_leaf=2,
                random_state=args.rf_seed + fold, n_jobs=1)
            model.fit(X[train], y[train])
            predictions[valid] = model.predict(X[valid])
        final = RandomForestRegressor(
            n_estimators=300, max_depth=6, min_samples_leaf=2,
            random_state=args.rf_seed, n_jobs=1).fit(X, y)
        names = [FEATURE_NAMES[i] for i in action_indices]
        importance = sorted(zip(names, final.feature_importances_),
                            key=lambda item: (-item[1], item[0]))[:10]
        relationship = {
            "status": "exploratory_event_grouped_cv",
            "folds": folds,
            "samples": len(y),
            "events": len(set(groups)),
            "event_centered_zero_baseline_mae": float(np.mean(np.abs(y))),
            "event_centered_rf_mae": float(np.mean(np.abs(predictions - y))),
            "event_centered_rf_r2": float(1.0 - np.sum((predictions - y) ** 2)
                                          / np.sum(y ** 2)),
            "prediction_correlation": (float(np.corrcoef(predictions, y)[0, 1])
                                       if np.std(predictions) > 0 else None),
            "top_in_sample_feature_importance": [
                {"feature": name, "importance": float(value)}
                for name, value in importance],
            "interpretation": "diagnostic only; not hyperparameter or formal-performance evidence",
        }
    rewards = np.asarray([row["reward"] for row in rows], dtype=float)
    report = {
        "schema_version": 1,
        "training_only": True,
        "probe_rows": len(rows),
        "positions": len(aggregated),
        "events": len(by_event),
        "instances": sorted({row["instance_id"] for row in rows}),
        "extra_ccp100_evaluations": len(rows),
        "effective_modification_rate": float(np.mean([row["effective_mutation"] for row in rows])),
        "no_change_rate": float(np.mean([not row["decision_changed"] for row in rows])),
        "mutation_failure_rate": float(np.mean([not row["mutation_success"] for row in rows])),
        "repair_failure_or_revert_rate": float(np.mean([
            (not row["repair_success"]) or row["mutation_reverted"] for row in rows])),
        "infeasible_after_rate": float(np.mean([not row["feasible_after"] for row in rows])),
        "reward": {"mean": float(rewards.mean()), "std": float(rewards.std()),
                   "minimum": float(rewards.min()), "q25": float(np.quantile(rewards, .25)),
                   "median": float(np.median(rewards)), "q75": float(np.quantile(rewards, .75)),
                   "maximum": float(rewards.max()),
                   "positive_rate": float(np.mean(rewards > 0)),
                   "negative_rate": float(np.mean(rewards < 0))},
        "position_difference": {
            "events_with_nonzero_mean_reward_range": int(sum(
                row["reward_range"] > 1e-12 for row in event_ranges)),
            "fraction_with_nonzero_mean_reward_range": float(np.mean([
                row["reward_range"] > 1e-12 for row in event_ranges])),
            "mean_within_event_reward_range": float(np.mean([
                row["reward_range"] for row in event_ranges])),
            "median_within_event_reward_range": float(np.median([
                row["reward_range"] for row in event_ranges])),
            "events": event_ranges,
        },
        "action_feature_relationship": relationship,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.out, report)
    print({key: report[key] for key in (
        "probe_rows", "events", "effective_modification_rate",
        "position_difference", "action_feature_relationship")})


if __name__ == "__main__":
    main()
