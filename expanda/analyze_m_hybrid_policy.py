#!/usr/bin/env python3
"""Combine an OOF M state gate with OOF Transformer candidate choices."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


EPSILON = 1e-12


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _load_transformer_rows(root: Path):
    reports = sorted(root.glob("heldout_S0_*/report.json"))
    if not reports:
        raise ValueError(f"no Transformer fold reports found below {root}")
    rows = []
    instances = set()
    for path in reports:
        report = _read_json(path)
        fold = report["heldout_instance"]
        instances.add(fold)
        for row in report["validation"]:
            if row["instance"] != fold:
                raise ValueError(
                    f"{path}: validation instance {row['instance']} != {fold}")
            rows.append(row)
    keys = [row["state_key"] for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("Transformer validation state keys are not unique")
    return rows, sorted(instances)


def _load_gate_predictions(path: Path, label: str):
    report = _read_json(path)
    try:
        label_report = report["labels"][label]
    except KeyError as error:
        raise ValueError(f"gate report does not contain label {label}") from error
    predictions = label_report["predictions"]
    by_key = {row["state_key"]: row for row in predictions}
    if len(by_key) != len(predictions):
        raise ValueError("gate prediction state keys are not unique")
    return report, label_report, by_key


def _fraction_metrics(rows, fraction, instance_count):
    selected_count = max(1, math.ceil(len(rows) * fraction))
    selected = sorted(
        rows, key=lambda row: row["state_gate_probability"], reverse=True
    )[:selected_count]
    deployed_by_instance = {}
    for row in selected:
        deployed_by_instance.setdefault(row["instance"], 0.0)
        deployed_by_instance[row["instance"]] += row["selected_gain_percent"]
    selected_gains = [row["selected_gain_percent"] for row in selected]
    selected_oracles = [row["oracle_percent"] for row in selected]
    positive_selected = sum(gain > EPSILON for gain in selected_gains)
    negative_selected = sum(gain < -EPSILON for gain in selected_gains)
    return {
        "fraction": fraction,
        "selected_states": selected_count,
        "oracle_above_10_precision": (
            sum(value >= 10.0 for value in selected_oracles) / selected_count
        ),
        "mean_selected_oracle_percent": sum(selected_oracles) / selected_count,
        "selected_actions_above_base_fraction": (
            positive_selected / selected_count
        ),
        "selected_actions_below_base_fraction": (
            negative_selected / selected_count
        ),
        "mean_selected_action_gain_percent": sum(selected_gains) / selected_count,
        "overall_mean_deployed_gain_percent": sum(selected_gains) / len(rows),
        "positive_folds": sum(
            total > EPSILON for total in deployed_by_instance.values()
        ),
        "instance_count": instance_count,
        "minimum_selected_gain_percent": min(selected_gains),
        "maximum_selected_gain_percent": max(selected_gains),
        "selected_state_keys": [row["state_key"] for row in selected],
    }


def build_report(gate_report_path, transformer_root, fractions, label):
    transformer_rows, instances = _load_transformer_rows(transformer_root)
    gate_report, label_report, gate_by_key = _load_gate_predictions(
        gate_report_path, label)
    transformer_keys = {row["state_key"] for row in transformer_rows}
    gate_keys = set(gate_by_key)
    if transformer_keys != gate_keys:
        missing_gate = sorted(transformer_keys - gate_keys)
        missing_transformer = sorted(gate_keys - transformer_keys)
        raise ValueError(
            "state-key mismatch: "
            f"missing_gate={missing_gate[:5]} "
            f"missing_transformer={missing_transformer[:5]}")
    joined = []
    for row in transformer_rows:
        gate = gate_by_key[row["state_key"]]
        if row["instance"] != gate["instance"]:
            raise ValueError(f"instance mismatch for {row['state_key']}")
        if abs(row["oracle_percent"] - gate["oracle_percent"]) > EPSILON:
            raise ValueError(f"oracle mismatch for {row['state_key']}")
        joined.append({
            "state_key": row["state_key"],
            "instance": row["instance"],
            "generation": row["generation"],
            "state_gate_probability": gate["probability"],
            "oracle_percent": row["oracle_percent"],
            "selected_candidate_index": row["selected_candidate_index"],
            "selected_gain_percent": row["selected_gain_percent"],
        })
    sweep = [
        _fraction_metrics(joined, fraction, len(instances))
        for fraction in fractions
    ]
    admission = [
        row for row in sweep
        if row["oracle_above_10_precision"] >= 0.60
        and row["mean_selected_oracle_percent"] >= 10.0
        and row["selected_actions_above_base_fraction"] >= 0.60
        and row["positive_folds"] >= 7
        and row["overall_mean_deployed_gain_percent"] >= 1.0
    ]
    return {
        "completed": True,
        "definition": (
            "leave-one-instance-out balanced logistic M state gate joined "
            "with leave-one-instance-out path-v4 Transformer candidate choices"
        ),
        "state_count": len(joined),
        "instance_count": len(instances),
        "state_keys_exact": True,
        "gate": {
            "model": gate_report.get("model"),
            "label": label,
            "loo_roc_auc": label_report["loo_roc_auc"],
            "loo_average_precision": label_report["loo_average_precision"],
        },
        "threshold_sweep": sweep,
        "admission_thresholds": {
            "oracle_above_10_precision": 0.60,
            "mean_selected_oracle_percent": 10.0,
            "selected_actions_above_base_fraction": 0.60,
            "positive_folds": 7,
            "overall_mean_deployed_gain_percent": 1.0,
        },
        "admission_pass": bool(admission),
        "admitted_fractions": [row["fraction"] for row in admission],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate-report", type=Path, required=True)
    parser.add_argument("--transformer-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--label", default="positive_10")
    parser.add_argument(
        "--fractions", type=float, nargs="+",
        default=(0.05, 0.10, 0.15, 0.20, 0.25, 0.30))
    args = parser.parse_args(argv)
    if any(not 0 < fraction <= 1 for fraction in args.fractions):
        parser.error("all fractions must be in (0, 1]")
    report = build_report(
        args.gate_report, args.transformer_root, args.fractions, args.label)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n",
        encoding="utf-8")
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
