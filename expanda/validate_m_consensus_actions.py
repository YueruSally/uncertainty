#!/usr/bin/env python3
"""Independently validate the M actions frozen by the consensus gate."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics

from collect_m_continuation_stability import (
    EPSILON, _analyse, _command, _execute, _read,
)


REQUIRED_AUDIT_FLAGS = (
    "completed",
    "p000_exact_base",
    "scenario_exact",
    "duplicate_replay_consistent",
    "forced_execution_exact",
)


def _state_key(row):
    return (
        str(row["instance"]),
        int(row["algorithm_seed"]),
        int(row["event"]),
    )


def _parse_prediction_key(value):
    parts = str(value).split(":")
    if len(parts) != 3:
        raise ValueError(f"invalid consensus state key: {value}")
    return parts[0], int(parts[1]), int(parts[2])


def build_validation_manifest(consensus, source_manifest, audit,
                              continuation_seeds):
    """Freeze exactly the actions selected by the completed consensus gate."""
    if consensus.get("completed") is not True:
        raise ValueError("consensus report is incomplete")
    selected = [
        row for row in consensus.get("predictions", [])
        if row.get("intervene") is True
    ]
    if len(selected) != int(consensus.get("selected_state_count", -1)):
        raise ValueError("consensus selected-state count is inconsistent")
    if not selected:
        raise ValueError("consensus report selected no actions")
    if len({row["state_key"] for row in selected}) != len(selected):
        raise ValueError("consensus report repeats a selected state")

    old_seeds = {int(value) for value in source_manifest["continuation_seeds"]}
    new_seeds = [int(value) for value in continuation_seeds]
    if len(set(new_seeds)) != len(new_seeds):
        raise ValueError("independent continuation seeds must be unique")
    overlap = old_seeds.intersection(new_seeds)
    if overlap:
        raise ValueError(
            f"independent continuation seeds reuse source seeds: {sorted(overlap)}")

    source_states = {
        _state_key(row): row for row in source_manifest["states"]
    }
    audited_states = {
        _state_key(row): row for row in audit["states"]
    }
    manifest_states = []
    for prediction in sorted(selected, key=lambda row: row["state_key"]):
        key = _parse_prediction_key(prediction["state_key"])
        if key not in source_states or key not in audited_states:
            raise ValueError(f"selected state is missing from source data: {key}")
        source_state = source_states[key]
        audited_state = audited_states[key]
        candidate_index = int(prediction["selected_candidate_id"])
        source_actions = [
            action for action in source_state["actions"]
            if int(action["candidate_index"]) == candidate_index
        ]
        audited_actions = [
            action for action in audited_state["actions"]
            if int(action["candidate_index"]) == candidate_index
            and action.get("selected_replacement_executed") is True
        ]
        if len(source_actions) != 1 or len(audited_actions) != 1:
            raise ValueError(
                f"selected candidate is not uniquely executable: {key}:"
                f"{candidate_index}")
        if int(prediction["generation"]) != int(source_state["generation"]):
            raise ValueError(f"generation mismatch for selected state: {key}")
        action = dict(source_actions[0])
        action["selection_role"] = "top_1"
        manifest_states.append({
            "instance": key[0],
            "algorithm_seed": key[1],
            "event": key[2],
            "generation": int(source_state["generation"]),
            "audited_oracle_percent": float(
                audited_state["audited_oracle_percent"]),
            "actions": [action],
        })

    return {
        "schema_version": 1,
        "selection": {
            "state_selection": "predeclared_consensus",
            "action_selection": "predeclared_consensus",
            "state_count": len(manifest_states),
            "actions_per_state": 1,
            "action_rule": (
                "exact actions frozen by the completed late-stage LOO "
                "consensus report before independent continuation seeds"),
        },
        "continuation_seeds": new_seeds,
        "states": manifest_states,
    }


def _validation_summary(stability):
    actions = stability["action_stability"]
    records = stability["records"]
    action_means = [float(row["mean_final_percent"]) for row in actions]
    risk_scores = [
        float(row["mean_final_percent"])
        - statistics.pstdev(float(value) for value in row["final_percent_each"])
        for row in actions
    ]
    record_values = [float(row["final_percent"]) for row in records]
    thresholds = {
        "mean_action_final_percent": 1.0,
        "positive_action_mean_fraction": 0.60,
        "negative_action_mean_fraction": 0.0,
        "risk_positive_action_fraction": 0.60,
        "maximum_pooled_negative_fraction": 0.20,
    }
    result = {
        "completed": True,
        "definition": (
            "five consensus-selected M actions evaluated with eight new "
            "matched Base continuation seeds under the frozen CCP scenarios"),
        "selected_state_count": stability["selected_state_count"],
        "selected_action_count": stability["selected_action_count"],
        "independent_continuation_seed_count": stability[
            "continuation_seed_count"],
        "paired_action_continuation_count": len(record_values),
        "mean_action_final_percent": statistics.fmean(action_means),
        "median_action_final_percent": statistics.median(action_means),
        "minimum_action_mean_final_percent": min(action_means),
        "maximum_action_mean_final_percent": max(action_means),
        "positive_action_mean_fraction": sum(
            value > EPSILON for value in action_means) / len(action_means),
        "negative_action_mean_fraction": sum(
            value < -EPSILON for value in action_means) / len(action_means),
        "risk_positive_action_fraction": sum(
            value > EPSILON for value in risk_scores) / len(risk_scores),
        "pooled_positive_continuation_fraction": sum(
            value > EPSILON for value in record_values) / len(record_values),
        "pooled_negative_continuation_fraction": sum(
            value < -EPSILON for value in record_values) / len(record_values),
        "scenario_exact": stability["scenario_exact"],
        "forced_execution_exact": stability["forced_execution_exact"],
        "pre_intervention_state_replay_exact": stability[
            "pre_intervention_state_replay_exact"],
        "actions": [
            {
                "instance": row["instance"],
                "algorithm_seed": row["algorithm_seed"],
                "event": row["event"],
                "candidate_index": row["candidate_index"],
                "mean_final_percent": row["mean_final_percent"],
                "standard_deviation_final_percent": statistics.pstdev(
                    float(value) for value in row["final_percent_each"]),
                "risk_adjusted_final_percent": risk,
                "positive_fraction": row["positive_fraction"],
                "negative_fraction": row["negative_fraction"],
            }
            for row, risk in zip(actions, risk_scores)
        ],
        "admission_thresholds": thresholds,
    }
    result["admission_pass"] = bool(
        result["selected_action_count"] == 5
        and result["independent_continuation_seed_count"] == 8
        and result["paired_action_continuation_count"] == 40
        and result["scenario_exact"] is True
        and result["forced_execution_exact"] is True
        and result["pre_intervention_state_replay_exact"] is True
        and result["mean_action_final_percent"] >=
        thresholds["mean_action_final_percent"]
        and result["positive_action_mean_fraction"] >=
        thresholds["positive_action_mean_fraction"]
        and result["negative_action_mean_fraction"] <=
        thresholds["negative_action_mean_fraction"]
        and result["risk_positive_action_fraction"] >=
        thresholds["risk_positive_action_fraction"]
        and result["pooled_negative_continuation_fraction"] <=
        thresholds["maximum_pooled_negative_fraction"]
    )
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--consensus-report", type=Path, required=True)
    parser.add_argument("--source-stability-root", type=Path, required=True)
    parser.add_argument("--audit-root", type=Path, required=True)
    parser.add_argument("--instances", type=Path, required=True)
    parser.add_argument("--normalization", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--continuation-seeds", nargs="+", type=int,
                        required=True)
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--evaluation-budget", type=int, default=400)
    parser.add_argument("--evaluation-mode", choices=("ccp", "ccp100"),
                        default="ccp")
    parser.add_argument("--uncertainty-profile",
                        default="transport_border_saa_v1")
    parser.add_argument("--scenario-count", type=int, default=200)
    parser.add_argument("--ccp-confidence", type=float, default=0.90)
    parser.add_argument("--pop", type=int, default=20)
    parser.add_argument("--gens", type=int, default=1000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--runner", type=Path,
        default=Path(__file__).resolve().with_name(
            "run_learning_components_deterministic.py"))
    args = parser.parse_args(argv)

    source_manifest_path = args.source_stability_root / "selection_manifest.json"
    required = (
        (args.consensus_report, "consensus report"),
        (source_manifest_path, "source stability manifest"),
        (args.audit_root / "AUDIT_COMPLETE.json", "source audit"),
        (args.normalization, "normalization"),
        (args.checkpoint, "checkpoint"),
        (args.runner, "runner"),
    )
    for path, name in required:
        if not path.is_file():
            parser.error(f"missing {name}: {path}")
    if args.max_workers < 1 or len(args.continuation_seeds) != 8:
        parser.error("validation requires eight seeds and at least one worker")

    audit = _read(args.audit_root / "AUDIT_COMPLETE.json")
    for field in REQUIRED_AUDIT_FLAGS:
        if audit.get(field) is not True:
            parser.error(f"source audit failed prerequisite: {field}")
    consensus = _read(args.consensus_report)
    source_manifest = _read(source_manifest_path)
    try:
        manifest = build_validation_manifest(
            consensus, source_manifest, audit, args.continuation_seeds)
    except (KeyError, TypeError, ValueError) as error:
        parser.error(str(error))
    if len(manifest["states"]) != 5:
        parser.error(
            f"expected five frozen consensus actions; found "
            f"{len(manifest['states'])}")

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "selection_manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n",
        encoding="utf-8")

    jobs = []
    for state in manifest["states"]:
        short = state["instance"]
        algorithm_seed = int(state["algorithm_seed"])
        event = int(state["event"])
        instance_path = args.instances / f"train-{short.replace('_', '-')}.json"
        if not instance_path.is_file():
            parser.error(f"missing instance: {instance_path}")
        action = state["actions"][0]
        candidate_index = int(action["candidate_index"])
        for continuation_seed in manifest["continuation_seeds"]:
            root = (
                args.out / short / f"seed_{algorithm_seed}" /
                f"event_{event}" / f"continuation_{continuation_seed}")
            reference = root / "REFERENCE"
            jobs.append((
                reference, reference.with_suffix(".nohup.log"),
                _command(args, instance_path, reference, algorithm_seed, event,
                         continuation_seed, None),
                algorithm_seed, event, continuation_seed, None))
            run = root / f"candidate_{candidate_index}"
            jobs.append((
                run, run.with_suffix(".nohup.log"),
                _command(args, instance_path, run, algorithm_seed, event,
                         continuation_seed, candidate_index),
                algorithm_seed, event, continuation_seed, candidate_index))

    counts = _execute(jobs, min(args.max_workers, len(jobs)))
    stability = _analyse(args, manifest, counts)
    (args.out / "STABILITY_COMPLETE.json").write_text(
        json.dumps(stability, indent=2, allow_nan=False) + "\n",
        encoding="utf-8")
    validation = _validation_summary(stability)
    (args.out / "QUICK_RESULT.json").write_text(
        json.dumps(validation, indent=2, allow_nan=False) + "\n",
        encoding="utf-8")
    (args.out / "VALIDATION_COMPLETE.json").write_text(
        json.dumps(validation, indent=2, allow_nan=False) + "\n",
        encoding="utf-8")
    print(json.dumps(validation, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
