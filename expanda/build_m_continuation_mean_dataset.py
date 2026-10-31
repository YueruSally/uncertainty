#!/usr/bin/env python3
"""Build M datasets labelled by matched-continuation mean and uncertainty."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import statistics


EPSILON = 1e-12


def _read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl(path: Path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _sha256(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, allow_nan=False) + "\n")


def _aggregate_records(records, risk_lambda):
    continuation_seeds = [int(row["continuation_seed"]) for row in records]
    if len(set(continuation_seeds)) != len(continuation_seeds):
        raise ValueError("duplicate continuation seeds for an M state/action")
    values = [float(row["final_percent"]) for row in records]
    mean = statistics.fmean(values)
    standard_deviation = statistics.pstdev(values)
    return {
        "continuation_count": len(values),
        "continuation_seeds": sorted(continuation_seeds),
        "continuation_final_percent_each": values,
        "continuation_mean_final_percent": mean,
        "continuation_median_final_percent": statistics.median(values),
        "continuation_standard_deviation_final_percent": standard_deviation,
        "continuation_minimum_final_percent": min(values),
        "continuation_maximum_final_percent": max(values),
        "continuation_positive_fraction": sum(
            value > EPSILON for value in values) / len(values),
        "continuation_negative_fraction": sum(
            value < -EPSILON for value in values) / len(values),
        "continuation_risk_adjusted_final_percent": (
            mean - risk_lambda * standard_deviation),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stability-root", type=Path, required=True)
    parser.add_argument("--source-candidates", type=Path, required=True)
    parser.add_argument("--source-states", type=Path, required=True)
    parser.add_argument("--candidate-out", type=Path, required=True)
    parser.add_argument("--state-out", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--risk-lambda", type=float, default=1.0)
    args = parser.parse_args(argv)
    if args.risk_lambda < 0:
        parser.error("risk lambda must be non-negative")

    stability_path = args.stability_root / "STABILITY_COMPLETE.json"
    if not stability_path.is_file():
        parser.error(f"missing completed stability audit: {stability_path}")
    stability = _read(stability_path)
    for field in (
            "completed", "scenario_exact", "forced_execution_exact",
            "pre_intervention_state_replay_exact"):
        if stability.get(field) is not True:
            parser.error(f"stability audit failed prerequisite: {field}")
    selection = stability.get("selection", {})
    if (selection.get("state_selection") != "stratified"
            or selection.get("action_selection") != "index_spread"):
        parser.error(
            "continuation data were not sampled with outcome-independent "
            "stratified states and index-spread actions")

    source_candidates = {
        (
            row["instance"], int(row["algorithm_seed"]), int(row["event"]),
            int(row["candidate_id"]),
        ): row
        for row in _jsonl(args.source_candidates)
    }
    source_states = {
        row["state_key"]: row for row in _jsonl(args.source_states)
    }
    grouped = defaultdict(list)
    for record in stability["records"]:
        key = (
            record["instance"], int(record["algorithm_seed"]),
            int(record["event"]), int(record["candidate_index"]),
        )
        grouped[key].append(record)

    expected_continuations = int(stability["continuation_seed_count"])
    candidate_rows = []
    by_state = defaultdict(list)
    for key, records in sorted(grouped.items()):
        if key not in source_candidates:
            raise ValueError(f"continuation action missing source features: {key}")
        aggregate = _aggregate_records(records, args.risk_lambda)
        if aggregate["continuation_count"] != expected_continuations:
            raise ValueError(
                f"{key} has {aggregate['continuation_count']} continuations; "
                f"expected {expected_continuations}")
        row = dict(source_candidates[key])
        row.update(aggregate)
        row["continuation_risk_adjusted_final_advantage"] = aggregate[
            "continuation_risk_adjusted_final_percent"]
        row["continuation_mean_final_advantage"] = aggregate[
            "continuation_mean_final_percent"]
        candidate_rows.append(row)
        by_state[row["state_key"]].append(row)

    state_rows = []
    for state_key, rows in sorted(by_state.items()):
        if state_key not in source_states:
            raise ValueError(f"continuation state missing source features: {state_key}")
        state = dict(source_states[state_key])
        risk_oracle = max(
            row["continuation_risk_adjusted_final_percent"] for row in rows)
        mean_oracle = max(
            row["continuation_mean_final_percent"] for row in rows)
        state.update({
            "audited_oracle_percent": risk_oracle,
            "continuation_mean_oracle_percent": mean_oracle,
            "target_definition": "mean_final_percent_minus_lambda_times_std",
            "risk_lambda": args.risk_lambda,
            "positive_any": risk_oracle > EPSILON,
            "positive_5": risk_oracle >= 5.0,
            "positive_10": risk_oracle >= 10.0,
            "tested_candidate_count": len(rows),
        })
        state_rows.append(state)

    expected_candidates = (
        int(stability["selected_action_count"])
    )
    if len(candidate_rows) != expected_candidates:
        raise ValueError(
            f"built {len(candidate_rows)} candidates; "
            f"expected {expected_candidates}")
    expected_states = int(stability["selected_state_count"])
    if len(state_rows) != expected_states:
        raise ValueError(
            f"built {len(state_rows)} states; expected {expected_states}")

    _write_jsonl(args.candidate_out, candidate_rows)
    _write_jsonl(args.state_out, state_rows)
    summary = {
        "completed": True,
        "definition": (
            "outcome-independent stratified M states/actions labelled by "
            "matched-continuation final CCP HV percent"),
        "source_stability": str(stability_path.resolve()),
        "source_stability_sha256": _sha256(stability_path),
        "risk_lambda": args.risk_lambda,
        "instance_count": len({row["instance"] for row in state_rows}),
        "state_count": len(state_rows),
        "candidate_count": len(candidate_rows),
        "continuation_count_per_action": expected_continuations,
        "positive_risk_states": sum(row["positive_any"] for row in state_rows),
        "positive_5_risk_states": sum(row["positive_5"] for row in state_rows),
        "positive_10_risk_states": sum(row["positive_10"] for row in state_rows),
        "positive_risk_candidates": sum(
            row["continuation_risk_adjusted_final_percent"] > EPSILON
            for row in candidate_rows),
        "positive_mean_candidates": sum(
            row["continuation_mean_final_percent"] > EPSILON
            for row in candidate_rows),
        "mean_action_standard_deviation": statistics.fmean(
            row["continuation_standard_deviation_final_percent"]
            for row in candidate_rows),
        "mean_action_positive_fraction": statistics.fmean(
            row["continuation_positive_fraction"] for row in candidate_rows),
        "candidate_dataset": str(args.candidate_out.resolve()),
        "candidate_dataset_sha256": _sha256(args.candidate_out),
        "state_dataset": str(args.state_out.resolve()),
        "state_dataset_sha256": _sha256(args.state_out),
        "scenario_exact": True,
        "forced_execution_exact": True,
        "pre_intervention_state_replay_exact": True,
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n",
        encoding="utf-8")
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
