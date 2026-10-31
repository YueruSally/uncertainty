#!/usr/bin/env python3
"""Build M state-gate and candidate-ranker datasets from a completed audit."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from transformer_ppo.supervised_prior import (
    _candidate_absolute_features, state_relative_features,
)


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


def _policy_state(run: Path, event: int):
    rows = [
        row for row in _jsonl(run / "mutation_policy_states.jsonl")
        if int(row.get("mixture_event_index", -1)) == event
        and row.get("decision_stage", "location") == "location"
    ]
    if len(rows) != 1:
        raise ValueError(
            f"expected one M policy state at event {event}, found "
            f"{len(rows)}: {run}")
    return rows[0]


def _state_features(absolute):
    return np.concatenate((
        absolute.mean(axis=0),
        absolute.std(axis=0),
        absolute.min(axis=0),
        absolute.max(axis=0),
        np.asarray([np.log1p(len(absolute))], dtype=float),
    ))


def _write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, allow_nan=False) + "\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--candidate-out", type=Path, required=True)
    parser.add_argument("--state-out", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args(argv)

    audit_path = args.root / "AUDIT_COMPLETE.json"
    manifest_path = args.root / "collection_manifest.json"
    if not audit_path.is_file() or not manifest_path.is_file():
        parser.error(f"completed M audit is missing: {args.root}")
    audit = _read(audit_path)
    manifest = _read(manifest_path)
    if audit.get("completed") is not True:
        parser.error(f"M audit is incomplete: {args.root}")

    manifest_states = {}
    for instance in manifest["instances"]:
        for seed_row in instance["seeds"]:
            for state in seed_row["states"]:
                key = (
                    instance["short_id"], int(seed_row["algorithm_seed"]),
                    int(state["event"]),
                )
                manifest_states[key] = state

    candidate_rows = []
    state_rows = []
    observation_failures = []
    for summary_state in audit["states"]:
        instance = summary_state["instance"]
        seed = int(summary_state["algorithm_seed"])
        event = int(summary_state["event"])
        key = (instance, seed, event)
        if key not in manifest_states:
            raise ValueError(f"audit state missing from manifest: {key}")
        manifest_state = manifest_states[key]
        state_root = args.root / instance / f"seed_{seed}" / f"event_{event}"
        probe_state = _policy_state(state_root / "PROBE", event)
        observation = probe_state["observation"]
        eligible = np.asarray(observation["eligible"], dtype=bool)
        eligible_indices = np.flatnonzero(eligible)
        positions = {
            int(candidate_index): position
            for position, candidate_index in enumerate(eligible_indices)
        }
        absolute = _candidate_absolute_features(observation, eligible_indices)
        relative = state_relative_features(absolute)
        state_key = f"{instance}:{seed}:{event}"
        oracle = float(summary_state["audited_oracle_percent"])
        aggregate = _state_features(absolute)
        state_rows.append({
            "instance": instance,
            "algorithm_seed": seed,
            "event": event,
            "generation": int(summary_state["generation"]),
            "state_key": state_key,
            "feature_dimension": int(len(aggregate)),
            "features": aggregate.astype(float).tolist(),
            "audited_oracle_percent": oracle,
            "positive_any": bool(oracle > EPSILON),
            "positive_5": bool(oracle >= 5.0),
            "positive_10": bool(oracle >= 10.0),
            "tested_candidate_count": int(
                summary_state["audited_candidate_count"]),
            "full_candidate_count": int(len(observation["candidates"])),
        })

        expected_indices = set(
            int(value) for value in manifest_state["forced_candidate_indices"])
        action_indices = {
            int(action["candidate_index"])
            for action in summary_state["actions"]
        }
        if action_indices != expected_indices:
            raise ValueError(
                f"manifest/action candidate mismatch: {state_key}")
        for action in summary_state["actions"]:
            candidate_index = int(action["candidate_index"])
            if candidate_index not in positions:
                raise ValueError(
                    f"audited candidate {candidate_index} is not eligible: "
                    f"{state_key}")
            branch = state_root / f"candidate_{candidate_index}"
            branch_state = _policy_state(branch, event)
            if branch_state["observation"] != observation:
                observation_failures.append(
                    f"{state_key}:{candidate_index}")
            position = positions[candidate_index]
            final_percent = float(action["final_percent"])
            base_probability = float(
                observation["action_prior"][candidate_index])
            candidate_rows.append({
                "instance": instance,
                "algorithm_seed": seed,
                "event": event,
                "generation": int(summary_state["generation"]),
                "state_key": state_key,
                "candidate_id": candidate_index,
                "candidate_key": action["action_signature"],
                "feature_dimension": int(relative.shape[1]),
                "features": relative[position].astype(float).tolist(),
                "base_probability": base_probability,
                "proposal_probability": base_probability,
                "immediate_archive_delta": float(
                    action.get("immediate_archive_delta", 0.0)),
                # Compatibility aliases for the existing LOO ranker probe.
                "plus_1_generation_archive_advantage": final_percent,
                "final_archive_advantage": final_percent,
                "final_percent": final_percent,
            })

    if observation_failures:
        raise ValueError(
            "forced branches do not replay an identical M state: "
            f"{observation_failures[:5]}")
    candidate_rows.sort(key=lambda row: (
        row["instance"], row["algorithm_seed"], row["event"],
        row["candidate_id"]))
    state_rows.sort(key=lambda row: (
        row["instance"], row["algorithm_seed"], row["event"]))
    candidate_dimensions = {
        row["feature_dimension"] for row in candidate_rows}
    state_dimensions = {row["feature_dimension"] for row in state_rows}
    if len(candidate_dimensions) != 1 or len(state_dimensions) != 1:
        raise ValueError("inconsistent M dataset feature dimensions")

    _write_jsonl(args.candidate_out, candidate_rows)
    _write_jsonl(args.state_out, state_rows)
    summary = {
        "completed": True,
        "source_root": str(args.root.resolve()),
        "audit_sha256": _sha256(audit_path),
        "algorithm_seeds": sorted({
            row["algorithm_seed"] for row in state_rows}),
        "instance_count": len({row["instance"] for row in state_rows}),
        "state_count": len(state_rows),
        "candidate_count": len(candidate_rows),
        "candidate_feature_dimension": candidate_dimensions.pop(),
        "state_feature_dimension": state_dimensions.pop(),
        "positive_any_states": sum(row["positive_any"] for row in state_rows),
        "positive_5_states": sum(row["positive_5"] for row in state_rows),
        "positive_10_states": sum(row["positive_10"] for row in state_rows),
        "positive_candidates": sum(
            row["final_percent"] > EPSILON for row in candidate_rows),
        "negative_candidates": sum(
            row["final_percent"] < -EPSILON for row in candidate_rows),
        "candidate_dataset": str(args.candidate_out.resolve()),
        "candidate_dataset_sha256": _sha256(args.candidate_out),
        "state_dataset": str(args.state_out.resolve()),
        "state_dataset_sha256": _sha256(args.state_out),
        "fixed_state_replay_consistent": True,
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n",
        encoding="utf-8")
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
