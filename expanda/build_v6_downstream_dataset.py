#!/usr/bin/env python3
"""Join one-shot policy observations to paired one-generation HV labels."""
from __future__ import annotations

import argparse
from collections import defaultdict
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


def _progress(run: Path):
    result = {}
    for row in _jsonl(run / "search_progress.jsonl"):
        if row.get("checkpoint") != "environmental_selection":
            continue
        generation = int(row["generation"])
        if generation in result:
            raise ValueError(f"duplicate progress generation {generation}: {run}")
        result[generation] = row
    return result


def _event_row(run: Path, event: int):
    rows = [
        row for row in _jsonl(run / "mutation_outcomes.jsonl")
        if row.get("mixture_event_index") == event
        and row.get("decision_kind") == "model_choice"
    ]
    if len(rows) != 1:
        raise ValueError(
            f"expected one model action at event {event}, found {len(rows)}: {run}")
    return rows[0]


def _policy_state(run: Path, event: int):
    rows = [
        row for row in _jsonl(run / "mutation_policy_states.jsonl")
        if row.get("mixture_event_index") == event
        and row.get("operator") == "replace"
        and row.get("decision_stage", "location") == "location"
    ]
    if len(rows) != 1:
        raise ValueError(
            f"expected one replace policy state at event {event}, "
            f"found {len(rows)}: {run}")
    return rows[0]


def _same_target(candidate, outcome):
    target = candidate.get("target") or {}
    for name in ("batch_id", "allocation_index", "replacement_path_id"):
        left = target.get(name)
        right = outcome.get(name)
        if name == "replacement_path_id":
            left = None if left is None else str(left)
            right = None if right is None else str(right)
        if left != right:
            return False
    return True


def _paired_delta(base_progress, branch_progress, generation, field):
    if generation not in base_progress or generation not in branch_progress:
        raise ValueError(f"missing generation {generation} for {field}")
    base = base_progress[generation].get(field)
    branch = branch_progress[generation].get(field)
    if base is None or branch is None:
        raise ValueError(f"null generation {generation} value for {field}")
    return float(branch) - float(base)


def _branch_record(instance, seed, event, run, base_run, base_progress):
    outcome = _event_row(run, event)
    state = _policy_state(run, event)
    observation = state["observation"]
    matches = [
        int(candidate["candidate_id"])
        for candidate in state["candidates"]
        if _same_target(candidate, outcome)
    ]
    if len(matches) != 1:
        raise ValueError(
            f"expected one selected candidate, found {len(matches)}: {run}")
    candidate_index = matches[0]
    if str(outcome.get("executed_replacement_path_id")) != str(
            outcome.get("replacement_path_id")):
        raise ValueError(f"selected replacement path was not executed: {run}")
    eligible = np.asarray(observation["eligible"], dtype=bool)
    eligible_indices = np.flatnonzero(eligible)
    eligible_positions = {
        int(index): position for position, index in enumerate(eligible_indices)
    }
    if candidate_index not in eligible_positions:
        raise ValueError(f"selected candidate is not eligible: {run}")
    absolute = _candidate_absolute_features(observation, eligible_indices)
    feature_matrix = state_relative_features(absolute)
    feature_vector = feature_matrix[eligible_positions[candidate_index]]
    generation = int(outcome["generation"])
    branch_progress = _progress(run)
    base_complete = _read(base_run / "COMPLETE.json")
    branch_complete = _read(run / "COMPLETE.json")
    base_prior = observation.get(
        "base_action_prior", observation["action_prior"])
    supervised_prior = observation["action_prior"]
    return {
        "instance": instance,
        "algorithm_seed": seed,
        "event": event,
        "state_key": f"{instance}:{seed}:{event}",
        "generation": generation,
        "policy_seed": int(run.name.removeprefix("policy_")),
        "run": str(run),
        "candidate_id": candidate_index,
        "batch_id": outcome.get("batch_id"),
        "allocation_index": outcome.get("allocation_index"),
        "replacement_path_id": str(outcome["replacement_path_id"]),
        "candidate_key": (
            f"{outcome.get('batch_id')}:{outcome.get('allocation_index')}:"
            f"{outcome['replacement_path_id']}"),
        "feature_dimension": int(len(feature_vector)),
        "features": feature_vector.astype(float).tolist(),
        "base_probability": float(base_prior[candidate_index]),
        "proposal_probability": float(supervised_prior[candidate_index]),
        "immediate_archive_delta": float(outcome.get("reward_hv_delta", 0.0)),
        "survived_environmental_selection": bool(
            outcome.get("survived_environmental_selection")),
        "post_selection_archive_advantage": _paired_delta(
            base_progress, branch_progress, generation, "archive_hv"),
        "plus_1_generation_archive_advantage": _paired_delta(
            base_progress, branch_progress, generation + 1, "archive_hv"),
        "plus_1_generation_population_advantage": _paired_delta(
            base_progress, branch_progress, generation + 1, "population_hv"),
        "final_archive_advantage": (
            float(branch_complete["final_hypervolume"])
            - float(base_complete["final_hypervolume"])),
        "final_archive_sha256": _sha256(run / "final_archive_lineage.json"),
    }


def _span(values):
    return max(values) - min(values) if values else 0.0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args(argv)
    complete = args.root / "COLLECTION_COMPLETE.json"
    manifest_path = args.root / "collection_manifest.json"
    if not complete.is_file() or _read(complete).get("completed") is not True:
        parser.error(f"collection is incomplete: {args.root}")
    manifest = _read(manifest_path)

    raw = []
    for instance_row in manifest["instances"]:
        instance = instance_row["short_id"]
        seed = int(manifest["algorithm_seed"])
        seed_root = args.root / instance / f"seed_{seed}"
        base_run = seed_root / "P000"
        base_progress = _progress(base_run)
        for state in instance_row["states"]:
            event = int(state["event"])
            state_root = seed_root / f"event_{event}"
            for policy_seed in manifest["policy_seeds"]:
                run = state_root / f"policy_{policy_seed}"
                raw.append(_branch_record(
                    instance, seed, event, run, base_run, base_progress))

    dimensions = {record["feature_dimension"] for record in raw}
    if len(dimensions) != 1:
        raise ValueError(f"inconsistent feature dimensions: {dimensions}")
    grouped = defaultdict(list)
    for record in raw:
        grouped[(record["state_key"], record["candidate_key"])].append(record)
    failures = []
    deduplicated = []
    numeric = (
        "immediate_archive_delta", "post_selection_archive_advantage",
        "plus_1_generation_archive_advantage",
        "plus_1_generation_population_advantage", "final_archive_advantage",
    )
    for key, records in grouped.items():
        first = records[0]
        for other in records[1:]:
            if not np.allclose(first["features"], other["features"],
                               rtol=0.0, atol=EPSILON):
                failures.append({"key": key, "field": "features"})
            for field in numeric:
                if abs(first[field] - other[field]) > EPSILON:
                    failures.append({"key": key, "field": field})
            if first["final_archive_sha256"] != other["final_archive_sha256"]:
                failures.append({"key": key, "field": "final_archive_sha256"})
        merged = dict(first)
        merged["replicate_count"] = len(records)
        merged["policy_seeds"] = [record["policy_seed"] for record in records]
        deduplicated.append(merged)
    if failures:
        raise ValueError(f"duplicate branch inconsistency: {failures[:5]}")
    deduplicated.sort(key=lambda record: (
        record["instance"], record["event"], record["candidate_key"]))

    by_state = defaultdict(list)
    for record in deduplicated:
        by_state[record["state_key"]].append(record)
    state_summary = []
    label = "plus_1_generation_archive_advantage"
    for state_key, records in sorted(by_state.items()):
        state_summary.append({
            "state_key": state_key,
            "instance": records[0]["instance"],
            "candidate_count": len(records),
            "label_span": _span([record[label] for record in records]),
            "final_span": _span([
                record["final_archive_advantage"] for record in records]),
        })

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        for record in deduplicated:
            handle.write(json.dumps(record, allow_nan=False) + "\n")
    summary = {
        "completed": True,
        "source_root": str(args.root),
        "source_manifest_sha256": _sha256(manifest_path),
        "target": label,
        "feature_mode": "state_relative",
        "feature_dimension": dimensions.pop(),
        "raw_branch_count": len(raw),
        "unique_candidate_count": len(deduplicated),
        "duplicate_branch_count": len(raw) - len(deduplicated),
        "duplicate_replay_consistent": True,
        "state_count": len(by_state),
        "states_with_at_least_two_candidates": sum(
            row["candidate_count"] >= 2 for row in state_summary),
        "informative_label_states": sum(
            row["label_span"] > EPSILON for row in state_summary),
        "informative_final_states": sum(
            row["final_span"] > EPSILON for row in state_summary),
        "zero_immediate_reward_candidates": sum(
            abs(record["immediate_archive_delta"]) <= EPSILON
            for record in deduplicated),
        "dataset": str(args.out.resolve()),
        "dataset_sha256": _sha256(args.out),
        "per_state": state_summary,
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n",
        encoding="utf-8")
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
