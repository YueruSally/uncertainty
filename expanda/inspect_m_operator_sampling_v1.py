#!/usr/bin/env python3
"""Inspect existing policy opportunities and audit coverage. No search or fitting."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from analyze_m_operator_oracle_audit import analyze_runs, phase
from collect_m_operator_feedback_v1 import digest, json_hash, read_json, write_json


PHASES = ("early", "middle", "late")


def iter_jsonl(path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def phase_allocation(available, target=32):
    """Can the original 8..14 coverage rule be met, without looking at gains?"""
    if any(available.get(name, 0) < 8 for name in PHASES):
        return None
    capacities = {name: min(14, available[name]) for name in PHASES}
    if sum(capacities.values()) < target:
        return None
    counts = dict.fromkeys(PHASES, 8)
    while sum(counts.values()) < target:
        for name in PHASES:
            if counts[name] < capacities[name] and sum(counts.values()) < target:
                counts[name] += 1
    return counts


def inspect_run(run, target=32):
    config, result = read_json(run / "configuration.json"), read_json(run / "COMPLETE.json")
    if result.get("completed") is not True or result.get("mutation_control") != "operator":
        raise ValueError(f"requires a completed operator trajectory: {run}")
    budget = int(config["evaluation_budget"])
    policy_file = run / "mutation_policy_states.jsonl"
    available = Counter()
    policy_states = 0
    seen_ids = set()
    for row in iter_jsonl(policy_file):
        policy_states += 1
        identity = row["policy_state_id"]
        if identity in seen_ids or row["instance_id"] != config["instance_id"]:
            raise ValueError(f"duplicate/mismatched policy state: {run}: {identity}")
        seen_ids.add(identity)
        candidates = row["candidates"]
        mask = row["observation"]["eligible"]
        if len(mask) != len(candidates) or any(
                bool(mask[i]) != bool(candidate["eligible"]) for i, candidate in enumerate(candidates)):
            raise ValueError(f"policy candidate mask mismatch: {run}: {identity}")
        if sum(bool(value) for value in mask) >= 2:
            available[phase(float(row["evaluation"]) / budget)] += 1
    summary, _ = analyze_runs([run], expected_replicates=3)
    summary_matches = (
        result.get("candidate_audit_state_count") == summary["raw_state_count"]
        and result.get("candidate_audit_evaluation_count") == summary["counterfactual_evaluation_count"])
    allocation = phase_allocation(available, target)
    sampled = summary["phase_counts"]
    feedback_complete = summary["completeness_pass"]
    target_met = summary["state_count"] == target
    phase_met = all(8 <= sampled.get(name, 0) <= 14 for name in PHASES)
    if not summary_matches:
        diagnosis = "COMPLETION_COUNTERS_DISAGREE_WITH_SAVED_LOGS"
    elif not feedback_complete:
        diagnosis = "MISSING_OR_INVALID_OPERATOR_FEEDBACK"
    elif target_met and phase_met:
        diagnosis = "ORIGINAL_SAMPLE_CHECKS_MET"
    elif allocation is not None:
        diagnosis = "ENOUGH_LOGGED_OPPORTUNITIES_BUT_SAMPLER_MISSED_QUOTA"
    else:
        diagnosis = "ORIGINAL_PHASE_QUOTA_NOT_SUPPORTED_BY_LOGGED_OPPORTUNITIES"
    names = ("configuration.json", "COMPLETE.json", "mutation_policy_states.jsonl",
             "mutation_candidate_audit_states.jsonl", "mutation_candidate_audit_outcomes.jsonl")
    return {
        "instance": config["instance_id"], "algorithm_seed": config["seeds"]["algorithm"],
        "policy_state_count": policy_states,
        "eligible_opportunities_by_phase": {name: available[name] for name in PHASES},
        "sampled_states": summary["raw_state_count"], "sampled_phases": sampled,
        "recorded_state_feedback_complete": feedback_complete,
        "completion_counters_match_logs": summary_matches,
        "excluded_states": summary["excluded_states"],
        "feedback_rows": summary["counterfactual_evaluation_count"],
        "expected_feedback_rows_for_recorded_states": summary["expected_counterfactual_evaluation_count"],
        "sample_target_met": target_met, "original_phase_rule_met": phase_met,
        "possible_phase_allocation_from_logged_states": allocation,
        "diagnosis": diagnosis,
        "qualification": "logged eligible observations; successful replay has not been verified",
        "source_sha256": {name: digest(run / name) for name in names},
    }


def inspect_collection(root, batch=1):
    plan = read_json(root / "PLAN.json")
    seed = int(plan["batches"][str(batch)]["seed"])
    expected = int(plan["states_per_run"])
    rows, errors = [], []
    for instance in plan["instance_ids"]:
        run = root / "runs" / f"{instance}_seed_{seed}"
        try:
            request = read_json(root / "requests" / (run.name + ".json"))
            if request != {"plan_sha256": json_hash(plan), "instance": instance, "seed": seed}:
                raise ValueError("run request does not match saved plan")
            row = inspect_run(run, expected)
            if row["instance"] != instance or row["algorithm_seed"] != seed:
                raise ValueError("run identity differs from saved plan")
            rows.append(row)
        except (OSError, ValueError, KeyError, TypeError) as error:
            errors.append({"instance": instance, "error": str(error)})
    return {
        "diagnostic_completed": not errors, "new_objective_evaluations": 0,
        "new_training_steps": 0, "expected_states": expected * len(plan["instance_ids"]),
        "recorded_states_in_checked_runs": sum(row["sampled_states"] for row in rows),
        "saved_plan_sha256": json_hash(plan), "per_run": rows, "errors": errors,
        "training_data_approved": False,
        "original_completion_markers_unchanged": True,
        "next_action": "resolve sampling capacity before any replay; no automatic training or rerun",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--batch", type=int, choices=(1, 2), default=1)
    args = parser.parse_args(argv)
    report = inspect_collection(args.root, args.batch)
    write_json(args.root / f"sampling_check_batch_{args.batch}.json", report)
    visible = {key: value for key, value in report.items() if key != "per_run"}
    visible["per_run"] = [{key: row[key] for key in (
        "instance", "eligible_opportunities_by_phase", "sampled_states", "sampled_phases",
        "recorded_state_feedback_complete", "completion_counters_match_logs",
        "possible_phase_allocation_from_logged_states", "diagnosis")}
        for row in report["per_run"]]
    print(json.dumps(visible, indent=2))
    if report["errors"]:
        parser.exit(1, "Some trajectories could not be inspected; see errors in report.\n")


if __name__ == "__main__":
    main()
