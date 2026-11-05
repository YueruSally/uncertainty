#!/usr/bin/env python3
"""Held-out-repeat diagnostics only: no new evaluations, fitting, or PPO."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import statistics

from analyze_m_operator_oracle_audit import (
    OPERATORS, TOLERANCE, analyze_runs, read_jsonl,
)


def mean(values):
    return statistics.fmean(values) if values else None


def quantile(values, probability):
    if not values:
        return None
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (position - lower) * (ordered[upper] - ordered[lower])


def heldout_repeats(state):
    """Choose on K-1 replicates; score exclusively on the held-out replicate."""
    if not state["balanced_available"]:
        return []
    scores = {
        operator: row["population_delta_hv_by_replicate"]
        for operator, row in state["operators"].items()
    }
    count = len(scores["replace"])
    if count < 3:
        raise ValueError("cross-repeat diagnostic requires at least three repeats")
    folds = []
    for heldout in range(count):
        selection_means = {
            operator: mean([value for i, value in enumerate(values) if i != heldout])
            for operator, values in scores.items()
        }
        best = max(selection_means.values())
        tied = [operator for operator in OPERATORS
                if operator in scores and best - selection_means[operator] <= TOLERANCE]
        action = "replace" if "replace" in tied else tied[0]
        folds.append({
            "heldout_repeat": heldout,
            "selected_operator": action,
            "selection_mean_advantage": selection_means[action] - selection_means["replace"],
            "heldout_advantage": scores[action][heldout] - scores["replace"][heldout],
        })
    return folds


def metrics(states):
    decisions = [fold for state in states for fold in state["repeat_folds"]]
    values = [row["heldout_advantage"] for row in decisions]
    switched = [row["heldout_advantage"] for row in decisions
                if row["selected_operator"] != "replace"]
    # Repeats reuse states; their count is not an independent sample size.
    state_means = [mean([r["heldout_advantage"] for r in s["repeat_folds"]])
                   for s in states if s["repeat_folds"]]
    return {
        "state_count": len(state_means),
        "rotation_decision_count": len(decisions),
        "mean_net_gain_all_states": mean(state_means),
        "switch_fraction": len(switched) / len(decisions) if decisions else None,
        "mean_gain_when_switched": mean(switched),
        "switched_gain_p10": quantile(switched, 0.10),
        # Totals average over rotations so each state has total weight one.
        "positive_gain_total": sum(
            sum(max(0.0, r["heldout_advantage"]) for r in s["repeat_folds"])
            / len(s["repeat_folds"]) for s in states if s["repeat_folds"]),
        "negative_gain_total": sum(
            sum(min(0.0, r["heldout_advantage"]) for r in s["repeat_folds"])
            / len(s["repeat_folds"]) for s in states if s["repeat_folds"]),
        "selected_operator_counts": dict(Counter(row["selected_operator"] for row in decisions)),
    }


def repair_evidence(rows):
    if all(row.get("execution_contract") == "operator_shared_v1" for row in rows):
        return "shared_execution_contract_requires_consistency_check"
    if any(
        row.get("repair", {}).get("mutation_reverted")
        or row.get("repair", {}).get("repair_action_count", 0) > 0
        for row in rows
    ):
        return "legacy_repair_modified_child_replay_required"
    return "legacy_unverified_even_if_no_repair_change_logged"


def analyze(runs, expected_state_count=None):
    completeness, dataset = analyze_runs(
        runs, expected_replicates=3, expected_state_count=expected_state_count)
    if not completeness["completeness_pass"]:
        raise ValueError("input audit incomplete: " + json.dumps(completeness["excluded_states"][:5]))
    raw = defaultdict(list)
    inputs = []
    for run in map(Path, runs):
        for row in read_jsonl(run / "mutation_candidate_audit_outcomes.jsonl"):
            raw[(str(run), int(row["state_id"]))].append(row)
        for filename in ("mutation_candidate_audit_states.jsonl", "mutation_candidate_audit_outcomes.jsonl"):
            path = run / filename
            inputs.append({"path": str(path.resolve()),
                           "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    states = []
    for state in dataset:
        rows = raw[(state["run"], state["state_id"])]
        same_fingerprint = None
        if all(row.get("fingerprint") is not None for row in rows):
            same_fingerprint = all(len({row["fingerprint"] for row in rows
                                        if row["replicate"] == repeat}) == 1
                                   for repeat in range(3))
        state.update({
            "repeat_folds": heldout_repeats(state),
            "data_reuse_status": repair_evidence(rows),
            "all_replicate_contributions_zero": all(
                abs(row["population_delta_hv"]) <= TOLERANCE for row in rows),
            "all_operators_same_child_each_repeat": same_fingerprint,
        })
        states.append(state)

    comparable = [state for state in states if state["balanced_available"]]
    per_operator = {}
    for operator in OPERATORS:
        values = [(state, state["operators"][operator]["mean_advantage_over_balanced"])
                  for state in comparable if operator in state["operators"]]
        per_operator[operator] = {
            "comparable_legal_states": len(values),
            "mean_advantage": mean([value for _, value in values]),
            "positive_gain_total": sum(max(0.0, value) for _, value in values),
            "negative_gain_total": sum(min(0.0, value) for _, value in values),
            "worst_five": [{"run": state["run"], "state_id": state["state_id"],
                            "instance": state["instance"], "phase": state["phase"], "advantage": value}
                           for state, value in sorted(values, key=lambda item: item[1])[:5]
                           if value < -TOLERANCE],
        }
    overall = metrics(comparable)
    repair_comparisons = [row["legacy_repair_only_comparison"] for rows in raw.values()
                          for row in rows if row.get("legacy_repair_only_comparison") is not None]
    grouped = {}
    for key in ("instance", "run", "phase"):
        groups = defaultdict(list)
        for state in comparable:
            groups[state[key]].append(state)
        grouped["per_" + key] = {name: metrics(rows) for name, rows in sorted(groups.items())}
    gain = overall["mean_net_gain_all_states"]
    summary = {
        "completed": True,
        "definition": "select a legal operator using two repeats; evaluate on the third; rotate three times",
        "input_completeness_pass": True,
        "state_count": len(states),
        "balanced_comparable_state_count": len(comparable),
        "balanced_unavailable_state_count": len(states) - len(comparable),
        "counterfactual_evaluations_in_input": completeness["counterfactual_evaluation_count"],
        "new_objective_evaluations": 0,
        "new_training_steps": 0,
        "absolute_tie_tolerance": TOLERANCE,
        "tie_policy": "prefer Balanced; otherwise fixed operator order",
        "posthoc_mean_oracle_gain": completeness["mean_one_step_oracle_gain_over_balanced"],
        "heldout_repeat": overall,
        **grouped,
        "per_operator": per_operator,
        "tie_diagnostics_nonexclusive": {
            "mean_contributions_tied": sum(s["all_actions_tied"] for s in comparable),
            "all_replicate_contributions_zero": sum(s["all_replicate_contributions_zero"] for s in comparable),
            "same_child_each_repeat": sum(s["all_operators_same_child_each_repeat"] is True for s in comparable),
            "fingerprint_unavailable_states": sum(s["all_operators_same_child_each_repeat"] is None for s in comparable),
        },
        "data_reuse_status_counts": dict(Counter(s["data_reuse_status"] for s in states)),
        "same_mutated_child_repair_comparison": {
            "checked_outcomes": len(repair_comparisons),
            "different_fingerprints": sum(not row["same_child_fingerprint"] for row in repair_comparisons),
            "difference_fraction": mean([float(not row["same_child_fingerprint"]) for row in repair_comparisons]),
            "is_replay_of_old_states": False,
        },
        "training_data_approved": False,
        "next_action": ("REVIEW_ONE_STEP_SIGNAL_BEFORE_TRAINING" if gain is None or gain <= 0.0
                        else "VERIFY_EXECUTION_THEN_COLLECT_PLANNED_TRAJECTORIES"),
        "limitations": [
            "Old repair evidence cannot certify equality with search execution; saved observations are not full replayable parents.",
            "Three rotations are correlated and trajectories contain correlated states; no independent-sample confidence claim.",
            "Selection uses counterfactual feedback, not a deployable state predictor; no claim about final-search HV.",
            "A nonpositive result pauses this one-step approach; it does not prove no longer-horizon policy can improve.",
        ],
        "inputs": inputs,
    }
    return summary, states


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--expected-state-count", type=int)
    args = parser.parse_args(argv)
    if len({run.resolve() for run in args.runs}) != len(args.runs):
        parser.error("duplicate input run directory")
    if args.out.resolve() in {run.resolve() for run in args.runs}:
        parser.error("output must be separate from raw runs")
    try:
        summary, states = analyze(args.runs, args.expected_state_count)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))
    args.out.mkdir(parents=True, exist_ok=True)
    for name, data in (("REPEAT_CHECK.json", summary),
                       ("REPEAT_CHECK_STATES.json", {"states": states})):
        temporary = args.out / (name + ".tmp")
        temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        temporary.replace(args.out / name)
    print(json.dumps({key: summary[key] for key in (
        "completed", "state_count", "new_objective_evaluations", "heldout_repeat",
        "data_reuse_status_counts", "training_data_approved", "next_action")}, indent=2))


if __name__ == "__main__":
    main()
