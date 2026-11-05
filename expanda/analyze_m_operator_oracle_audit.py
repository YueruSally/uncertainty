#!/usr/bin/env python3
"""Analyze repeated legal-operator feedback from saved logs, without evaluation."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import statistics


OPERATORS = ("add", "del", "mod", "mode", "replace")
TOLERANCE = 1e-12


def read_jsonl(path):
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def mean(values):
    return statistics.fmean(values) if values else None


def phase(fraction):
    if fraction < 1.0 / 3.0:
        return "early"
    if fraction < 2.0 / 3.0:
        return "middle"
    return "late"


def summarize_state(state, outcomes, expected_replicates=3):
    """Require feedback for the declared legal mask, never for illegal actions."""
    candidates = state["candidates"]
    by_id = {int(row["candidate_id"]): row for row in candidates}
    if len(by_id) != len(candidates):
        raise ValueError("duplicate candidate id")
    if (set(by_id) != set(range(len(OPERATORS)))
            or {row["candidate_operator"] for row in candidates} != set(OPERATORS)):
        raise ValueError("state must declare all five operator identities")
    legal_by_id = {index: row for index, row in by_id.items() if row["eligible"]}
    if not legal_by_id:
        raise ValueError("state has no legal operators")
    audited_ids = [int(index) for index in state["audited_candidate_ids"]]
    if len(set(audited_ids)) != len(audited_ids):
        raise ValueError("duplicate audited candidate id")
    if set(audited_ids) != set(legal_by_id):
        raise ValueError("audit does not cover exactly all legal operators")

    eligible_mask = state["observation"]["eligible"]
    if (len(eligible_mask) != len(candidates)
            or any(bool(eligible_mask[index]) != bool(row["eligible"])
                   for index, row in by_id.items())):
        raise ValueError("candidate eligibility disagrees with observation mask")

    by_operator = defaultdict(dict)
    for row in outcomes:
        candidate_id = int(row["candidate_id"])
        candidate = legal_by_id.get(candidate_id)
        if candidate is None:
            raise ValueError("outcome belongs to an illegal or unknown action")
        operator = row["operator"]
        if operator != candidate["candidate_operator"]:
            raise ValueError("outcome operator disagrees with candidate identity")
        replicate = int(row["replicate"])
        if replicate in by_operator[operator]:
            raise ValueError("duplicate operator replicate")
        score = float(row["population_delta_hv"])
        if not math.isfinite(score):
            raise ValueError("non-finite population HV contribution")
        by_operator[operator][replicate] = row

    legal_operators = [
        operator for operator in OPERATORS
        if any(row["candidate_operator"] == operator for row in legal_by_id.values())
    ]
    summaries = {}
    for operator in legal_operators:
        replicates = by_operator[operator]
        if set(replicates) != set(range(expected_replicates)):
            raise ValueError(f"missing or unexpected repeats for {operator}")
        rows = [replicates[index] for index in range(expected_replicates)]
        scores = [float(row["population_delta_hv"]) for row in rows]
        summaries[operator] = {
            "replicates": len(scores),
            "population_delta_hv_by_replicate": scores,
            "mean_population_delta_hv": mean(scores),
            "standard_deviation_population_delta_hv": (
                statistics.stdev(scores) if len(scores) >= 2 else 0.0),
            "minimum_population_delta_hv": min(scores),
            "maximum_population_delta_hv": max(scores),
            "success_fraction": mean([
                float(bool(row["mutation_success"])) for row in rows]),
            "feasible_fraction": mean([float(bool(row["feasible"])) for row in rows]),
        }

    best_value = max(row["mean_population_delta_hv"] for row in summaries.values())
    tied_best = [
        operator for operator in legal_operators
        if abs(summaries[operator]["mean_population_delta_hv"] - best_value) <= TOLERANCE
    ]
    best_operator = "replace" if "replace" in tied_best else tied_best[0]
    spread = best_value - min(row["mean_population_delta_hv"] for row in summaries.values())
    comparable = "replace" in summaries
    balanced = summaries["replace"]["mean_population_delta_hv"] if comparable else None
    for row in summaries.values():
        row["mean_advantage_over_balanced"] = (
            row["mean_population_delta_hv"] - balanced if comparable else None)

    alternatives = [operator for operator in legal_operators if operator != "replace"]
    best_nonbalanced = max(
        alternatives,
        key=lambda operator: summaries[operator]["mean_population_delta_hv"],
        default=None)
    repeat_advantages = []
    if comparable and best_nonbalanced is not None:
        repeat_advantages = [
            value - baseline
            for value, baseline in zip(
                summaries[best_nonbalanced]["population_delta_hv_by_replicate"],
                summaries["replace"]["population_delta_hv_by_replicate"])
        ]
    positive_fraction = mean([float(value > TOLERANCE) for value in repeat_advantages])
    nonbalanced_advantage = mean(repeat_advantages)
    repeat_consistent = bool(
        repeat_advantages and nonbalanced_advantage > TOLERANCE
        and positive_fraction >= 2.0 / 3.0)

    return {
        "state_id": int(state["state_id"]),
        "instance": state["instance_id"],
        "generation": int(state["generation"]),
        "evaluation": int(state["evaluation"]),
        "evaluation_fraction": float(state["evaluation_fraction"]),
        "phase": phase(float(state["evaluation_fraction"])),
        "population_hv": float(state["population_hv"]),
        "observation": state["observation"],
        "eligible_operators": legal_operators,
        "illegal_operators": [operator for operator in OPERATORS if operator not in summaries],
        "operators": summaries,
        # Null entries stay masked; an illegal action is never assigned zero reward.
        "action_order": list(OPERATORS),
        "mean_q_values": [
            summaries[operator]["mean_population_delta_hv"] if operator in summaries else None
            for operator in OPERATORS],
        "mean_relative_q_values": [
            summaries[operator]["mean_advantage_over_balanced"] if operator in summaries else None
            for operator in OPERATORS],
        "balanced_available": comparable,
        "best_operator": best_operator,
        "best_nonbalanced_operator": best_nonbalanced,
        "oracle_gain_over_balanced": best_value - balanced if comparable else None,
        "best_nonbalanced_advantage": nonbalanced_advantage,
        "best_nonbalanced_advantage_by_replicate": repeat_advantages,
        "best_nonbalanced_positive_replicate_fraction": positive_fraction,
        "stable_nonbalanced_opportunity": repeat_consistent,
        "action_spread": spread,
        "all_actions_tied": spread <= TOLERANCE,
    }


def analyze_runs(runs, expected_replicates=3, expected_state_count=None):
    dataset, excluded = [], []
    raw_state_count = raw_outcome_count = expected_outcome_count = 0
    for run in map(Path, runs):
        states = read_jsonl(run / "mutation_candidate_audit_states.jsonl")
        outcomes = read_jsonl(run / "mutation_candidate_audit_outcomes.jsonl")
        raw_state_count += len(states)
        raw_outcome_count += len(outcomes)
        by_state = defaultdict(list)
        known_ids = {int(row["state_id"]) for row in states}
        if len(known_ids) != len(states):
            raise ValueError(f"duplicate state ids under {run}")
        for row in outcomes:
            state_id = int(row["state_id"])
            if state_id not in known_ids:
                raise ValueError(f"outcome for unknown state {state_id} under {run}")
            by_state[state_id].append(row)
        for state in states:
            expected_outcome_count += (
                sum(bool(row["eligible"]) for row in state["candidates"]) * expected_replicates)
            try:
                entry = summarize_state(
                    state, by_state[int(state["state_id"])], expected_replicates)
            except (KeyError, TypeError, ValueError, IndexError) as error:
                excluded.append({
                    "run": str(run), "state_id": state.get("state_id"), "reason": str(error)})
                continue
            entry["run"] = str(run)
            dataset.append(entry)
    if not dataset:
        raise ValueError(f"no complete legal-operator states; exclusions={excluded[:5]}")

    comparable = [row for row in dataset if row["balanced_available"]]
    advantages = [row["oracle_gain_over_balanced"] for row in comparable]
    positive_advantages = [value for value in advantages if value > TOLERANCE]
    stable = [row for row in comparable if row["stable_nonbalanced_opportunity"]]
    per_operator = {}
    for operator in OPERATORS:
        available = [row for row in dataset if operator in row["operators"]]
        values = [
            row["operators"][operator]["mean_advantage_over_balanced"]
            for row in comparable if operator in row["operators"]]
        per_operator[operator] = {
            "eligible_state_count": len(available),
            "comparable_state_count": len(values),
            "mean_advantage_over_balanced": mean(values),
            "positive_state_fraction": mean([float(value > TOLERANCE) for value in values]),
            "negative_state_fraction": mean([float(value < -TOLERANCE) for value in values]),
        }

    completeness_pass = bool(
        not excluded and raw_state_count == len(dataset)
        and raw_outcome_count == expected_outcome_count
        and (expected_state_count is None or raw_state_count == expected_state_count))
    summary = {
        "schema_version": 2,
        "completed": True,
        "definition": (
            "repeated same-state CCP100 audit over each state's legal mutation operators; "
            "Replace uses Balanced; unavailable operators are masked and never imputed"),
        "run_count": len(runs),
        "raw_state_count": raw_state_count,
        "state_count": len(dataset),
        "excluded_state_count": len(excluded),
        "excluded_states": excluded,
        "expected_state_count": expected_state_count,
        "operator_count": len(OPERATORS),
        "legal_operator_count_distribution": dict(Counter(len(row["eligible_operators"]) for row in dataset)),
        "all_five_legal_state_count": sum(len(row["eligible_operators"]) == 5 for row in dataset),
        "replicates_per_operator": expected_replicates,
        "counterfactual_evaluation_count": raw_outcome_count,
        "expected_counterfactual_evaluation_count": expected_outcome_count,
        "analyzed_counterfactual_evaluation_count": sum(
            entry["operators"][operator]["replicates"]
            for entry in dataset for operator in entry["eligible_operators"]),
        "completeness_pass": completeness_pass,
        "phase_counts": dict(Counter(row["phase"] for row in dataset)),
        "balanced_comparable_state_count": len(comparable),
        "balanced_unavailable_state_count": len(dataset) - len(comparable),
        "best_operator_counts": dict(Counter(row["best_operator"] for row in comparable)),
        "all_actions_tied_fraction": mean([float(row["all_actions_tied"]) for row in comparable]),
        "nonbalanced_oracle_fraction": mean([float(row["best_operator"] != "replace") for row in comparable]),
        "stable_nonbalanced_opportunity_count": len(stable),
        "stable_nonbalanced_opportunity_fraction": len(stable) / len(comparable) if comparable else None,
        "mean_one_step_oracle_gain_over_balanced": mean(advantages),
        "median_positive_oracle_gain_over_balanced": statistics.median(positive_advantages) if positive_advantages else None,
        "maximum_oracle_gain_over_balanced": max(advantages) if advantages else None,
        "per_operator": per_operator,
        "interpretation": {
            "is_full_search_oracle": False,
            "metric_units": "absolute normalized population HV contribution, not final-HV percent",
            "sampling_policy": "frozen V6 Transformer operator policy",
            "repeat_consistency_is_independent_validation": False,
            "why": (
                "The maximum of sampled one-step mean contributions is optimistic and cannot bound "
                "long-term search performance. K=3 repeat agreement is descriptive; use fresh "
                "operator randomness and held-out trajectories before claiming stable advantage."),
            "pretraining_recommended": None,
            "next_step": (
                "validate opportunity size and action ranking on held-out feedback"
                if stable else
                "no repeat-consistent one-step advantage observed; inspect sparse feedback "
                "and longer-term effects before more training"),
        },
    }
    return summary, dataset


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--expected-replicates", type=int, default=3)
    parser.add_argument("--expected-state-count", type=int)
    args = parser.parse_args(argv)
    if args.expected_replicates < 1 or (args.expected_state_count is not None and args.expected_state_count < 1):
        parser.error("expected replicates and state count must be positive")
    try:
        summary, dataset = analyze_runs(args.runs, args.expected_replicates, args.expected_state_count)
    except (OSError, KeyError, TypeError, ValueError) as error:
        parser.error(str(error))
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "OPERATOR_ORACLE_AUDIT.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (args.out / "OPERATOR_COUNTERFACTUAL_DATASET.json").write_text(
        json.dumps({"schema_version": 2, "completed": True, "states": dataset},
                   indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
