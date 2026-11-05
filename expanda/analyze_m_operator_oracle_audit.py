#!/usr/bin/env python3
"""Summarize repeated five-operator counterfactuals without training a model."""
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
        json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def mean(values):
    return statistics.fmean(values) if values else 0.0


def standard_deviation(values):
    return statistics.stdev(values) if len(values) >= 2 else 0.0


def phase(fraction):
    if fraction < 1.0 / 3.0:
        return "early"
    if fraction < 2.0 / 3.0:
        return "middle"
    return "late"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    dataset = []
    missing = []
    for run in args.runs:
        state_path = run / "mutation_candidate_audit_states.jsonl"
        outcome_path = run / "mutation_candidate_audit_outcomes.jsonl"
        if not state_path.is_file() or not outcome_path.is_file():
            missing.append(str(run))
            continue
        states = {row["state_id"]: row for row in read_jsonl(state_path)}
        outcomes = defaultdict(list)
        for row in read_jsonl(outcome_path):
            outcomes[row["state_id"]].append(row)
        for state_id, state in sorted(states.items()):
            rows = outcomes.get(state_id, [])
            by_operator = defaultdict(list)
            for row in rows:
                by_operator[row["operator"]].append(row)
            if any(operator not in by_operator for operator in OPERATORS):
                continue
            summaries = {}
            replicate_scores = {}
            for operator in OPERATORS:
                operator_rows = by_operator[operator]
                scores = [float(row["population_delta_hv"])
                          for row in operator_rows]
                replicate_scores[operator] = {
                    int(row["replicate"]): float(row["population_delta_hv"])
                    for row in operator_rows
                }
                summaries[operator] = {
                    "replicates": len(scores),
                    "mean_population_delta_hv": mean(scores),
                    "standard_deviation_population_delta_hv": (
                        standard_deviation(scores)),
                    "minimum_population_delta_hv": min(scores),
                    "maximum_population_delta_hv": max(scores),
                    "success_fraction": mean([
                        float(bool(row["mutation_success"]))
                        for row in operator_rows]),
                    "feasible_fraction": mean([
                        float(bool(row["feasible"]))
                        for row in operator_rows]),
                }

            balanced = summaries["replace"]["mean_population_delta_hv"]
            for operator in OPERATORS:
                summaries[operator]["mean_advantage_over_balanced"] = (
                    summaries[operator]["mean_population_delta_hv"] - balanced)
            best_value = max(
                summaries[operator]["mean_population_delta_hv"]
                for operator in OPERATORS)
            tied_best = [
                operator for operator in OPERATORS
                if abs(summaries[operator]["mean_population_delta_hv"]
                       - best_value) <= TOLERANCE
            ]
            best_operator = "replace" if "replace" in tied_best else tied_best[0]
            best_nonbalanced = max(
                OPERATORS[:-1],
                key=lambda operator: (
                    summaries[operator]["mean_population_delta_hv"],
                    -OPERATORS.index(operator)))
            common_replicates = sorted(set(
                replicate_scores["replace"]
            ).intersection(replicate_scores[best_nonbalanced]))
            replicate_advantages = [
                replicate_scores[best_nonbalanced][replicate]
                - replicate_scores["replace"][replicate]
                for replicate in common_replicates
            ]
            positive_fraction = mean([
                float(value > TOLERANCE) for value in replicate_advantages])
            mean_nonbalanced_advantage = summaries[best_nonbalanced][
                "mean_advantage_over_balanced"]
            stable_nonbalanced = bool(
                mean_nonbalanced_advantage > TOLERANCE
                and positive_fraction >= 2.0 / 3.0)
            spread = best_value - min(
                summaries[operator]["mean_population_delta_hv"]
                for operator in OPERATORS)
            dataset.append({
                "run": str(run),
                "state_id": int(state_id),
                "instance": state["instance_id"],
                "generation": int(state["generation"]),
                "evaluation": int(state["evaluation"]),
                "evaluation_fraction": float(state["evaluation_fraction"]),
                "phase": phase(float(state["evaluation_fraction"])),
                "population_hv": float(state["population_hv"]),
                "observation": state["observation"],
                "operators": summaries,
                "best_operator": best_operator,
                "best_nonbalanced_operator": best_nonbalanced,
                "oracle_gain_over_balanced": best_value - balanced,
                "best_nonbalanced_advantage": mean_nonbalanced_advantage,
                "best_nonbalanced_positive_replicate_fraction": positive_fraction,
                "stable_nonbalanced_opportunity": stable_nonbalanced,
                "action_spread": spread,
                "all_actions_tied": spread <= TOLERANCE,
            })

    if missing:
        parser.error(f"missing audit logs: {missing}")
    if not dataset:
        parser.error("no complete five-operator audit states")

    state_count = len(dataset)
    phase_counts = Counter(row["phase"] for row in dataset)
    best_counts = Counter(row["best_operator"] for row in dataset)
    advantages = [row["oracle_gain_over_balanced"] for row in dataset]
    nonzero_advantages = [value for value in advantages if value > TOLERANCE]
    stable = [row for row in dataset if row["stable_nonbalanced_opportunity"]]
    per_operator = {}
    for operator in OPERATORS:
        values = [
            row["operators"][operator]["mean_advantage_over_balanced"]
            for row in dataset
        ]
        per_operator[operator] = {
            "mean_advantage_over_balanced": mean(values),
            "positive_state_fraction": mean([
                float(value > TOLERANCE) for value in values]),
            "negative_state_fraction": mean([
                float(value < -TOLERANCE) for value in values]),
        }

    summary = {
        "completed": True,
        "definition": (
            "repeated same-state CCP100 five-operator audit; Replace uses "
            "the frozen Balanced target rule and audit evaluations are outside "
            "the NSGA-II search budget"),
        "run_count": len(args.runs),
        "state_count": state_count,
        "operator_count": len(OPERATORS),
        "replicates_per_operator": min(
            entry["operators"][operator]["replicates"]
            for entry in dataset for operator in OPERATORS),
        "counterfactual_evaluation_count": sum(
            entry["operators"][operator]["replicates"]
            for entry in dataset for operator in OPERATORS),
        "phase_counts": dict(phase_counts),
        "best_operator_counts": dict(best_counts),
        "all_actions_tied_fraction": mean([
            float(row["all_actions_tied"]) for row in dataset]),
        "nonbalanced_oracle_fraction": mean([
            float(row["best_operator"] != "replace") for row in dataset]),
        "stable_nonbalanced_opportunity_count": len(stable),
        "stable_nonbalanced_opportunity_fraction": len(stable) / state_count,
        "mean_one_step_oracle_gain_over_balanced": mean(advantages),
        "median_positive_oracle_gain_over_balanced": (
            statistics.median(nonzero_advantages)
            if nonzero_advantages else 0.0),
        "maximum_oracle_gain_over_balanced": max(advantages),
        "per_operator": per_operator,
        "interpretation": {
            "is_full_search_oracle": False,
            "why": (
                "This measures a repeated one-step upper bound on sampled "
                "states. A full oracle rollout is justified only if stable "
                "non-Balanced opportunities are present."),
            "pretraining_recommended": bool(stable),
        },
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "OPERATOR_ORACLE_AUDIT.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n",
        encoding="utf-8")
    (args.out / "OPERATOR_COUNTERFACTUAL_DATASET.json").write_text(
        json.dumps({"completed": True, "states": dataset},
                   indent=2, allow_nan=False) + "\n",
        encoding="utf-8")
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
