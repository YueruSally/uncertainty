#!/usr/bin/env python3
"""Analyze fixed-state mutation candidate counterfactuals without rerunning search."""
from __future__ import annotations

import argparse
from collections import defaultdict
import itertools
import json
import math
from pathlib import Path
import random
import statistics


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON in {path}:{line_number}: {error}") from error


def mean(values):
    return float(statistics.fmean(values)) if values else None


def conditional(values):
    total = sum(values)
    if total <= 0.0:
        raise ValueError("audited probability mass must be positive")
    return [float(value / total) for value in values]


def expectation(probabilities, values):
    return float(sum(probability * value for probability, value in zip(
        probabilities, values)))


def average_ranks(values):
    result = [0.0] * len(values)
    ordered = sorted(range(len(values)), key=lambda index: values[index])
    offset = 0
    while offset < len(ordered):
        end = offset + 1
        while end < len(ordered) and values[ordered[end]] == values[ordered[offset]]:
            end += 1
        rank = (offset + end - 1) / 2.0
        for position in range(offset, end):
            result[ordered[position]] = rank
        offset = end
    return result


def correlation(left, right):
    if len(left) < 2 or len(set(left)) < 2 or len(set(right)) < 2:
        return None
    left_mean, right_mean = mean(left), mean(right)
    numerator = sum(
        (a - left_mean) * (b - right_mean) for a, b in zip(left, right))
    left_scale = sum((value - left_mean) ** 2 for value in left)
    right_scale = sum((value - right_mean) ** 2 for value in right)
    denominator = math.sqrt(left_scale * right_scale)
    return None if denominator <= 0.0 else float(numerator / denominator)


def shuffled_residual_expectations(prior, residual, values, sample_count=4096):
    """Evaluate the shuffled residual null without factorial blow-up.

    Exact enumeration is retained for small audits.  Larger joint path action
    sets use a deterministic Monte Carlo permutation sample so the report is
    reproducible and bounded in both time and memory.
    """
    count = len(residual)
    if count <= 8:
        permutations = itertools.permutations(residual)
    else:
        rng = random.Random(0)

        def sampled():
            indices = list(range(count))
            for _ in range(sample_count):
                rng.shuffle(indices)
                yield [residual[index] for index in indices]

        permutations = sampled()
    return [expectation(conditional([
        prior_value * residual_value
        for prior_value, residual_value in zip(prior, permutation)
    ]), values) for permutation in permutations]


def metric_comparison(prior, trained, values):
    prior_expected = expectation(prior, values)
    trained_expected = expectation(trained, values)
    residual = [
        trained_value / prior_value
        for prior_value, trained_value in zip(prior, trained)
    ]
    shuffled = shuffled_residual_expectations(prior, residual, values)
    oracle = max(values)
    trained_top = max(range(len(values)), key=lambda index: trained[index])
    return {
        "values": values,
        "prior_expected": prior_expected,
        "trained_expected": trained_expected,
        "trained_minus_prior": trained_expected - prior_expected,
        "shuffled_residual_expected_mean": mean(shuffled),
        "trained_minus_shuffled_mean": trained_expected - mean(shuffled),
        "oracle": oracle,
        "oracle_minus_prior": oracle - prior_expected,
        "oracle_minus_trained": oracle - trained_expected,
        "trained_top_is_oracle": bool(
            math.isclose(values[trained_top], oracle, rel_tol=0.0, abs_tol=1e-15)),
        "trained_probability_value_spearman": correlation(
            average_ranks(trained), average_ranks(values)),
    }


def analyze(run: Path):
    states_path = run / "mutation_candidate_audit_states.jsonl"
    outcomes_path = run / "mutation_candidate_audit_outcomes.jsonl"
    if not states_path.is_file() or not outcomes_path.is_file():
        raise ValueError(f"candidate audit logs are missing in {run}")
    states = {int(row["state_id"]): row for row in read_jsonl(states_path)}
    grouped = defaultdict(lambda: defaultdict(list))
    for row in read_jsonl(outcomes_path):
        grouped[int(row["state_id"])][int(row["candidate_id"])].append(row)
    if set(states) != set(grouped):
        raise ValueError("candidate audit state and outcome IDs differ")

    state_results = []
    for state_id in sorted(states):
        state = states[state_id]
        candidate_ids = [int(value) for value in state["audited_candidate_ids"]]
        if set(candidate_ids) != set(grouped[state_id]):
            raise ValueError(f"state {state_id} candidate IDs differ")
        candidates = []
        for candidate_id in candidate_ids:
            rows = grouped[state_id][candidate_id]
            candidates.append({
                "candidate_id": candidate_id,
                "replicates": len(rows),
                "baseline_probability": float(rows[0]["baseline_probability"]),
                "trained_probability": float(rows[0]["trained_probability"]),
                "feasible_fraction": mean([
                    float(bool(row["feasible"])) for row in rows]),
                "mean_normalized_violation": mean([
                    float(row["normalized_violation"]) for row in rows]),
                "mean_archive_delta_hv": mean([
                    float(row["archive_delta_hv"]) for row in rows]),
                "mean_single_point_hv": mean([
                    float(row["single_point_hv"]) for row in rows]),
                "distinct_fingerprints": len({row["fingerprint"] for row in rows}),
                "would_change_archive_fraction": mean([
                    float(bool(row["archive_diagnostics"]["would_change_archive"]))
                    for row in rows]),
            })
        prior = conditional([
            candidate["baseline_probability"] for candidate in candidates])
        trained = conditional([
            candidate["trained_probability"] for candidate in candidates])
        delta_values = [
            candidate["mean_archive_delta_hv"] for candidate in candidates]
        quality_values = [
            candidate["mean_single_point_hv"] for candidate in candidates]
        state_results.append({
            "state_id": state_id,
            "generation": state["generation"],
            "evaluation": state["evaluation"],
            "evaluation_fraction": state["evaluation_fraction"],
            "operator": state["operator"],
            "archive_hv": state["archive_hv"],
            "audited_prior_probability_mass": state[
                "audited_prior_probability_mass"],
            "audited_trained_probability_mass": state[
                "audited_trained_probability_mass"],
            "distinct_candidate_fingerprints": len({
                row["fingerprint"]
                for candidate_id in candidate_ids
                for row in grouped[state_id][candidate_id]
            }),
            "candidates": candidates,
            "conditional_prior": prior,
            "conditional_trained": trained,
            "archive_delta_hv": metric_comparison(prior, trained, delta_values),
            "single_point_hv": metric_comparison(prior, trained, quality_values),
        })

    def collect(metric, field):
        return [row[metric][field] for row in state_results
                if row[metric][field] is not None]

    return {
        "schema_version": 1,
        "run": str(run.resolve()),
        "state_count": len(state_results),
        "candidate_count": sum(len(row["candidates"]) for row in state_results),
        "counterfactual_evaluation_count": sum(
            candidate["replicates"]
            for row in state_results for candidate in row["candidates"]),
        "mean_audited_prior_probability_mass": mean([
            row["audited_prior_probability_mass"] for row in state_results]),
        "mean_audited_trained_probability_mass": mean([
            row["audited_trained_probability_mass"] for row in state_results]),
        "states_with_distinct_outcomes": sum(
            row["distinct_candidate_fingerprints"] > 1 for row in state_results),
        "states_with_positive_archive_gap": sum(
            row["archive_delta_hv"]["oracle_minus_prior"] > 1e-15
            for row in state_results),
        "states_trained_beats_prior_archive_delta": sum(
            row["archive_delta_hv"]["trained_minus_prior"] > 1e-15
            for row in state_results),
        "states_trained_beats_prior_quality": sum(
            row["single_point_hv"]["trained_minus_prior"] > 1e-15
            for row in state_results),
        "mean_archive_oracle_minus_prior": mean(collect(
            "archive_delta_hv", "oracle_minus_prior")),
        "mean_archive_trained_minus_prior": mean(collect(
            "archive_delta_hv", "trained_minus_prior")),
        "mean_quality_oracle_minus_prior": mean(collect(
            "single_point_hv", "oracle_minus_prior")),
        "mean_quality_trained_minus_prior": mean(collect(
            "single_point_hv", "trained_minus_prior")),
        "mean_quality_trained_minus_shuffled": mean(collect(
            "single_point_hv", "trained_minus_shuffled_mean")),
        "states": state_results,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    try:
        result = analyze(args.run)
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))
    output = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.out is not None:
        args.out.write_text(output, encoding="utf-8")
    print(output, end="")


if __name__ == "__main__":
    main()
