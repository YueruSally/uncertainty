#!/usr/bin/env python3
"""Audit the completed small-budget S0 learning-component experiment.

This command is read-only with respect to training and test artifacts.  It
aggregates the existing 50-episode I/C/M training runs and the repeated frozen
policy test without launching NSGA-II or updating a policy.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import statistics


COMPONENTS = ("I", "C", "M")
METHODS = ("BASE", "I", "C", "M", "ICM")
OUTCOME_LOG = {
    "I": "initialization_outcomes.jsonl",
    "C": "crossover_outcomes.jsonl",
    "M": "mutation_outcomes.jsonl",
}
CANDIDATE_LOG = {
    "I": "initialization_candidates.jsonl",
    "C": "crossover_candidates.jsonl",
    "M": "mutation_candidates.jsonl",
}


def _read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl(path: Path):
    if not path.is_file():
        return
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"invalid JSON in {path}:{line_number}: {error}") from error


def _atomic_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _mean(values):
    return float(statistics.fmean(values)) if values else None


def _ratio(numerator, denominator):
    return float(numerator / denominator) if denominator else None


def _sum_counts(rows):
    counts = Counter()
    for row in rows:
        counts.update(row or {})
    return dict(sorted(counts.items()))


def _training_audit(root: Path, component: str):
    summary_path = root / "training_summary.json"
    config_path = root / "training_configuration.json"
    if not summary_path.is_file() or not config_path.is_file():
        raise ValueError(f"incomplete {component} training directory: {root}")
    summary, config = _read(summary_path), _read(config_path)
    ppo_config = config.get("ppo") or {}
    episodes = summary.get("episodes", [])
    if (summary.get("completed") is not True
            or str(summary.get("component")).upper() != component):
        raise ValueError(f"wrong or incomplete {component} training summary: {root}")

    model_actions = sum(
        int(row.get("decision_counts", {}).get("model_choice", 0))
        for row in episodes)
    decision_counts = _sum_counts(
        row.get("decision_counts", {}) for row in episodes)
    ppo_updates = int(episodes[-1].get("ppo_updates_total", 0)) if episodes else 0
    event_count = positive_count = 0
    reward_total_from_events = 0.0
    missing_logs = []
    missing_configuration_logs = []
    missing_update_logs = []
    per_episode_events = []
    episode_evaluation_modes = set()
    episode_training_sizes = set()
    update_rows = []
    for episode in episodes:
        episode_id = episode["instance"]
        episode_root = root / "episodes" / episode_id
        episode_configuration = episode_root / "configuration.json"
        if episode_configuration.is_file():
            episode_config = _read(episode_configuration)
            if episode_config.get("evaluation_mode") is not None:
                episode_evaluation_modes.add(str(
                    episode_config["evaluation_mode"]))
            if episode_config.get("training_size") is not None:
                episode_training_sizes.add(int(episode_config["training_size"]))
        else:
            missing_configuration_logs.append(str(episode_configuration))
        update_path = episode_root / "ppo_updates.jsonl"
        if update_path.is_file():
            update_rows.extend(_jsonl(update_path))
        else:
            missing_update_logs.append(str(update_path))
        path = episode_root / OUTCOME_LOG[component]
        if not path.is_file():
            missing_logs.append(str(path))
            continue
        current_events = current_positive = 0
        for row in _jsonl(path):
            if not row.get("reward_recorded_for_policy", False):
                continue
            reward = float(row.get("reward", 0.0) or 0.0)
            current_events += 1
            current_positive += int(reward > 0.0)
            reward_total_from_events += reward
        event_count += current_events
        positive_count += current_positive
        per_episode_events.append(current_events)

    update_transition_counts = [
        int(row["transition_count"]) for row in update_rows
        if row.get("transition_count") is not None
    ]
    optimizer_step_counts = [
        int(row["optimizer_step_count"]) for row in update_rows
        if row.get("optimizer_step_count") is not None
    ]
    pre_step_logp_errors = [
        float(row["first_minibatch_pre_step_logp_abs_error_mean"])
        for row in update_rows
        if row.get("first_minibatch_pre_step_logp_abs_error_mean") is not None
    ]
    complete_optimizer_diagnostics = (
        bool(update_rows)
        and len(optimizer_step_counts) == len(update_rows)
        and len(pre_step_logp_errors) == len(update_rows)
    )
    deterministic_configuration_confirmed = (
        config.get("evaluation_mode") == "deterministic"
        and episode_evaluation_modes == {"deterministic"}
        and episode_training_sizes == {1}
        and not missing_configuration_logs
    )
    result = {
        "component": component,
        "training_root": str(root.resolve()),
        "architecture": config.get("architecture"),
        "actor_conditioning": config.get("actor_conditioning"),
        "configuration": config.get("configuration"),
        "training_evaluation_mode": config.get("evaluation_mode"),
        "configured_ppo_dropout": ppo_config.get("dropout"),
        "episode_evaluation_modes": sorted(episode_evaluation_modes),
        "episode_training_sizes": sorted(episode_training_sizes),
        "deterministic_configuration_confirmed": (
            deterministic_configuration_confirmed),
        "episode_count": len(episodes),
        "population": config.get("population"),
        "evaluation_budget_per_episode": config.get(
            "evaluation_budget_per_episode"),
        "total_actual_evaluations": summary.get("total_actual_evaluations"),
        "model_action_transition_count": model_actions,
        "effective_reward_event_count": event_count,
        "positive_reward_event_count": positive_count,
        "positive_reward_fraction": _ratio(positive_count, event_count),
        "reward_total_from_event_logs": reward_total_from_events,
        "reward_total_from_episode_summaries": float(sum(
            row.get("reward_total", 0.0) for row in episodes)),
        "positive_reward_count_from_episode_summaries": int(sum(
            row.get("positive_reward_actions", 0) for row in episodes)),
        "ppo_update_count": ppo_updates,
        "logged_ppo_update_count": len(update_rows),
        "ppo_update_log_matches_summary": len(update_rows) == ppo_updates,
        "logged_update_transition_count": sum(update_transition_counts),
        "update_transition_count_min": (
            min(update_transition_counts) if update_transition_counts else None),
        "update_transition_count_max": (
            max(update_transition_counts) if update_transition_counts else None),
        "mean_update_transition_count": _mean(update_transition_counts),
        "logged_optimizer_step_count": (
            sum(optimizer_step_counts) if complete_optimizer_diagnostics else None),
        "early_stopped_update_count": (
            sum(bool(row.get("early_stopped_on_target_kl")) for row in update_rows)
            if complete_optimizer_diagnostics else None),
        "mean_first_minibatch_pre_step_logp_abs_error": (
            _mean(pre_step_logp_errors) if complete_optimizer_diagnostics else None),
        "max_first_minibatch_pre_step_logp_abs_error": (
            max(pre_step_logp_errors) if complete_optimizer_diagnostics else None),
        "mean_model_actions_per_ppo_update": _ratio(model_actions, ppo_updates),
        "decision_counts": decision_counts,
        "mean_reward_events_per_episode": _mean(per_episode_events),
        "optimisation_seconds": float(sum(
            row.get("optimisation_seconds", 0.0) for row in episodes)),
        "inference_seconds": float(sum(
            row.get("inference_seconds", 0.0) for row in episodes)),
        "missing_outcome_log_count": len(missing_logs),
        "missing_outcome_logs": missing_logs[:10],
        "missing_episode_configuration_count": len(missing_configuration_logs),
        "missing_episode_configurations": missing_configuration_logs[:10],
        "missing_ppo_update_log_count": len(missing_update_logs),
        "missing_ppo_update_logs": missing_update_logs[:10],
        "logging_limit": (
            None if complete_optimizer_diagnostics else
            "This run predates optimizer-step and fixed-parameter pre-step logp "
            "diagnostics; per-update transition/loss/KL fields may still exist."),
    }
    result["non_inference_seconds"] = (
        result["optimisation_seconds"] - result["inference_seconds"])
    result["event_log_matches_episode_reward_count"] = (
        positive_count == result["positive_reward_count_from_episode_summaries"])
    result["event_log_matches_episode_reward_total"] = math.isclose(
        reward_total_from_events, result["reward_total_from_episode_summaries"],
        rel_tol=1e-10, abs_tol=1e-12)
    return result


def _test_run_roots(first_batch: Path, repeats: Path):
    roots = []
    for batch in [first_batch, *sorted((repeats / "repeats").glob("repeat_*"))]:
        roots.extend(sorted((batch / "instances").glob("test-S0-*")))
    return roots


def _semantic_position(component: str, row: dict):
    if component == "I":
        return f"path:{row.get('path_id')}"
    if component == "C":
        return f"node:{row.get('join_node')}"
    return "mutation:" + "|".join(str(row.get(name)) for name in (
        "operator", "batch_id", "allocation_index", "arc_index"))


def _candidate_file_audit(path: Path, component: str):
    events = defaultdict(list)
    for row in _jsonl(path):
        if row.get("decision_kind") == "model_choice":
            # I and C can contain several policy decisions in one environment
            # event.  Each candidate distribution must be audited separately.
            if component == "I":
                suffix = (row.get("batch_id"), row.get("selection_step"))
            elif component == "C":
                suffix = (row.get("cut_index"),)
            else:
                suffix = ()
            events[(str(row.get("event_id")), *suffix)].append(row)
    metrics = []
    positions = Counter()
    chosen_indices = Counter()
    for rows in events.values():
        probabilities = [float(row.get("selection_probability", 0.0) or 0.0)
                         for row in rows]
        chosen = [index for index, row in enumerate(rows) if row.get("chosen")]
        if len(chosen) != 1 or len(probabilities) < 2:
            continue
        total = sum(probabilities)
        if not math.isfinite(total) or total <= 0.0:
            continue
        probabilities = [value / total for value in probabilities]
        selected = chosen[0]
        entropy = -sum(value * math.log(value) for value in probabilities if value > 0.0)
        uniform = 1.0 / len(probabilities)
        metrics.append({
            "candidate_count": len(probabilities),
            "normalised_entropy": entropy / math.log(len(probabilities)),
            "total_variation_from_uniform": 0.5 * sum(
                abs(value - uniform) for value in probabilities),
            "selected_probability": probabilities[selected],
            "selected_is_argmax": (
                probabilities[selected] >= max(probabilities) - 1e-12),
        })
        chosen_indices[selected] += 1
        positions[_semantic_position(component, rows[selected])] += 1
    return metrics, positions, chosen_indices


def _test_action_audit(run_roots: list[Path], component: str):
    summaries = []
    decision_rows = []
    all_metrics = []
    positions = Counter()
    chosen_indices = Counter()
    fallback_count = 0
    shadow_comparisons = shadow_differences = 0
    missing_candidate_logs = 0
    for run_root in run_roots:
        method_root = run_root / component
        complete = method_root / "COMPLETE.json"
        if not complete.is_file():
            raise ValueError(f"missing test completion marker: {complete}")
        summary = _read(complete)
        summaries.append(summary)
        decision_rows.append(summary.get("decision_counts", {}))
        candidates = method_root / CANDIDATE_LOG[component]
        if candidates.is_file():
            metrics, current_positions, indices = _candidate_file_audit(
                candidates, component)
            all_metrics.extend(metrics)
            positions.update(current_positions)
            chosen_indices.update(indices)
        else:
            missing_candidate_logs += 1
        outcomes = method_root / OUTCOME_LOG[component]
        for row in _jsonl(outcomes):
            metadata = row.get("policy_metadata") or {}
            fallback_count += int(metadata.get("fallback") is not None)
            if component == "I" and row.get("model_decisions", 0):
                shadow_comparisons += 1
                shadow_differences += int(row.get("different_from_shadow_base", False))
            elif component == "C":
                for cut in row.get("selected_cuts", []):
                    if cut.get("decision_kind") == "model_choice":
                        shadow_comparisons += 1
                        shadow_differences += int(
                            cut.get("different_from_shadow_base", False))
            elif component == "M" and row.get("decision_kind") == "model_choice":
                different = row.get("different_from_shadow_base_batch")
                if different is not None:
                    shadow_comparisons += 1
                    shadow_differences += int(different)

    total_seconds = sum(float(row.get("optimisation_seconds", 0.0))
                        for row in summaries)
    inference_seconds = sum(float(row.get("inference_seconds", 0.0))
                            for row in summaries)
    metric = lambda name: _mean([row[name] for row in all_metrics])
    total_positions = sum(positions.values())
    return {
        "component": component,
        "run_count": len(summaries),
        "evaluation_count_min": min(
            int(row["real_objective_evaluation_count"]) for row in summaries),
        "evaluation_count_max": max(
            int(row["real_objective_evaluation_count"]) for row in summaries),
        "decision_counts": _sum_counts(decision_rows),
        "fallback_count": fallback_count,
        "model_choice_event_count_from_candidate_logs": len(all_metrics),
        "mean_candidate_count": metric("candidate_count"),
        "mean_normalised_entropy": metric("normalised_entropy"),
        "mean_total_variation_from_uniform": metric(
            "total_variation_from_uniform"),
        "mean_selected_probability": metric("selected_probability"),
        "argmax_selection_fraction": _mean([
            float(row["selected_is_argmax"]) for row in all_metrics]),
        "shadow_random_comparison_count": shadow_comparisons,
        "different_from_shadow_random_fraction": _ratio(
            shadow_differences, shadow_comparisons),
        "most_common_semantic_positions": [
            {"position": name, "count": count,
             "fraction": _ratio(count, total_positions)}
            for name, count in positions.most_common(10)
        ],
        "chosen_candidate_index_counts": dict(sorted(chosen_indices.items())),
        "optimisation_seconds": total_seconds,
        "inference_seconds": inference_seconds,
        "non_inference_seconds": total_seconds - inference_seconds,
        "inference_fraction": _ratio(inference_seconds, total_seconds),
        "missing_candidate_log_count": missing_candidate_logs,
    }


def _method_time_and_budget(run_roots: list[Path]):
    result = {}
    for method in METHODS:
        summaries = [_read(root / method / "COMPLETE.json") for root in run_roots]
        evaluations = [int(row["real_objective_evaluation_count"]) for row in summaries]
        optimisation = [float(row.get("optimisation_seconds", 0.0)) for row in summaries]
        inference = [float(row.get("inference_seconds", 0.0)) for row in summaries]
        result[method] = {
            "run_count": len(summaries),
            "evaluation_count_min": min(evaluations),
            "evaluation_count_max": max(evaluations),
            "all_runs_have_equal_declared_budget": len(set(evaluations)) == 1,
            "mean_optimisation_seconds": _mean(optimisation),
            "total_optimisation_seconds": sum(optimisation),
            "mean_inference_seconds": _mean(inference),
            "mean_non_inference_seconds": _mean([
                total - model for total, model in zip(optimisation, inference)]),
            "timing_limit": (
                "Existing logs combine objective evaluation, feature construction, "
                "and other NSGA-II work in non_inference_seconds."),
        }
    base = result["BASE"]["mean_optimisation_seconds"]
    for row in result.values():
        row["mean_runtime_ratio_vs_base"] = _ratio(
            row["mean_optimisation_seconds"], base)
    return result


def _markdown(audit: dict):
    lines = [
        "# S0 learning-components stage-1 audit",
        "",
        "This report audits existing artifacts only; it does not launch or train a model.",
        "",
        "## Budget and provenance",
        "",
        f"- Population: {audit['test_design']['population']}",
        f"- Evaluations per independent run: {audit['test_design']['evaluation_budget']}",
        f"- Algorithm repeats per instance: {audit['test_design']['algorithm_repeats_per_instance']}",
        f"- Policy/training seed: {audit['test_design']['policy_seed']}",
        "- Status: small-budget deterministic screening; not the final full-scale experiment.",
        "",
        "## Training configuration check",
        "",
        "| Component | Architecture | Mode | Scene | PPO dropout | Episode modes | Training sizes | Deterministic confirmed |",
        "|---|---|---|---|---:|---|---|---|",
    ]
    for component in COMPONENTS:
        row = audit["training"][component]
        modes = ", ".join(row["episode_evaluation_modes"]) or "missing"
        sizes = ", ".join(map(str, row["episode_training_sizes"])) or "missing"
        lines.append(
            f"| {component} | {row['architecture']} | "
            f"{row['training_evaluation_mode']} | {row['configuration']} | "
            f"{row['configured_ppo_dropout']} | {modes} | {sizes} | "
            f"{row['deterministic_configuration_confirmed']} |")
    lines.extend([
        "",
        "## Training signal",
        "",
        "| Component | Episodes | Model actions | Reward events | Positive rewards | Positive fraction | PPO updates | Actions/update |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for component in COMPONENTS:
        row = audit["training"][component]
        fraction = row["positive_reward_fraction"]
        actions = row["mean_model_actions_per_ppo_update"]
        lines.append(
            f"| {component} | {row['episode_count']} | "
            f"{row['model_action_transition_count']} | "
            f"{row['effective_reward_event_count']} | "
            f"{row['positive_reward_event_count']} | "
            f"{fraction:.4f} | {row['ppo_update_count']} | {actions:.2f} |")
    lines.extend([
        "",
        "## Frozen-policy test actions",
        "",
        "| Component | Model-choice events | Single | No candidate | Mean candidates | Normalised entropy | TV from uniform | Differs from paired random |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for component in COMPONENTS:
        row = audit["test_actions"][component]
        counts = row["decision_counts"]
        lines.append(
            f"| {component} | {row['model_choice_event_count_from_candidate_logs']} | "
            f"{counts.get('single_candidate', 0)} | {counts.get('no_candidate', 0)} | "
            f"{row['mean_candidate_count']:.2f} | "
            f"{row['mean_normalised_entropy']:.4f} | "
            f"{row['mean_total_variation_from_uniform']:.4f} | "
            f"{row['different_from_shadow_random_fraction']:.4f} |")
    lines.extend([
        "",
        "The shadow-random mismatch is diagnostic only. Its chance expectation "
        "increases with candidate-set size, so it is not evidence of learning by itself.",
        "",
        "## Runtime and equal-budget check",
        "",
        "| Method | Runs | Evaluation range | Mean seconds | Mean inference | Ratio vs Base |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for method in METHODS:
        row = audit["test_time_and_budget"][method]
        lines.append(
            f"| {method} | {row['run_count']} | "
            f"{row['evaluation_count_min']}–{row['evaluation_count_max']} | "
            f"{row['mean_optimisation_seconds']:.3f} | "
            f"{row['mean_inference_seconds']:.3f} | "
            f"{row['mean_runtime_ratio_vs_base']:.2f} |")
    lines.extend([
        "",
        "Existing logs do not separate objective evaluation from feature construction. "
        "That split requires new timers and a training-only diagnostic rerun.",
        "",
    ])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--i-training", type=Path, required=True)
    parser.add_argument("--c-training", type=Path, required=True)
    parser.add_argument("--m-training", type=Path, required=True)
    parser.add_argument("--first-test-batch", type=Path, required=True)
    parser.add_argument("--test-repeats", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    training_roots = {
        "I": args.i_training.resolve(),
        "C": args.c_training.resolve(),
        "M": args.m_training.resolve(),
    }
    first_batch = args.first_test_batch.resolve()
    repeats = args.test_repeats.resolve()
    out = args.out.resolve()
    if out.exists() and any(out.iterdir()):
        parser.error("audit output directory must be empty")
    try:
        repeat_complete = _read(repeats / "MULTISEED_COMPLETE.json")
        first_complete = _read(first_batch / "BATCH_COMPLETE.json")
        if repeat_complete.get("completed") is not True:
            raise ValueError("multi-seed test is incomplete")
        run_roots = _test_run_roots(first_batch, repeats)
        expected = int(repeat_complete["algorithm_repeats_per_instance"]) * int(
            repeat_complete["independent_instance_count"])
        if len(run_roots) != expected:
            raise ValueError(
                f"expected {expected} complete instance runs, found {len(run_roots)}")
        time_and_budget = _method_time_and_budget(run_roots)
        audit = {
            "schema_version": 1,
            "artifact_role": "existing-results audit; no optimisation or training",
            "test_design": {
                "population": repeat_complete["population"],
                "evaluation_budget": repeat_complete["evaluation_budget"],
                "algorithm_repeats_per_instance": repeat_complete[
                    "algorithm_repeats_per_instance"],
                "independent_instance_count": repeat_complete[
                    "independent_instance_count"],
                "policy_seed": repeat_complete["policy_seed"],
                "methods": repeat_complete["methods"],
                "first_batch_git_commit": first_complete.get("git_commit"),
                "repeats_git_commit": repeat_complete.get("git_commit"),
            },
            "training": {
                component: _training_audit(training_roots[component], component)
                for component in COMPONENTS
            },
            "test_actions": {
                component: _test_action_audit(run_roots, component)
                for component in COMPONENTS
            },
            "test_time_and_budget": time_and_budget,
            "equal_evaluation_count_across_all_methods": len({
                (row["evaluation_count_min"], row["evaluation_count_max"])
                for row in time_and_budget.values()
            }) == 1,
            "limits": [
                "Existing logs do not persist per-update PPO batch size/loss/KL traces.",
                "Existing timers do not separate objective evaluation and feature construction.",
                "Candidate-index frequencies are diagnostic because candidate sets vary by event.",
            ],
        }
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))
    out.mkdir(parents=True, exist_ok=True)
    _atomic_json(out / "audit.json", audit)
    (out / "audit.md").write_text(_markdown(audit), encoding="utf-8")
    _atomic_json(out / "COMPLETE.json", {
        "completed": True,
        "audit": "audit.json",
        "report": "audit.md",
        "training_episode_count": {
            component: audit["training"][component]["episode_count"]
            for component in COMPONENTS
        },
        "test_instance_run_count": len(run_roots),
        "launched_optimisation": False,
        "updated_policy": False,
    })
    print(json.dumps({
        "completed": True,
        "audit": str(out / "audit.json"),
        "report": str(out / "audit.md"),
        "test_instance_run_count": len(run_roots),
    }, indent=2))


if __name__ == "__main__":
    main()
