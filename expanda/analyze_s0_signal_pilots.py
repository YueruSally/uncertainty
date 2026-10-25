#!/usr/bin/env python3
"""Analyze training-only C/M reward signal pilots without rerunning search."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import statistics


OUTCOME_FILE = {"C": "crossover_outcomes.jsonl", "M": "mutation_outcomes.jsonl"}
PHASES = ("early", "middle", "late")


def _read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _phase(evaluation: int, budget: int) -> str:
    if evaluation <= budget / 3:
        return "early"
    if evaluation <= 2 * budget / 3:
        return "middle"
    return "late"


def _outcome_evaluation(row: dict) -> int:
    """Return the action settlement evaluation across C and legacy M logs."""
    for name in ("evaluation", "evaluation_after", "ccp_evaluation_count"):
        value = row.get(name)
        if value is not None:
            return int(value)
    raise ValueError(
        "reward outcome has no evaluation, evaluation_after, or "
        "ccp_evaluation_count field")


def _mean(values):
    return float(statistics.fmean(values)) if values else None


def _ratio(numerator, denominator):
    return float(numerator / denominator) if denominator else None


def _validate(root: Path, component: str):
    summary = _read(root / "training_summary.json")
    config = _read(root / "training_configuration.json")
    if summary.get("completed") is not True or summary.get("component") != component:
        raise ValueError(f"incomplete or wrong {component} training directory: {root}")
    if config.get("configuration") != "S0" or config.get("evaluation_mode") != "deterministic":
        raise ValueError(f"{component} pilot is not deterministic S0 training")
    if any(not str(row.get("instance", "")).startswith("train-S0-")
           for row in summary.get("episodes", [])):
        raise ValueError(f"{component} pilot contains a non-training instance")
    return summary, config


def _analyze_component(root: Path, component: str):
    summary, config = _validate(root, component)
    budget = int(config["evaluation_budget_per_episode"])
    phase_rows = []
    update_rows = []
    for episode in summary["episodes"]:
        instance = episode["instance"]
        episode_root = root / "episodes" / instance
        grouped = {name: [] for name in PHASES}
        for row in _jsonl(episode_root / OUTCOME_FILE[component]):
            if row.get("reward_recorded_for_policy"):
                grouped[_phase(_outcome_evaluation(row), budget)].append(row)
        for phase in PHASES:
            rows = grouped[phase]
            rewards = [float(row.get("reward", 0.0) or 0.0) for row in rows]
            phase_rows.append({
                "component": component,
                "instance": instance,
                "phase": phase,
                "reward_events": len(rows),
                "positive_reward_events": sum(value > 0.0 for value in rewards),
                "positive_reward_fraction": _ratio(
                    sum(value > 0.0 for value in rewards), len(rows)),
                "reward_total": sum(rewards),
                "hv_delta_total": sum(
                    float(row.get("reward_hv_delta", 0.0) or 0.0) for row in rows),
                "mean_archive_size": _mean([
                    float(row.get("reward_archive_size", 0.0) or 0.0) for row in rows]),
            })
        updates = list(_jsonl(episode_root / "ppo_updates.jsonl"))
        for local_index, row in enumerate(updates, 1):
            transitions = int(row.get("transition_count", 0))
            positives = int(row.get("positive_reward_transition_count", 0))
            update_rows.append({
                "component": component,
                "instance": instance,
                "episode_update": local_index,
                "global_update": int(row["update"]),
                "transition_count": transitions,
                "positive_reward_transition_count": positives,
                "positive_reward_fraction": _ratio(positives, transitions),
                "reward_total": float(row.get("reward_total", 0.0) or 0.0),
                "policy_loss": row.get("policy_loss"),
                "value_loss": row.get("value_loss"),
                "entropy": row.get("entropy"),
                "kl": row.get("kl"),
                "update_seconds": float(row.get("update_seconds", 0.0) or 0.0),
            })
    return summary, config, phase_rows, update_rows


def _aggregate_phases(rows):
    result = []
    for component in ("C", "M"):
        for phase in PHASES:
            selected = [row for row in rows
                        if row["component"] == component and row["phase"] == phase]
            events = sum(row["reward_events"] for row in selected)
            positive = sum(row["positive_reward_events"] for row in selected)
            result.append({
                "component": component,
                "phase": phase,
                "reward_events": events,
                "positive_reward_events": positive,
                "positive_reward_fraction": _ratio(positive, events),
                "reward_total": sum(row["reward_total"] for row in selected),
                "hv_delta_total": sum(row["hv_delta_total"] for row in selected),
            })
    return result


def _update_bins(rows, bin_count=10):
    result = []
    for component in ("C", "M"):
        selected = [row for row in rows if row["component"] == component]
        selected.sort(key=lambda row: row["global_update"])
        for bin_index in range(bin_count):
            start = len(selected) * bin_index // bin_count
            end = len(selected) * (bin_index + 1) // bin_count
            current = selected[start:end]
            if not current:
                continue
            transitions = sum(row["transition_count"] for row in current)
            positives = sum(row["positive_reward_transition_count"] for row in current)
            result.append({
                "component": component,
                "bin": bin_index + 1,
                "progress_fraction": (bin_index + 0.5) / bin_count,
                "updates": len(current),
                "transition_count": transitions,
                "positive_reward_transition_count": positives,
                "positive_reward_fraction": _ratio(positives, transitions),
                "mean_reward_per_transition": _ratio(
                    sum(row["reward_total"] for row in current), transitions),
                "mean_entropy": _mean([
                    float(row["entropy"]) for row in current
                    if row.get("entropy") is not None]),
                "mean_kl": _mean([
                    float(row["kl"]) for row in current
                    if row.get("kl") is not None]),
            })
    return result


def _write_csv(path: Path, rows):
    rows = list(rows)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _svg(path: Path, phase_summary):
    width, height = 720, 360
    bars = {(row["component"], row["phase"]):
            float(row["positive_reward_fraction"] or 0.0) for row in phase_summary}
    maximum = max([0.01, *bars.values()])
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="360" y="24" text-anchor="middle" font-size="16">Positive reward fraction by search phase</text>',
        '<line x1="60" y1="310" x2="690" y2="310" stroke="black"/>',
        '<line x1="60" y1="45" x2="60" y2="310" stroke="black"/>',
    ]
    labels = [(component, phase) for component in ("C", "M") for phase in PHASES]
    colors = {"C": "#3b82f6", "M": "#ef4444"}
    for index, (component, phase) in enumerate(labels):
        value = bars[(component, phase)]
        x = 85 + index * 98
        bar_height = 240 * value / maximum
        y = 310 - bar_height
        elements.extend([
            f'<rect x="{x}" y="{y:.2f}" width="58" height="{bar_height:.2f}" fill="{colors[component]}"/>',
            f'<text x="{x + 29}" y="{y - 6:.2f}" text-anchor="middle" font-size="11">{value:.3f}</text>',
            f'<text x="{x + 29}" y="330" text-anchor="middle" font-size="11">{component}-{phase[0].upper()}</text>',
        ])
    elements.append('</svg>')
    path.write_text("\n".join(elements) + "\n", encoding="utf-8")


def analyze(c_training: Path, m_training: Path, out: Path):
    phase_rows, update_rows, provenance = [], [], {}
    for component, root in (("C", c_training), ("M", m_training)):
        summary, config, current_phases, current_updates = _analyze_component(
            root, component)
        phase_rows.extend(current_phases)
        update_rows.extend(current_updates)
        provenance[component] = {
            "training_root": str(root.resolve()),
            "episodes": len(summary["episodes"]),
            "evaluations": summary["total_actual_evaluations"],
            "ppo_updates": summary["episodes"][-1]["ppo_updates_total"],
            "population": config["population"],
            "evaluation_budget": config["evaluation_budget_per_episode"],
        }
    out.mkdir(parents=True, exist_ok=False)
    phase_summary = _aggregate_phases(phase_rows)
    bins = _update_bins(update_rows)
    result = {
        "completed": True,
        "stage": "training-only deterministic S0 C/M signal analysis",
        "test_instances_used": False,
        "phase_definition": "equal thirds of each episode evaluation budget",
        "provenance": provenance,
        "phase_summary": phase_summary,
        "outputs": {
            "phase_by_instance": "phase_by_instance.csv",
            "ppo_update_trace": "ppo_update_trace.csv",
            "ppo_update_bins": "ppo_update_bins.csv",
            "phase_chart": "positive_reward_by_phase.svg",
            "report": "analysis.md",
        },
    }
    (out / "analysis.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    _write_csv(out / "phase_by_instance.csv", phase_rows)
    _write_csv(out / "ppo_update_trace.csv", update_rows)
    _write_csv(out / "ppo_update_bins.csv", bins)
    _svg(out / "positive_reward_by_phase.svg", phase_summary)

    lines = [
        "# S0 C/M training-signal analysis", "",
        "This analysis uses only the three preselected S0 training instances.", "",
        "## Reward signal by search phase", "",
        "| Component | Phase | Events | Positive | Fraction | Reward total |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in phase_summary:
        lines.append(
            f"| {row['component']} | {row['phase']} | {row['reward_events']} | "
            f"{row['positive_reward_events']} | "
            f"{row['positive_reward_fraction']:.4f} | {row['reward_total']:.6f} |")
    lines.extend(["", "## Training volume", ""])
    for component in ("C", "M"):
        row = provenance[component]
        lines.append(
            f"- {component}: {row['episodes']} episodes, {row['evaluations']:,} "
            f"evaluations, {row['ppo_updates']} PPO updates.")
    lines.extend([
        "", "PPO update bins are chronological within each component and are not "
        "mapped back to exact evaluation phases; use `phase_by_instance.csv` for "
        "the evaluation-based early/middle/late comparison.", "",
    ])
    (out / "analysis.md").write_text("\n".join(lines), encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--c-training", type=Path, required=True)
    parser.add_argument("--m-training", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.c_training, args.m_training, args.out)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
