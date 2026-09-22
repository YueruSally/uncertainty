#!/usr/bin/env python3
"""Build run-split, multi-label CCP100 mutation-location training data."""

import argparse
import json
from pathlib import Path
import re

import pandas as pd

from build_learning_dataset import file_sha256, parse_runs, selected_candidates
from mutation_logging import write_json
from segment_learning_features import FEATURES, feature_record


TARGETS = [
    "cost_improved", "emission_improved", "makespan_improved",
    "violation_reduced", "schedule_failure_introduced",
    "capacity_worsened", "pareto_promising", "selection_survivor",
    "nondominated_after_selection", "scenario_stable_cost",
    "scenario_stable_emission", "scenario_stable_makespan",
]

TARGET_DEFINITIONS = {
    "cost_improved": "feasible effective mutation decreases training q90 cost",
    "emission_improved": "feasible effective mutation decreases training q90 emissions",
    "makespan_improved": "feasible effective mutation decreases training q90 makespan",
    "violation_reduced": "effective mutation decreases normalized constraint violation",
    "schedule_failure_introduced": "effective mutation changes miss_tt from zero to positive",
    "capacity_worsened": "effective mutation introduces or increases capacity excess",
    "pareto_promising": "feasible effective mutation dominates its parent or makes a trade-off move",
    "selection_survivor": "effective offspring survives environmental selection",
    "nondominated_after_selection": "surviving effective offspring is feasible with NSGA-II rank zero",
    "scenario_stable_cost": "cost improves in at least the configured fraction of paired CCP100 scenarios",
    "scenario_stable_emission": "emissions improve in at least the configured fraction of paired CCP100 scenarios",
    "scenario_stable_makespan": "makespan improves in at least the configured fraction of paired CCP100 scenarios",
}

RUN_DIRECTORY = re.compile(r"^run(\d+)$")


def nested_run_number(path):
    """Infer run number from a run directory or one of its method children."""
    for part in reversed(path.parts):
        match = RUN_DIRECTORY.match(part)
        if match:
            return int(match.group(1))
    raise ValueError(f"Cannot infer run number from {path}")


def _value(mapping, key, default=0.0):
    value = (mapping or {}).get(key, default)
    return default if value is None else value


def build_row(event, candidate, number, split, run_name, stability_threshold=.6):
    context = {
        "operator": event["operator"], "generation": event["generation"],
        "phase": event["phase"], "feasible_before": event["feasible_before"],
        "q90_cost_before": event["q90_cost_before"],
        "q90_emission_before": event["q90_emission_before"],
        "q90_makespan_before": event["q90_makespan_before"],
        "violation_before": event["violation_before"],
        "violation_breakdown_before": event.get("violation_breakdown_before", {}),
        "candidate_count": event["candidate_count"],
    }
    row = feature_record(candidate, context)
    attributable = bool(event["outcome_attributable_to_selected_target"])
    effective = bool(event["effective_mutation"])
    selected_eligible = bool(candidate["eligible"])
    objective_eligible = bool(
        selected_eligible and event["objective_label_eligible"] and effective)
    offspring = event["phase"] == "offspring"
    before_vio, after_vio = event.get("violation_before"), event.get("violation_after")
    before_breakdown = event.get("violation_breakdown_before") or {}
    after_breakdown = event.get("violation_breakdown_after") or {}
    schedule_before = bool(event.get(
        "schedule_failure_before", _value(before_breakdown, "miss_tt") > 0))
    schedule_after = bool(event.get(
        "schedule_failure_after", _value(after_breakdown, "miss_tt") > 0))
    capacity_before = bool(event.get("capacity_infeasible_before",
        _value(before_breakdown, "cap_excess") > 0
        or _value(before_breakdown, "border_cap_excess") > 0))
    capacity_after = bool(event.get("capacity_infeasible_after",
        _value(after_breakdown, "cap_excess") > 0
        or _value(after_breakdown, "border_cap_excess") > 0))
    violation_eligible = bool(selected_eligible and attributable and effective
                              and before_vio is not None and after_vio is not None)
    common = {
        "run_number": number, "run_name": run_name, "split": split,
        "event_id": event["event_id"], "candidate_id": candidate["candidate_id"],
        "selected_target_eligible": selected_eligible,
        "outcome_attributable": attributable, "effective_mutation": effective,
        "objective_label_eligible": objective_eligible,
        "violation_label_eligible": violation_eligible,
        "offspring_label_eligible": bool(offspring and attributable and effective),
        "feasible_after": bool(event["feasible_after"]),
        "delta_cost": event.get("delta_cost"),
        "delta_emission": event.get("delta_emission"),
        "delta_makespan": event.get("delta_makespan"),
        "delta_violation": (before_vio - after_vio) if violation_eligible else None,
        "scenario_improvement_rate_cost": event.get("scenario_improvement_rate_cost"),
        "scenario_improvement_rate_emission": event.get("scenario_improvement_rate_emission"),
        "scenario_improvement_rate_makespan": event.get("scenario_improvement_rate_makespan"),
    }
    row.update(common)
    for name in ("cost", "emission", "makespan"):
        delta = event.get(f"delta_{name}")
        row[f"{name}_improved"] = bool(objective_eligible and delta is not None and delta > 0)
        rate = event.get(f"scenario_improvement_rate_{name}")
        row[f"scenario_stable_{name}"] = bool(
            objective_eligible and rate is not None and rate >= stability_threshold)
        row[f"scenario_stable_{name}_eligible"] = bool(
            objective_eligible and rate is not None)
    row["violation_reduced"] = bool(
        violation_eligible and before_vio - after_vio > 1e-12)
    row["schedule_failure_introduced"] = bool(
        violation_eligible and not schedule_before and schedule_after)
    row["capacity_worsened"] = bool(
        violation_eligible and (not capacity_before and capacity_after
        or _value(after_breakdown, "cap_excess")
           + _value(after_breakdown, "border_cap_excess")
           > _value(before_breakdown, "cap_excess")
             + _value(before_breakdown, "border_cap_excess") + 1e-12))
    row["pareto_promising"] = bool(
        objective_eligible and event["feasible_after"]
        and (event.get("child_dominates_before")
             or (event["feasible_before"] and event.get("tradeoff_move"))))
    row["selection_survivor"] = bool(
        offspring and effective and event["feasible_after"]
        and event.get("survived_environmental_selection"))
    row["nondominated_after_selection"] = bool(
        event.get("nondominated_after_selection", False))
    for target in ("cost_improved", "emission_improved", "makespan_improved",
                   "pareto_promising"):
        row[f"{target}_eligible"] = objective_eligible
    for target in ("violation_reduced", "schedule_failure_introduced",
                   "capacity_worsened"):
        row[f"{target}_eligible"] = violation_eligible
    row["selection_survivor_eligible"] = bool(
        selected_eligible and offspring and attributable and effective)
    row["nondominated_after_selection_eligible"] = bool(
        selected_eligible and offspring and attributable and effective
        and event.get("nondominated_after_selection") is not None)
    return row


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--pattern", default="random_final_v3_run*")
    parser.add_argument("--train-runs", default="1-7")
    parser.add_argument("--validation-runs", default="8-10")
    parser.add_argument("--stability-threshold", type=float, default=.6)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    train_runs, validation_runs = parse_runs(args.train_runs), parse_runs(args.validation_runs)
    if train_runs & validation_runs:
        parser.error("train-runs and validation-runs overlap")
    if not 0.0 <= args.stability_threshold <= 1.0:
        parser.error("stability-threshold must be between zero and one")
    if args.out.exists() and any(args.out.iterdir()):
        parser.error("output directory must be empty")
    args.out.mkdir(parents=True, exist_ok=True)
    expected = train_runs | validation_runs
    runs = sorted((p for p in args.runs_root.glob(args.pattern) if p.is_dir()),
                  key=nested_run_number)
    found = [nested_run_number(p) for p in runs]
    if len(found) != len(set(found)):
        raise ValueError(f"More than one input directory found for a run: {found}")
    if set(found) != expected:
        raise ValueError(f"Found runs {found}, expected {sorted(expected)}")
    records, manifest_runs = [], []
    for run in runs:
        number = nested_run_number(run)
        split = "train" if number in train_runs else "validation"
        complete = json.loads((run / "COMPLETE.json").read_text())
        config = json.loads((run / "configuration.json").read_text())
        if not complete["completed"] or complete["stop_reason"] != "evaluation_budget":
            raise ValueError(f"Run {number} is not a completed budget-controlled run")
        if config["method"] != "Random-CCP100":
            raise ValueError(f"Run {number} is not Random-CCP100 logging data")
        candidates, counts, probability_sums = selected_candidates(run / "mutation_candidates.jsonl")
        run_rows = []
        with (run / "mutation_events.jsonl").open(encoding="utf-8") as handle:
            for line in handle:
                event = json.loads(line)
                event_id = event["event_id"]
                if event["selection_policy"] != "random":
                    raise ValueError(f"Non-random logging policy in {event_id}")
                if event_id not in candidates or counts[event_id] != event["candidate_count"]:
                    raise ValueError(f"Candidate/event mismatch for {event_id}")
                if abs(probability_sums[event_id] - 1.0) > 1e-9:
                    raise ValueError(f"Baseline probabilities do not sum to one for {event_id}")
                run_rows.append(build_row(event, candidates[event_id], number, split,
                                          str(run.relative_to(args.runs_root)),
                                          args.stability_threshold))
        records.extend(run_rows)
        manifest_runs.append({
            "run_number": number, "run_name": str(run.relative_to(args.runs_root)),
            "split": split, "algorithm_seed": config["seeds"]["algorithm"],
            "training_seed": config["seeds"]["training"],
            "scenario_digest": config["scenario_digest"],
            "git_commit": config["git_commit"],
            "events": len(run_rows),
            "events_sha256": file_sha256(run / "mutation_events.jsonl"),
            "candidates_sha256": file_sha256(run / "mutation_candidates.jsonl"),
        })
    frame = pd.DataFrame(records)
    frame.to_csv(args.out / "segment_mutations.csv", index=False)
    target_summary = {}
    for target in TARGETS:
        eligible = frame[frame[f"{target}_eligible"].astype(bool)]
        target_summary[target] = {
            "eligible": int(len(eligible)),
            "positive": int(eligible[target].astype(bool).sum()),
            "prevalence": float(eligible[target].astype(bool).mean()) if len(eligible) else None,
            "by_split": {
                split: {
                    "eligible": int(len(part)),
                    "positive": int(part[target].astype(bool).sum()),
                    "prevalence": (float(part[target].astype(bool).mean())
                                   if len(part) else None),
                }
                for split, part in eligible.groupby("split")
            },
        }
    manifest = {
        "schema_version": 1, "oos_used": False,
        "counterfactuals_created": False,
        "stability_threshold": args.stability_threshold,
        "features": FEATURES, "targets": TARGETS,
        "target_definitions": TARGET_DEFINITIONS,
        "train_runs": sorted(train_runs), "validation_runs": sorted(validation_runs),
        "rows": len(frame), "target_summary": target_summary, "runs": manifest_runs,
    }
    write_json(args.out / "dataset_manifest.json", manifest)
    print(json.dumps({"rows": len(frame), "targets": target_summary}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
