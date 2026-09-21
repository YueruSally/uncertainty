#!/usr/bin/env python3
"""Join selected Random-CCP100 candidates to attributable mutation outcomes."""
import argparse
import hashlib
import json
from pathlib import Path
import re

import pandas as pd

from learning_features import FEATURES, feature_record
from mutation_logging import write_json


RUN_NUMBER = re.compile(r"(?:^|_)run(\d+)(?:_|$)")


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_runs(spec):
    values = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = (int(x) for x in part.split("-", 1))
            values.update(range(lo, hi + 1))
        else:
            values.add(int(part))
    return values


def run_number(path):
    match = RUN_NUMBER.search(path.name)
    if not match:
        raise ValueError(f"Cannot infer run number from {path.name!r}")
    return int(match.group(1))


def selected_candidates(path):
    selected = {}
    counts, probability_sums = {}, {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            event_id = row["event_id"]
            counts[event_id] = counts.get(event_id, 0) + 1
            probability_sums[event_id] = probability_sums.get(event_id, 0.0) + float(
                row.get("baseline_selection_probability", row["selection_probability"]))
            if row["chosen"]:
                if event_id in selected:
                    raise ValueError(f"Multiple chosen candidates for {event_id}")
                selected[event_id] = row
    return selected, counts, probability_sums


def build_row(event, candidate, number, split, run_name):
    context = {
        "operator": event["operator"], "generation": event["generation"],
        "phase": event["phase"], "feasible_before": event["feasible_before"],
        "q90_cost_before": event["q90_cost_before"],
        "q90_emission_before": event["q90_emission_before"],
        "q90_makespan_before": event["q90_makespan_before"],
        "violation_before": event["violation_before"],
    }
    row = feature_record(candidate, context)
    row.update({
        "run_number": number, "run_name": run_name, "split": split,
        "event_id": event["event_id"], "candidate_id": candidate["candidate_id"],
        "selected_target_eligible": bool(candidate["eligible"]),
        "outcome_attributable": bool(event["outcome_attributable_to_selected_target"]),
        "finite_objective_label": bool(event["finite_objective_label"]),
        "objective_label_eligible": bool(event["objective_label_eligible"]),
        "effective_mutation": bool(event["effective_mutation"]),
        "feasible_after": bool(event["feasible_after"]),
        "child_dominates_before": bool(event["child_dominates_before"]),
        "tradeoff_move": bool(event["tradeoff_move"]),
        "before_dominates_child": bool(event["before_dominates_child"]),
        "survived_environmental_selection": event["survived_environmental_selection"],
        "delta_cost": event["delta_cost"],
        "delta_emission": event["delta_emission"],
        "delta_makespan": event["delta_makespan"],
    })
    row["model_row_eligible"] = bool(
        row["selected_target_eligible"] and row["outcome_attributable"]
        and row["finite_objective_label"] and event["phase"] == "offspring"
        and row["survived_environmental_selection"] is not None)
    row["pareto_promising"] = bool(
        row["effective_mutation"] and row["feasible_after"]
        and (row["child_dominates_before"]
             or (event["feasible_before"] and row["tradeoff_move"])))
    row["selection_survivor"] = bool(
        row["effective_mutation"] and row["feasible_after"]
        and row["survived_environmental_selection"])
    return row


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, default=Path("expanda/learning_runs"))
    parser.add_argument("--pattern", default="random_final_v3_run*")
    parser.add_argument("--train-runs", default="1-7")
    parser.add_argument("--validation-runs", default="8-10")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    train_runs, validation_runs = parse_runs(args.train_runs), parse_runs(args.validation_runs)
    if train_runs & validation_runs:
        parser.error("train-runs and validation-runs overlap")
    if args.out.exists() and any(args.out.iterdir()):
        parser.error("output directory must be empty")
    args.out.mkdir(parents=True, exist_ok=True)
    runs = sorted((path for path in args.runs_root.glob(args.pattern) if path.is_dir()),
                  key=run_number)
    expected = train_runs | validation_runs
    if {run_number(x) for x in runs} != expected:
        raise ValueError(f"Found runs {[run_number(x) for x in runs]}, expected {sorted(expected)}")
    records, manifest_runs = [], []
    for run in runs:
        number = run_number(run)
        split = "train" if number in train_runs else "validation"
        complete = json.loads((run / "COMPLETE.json").read_text())
        config = json.loads((run / "configuration.json").read_text())
        if not complete["completed"] or complete["stop_reason"] != "evaluation_budget":
            raise ValueError(f"Run {number} is not a completed budget-controlled run")
        if config["method"] != "Random-CCP100":
            raise ValueError(f"Run {number} is not Random-CCP100 logging data")
        candidates, counts, probability_sums = selected_candidates(
            run / "mutation_candidates.jsonl")
        events = []
        with (run / "mutation_events.jsonl").open(encoding="utf-8") as handle:
            for line in handle:
                event = json.loads(line)
                event_id = event["event_id"]
                if event["selection_policy"] != "random":
                    raise ValueError(f"Non-random logging policy in {event_id}")
                if event_id not in candidates:
                    raise ValueError(f"Missing chosen candidate for {event_id}")
                if counts[event_id] != event["candidate_count"]:
                    raise ValueError(f"Candidate count mismatch for {event_id}")
                if abs(probability_sums[event_id] - 1.0) > 1e-9:
                    raise ValueError(f"Baseline probabilities do not sum to one for {event_id}")
                records.append(build_row(event, candidates[event_id], number, split, run.name))
                events.append(event)
        manifest_runs.append({
            "run_number": number, "run_name": run.name, "split": split,
            "algorithm_seed": config["seeds"]["algorithm"],
            "training_seed": config["seeds"]["training"],
            "scenario_digest": config["scenario_digest"],
            "git_commit": config["git_commit"], "events": len(events),
            "model_rows": sum(r["model_row_eligible"] for r in records if r["run_number"] == number),
            "events_sha256": file_sha256(run / "mutation_events.jsonl"),
            "candidates_sha256": file_sha256(run / "mutation_candidates.jsonl"),
        })
    frame = pd.DataFrame(records)
    frame.to_csv(args.out / "selected_mutations.csv", index=False)
    manifest = {
        "schema_version": 2, "target": "selection_survivor",
        "target_definition": ("offspring AND selected_target_eligible AND attributable AND "
                              "finite_objectives AND effective_mutation AND feasible_after "
                              "AND survived_environmental_selection"),
        "counterfactuals_created": False, "oos_used": False,
        "features": FEATURES, "train_runs": sorted(train_runs),
        "validation_runs": sorted(validation_runs), "rows": len(frame),
        "model_rows": int(frame["model_row_eligible"].sum()), "runs": manifest_runs,
    }
    write_json(args.out / "dataset_manifest.json", manifest)
    print({"rows": len(frame), "model_rows": manifest["model_rows"],
           "train": int((frame["split"] == "train").sum()),
           "validation": int((frame["split"] == "validation").sum())})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
