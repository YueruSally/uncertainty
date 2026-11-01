#!/usr/bin/env python3
"""Diagnose whether observable initial-population quality gates Balanced I."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics

from transformer_ppo.archive import (
    ObjectiveNormalizer,
    exact_hv_3d,
    nondominated,
)


EPSILON = 1e-12
TARGET_STRATEGY = "I_DIRECTED_BALANCED"


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write(path, value):
    Path(path).write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n",
        encoding="utf-8")


def _initial_metrics(run):
    run = Path(run)
    config = _read(run / "configuration.json")
    rows = _read(run / "initial_feasible_objectives.json")
    normalization = config["normalization"]
    normalizer = ObjectiveNormalizer(
        tuple(normalization["minimum"]),
        tuple(normalization["maximum"]),
        tuple(normalization["reference"]),
        tuple(normalization["objective_divisor"]),
    )
    points = [
        normalizer.transform(tuple(row["objectives"]))
        for row in rows
    ]
    front = nondominated(points)
    coverage = sum(
        max(point[axis] for point in front)
        - min(point[axis] for point in front)
        for axis in range(3)
    ) if front else 0.0
    return {
        "hypervolume": exact_hv_3d(points, normalizer.reference),
        "feasible_count": len(points),
        "nondominated_count": len(front),
        "coverage": float(coverage),
    }


def _correlation(rows, signal):
    x = [float(row[signal]) for row in rows]
    y = [float(row["final_paired_percent"]) for row in rows]
    if len(x) < 2 or max(x) - min(x) <= EPSILON:
        return None
    return float(statistics.correlation(x, y))


def _gate_summary(rows, name, predicate):
    selected = [row for row in rows if predicate(row)]
    selected_values = [row["final_paired_percent"] for row in selected]
    deployed = [
        row["final_paired_percent"] if predicate(row) else 0.0
        for row in rows
    ]
    return {
        "gate": name,
        "pairs": len(rows),
        "selected": len(selected),
        "selected_fraction": len(selected) / len(rows),
        "mean_deployed_final_percent": statistics.fmean(deployed),
        "mean_selected_final_percent": (
            statistics.fmean(selected_values) if selected_values else None),
        "minimum_selected_final_percent": (
            min(selected_values) if selected_values else None),
        "selected_wins": sum(value > EPSILON for value in selected_values),
        "selected_losses": sum(value < -EPSILON for value in selected_values),
        "selected_runs_below_minus_5": sum(
            value < -5.0 for value in selected_values),
    }


GATES = {
    "always_balanced": lambda row: True,
    "initial_hv_at_least_base": (
        lambda row: row["initial_hv_delta"] >= -EPSILON),
    "initial_feasible_count_at_least_base": (
        lambda row: row["initial_feasible_count_delta"] >= 0),
    "initial_nondominated_count_at_least_base": (
        lambda row: row["initial_nondominated_count_delta"] >= 0),
    "initial_coverage_at_least_base": (
        lambda row: row["initial_coverage_delta"] >= -EPSILON),
    "initial_hv_and_feasible_count_at_least_base": (
        lambda row: row["initial_hv_delta"] >= -EPSILON
        and row["initial_feasible_count_delta"] >= 0),
}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development-complete", type=Path, required=True)
    parser.add_argument("--holdout-complete", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    for path in (args.development_complete, args.holdout_complete):
        if not path.is_file():
            parser.error(f"missing COMPLETE.json: {path}")
    args.out.mkdir(parents=True, exist_ok=True)

    pairs = []
    for split, complete_path in (
            ("development", args.development_complete),
            ("holdout", args.holdout_complete)):
        result = _read(complete_path)
        indexed = {
            (row["instance"], int(row["algorithm_seed"]), row["strategy"]): row
            for row in result["runs"]
        }
        targets = [
            row for row in result["runs"]
            if row["strategy"] == TARGET_STRATEGY
        ]
        for target in targets:
            key = (target["instance"], int(target["algorithm_seed"]))
            base = indexed[(*key, "BASE")]
            base_initial = _initial_metrics(base["run_path"])
            target_initial = _initial_metrics(target["run_path"])
            pairs.append({
                "split": split,
                "instance": key[0],
                "algorithm_seed": key[1],
                "final_paired_percent": float(target["paired_percent"]),
                "base_initial": base_initial,
                "balanced_initial": target_initial,
                "initial_hv_delta": (
                    target_initial["hypervolume"]
                    - base_initial["hypervolume"]),
                "initial_feasible_count_delta": (
                    target_initial["feasible_count"]
                    - base_initial["feasible_count"]),
                "initial_nondominated_count_delta": (
                    target_initial["nondominated_count"]
                    - base_initial["nondominated_count"]),
                "initial_coverage_delta": (
                    target_initial["coverage"]
                    - base_initial["coverage"]),
            })

    signals = (
        "initial_hv_delta",
        "initial_feasible_count_delta",
        "initial_nondominated_count_delta",
        "initial_coverage_delta",
    )
    splits = {}
    for split_name in ("development", "holdout", "all"):
        rows = (
            pairs if split_name == "all" else
            [row for row in pairs if row["split"] == split_name]
        )
        splits[split_name] = {
            "pairs": len(rows),
            "signal_final_correlations": {
                signal: _correlation(rows, signal) for signal in signals
            },
            "gates": [
                _gate_summary(rows, name, predicate)
                for name, predicate in GATES.items()
            ],
        }

    result = {
        "completed": True,
        "definition": (
            "offline paired diagnosis of whether initial population quality "
            "can safely gate I_DIRECTED_BALANCED"),
        "pair_count": len(pairs),
        "splits": splits,
        "pairs": pairs,
    }
    _write(args.out / "COMPLETE.json", result)
    quick = {
        "completed": True,
        "definition": result["definition"],
        "pair_count": len(pairs),
        "splits": splits,
    }
    _write(args.out / "QUICK_RESULT.json", quick)
    print(json.dumps(quick, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
