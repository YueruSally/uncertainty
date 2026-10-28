#!/usr/bin/env python3
"""Analyze paired Base/I/C/M S0 V2 development validation runs."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import statistics


METHODS = ("BASE", "I", "C", "M")


def _read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _atomic_json(path: Path, value):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n",
        encoding="utf-8")
    temporary.replace(path)


def _load_pilot(root: Path, component: str, plan):
    complete = _read(root / "COMPLETE.json")
    config = _read(root / "pilot_configuration.json")
    rows = complete.get("runs", [])
    if (complete.get("completed") is not True
            or complete.get("run_count") != 9
            or len(rows) != 9
            or config.get("component") != component
            or config.get("selection_digest") != plan["selection_digest"]
            or config.get("normalization_digest")
            != plan["normalization"]["normalization_digest"]
            or config.get("test_instances_used") is not False):
        raise ValueError(f"invalid or mismatched {component} pilot: {root}")
    if component in ("C", "M"):
        probability = config.get("learning_control", {}).get(
            "model_choice_probability")
        if probability != 0.5:
            raise ValueError(f"{component} pilot is not q=0.5: {probability}")
    if component == "I":
        top_k = config.get("learning_control", {}).get(
            "initialization_rule_top_k")
        if top_k is not None:
            raise ValueError(f"I pilot is not I-All: top_k={top_k}")
    result = {}
    for row in rows:
        key = (row["role"], int(row["repeat"]), int(row["algorithm_seed"]))
        hv = float(row["final_hypervolume"])
        if key in result or not math.isfinite(hv):
            raise ValueError(f"invalid or duplicate run in {root}: {key}")
        result[key] = {**row, "final_hypervolume": hv}
    return result


def paired_summary(method, values, base):
    if set(values) != set(base):
        raise ValueError(f"paired keys differ for {method}")
    candidate_hv = [values[key]["final_hypervolume"] for key in sorted(base)]
    base_hv = [base[key]["final_hypervolume"] for key in sorted(base)]
    deltas = [candidate - control
              for candidate, control in zip(candidate_hv, base_hv)]
    percentages = [100.0 * delta / control
                   for delta, control in zip(deltas, base_hv)]
    wins = sum(value > 1e-12 for value in deltas)
    losses = sum(value < -1e-12 for value in deltas)
    return {
        "method": method,
        "run_count": len(candidate_hv),
        "mean_hv": statistics.mean(candidate_hv),
        "base_mean_hv": statistics.mean(base_hv),
        "mean_paired_delta_hv": statistics.mean(deltas),
        "mean_paired_delta_percent": statistics.mean(percentages),
        "paired_delta_standard_deviation": statistics.stdev(deltas),
        "wins": wins,
        "ties": len(deltas) - wins - losses,
        "losses": losses,
    }


def analyze(plan_path: Path, roots, out: Path):
    plan = _read(plan_path)
    if (plan.get("stage")
            != "training-only S0 V2 initialization top-k ablation"
            or plan.get("test_instances_used") is not False):
        raise ValueError("invalid training-only V2 development plan")
    pilots = {
        method: _load_pilot(Path(roots[method]), method, plan)
        for method in METHODS
    }
    keys = set(pilots["BASE"])
    if any(set(pilots[method]) != keys for method in METHODS[1:]):
        raise ValueError("the four methods do not have identical paired runs")

    base_values = [pilots["BASE"][key]["final_hypervolume"]
                   for key in sorted(keys)]
    base_summary = {
        "method": "BASE", "run_count": len(keys),
        "mean_hv": statistics.mean(base_values),
        "base_mean_hv": statistics.mean(base_values),
        "mean_paired_delta_hv": 0.0,
        "mean_paired_delta_percent": 0.0,
        "paired_delta_standard_deviation": 0.0,
        "wins": 0, "ties": len(keys), "losses": 0,
    }
    summaries = [base_summary] + [
        paired_summary(method, pilots[method], pilots["BASE"])
        for method in METHODS[1:]
    ]

    rows = []
    for key in sorted(keys):
        row = {
            "role": key[0],
            "repeat": key[1],
            "algorithm_seed": key[2],
            "instance_id": pilots["BASE"][key]["instance_id"],
        }
        for method in METHODS:
            row[f"{method}_hv"] = pilots[method][key]["final_hypervolume"]
        for method in METHODS[1:]:
            row[f"{method}_delta_percent"] = 100.0 * (
                row[f"{method}_hv"] / row["BASE_hv"] - 1.0)
        rows.append(row)

    role_summaries = []
    for role in ("low", "median", "high"):
        selected = [row for row in rows if row["role"] == role]
        for method in METHODS:
            values = [row[f"{method}_hv"] for row in selected]
            base_role = [row["BASE_hv"] for row in selected]
            deltas = [value - control
                      for value, control in zip(values, base_role)]
            wins = sum(value > 1e-12 for value in deltas)
            losses = sum(value < -1e-12 for value in deltas)
            role_summaries.append({
                "role": role,
                "method": method,
                "mean_hv": statistics.mean(values),
                "mean_delta_percent": statistics.mean([
                    100.0 * delta / control
                    for delta, control in zip(deltas, base_role)]),
                "wins": wins,
                "ties": len(deltas) - wins - losses,
                "losses": losses,
            })

    best = max(summaries, key=lambda row: row["mean_hv"])["method"]
    result = {
        "completed": True,
        "stage": "training-only S0 V2 component development validation",
        "test_instances_used": False,
        "plan_digest": plan["plan_digest"],
        "pair_count_per_method": len(keys),
        "methods": list(METHODS),
        "summaries": summaries,
        "role_summaries": role_summaries,
        "highest_mean_hv_method": best,
        "interpretation_scope": (
            "development validation only; not a final-test conclusion"),
    }
    out.mkdir(parents=True, exist_ok=False)
    _atomic_json(out / "analysis.json", result)
    with (out / "paired_hv.csv").open(
            "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# S0 V2 component development validation",
        "",
        "Training-only paired validation; no final test instances used.",
        "",
        "| Method | Mean HV | Mean paired delta % | W-T-L | Delta SD |",
        "|---|---:|---:|:---:|---:|",
    ]
    for row in summaries:
        lines.append(
            f"| {row['method']} | {row['mean_hv']:.6f} | "
            f"{row['mean_paired_delta_percent']:+.3f}% | "
            f"{row['wins']}-{row['ties']}-{row['losses']} | "
            f"{row['paired_delta_standard_deviation']:.6f} |")
    lines.extend(["", "## By complexity role", ""])
    for role in ("low", "median", "high"):
        lines.extend([
            f"### {role}", "",
            "| Method | Mean HV | Mean delta % | W-T-L |",
            "|---|---:|---:|:---:|",
        ])
        for row in role_summaries:
            if row["role"] == role:
                lines.append(
                    f"| {row['method']} | {row['mean_hv']:.6f} | "
                    f"{row['mean_delta_percent']:+.3f}% | "
                    f"{row['wins']}-{row['ties']}-{row['losses']} |")
        lines.append("")
    lines.extend([
        "## Scope", "",
        f"Highest development mean HV: `{best}`.", "",
        "This is a development result and must not be reported as a "
        "final-test significance conclusion.",
    ])
    (out / "analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--initialization", type=Path, required=True)
    parser.add_argument("--crossover", type=Path, required=True)
    parser.add_argument("--mutation", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = analyze(args.plan, {
            "BASE": args.base,
            "I": args.initialization,
            "C": args.crossover,
            "M": args.mutation,
        }, args.out)
    except (KeyError, OSError, TypeError, ValueError,
            json.JSONDecodeError) as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
