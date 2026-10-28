#!/usr/bin/env python3
"""Analyze paired training-only S0 V2 initialization top-k runs."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import statistics


def _read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _atomic_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n",
        encoding="utf-8")
    temporary.replace(path)


def _runs(root: Path, expected_component: str):
    complete = _read(root / "COMPLETE.json")
    config = _read(root / "pilot_configuration.json")
    rows = complete.get("runs", [])
    if (complete.get("completed") is not True
            or complete.get("run_count") != 9
            or len(rows) != 9
            or config.get("component") != expected_component
            or config.get("test_instances_used") is not False):
        raise ValueError(f"invalid completed {expected_component} pilot: {root}")
    result = {}
    for row in rows:
        key = (row["role"], int(row["repeat"]), int(row["algorithm_seed"]))
        hv = float(row["final_hypervolume"])
        if key in result or not math.isfinite(hv):
            raise ValueError(f"invalid or duplicate paired run in {root}: {key}")
        result[key] = {**row, "final_hypervolume": hv}
    return config, result


def _mean(values):
    return sum(values) / len(values)


def _summary(name, candidate, base):
    if set(candidate) != set(base):
        raise ValueError(f"paired keys differ for {name}")
    rows = []
    for key in sorted(base):
        base_hv = base[key]["final_hypervolume"]
        candidate_hv = candidate[key]["final_hypervolume"]
        delta = candidate_hv - base_hv
        rows.append({
            "setting": name,
            "role": key[0],
            "repeat": key[1],
            "algorithm_seed": key[2],
            "instance_id": candidate[key]["instance_id"],
            "base_hv": base_hv,
            "candidate_hv": candidate_hv,
            "delta_hv": delta,
            "delta_percent": 100.0 * delta / base_hv,
        })
    deltas = [row["delta_hv"] for row in rows]
    wins = sum(value > 1e-12 for value in deltas)
    losses = sum(value < -1e-12 for value in deltas)
    result = {
        "setting": name,
        "pair_count": len(rows),
        "base_mean_hv": _mean([row["base_hv"] for row in rows]),
        "candidate_mean_hv": _mean([
            row["candidate_hv"] for row in rows]),
        "mean_paired_delta_hv": _mean(deltas),
        "mean_paired_delta_percent": _mean([
            row["delta_percent"] for row in rows]),
        "paired_delta_standard_deviation": statistics.stdev(deltas),
        "wins": wins,
        "ties": len(rows) - wins - losses,
        "losses": losses,
        "roles": {},
    }
    for role in ("low", "median", "high"):
        selected = [row for row in rows if row["role"] == role]
        role_deltas = [row["delta_hv"] for row in selected]
        role_wins = sum(value > 1e-12 for value in role_deltas)
        role_losses = sum(value < -1e-12 for value in role_deltas)
        result["roles"][role] = {
            "base_mean_hv": _mean([row["base_hv"] for row in selected]),
            "candidate_mean_hv": _mean([
                row["candidate_hv"] for row in selected]),
            "mean_delta_percent": _mean([
                row["delta_percent"] for row in selected]),
            "wins": role_wins,
            "ties": len(selected) - role_wins - role_losses,
            "losses": role_losses,
        }
    return result, rows


def _choose(summaries, top_k_by_name, near_tie_fraction):
    best_mean = max(row["candidate_mean_hv"] for row in summaries)
    eligible = [
        row for row in summaries
        if row["candidate_mean_hv"] >= best_mean * (1.0 - near_tie_fraction)
    ]
    selected = min(
        eligible,
        key=lambda row: (
            row["paired_delta_standard_deviation"],
            (top_k_by_name[row["setting"]]
             if top_k_by_name[row["setting"]] is not None else 10**9),
        ))
    all_row = next(row for row in summaries if row["setting"] == "all")
    stable_top_k = (
        top_k_by_name[selected["setting"]] is not None
        and selected["candidate_mean_hv"] >= all_row["candidate_mean_hv"]
        and selected["candidate_mean_hv"] > selected["base_mean_hv"]
        and selected["wins"] > selected["losses"])
    if top_k_by_name[selected["setting"]] is not None and not stable_top_k:
        selected = all_row
        reason = (
            "no eligible finite top-k met the predeclared stability guard; "
            "do not force a prefilter")
    else:
        reason = (
            "highest mean-HV near-tie set, then lowest paired-delta "
            "standard deviation, then smaller finite top-k")
    return selected["setting"], reason, [row["setting"] for row in eligible]


def analyze(plan_path: Path, base_root: Path, candidates, out: Path):
    plan = _read(plan_path)
    expected = {
        row["name"]: row["top_k"] for row in plan.get("settings", [])}
    if (plan.get("stage")
            != "training-only S0 V2 initialization top-k ablation"
            or set(expected) != {"top4", "top8", "top16", "all"}
            or plan.get("test_instances_used") is not False):
        raise ValueError("invalid frozen I-ablation plan")
    supplied = dict(candidates)
    if set(supplied) != set(expected):
        raise ValueError("candidates must supply top4, top8, top16, and all")

    base_config, base = _runs(base_root, "BASE")
    expected_ids = {
        row["instance_id"] for row in plan["validation_instances"]}
    if ({row["instance_id"] for row in base.values()} != expected_ids
            or base_config.get("selection_digest")
            != plan["selection_digest"]):
        raise ValueError("Base pilot does not match the frozen validation set")

    summaries = []
    paired_rows = []
    for name in ("top4", "top8", "top16", "all"):
        config, current = _runs(supplied[name], "I")
        if (config.get("selection_digest") != plan["selection_digest"]
                or config.get("normalization_digest")
                != plan["normalization"]["normalization_digest"]):
            raise ValueError(f"{name} pilot does not match the frozen plan")
        configured_top_k = (
            config.get("learning_control", {}).get(
                "initialization_rule_top_k"))
        if configured_top_k != expected[name]:
            raise ValueError(
                f"{name} top-k mismatch: {configured_top_k} != {expected[name]}")
        summary, rows = _summary(name, current, base)
        summaries.append(summary)
        paired_rows.extend(rows)

    criterion = plan["selection_criterion"]
    recommendation, reason, eligible = _choose(
        summaries, expected, criterion["near_tie_relative_fraction"])
    result = {
        "completed": True,
        "stage": "training-only S0 V2 initialization top-k analysis",
        "test_instances_used": False,
        "plan_digest": plan["plan_digest"],
        "pair_count_per_setting": 9,
        "summaries": summaries,
        "recommendation": {
            "setting": recommendation,
            "top_k": expected[recommendation],
            "near_tie_eligible": eligible,
            "reason": reason,
            "role": "development recommendation; not a test-set result",
        },
    }
    out.mkdir(parents=True, exist_ok=False)
    _atomic_json(out / "analysis.json", result)

    fields = list(paired_rows[0])
    with (out / "paired_hv.csv").open(
            "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(paired_rows)

    lines = [
        "# S0 V2 initialization top-k ablation",
        "",
        "Training-only frozen-policy validation; no final test instances used.",
        "",
        "| Setting | Mean HV | Mean paired delta % | W-T-L | Delta SD |",
        "|---|---:|---:|:---:|---:|",
    ]
    for row in summaries:
        lines.append(
            f"| {row['setting']} | {row['candidate_mean_hv']:.6f} | "
            f"{row['mean_paired_delta_percent']:+.3f}% | "
            f"{row['wins']}-{row['ties']}-{row['losses']} | "
            f"{row['paired_delta_standard_deviation']:.6f} |")
    lines.extend([
        "", "## Recommendation", "",
        f"- Setting: `{recommendation}`",
        f"- top-k: `{expected[recommendation]}`",
        f"- Near-tie eligible: {', '.join(eligible)}",
        f"- Rule: {reason}.",
        "",
        "This is a development recommendation, not a final-test conclusion.",
    ])
    (out / "analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument(
        "--candidate", action="append", nargs=2,
        metavar=("NAME", "DIRECTORY"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = analyze(
            args.plan, args.base,
            [(name, Path(path)) for name, path in args.candidate], args.out)
    except (KeyError, OSError, TypeError, ValueError,
            json.JSONDecodeError) as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
