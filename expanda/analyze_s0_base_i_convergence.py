#!/usr/bin/env python3
"""Paired training-only convergence analysis for deterministic S0 Base and I."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import statistics


ROLES = ("low", "median", "high")
DEFAULT_BUDGETS = (250, 5_000, 10_000, 20_000, 30_000)


def _read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _csv(path: Path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows):
    rows = list(rows)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _validate(base: Path, learned: Path):
    base_config = _read(base / "pilot_configuration.json")
    learned_config = _read(learned / "pilot_configuration.json")
    base_complete = _read(base / "COMPLETE.json")
    learned_complete = _read(learned / "COMPLETE.json")
    if base_complete.get("completed") is not True or learned_complete.get("completed") is not True:
        raise ValueError("both convergence pilots must be complete")
    if base_config.get("component") != "BASE" or learned_config.get("component") != "I":
        raise ValueError("expected paired BASE and I convergence pilots")
    if (base_config.get("test_instances_used") is not False
            or learned_config.get("test_instances_used") is not False):
        raise ValueError("convergence analysis must remain training-only")
    shared = ("selection_digest", "normalization_digest", "population", "repeats",
              "algorithm_seed_base", "seed_formula")
    different = [name for name in shared if base_config.get(name) != learned_config.get(name)]
    if different:
        raise ValueError(f"paired pilot configuration differs in {different}")
    if int(base_config["evaluation_budget"]) < int(learned_config["evaluation_budget"]):
        raise ValueError("Base pilot does not cover the I evaluation budget")
    return base_config, learned_config, base_complete, learned_complete


def _index(rows, budgets):
    result = {}
    for row in rows:
        evaluation = int(row["real_objective_evaluation_count"])
        if evaluation not in budgets:
            continue
        key = (
            row["role"], row["instance_id"], int(row["repeat"]),
            int(row["algorithm_seed"]), evaluation,
        )
        if key in result:
            raise ValueError(f"duplicate convergence row: {key}")
        hv = float(row["archive_hv"])
        if not math.isfinite(hv):
            raise ValueError(f"non-finite archive HV: {key}")
        result[key] = {**row, "archive_hv": hv}
    return result


def pair_curves(base_rows, learned_rows, budgets):
    base, learned = _index(base_rows, budgets), _index(learned_rows, budgets)
    if set(base) != set(learned):
        missing_i = sorted(set(base) - set(learned))
        missing_base = sorted(set(learned) - set(base))
        raise ValueError(
            f"unpaired convergence checkpoints: missing_i={missing_i[:3]} "
            f"missing_base={missing_base[:3]}")
    pairs = []
    for key in sorted(base):
        role, instance, repeat, seed, evaluation = key
        base_hv, i_hv = base[key]["archive_hv"], learned[key]["archive_hv"]
        delta = i_hv - base_hv
        tolerance = 1e-12
        outcome = "win" if delta > tolerance else "loss" if delta < -tolerance else "tie"
        pairs.append({
            "role": role, "instance_id": instance, "repeat": repeat,
            "algorithm_seed": seed, "evaluation_budget": evaluation,
            "base_archive_hv": base_hv, "i_archive_hv": i_hv,
            "delta_i_minus_base": delta,
            "percent_delta_vs_base": 100.0 * delta / base_hv if base_hv else None,
            "outcome": outcome,
        })
    return pairs


def summarise(pairs, budgets):
    result = []
    for role in (*ROLES, "overall"):
        for budget in budgets:
            rows = [row for row in pairs if row["evaluation_budget"] == budget
                    and (role == "overall" or row["role"] == role)]
            if not rows:
                raise ValueError(f"no paired rows for role={role}, budget={budget}")
            deltas = [row["delta_i_minus_base"] for row in rows]
            percents = [row["percent_delta_vs_base"] for row in rows]
            result.append({
                "role": role, "evaluation_budget": budget,
                "pair_count": len(rows),
                "mean_base_hv": statistics.fmean(row["base_archive_hv"] for row in rows),
                "mean_i_hv": statistics.fmean(row["i_archive_hv"] for row in rows),
                "mean_delta_i_minus_base": statistics.fmean(deltas),
                "median_delta_i_minus_base": statistics.median(deltas),
                "mean_percent_delta_vs_base": statistics.fmean(percents),
                "median_percent_delta_vs_base": statistics.median(percents),
                "wins": sum(row["outcome"] == "win" for row in rows),
                "ties": sum(row["outcome"] == "tie" for row in rows),
                "losses": sum(row["outcome"] == "loss" for row in rows),
            })
    return result


def _runtime_pairs(base_complete, learned_complete):
    def index(complete):
        return {(row["role"], row["instance_id"], int(row["repeat"]),
                 int(row["algorithm_seed"])): float(row["optimisation_seconds"])
                for row in complete["runs"]}
    base, learned = index(base_complete), index(learned_complete)
    if set(base) != set(learned):
        raise ValueError("runtime summaries are not paired")
    ratios = [learned[key] / base[key] for key in sorted(base)]
    return {
        "pair_count": len(ratios),
        "mean_base_seconds": statistics.fmean(base.values()),
        "mean_i_seconds": statistics.fmean(learned.values()),
        "mean_paired_i_over_base_ratio": statistics.fmean(ratios),
        "median_paired_i_over_base_ratio": statistics.median(ratios),
    }


def _svg(summary, budgets):
    rows = {(row["role"], row["evaluation_budget"]): row for row in summary}
    width, height = 880, 500
    left, right, top, bottom = 80, 35, 50, 70
    plot_w, plot_h = width - left - right, height - top - bottom
    overall = [rows[("overall", budget)] for budget in budgets]
    values = [row[name] for row in overall for name in ("mean_base_hv", "mean_i_hv")]
    ymin = min(values) - 0.05 * (max(values) - min(values) or 1.0)
    ymax = max(values) + 0.05 * (max(values) - min(values) or 1.0)

    def x(index):
        return left + index / max(1, len(budgets) - 1) * plot_w

    def y(value):
        return top + (ymax - value) / (ymax - ymin) * plot_h

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="440" y="27" text-anchor="middle" font-size="18">Paired S0 Base vs I convergence (training-only)</text>',
    ]
    for method, field, color in (("Base", "mean_base_hv", "#334155"),
                                 ("I", "mean_i_hv", "#2563eb")):
        points = " ".join(
            f'{x(index):.2f},{y(row[field]):.2f}' for index, row in enumerate(overall))
        parts.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="3"/>')
        for index, row in enumerate(overall):
            parts.append(f'<circle cx="{x(index):.2f}" cy="{y(row[field]):.2f}" r="4" fill="{color}"/>')
        parts.append(f'<text x="{width-right-70}" y="{top + 20 * (1 if method == "Base" else 2)}" fill="{color}" font-size="13">{method}</text>')
    parts.extend([
        f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="black"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="black"/>',
    ])
    for index, budget in enumerate(budgets):
        parts.append(f'<text x="{x(index):.2f}" y="{height-bottom+24}" text-anchor="middle" font-size="12">{budget:,}</text>')
    parts.append('</svg>')
    return "\n".join(parts) + "\n"


def analyze(base: Path, learned: Path, out: Path, budgets=DEFAULT_BUDGETS):
    base_config, learned_config, base_complete, learned_complete = _validate(base, learned)
    pairs = pair_curves(
        _csv(base / "convergence_progress.csv"),
        _csv(learned / "convergence_progress.csv"), budgets)
    summary = summarise(pairs, budgets)
    runtime = _runtime_pairs(base_complete, learned_complete)
    out.mkdir(parents=True, exist_ok=False)
    _write_csv(out / "paired_checkpoints.csv", pairs)
    _write_csv(out / "checkpoint_summary.csv", summary)
    (out / "convergence.svg").write_text(_svg(summary, budgets), encoding="utf-8")
    result = {
        "completed": True,
        "stage": "training-only paired S0 Base/I convergence analysis",
        "test_instances_used": False,
        "budgets": list(budgets),
        "pair_count_per_budget": len(pairs) // len(budgets),
        "normalization_digest": base_config["normalization_digest"],
        "i_checkpoint_sha256": learned_config["policy_checkpoint_sha256"],
        "runtime": runtime,
        "outputs": {
            "pairs": "paired_checkpoints.csv", "summary": "checkpoint_summary.csv",
            "curve": "convergence.svg", "report": "analysis.md",
        },
    }
    (out / "analysis.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    by_key = {(row["role"], row["evaluation_budget"]): row for row in summary}
    lines = [
        "# S0 Base versus I convergence analysis", "",
        "This paired diagnostic uses only the three preselected S0 training instances.", "",
        "| Budget | Base mean HV | I mean HV | Mean delta % | Wins-Ties-Losses |",
        "|---:|---:|---:|---:|:---:|",
    ]
    for budget in budgets:
        row = by_key[("overall", budget)]
        lines.append(
            f"| {budget:,} | {row['mean_base_hv']:.6f} | {row['mean_i_hv']:.6f} | "
            f"{row['mean_percent_delta_vs_base']:+.3f}% | "
            f"{row['wins']}-{row['ties']}-{row['losses']} |")
    lines.extend(["", "## 30,000 evaluations by complexity role", "",
                  "| Role | Base mean HV | I mean HV | Mean delta % | Wins-Ties-Losses |",
                  "|---|---:|---:|---:|:---:|"])
    for role in ROLES:
        row = by_key[(role, budgets[-1])]
        lines.append(
            f"| {role} | {row['mean_base_hv']:.6f} | {row['mean_i_hv']:.6f} | "
            f"{row['mean_percent_delta_vs_base']:+.3f}% | "
            f"{row['wins']}-{row['ties']}-{row['losses']} |")
    lines.extend(["", "## Runtime", "",
                  f"- Base mean: {runtime['mean_base_seconds'] / 60:.2f} minutes.",
                  f"- I mean: {runtime['mean_i_seconds'] / 60:.2f} minutes.",
                  f"- Mean paired I/Base ratio: {runtime['mean_paired_i_over_base_ratio']:.2f}.",
                  "", "This is a development diagnostic, not a test-set significance result.", ""])
    (out / "analysis.md").write_text("\n".join(lines), encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--initialization", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(analyze(args.base, args.initialization, args.out),
                     indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
