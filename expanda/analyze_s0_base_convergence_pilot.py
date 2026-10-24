#!/usr/bin/env python3
"""Analyse the training-only S0 Base convergence pilot without rerunning search."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import statistics


ROLES = ("low", "median", "high")
DEFAULT_BUDGETS = (5_000, 10_000, 20_000, 30_000, 40_000, 50_000)


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _atomic_text(path: Path, value: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def _atomic_json(path: Path, value):
    _atomic_text(path, json.dumps(value, indent=2, allow_nan=False) + "\n")


def _read_progress(path: Path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def extract_checkpoints(rows, budgets):
    """Return one exact row per run and requested evaluation budget."""
    keys = ("role", "instance_id", "repeat", "algorithm_seed")
    grouped = {}
    for row in rows:
        key = tuple(row[name] for name in keys)
        evaluation = int(row["real_objective_evaluation_count"])
        grouped.setdefault(key, {})[evaluation] = row

    extracted = []
    for key, by_evaluation in sorted(grouped.items()):
        missing = [budget for budget in budgets if budget not in by_evaluation]
        if missing:
            raise ValueError(
                f"run {key} lacks exact evaluation checkpoints {missing}")
        final_hv = float(by_evaluation[budgets[-1]]["archive_hv"])
        if not math.isfinite(final_hv) or final_hv <= 0:
            raise ValueError(f"run {key} has invalid final archive HV")
        previous_hv = None
        for budget in budgets:
            source = by_evaluation[budget]
            hv = float(source["archive_hv"])
            if not math.isfinite(hv):
                raise ValueError(f"run {key} has non-finite archive HV")
            extracted.append({
                **dict(zip(keys, key)),
                "evaluation_budget": budget,
                "archive_hv": hv,
                "fraction_of_run_final": hv / final_hv,
                "remaining_to_run_final": final_hv - hv,
                "gain_from_previous_checkpoint": (
                    None if previous_hv is None else hv - previous_hv),
                "runtime_seconds": float(source["runtime_seconds"]),
            })
            previous_hv = hv
    return extracted


def _stats(values):
    return {
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "minimum": min(values),
        "maximum": max(values),
        "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def summarise_checkpoints(extracted, budgets):
    summary = []
    for role in (*ROLES, "overall"):
        role_rows = (
            extracted if role == "overall"
            else [row for row in extracted if row["role"] == role])
        for budget in budgets:
            rows = [
                row for row in role_rows
                if row["evaluation_budget"] == budget]
            if not rows:
                raise ValueError(f"no rows for role={role}, budget={budget}")
            hv = _stats([row["archive_hv"] for row in rows])
            fraction = _stats([row["fraction_of_run_final"] for row in rows])
            remaining = _stats([row["remaining_to_run_final"] for row in rows])
            gains = [
                row["gain_from_previous_checkpoint"] for row in rows
                if row["gain_from_previous_checkpoint"] is not None]
            runtime = _stats([row["runtime_seconds"] for row in rows])
            summary.append({
                "role": role,
                "evaluation_budget": budget,
                "run_count": len(rows),
                "mean_archive_hv": hv["mean"],
                "median_archive_hv": hv["median"],
                "stdev_archive_hv": hv["stdev"],
                "minimum_archive_hv": hv["minimum"],
                "maximum_archive_hv": hv["maximum"],
                "mean_fraction_of_run_final": fraction["mean"],
                "minimum_fraction_of_run_final": fraction["minimum"],
                "mean_remaining_to_run_final": remaining["mean"],
                "mean_gain_from_previous_checkpoint": (
                    statistics.fmean(gains) if gains else None),
                "mean_runtime_seconds": runtime["mean"],
            })
    return summary


def diagnostic_flags(summary, budgets):
    """Provide transparent diagnostics; do not automatically freeze a budget."""
    rows = {
        (row["role"], row["evaluation_budget"]): row for row in summary}
    flags = []
    for budget in budgets[:-1]:
        overall = rows[("overall", budget)]
        role_fractions = [
            rows[(role, budget)]["mean_fraction_of_run_final"]
            for role in ROLES]
        flags.append({
            "evaluation_budget": budget,
            "overall_mean_fraction_of_50k":
                overall["mean_fraction_of_run_final"],
            "worst_role_mean_fraction_of_50k": min(role_fractions),
            "all_runs_at_least_98pct_of_50k":
                overall["minimum_fraction_of_run_final"] >= 0.98,
            "screening_threshold_met": (
                overall["mean_fraction_of_run_final"] >= 0.99
                and min(role_fractions) >= 0.98),
        })
    return flags


def _write_csv(path: Path, rows, fields):
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _svg(summary, budgets):
    rows = {
        (row["role"], row["evaluation_budget"]): row for row in summary}
    width, height = 900, 540
    left, right, top, bottom = 90, 35, 55, 75
    plot_w, plot_h = width - left - right, height - top - bottom
    values = [
        rows[(role, budget)]["mean_archive_hv"]
        for role in (*ROLES, "overall") for budget in budgets]
    ymin = max(0.0, min(values) - 0.05 * (max(values) - min(values) or 1.0))
    ymax = max(values) + 0.05 * (max(values) - min(values) or 1.0)

    def x(value):
        return left + (value - budgets[0]) / (budgets[-1] - budgets[0]) * plot_w

    def y(value):
        return top + (ymax - value) / (ymax - ymin) * plot_h

    colours = {
        "low": "#277da1", "median": "#43aa8b", "high": "#f8961e",
        "overall": "#9b5de5",
    }
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="450" y="30" text-anchor="middle" font-family="sans-serif" font-size="20">S0 Base convergence on training-only pilot instances</text>',
    ]
    for index in range(6):
        value = ymin + index * (ymax - ymin) / 5
        py = y(value)
        parts.extend([
            f'<line x1="{left}" y1="{py:.2f}" x2="{width-right}" y2="{py:.2f}" stroke="#e5e7eb"/>',
            f'<text x="{left-12}" y="{py+5:.2f}" text-anchor="end" font-family="sans-serif" font-size="12">{value:.3f}</text>',
        ])
    for budget in budgets:
        px = x(budget)
        parts.extend([
            f'<line x1="{px:.2f}" y1="{top}" x2="{px:.2f}" y2="{height-bottom}" stroke="#f1f5f9"/>',
            f'<text x="{px:.2f}" y="{height-bottom+24}" text-anchor="middle" font-family="sans-serif" font-size="12">{budget//1000}k</text>',
        ])
    for role in (*ROLES, "overall"):
        points = " ".join(
            f'{x(budget):.2f},{y(rows[(role, budget)]["mean_archive_hv"]):.2f}'
            for budget in budgets)
        stroke_width = 4 if role == "overall" else 2.5
        parts.append(
            f'<polyline points="{points}" fill="none" stroke="{colours[role]}" stroke-width="{stroke_width}"/>')
        for budget in budgets:
            parts.append(
                f'<circle cx="{x(budget):.2f}" cy="{y(rows[(role, budget)]["mean_archive_hv"]):.2f}" r="4" fill="{colours[role]}"/>')
    parts.extend([
        f'<text x="{left + plot_w/2:.2f}" y="{height-20}" text-anchor="middle" font-family="sans-serif" font-size="14">Real objective evaluations</text>',
        f'<text x="20" y="{top + plot_h/2:.2f}" transform="rotate(-90 20 {top + plot_h/2:.2f})" text-anchor="middle" font-family="sans-serif" font-size="14">Mean archive HV</text>',
    ])
    legend_x = width - right - 270
    for index, role in enumerate((*ROLES, "overall")):
        lx = legend_x + (index % 2) * 135
        ly = top + 18 + (index // 2) * 24
        parts.extend([
            f'<line x1="{lx}" y1="{ly}" x2="{lx+24}" y2="{ly}" stroke="{colours[role]}" stroke-width="4"/>',
            f'<text x="{lx+31}" y="{ly+5}" font-family="sans-serif" font-size="13">{role}</text>',
        ])
    parts.append("</svg>\n")
    return "\n".join(parts)


def _markdown(pilot, summary, flags, budgets):
    rows = {
        (row["role"], row["evaluation_budget"]): row for row in summary}
    lines = [
        "# S0 Base convergence analysis",
        "",
        "This analysis uses only the three preselected S0 training instances. "
        "It does not use test instances and does not rerun optimisation.",
        "",
        "## Mean archive hypervolume",
        "",
        "| Budget | Low | Median | High | Overall | Overall fraction of 50k | Mean runtime |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for budget in budgets:
        overall = rows[("overall", budget)]
        lines.append(
            f'| {budget:,} | {rows[("low", budget)]["mean_archive_hv"]:.6f} '
            f'| {rows[("median", budget)]["mean_archive_hv"]:.6f} '
            f'| {rows[("high", budget)]["mean_archive_hv"]:.6f} '
            f'| {overall["mean_archive_hv"]:.6f} '
            f'| {100 * overall["mean_fraction_of_run_final"]:.2f}% '
            f'| {overall["mean_runtime_seconds"] / 60:.2f} min |')
    lines.extend([
        "",
        "## Budget diagnostics",
        "",
        "The screening flag is descriptive, not an automatic decision. It is "
        "true when overall mean HV has reached at least 99% of each run's 50k "
        "HV and every complexity role mean has reached at least 98%.",
        "",
        "| Budget | Overall fraction of 50k | Worst role fraction | Every run >=98% | Screening flag |",
        "|---:|---:|---:|:---:|:---:|",
    ])
    for row in flags:
        lines.append(
            f'| {row["evaluation_budget"]:,} '
            f'| {100 * row["overall_mean_fraction_of_50k"]:.2f}% '
            f'| {100 * row["worst_role_mean_fraction_of_50k"]:.2f}% '
            f'| {"yes" if row["all_runs_at_least_98pct_of_50k"] else "no"} '
            f'| {"yes" if row["screening_threshold_met"] else "no"} |')
    lines.extend([
        "",
        "## Provenance",
        "",
        f'- Pilot runs: {pilot["run_count"]}',
        f'- Population: {pilot["population"]}',
        f'- Maximum evaluation budget: {pilot["evaluation_budget"]:,}',
        "- Dataset role: training-only budget selection",
        "- Formal budget: not frozen by this script; review the full curve and "
        "all three role-specific trajectories first.",
        "",
    ])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--budgets", type=int, nargs="+", default=list(DEFAULT_BUDGETS))
    args = parser.parse_args()
    args.pilot = args.pilot.resolve()
    args.out = args.out.resolve()
    budgets = tuple(args.budgets)
    if (not budgets or tuple(sorted(set(budgets))) != budgets
            or any(value <= 0 for value in budgets)):
        parser.error("--budgets must be unique positive values in ascending order")
    try:
        config = _read_json(args.pilot / "pilot_configuration.json")
        complete = _read_json(args.pilot / "COMPLETE.json")
        if (config.get("configuration") != "S0"
                or config.get("test_instances_used") is not False
                or config.get("validation_set_used") is not False
                or config.get("component") != "BASE"
                or complete.get("completed") is not True
                or complete.get("run_count") != 9
                or budgets[-1] != config.get("evaluation_budget")):
            raise ValueError("pilot is not the complete training-only S0 Base 3x3 run")
        rows = _read_progress(args.pilot / complete["progress_csv"])
        extracted = extract_checkpoints(rows, budgets)
        summary = summarise_checkpoints(extracted, budgets)
        flags = diagnostic_flags(summary, budgets)
    except (OSError, KeyError, TypeError, ValueError,
            json.JSONDecodeError) as error:
        parser.error(str(error))

    args.out.mkdir(parents=True, exist_ok=True)
    run_fields = list(extracted[0])
    summary_fields = list(summary[0])
    _write_csv(args.out / "run_checkpoints.csv", extracted, run_fields)
    _write_csv(args.out / "checkpoint_summary.csv", summary, summary_fields)
    _atomic_text(args.out / "convergence_curve.svg", _svg(summary, budgets))
    report_metadata = {**config, "run_count": complete["run_count"]}
    _atomic_text(
        args.out / "analysis.md",
        _markdown(report_metadata, summary, flags, budgets))
    analysis = {
        "completed": True,
        "stage": "training-only S0 Base convergence analysis",
        "test_instances_used": False,
        "pilot": str(args.pilot),
        "run_count": complete["run_count"],
        "population": config["population"],
        "evaluation_budget": config["evaluation_budget"],
        "budgets": list(budgets),
        "diagnostic_flags": flags,
        "formal_budget_frozen": False,
        "outputs": {
            "run_checkpoints": "run_checkpoints.csv",
            "checkpoint_summary": "checkpoint_summary.csv",
            "curve": "convergence_curve.svg",
            "report": "analysis.md",
        },
    }
    _atomic_json(args.out / "analysis.json", analysis)
    print(json.dumps(analysis, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
