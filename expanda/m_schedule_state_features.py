"""Pure helpers for the pre-intervention M schedule state gate."""
from __future__ import annotations

import math
import statistics


EPSILON = 1e-12
PROGRESS_FIELDS = (
    "population_hv",
    "archive_hv",
    "feasible_fraction",
    "front_fraction",
    "evaluation_fraction",
)


def short_instance_id(value):
    """Map instance-file identifiers to the directory naming convention."""
    result = str(value).removeprefix("train-")
    if result.startswith("S0-"):
        result = "S0_" + result[3:]
    return result


def _finite(value, default=0.0):
    if value is None:
        return float(default)
    result = float(value)
    return result if math.isfinite(result) else float(default)


def pre_intervention_progress(rows, minimum_generation, evaluation_budget):
    """Return deterministic search rows strictly before an intervention."""
    selected = []
    for raw in rows:
        generation = int(raw.get("generation", -1))
        if generation >= int(minimum_generation):
            continue
        population_size = max(1, int(raw.get("population_size", 0)))
        row = {
            "checkpoint": str(raw.get("checkpoint", "")),
            "generation": generation,
            "real_objective_evaluation_count": int(
                raw.get("real_objective_evaluation_count", 0)),
            "population_size": int(raw.get("population_size", 0)),
            "feasible_count": int(raw.get("feasible_count", 0)),
            "front_size": int(raw.get("front_size", 0)),
            "population_hv": _finite(raw.get("population_hv")),
            "archive_hv": _finite(raw.get("archive_hv")),
        }
        row["feasible_fraction"] = row["feasible_count"] / population_size
        row["front_fraction"] = row["front_size"] / population_size
        row["evaluation_fraction"] = (
            row["real_objective_evaluation_count"]
            / max(1, int(evaluation_budget)))
        selected.append(row)
    selected.sort(key=lambda row: (
        row["generation"],
        row["real_objective_evaluation_count"],
        row["checkpoint"],
    ))
    if not selected:
        raise ValueError("no pre-intervention search progress rows")
    return selected


def assert_progress_replay(reference, candidate, tolerance=1e-12):
    """Require the scheduled run to replay Base before intervention."""
    if len(reference) != len(candidate):
        raise ValueError(
            "pre-intervention progress length mismatch: "
            f"base={len(reference)} candidate={len(candidate)}")
    identity_fields = (
        "checkpoint", "generation", "real_objective_evaluation_count",
        "population_size", "feasible_count", "front_size",
    )
    numeric_fields = ("population_hv", "archive_hv")
    for index, (base_row, candidate_row) in enumerate(
            zip(reference, candidate)):
        for field in identity_fields:
            if base_row[field] != candidate_row[field]:
                raise ValueError(
                    f"pre-intervention replay mismatch row={index} "
                    f"field={field}: {base_row[field]!r} != "
                    f"{candidate_row[field]!r}")
        for field in numeric_fields:
            if not math.isclose(
                    float(base_row[field]), float(candidate_row[field]),
                    rel_tol=0.0, abs_tol=tolerance):
                raise ValueError(
                    f"pre-intervention replay mismatch row={index} "
                    f"field={field}: {base_row[field]!r} != "
                    f"{candidate_row[field]!r}")


def _slope(values):
    if len(values) < 2:
        return 0.0
    x_mean = (len(values) - 1) / 2.0
    y_mean = statistics.fmean(values)
    denominator = sum((index - x_mean) ** 2 for index in range(len(values)))
    if denominator <= EPSILON:
        return 0.0
    return sum(
        (index - x_mean) * (value - y_mean)
        for index, value in enumerate(values)
    ) / denominator


def _stagnation(values, window):
    recent = values[-max(1, int(window)):]
    if len(recent) < 2:
        return 0.0
    improvements = sum(
        current > previous + EPSILON
        for previous, current in zip(recent, recent[1:]))
    return 1.0 - improvements / (len(recent) - 1)


def progress_features(rows):
    """Build a fixed-size, runtime-independent feature vector."""
    names = []
    values = []
    for field in PROGRESS_FIELDS:
        series = [_finite(row[field]) for row in rows]
        metrics = {
            "first": series[0],
            "last": series[-1],
            "delta": series[-1] - series[0],
            "mean": statistics.fmean(series),
            "std": statistics.pstdev(series) if len(series) > 1 else 0.0,
            "minimum": min(series),
            "maximum": max(series),
            "slope_all": _slope(series),
            "slope_last3": _slope(series[-3:]),
        }
        for metric, value in metrics.items():
            names.append(f"{field}_{metric}")
            values.append(float(value))

    archive = [_finite(row["archive_hv"]) for row in rows]
    population = [_finite(row["population_hv"]) for row in rows]
    gaps = [a - p for a, p in zip(archive, population)]
    extra = {
        "progress_point_count": math.log1p(len(rows)),
        "last_generation": float(rows[-1]["generation"]),
        "last_evaluation_fraction": float(rows[-1]["evaluation_fraction"]),
        "archive_population_gap_first": gaps[0],
        "archive_population_gap_last": gaps[-1],
        "archive_population_gap_mean": statistics.fmean(gaps),
        "archive_stagnation_last3": _stagnation(archive, 3),
        "archive_stagnation_last5": _stagnation(archive, 5),
        "population_stagnation_last3": _stagnation(population, 3),
        "population_stagnation_last5": _stagnation(population, 5),
    }
    names.extend(extra)
    values.extend(float(value) for value in extra.values())
    if not all(math.isfinite(value) for value in values):
        raise ValueError("non-finite pre-intervention feature")
    return names, values
