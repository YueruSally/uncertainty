"""Shared, pre-mutation feature schema for CCP100 location learning."""
from dataclasses import asdict, is_dataclass

import pandas as pd


NUMERIC_FEATURES = [
    "batch_id", "allocation_index", "arc_index",
    "baseline_selection_probability", "batch_quantity", "path_count", "share",
    "path_cost_per_teu", "path_emission_per_teu", "path_nominal_time_h",
    "alternative_mode_count", "generation", "feasible_before",
    "q90_cost_before", "q90_emission_before", "q90_makespan_before",
    "violation_before",
]

CATEGORICAL_FEATURES = [
    "operator", "phase", "from_node", "to_node", "current_mode",
]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def _target_fields(candidate):
    target = candidate.get("target")
    if target is None:
        return candidate
    if is_dataclass(target):
        return asdict(target)
    if isinstance(target, dict):
        return target
    raise TypeError(f"Unsupported mutation target: {type(target)!r}")


def feature_record(candidate, context):
    """Return only information available before applying the mutation."""
    target = _target_fields(candidate)
    baseline_probability = candidate.get(
        "baseline_selection_probability", candidate.get("selection_probability"))
    row = {
        "batch_id": target.get("batch_id"),
        "allocation_index": target.get("allocation_index"),
        "arc_index": target.get("arc_index"),
        "baseline_selection_probability": baseline_probability,
        "batch_quantity": candidate.get("batch_quantity"),
        "path_count": candidate.get("path_count"),
        "share": candidate.get("share"),
        "path_cost_per_teu": candidate.get("path_cost_per_teu"),
        "path_emission_per_teu": candidate.get("path_emission_per_teu"),
        "path_nominal_time_h": candidate.get("path_nominal_time_h"),
        "alternative_mode_count": candidate.get("alternative_mode_count"),
        "generation": context.get("generation"),
        "feasible_before": int(bool(context.get("feasible_before"))),
        "q90_cost_before": context.get("q90_cost_before"),
        "q90_emission_before": context.get("q90_emission_before"),
        "q90_makespan_before": context.get("q90_makespan_before"),
        "violation_before": context.get("violation_before"),
        "operator": context.get("operator"),
        "phase": context.get("phase"),
        "from_node": candidate.get("from_node"),
        "to_node": candidate.get("to_node"),
        "current_mode": candidate.get("current_mode"),
    }
    return row


def feature_frame(records):
    return pd.DataFrame(records, columns=FEATURES)
