"""Pre-mutation feature schema for segment-level CCP100 learning."""

from learning_features import CATEGORICAL_FEATURES as V1_CATEGORICAL_FEATURES
from learning_features import NUMERIC_FEATURES as V1_NUMERIC_FEATURES
from learning_features import feature_record as v1_feature_record


NUMERIC_FEATURES = V1_NUMERIC_FEATURES + [
    "path_id", "candidate_count",
    "miss_tt_before", "cap_excess_before", "border_cap_excess_before",
    "late_teu_h_before", "max_border_util_before",
    "max_observed_late_h_before", "wait_teu_h_before",
]

CATEGORICAL_FEATURES = list(V1_CATEGORICAL_FEATURES)
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def feature_record(candidate, context):
    row = v1_feature_record(candidate, context)
    breakdown = context.get("violation_breakdown_before") or {}
    row.update({
        "path_id": candidate.get("path_id"),
        "candidate_count": context.get("candidate_count"),
        "miss_tt_before": breakdown.get("miss_tt"),
        "cap_excess_before": breakdown.get("cap_excess"),
        "border_cap_excess_before": breakdown.get("border_cap_excess"),
        "late_teu_h_before": breakdown.get("late_teu_h"),
        "max_border_util_before": breakdown.get("max_border_util"),
        "max_observed_late_h_before": breakdown.get("max_observed_late_h"),
        "wait_teu_h_before": breakdown.get("wait_teu_h"),
    })
    return row
