"""Dependency-light helpers for deterministic resource-pilot selection."""


METRIC_NAME = "candidate_path_token_count"
SELECTION_RULE = (
    "sort ascending by candidate_path_token_count, then candidate_path_count, "
    "then instance_id; select ranks 1, lower median, and N"
)


def select_records(records):
    """Return deterministic low/lower-median/high records with rank metadata."""
    if len(records) < 3:
        raise ValueError("at least three complexity records are required")
    ordered = sorted(
        records,
        key=lambda row: (
            int(row[METRIC_NAME]),
            int(row["candidate_path_count"]),
            row["instance_id"],
        ),
    )
    indices = (0, (len(ordered) - 1) // 2, len(ordered) - 1)
    selected = []
    for role, index in zip(("low", "median", "high"), indices):
        selected.append({
            "role": role,
            "rank": index + 1,
            **ordered[index],
        })
    return ordered, selected
