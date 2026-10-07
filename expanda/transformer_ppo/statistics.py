"""Per-configuration paired inference for formal OOS hypervolume results."""
from __future__ import annotations

from collections import Counter

import numpy as np

from .config import TEST_CONFIGS


def holm(p_values):
    order = sorted(range(len(p_values)), key=lambda index: p_values[index])
    adjusted = [1.0] * len(p_values)
    running = 0.0
    count = len(p_values)
    for rank, index in enumerate(order):
        value = min(1.0, (count - rank) * float(p_values[index]))
        running = max(running, value)
        adjusted[index] = running
    return adjusted


def paired_bootstrap_ci(differences, seed=71_000_001, samples=20_000):
    values = np.asarray(differences, dtype=float)
    if not len(values) or not np.all(np.isfinite(values)):
        raise ValueError("finite paired differences required")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(samples, len(values)))
    means = values[indices].mean(axis=1)
    return [float(x) for x in np.quantile(means, [0.025, 0.975])]


def _one_configuration(rows, learned, control, bootstrap_seed):
    from scipy.stats import wilcoxon

    left_rows = [row for row in rows if row["method"] == learned]
    right_rows = [row for row in rows if row["method"] == control]
    left = {row["instance_id"]: float(row["oos_hv"])
            for row in left_rows}
    right = {row["instance_id"]: float(row["oos_hv"])
             for row in right_rows}
    if (len(left) != len(left_rows) or len(right) != len(right_rows)
            or set(left) != set(right) or len(left) != 10):
        raise ValueError(f"{learned} vs {control} requires the same 10 test instances")
    identifiers = sorted(left)
    differences = [left[key] - right[key] for key in identifiers]
    nonzero = [value for value in differences if abs(value) > 1e-12]
    p_value = float(wilcoxon(differences, zero_method="wilcox").pvalue) if nonzero else 1.0
    outcome = Counter("win" if value > 1e-12 else "loss" if value < -1e-12 else "tie"
                      for value in differences)
    return {
        "learned": learned, "control": control,
        "instance_ids": identifiers, "paired_differences": differences,
        "mean_difference": float(np.mean(differences)),
        "median_difference": float(np.median(differences)),
        "mean_difference_bootstrap_95ci": paired_bootstrap_ci(
            differences, seed=bootstrap_seed),
        "wilcoxon_p_value": p_value,
        "wins": outcome["win"], "ties": outcome["tie"], "losses": outcome["loss"],
    }


def formal_report(rows):
    required = {"configuration", "instance_id", "method", "oos_hv"}
    if any(not required.issubset(row) for row in rows):
        raise ValueError(f"each row requires {sorted(required)}")
    families = {
        "primary_transformer_vs_rule": ("transformer", "rule"),
        "ablation_transformer_vs_mlp": ("transformer", "mlp"),
        "mask_rule_vs_random": ("rule", "random"),
    }
    result = {"schema_version": 1, "pairing_unit": "new test instance",
              "tests_per_family": len(TEST_CONFIGS), "families": {}}
    for family_index, (family, methods) in enumerate(families.items()):
        entries = []
        for config_index, configuration in enumerate(TEST_CONFIGS):
            subset = [row for row in rows if row["configuration"] == configuration]
            entries.append(_one_configuration(
                subset, *methods,
                bootstrap_seed=71_000_001 + family_index * 100 + config_index))
            entries[-1]["configuration"] = configuration
        adjusted = holm([entry["wilcoxon_p_value"] for entry in entries])
        for entry, value in zip(entries, adjusted):
            entry["holm_adjusted_p_value"] = value
        result["families"][family] = entries
    return result
