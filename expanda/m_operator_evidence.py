"""Repeat reliability, recorded structure, and decision metrics. No objectives."""
from __future__ import annotations

from itertools import combinations

import numpy as np

from m_operator_offline import BALANCED, OPERATORS

EPS = 1e-12


def correlation(x, y):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return None
    x, y = x - x.mean(), y - y.mean()
    denominator = np.linalg.norm(x) * np.linalg.norm(y)
    if denominator <= 0 or not np.isfinite(denominator):
        return None
    return float(np.clip(np.sum(x * y) / denominator, -1, 1))


def ranks(x):
    order = np.argsort(x, kind="stable")
    ranked = np.empty(len(x), float)
    start = 0
    while start < len(x):
        stop = start + 1
        while stop < len(x) and x[order[stop]] == x[order[start]]:
            stop += 1
        ranked[order[start:stop]] = (start + stop - 1) / 2
        start = stop
    return ranked


def repeat_statistics(delta):
    """Diagnostic method-of-moments reliability, NOT a measured R² ceiling.

    delta[s,r] = latent mean[s] + zero-mean independent repeat error[s,r].
    Under that assumption E Var(state means) = Var(latent means) + E Var(error)/K.
    K=3 and correlated states make the variance-component estimate uncertain.
    """
    delta = np.asarray(delta, dtype=float)
    if delta.ndim != 2 or delta.shape[1] != 3 or not np.isfinite(delta).all():
        raise ValueError("repeat diagnostic requires finite [states, 3] relative gains")
    n = len(delta)
    rotations = []
    for r in range(3):
        a, b = delta[:, r], delta[:, [k for k in range(3) if k != r]].mean(1)
        rotations.append({"repeat": r + 1, "pearson": correlation(a, b),
                          "spearman": correlation(ranks(a), ranks(b)),
                          "comparison": "one repeat versus mean of the other two"})
    # Shift before variance to avoid spurious variance for identical decimals.
    within = float(np.var(delta - delta[:, :1], axis=1, ddof=1).mean()) if n else None
    means = delta.mean(1)
    between = float(np.var(means - means[0], ddof=1)) if n >= 2 else None
    raw = 1.0 - within / (3.0 * between) if between is not None and between > 0 else None
    return {"legal_state_count": n, "correlations": rotations,
            "all_repeats_zero_fraction": float(np.mean(np.all(np.abs(delta) <= EPS, axis=1))) if n else None,
            "mean_within_state_repeat_variance": within,
            "variance_of_three_repeat_state_means": between,
            "estimated_variance_of_mean_label_noise": within / 3 if within is not None else None,
            "estimated_mean_label_reliability_unclipped": raw,
            "estimated_mean_label_reliability_clipped": float(np.clip(raw, 0, 1)) if raw is not None else None,
            "undefined_correlation_reason": "too few states or constant repeat vectors" if any(r["pearson"] is None for r in rotations) else None}


def noise_report(rows, q, mask):
    delta = q - q[:, BALANCED:BALANCED + 1, :]
    names = np.array([r["instance"] for r in rows])
    groups = {"pooled": np.arange(len(rows))}
    groups.update({name: np.flatnonzero(names == name) for name in sorted(set(names))})
    result = {}
    for name, ids in groups.items():
        result[name] = {op: repeat_statistics(delta[ids[mask[ids, a]], a]) for a, op in enumerate(OPERATORS[:4])}
    return {"scope": "operator/target randomness conditional on fixed CCP100; does not estimate independent-scenario uncertainty",
            "assumptions": "variance components assume independent zero-mean repeat errors within a state; only K=3; not a strict predictability ceiling",
            "interpretation": "Low correlation is not proof that richer inputs cannot help; constant labels have undefined correlation, not zero reliability. High reliability does not prove available features predict the signal.",
            "sample_unit": "state; repeated states within trajectories are correlated; no independent-384-state confidence interval",
            "by_group": result}


def recover_path_groups(parent):
    """Recover only grouping actually encoded by parent_tokens: header then arcs.

    The 19-column legacy schema omits batch ID and global path ID. A local path
    block index is a grouping label only; it must not be invented as batch ID.
    """
    parent = np.asarray(parent, dtype=float)
    if parent.ndim != 2 or parent.shape[1] != 19 or not np.isfinite(parent).all():
        raise ValueError("parent must use the finite 19-column legacy schema")
    groups, cursor = [], 0
    while cursor < len(parent):
        header = parent[cursor]
        length = int(round(header[7]))
        if header[8] != 0 or length < 0 or abs(header[7] - length) > EPS:
            raise ValueError("missing allocation header or invalid path length")
        end = cursor + 1 + length
        if end > len(parent):
            raise ValueError("truncated path arc block")
        arcs = parent[cursor + 1:end]
        for position, arc in enumerate(arcs):
            if arc[8] != 1 or not np.isclose(arc[6], position / max(1, length - 1), atol=1e-12, rtol=0):
                raise ValueError("arc type/position disagrees with path block")
            if not np.allclose(arc[[0, 1, 7, 9, 10, 11, 12]], header[[0, 1, 7, 9, 10, 11, 12]], atol=1e-12, rtol=0):
                raise ValueError("arc allocation attributes disagree with path header")
        groups.append({"local_path_group": len(groups), "header": cursor,
                       "arc_indices": list(range(cursor + 1, end)), "batch_id": None})
        cursor = end
    return groups


def structure_report(rows):
    counts, errors = [], []
    for row in rows:
        try:
            groups = recover_path_groups(row["observation"]["parent"])
            counts.append({"paths": len(groups), "arcs": sum(len(g["arc_indices"]) for g in groups)})
        except ValueError as error:
            errors.append({"instance": row["instance"], "run": row["run"], "state_id": row["state_id"], "error": str(error)})
    return {"state_count": len(rows), "path_groups_recovered_state_count": len(counts),
            "path_grouping_exact": not errors, "invalid_states": errors,
            "total_recorded_path_blocks": sum(c["paths"] for c in counts),
            "total_recorded_arcs": sum(c["arcs"] for c in counts),
            "batch_identity_available": False, "full_batch_path_token_experiment_ready": False,
            "available": ["allocation/path block boundary", "arc position", "quantity and share",
                          "origin/destination/from/to/mode category signatures", "distance/cost-rate/emission-rate/speed"],
            "missing": ["explicit transport batch identity per allocation", "global path identity"],
            "qualification": "OD, quantity and share are not guaranteed unique batch identifiers. Local path blocks can support a path-only study, not a claim of complete batch/path structure. Do not reconstruct batch IDs from outcomes."}


def decision_quality(predictions, target, mask, train_target, train_mask, scale):
    """Auxiliary metrics; primary policy gain is always evaluated separately."""
    legal = mask.copy()
    legal[:, BALANCED] = False
    active = legal & (np.abs(target) > EPS)
    weights = np.abs(target[active])
    total = float(weights.sum())
    truth, predicted = target[active] > 0, predictions[active] > EPS
    positive_mass = float(weights[truth].sum())
    negative_mass = float(weights[~truth].sum())
    constant = np.array([np.mean(train_target[train_mask[:, a], a]) if train_mask[:, a].any() else 0 for a in range(5)])
    mse = float(np.mean(((predictions[legal] - target[legal]) / scale) ** 2))
    baseline = float(np.mean(((np.broadcast_to(constant, target.shape)[legal] - target[legal]) / scale) ** 2))
    return {"nonzero_legal_state_action_pairs": int(active.sum()),
            "weighted_sign_accuracy": float(weights[predicted == truth].sum() / total) if total else None,
            "always_nonpositive_weighted_accuracy": negative_mass / total if total else None,
            "positive_gain_weighted_recall": float(weights[predicted & truth].sum() / positive_mass) if positive_mass else None,
            "negative_gain_weighted_specificity": float(weights[~predicted & ~truth].sum() / negative_mass) if negative_mass else None,
            "model_mse": mse, "training_operator_mean_mse": baseline,
            "mse_ratio_to_training_operator_mean": mse / baseline if baseline > EPS else None,
            "mse_ratio_undefined": baseline <= EPS,
            "label_unit": "legal non-Balanced state-action; absolute K3 mean gain weights"}


def training_subsets(fold, size):
    return [list(group) for group in combinations(sorted(fold["training"]), size)]


def feature_column_names(observation):
    from transformer_ppo.observations import (
        CANDIDATE_FEATURES, CANDIDATE_PATH_V2_FEATURES, CANDIDATE_PATH_V3_FEATURES,
        CANDIDATE_PATH_V4_FEATURES, GLOBAL_FEATURES, PARENT_FEATURES,
    )
    width = len(observation["candidates"][0])
    candidate = next((list(names) for names in (CANDIDATE_FEATURES, CANDIDATE_PATH_V2_FEATURES,
        CANDIDATE_PATH_V3_FEATURES, CANDIDATE_PATH_V4_FEATURES) if len(names) == width),
        [f"column_{i}" for i in range(width)])
    parent = list(PARENT_FEATURES) if len(observation["parent"][0]) == len(PARENT_FEATURES) else [f"column_{i}" for i in range(len(observation["parent"][0]))]
    global_names = list(GLOBAL_FEATURES) if len(observation["global"]) == len(GLOBAL_FEATURES) else [f"column_{i}" for i in range(len(observation["global"]))]
    return (["candidate." + n for n in candidate] + ["global." + n for n in global_names]
            + ["parent_mean." + n for n in parent] + ["parent_std." + n for n in parent]
            + ["log_parent_token_count"] + ["eligible." + a for a in OPERATORS] + ["operator_identity." + a for a in OPERATORS])
