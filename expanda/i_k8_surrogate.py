"""Small Transformer utilities for complete-individual I-K8 ranking."""
from __future__ import annotations

from collections import Counter

import numpy as np
import torch

from m_prescreen_surrogate import CandidateGainTransformer


def state_context(row):
    points = np.asarray(row["population_points"], dtype=np.float64)
    if points.size == 0:
        summary = np.zeros(12, dtype=np.float64)
    else:
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("I-K8 population points must be Nx3")
        summary = np.concatenate((
            points.mean(0), points.std(0), points.min(0), points.max(0)))
    context = np.concatenate((summary, [
        float(row["slot_fraction"]),
        len(points) / 20.0,
    ]))
    if not np.isfinite(context).all():
        raise ValueError("non-finite I-K8 context")
    return context


def _moments(values):
    values = np.asarray(values, dtype=np.float64)
    standard = values.std(0)
    return {
        "mean": values.mean(0).tolist(),
        "std": np.where(standard > 1e-8, standard, 1.0).tolist(),
    }


def fit_scaler(rows):
    candidate = np.concatenate([
        np.asarray(row["candidate_features"], dtype=float) for row in rows
    ], axis=0)
    batch = np.concatenate([
        np.asarray(row["candidate_batch_tokens"], dtype=float).reshape(
            -1, len(row["candidate_batch_tokens"][0][0]))
        for row in rows
    ], axis=0)
    context = np.stack([state_context(row) for row in rows])
    return {
        "candidate": _moments(candidate),
        "batch": _moments(batch),
        "context": _moments(context),
        "clip": 10.0,
    }


def _scaled(values, part, scaler):
    values = np.asarray(values, dtype=np.float32)
    mean = np.asarray(scaler[part]["mean"], dtype=np.float32)
    standard = np.asarray(scaler[part]["std"], dtype=np.float32)
    return np.clip((values - mean) / standard,
                   -float(scaler["clip"]), float(scaler["clip"]))


def pack_rows(rows, scaler):
    batch = np.stack([
        _scaled(row["candidate_batch_tokens"], "batch", scaler)
        for row in rows
    ])
    candidate = np.stack([
        _scaled(row["candidate_features"], "candidate", scaler)
        for row in rows
    ])
    context = _scaled(
        np.stack([state_context(row) for row in rows]), "context", scaler)
    base = np.zeros(
        (len(rows), len(rows[0]["candidate_features"]), 1), np.float32)
    base[:, 0, 0] = 1.0
    return tuple(torch.as_tensor(value) for value in (
        batch, candidate, context, base))


def gain_targets(rows, target_scale):
    values = np.asarray([
        [float(label["hv_contribution"])
         - float(row["labels"][0]["hv_contribution"])
         for label in row["labels"]]
        for row in rows
    ], dtype=np.float32)
    return torch.as_tensor(values / float(target_scale))


def pairwise_ranking_loss(predicted, target):
    """Magnitude-weighted within-state ranking; exact ties carry no gradient."""
    difference = predicted[:, :, None] - predicted[:, None, :]
    truth = target[:, :, None] - target[:, None, :]
    candidates = predicted.shape[1]
    upper = torch.triu(torch.ones(
        (candidates, candidates), dtype=torch.bool,
        device=predicted.device), diagonal=1)
    mask = upper[None] & (truth.abs() > 1e-6)
    if not bool(mask.any()):
        return predicted.sum() * 0.0
    sign = truth[mask].sign()
    weight = truth[mask].abs().clamp(0.1, 5.0)
    return (weight * torch.nn.functional.softplus(
        -sign * difference[mask])).sum() / weight.sum()


def choose_first_maximum(values):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or len(values) != 8 or not np.isfinite(values).all():
        raise ValueError("I-K8 predictions must be a finite length-eight vector")
    best = float(values.max())
    return next(index for index, value in enumerate(values)
                if value >= best - 1e-12)


def predict(model, rows, scaler, target_scale, batch_size=128):
    values = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(rows), batch_size):
            packed = pack_rows(rows[start:start + batch_size], scaler)
            values.append(model(*packed).numpy() * float(target_scale))
    return np.concatenate(values)


def score(rows, predictions):
    decisions = []
    for row, values in zip(rows, np.asarray(predictions)):
        selected = choose_first_maximum(values)
        labels = row["labels"]
        base_value = float(labels[0]["hv_contribution"])
        simple = int(row["simple_selected_index"])
        actual = float(labels[selected]["hv_contribution"]) - base_value
        simple_gain = float(labels[simple]["hv_contribution"]) - base_value
        oracle = max(float(label["hv_contribution"]) - base_value
                     for label in labels)
        decisions.append({
            "instance": row["instance"],
            "trajectory_seed": row["trajectory_seed"],
            "event_index": row["event_index"],
            "selected": selected,
            "simple_selected": simple,
            "net_gain_vs_base": actual,
            "simple_gain_vs_base": simple_gain,
            "transformer_minus_simple": actual - simple_gain,
            "oracle_gain_vs_base": oracle,
        })

    def summarize(part):
        transformer = np.asarray(
            [row["net_gain_vs_base"] for row in part], dtype=float)
        simple = np.asarray(
            [row["simple_gain_vs_base"] for row in part], dtype=float)
        difference = transformer - simple
        return {
            "states": len(part),
            "transformer_mean_net_gain_vs_base": float(transformer.mean()),
            "simple_mean_net_gain_vs_base": float(simple.mean()),
            "transformer_minus_simple_mean": float(difference.mean()),
            "transformer_switch_fraction": float(np.mean([
                row["selected"] != 0 for row in part])),
            "simple_switch_fraction": float(np.mean([
                row["simple_selected"] != 0 for row in part])),
            "transformer_positive_gain_total": float(
                transformer[transformer > 0].sum()),
            "transformer_negative_gain_total": float(
                transformer[transformer < 0].sum()),
            "transformer_gain_p10": float(np.quantile(transformer, .1)),
            "mean_oracle_gain_vs_base": float(np.mean([
                row["oracle_gain_vs_base"] for row in part])),
            "action_counts": dict(Counter(
                str(row["selected"]) for row in part)),
        }

    per_instance = []
    for instance in sorted({row["instance"] for row in decisions}):
        part = [row for row in decisions if row["instance"] == instance]
        per_instance.append({"instance": instance, **summarize(part)})
    return {**summarize(decisions), "per_instance": per_instance,
            "decisions": decisions}


__all__ = [
    "CandidateGainTransformer", "fit_scaler", "pack_rows", "gain_targets",
    "pairwise_ranking_loss", "predict", "score", "choose_first_maximum",
]
