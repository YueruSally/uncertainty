"""Directed-default residual learning for I-K8 initialization."""
from __future__ import annotations

from collections import Counter
import copy

import numpy as np
import torch

from i_k8_surrogate import CandidateGainTransformer, pack_rows


DIRECTED_ORIGINAL_INDEX = 1
MODEL_ORDER = (1, 0, 2, 3, 4, 5, 6, 7)


def reorder_row(row):
    """Move Directed to model index zero, which is fixed to residual zero."""
    result = copy.deepcopy(row)
    for key in ("candidate_features", "candidate_batch_tokens", "labels",
                "candidate_fingerprints", "candidate_roles"):
        if key in result:
            values = result[key]
            if len(values) != 8:
                raise ValueError(f"I-K8 residual {key} must have length eight")
            result[key] = [values[index] for index in MODEL_ORDER]
    return result


def original_index(model_index):
    if not 0 <= int(model_index) < 8:
        raise ValueError("I-K8 residual model index is invalid")
    return MODEL_ORDER[int(model_index)]


def ensemble_predictions(models, rows, scaler, target_scale):
    ordered = [reorder_row(row) for row in rows]
    packed = pack_rows(ordered, scaler)
    values = []
    with torch.inference_mode():
        for model in models:
            model.eval()
            values.append(model(*packed).numpy() * float(target_scale))
    predictions = np.asarray(values)
    if predictions.shape != (len(models), len(rows), 8):
        raise ValueError("I-K8 residual ensemble output shape changed")
    if not np.allclose(predictions[:, :, 0], 0.0, atol=1e-12):
        raise ValueError("Directed residual must be exactly zero")
    return predictions


def gate_decisions(predictions, uncertainty_weight, threshold):
    predictions = np.asarray(predictions, dtype=float)
    mean = predictions.mean(0)
    standard = predictions.std(0)
    conservative = mean - float(uncertainty_weight) * standard
    result = []
    for values in conservative:
        best = int(np.argmax(values))
        result.append(best if values[best] > float(threshold) else 0)
    return np.asarray(result, dtype=int), mean, standard, conservative


def score_gate(rows, predictions, uncertainty_weight, threshold):
    ordered = [reorder_row(row) for row in rows]
    selected, mean, standard, conservative = gate_decisions(
        predictions, uncertainty_weight, threshold)
    decisions = []
    for index, (row, action) in enumerate(zip(ordered, selected)):
        labels = row["labels"]
        reference = float(labels[0]["hv_contribution"])
        gain = float(labels[int(action)]["hv_contribution"]) - reference
        oracle = max(float(label["hv_contribution"]) - reference
                     for label in labels)
        decisions.append({
            "instance": row["instance"],
            "trajectory_seed": row["trajectory_seed"],
            "event_index": row["event_index"],
            "selected_model_index": int(action),
            "selected_original_index": original_index(action),
            "net_gain_vs_directed": gain,
            "oracle_gain_vs_directed": oracle,
            "predicted_mean": float(mean[index, action]),
            "predicted_std": float(standard[index, action]),
            "conservative_score": float(conservative[index, action]),
        })

    def summarize(part):
        gains = np.asarray([row["net_gain_vs_directed"] for row in part])
        oracle = np.asarray([row["oracle_gain_vs_directed"] for row in part])
        return {
            "states": len(part),
            "mean_net_gain_vs_directed": float(gains.mean()),
            "switch_fraction": float(np.mean([
                row["selected_model_index"] != 0 for row in part])),
            "positive_gain_total": float(gains[gains > 0].sum()),
            "negative_gain_total": float(gains[gains < 0].sum()),
            "gain_p10": float(np.quantile(gains, .1)),
            "mean_oracle_gain_vs_directed": float(oracle.mean()),
            "oracle_gain_fraction_captured": (
                float(gains.mean() / oracle.mean()) if oracle.mean() > 0 else None),
            "action_counts": dict(Counter(
                str(row["selected_original_index"]) for row in part)),
        }

    per_instance = []
    for instance in sorted({row["instance"] for row in decisions}):
        part = [row for row in decisions if row["instance"] == instance]
        per_instance.append({"instance": instance, **summarize(part)})
    return {**summarize(decisions), "per_instance": per_instance,
            "decisions": decisions}


class FrozenIResidualGate:
    """Frozen three-seed residual ensemble with Directed safe fallback."""

    def __init__(self, bundle, digest_function):
        if bundle.get("protocol") != "i-k8-directed-residual-v1":
            raise ValueError("incorrect I-K8 residual bundle")
        self.scaler = bundle["scaler"]
        self.target_scale = float(bundle["target_scale"])
        self.uncertainty_weight = float(bundle["gate"]["uncertainty_weight"])
        self.threshold = float(bundle["gate"]["threshold"])
        self.models = []
        for filename in bundle["checkpoints"]:
            if digest_function(filename) != bundle["checkpoint_sha256"][filename]:
                raise ValueError(f"I residual checkpoint digest mismatch: {filename}")
            try:
                checkpoint = torch.load(
                    filename, map_location="cpu", weights_only=True)
            except TypeError as exc:
                if "weights_only" not in str(exc): raise
                checkpoint = torch.load(filename, map_location="cpu")
            if (checkpoint.get("protocol") != "i-k8-directed-residual-model-v1"
                    or checkpoint.get("training_instances") != 200
                    or checkpoint.get("width") != 16):
                raise ValueError("I residual checkpoint identity changed")
            model = CandidateGainTransformer(
                checkpoint["batch_feature_count"],
                checkpoint["candidate_feature_count"],
                checkpoint["context_feature_count"], width=16)
            model.load_state_dict(checkpoint["model"], strict=True)
            model.eval()
            for parameter in model.parameters(): parameter.requires_grad_(False)
            self.models.append(model)
        if len(self.models) != 3:
            raise ValueError("I residual gate requires three models")

    def select(self, row):
        predictions = ensemble_predictions(
            self.models, [row], self.scaler, self.target_scale)
        selected, _, _, _ = gate_decisions(
            predictions, self.uncertainty_weight, self.threshold)
        return original_index(int(selected[0]))


__all__ = [
    "DIRECTED_ORIGINAL_INDEX", "MODEL_ORDER", "FrozenIResidualGate",
    "reorder_row", "original_index", "ensemble_predictions",
    "gate_decisions", "score_gate",
]
