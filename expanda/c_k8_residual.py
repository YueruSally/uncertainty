"""Simple-default residual learning for C-K8 crossover."""
from __future__ import annotations

from collections import Counter
import copy

import numpy as np
import torch

from i_k8_surrogate import CandidateGainTransformer, pack_rows


def simple_index(row):
    """Reproduce C-K8 Simple using only pre-evaluation candidate features."""
    features = row.get("candidate_features", ())
    if len(features) != 8:
        raise ValueError("C-K8 residual requires eight candidate features")
    selected = min(range(8), key=lambda index: (
        float(features[index][0]), index != 1, index))
    recorded = row.get("simple_selected_index")
    if recorded is not None and int(recorded) != selected:
        raise ValueError("recorded C-K8 Simple decision is not reproducible")
    return selected


def model_order(row):
    selected = simple_index(row)
    return (selected, *(index for index in range(8) if index != selected))


def reorder_row(row):
    """Move this state's Simple choice to model index zero."""
    result = copy.deepcopy(row)
    order = model_order(row)
    for key in ("candidate_features", "candidate_batch_tokens", "labels",
                "candidate_fingerprints", "candidate_roles"):
        if key in result:
            values = result[key]
            if len(values) != 8:
                raise ValueError(f"C-K8 residual {key} must have length eight")
            result[key] = [values[index] for index in order]
    result["residual_original_order"] = list(order)
    result["simple_selected_index"] = 0
    return result


def original_index(row, residual_index):
    order = model_order(row)
    if not 0 <= int(residual_index) < len(order):
        raise ValueError("C-K8 residual index is invalid")
    return int(order[int(residual_index)])


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
        raise ValueError("C-K8 residual ensemble output shape changed")
    if not np.allclose(predictions[:, :, 0], 0.0, atol=1e-12):
        raise ValueError("Simple residual must be exactly zero")
    return predictions


def gate_decisions(predictions, uncertainty_weight, threshold):
    predictions = np.asarray(predictions, dtype=float)
    mean = predictions.mean(0)
    standard = predictions.std(0)
    conservative = mean - float(uncertainty_weight) * standard
    selected = []
    for values in conservative:
        best = int(np.argmax(values))
        selected.append(best if values[best] > float(threshold) else 0)
    return np.asarray(selected, dtype=int), mean, standard, conservative


def score_gate(rows, predictions, uncertainty_weight, threshold):
    selected, mean, standard, conservative = gate_decisions(
        predictions, uncertainty_weight, threshold)
    decisions = []
    for index, (row, action) in enumerate(zip(rows, selected)):
        reference_index = simple_index(row)
        chosen_index = original_index(row, action)
        labels = row["labels"]
        reference = float(labels[reference_index]["hv_contribution"])
        gain = float(labels[chosen_index]["hv_contribution"]) - reference
        oracle = max(float(label["hv_contribution"]) - reference
                     for label in labels)
        decisions.append({
            "instance": row["instance"],
            "trajectory_seed": row["trajectory_seed"],
            "event_index": row["event_index"],
            "simple_original_index": reference_index,
            "selected_residual_index": int(action),
            "selected_original_index": chosen_index,
            "net_gain_vs_simple": gain,
            "oracle_gain_vs_simple": oracle,
            "predicted_mean": float(mean[index, action]),
            "predicted_std": float(standard[index, action]),
            "conservative_score": float(conservative[index, action]),
        })

    def summarize(part):
        gains = np.asarray([row["net_gain_vs_simple"] for row in part])
        oracle = np.asarray([row["oracle_gain_vs_simple"] for row in part])
        return {
            "states": len(part),
            "mean_net_gain_vs_simple": float(gains.mean()),
            "switch_fraction": float(np.mean([
                row["selected_residual_index"] != 0 for row in part])),
            "positive_gain_total": float(gains[gains > 0].sum()),
            "negative_gain_total": float(gains[gains < 0].sum()),
            "gain_p10": float(np.quantile(gains, .1)),
            "mean_oracle_gain_vs_simple": float(oracle.mean()),
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


class FrozenCResidualGate:
    """Frozen three-seed residual ensemble with C-K8 Simple fallback."""

    def __init__(self, bundle, digest_function):
        if bundle.get("protocol") != "c-k8-simple-residual-v1":
            raise ValueError("incorrect C-K8 residual bundle")
        self.scaler = bundle["scaler"]
        self.target_scale = float(bundle["target_scale"])
        self.uncertainty_weight = float(bundle["gate"]["uncertainty_weight"])
        self.threshold = float(bundle["gate"]["threshold"])
        self.models = []
        for filename in bundle["checkpoints"]:
            if digest_function(filename) != bundle["checkpoint_sha256"][filename]:
                raise ValueError(f"C residual checkpoint digest mismatch: {filename}")
            try:
                checkpoint = torch.load(
                    filename, map_location="cpu", weights_only=True)
            except TypeError as exc:
                if "weights_only" not in str(exc):
                    raise
                checkpoint = torch.load(filename, map_location="cpu")
            if (checkpoint.get("protocol") != "c-k8-simple-residual-model-v1"
                    or checkpoint.get("training_instances") != 200
                    or checkpoint.get("width") != 16):
                raise ValueError("C residual checkpoint identity changed")
            model = CandidateGainTransformer(
                checkpoint["batch_feature_count"],
                checkpoint["candidate_feature_count"],
                checkpoint["context_feature_count"], width=16)
            model.load_state_dict(checkpoint["model"], strict=True)
            model.eval()
            for parameter in model.parameters():
                parameter.requires_grad_(False)
            self.models.append(model)
        if len(self.models) != 3:
            raise ValueError("C residual gate requires three models")

    def select(self, row):
        predictions = ensemble_predictions(
            self.models, [row], self.scaler, self.target_scale)
        selected, _, _, _ = gate_decisions(
            predictions, self.uncertainty_weight, self.threshold)
        return original_index(row, int(selected[0]))


__all__ = [
    "FrozenCResidualGate", "simple_index", "model_order", "reorder_row",
    "original_index", "ensemble_predictions", "gate_decisions", "score_gate",
]
