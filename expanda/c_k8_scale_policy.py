"""Frozen size-200 C-K8 Transformer inference."""
from __future__ import annotations

import numpy as np
import torch

from i_k8_surrogate import (
    CandidateGainTransformer, choose_first_maximum, pack_rows)


class FrozenCCrossoverRanker:
    """Load the fixed three-seed C-K8 ensemble once per search run."""

    def __init__(self, bundle, digest_function):
        if (bundle.get("protocol") != "c-k8-scale-transformer-curve-v1"
                or bundle.get("selected_size") != 200):
            raise ValueError("incorrect C-K8 Transformer bundle")
        selected = bundle.get("models", {}).get("200")
        if selected is None:
            raise ValueError("C-K8 bundle has no size-200 model")
        self.scaler = selected["scaler"]
        self.target_scale = float(selected["target_scale"])
        self.models = []
        for filename in selected["checkpoints"]:
            expected = selected["checkpoint_sha256"].get(filename)
            if expected is None or digest_function(filename) != expected:
                raise ValueError(f"C-K8 checkpoint digest mismatch: {filename}")
            try:
                checkpoint = torch.load(
                    filename, map_location="cpu", weights_only=True)
            except TypeError as exc:
                if "weights_only" not in str(exc):
                    raise
                checkpoint = torch.load(filename, map_location="cpu")
            if (checkpoint.get("protocol") != "c-k8-scale-transformer-v1"
                    or checkpoint.get("training_instances") != 200
                    or checkpoint.get("width") != 16):
                raise ValueError("C-K8 checkpoint identity mismatch")
            model = CandidateGainTransformer(
                checkpoint["batch_feature_count"],
                checkpoint["candidate_feature_count"],
                checkpoint["context_feature_count"],
                width=checkpoint["width"])
            model.load_state_dict(checkpoint["model"], strict=True)
            model.eval()
            for parameter in model.parameters(): parameter.requires_grad_(False)
            self.models.append(model)
        if len(self.models) != 3:
            raise ValueError("C-K8 size-200 ensemble must contain three models")

    def predict(self, row):
        packed = pack_rows([row], self.scaler)
        with torch.inference_mode():
            predictions = np.mean([
                model(*packed).numpy() * self.target_scale
                for model in self.models], axis=0)[0]
        if predictions.shape != (8,) or abs(float(predictions[0])) > 1e-12:
            raise ValueError("C-K8 ranker output contract changed")
        return predictions

    def select(self, row):
        return choose_first_maximum(self.predict(row))
