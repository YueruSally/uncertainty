"""Transformer and MLP actor-critic models with identical action semantics."""
from __future__ import annotations

from typing import Dict

import torch
from torch import nn

from .config import OPERATORS
from .observations import CANDIDATE_FEATURES, GLOBAL_FEATURES, PARENT_FEATURES


def _signed_log1p(value):
    return torch.sign(value) * torch.log1p(torch.abs(value))


def _masked_mean(values, padding_mask):
    valid = (~padding_mask).unsqueeze(-1).to(values.dtype)
    return (values * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)


class _Heads(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.actor_heads = nn.ModuleList([nn.Linear(hidden_dim, 1) for _ in OPERATORS])
        self.critic_head = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1))

    def actor_logits(self, candidate_hidden, operator_id):
        all_logits = torch.stack(
            [head(candidate_hidden).squeeze(-1) for head in self.actor_heads], dim=1)
        index = operator_id[:, None, None].expand(-1, 1, candidate_hidden.shape[1])
        return all_logits.gather(1, index).squeeze(1)


class TransformerActorCritic(nn.Module):
    """Candidate self-attention followed by parent-candidate cross-attention."""

    architecture = "transformer"

    def __init__(self, hidden_dim=128, heads=4, layers=2, dropout=0.1):
        super().__init__()
        self.parent_projection = nn.Sequential(
            nn.Linear(len(PARENT_FEATURES), hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU())
        self.candidate_projection = nn.Sequential(
            nn.Linear(len(CANDIDATE_FEATURES), hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU())
        self.global_projection = nn.Sequential(
            nn.Linear(len(GLOBAL_FEATURES), hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU())
        self.operator_embedding = nn.Embedding(len(OPERATORS), hidden_dim)
        encoder_layer = lambda: nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=heads, dim_feedforward=hidden_dim * 4,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True)
        self.parent_encoder = nn.TransformerEncoder(encoder_layer(), num_layers=layers)
        self.candidate_encoder = nn.TransformerEncoder(encoder_layer(), num_layers=layers)
        self.cross_attention = nn.MultiheadAttention(
            hidden_dim, heads, dropout=dropout, batch_first=True)
        self.cross_norm = nn.LayerNorm(hidden_dim)
        self.heads = _Heads(hidden_dim)

    def forward(self, batch: Dict[str, torch.Tensor]):
        parent = self.parent_projection(_signed_log1p(batch["parent"]))
        candidate = self.candidate_projection(_signed_log1p(batch["candidates"]))
        operator = self.operator_embedding(batch["operator_id"]).unsqueeze(1)
        candidate = candidate + operator
        parent = self.parent_encoder(parent, src_key_padding_mask=batch["parent_padding_mask"])
        candidate = self.candidate_encoder(
            candidate, src_key_padding_mask=batch["candidate_padding_mask"])
        attended, _ = self.cross_attention(
            query=candidate, key=parent, value=parent,
            key_padding_mask=batch["parent_padding_mask"], need_weights=False)
        candidate = self.cross_norm(candidate + attended)
        logits = self.heads.actor_logits(candidate, batch["operator_id"])
        valid = batch["eligible_mask"] & ~batch["candidate_padding_mask"]
        logits = logits.masked_fill(~valid, torch.finfo(logits.dtype).min)
        parent_pool = _masked_mean(parent, batch["parent_padding_mask"])
        candidate_pool = _masked_mean(candidate, batch["candidate_padding_mask"])
        global_hidden = self.global_projection(_signed_log1p(batch["global"]))
        value = self.heads.critic_head(
            torch.cat([parent_pool, candidate_pool, global_hidden], dim=-1)).squeeze(-1)
        return logits, value


class MLPActorCritic(nn.Module):
    """Fair non-attention ablation using the same tokens, mask and PPO."""

    architecture = "mlp"

    def __init__(self, hidden_dim=128, **_):
        super().__init__()
        self.parent_projection = nn.Sequential(
            nn.Linear(len(PARENT_FEATURES), hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim))
        self.candidate_projection = nn.Sequential(
            nn.Linear(len(CANDIDATE_FEATURES) + len(GLOBAL_FEATURES) + hidden_dim,
                      hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim), nn.GELU())
        self.global_projection = nn.Sequential(
            nn.Linear(len(GLOBAL_FEATURES), hidden_dim), nn.GELU())
        self.operator_embedding = nn.Embedding(len(OPERATORS), hidden_dim)
        self.heads = _Heads(hidden_dim)

    def forward(self, batch: Dict[str, torch.Tensor]):
        parent = self.parent_projection(_signed_log1p(batch["parent"]))
        parent_pool = _masked_mean(parent, batch["parent_padding_mask"])
        count = batch["candidates"].shape[1]
        global_raw = _signed_log1p(batch["global"])
        inputs = torch.cat([
            _signed_log1p(batch["candidates"]),
            global_raw.unsqueeze(1).expand(-1, count, -1),
            parent_pool.unsqueeze(1).expand(-1, count, -1),
        ], dim=-1)
        candidate = self.candidate_projection(inputs)
        candidate = candidate + self.operator_embedding(batch["operator_id"]).unsqueeze(1)
        logits = self.heads.actor_logits(candidate, batch["operator_id"])
        valid = batch["eligible_mask"] & ~batch["candidate_padding_mask"]
        logits = logits.masked_fill(~valid, torch.finfo(logits.dtype).min)
        candidate_pool = _masked_mean(candidate, batch["candidate_padding_mask"])
        global_hidden = self.global_projection(global_raw)
        value = self.heads.critic_head(
            torch.cat([parent_pool, candidate_pool, global_hidden], dim=-1)).squeeze(-1)
        return logits, value


def build_model(architecture, config):
    kwargs = dict(hidden_dim=config.hidden_dim, heads=config.attention_heads,
                  layers=config.transformer_layers, dropout=config.dropout)
    if architecture == "transformer":
        return TransformerActorCritic(**kwargs)
    if architecture == "mlp":
        return MLPActorCritic(**kwargs)
    raise ValueError(f"unknown architecture: {architecture}")
