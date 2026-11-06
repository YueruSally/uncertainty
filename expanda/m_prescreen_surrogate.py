"""Small set Transformer for CCP100 objective prediction over K8 children."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from transformer_ppo.archive import exact_hv_3d


class CandidateObjectiveTransformer(nn.Module):
    def __init__(self, batch_feature_count, candidate_feature_count,
                 context_feature_count, width=32):
        super().__init__()
        self.batch_projection = nn.Linear(batch_feature_count, width)
        batch_layer = nn.TransformerEncoderLayer(width, 4, width*2,
            dropout=0., activation='gelu', batch_first=True, norm_first=False)
        self.batch_encoder = nn.TransformerEncoder(batch_layer, 1)
        self.candidate_projection = nn.Linear(
            width+candidate_feature_count+context_feature_count+1, width)
        candidate_layer = nn.TransformerEncoderLayer(width, 4, width*2,
            dropout=0., activation='gelu', batch_first=True, norm_first=False)
        self.candidate_encoder = nn.TransformerEncoder(candidate_layer, 1)
        self.objective_head = nn.Linear(width, 3)
        self.feasibility_head = nn.Linear(width, 1)

    def forward(self, batch_tokens, candidate_features, context, balanced):
        if batch_tokens.ndim != 4 or candidate_features.ndim != 3:
            raise ValueError('expected [state,candidate,batch,feature] inputs')
        states, candidates, batches, features = batch_tokens.shape
        hidden = self.batch_projection(batch_tokens.reshape(
            states*candidates, batches, features))
        hidden = self.batch_encoder(hidden).mean(1).reshape(states, candidates, -1)
        repeated = context[:, None, :].expand(-1, candidates, -1)
        hidden = self.candidate_projection(torch.cat(
            (hidden, candidate_features, repeated, balanced), dim=-1))
        hidden = self.candidate_encoder(hidden)
        return self.objective_head(hidden), self.feasibility_head(hidden).squeeze(-1)


def state_context(row):
    points = np.asarray(row['population_points'], dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not len(points):
        raise ValueError('state lacks a finite three-objective population')
    context = np.concatenate((points.mean(0), points.std(0),
        points.min(0), points.max(0),
        [float(row['generation'])/20., float(row['evaluation'])/400.]))
    if not np.isfinite(context).all():
        raise ValueError('non-finite prescreen context')
    return context


def fit_scaler(rows, normalizer):
    candidate = np.concatenate([np.asarray(row['candidate_features'], dtype=float)
                                for row in rows], axis=0)
    batch = np.concatenate([np.asarray(row['candidate_batch_tokens'], dtype=float).reshape(
        -1, len(row['candidate_batch_tokens'][0][0])) for row in rows], axis=0)
    context = np.stack([state_context(row) for row in rows])
    targets = np.asarray([normalizer.transform(label['objectives'])
        for row in rows for label in row['labels'] if label['feasible']], dtype=float)

    def moments(values):
        std = values.std(0)
        return {'mean': values.mean(0).tolist(),
                'std': np.where(std > 1e-8, std, 1.).tolist()}
    if not len(targets):
        raise ValueError('no feasible objective labels')
    return {'candidate': moments(candidate), 'batch': moments(batch),
        'context': moments(context), 'target': moments(targets), 'clip': 10.}


def scaled(values, part, scaler):
    values = np.asarray(values, dtype=np.float32)
    mean = np.asarray(scaler[part]['mean'], dtype=np.float32)
    std = np.asarray(scaler[part]['std'], dtype=np.float32)
    return np.clip((values-mean)/std, -scaler['clip'], scaler['clip'])


def pack_rows(rows, scaler, normalizer):
    batch = np.stack([scaled(row['candidate_batch_tokens'], 'batch', scaler)
                      for row in rows])
    candidate = np.stack([scaled(row['candidate_features'], 'candidate', scaler)
                          for row in rows])
    context = scaled(np.stack([state_context(row) for row in rows]), 'context', scaler)
    balanced = np.zeros((len(rows), len(rows[0]['labels']), 1), np.float32)
    balanced[:, 0, 0] = 1.
    feasible = np.asarray([[label['feasible'] for label in row['labels']]
                           for row in rows], dtype=np.float32)
    mean = np.asarray(scaler['target']['mean'])
    std = np.asarray(scaler['target']['std'])
    objectives = np.zeros((len(rows), len(rows[0]['labels']), 3), np.float32)
    for i, row in enumerate(rows):
        for j, label in enumerate(row['labels']):
            if label['feasible']:
                objectives[i, j] = (np.asarray(normalizer.transform(
                    label['objectives']))-mean)/std
    return tuple(torch.as_tensor(value) for value in (
        batch, candidate, context, balanced, objectives, feasible))


def loss_value(predicted, logits, target, feasible):
    mask = feasible.bool()
    if not bool(mask.any()):
        raise ValueError('batch contains no feasible candidate labels')
    objective = nn.functional.smooth_l1_loss(predicted[mask], target[mask])
    classification = nn.functional.binary_cross_entropy_with_logits(logits, feasible)
    pairwise = predicted[:, :, None, :]-predicted[:, None, :, :]
    true_pairwise = target[:, :, None, :]-target[:, None, :, :]
    pair_mask = mask[:, :, None] & mask[:, None, :]
    ranking = nn.functional.smooth_l1_loss(pairwise[pair_mask], true_pairwise[pair_mask])
    return objective + .2*classification + .25*ranking


def predict(model, rows, scaler, normalizer):
    packed = pack_rows(rows, scaler, normalizer)
    with torch.inference_mode():
        objective, logits = model(*packed[:4])
    mean = np.asarray(scaler['target']['mean'])
    std = np.asarray(scaler['target']['std'])
    return objective.numpy()*std+mean, torch.sigmoid(logits).numpy()


def choose_by_predicted_hv(rows, normalized_objectives,
                           feasibility_probability, reference):
    selected = []
    for row, objectives, probability in zip(
            rows, normalized_objectives, feasibility_probability):
        before = exact_hv_3d(row['population_points'], reference)
        contribution = []
        for point, feasible in zip(objectives, probability >= .5):
            contribution.append(exact_hv_3d(
                row['population_points']+([point.tolist()] if feasible else []),
                reference)-before)
        best = max(contribution)
        # Deterministic tie handling preserves Balanced candidate zero.
        selected.append(next(index for index, value in enumerate(contribution)
                             if value >= best-1e-12))
    return np.asarray(selected, dtype=int)
