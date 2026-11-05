"""Matched summary/path-input models. Reads pre-action observations only.

Both arms use the identical hierarchical network. SUMMARY supplies the parent
mean/std as one two-token block; PATH_TOKEN supplies recorded allocation headers
and their ordered arcs. Neither arm receives invented batch/path IDs or outcomes.
"""
from __future__ import annotations

import copy
import hashlib
import time

import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence

from m_operator_evidence import recover_path_groups
from m_operator_offline import BALANCED, masked_loss

ARMS = ("SUMMARY", "PATH_TOKEN")


def parent_arrays(rows):
    arrays, groups = [], []
    for row in rows:
        parent = np.asarray(row["observation"]["parent"], dtype=np.float64)
        groups.append(recover_path_groups(parent))
        arrays.append(parent)
    return arrays, groups


def fit_parent_scaler(parents, training):
    # Fit on TRAINING instances only; the same values feed both arms.
    values = np.concatenate([parents[i] for i in training])
    mean, std = values.mean(0), values.std(0)
    std = np.where(std > 1e-8, std, 1.)
    if not np.isfinite(mean).all() or not np.isfinite(std).all():
        raise FloatingPointError("non-finite parent scaler")
    return {"mean": mean.tolist(), "std": std.tolist(), "feature_clip": 10.,
            "scope": "training-instance parent rows only; identical in both arms"}


def build_parent_inputs(parents, groups, scaler, arm):
    if arm not in ARMS:
        raise ValueError("unknown representation arm")
    result = []
    for parent, blocks in zip(parents, groups):
        z = np.clip((parent - np.asarray(scaler["mean"])) / np.asarray(scaler["std"]),
                    -scaler["feature_clip"], scaler["feature_clip"])
        if arm == "SUMMARY":
            # Same transformed primitive columns as PATH_TOKEN; aggregate only
            # after transformation. Roles identify mean vs std (not arc identity).
            values = np.concatenate([np.stack([z.mean(0), z.std(0)]), np.eye(2)], axis=1)
            result.append([values.astype(np.float32)])
        else:
            paths = []
            for block in blocks:
                indices = [block["header"], *block["arc_indices"]]
                roles = np.zeros((len(indices), 2))
                roles[0, 0], roles[1:, 1] = 1., 1.
                # Column 6 retains the recorded position within this path.
                paths.append(np.concatenate([z[indices], roles], axis=1).astype(np.float32))
            result.append(paths)
    return result


def pack_paths(inputs, indices):
    paths, counts = [], []
    for i in indices:
        blocks = inputs[int(i)]
        if not blocks or any(len(b) == 0 for b in blocks):
            raise ValueError("empty path block would create all-masked attention")
        counts.append(len(blocks))
        paths.extend(torch.as_tensor(b) for b in blocks)
    lengths = torch.tensor([len(p) for p in paths])
    padded = pad_sequence(paths, batch_first=True)
    valid = torch.arange(padded.shape[1])[None, :] < lengths[:, None]
    return padded, valid, counts


class AttentionBlock(nn.Module):
    def __init__(self, width=32):
        super().__init__()
        self.attention = nn.MultiheadAttention(width, 4, dropout=0., batch_first=True)
        self.norm1, self.norm2 = nn.LayerNorm(width), nn.LayerNorm(width)
        self.ff = nn.Sequential(nn.Linear(width, 2 * width), nn.GELU(), nn.Linear(2 * width, width))

    def forward(self, hidden, valid):
        if not bool(valid.any(1).all()):
            raise ValueError("all-masked attention row")
        attended, _ = self.attention(hidden, hidden, hidden, key_padding_mask=~valid, need_weights=False)
        hidden = self.norm1(hidden + attended)
        return self.norm2(hidden + self.ff(hidden))


class MatchedPathTransformer(nn.Module):
    def __init__(self, feature_count, width=32):
        super().__init__()
        self.projection = nn.Linear(feature_count, width)
        self.parent_projection = nn.Linear(21, width)
        self.path_block = AttentionBlock(width)
        self.cross_attention = nn.MultiheadAttention(width, 4, dropout=0., batch_first=True)
        self.cross_norm = nn.LayerNorm(width)
        self.operator_block = AttentionBlock(width)
        self.head = nn.Linear(width, 1)

    def forward(self, tokens, eligible, paths, path_valid, path_counts):
        if len(path_counts) != len(tokens) or sum(path_counts) != len(paths) or min(path_counts) < 1:
            raise ValueError("path/state grouping mismatch")
        # No artificial path ordering: attention inside each recorded path,
        # followed by masked pooling, then candidate queries over the path set.
        hidden = self.path_block(self.parent_projection(paths), path_valid)
        pooled = hidden.masked_fill(~path_valid[..., None], 0.).sum(1) / path_valid.sum(1)[:, None]
        memory = pad_sequence(torch.split(pooled, path_counts), batch_first=True)
        memory_valid = torch.arange(memory.shape[1])[None, :] < torch.tensor(path_counts)[:, None]
        candidates = self.projection(tokens)
        context, _ = self.cross_attention(candidates, memory, memory,
                                         key_padding_mask=~memory_valid, need_weights=False)
        candidates = self.operator_block(self.cross_norm(candidates + context), eligible)
        raw = self.head(candidates).squeeze(-1)
        return torch.cat([raw[:, :BALANCED], torch.zeros_like(raw[:, BALANCED:])], dim=1)


def weight_digest(model):
    sha = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        sha.update(name.encode())
        sha.update(tensor.detach().cpu().numpy().tobytes())
    return sha.hexdigest()


def forward_indices(model, x, mask, inputs, ids):
    return model(x[ids], mask[ids], *pack_paths(inputs, ids))


def predict(model, x, mask, inputs, indices=None, microbatch=4):
    model.eval()
    ids = np.arange(len(x)) if indices is None else np.asarray(indices)
    xx, mm = torch.as_tensor(x), torch.as_tensor(mask)
    outputs = []
    with torch.no_grad():
        for start in range(0, len(ids), microbatch):
            outputs.append(forward_indices(model, xx, mm, inputs, ids[start:start + microbatch]).numpy())
    result = np.concatenate(outputs).astype(float)
    if not np.isfinite(result).all():
        raise FloatingPointError("non-finite representation predictions")
    return result


def fit(x, mask, target, inputs, train, validation, seed, max_updates=320, progress=None):
    """Effective batch32, microbatch4, exactly weighted by legal action count.

    Equal optimizer-update caps, data permutations and validation schedule for
    both representations. Separate inner-validation stopping may end them at
    different steps; actual steps and elapsed time are retained.
    """
    if len(train) % 32 or max_updates < 1:
        raise ValueError("requires whole 32-state batches and positive updates")
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = MatchedPathTransformer(x.shape[-1])
    initial = weight_digest(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    xx, mm, yy = torch.as_tensor(x), torch.as_tensor(mask), torch.as_tensor(target)
    train, validation = np.asarray(train), np.asarray(validation)
    cursor, order = len(train), None
    started, last_log = time.monotonic(), time.monotonic()
    history, best, best_loss, best_step = [], None, float("inf"), 0
    for step in range(max_updates + 1):
        if step:
            model.train()
            if cursor >= len(train):
                order, cursor = rng.permutation(train), 0
            batch = order[cursor:cursor + 32]
            cursor += 32
            count = int(mm[batch, :BALANCED].sum())
            if not count:
                raise ValueError("no legal alternatives in training minibatch")
            optimizer.zero_grad(set_to_none=True)
            for start in range(0, len(batch), 4):
                ids = batch[start:start + 4]
                n = int(mm[ids, :BALANCED].sum())
                if not n:
                    continue
                output = forward_indices(model, xx, mm, inputs, ids)
                loss = masked_loss(output, yy[ids], mm[ids]) * (n / count)
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError("non-finite path representation loss")
                loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
            optimizer.step()
            now = time.monotonic()
            if progress and now - last_log >= 10:
                progress(f"updates={step}/{max_updates} elapsed={now-started:.1f}s")
                last_log = now
        if step % 8 and step != max_updates:
            continue
        tr = predict(model, x, mask, inputs, train)
        va = predict(model, x, mask, inputs, validation)
        tr_mask, va_mask = mask[train].copy(), mask[validation].copy()
        tr_mask[:, BALANCED], va_mask[:, BALANCED] = False, False
        tr_loss = float(np.mean((tr[tr_mask] - target[train][tr_mask]) ** 2))
        va_loss = float(np.mean((va[va_mask] - target[validation][va_mask]) ** 2))
        if not np.isfinite([tr_loss, va_loss]).all():
            raise FloatingPointError("non-finite representation validation loss")
        history.append({"update": step, "training_mse": tr_loss, "validation_mse": va_loss})
        if va_loss < best_loss - 1e-10:
            best_loss, best_step, best = va_loss, step, copy.deepcopy(model.state_dict())
        if progress:
            progress(f"updates={step}/{max_updates} train_mse={tr_loss:.6g} valid_mse={va_loss:.6g} elapsed={time.monotonic()-started:.1f}s")
            last_log = time.monotonic()
        if step - best_step >= 64:
            break
    model.load_state_dict(best)
    return model, {"seed": seed, "selected_update": best_step, "optimizer_updates": step,
                   "history": history, "initial_weights_sha256": initial,
                   "selected_weights_sha256": weight_digest(model),
                   "parameter_count": sum(p.numel() for p in model.parameters()),
                   "elapsed_seconds": time.monotonic() - started}
