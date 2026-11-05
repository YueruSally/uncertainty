"""Small, grouped offline operator-return experiment (no search or PPO imports)."""
from __future__ import annotations

import copy
import hashlib
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch import nn

OPERATORS = ("add", "del", "mod", "mode", "replace")
BALANCED = 4
PHASES = ("early", "middle", "late")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def grouped_folds(instances):
    """Outer instance is never used for preprocessing, stopping or thresholds."""
    names = sorted(set(instances))
    if len(names) < 4:
        raise ValueError("need at least four distinct instances")
    return [{"test": name, "validation": names[(i + 1) % len(names)],
             "training": [n for n in names if n not in (name, names[(i + 1) % len(names)])]}
            for i, name in enumerate(names)]


def preaction_features(observation):
    """Both learners receive exactly the same five candidate/state vectors.

    Parent tokens are summarized to bound memory. This does NOT test a full
    path/graph Transformer. Neither learner receives counterfactual outcomes,
    instance IDs, policy checkpoints, best-operator labels or future states.
    """
    candidates = np.asarray(observation["candidates"], dtype=np.float64)
    parent = np.asarray(observation["parent"], dtype=np.float64)
    global_features = np.asarray(observation["global"], dtype=np.float64)
    mask = np.asarray(observation["eligible"], dtype=bool)
    if candidates.ndim != 2 or len(candidates) != 5 or mask.shape != (5,):
        raise ValueError("expected five operator tokens and a five-action legal mask")
    if parent.ndim != 2 or len(parent) == 0 or global_features.ndim != 1:
        raise ValueError("missing pre-action parent/global features")
    context = np.concatenate([global_features, parent.mean(0), parent.std(0),
                              [np.log1p(len(parent))], mask.astype(float)])
    tokens = np.concatenate([candidates, np.tile(context, (5, 1)), np.eye(5)], axis=1)
    if not np.isfinite(tokens).all():
        raise ValueError("non-finite pre-action feature; do not silently replace with zero")
    return tokens, mask


def load_panel(collection):
    collection = Path(collection)
    quality = read_json(collection / "batch_2/QUALITY.json")
    data = read_json(collection / "batch_2/DATASET.json")
    completion = read_json(collection / "BATCH_2_COMPLETE")
    plan = read_json(collection / "PLAN.json")
    design = {"scenario_count": 100, "ccp_confidence": 0.9, "population_size": 20,
              "evaluation_budget": 400, "replicates_per_legal_operator": 3,
              "execution_contract": "operator_shared_v1"}
    if any(plan.get(k) != v for k, v in design.items()):
        raise ValueError("source plan must use the shared execution contract and CCP100/pop20/B400/K3")
    plan_hash = hashlib.sha256(json.dumps(plan, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if any(value.get("plan_sha256") != plan_hash for value in (quality, data, completion)):
        raise ValueError("collection plan hashes disagree")
    required = {"data_quality_pass": True, "replay_exact": True, "state_count": 384,
                "trajectory_count": 12, "instance_count": 6, "scenario_count": 100,
                "population_size": 20, "evaluation_budget_per_trajectory": 400,
                "new_training_steps": 0}
    if any(quality.get(k) != v for k, v in required.items()):
        raise ValueError("requires completed, verified 6-instance/12-trajectory CCP100 panel")
    rows = data["states"]
    if len(rows) != 384 or completion.get("states") != 384:
        raise ValueError("panel state count differs from quality report")
    checked = {r["run"]: r for r in quality["checked_runs"]}
    if len(checked) != 12 or not all(r.get("replay_exact") is True for r in checked.values()):
        raise ValueError("missing trajectory replay evidence")
    keys, tokens, masks, outcomes = set(), [], [], []
    for row in rows:
        key = (row["run"], row["state_id"])
        if key in keys or row["run"] not in checked:
            raise ValueError("duplicate or unverified state")
        keys.add(key)
        if row["instance"] != checked[row["run"]]["instance"]:
            raise ValueError("trajectory/instance mismatch")
        if tuple(row["action_order"]) != OPERATORS or row["phase"] not in PHASES:
            raise ValueError("unexpected action order or search phase")
        x, mask = preaction_features(row["observation"])
        if mask.tolist() != [a in row["eligible_operators"] for a in OPERATORS]:
            raise ValueError("legal mask disagrees with recorded feedback")
        # A missing Balanced action is not a zero-return training example.
        if not mask[BALANCED] or not row["balanced_available"]:
            raise ValueError(f"Balanced unavailable in {key}; explicitly redesign comparison before fitting")
        q = np.full((5, 3), np.nan)
        for action, name in enumerate(OPERATORS):
            if mask[action]:
                values = np.asarray(row["operators"][name]["population_delta_hv_by_replicate"], dtype=float)
                if values.shape != (3,) or not np.isfinite(values).all():
                    raise ValueError("legal action needs three finite repeats")
                q[action] = values
            elif name in row["operators"]:
                raise ValueError("illegal action has feedback")
        tokens.append(x)
        masks.append(mask)
        outcomes.append(q)
    counts = Counter(r["run"] for r in rows)
    instance_counts = Counter(r["instance"] for r in rows)
    if len(counts) != 12 or set(counts.values()) != {32} or len(instance_counts) != 6 or set(instance_counts.values()) != {64}:
        raise ValueError("panel must have 32 states/trajectory and two trajectories/instance")
    if dict(Counter(r["phase"] for r in rows)) != quality["phase_counts"]:
        raise ValueError("phase counts disagree with quality report")
    q = np.stack(outcomes)
    # All three repeats stay together; they are not three independent samples.
    y = (q - q[:, BALANCED:BALANCED + 1]).mean(axis=2)
    return rows, np.stack(tokens), np.stack(masks), y, q, quality


def fit_scaler(x, mask, y):
    values = x[mask]
    mean, std = values.mean(0), values.std(0)
    std = np.where(std > 1e-8, std, 1.0)
    alternatives = mask.copy()
    alternatives[:, BALANCED] = False
    scale = max(float(np.sqrt(np.mean(y[alternatives] ** 2))), 1e-10)
    if not np.isfinite(mean).all() or not np.isfinite(std).all() or not np.isfinite(scale):
        raise FloatingPointError("non-finite training-only normalization")
    return {"mean": mean.tolist(), "std": std.tolist(), "target_scale": scale,
            "feature_clip": 10.0}


def transform(x, scaler):
    return np.clip((x - np.array(scaler["mean"])) / np.array(scaler["std"]),
                   -scaler["feature_clip"], scaler["feature_clip"]).astype(np.float32)


def fit_ridge(x, mask, y, alpha=10.0):
    # Separate heads permit different state dependence for each operator.
    # Each head sees all five tokens, exactly as the Transformer does.
    full = x.reshape(len(x), -1).astype(float)
    coefficients = []
    for action in range(4):
        valid = mask[:, action]
        if valid.sum() < 2:
            coefficients.append(None)
            continue
        a, b = full[valid], y[valid, action]
        am, bm = a.mean(0), b.mean()
        ac = a - am
        # Dual form is cheap for <300 training states and many token features.
        gram = np.einsum("ik,jk->ij", ac, ac, optimize=False)
        dual = np.linalg.solve(gram + alpha * np.eye(len(a)), b - bm)
        weight = np.einsum("ij,i->j", ac, dual, optimize=False)
        bias = float(bm - np.sum(am * weight))
        if not np.isfinite(weight).all() or not np.isfinite(bias):
            raise FloatingPointError("non-finite ridge coefficients")
        coefficients.append({"weight": weight.tolist(), "bias": bias})
    return coefficients


def predict_ridge(coefficients, x):
    full = x.reshape(len(x), -1)
    pred = np.zeros((len(x), 5))
    for action, coefficient in enumerate(coefficients):
        pred[:, action] = (-1e9 if coefficient is None else
                           np.einsum("ij,j->i", full, np.array(coefficient["weight"]), optimize=False) + coefficient["bias"])
    return pred


class TinyOperatorTransformer(nn.Module):
    def __init__(self, feature_count, width=32):
        super().__init__()
        self.projection = nn.Linear(feature_count, width)
        self.attention = nn.MultiheadAttention(width, 4, dropout=0.0, batch_first=True)
        self.norm1 = nn.LayerNorm(width)
        self.ff = nn.Sequential(nn.Linear(width, width * 2), nn.GELU(), nn.Linear(width * 2, width))
        self.norm2 = nn.LayerNorm(width)
        self.head = nn.Linear(width, 1)

    def forward(self, tokens, eligible):
        if not bool(eligible.any(dim=1).all()):
            raise ValueError("all-masked attention row")
        hidden = self.projection(tokens)
        attended, _ = self.attention(hidden, hidden, hidden, key_padding_mask=~eligible, need_weights=False)
        hidden = self.norm1(hidden + attended)
        hidden = self.norm2(hidden + self.ff(hidden))
        raw = self.head(hidden).squeeze(-1)
        # Balanced has known relative return zero, not a fitted offset.
        return torch.cat([raw[:, :4], torch.zeros_like(raw[:, 4:])], dim=1)


def masked_loss(prediction, target, eligible):
    mask = eligible.clone()
    mask[:, BALANCED] = False
    if not bool(mask.any()):
        raise ValueError("no legal non-Balanced labels")
    # Index BEFORE subtracting: NaN illegal labels must not contaminate gradients.
    return ((prediction[mask] - target[mask]) ** 2).mean()


def train_transformer(train, validation, seed, epochs=40, patience=8, progress=None):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    x, mask, y = [torch.as_tensor(a) for a in train]
    vx, vm, vy = [torch.as_tensor(a) for a in validation]
    model = TinyOperatorTransformer(x.shape[-1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    history, best, best_loss, best_epoch = [], None, float("inf"), 0
    started = time.monotonic()
    last_log = started
    for epoch in range(epochs + 1):
        model.train()
        if epoch:
            for offset in range(0, len(x), 32):
                # One permutation per epoch, not sampling with replacement.
                if offset == 0:
                    order = rng.permutation(len(x))
                indices = order[offset:offset + 32]
                optimizer.zero_grad(set_to_none=True)
                loss = masked_loss(model(x[indices], mask[indices]), y[indices], mask[indices])
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError("non-finite offline regression loss")
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
                optimizer.step()
        model.eval()
        with torch.no_grad():
            training_loss = float(masked_loss(model(x, mask), y, mask))
            validation_loss = float(masked_loss(model(vx, vm), vy, vm))
        if not np.isfinite([training_loss, validation_loss]).all():
            raise FloatingPointError("non-finite offline validation loss")
        history.append({"epoch": epoch, "train_mse": training_loss, "validation_mse": validation_loss})
        if validation_loss < best_loss - 1e-10:
            best_loss, best_epoch, best = validation_loss, epoch, copy.deepcopy(model.state_dict())
        now = time.monotonic()
        if progress and (epoch % 5 == 0 or now - last_log >= 15 or epoch == epochs):
            progress(f"epoch={epoch}/{epochs} train_mse={training_loss:.6g} valid_mse={validation_loss:.6g} elapsed={now-started:.1f}s")
            last_log = now
        if epoch - best_epoch >= patience:
            break
    model.load_state_dict(best)
    return model, {"seed": seed, "selected_epoch": best_epoch, "history": history,
                   "elapsed_seconds": time.monotonic() - started,
                   "parameter_count": sum(p.numel() for p in model.parameters())}


def predict_transformer(model, x, mask):
    model.eval()
    with torch.no_grad():
        pred = model(torch.as_tensor(x), torch.as_tensor(mask)).numpy().astype(float)
    if not np.isfinite(pred).all():
        raise FloatingPointError("non-finite prediction")
    return pred


def choose(prediction, mask, threshold):
    if not np.isfinite(prediction).all():
        raise ValueError("non-finite prediction")
    if threshold is None:  # explicit always-Balanced option, JSON-safe infinity
        return np.full(len(prediction), BALANCED, dtype=int)
    alternatives = mask.copy()
    alternatives[:, BALANCED] = False
    scores = np.where(alternatives, prediction, -np.inf)
    action = scores.argmax(1)
    return np.where(scores[np.arange(len(scores)), action] > max(float(threshold), 1e-12), action, BALANCED)


def tune_threshold(prediction, mask, actual, target_scale):
    # Predeclared grid, validation INSTANCE only. Ties prefer fewer switches.
    candidates = [None, 0.0, 0.1 * target_scale, 0.25 * target_scale,
                  0.5 * target_scale, target_scale]
    best, best_score = None, (0.0, 0)
    for threshold in candidates:
        selected = choose(prediction, mask, threshold)
        score = (float(actual[np.arange(len(actual)), selected].mean()), -int((selected != BALANCED).sum()))
        if score > best_score:
            best, best_score = threshold, score
    return best


def decisions(rows, selected, y, prediction):
    return [{"instance": r["instance"], "run": r["run"], "state_id": r["state_id"],
             "phase": r["phase"], "selected_operator": OPERATORS[int(a)],
             "mean_heldout_gain": float(y[i, a]), "predicted_gain": float(prediction[i, a])}
            for i, (r, a) in enumerate(zip(rows, selected))]


def gain_metrics(records):
    gain = np.array([r["mean_heldout_gain"] for r in records])
    switched = np.array([r["selected_operator"] != "replace" for r in records])
    return {"state_count": len(records), "mean_net_gain_all_states": float(gain.mean()),
            "switch_fraction": float(switched.mean()),
            "mean_gain_when_switched": float(gain[switched].mean()) if switched.any() else None,
            "positive_gain_total": float(gain[gain > 0].sum()),
            "negative_gain_total": float(gain[gain < 0].sum()),
            "switched_gain_p10": float(np.quantile(gain[switched], 0.1)) if switched.any() else None,
            "selected_operator_counts": dict(Counter(r["selected_operator"] for r in records))}


def cross_repeat_check(rows):
    # This selector sees two counterfactual outcomes: it is NOT deployable.
    from analyze_m_operator_repeat_check import heldout_repeats, metrics
    checked = [dict(r, repeat_folds=heldout_repeats(r)) for r in rows]
    return {"overall": metrics(checked), "per_instance": {
        name: metrics([r for r in checked if r["instance"] == name])
        for name in sorted({r["instance"] for r in rows})},
        "qualification": "cross-repeat diagnostic with two observed branches; not model performance or long-run HV"}
