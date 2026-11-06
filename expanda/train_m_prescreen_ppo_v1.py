#!/usr/bin/env python3
"""Bounded contextual-bandit PPO updates for the frozen K8 Transformer."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch import nn

from m_operator_offline import digest, read_json, write_json
from m_prescreen_scale_policy import FrozenScaleRanker
from m_prescreen_surrogate import gain_targets, pack_rows
from run_learning_components_deterministic import _normalizer
from train_m_prescreen_scale_curve_v1 import load_dataset


UPDATES = 8
CHECKPOINT_UPDATES = (0, 2, 4, 8)
STATES_PER_UPDATE = 1024
OPTIMIZATION_EPOCHS = 4
MINIBATCH_SIZE = 128
POLICY_TEMPERATURE = 1.0
CLIP_RATIO = .2
ACTOR_LEARNING_RATE = 1e-5
CRITIC_LEARNING_RATE = 1e-4
ENTROPY_COEFFICIENT = .002
SEED = 920001


def log(message):
    print(f"[M-K8-PPO-TRAIN] {message}", flush=True)


def ensemble_logits(models, packed):
    return torch.stack([model(*packed[:4]) for model in models]).mean(0) \
        / POLICY_TEMPERATURE


def save_checkpoint_bundle(out, update, models, critic, original_bundle,
                           model_spec, training_names, reports):
    root = out/"checkpoints"/f"update_{update:03d}"
    root.mkdir(parents=True, exist_ok=True)
    checkpoints, hashes = [], {}
    for index, model in enumerate(models):
        path = root/f"model_{index+1}.pt"
        torch.save({"protocol": "m-prescreen-scale-ppo-v1",
            "training_instances": 200, "seed": SEED, "ppo_update": update,
            "epoch": original_bundle["epochs"], "width": 16,
            "model": model.state_dict(),
            "batch_feature_count": model_spec["batch_feature_count"],
            "candidate_feature_count": model_spec["candidate_feature_count"],
            "context_feature_count": model_spec["context_feature_count"]}, path)
        checkpoints.append(str(path.resolve())); hashes[str(path.resolve())] = digest(path)
    critic_path = root/"critic.pt"
    torch.save({"protocol": "m-prescreen-scale-ppo-critic-v1",
        "ppo_update": update, "model": critic.state_dict()}, critic_path)
    bundle = {"protocol": "m-prescreen-scale-ppo-bundle-v1",
        "source_bundle": str(Path(original_bundle["collection"]).resolve()),
        "ppo_update": update, "training_instances": training_names,
        "training_seed": SEED, "gamma": 0.0,
        "reward": "selected candidate actual one-step HV contribution minus Balanced contribution",
        "models": {"200": {"training_instances": training_names,
            "scaler": model_spec["scaler"],
            "target_scale": model_spec["target_scale"],
            "checkpoints": checkpoints, "checkpoint_sha256": hashes}},
        "critic": str(critic_path.resolve()),
        "critic_sha256": digest(critic_path),
        "training_reports_through_update": reports}
    bundle_path = root/"BUNDLE.json"
    write_json(bundle_path, bundle)
    return {"update": update, "bundle": str(bundle_path.resolve()),
        "bundle_sha256": digest(bundle_path), "checkpoints": checkpoints,
        "checkpoint_sha256": hashes}


def ppo_update(models, critic, actor_optimizer, critic_optimizer, rows,
               scaler, normalizer, target_scale, rng):
    count = min(STATES_PER_UPDATE, len(rows))
    selected = rng.choice(len(rows), size=count, replace=False)
    part = [rows[int(index)] for index in selected]
    packed = pack_rows(part, scaler, normalizer)
    rewards_all = gain_targets(part, target_scale).clamp(-5.0, 5.0)
    with torch.no_grad():
        old_logits = ensemble_logits(models, packed)
        distribution = torch.distributions.Categorical(logits=old_logits)
        actions = distribution.sample()
        old_log_probability = distribution.log_prob(actions)
        rewards = rewards_all.gather(1, actions[:, None]).squeeze(1)
        old_values = critic(packed[2]).squeeze(1)
        advantage = rewards-old_values
        advantage = (advantage-advantage.mean()) / advantage.std().clamp_min(1e-6)
    order = np.arange(count); approximate_kl = entropy_total = 0.0
    actor_steps = critic_steps = 0
    for _ in range(OPTIMIZATION_EPOCHS):
        rng.shuffle(order)
        for start in range(0, count, MINIBATCH_SIZE):
            indices = torch.as_tensor(order[start:start+MINIBATCH_SIZE])
            mini = tuple(value[indices] for value in packed)
            logits = ensemble_logits(models, mini)
            distribution = torch.distributions.Categorical(logits=logits)
            new_log_probability = distribution.log_prob(actions[indices])
            ratio = torch.exp(new_log_probability-old_log_probability[indices])
            unclipped = ratio*advantage[indices]
            clipped = ratio.clamp(1-CLIP_RATIO, 1+CLIP_RATIO)*advantage[indices]
            entropy = distribution.entropy().mean()
            actor_loss = -torch.minimum(unclipped, clipped).mean() \
                - ENTROPY_COEFFICIENT*entropy
            actor_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [parameter for model in models for parameter in model.parameters()], 1.0)
            actor_optimizer.step(); actor_steps += 1

            value = critic(mini[2]).squeeze(1)
            critic_loss = nn.functional.mse_loss(value, rewards[indices])
            critic_optimizer.zero_grad(set_to_none=True)
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
            critic_optimizer.step(); critic_steps += 1
            with torch.no_grad():
                approximate_kl += float((old_log_probability[indices]-
                    new_log_probability).mean().abs())
                entropy_total += float(entropy)
    return {"states": count, "actions": dict(Counter(
        str(int(action)) for action in actions)),
        "mean_reward": float(rewards.mean()),
        "positive_rewards": int((rewards > 1e-12).sum()),
        "zero_rewards": int((rewards.abs() <= 1e-12).sum()),
        "negative_rewards": int((rewards < -1e-12).sum()),
        "actor_optimizer_steps": actor_steps,
        "critic_optimizer_steps": critic_steps,
        "approximate_kl": approximate_kl/max(actor_steps, 1),
        "mean_entropy": entropy_total/max(actor_steps, 1)}


def execute(a):
    collection = a.collection.resolve(); curve_root = a.curve_root.resolve()
    out = a.out.resolve(); out.mkdir(parents=True, exist_ok=True)
    original_bundle = read_json(curve_root/"BUNDLE.json")
    curve_result = read_json(curve_root/"QUICK_RESULT.json")
    if (original_bundle.get("protocol") != "m-prescreen-scale-learning-curve-v1"
            or curve_result.get("curve", [{}])[-1].get("training_instances") != 200
            or not (curve_root/"TRAINING_COMPLETE").exists()):
        raise ValueError("requires completed frozen size-200 ranker")
    rows, sources, _ = load_dataset(collection)
    training = [row for row in rows if row["development_role"] == "train"]
    validation = [row for row in rows if row["development_role"] == "validation"]
    training_names = sorted({row["instance"] for row in training})
    validation_names = sorted({row["instance"] for row in validation})
    if len(training_names) != 200 or len(validation_names) != 20:
        raise ValueError("PPO split must remain 200 training / 20 validation")
    model_spec = original_bundle["models"]["200"]
    plan = {"protocol": "m-prescreen-scale-ppo-training-v1",
        "collection": str(collection), "curve_root": str(curve_root),
        "normalization": str(a.normalization.resolve()),
        "normalization_sha256": digest(a.normalization),
        "training_instances": training_names,
        "validation_instances_reserved_for_search_selection": validation_names,
        "training_states": len(training), "validation_states": len(validation),
        "updates": UPDATES, "checkpoint_updates": list(CHECKPOINT_UPDATES),
        "states_per_update": STATES_PER_UPDATE,
        "optimization_epochs": OPTIMIZATION_EPOCHS,
        "minibatch_size": MINIBATCH_SIZE,
        "policy_temperature": POLICY_TEMPERATURE,
        "clip_ratio": CLIP_RATIO,
        "actor_learning_rate": ACTOR_LEARNING_RATE,
        "critic_learning_rate": CRITIC_LEARNING_RATE,
        "entropy_coefficient": ENTROPY_COEFFICIENT,
        "gamma": 0.0, "seed": SEED,
        "reward": "stored exact CCP100 selected-minus-Balanced one-step HV contribution",
        "gradient_data": "training instances only; validation receives no gradient",
        "source_sha256": {**sources,
            str((curve_root/"BUNDLE.json").resolve()): digest(curve_root/"BUNDLE.json"),
            str(Path(__file__).resolve()): digest(Path(__file__).resolve())}}
    if (out/"PLAN.json").exists() and read_json(out/"PLAN.json") != plan:
        raise ValueError("frozen PPO training plan changed")
    write_json(out/"PLAN.json", plan)
    if (out/"TRAINING_COMPLETE").exists():
        log("already complete; see TRAINING_PROGRESS.json"); return
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    torch.set_num_threads(1); rng = np.random.default_rng(SEED)
    ranker = FrozenScaleRanker(original_bundle, digest)
    models = ranker.models
    normalizer = _normalizer(a.normalization)
    packed_one = pack_rows(training[:1], model_spec["scaler"], normalizer)
    context_count = packed_one[2].shape[-1]
    critic = nn.Sequential(nn.Linear(context_count, 32), nn.Tanh(),
                           nn.Linear(32, 1))
    actor_optimizer = torch.optim.AdamW(
        [parameter for model in models for parameter in model.parameters()],
        lr=ACTOR_LEARNING_RATE, weight_decay=1e-4)
    critic_optimizer = torch.optim.AdamW(
        critic.parameters(), lr=CRITIC_LEARNING_RATE, weight_decay=1e-4)
    shape = {"batch_feature_count": packed_one[0].shape[-1],
        "candidate_feature_count": packed_one[1].shape[-1],
        "context_feature_count": packed_one[2].shape[-1],
        "scaler": model_spec["scaler"],
        "target_scale": model_spec["target_scale"]}
    started = time.monotonic(); reports = []; checkpoints = []
    checkpoints.append(save_checkpoint_bundle(out, 0, models, critic,
        original_bundle, shape, training_names, reports))
    for update in range(1, UPDATES+1):
        report = {"ppo_update": update, **ppo_update(models, critic,
            actor_optimizer, critic_optimizer, training, model_spec["scaler"],
            normalizer, model_spec["target_scale"], rng)}
        reports.append(report)
        log(f"update={update}/{UPDATES} reward={report['mean_reward']:+.6f} KL={report['approximate_kl']:.6f}")
        if update in CHECKPOINT_UPDATES:
            checkpoints.append(save_checkpoint_bundle(out, update, models,
                critic, original_bundle, shape, training_names, reports))
        write_json(out/"STATUS.json", {"stage": "training",
            "completed_updates": update, "total_updates": UPDATES,
            "elapsed_seconds": time.monotonic()-started})
    result = {"completed": True, "ppo_updates": UPDATES,
        "checkpoints": checkpoints, "training": reports,
        "training_instances": 200, "validation_instances_used_for_gradient": 0,
        "new_objective_evaluations": 0,
        "elapsed_seconds": time.monotonic()-started,
        "next_action": "select one checkpoint by full B400 search on 20 validation instances"}
    write_json(out/"TRAINING_PROGRESS.json", result)
    (out/"TRAINING_COMPLETE").write_text(digest(out/"TRAINING_PROGRESS.json")+"\n")
    write_json(out/"STATUS.json", {"stage": "complete",
        "completed_updates": UPDATES, "total_updates": UPDATES})
    log(json.dumps({key: result[key] for key in (
        "ppo_updates", "training_instances",
        "validation_instances_used_for_gradient", "elapsed_seconds")}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection", type=Path, required=True)
    parser.add_argument("--curve-root", type=Path, required=True)
    parser.add_argument("--normalization", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    execute(parser.parse_args())


if __name__ == "__main__":
    main()
