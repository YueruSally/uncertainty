#!/usr/bin/env python3
"""Train the frozen 25/50/100/200-instance K8 ranking learning curve."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch import nn

from m_operator_offline import digest, read_json, write_json
from m_prescreen_surrogate import (CandidateGainTransformer, fit_scaler,
    gain_targets, pack_rows)
from run_learning_components_deterministic import _normalizer
from train_m_prescreen_ranker_v2 import predict, score


SIZES = (25, 50, 100, 200)
SEEDS = (917001, 917002, 917003)
EPOCHS = 30
BATCH_SIZE = 64
WIDTH = 16
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4


def log(message):
    print(f"[M-K8-SCALE-TRAIN] {message}", flush=True)


def instance_order(names):
    return sorted(names, key=lambda name: hashlib.sha256(
        f"m-k8-scale-order-v1:{name}".encode()).hexdigest())


def pairwise_ranking_loss(predicted, target):
    """Within-state ordering only; exact target ties carry no gradient."""
    difference = predicted[:, :, None]-predicted[:, None, :]
    truth = target[:, :, None]-target[:, None, :]
    candidates = predicted.shape[1]
    upper = torch.triu(torch.ones((candidates, candidates), dtype=torch.bool,
                                  device=predicted.device), diagonal=1)
    mask = upper[None] & (truth.abs() > 1e-6)
    if not bool(mask.any()):
        return predicted.sum()*0.0
    sign = truth[mask].sign()
    weight = truth[mask].abs().clamp(0.1, 5.0)
    return (weight*nn.functional.softplus(-sign*difference[mask])).sum() \
        / weight.sum()


def load_dataset(collection):
    plan = read_json(collection/"PLAN.json")
    result = read_json(collection/"QUICK_RESULT.json")
    if (plan.get("protocol") != "m-prescreen-scale-data-v1"
            or plan.get("train_count") != 200
            or plan.get("validation_count") != 20
            or result.get("state_count") != 6600
            or not (collection/"COLLECTION_COMPLETE").exists()):
        raise ValueError("requires completed 220-instance scale collection")
    rows, sources = [], {}
    for record in plan["instances"]:
        folder = collection/"runs"/record["instance_id"]
        complete = read_json(folder/"RUN_COMPLETE.json")
        feedback_path = folder/"FEEDBACK.json"
        feedback = read_json(feedback_path)
        if (complete.get("feedback_sha256") != digest(feedback_path)
                or complete.get("feedback_state_count") != 30
                or len(feedback["rows"]) != 30):
            raise ValueError(f"scale feedback provenance mismatch: {folder}")
        for row in feedback["rows"]:
            if len(row["labels"]) != 8:
                raise ValueError("scale state is not exact K8")
            rows.append({**row, "instance": record["instance_id"],
                "development_role": record["development_role"],
                "trajectory_seed": record["algorithm_seed"]})
        sources[str(feedback_path.resolve())] = digest(feedback_path)
    return rows, sources, plan


def batches(length, rng):
    order = np.arange(length); rng.shuffle(order)
    return [order[start:start+BATCH_SIZE]
            for start in range(0, length, BATCH_SIZE)]


def train_one(size, seed, rows, scaler, normalizer, target_scale, out):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.set_num_threads(1)
    shape = pack_rows(rows[:1], scaler, normalizer)
    model = CandidateGainTransformer(shape[0].shape[-1],
        shape[1].shape[-1], shape[2].shape[-1], width=WIDTH)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE,
                                  weight_decay=WEIGHT_DECAY)
    rng = np.random.default_rng(seed); history = []
    started = time.monotonic()
    for epoch in range(1, EPOCHS+1):
        model.train(); total = count = 0
        for indices in batches(len(rows), rng):
            part = [rows[int(index)] for index in indices]
            packed = pack_rows(part, scaler, normalizer)
            loss = pairwise_ranking_loss(
                model(*packed[:4]), gain_targets(part, target_scale))
            optimizer.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item()*len(part); count += len(part)
        if epoch in (1, 10, 20, 30):
            history.append({"epoch": epoch, "training_rank_loss": total/count})
            log(f"size={size} seed={seed} epoch={epoch}/{EPOCHS} loss={total/count:.6f}")
    checkpoint = out/f"size_{size}_seed_{seed}.pt"
    torch.save({"protocol": "m-prescreen-scale-ranker-v1",
        "training_instances": size, "seed": seed, "epoch": EPOCHS,
        "width": WIDTH, "model": model.state_dict(),
        "batch_feature_count": shape[0].shape[-1],
        "candidate_feature_count": shape[1].shape[-1],
        "context_feature_count": shape[2].shape[-1]}, checkpoint)
    return model, {"seed": seed, "history": history,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": digest(checkpoint),
        "elapsed_seconds": time.monotonic()-started}


def execute(a):
    collection = a.collection.resolve(); out = a.out.resolve()
    rows, sources, collection_plan = load_dataset(collection)
    train_names = instance_order({row["instance"] for row in rows
                                  if row["development_role"] == "train"})
    validation_names = sorted({row["instance"] for row in rows
                               if row["development_role"] == "validation"})
    if len(train_names) != 200 or len(validation_names) != 20:
        raise ValueError("scale dataset role split changed")
    validation = [row for row in rows if row["instance"] in validation_names]
    normalizer = _normalizer(a.normalization)
    frozen = {"protocol": "m-prescreen-scale-learning-curve-v1",
        "collection": str(collection),
        "collection_plan_sha256": digest(collection/"PLAN.json"),
        "normalization": str(a.normalization.resolve()),
        "normalization_sha256": digest(a.normalization),
        "nested_training_order": train_names,
        "validation_instances": validation_names,
        "sizes": list(SIZES), "training_seeds": list(SEEDS),
        "epochs": EPOCHS, "batch_size": BATCH_SIZE, "width": WIDTH,
        "learning_rate": LEARNING_RATE, "weight_decay": WEIGHT_DECAY,
        "loss": "within-state magnitude-weighted pairwise logistic ranking; exact ties ignored; Balanced output fixed zero",
        "checkpoint_selection": "fixed epoch 30; validation never selects epoch, threshold, architecture or seed",
        "pass": "size-200 mean net gain positive; >=12/20 positive validation instances; positive log2-size slope; size-200 gain > size-25 gain",
        "source_sha256": {**sources,
            str(Path(__file__).resolve()): digest(Path(__file__).resolve()),
            str(Path(__file__).parent/"m_prescreen_surrogate.py"):
                digest(Path(__file__).parent/"m_prescreen_surrogate.py")}}
    out.mkdir(parents=True, exist_ok=True)
    if (out/"PLAN.json").exists() and read_json(out/"PLAN.json") != frozen:
        raise ValueError("frozen scale learning-curve inputs changed")
    write_json(out/"PLAN.json", frozen)
    if (out/"TRAINING_COMPLETE").exists():
        log("already complete; see QUICK_RESULT.json")
        return
    started = time.monotonic(); curve = []; bundles = {}
    for size in SIZES:
        selected_names = set(train_names[:size])
        training = [row for row in rows if row["instance"] in selected_names]
        if len(training) != size*30:
            raise ValueError("nested training-state count mismatch")
        scaler = fit_scaler(training, normalizer)
        alternative = np.asarray([float(label["hv_contribution"])-
            float(row["labels"][0]["hv_contribution"])
            for row in training for label in row["labels"][1:]])
        target_scale = max(float(np.sqrt(np.mean(alternative**2))), 1e-10)
        models, reports = [], []
        for seed in SEEDS:
            model, report = train_one(size, seed, training, scaler,
                normalizer, target_scale, out)
            models.append(model); reports.append(report)
        predictions = np.mean([predict(model, validation, scaler,
            normalizer, target_scale) for model in models], axis=0)
        validation_score = score(validation, predictions)
        positive_instances = sum(row["mean_net_gain"] > 0
                                 for row in validation_score["per_instance"])
        mean_oracle = float(np.mean([row["oracle_gain"]
                                    for row in validation_score["decisions"]]))
        item = {"training_instances": size,
            "training_states": len(training),
            "validation_instances": len(validation_names),
            "validation_states": len(validation),
            "mean_net_gain_all_states": validation_score["mean_net_gain"],
            "switch_fraction": validation_score["switch_fraction"],
            "positive_instances": positive_instances,
            "mean_oracle_gain": mean_oracle,
            "oracle_gain_fraction_captured": (
                validation_score["mean_net_gain"]/mean_oracle
                if mean_oracle > 0 else None),
            "positive_gain_total": validation_score["positive_gain_total"],
            "negative_gain_total": validation_score["negative_gain_total"],
            "gain_p10": validation_score["gain_p10"],
            "target_scale": target_scale,
            "training": reports,
            "per_instance": validation_score["per_instance"]}
        curve.append(item)
        bundles[str(size)] = {"training_instances": sorted(selected_names),
            "scaler": scaler, "target_scale": target_scale,
            "checkpoints": [row["checkpoint"] for row in reports],
            "checkpoint_sha256": {row["checkpoint"]: row["checkpoint_sha256"]
                                  for row in reports}}
        write_json(out/"STATUS.json", {"stage": "training",
            "completed_sizes": len(curve), "total_sizes": len(SIZES),
            "last": {key: item[key] for key in (
                "training_instances", "mean_net_gain_all_states",
                "positive_instances", "switch_fraction")},
            "elapsed_seconds": time.monotonic()-started})
        log(f"size={size} gain={item['mean_net_gain_all_states']:+.8f} positive_instances={positive_instances}/20 switches={item['switch_fraction']:.3f}")
    gains = np.asarray([row["mean_net_gain_all_states"] for row in curve])
    slope = float(np.polyfit(np.log2(np.asarray(SIZES, dtype=float)), gains, 1)[0])
    last = curve[-1]
    passed = (last["mean_net_gain_all_states"] > 0
              and last["positive_instances"] >= 12
              and slope > 0
              and last["mean_net_gain_all_states"] > curve[0]["mean_net_gain_all_states"])
    result = {"completed": True, "new_objective_evaluations": 0,
        "new_searches": 0, "new_ppo_updates": 0,
        "curve": curve, "log2_instance_count_slope": slope,
        "learning_curve_pass": passed,
        "conclusion": ("FREEZE_SIZE_200_FOR_BOUNDED_B400_SEARCH"
                       if passed else "STOP_K8_SCALE_CURVE_FLAT_OR_NONPOSITIVE"),
        "next_action": ("run one frozen size-200 model in paired development search before PPO"
                        if passed else
                        "do not collect more K8 data or tune the ranker")}
    write_json(out/"BUNDLE.json", {**frozen, "models": bundles,
        "selected_size": 200 if passed else None,
        "learning_curve_pass": passed})
    write_json(out/"QUICK_RESULT.json", result)
    write_json(out/"STATUS.json", {"stage": "complete",
        "learning_curve_pass": passed,
        "elapsed_seconds": time.monotonic()-started})
    (out/"TRAINING_COMPLETE").write_text(digest(out/"BUNDLE.json")+"\n")
    log(json.dumps({"curve": [{key: row[key] for key in (
        "training_instances", "mean_net_gain_all_states",
        "positive_instances", "switch_fraction",
        "oracle_gain_fraction_captured")} for row in curve],
        "log2_instance_count_slope": slope,
        "learning_curve_pass": passed,
        "conclusion": result["conclusion"]}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection", type=Path, required=True)
    parser.add_argument("--normalization", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    execute(parser.parse_args())


if __name__ == "__main__":
    main()
