#!/usr/bin/env python3
"""Train the frozen 25/50/100/200-instance I-K8 Transformer curve."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
import torch

from i_k8_surrogate import (CandidateGainTransformer, fit_scaler,
    gain_targets, pack_rows, pairwise_ranking_loss, predict, score)
from m_operator_offline import digest, read_json, write_json


SIZES = (25, 50, 100, 200)
SEEDS = (921001, 921002, 921003)
EPOCHS = 30
BATCH_SIZE = 64
WIDTH = 16
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4


def log(message):
    print(f"[I-K8-TRANSFORMER] {message}", flush=True)


def instance_order(names):
    return sorted(names, key=lambda name: hashlib.sha256(
        f"i-k8-scale-order-v1:{name}".encode()).hexdigest())


def load_dataset(collection):
    plan = read_json(collection / "PLAN.json")
    result = read_json(collection / "QUICK_RESULT.json")
    if (plan.get("protocol") != "i-k8-scale-data-v1"
            or plan.get("train_count") != 200
            or plan.get("validation_count") != 20
            or plan.get("states_per_trajectory") != 20
            or result.get("state_count") != 4400
            or not (collection / "COLLECTION_COMPLETE").exists()):
        raise ValueError("requires completed 220-instance I-K8 collection")
    rows = []
    sources = {}
    for record in plan["instances"]:
        folder = collection / "runs" / record["instance_id"]
        complete = read_json(folder / "RUN_COMPLETE.json")
        feedback_path = folder / "FEEDBACK.json"
        feedback = read_json(feedback_path)
        if (complete.get("feedback_sha256") != digest(feedback_path)
                or complete.get("feedback_state_count") != 20
                or len(feedback.get("rows", ())) != 20):
            raise ValueError(f"I-K8 feedback provenance mismatch: {folder}")
        for row in feedback["rows"]:
            if (len(row.get("labels", ())) != 8
                    or row.get("candidate_roles", [None])[0] != "base"):
                raise ValueError("I-K8 state is not exact K8 with Base index zero")
            rows.append({
                **row,
                "instance": record["instance_id"],
                "development_role": record["development_role"],
                "trajectory_seed": record["algorithm_seed"],
            })
        sources[str(feedback_path.resolve())] = digest(feedback_path)
    return rows, sources, plan


def shuffled_batches(length, generator):
    order = np.arange(length)
    generator.shuffle(order)
    return [order[start:start + BATCH_SIZE]
            for start in range(0, length, BATCH_SIZE)]


def train_one(size, seed, rows, scaler, target_scale, out):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    shape = pack_rows(rows[:1], scaler)
    model = CandidateGainTransformer(
        shape[0].shape[-1], shape[1].shape[-1], shape[2].shape[-1],
        width=WIDTH)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    generator = np.random.default_rng(seed)
    history = []
    started = time.monotonic()
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total = 0.0
        count = 0
        for indices in shuffled_batches(len(rows), generator):
            part = [rows[int(index)] for index in indices]
            packed = pack_rows(part, scaler)
            loss = pairwise_ranking_loss(
                model(*packed), gain_targets(part, target_scale))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item() * len(part)
            count += len(part)
        if epoch in (1, 10, 20, 30):
            history.append({"epoch": epoch,
                            "training_rank_loss": total / count})
            log(f"size={size} seed={seed} epoch={epoch}/{EPOCHS} "
                f"loss={total / count:.6f}")
    checkpoint = out / f"size_{size}_seed_{seed}.pt"
    torch.save({
        "protocol": "i-k8-scale-transformer-v1",
        "training_instances": size,
        "seed": seed,
        "epoch": EPOCHS,
        "width": WIDTH,
        "model": model.state_dict(),
        "batch_feature_count": shape[0].shape[-1],
        "candidate_feature_count": shape[1].shape[-1],
        "context_feature_count": shape[2].shape[-1],
    }, checkpoint)
    return model, {
        "seed": seed,
        "history": history,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": digest(checkpoint),
        "elapsed_seconds": time.monotonic() - started,
    }


def execute(a):
    collection = a.collection.resolve()
    out = a.out.resolve()
    rows, sources, collection_plan = load_dataset(collection)
    train_names = instance_order({
        row["instance"] for row in rows
        if row["development_role"] == "train"
    })
    validation_names = sorted({
        row["instance"] for row in rows
        if row["development_role"] == "validation"
    })
    if len(train_names) != 200 or len(validation_names) != 20:
        raise ValueError("I-K8 dataset role split changed")
    validation = [row for row in rows if row["instance"] in validation_names]
    frozen = {
        "protocol": "i-k8-scale-transformer-curve-v1",
        "collection": str(collection),
        "collection_plan_sha256": digest(collection / "PLAN.json"),
        "nested_training_order": train_names,
        "validation_instances": validation_names,
        "sizes": list(SIZES),
        "training_seeds": list(SEEDS),
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "width": WIDTH,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "loss": "within-state magnitude-weighted pairwise logistic ranking; exact ties ignored; Base output fixed zero",
        "model_selection": "size 200 and fixed epoch 30; validation does not select architecture, threshold, epoch or seed",
        "reference_method": "BASE",
        "learning_curve_role": "diagnostic only; not an automatic gate",
        "final_test_instances_used": False,
        "source_sha256": {
            **sources,
            str(Path(__file__).resolve()): digest(Path(__file__).resolve()),
            str(Path(__file__).parent / "i_k8_surrogate.py"):
                digest(Path(__file__).parent / "i_k8_surrogate.py"),
        },
    }
    out.mkdir(parents=True, exist_ok=True)
    if (out / "PLAN.json").exists() and read_json(out / "PLAN.json") != frozen:
        raise ValueError("frozen I-K8 Transformer inputs changed")
    write_json(out / "PLAN.json", frozen)
    if (out / "TRAINING_COMPLETE").exists():
        log("already complete; see QUICK_RESULT.json")
        return
    started = time.monotonic()
    curve = []
    bundles = {}
    for size in SIZES:
        selected_names = set(train_names[:size])
        training = [row for row in rows if row["instance"] in selected_names]
        if (len(training) != size * 20
                or {row["instance"] for row in training} != selected_names):
            raise ValueError("nested I-K8 training-state count mismatch")
        scaler = fit_scaler(training)
        alternatives = np.asarray([
            float(label["hv_contribution"])
            - float(row["labels"][0]["hv_contribution"])
            for row in training for label in row["labels"][1:]
        ])
        target_scale = max(
            float(np.sqrt(np.mean(alternatives ** 2))), 1e-10)
        models = []
        reports = []
        for seed in SEEDS:
            model, report = train_one(
                size, seed, training, scaler, target_scale, out)
            models.append(model)
            reports.append(report)
        predictions = np.mean([
            predict(model, validation, scaler, target_scale)
            for model in models
        ], axis=0)
        validation_score = score(validation, predictions)
        positive_base = sum(
            row["transformer_mean_net_gain_vs_base"] > 0
            for row in validation_score["per_instance"])
        positive_simple = sum(
            row["transformer_minus_simple_mean"] > 0
            for row in validation_score["per_instance"])
        item = {
            "training_instances": size,
            "training_states": len(training),
            "validation_instances": len(validation_names),
            "validation_states": len(validation),
            "transformer_mean_net_gain_vs_base": validation_score[
                "transformer_mean_net_gain_vs_base"],
            "simple_mean_net_gain_vs_base": validation_score[
                "simple_mean_net_gain_vs_base"],
            "transformer_minus_simple_mean": validation_score[
                "transformer_minus_simple_mean"],
            "positive_instances_vs_base": positive_base,
            "positive_instances_vs_simple": positive_simple,
            "transformer_switch_fraction": validation_score[
                "transformer_switch_fraction"],
            "mean_oracle_gain_vs_base": validation_score[
                "mean_oracle_gain_vs_base"],
            "oracle_gain_fraction_captured": (
                validation_score["transformer_mean_net_gain_vs_base"]
                / validation_score["mean_oracle_gain_vs_base"]
                if validation_score["mean_oracle_gain_vs_base"] > 0 else None),
            "target_scale": target_scale,
            "training": reports,
            "per_instance": validation_score["per_instance"],
        }
        curve.append(item)
        bundles[str(size)] = {
            "training_instances": sorted(selected_names),
            "scaler": scaler,
            "target_scale": target_scale,
            "checkpoints": [row["checkpoint"] for row in reports],
            "checkpoint_sha256": {
                row["checkpoint"]: row["checkpoint_sha256"]
                for row in reports
            },
        }
        write_json(out / "STATUS.json", {
            "stage": "training",
            "completed_sizes": len(curve),
            "total_sizes": len(SIZES),
            "last": {key: item[key] for key in (
                "training_instances", "transformer_mean_net_gain_vs_base",
                "transformer_minus_simple_mean",
                "positive_instances_vs_base")},
            "elapsed_seconds": time.monotonic() - started,
        })
        log(f"size={size} vs_base="
            f"{item['transformer_mean_net_gain_vs_base']:+.8f} "
            f"vs_simple={item['transformer_minus_simple_mean']:+.8f} "
            f"positive={positive_base}/20")
    gains = np.asarray([
        row["transformer_mean_net_gain_vs_base"] for row in curve
    ])
    slope = float(np.polyfit(
        np.log2(np.asarray(SIZES, dtype=float)), gains, 1)[0])
    last = curve[-1]
    offline_pass = bool(
        last["transformer_mean_net_gain_vs_base"] > 0
        and last["transformer_minus_simple_mean"] > 0
        and last["positive_instances_vs_base"] >= 12)
    result = {
        "completed": True,
        "new_objective_evaluations": 0,
        "new_searches": 0,
        "new_ppo_updates": 0,
        "curve": curve,
        "log2_instance_count_slope": slope,
        "learning_curve_is_diagnostic_only": True,
        "selected_training_instances": 200,
        "offline_screen_pass": offline_pass,
        "conclusion": (
            "I_K8_TRANSFORMER_OFFLINE_POSITIVE"
            if offline_pass else
            "I_K8_TRANSFORMER_OFFLINE_INCONCLUSIVE_RUN_PREDECLARED_BOUNDED_SEARCH"),
        "next_action": (
            "run the predeclared paired development search; do not open final test"),
    }
    write_json(out / "BUNDLE.json", {
        **frozen,
        "models": bundles,
        "selected_size": 200,
        "offline_screen_pass": offline_pass,
    })
    write_json(out / "QUICK_RESULT.json", result)
    write_json(out / "STATUS.json", {
        "stage": "complete",
        "offline_screen_pass": offline_pass,
        "elapsed_seconds": time.monotonic() - started,
    })
    (out / "TRAINING_COMPLETE").write_text(
        digest(out / "BUNDLE.json") + "\n")
    log(json.dumps({
        "curve": [{key: row[key] for key in (
            "training_instances", "transformer_mean_net_gain_vs_base",
            "simple_mean_net_gain_vs_base", "transformer_minus_simple_mean",
            "positive_instances_vs_base", "oracle_gain_fraction_captured")}
            for row in curve],
        "log2_instance_count_slope": slope,
        "offline_screen_pass": offline_pass,
        "conclusion": result["conclusion"],
    }, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    execute(parser.parse_args())


if __name__ == "__main__":
    main()
