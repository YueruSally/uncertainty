#!/usr/bin/env python3
"""Train Directed-default I-K8 residual models using existing exact labels."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import time

import numpy as np
import torch

from i_k8_residual import ensemble_predictions, reorder_row, score_gate
from i_k8_surrogate import (CandidateGainTransformer, fit_scaler,
    gain_targets, pack_rows, pairwise_ranking_loss)
from m_operator_offline import digest, read_json, write_json
from train_i_k8_transformer_scale_v1 import load_dataset


SEEDS = (926001, 926002, 926003)
EPOCHS = 30
BATCH_SIZE = 64
WIDTH = 16
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
UNCERTAINTY_WEIGHTS = (0.0, 0.5, 1.0, 2.0)
THRESHOLD_SCALE = (0.0, 0.05, 0.10, 0.20)


def log(message):
    print(f"[I-K8-RESIDUAL] {message}", flush=True)


def batches(length, generator):
    order = np.arange(length); generator.shuffle(order)
    return [order[start:start+BATCH_SIZE]
            for start in range(0, length, BATCH_SIZE)]


def train_one(seed, rows, scaler, target_scale, out):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.set_num_threads(1)
    packed = pack_rows(rows[:1], scaler)
    model = CandidateGainTransformer(
        packed[0].shape[-1], packed[1].shape[-1], packed[2].shape[-1],
        width=WIDTH)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    generator = np.random.default_rng(seed); history = []
    started = time.monotonic()
    for epoch in range(1, EPOCHS+1):
        model.train(); total = 0.0; count = 0
        for indices in batches(len(rows), generator):
            part = [rows[int(index)] for index in indices]
            loss = pairwise_ranking_loss(
                model(*pack_rows(part, scaler)),
                gain_targets(part, target_scale))
            optimizer.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); total += float(loss) * len(part); count += len(part)
        if epoch in (1, 10, 20, 30):
            value = total / count
            history.append({"epoch": epoch, "training_rank_loss": value})
            log(f"seed={seed} epoch={epoch}/{EPOCHS} loss={value:.6f}")
    path = out / f"residual_seed_{seed}.pt"
    torch.save({
        "protocol": "i-k8-directed-residual-model-v1",
        "training_instances": 200, "seed": seed, "epoch": EPOCHS,
        "width": WIDTH, "model": model.state_dict(),
        "batch_feature_count": packed[0].shape[-1],
        "candidate_feature_count": packed[1].shape[-1],
        "context_feature_count": packed[2].shape[-1],
    }, path)
    return model, {"seed": seed, "history": history,
        "checkpoint": str(path.resolve()), "checkpoint_sha256": digest(path),
        "elapsed_seconds": time.monotonic()-started}


def execute(a):
    collection = a.collection.resolve(); out = a.out.resolve()
    rows, sources, _ = load_dataset(collection)
    training_original = [row for row in rows if row["development_role"] == "train"]
    validation_original = [row for row in rows if row["development_role"] == "validation"]
    training_names = sorted({row["instance"] for row in training_original})
    validation_names = sorted({row["instance"] for row in validation_original})
    if (len(training_names) != 200 or len(training_original) != 4000
            or len(validation_names) != 20 or len(validation_original) != 400):
        raise ValueError("I residual split must remain 200/20 instances")
    training = [reorder_row(row) for row in training_original]
    scaler = fit_scaler(training)
    residuals = np.asarray([
        float(label["hv_contribution"])
        - float(row["labels"][0]["hv_contribution"])
        for row in training for label in row["labels"][1:]])
    target_scale = max(float(np.sqrt(np.mean(residuals**2))), 1e-10)
    plan = {
        "protocol": "i-k8-directed-residual-training-v1",
        "source_snapshot_tag": "ccp-ic-k8-before-residual-v1",
        "collection": str(collection),
        "collection_plan_sha256": digest(collection / "PLAN.json"),
        "training_instances": training_names,
        "validation_instances": validation_names,
        "training_states": len(training), "validation_states": len(validation_original),
        "directed_original_candidate_index": 1,
        "model_candidate_order": [1, 0, 2, 3, 4, 5, 6, 7],
        "training_seeds": list(SEEDS), "epochs": EPOCHS,
        "batch_size": BATCH_SIZE, "width": WIDTH,
        "learning_rate": LEARNING_RATE, "weight_decay": WEIGHT_DECAY,
        "loss": "within-state magnitude-weighted pairwise ranking relative to Directed",
        "gate_grid": {"uncertainty_weights": list(UNCERTAINTY_WEIGHTS),
                      "threshold_target_scale_fractions": list(THRESHOLD_SCALE)},
        "gate_selection": "maximum validation mean residual, then positive instances, p10, lower switch fraction",
        "new_objective_evaluations": 0, "final_test_opened": False,
        "source_sha256": {**sources,
            str(Path(__file__).resolve()): digest(Path(__file__).resolve()),
            str(Path(__file__).parent / "i_k8_residual.py"):
                digest(Path(__file__).parent / "i_k8_residual.py")},
    }
    out.mkdir(parents=True, exist_ok=True)
    if (out / "PLAN.json").exists() and read_json(out / "PLAN.json") != plan:
        raise ValueError("frozen I residual training plan changed")
    write_json(out / "PLAN.json", plan)
    if (out / "TRAINING_COMPLETE").exists():
        log("already complete; see QUICK_RESULT.json"); return
    started = time.monotonic(); models = []; reports = []
    for seed in SEEDS:
        model, report = train_one(seed, training, scaler, target_scale, out)
        models.append(model); reports.append(report)
    predictions = ensemble_predictions(
        models, validation_original, scaler, target_scale)
    grid = []
    for weight in UNCERTAINTY_WEIGHTS:
        for fraction in THRESHOLD_SCALE:
            threshold = fraction * target_scale
            result = score_gate(
                validation_original, predictions, weight, threshold)
            positive_instances = sum(
                row["mean_net_gain_vs_directed"] > 0
                for row in result["per_instance"])
            grid.append({
                "uncertainty_weight": weight,
                "threshold_scale_fraction": fraction,
                "threshold": threshold,
                "mean_net_gain_vs_directed": result[
                    "mean_net_gain_vs_directed"],
                "positive_instances": positive_instances,
                "switch_fraction": result["switch_fraction"],
                "gain_p10": result["gain_p10"],
                "positive_gain_total": result["positive_gain_total"],
                "negative_gain_total": result["negative_gain_total"],
                "oracle_gain_fraction_captured": result[
                    "oracle_gain_fraction_captured"],
                "per_instance": result["per_instance"],
            })
    selected = max(grid, key=lambda row: (
        row["mean_net_gain_vs_directed"], row["positive_instances"],
        row["gain_p10"], -row["switch_fraction"],
        row["uncertainty_weight"], row["threshold"]))
    checkpoints = [row["checkpoint"] for row in reports]
    hashes = {row["checkpoint"]: row["checkpoint_sha256"] for row in reports}
    bundle = {
        "protocol": "i-k8-directed-residual-v1",
        "source_snapshot_tag": "ccp-ic-k8-before-residual-v1",
        "training_instances": training_names,
        "validation_instances": validation_names,
        "scaler": scaler, "target_scale": target_scale,
        "checkpoints": checkpoints, "checkpoint_sha256": hashes,
        "gate": {key: selected[key] for key in (
            "uncertainty_weight", "threshold_scale_fraction", "threshold")},
        "training_reports": reports,
    }
    write_json(out / "BUNDLE.json", bundle)
    quick = {
        "completed": True, "new_objective_evaluations": 0,
        "new_searches": 0, "new_ppo_updates": 0,
        "training_instances": 200, "training_states": len(training),
        "validation_instances": 20, "validation_states": len(validation_original),
        "target_scale": target_scale, "gate_candidates": grid,
        "selected_gate": selected,
        "offline_residual_positive": bool(
            selected["mean_net_gain_vs_directed"] > 0
            and selected["positive_instances"] >= 12),
        "next_action": "run one frozen bounded development search; no PPO and no final test",
        "elapsed_seconds": time.monotonic()-started,
    }
    write_json(out / "QUICK_RESULT.json", quick)
    write_json(out / "STATUS.json", {"stage": "complete",
        "selected_gate": selected, "elapsed_seconds": time.monotonic()-started})
    (out / "TRAINING_COMPLETE").write_text(digest(out / "BUNDLE.json") + "\n")
    log(json.dumps({"selected_gate": selected,
                    "offline_residual_positive": quick["offline_residual_positive"]},
                   indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    execute(parser.parse_args())


if __name__ == "__main__": main()
