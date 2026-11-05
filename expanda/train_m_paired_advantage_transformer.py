#!/usr/bin/env python3
"""Pretrain M to choose the better of its action and the paired heuristic."""
from __future__ import annotations

import argparse
from dataclasses import replace
import gc
import json
import math
from pathlib import Path
import random

def _jsonl(path):
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _instance_key(value):
    return str(value).removeprefix("train-").replace("-", "_")


def _state_key(row):
    return f"{row['instance']}:g{row['generation']}:e{row['evaluation']}"


def load_dataset(root, tolerance=1e-12):
    """Join V4 policy states to same-state Transformer/heuristic outcomes."""
    rows = []
    episode_root = Path(root) / "episodes"
    for run in sorted(episode_root.glob("train-S0-*")):
        state_path = run / "mutation_policy_states.jsonl"
        outcome_path = run / "mutation_outcomes.jsonl"
        if not state_path.is_file() or not outcome_path.is_file():
            continue
        states = {
            int(row["mixture_event_index"]): row
            for row in _jsonl(state_path)
            if row.get("operator") == "replace"
        }
        for outcome in _jsonl(outcome_path):
            paired = outcome.get("paired_heuristic_reward")
            if not isinstance(paired, dict):
                continue
            state = states.get(int(outcome["mixture_event_index"]))
            if state is None:
                continue
            selected = int(paired["selected_candidate_index"])
            heuristic = int(paired["heuristic_candidate_index"])
            eligible = state["observation"].get("eligible", [])
            if not (0 <= selected < len(eligible)
                    and 0 <= heuristic < len(eligible)
                    and eligible[selected] and eligible[heuristic]):
                raise ValueError(
                    f"paired action is not legal for event "
                    f"{outcome['mixture_event_index']} under {run}")
            advantage = float(paired["selected_minus_heuristic_hv"])
            override = selected != heuristic and advantage > tolerance
            target = selected if override else heuristic
            alternative = heuristic if override else selected
            rows.append({
                "instance": _instance_key(state["instance_id"]),
                "generation": int(state["generation"]),
                "evaluation": int(state["evaluation"]),
                "event": int(state["mixture_event_index"]),
                "observation": state["observation"],
                "target_action": target,
                "alternative_action": alternative,
                "selected_action": selected,
                "heuristic_action": heuristic,
                "selected_minus_heuristic_hv": advantage,
                "override_target": override,
                "pairwise_target": target != alternative,
                "label_source": paired.get("label_source"),
                "source": str(run),
            })
    if not rows:
        raise ValueError(f"no paired M reward states under {root}")
    return rows


def _groups(rows, batch_size, rng=None):
    order = list(range(len(rows)))
    if rng is not None:
        rng.shuffle(order)
    for start in range(0, len(order), batch_size):
        yield [rows[index] for index in order[start:start + batch_size]]


def _preference_loss(torch, logits, rows, positive_weight, pairwise_weight,
                     margin):
    """Weighted winner CE plus an explicit winner-over-loser margin."""
    losses = []
    for row_index, item in enumerate(rows):
        target = torch.as_tensor(
            [item["target_action"]], dtype=torch.long, device=logits.device)
        hard = torch.nn.functional.cross_entropy(
            logits[row_index:row_index + 1], target)
        weight = positive_weight if item["override_target"] else 1.0
        loss = weight * hard
        if item["pairwise_target"]:
            winner = logits[row_index, item["target_action"]]
            loser = logits[row_index, item["alternative_action"]]
            loss = loss + pairwise_weight * torch.nn.functional.softplus(
                margin - (winner - loser))
        losses.append(loss)
    return torch.stack(losses).sum() / sum(
        positive_weight if item["override_target"] else 1.0
        for item in rows)


def _metrics(torch, model, rows, device, batch_size, positive_weight,
             pairwise_weight, margin):
    from transformer_ppo.observations import pad_observations

    model.eval()
    totals = {
        "states": len(rows),
        "override_states": 0,
        "fallback_states": 0,
        "pairwise_states": 0,
        "correct": 0,
        "override_correct": 0,
        "fallback_correct": 0,
        "pairwise_correct": 0,
    }
    weighted_loss_sum = 0.0
    weight_sum = 0.0
    with torch.no_grad():
        for group in _groups(rows, batch_size):
            batch = pad_observations(
                [item["observation"] for item in group], device)
            logits, _ = model(batch)
            loss = _preference_loss(
                torch, logits, group, positive_weight, pairwise_weight,
                margin)
            group_weight = sum(
                positive_weight if item["override_target"] else 1.0
                for item in group)
            weighted_loss_sum += float(loss.cpu()) * group_weight
            weight_sum += group_weight
            predictions = logits.argmax(dim=1).cpu().tolist()
            for row_index, (item, prediction) in enumerate(
                    zip(group, predictions)):
                correct = int(prediction == item["target_action"])
                totals["correct"] += correct
                if item["override_target"]:
                    totals["override_states"] += 1
                    totals["override_correct"] += correct
                else:
                    totals["fallback_states"] += 1
                    totals["fallback_correct"] += correct
                if item["pairwise_target"]:
                    totals["pairwise_states"] += 1
                    totals["pairwise_correct"] += int(
                        (logits[row_index, item["target_action"]]
                         > logits[row_index, item["alternative_action"]])
                        .item())
    return {
        **{key: int(value) for key, value in totals.items()},
        "top_1_accuracy": totals["correct"] / max(1, totals["states"]),
        "override_recall": (
            totals["override_correct"] / max(1, totals["override_states"])),
        "fallback_accuracy": (
            totals["fallback_correct"] / max(1, totals["fallback_states"])),
        "pairwise_accuracy": (
            totals["pairwise_correct"] / max(1, totals["pairwise_states"])),
        "mean_preference_loss": weighted_loss_sum / max(1.0, weight_sum),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--validation-instances", nargs="+", required=True)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--early-stopping-patience", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--positive-override-weight", type=float, default=4.0)
    parser.add_argument("--pairwise-weight", type=float, default=1.0)
    parser.add_argument("--pairwise-margin", type=float, default=0.05)
    parser.add_argument("--residual-logit-limit", type=float, default=2.0)
    parser.add_argument(
        "--minimum-validation-override-recall", type=float, default=0.10)
    parser.add_argument(
        "--minimum-validation-fallback-accuracy", type=float, default=0.80)
    parser.add_argument("--seed", type=int, default=66101)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    if not args.checkpoint.is_file():
        parser.error(f"missing checkpoint: {args.checkpoint}")
    if (args.epochs < 1 or args.batch_size < 1
            or args.early_stopping_patience < 1
            or args.learning_rate <= 0.0 or args.weight_decay < 0.0
            or args.positive_override_weight < 1.0
            or args.pairwise_weight < 0.0 or args.pairwise_margin < 0.0
            or args.residual_logit_limit <= 0.0
            or not 0.0 <= args.minimum_validation_override_recall <= 1.0
            or not 0.0 <= args.minimum_validation_fallback_accuracy <= 1.0):
        parser.error("invalid paired-advantage training hyperparameters")

    import torch
    import numpy as np
    from transformer_ppo.config import PPOConfig
    from transformer_ppo.observations import pad_observations
    from transformer_ppo.ppo import PPOTrainer

    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    rows = load_dataset(args.data_root)
    validation_names = {
        _instance_key(value) for value in args.validation_instances}
    training = [
        row for row in rows if row["instance"] not in validation_names]
    validation = [
        row for row in rows if row["instance"] in validation_names]
    if not training or not validation:
        parser.error(
            "paired dataset must contain training and validation instances; "
            f"available={sorted({row['instance'] for row in rows})}, "
            f"requested_validation={sorted(validation_names)}")
    if not any(row["override_target"] for row in training):
        parser.error("training split contains no Transformer-better overrides")
    if not any(row["override_target"] for row in validation):
        parser.error("validation split contains no Transformer-better overrides")
    print(
        "[M-ADV-DATASET] "
        f"states={len(rows)} training={len(training)} "
        f"validation={len(validation)} "
        f"train_overrides={sum(row['override_target'] for row in training)} "
        f"valid_overrides={sum(row['override_target'] for row in validation)} "
        f"validation_instances={sorted(validation_names)}",
        flush=True)
    dataset_summary = {
        "states": len(rows),
        "training_states": len(training),
        "validation_states": len(validation),
        "training_override_states": sum(
            row["override_target"] for row in training),
        "validation_override_states": sum(
            row["override_target"] for row in validation),
    }
    del rows
    gc.collect()

    trainer = PPOTrainer.load(
        args.checkpoint, PPOConfig, lambda *_: None,
        device=device, training=False)
    if (trainer.component != "M"
            or trainer.architecture != "transformer"
            or trainer.config.mutation_control != "replace_path"
            or trainer.config.candidate_representation != "path_v5"
            or trainer.config.actor_policy != "base_residual"
            or trainer.config.mutation_action_prior != "joint_balanced"):
        parser.error(
            "checkpoint must be a joint-balanced base-residual path_v5 "
            "M replace_path Transformer")
    trainer.config = replace(
        trainer.config, residual_logit_limit=args.residual_logit_limit)
    trainer.model.residual_logit_limit = args.residual_logit_limit
    model = trainer.model
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate,
        weight_decay=args.weight_decay)
    metric_args = (
        device, args.batch_size, args.positive_override_weight,
        args.pairwise_weight, args.pairwise_margin)
    baseline_training = _metrics(torch, model, training, *metric_args)
    baseline_validation = _metrics(torch, model, validation, *metric_args)
    print(
        "[M-ADV-TRAIN] epoch=0 "
        f"train_top1={baseline_training['top_1_accuracy']:.4f} "
        f"valid_top1={baseline_validation['top_1_accuracy']:.4f} "
        f"valid_override={baseline_validation['override_recall']:.4f} "
        f"valid_pairwise={baseline_validation['pairwise_accuracy']:.4f} "
        f"valid_loss={baseline_validation['mean_preference_loss']:.6f}",
        flush=True)

    best_loss = math.inf
    best_epoch = 0
    best_state = {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }
    history = []
    stale = 0
    rng = random.Random(args.seed)
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        steps = 0
        for group in _groups(training, args.batch_size, rng):
            batch = pad_observations(
                [item["observation"] for item in group], device)
            logits, _ = model(batch)
            loss = _preference_loss(
                torch, logits, group, args.positive_override_weight,
                args.pairwise_weight, args.pairwise_margin)
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"non-finite preference loss: "
                    f"{[_state_key(item) for item in group]}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), 1.0)
            if not torch.isfinite(gradient_norm):
                raise FloatingPointError(
                    f"non-finite preference gradient: "
                    f"{[_state_key(item) for item in group]}")
            optimizer.step()
            loss_sum += float(loss.detach().cpu())
            steps += 1
        train_metrics = _metrics(torch, model, training, *metric_args)
        validation_metrics = _metrics(
            torch, model, validation, *metric_args)
        record = {
            "epoch": epoch,
            "mean_training_batch_loss": loss_sum / max(1, steps),
            "training": train_metrics,
            "validation": validation_metrics,
        }
        history.append(record)
        print(
            f"[M-ADV-TRAIN] epoch={epoch}/{args.epochs} "
            f"loss={record['mean_training_batch_loss']:.6f} "
            f"train_top1={train_metrics['top_1_accuracy']:.4f} "
            f"valid_top1={validation_metrics['top_1_accuracy']:.4f} "
            f"valid_override={validation_metrics['override_recall']:.4f} "
            f"valid_pairwise={validation_metrics['pairwise_accuracy']:.4f} "
            f"valid_loss={validation_metrics['mean_preference_loss']:.6f} "
            "admissible="
            f"{int(validation_metrics['override_recall'] >= args.minimum_validation_override_recall and validation_metrics['fallback_accuracy'] >= args.minimum_validation_fallback_accuracy)}",
            flush=True)
        validation_loss = validation_metrics["mean_preference_loss"]
        admissible = (
            validation_metrics["override_recall"]
            >= args.minimum_validation_override_recall
            and validation_metrics["fallback_accuracy"]
            >= args.minimum_validation_fallback_accuracy)
        if admissible and validation_loss < best_loss - 1e-9:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
        if stale >= args.early_stopping_patience:
            print(
                f"[M-ADV-TRAIN] early-stop epoch={epoch} "
                f"best_epoch={best_epoch} "
                f"best_loss={best_loss if math.isfinite(best_loss) else None}",
                flush=True)
            break

    model.load_state_dict(best_state)
    final_training = _metrics(torch, model, training, *metric_args)
    final_validation = _metrics(torch, model, validation, *metric_args)
    if not math.isfinite(final_validation["mean_preference_loss"]):
        raise FloatingPointError("non-finite final validation metric")
    args.out.mkdir(parents=True, exist_ok=True)
    checkpoint = args.out / "final_policy.pt"
    trainer.save(checkpoint, metadata={
        "kind": "m_paired_advantage_pretraining",
        "source": str(args.data_root),
        "validation_instances": sorted(validation_names),
        "positive_override_weight": args.positive_override_weight,
        "pairwise_weight": args.pairwise_weight,
        "pairwise_margin": args.pairwise_margin,
        "residual_logit_limit": args.residual_logit_limit,
        "minimum_validation_override_recall":
            args.minimum_validation_override_recall,
        "minimum_validation_fallback_accuracy":
            args.minimum_validation_fallback_accuracy,
        "seed": args.seed,
    }, overwrite=True)
    report = {
        "completed": True,
        "definition": (
            "path-v5 residual M Transformer pretrained to select the "
            "same-state HV winner between its sampled action and the exact "
            "joint-balanced heuristic action"),
        **dataset_summary,
        "training_instances": sorted({row["instance"] for row in training}),
        "validation_instances": sorted({
            row["instance"] for row in validation}),
        "baseline": {
            "training": baseline_training,
            "validation": baseline_validation,
        },
        "training": final_training,
        "validation": final_validation,
        "selected_epoch": best_epoch,
        "requested_epochs": args.epochs,
        "completed_epochs": len(history),
        "positive_override_weight": args.positive_override_weight,
        "pairwise_weight": args.pairwise_weight,
        "pairwise_margin": args.pairwise_margin,
        "residual_logit_limit": args.residual_logit_limit,
        "minimum_validation_override_recall":
            args.minimum_validation_override_recall,
        "minimum_validation_fallback_accuracy":
            args.minimum_validation_fallback_accuracy,
        "checkpoint": str(checkpoint),
        "history": history,
    }
    (args.out / "TRAINING_COMPLETE.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n",
        encoding="utf-8")
    print(json.dumps(report, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
