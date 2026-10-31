#!/usr/bin/env python3
"""Train the real path-v4 M Transformer on audited downstream CCP labels.

Each leave-one-instance-out fold starts from the same M checkpoint.  The actor
ranks the ten audited replacement-path candidates in each saved state, while
the existing critic context predicts whether the state's audited oracle is at
least ``gate_threshold`` percent better than exact Base.  No new objective
evaluation is performed by this script.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
from pathlib import Path

import numpy as np


EPSILON = 1e-12
CANDIDATE_ALIGNED_KEYS = (
    "candidates", "eligible", "action_prior", "candidate_graph_nodes",
    "candidate_graph_tokens",
)


def _read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl(path: Path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _sha256(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _policy_state(run: Path, event: int):
    rows = [
        row for row in _jsonl(run / "mutation_policy_states.jsonl")
        if int(row.get("mixture_event_index", -1)) == int(event)
        and row.get("decision_stage", "location") == "location"
    ]
    if len(rows) != 1:
        raise ValueError(
            f"expected one M policy state at event {event}, found "
            f"{len(rows)}: {run}")
    return rows[0]


def subset_observation(observation, candidate_indices):
    """Keep audited actions while retaining the complete state and graph."""
    indices = [int(value) for value in candidate_indices]
    candidate_count = len(observation["candidates"])
    if not indices or len(set(indices)) != len(indices):
        raise ValueError("candidate indices must be non-empty and unique")
    if min(indices) < 0 or max(indices) >= candidate_count:
        raise ValueError("candidate index outside the saved observation")
    result = copy.deepcopy(observation)
    for key in CANDIDATE_ALIGNED_KEYS:
        if key not in observation:
            continue
        if len(observation[key]) != candidate_count:
            raise ValueError(f"{key} does not align with candidate actions")
        result[key] = [copy.deepcopy(observation[key][index]) for index in indices]
    if not all(result["eligible"]):
        raise ValueError("an audited M candidate is not eligible")
    return result


def _load_states(root: Path):
    audit_path = root / "AUDIT_COMPLETE.json"
    if not audit_path.is_file():
        raise ValueError(f"missing completed audit: {audit_path}")
    audit = _read(audit_path)
    if audit.get("completed") is not True:
        raise ValueError(f"M audit is incomplete: {root}")
    states = []
    for row in audit["states"]:
        instance = str(row["instance"])
        seed = int(row["algorithm_seed"])
        event = int(row["event"])
        actions = sorted(
            row["actions"], key=lambda item: int(item["candidate_index"]))
        candidate_indices = [int(item["candidate_index"]) for item in actions]
        labels = [float(item["final_percent"]) for item in actions]
        probe = root / instance / f"seed_{seed}" / f"event_{event}" / "PROBE"
        observation = _policy_state(probe, event)["observation"]
        states.append({
            "instance": instance,
            "algorithm_seed": seed,
            "event": event,
            "generation": int(row["generation"]),
            "state_key": f"{instance}:{seed}:{event}",
            "candidate_indices": candidate_indices,
            "labels": labels,
            "oracle_percent": float(row["audited_oracle_percent"]),
            "observation": subset_observation(observation, candidate_indices),
        })
    states.sort(key=lambda item: (
        item["instance"], item["algorithm_seed"], item["event"]))
    return states, audit_path


def _batches(rows, batch_size, rng):
    order = list(range(len(rows)))
    rng.shuffle(order)
    for start in range(0, len(order), batch_size):
        yield [rows[index] for index in order[start:start + batch_size]]


def _rank_loss(torch, logits, targets, target_mask, temperature):
    # Pairwise signed loss gives every informative ordering equal standing.
    differences = targets.unsqueeze(2) - targets.unsqueeze(1)
    signs = differences.sign()
    paired_mask = target_mask.unsqueeze(2) & target_mask.unsqueeze(1)
    informative = (differences.abs() > EPSILON) & paired_mask
    score_differences = logits.unsqueeze(2) - logits.unsqueeze(1)
    pair_losses = torch.nn.functional.softplus(
        -score_differences * signs)
    pair_loss = pair_losses[informative].mean() if informative.any() else logits.sum() * 0.0
    # A small listwise term distinguishes the best downstream candidates.
    masked_targets = (targets / temperature).masked_fill(
        ~target_mask, torch.finfo(targets.dtype).min)
    target_probabilities = torch.softmax(masked_targets, dim=1)
    log_probabilities = torch.log_softmax(logits, dim=1).masked_fill(
        ~target_mask, 0.0)
    list_loss = -(target_probabilities * log_probabilities).sum(dim=1).mean()
    return pair_loss + 0.25 * list_loss


def _evaluate(model, rows, device, batch_size, gate_threshold,
              training_objective, score_scale):
    import torch
    from transformer_ppo.observations import pad_observations

    model.eval()
    predictions = []
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            group = rows[start:start + batch_size]
            batch = pad_observations(
                [item["observation"] for item in group], device)
            logits, gate_logits = model(batch)
            scores = logits.cpu().numpy()
            critic_gate_probabilities = torch.sigmoid(
                gate_logits).cpu().numpy()
            for offset, item in enumerate(group):
                labels = np.asarray(item["labels"], dtype=float)
                candidate_scores = scores[offset, :len(labels)]
                if training_objective == "regression_sign":
                    candidate_scores = candidate_scores * score_scale
                    maximum_prediction = float(np.max(candidate_scores))
                    gate_probability = 1.0 / (
                        1.0 + math.exp(-max(-50.0, min(
                            50.0, maximum_prediction / 2.0))))
                else:
                    gate_probability = float(
                        critic_gate_probabilities[offset])
                selected = int(np.argmax(candidate_scores))
                predictions.append({
                    "instance": item["instance"],
                    "algorithm_seed": item["algorithm_seed"],
                    "event": item["event"],
                    "generation": item["generation"],
                    "state_key": item["state_key"],
                    "candidate_indices": item["candidate_indices"],
                    "labels": labels.tolist(),
                    "scores": candidate_scores.astype(float).tolist(),
                    "selected_position": selected,
                    "selected_candidate_index": item["candidate_indices"][selected],
                    "selected_gain_percent": float(labels[selected]),
                    "oracle_percent": float(item["oracle_percent"]),
                    "gate_target": bool(item["oracle_percent"] >= gate_threshold),
                    "gate_probability": gate_probability,
                })
    return predictions


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--folds", nargs="+", required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--rank-temperature", type=float, default=2.0)
    parser.add_argument("--gate-threshold", type=float, default=10.0)
    parser.add_argument("--gate-weight", type=float, default=1.0)
    parser.add_argument(
        "--training-objective",
        choices=("pairwise_gate", "regression_sign"),
        default="pairwise_gate")
    parser.add_argument("--score-scale", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=9971)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    if args.epochs < 1 or args.batch_size < 1 or args.score_scale <= 0:
        parser.error("epochs and batch-size must be positive")
    if not args.checkpoint.is_file():
        parser.error(f"checkpoint not found: {args.checkpoint}")

    import torch
    from transformer_ppo.config import PPOConfig
    from transformer_ppo.observations import pad_observations
    from transformer_ppo.ppo import PPOTrainer

    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    device = torch.device(args.device)
    states, audit_path = _load_states(args.audit_root)
    instances = sorted({item["instance"] for item in states})
    unknown = sorted(set(args.folds) - set(instances))
    if unknown:
        parser.error(f"unknown folds: {unknown}; available={instances}")
    args.out.mkdir(parents=True, exist_ok=True)

    for fold in args.folds:
        fold_out = args.out / f"heldout_{fold}"
        marker = fold_out / "FOLD_COMPLETE.json"
        if marker.is_file():
            print(f"[FOLD] {fold} existing", flush=True)
            continue
        if fold_out.exists() and any(fold_out.iterdir()):
            raise ValueError(
                f"incomplete fold output exists; use a fresh root: {fold_out}")
        fold_out.mkdir(parents=True, exist_ok=True)
        train_rows = [item for item in states if item["instance"] != fold]
        validation_rows = [item for item in states if item["instance"] == fold]
        fold_seed = args.seed + instances.index(fold) * 1009
        random.seed(fold_seed)
        np.random.seed(fold_seed)
        torch.manual_seed(fold_seed)

        trainer = PPOTrainer.load(
            args.checkpoint, PPOConfig, lambda *_: None,
            device=device, training=False)
        if trainer.component != "M":
            raise ValueError("checkpoint is not an M policy")
        if trainer.config.mutation_control != "replace_path":
            raise ValueError("checkpoint does not use replace_path M control")
        model = trainer.model
        if args.training_objective == "regression_sign":
            # Predict downstream gain directly rather than adding the old Base
            # action prior to an otherwise uncalibrated residual score.
            model.actor_policy = "direct"
        model.train()
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.learning_rate,
            weight_decay=args.weight_decay)
        positive = sum(
            item["oracle_percent"] >= args.gate_threshold
            for item in train_rows)
        negative = len(train_rows) - positive
        positive_weight = max(1.0, negative / max(1, positive))
        gate_loss_function = torch.nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(positive_weight, device=device))
        candidate_positive = sum(
            label > EPSILON
            for item in train_rows for label in item["labels"])
        candidate_nonpositive = sum(
            label <= EPSILON
            for item in train_rows for label in item["labels"])
        candidate_positive_weight = max(
            1.0, candidate_nonpositive / max(1, candidate_positive))
        candidate_sign_loss_function = torch.nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(
                candidate_positive_weight, device=device))
        rng = random.Random(fold_seed)
        history = []
        for epoch in range(1, args.epochs + 1):
            epoch_rank = 0.0
            epoch_gate = 0.0
            epoch_regression = 0.0
            epoch_sign = 0.0
            steps = 0
            for group in _batches(train_rows, args.batch_size, rng):
                batch = pad_observations(
                    [item["observation"] for item in group], device)
                maximum_candidates = max(len(item["labels"]) for item in group)
                labels = torch.zeros(
                    len(group), maximum_candidates,
                    dtype=torch.float32, device=device)
                label_mask = torch.zeros(
                    len(group), maximum_candidates,
                    dtype=torch.bool, device=device)
                for row_index, item in enumerate(group):
                    count = len(item["labels"])
                    labels[row_index, :count] = torch.as_tensor(
                        item["labels"], dtype=torch.float32, device=device)
                    label_mask[row_index, :count] = True
                gate_targets = torch.as_tensor([
                    item["oracle_percent"] >= args.gate_threshold
                    for item in group
                ], dtype=torch.float32, device=device)
                logits, gate_logits = model(batch)
                logits = logits[:, :labels.shape[1]]
                rank_loss = _rank_loss(
                    torch, logits, labels, label_mask,
                    args.rank_temperature)
                if args.training_objective == "regression_sign":
                    scaled_targets = torch.clamp(
                        labels / args.score_scale, min=-3.0, max=3.0)
                    regression_elements = torch.nn.functional.smooth_l1_loss(
                        logits[label_mask], scaled_targets[label_mask],
                        reduction="none")
                    regression_weights = 1.0 + torch.clamp(
                        labels[label_mask].abs() / 5.0, max=4.0)
                    regression_loss = (
                        (regression_elements * regression_weights).sum()
                        / regression_weights.sum().clamp_min(1.0))
                    sign_loss = candidate_sign_loss_function(
                        logits[label_mask],
                        (labels[label_mask] > EPSILON).to(logits.dtype))
                    gate_loss = logits.sum() * 0.0
                    loss = rank_loss + regression_loss + 0.5 * sign_loss
                else:
                    regression_loss = logits.sum() * 0.0
                    sign_loss = logits.sum() * 0.0
                    gate_loss = gate_loss_function(gate_logits, gate_targets)
                    loss = rank_loss + args.gate_weight * gate_loss
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                epoch_rank += float(rank_loss.detach().cpu())
                epoch_gate += float(gate_loss.detach().cpu())
                epoch_regression += float(regression_loss.detach().cpu())
                epoch_sign += float(sign_loss.detach().cpu())
                steps += 1
            history.append({
                "epoch": epoch,
                "mean_rank_loss": epoch_rank / max(1, steps),
                "mean_gate_loss": epoch_gate / max(1, steps),
                "mean_regression_loss": epoch_regression / max(1, steps),
                "mean_sign_loss": epoch_sign / max(1, steps),
            })
            print(
                f"[FOLD {fold}] epoch={epoch}/{args.epochs} "
                f"rank={history[-1]['mean_rank_loss']:.6f} "
                f"gate={history[-1]['mean_gate_loss']:.6f} "
                f"reg={history[-1]['mean_regression_loss']:.6f} "
                f"sign={history[-1]['mean_sign_loss']:.6f}", flush=True)

        train_predictions = _evaluate(
            model, train_rows, device, args.batch_size, args.gate_threshold,
            args.training_objective, args.score_scale)
        validation_predictions = _evaluate(
            model, validation_rows, device, args.batch_size,
            args.gate_threshold, args.training_objective, args.score_scale)
        train_gate = np.asarray([
            item["gate_probability"] for item in train_predictions], dtype=float)
        deployment_threshold = float(np.quantile(train_gate, 0.80))
        for item in validation_predictions:
            item["gate_selected"] = bool(
                item["gate_probability"] >= deployment_threshold)
            item["deployed_gain_percent"] = (
                item["selected_gain_percent"] if item["gate_selected"] else 0.0)

        artifact = {
            "schema_version": 1,
            "kind": "m_downstream_transformer",
            "source_checkpoint_sha256": _sha256(args.checkpoint),
            "audit_sha256": _sha256(audit_path),
            "heldout_instance": fold,
            "instances": instances,
            "gate_threshold_percent": args.gate_threshold,
            "training_objective": args.training_objective,
            "score_scale": args.score_scale,
            "deployed_actor_policy": model.actor_policy,
            "deployment_gate_probability_threshold": deployment_threshold,
            "model_spec": trainer.model_spec,
            "ppo_config": trainer.config.__dict__,
            "model": model.state_dict(),
        }
        torch.save(artifact, fold_out / "model.pt")
        report = {
            "completed": True,
            "heldout_instance": fold,
            "train_state_count": len(train_rows),
            "validation_state_count": len(validation_rows),
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "training_objective": args.training_objective,
            "score_scale": args.score_scale,
            "gate_threshold_percent": args.gate_threshold,
            "deployment_gate_probability_threshold": deployment_threshold,
            "history": history,
            "validation": validation_predictions,
        }
        (fold_out / "report.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n",
            encoding="utf-8")
        marker.write_text(
            json.dumps({
                "completed": True,
                "heldout_instance": fold,
                "model_sha256": _sha256(fold_out / "model.pt"),
                "report_sha256": _sha256(fold_out / "report.json"),
            }, indent=2) + "\n", encoding="utf-8")
        print(f"[FOLD] {fold} complete", flush=True)


if __name__ == "__main__":
    main()
