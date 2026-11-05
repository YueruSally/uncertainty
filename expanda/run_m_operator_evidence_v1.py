#!/usr/bin/env python3
"""Bounded noise/structure audit and 2/3/4-instance development learning curves.

Reuses the completed CCP100 panel and four-instance fits. No NSGA-II evaluation,
new rollout, new labels, PPO, or automatic token experiment is started.
"""
from __future__ import annotations

import argparse
import copy
import fcntl
import math
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn

from m_operator_offline import (
    BALANCED, TinyOperatorTransformer, choose, decisions, digest, fit_scaler,
    gain_metrics, load_panel, masked_loss, predict_transformer, read_json,
    transform, tune_threshold, write_json,
)
from m_operator_evidence import decision_quality, feature_column_names, noise_report, structure_report, training_subsets


def log(message):
    print(f"[M-EVIDENCE] {message}", flush=True)


def train_fixed_updates(train, validation, seed, max_updates=320, progress=None):
    """Same tiny network/loss/optimizer as v1; equal update budget at every size.

    Evaluate every eight updates (one old four-instance epoch); stop after eight
    non-improving validations. Fewer instances mean more passes, not fewer steps.
    For the original 256-state fit this reproduces v1's 40-epoch schedule.
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    x, mask, y = [torch.as_tensor(a) for a in train]
    vx, vm, vy = [torch.as_tensor(a) for a in validation]
    if len(x) % 32:
        raise ValueError("learning-curve panel must have a whole number of 32-state minibatches")
    model = TinyOperatorTransformer(x.shape[-1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    best, best_loss, best_step, history = None, float("inf"), 0, []
    started = time.monotonic()
    cursor, order = len(x), None
    for step in range(max_updates + 1):
        if step:
            model.train()
            if cursor >= len(x):
                order, cursor = rng.permutation(len(x)), 0
            indices = order[cursor:cursor + 32]
            cursor += 32
            optimizer.zero_grad(set_to_none=True)
            loss = masked_loss(model(x[indices], mask[indices]), y[indices], mask[indices])
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("non-finite learning-curve loss")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
            optimizer.step()
        if step % 8 and step != max_updates:
            continue
        model.eval()
        with torch.no_grad():
            training_loss = float(masked_loss(model(x, mask), y, mask))
            validation_loss = float(masked_loss(model(vx, vm), vy, vm))
        if not np.isfinite([training_loss, validation_loss]).all():
            raise FloatingPointError("non-finite learning-curve validation")
        history.append({"step": step, "train_mse": training_loss, "validation_mse": validation_loss})
        if validation_loss < best_loss - 1e-10:
            best, best_loss, best_step = copy.deepcopy(model.state_dict()), validation_loss, step
        if progress and step % 40 == 0:
            progress(f"updates={step}/{max_updates} train_mse={training_loss:.5g} valid_mse={validation_loss:.5g} elapsed={time.monotonic()-started:.1f}s")
        if step - best_step >= 64:
            break
    model.load_state_dict(best)
    return model, {"seed": seed, "selected_update": best_step, "optimizer_updates": step,
                   "history": history, "elapsed_seconds": time.monotonic() - started}


def finite_mean(values):
    valid = [v for v in values if v is not None and math.isfinite(v)]
    return float(np.mean(valid)) if valid else None


def score(rows, y, mask, pred, validation, test, train, scale):
    threshold = tune_threshold(pred[validation], mask[validation], y[validation], scale)
    selected = choose(pred[test], mask[test], threshold)
    records = decisions([rows[i] for i in test], selected, y[test], pred[test])
    return {"threshold": threshold, "metrics": gain_metrics(records), "decisions": records}


def evaluate_subset(root, offline, fold, subset, rows, x, mask, y, seeds, max_updates, reuse_anchor):
    folder = root / fold["test"] / (f"n{len(subset)}_" + "__".join(subset))
    folder.mkdir(parents=True, exist_ok=True)
    complete = folder / "SUBSET_COMPLETE.json"
    if complete.exists():
        return read_json(complete)
    names = np.array([r["instance"] for r in rows])
    train = np.flatnonzero(np.isin(names, subset))
    validation = np.flatnonzero(names == fold["validation"])
    test = np.flatnonzero(names == fold["test"])
    scaler = fit_scaler(x[train], mask[train], y[train])
    xx, scale = transform(x, scaler), scaler["target_scale"]
    yy = (y / scale).astype(np.float32)
    predictions, seed_results = [], {}
    for seed in seeds:
        saved = folder / f"seed_{seed}.json"
        if saved.exists():
            fitted = read_json(saved)
        elif len(subset) == 4 and reuse_anchor:
            source = offline / fold["test"] / f"seed_{seed}.json"
            original = read_json(source)
            original_fold = read_json(offline / fold["test"] / "FOLD_COMPLETE.json")
            if original_fold["preprocessing"] != scaler:
                raise ValueError("cannot reuse four-instance fit: scaler differs")
            fitted = {**original, "selected_update": 8 * original["selected_epoch"],
                      "optimizer_updates": 8 * original["history"][-1]["epoch"],
                      "reused": True, "source": str(source), "source_sha256": digest(source)}
            write_json(saved, fitted)
        else:
            log(f"test={fold['test']} train_n={len(subset)} seed={seed} start")
            model, fitted = train_fixed_updates(
                (xx[train], mask[train], yy[train]), (xx[validation], mask[validation], yy[validation]),
                seed, max_updates=max_updates,
                progress=lambda message: log(f"test={fold['test']} train_n={len(subset)} seed={seed} {message}"))
            fitted["predictions"] = (predict_transformer(model, xx, mask) * scale).tolist()
            fitted["reused"] = False
            temporary = folder / f"seed_{seed}.pt.tmp"
            torch.save({"model": model.state_dict(), "preprocessing": scaler,
                        "training_instances": subset, "validation": fold["validation"], "test": fold["test"]}, temporary)
            temporary.replace(folder / f"seed_{seed}.pt")
            write_json(saved, fitted)
        pred = np.asarray(fitted["predictions"])
        if pred.shape != y.shape or not np.isfinite(pred).all():
            raise ValueError("invalid saved curve predictions")
        predictions.append(pred)
        seed_results[str(seed)] = {**score(rows, y, mask, pred, validation, test, train, scale),
            "selected_update": fitted["selected_update"], "optimizer_updates": fitted["optimizer_updates"],
            "reused": fitted["reused"]}
    mean, sd = np.mean(predictions, axis=0), np.std(predictions, axis=0)
    conservative = mean - sd
    conservative[:, BALANCED] = 0
    result = {"test_instance": fold["test"], "validation_instance": fold["validation"],
              "training_instances": subset, "training_instance_count": len(subset),
              "training_state_count": len(train), "seed_results": seed_results,
              "ensemble": score(rows, y, mask, conservative, validation, test, train, scale),
              "auxiliary": decision_quality(mean[test], y[test], mask[test], y[train], mask[train], scale)}
    if len(subset) == 4 and reuse_anchor:
        old = read_json(offline / fold["test"] / "FOLD_COMPLETE.json")["methods"]["TRANSFORMER_ENSEMBLE"]
        from analyze_m_operator_offline_v1 import assert_frozen_decisions
        assert_frozen_decisions(result["ensemble"]["decisions"], old["decisions"])
        result["original_four_instance_decisions_reproduced"] = True
    write_json(complete, result)
    return result


def summarize_curve(results, seeds):
    metrics = ("mean_net_gain_all_states", "switch_fraction", "positive_gain_total", "negative_gain_total")
    aux = ("mse_ratio_to_training_operator_mean", "weighted_sign_accuracy",
           "always_nonpositive_weighted_accuracy", "positive_gain_weighted_recall")
    curve = []
    for size in sorted({r["training_instance_count"] for r in results}):
        selected = [r for r in results if r["training_instance_count"] == size]
        per_instance = []
        for instance in sorted({r["test_instance"] for r in selected}):
            group = [r for r in selected if r["test_instance"] == instance]
            item = {"instance": instance, "training_subsets": len(group)}
            item.update({name: finite_mean([r["ensemble"]["metrics"][name] for r in group]) for name in metrics})
            item.update({name: finite_mean([r["auxiliary"][name] for r in group]) for name in aux})
            item["net_gain_std_across_training_subsets"] = float(np.std([r["ensemble"]["metrics"]["mean_net_gain_all_states"] for r in group]))
            per_instance.append(item)
        item = {"training_instances": size, "training_states_per_fit": size * 64,
                "subset_fits": len(selected), "per_instance": per_instance,
                "aggregation": "average across all training subsets within an instance, then equally across the six instances"}
        item.update({name: finite_mean([r[name] for r in per_instance]) for name in metrics + aux})
        item["positive_instances"] = sum(r["mean_net_gain_all_states"] > 1e-12 for r in per_instance)
        seed_gains = []
        for seed in seeds:
            values = [r["seed_results"][str(seed)]["metrics"]["mean_net_gain_all_states"] for r in selected]
            seed_gains.append({"seed": seed, "mean_net_gain_all_states": float(np.mean(values))})
        item["individual_training_seeds"] = seed_gains
        item["std_net_gain_across_training_seeds"] = float(np.std([r["mean_net_gain_all_states"] for r in seed_gains], ddof=1)) if len(seed_gains) > 1 else None
        item["mse_ratio_undefined_subset_count"] = sum(r["auxiliary"]["mse_ratio_undefined"] for r in selected)
        curve.append(item)
    return {"curve": curve, "role": "development diagnostics on previously inspected instances, not a new independent validation",
            "repeated_fit_warning": "subsets, seeds and states share data; neither fit count nor 384 states is an independent statistical sample count",
            "trend_interpretation": "an increasing curve supports more-data investigation; a flat curve does not prove a representation bottleneck",
            "five_instance_point": "omitted: retaining one outer test and one separate inner-validation instance among six leaves at most four training instances"}


def plot_curve(report, root):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    curve = report["curve"]
    sizes = [r["training_instances"] for r in curve]
    specs = [("mean_net_gain_all_states", "Mean selected gain vs Balanced", "Absolute one-step HV gain"),
             ("switch_fraction", "Switch fraction", "Fraction of all states"),
             ("mse_ratio_to_training_operator_mean", "MSE / constant-operator baseline", "Ratio (lower is better)"),
             ("weighted_sign_accuracy", "Absolute-gain-weighted sign accuracy", "Accuracy")]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    for ax, (metric, title, ylabel) in zip(axes.flat, specs):
        for instance in sorted(r["instance"] for r in curve[0]["per_instance"]):
            values = [next(r[metric] for r in point["per_instance"] if r["instance"] == instance) for point in curve]
            ax.plot(sizes, [np.nan if v is None else v for v in values], marker=".", alpha=.45, linewidth=1, label=instance)
        ax.plot(sizes, [np.nan if r[metric] is None else r[metric] for r in curve], "ko-", linewidth=2.5, label="Instance mean")
        if metric == "mean_net_gain_all_states":
            ax.axhline(0, color="grey", linestyle="--", linewidth=1)
            ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3))
        elif metric == "mse_ratio_to_training_operator_mean":
            ax.axhline(1, color="grey", linestyle="--", linewidth=1)
        elif metric == "weighted_sign_accuracy":
            ax.plot(sizes, [np.nan if r["always_nonpositive_weighted_accuracy"] is None else r["always_nonpositive_weighted_accuracy"] for r in curve],
                    "k--", label="Always nonpositive baseline")
            ax.set_ylim(-.03, 1.03)
        else:
            ax.set_ylim(-.03, 1.03)
        ax.set(title=title, xlabel="Training instances (one validation + one test held out)", ylabel=ylabel, xticks=sizes)
        ax.grid(alpha=.2)
    axes[0, 0].legend(fontsize=7, ncol=2)
    axes[1, 1].legend(fontsize=7, loc="best")
    prefix = "SYNTHETIC TEST DATA | " if report.get("synthetic_fixture") else ""
    fig.suptitle(prefix + "Development curves | CCP100 / source B400 | No new objective evaluations", fontsize=12)
    fig.savefig(root / "LEARNING_CURVE.png", dpi=170)
    plt.close(fig)


def execute(collection, offline, root, sizes=(2, 3, 4), max_updates=320):
    started = time.monotonic()
    original = read_json(offline / "PLAN.json")
    old_result = read_json(offline / "QUICK_RESULT.json")
    if not (offline / "OFFLINE_COMPLETE").exists() or not old_result.get("training_performed"):
        raise ValueError("requires completed original offline fitting")
    rows, x, mask, y, q, quality = load_panel(collection)
    inputs = [collection / p for p in ("PLAN.json", "BATCH_2_COMPLETE", "batch_2/DATASET.json", "batch_2/QUALITY.json")]
    source = Path(__file__).resolve().parent
    for path in inputs + [source / "m_operator_offline.py"]:
        if digest(path) != original["inputs_sha256"].get(str(path.resolve())):
            raise ValueError("source data/feature code changed relative to original offline run")
    if original["torch"] != torch.__version__ or original["numpy"] != np.__version__:
        raise ValueError("use the same torch/numpy environment for this paired curve")
    seeds, folds = original["training_seeds"], original["folds"]
    reuse_anchor = max_updates == 8 * original["maximum_epochs"] and original["patience"] == 8
    inputs += [offline / "PLAN.json", offline / "QUICK_RESULT.json", source / "m_operator_offline.py",
               source / "m_operator_evidence.py", source / "run_m_operator_evidence_v1.py", source / "transformer_ppo/observations.py"]
    for fold in folds:
        inputs.append(offline / fold["test"] / "FOLD_COMPLETE.json")
        inputs += [offline / fold["test"] / f"seed_{seed}.json" for seed in seeds]
    combinations_total = sum(len(training_subsets(fold, n)) for fold in folds for n in sizes)
    plan = {"protocol": "operator-evidence-v1", "inputs_sha256": {str(p.resolve()): digest(p) for p in inputs},
            "folds": folds, "sizes": list(sizes), "seeds": seeds,
            "training_compositions": "all subsets of the original four-instance training pool; identical held-out validation/test at every size",
            "composition_count": combinations_total, "training_fit_count_including_reused": combinations_total * len(seeds),
            "max_optimizer_updates": max_updates, "validation_every_updates": 8, "early_stop_after_stale_validations": 8,
            "reuse_four_instance_anchor": reuse_anchor,
            "optimization": "same width32 network, MSE, AdamW, batch32, seed, preprocessing and gate as original; same max updates across sizes",
            "source_design": {"CCP": 100, "population": 20, "search_budget": 400},
            "new_objective_evaluations": 0, "automatic_token_training": False, "automatic_ppo": False,
            "qualification": "repeatability is not an exact learnability ceiling; curves are exploratory; do not infer cause from a flat curve alone"}
    frozen = root / "PLAN.json"
    if frozen.exists() and read_json(frozen) != plan:
        raise ValueError("evidence inputs/plan changed; choose a new output directory")
    write_json(frozen, plan)
    if (root / "EVIDENCE_COMPLETE").exists():
        log("already complete; preserving existing results")
        return read_json(root / "QUICK_RESULT.json")
    log("stage=1 repeat reliability and recorded structure; zero new objective evaluations")
    noise = noise_report(rows, q, mask)
    structure = structure_report(rows)
    write_json(root / "REPEAT_RELIABILITY.json", noise)
    write_json(root / "STRUCTURE_AUDIT.json", structure)
    # Check clipping on current source folds; this is a diagnostic, no new scaler.
    clipping = []
    column_names = feature_column_names(rows[0]["observation"])
    if len(column_names) != x.shape[-1]:
        raise ValueError("feature-name audit disagrees with frozen feature width")
    for fold in folds:
        ids = [i for i, r in enumerate(rows) if r["instance"] == fold["test"]]
        scaler = read_json(offline / fold["test"] / "FOLD_COMPLETE.json")["preprocessing"]
        unbounded = (x[ids] - np.array(scaler["mean"])) / np.array(scaler["std"])
        clipped = np.abs(unbounded[mask[ids]]) > scaler["feature_clip"]
        rates = clipped.mean(0)
        top = np.argsort(-rates)[:8]
        clipping.append({"instance": fold["test"], "heldout_feature_value_clip_fraction": float(clipped.mean()),
                         "most_clipped_feature_columns": [{"index": int(i), "name": column_names[i], "fraction": float(rates[i])} for i in top if rates[i] > 0]})
    write_json(root / "NORMALIZATION_AUDIT.json", {"folds": clipping,
        "qualification": "clipping detects some scale shift, not all. Future token and summary arms must share any revised normalization; do not change scaling only in the token arm."})
    log(f"stage=2 learning curve compositions={combinations_total} fits_including_reused={combinations_total*len(seeds)}")
    completed = []
    for fold in folds:
        for size in sizes:
            for subset in training_subsets(fold, size):
                result = evaluate_subset(root, offline, fold, subset, rows, x, mask, y, seeds, max_updates, reuse_anchor)
                completed.append(result)
                write_json(root / "STATUS.json", {"stage": "learning_curve", "compositions_complete": len(completed),
                    "compositions_total": combinations_total, "elapsed_seconds": time.monotonic() - started})
                log(f"composition={len(completed)}/{combinations_total} complete test={fold['test']} train_n={size}")
    curve = summarize_curve(completed, seeds)
    curve["synthetic_fixture"] = bool(read_json(collection / "PLAN.json").get("synthetic_test_fixture", False))
    write_json(root / "LEARNING_CURVE.json", curve)
    plot_curve(curve, root)
    if any(digest(Path(p)) != value for p, value in plan["inputs_sha256"].items()):
        raise ValueError("source inputs/code changed during evidence collection")
    summary = {"completed": True, "new_objective_evaluations": 0, "new_ppo_updates": 0,
        "role": "development diagnosis; no automatic representation or PPO admission",
        "structure": {k: structure[k] for k in ("path_grouping_exact", "path_groups_recovered_state_count", "batch_identity_available", "full_batch_path_token_experiment_ready")},
        "repeat_reliability_pooled": {name: {"states": r["legal_state_count"],
            "pearson_by_rotation": [v["pearson"] for v in r["correlations"]],
            "mean_label_reliability_estimate": r["estimated_mean_label_reliability_clipped"]}
            for name, r in noise["by_group"]["pooled"].items()},
        "curve": [{k: r[k] for k in ("training_instances", "mean_net_gain_all_states", "switch_fraction",
                 "positive_instances", "mse_ratio_to_training_operator_mean", "weighted_sign_accuracy",
                 "always_nonpositive_weighted_accuracy", "positive_gain_weighted_recall",
                 "std_net_gain_across_training_seeds")} for r in curve["curve"]],
        "plot": str(root / "LEARNING_CURVE.png"), "elapsed_seconds_this_invocation": time.monotonic() - started,
        "qualification": "Review per-instance reliability and curves, not pooled scores alone. Old data lack explicit batch IDs; a full batch/path token claim is not supported."}
    write_json(root / "QUICK_RESULT.json", summary)
    write_json(root / "STATUS.json", {"stage": "complete", "compositions_complete": combinations_total})
    write_json(root / "EVIDENCE_COMPLETE", {"completed": True, "new_objective_evaluations": 0})
    log(f"COMPLETE elapsed={time.monotonic()-started:.1f}s; no PPO or token fitting started")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection-root", type=Path, required=True)
    parser.add_argument("--offline-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    collection, offline, root = (p.resolve() for p in (args.collection_root, args.offline_root, args.out))
    if root in (collection, offline) or collection in root.parents or offline in root.parents:
        parser.error("use a separate evidence directory; preserve the original panel and offline results")
    root.mkdir(parents=True, exist_ok=True)
    # Fail before fitting if the plotting dependency is missing.
    import matplotlib
    torch.set_num_threads(2)
    with (root / "evidence.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.exit(75, "another evidence job is running in this output directory\n")
        try:
            execute(collection, offline, root)
        except Exception as error:
            write_json(root / "STATUS.json", {"stage": "failed", "error": str(error)})
            raise


if __name__ == "__main__":
    main()
