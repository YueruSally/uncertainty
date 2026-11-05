#!/usr/bin/env python3
"""Six instance-held-out folds on the verified operator feedback panel."""
from __future__ import annotations

import argparse
import fcntl
from pathlib import Path
import time

import numpy as np
import torch

from m_operator_offline import (
    BALANCED, OPERATORS, PHASES, choose, cross_repeat_check, decisions, digest,
    fit_ridge, fit_scaler, gain_metrics, grouped_folds, load_panel,
    predict_ridge, predict_transformer, read_json, train_transformer, transform,
    tune_threshold, write_json,
)

SEEDS = (893001, 893002, 893003)


def log(message):
    print(f"[M-OFFLINE] {message}", flush=True)


def phase_predictor(rows, indices, y, mask):
    means = {}
    for phase in PHASES:
        selected = [i for i in indices if rows[i]["phase"] == phase]
        means[phase] = [float(np.mean(y[selected, a][mask[selected, a]]))
                        if any(mask[selected, a]) else -1e9 for a in range(5)]
    return np.array([means[row["phase"]] for row in rows]), means


def evaluate_prediction(rows, y, mask, prediction, validation, test, scale):
    threshold = tune_threshold(prediction[validation], mask[validation], y[validation], scale)
    chosen = choose(prediction[test], mask[test], threshold)
    records = decisions([rows[i] for i in test], chosen, y[test], prediction[test])
    return {"threshold": threshold, "always_balanced": threshold is None,
            "metrics": gain_metrics(records), "decisions": records}


def run_fold(root, fold, rows, x, mask, y, epochs, patience):
    folder = root / fold["test"]
    folder.mkdir(parents=True, exist_ok=True)
    names = np.array([r["instance"] for r in rows])
    train = np.flatnonzero(np.isin(names, fold["training"]))
    validation = np.flatnonzero(names == fold["validation"])
    test = np.flatnonzero(names == fold["test"])
    scaler = fit_scaler(x[train], mask[train], y[train])
    xx = transform(x, scaler)
    scale = scaler["target_scale"]
    yy = (y / scale).astype(np.float32)
    result = {"split": fold, "state_counts": {"training": len(train), "validation": len(validation), "test": len(test)},
              "preprocessing": scaler, "methods": {}, "training_seeds": {}}
    write_json(folder / "SPLIT.json", {"split": fold, "preprocessing": scaler,
               "state_counts": result["state_counts"]})
    result["methods"]["BALANCED"] = evaluate_prediction(rows, y, mask, np.zeros_like(y), validation, test, scale)
    phase_pred, phase_means = phase_predictor(rows, train, y, mask)
    result["methods"]["PHASE_RULE"] = evaluate_prediction(rows, y, mask, phase_pred, validation, test, scale)
    write_json(folder / "PHASE_MODEL.json", phase_means)
    coefficients = fit_ridge(xx[train], mask[train], yy[train])
    write_json(folder / "RIDGE_MODEL.json", {"alpha": 10.0, "coefficients": coefficients})
    ridge_pred = predict_ridge(coefficients, xx) * scale
    result["methods"]["RIDGE"] = evaluate_prediction(rows, y, mask, ridge_pred, validation, test, scale)
    predictions = []
    for seed in SEEDS:
        output = folder / f"seed_{seed}.json"
        if output.exists():
            fitted = read_json(output)
            log(f"test={fold['test']} seed={seed} reuse completed fit")
        else:
            log(f"test={fold['test']} validation={fold['validation']} seed={seed} start")
            model, fitted = train_transformer(
                (xx[train], mask[train], yy[train]),
                (xx[validation], mask[validation], yy[validation]), seed,
                epochs=epochs, patience=patience,
                progress=lambda msg: log(f"test={fold['test']} seed={seed} {msg}"))
            fitted["predictions"] = (predict_transformer(model, xx, mask) * scale).tolist()
            temporary = folder / f"seed_{seed}.pt.tmp"
            torch.save({"model": model.state_dict(), "feature_count": xx.shape[-1],
                        "preprocessing": scaler, "split": fold, "selected_epoch": fitted["selected_epoch"]}, temporary)
            temporary.replace(folder / f"seed_{seed}.pt")
            write_json(output, fitted)
        pred = np.asarray(fitted["predictions"])
        predictions.append(pred)
        scored = evaluate_prediction(rows, y, mask, pred, validation, test, scale)
        result["training_seeds"][str(seed)] = {**scored, "selected_epoch": fitted["selected_epoch"],
                                               "training_seconds": fitted["elapsed_seconds"]}
    # Three initializations are not independent datasets or a confidence interval.
    # The dispersion penalty is fixed beforehand; threshold still uses inner val.
    conservative = np.mean(predictions, axis=0) - np.std(predictions, axis=0)
    conservative[:, BALANCED] = 0.0
    result["methods"]["TRANSFORMER_ENSEMBLE"] = evaluate_prediction(rows, y, mask, conservative, validation, test, scale)
    write_json(folder / "FOLD_COMPLETE.json", result)
    return result


def summarize(folds):
    summary = {}
    for name in ("BALANCED", "PHASE_RULE", "RIDGE", "TRANSFORMER_ENSEMBLE"):
        records = [r for fold in folds for r in fold["methods"][name]["decisions"]]
        per_instance = [{"instance": fold["split"]["test"], **fold["methods"][name]["metrics"]} for fold in folds]
        per_trajectory = [{"run": run, **gain_metrics([r for r in records if r["run"] == run])}
                          for run in sorted({r["run"] for r in records})]
        summary[name] = {**gain_metrics(records), "per_instance": per_instance, "per_trajectory": per_trajectory,
                         "positive_instances": sum(r["mean_net_gain_all_states"] > 1e-12 for r in per_instance)}
    phase = summary["PHASE_RULE"]["mean_net_gain_all_states"]
    for name in ("RIDGE", "TRANSFORMER_ENSEMBLE"):
        row = summary[name]
        row["mean_gain_minus_phase_rule"] = row["mean_net_gain_all_states"] - phase
        row["offline_screen_pass"] = bool(row["mean_net_gain_all_states"] > 1e-12
                                            and row["positive_instances"] >= 4
                                            and row["mean_gain_minus_phase_rule"] > 1e-12)
    seeds = {}
    for seed in SEEDS:
        records = [r for f in folds for r in f["training_seeds"][str(seed)]["decisions"]]
        seeds[str(seed)] = gain_metrics(records)
    seed_gains = [r["mean_net_gain_all_states"] for r in seeds.values()]
    return summary, {"per_seed": seeds, "mean_net_gain_across_training_seeds": float(np.mean(seed_gains)),
                     "std_net_gain_across_training_seeds": float(np.std(seed_gains, ddof=1)),
                     "qualification": "training-initialization sensitivity, not an independent-sample confidence interval"}


def execute(collection, root, epochs, patience):
    start = time.monotonic()
    rows, x, mask, y, q, quality = load_panel(collection)
    folds = grouped_folds([r["instance"] for r in rows])
    source = Path(__file__).resolve().parent
    files = [collection / path for path in ("PLAN.json", "BATCH_2_COMPLETE", "batch_2/QUALITY.json", "batch_2/DATASET.json")]
    files += [source / name for name in ("m_operator_offline.py", "evaluate_m_operator_offline_v1.py", "analyze_m_operator_repeat_check.py")]
    plan = {"protocol": "operator-offline-v1", "inputs_sha256": {str(p.resolve()): digest(p) for p in files},
            "folds": folds, "training_seeds": list(SEEDS), "maximum_epochs": epochs, "patience": patience,
            "torch": torch.__version__, "numpy": np.__version__, "threads": torch.get_num_threads(),
            "features": "same five operator tokens + global + parent mean/std/count; no graph encoder; no outcome inputs",
            "ridge_alpha": 10.0, "transformer_width": 32, "transformer_heads": 4, "transformer_blocks": 1,
            "batch_size": 32, "learning_rate": 0.001, "weight_decay": 0.0001,
            "loss": "MSE of mean K3 gain vs Balanced, all legal alternatives, train-only RMS target scale",
            "checkpoint": "lowest inner-validation MSE, including epoch zero; random initialization",
            "threshold_grid_train_scale_multipliers": [None, 0, 0.1, 0.25, 0.5, 1],
            "threshold_selection": "inner-validation mean net gain; tie prefers fewer switches; null = always Balanced",
            "transformer_ensemble": "mean prediction minus one initialization std; dispersion heuristic, NOT calibrated lower confidence bound",
            "screen": "positive all-state mean, at least 4/6 positive instances, and better mean than phase rule",
            "repeat_gate": "held-out-repeat mean net gain > 1e-12 before fitting any model",
            "cost": "zero new objective evaluations; no PPO; existing counterfactual collection costs reported separately"}
    plan_path = root / "PLAN.json"
    if plan_path.exists() and read_json(plan_path) != plan:
        raise ValueError("offline plan/input/code changed; use a new output directory")
    write_json(plan_path, plan)
    if (root / "OFFLINE_COMPLETE").exists():
        log("already complete; preserving frozen results")
        return read_json(root / "QUICK_RESULT.json")
    log(f"states={len(rows)} instances={len(folds)} trajectories=12 feature_dim={x.shape[-1]} no new objective evaluations")
    repeat_check = cross_repeat_check(rows)
    write_json(root / "REPEAT_CHECK.json", repeat_check)
    # Always report the entire predeclared panel, never select observations by gain.
    diagnostics = {"all_legal_actions_zero_fraction": float(np.mean([np.all(v[m] == 0) for v, m in zip(q, mask)])),
                   "all_legal_mean_gains_tied_fraction": float(np.mean([np.ptp(v[m]) <= 1e-12 for v, m in zip(y, mask)])),
                   "qualification": "equal HV contributions do not establish identical offspring fingerprints",
                   "per_operator": {a: {"legal_states": int(mask[:, j].sum()),
                       "positive_gain_total": float(y[:, j][mask[:, j] & (y[:, j] > 0)].sum()),
                       "negative_gain_total": float(y[:, j][mask[:, j] & (y[:, j] < 0)].sum())}
                       for j, a in enumerate(OPERATORS)}}
    write_json(root / "LABEL_DIAGNOSTICS.json", diagnostics)
    repeat_pass = repeat_check["overall"]["mean_net_gain_all_states"] > 1e-12
    write_json(root / "REPEAT_GATE.json", {"pass": repeat_pass,
               "mean_net_gain_all_states": repeat_check["overall"]["mean_net_gain_all_states"],
               "minimum_mean_net_gain": 1e-12})
    if not repeat_pass:
        final = {"completed": True, "stage": "cross_repeat_check", "cross_repeat_pass": False,
                 "training_performed": False, "offline_screen_pass": False, "methods": {},
                 "new_objective_evaluations": 0, "new_ppo_updates": 0,
                 "next_action": "inspect label noise and operator execution; no model fitting or PPO started"}
        write_json(root / "QUICK_RESULT.json", final)
        write_json(root / "STATUS.json", {"stage": "complete", "folds_complete": 0, "cross_repeat_pass": False})
        write_json(root / "OFFLINE_COMPLETE", {"completed": True, "offline_screen_pass": False})
        log("COMPLETE: cross-repeat mean not positive; stopped before fitting")
        return final
    completed = []
    for number, fold in enumerate(folds, 1):
        saved = root / fold["test"] / "FOLD_COMPLETE.json"
        result = read_json(saved) if saved.exists() else run_fold(root, fold, rows, x, mask, y, epochs, patience)
        completed.append(result)
        write_json(root / "STATUS.json", {"stage": "offline", "folds_complete": number, "fold_count": len(folds),
                                         "elapsed_seconds": time.monotonic() - start})
        log(f"fold={number}/{len(folds)} test={fold['test']} complete")
    methods, seed_sensitivity = summarize(completed)
    any_pass = any(methods[n]["offline_screen_pass"] for n in ("RIDGE", "TRANSFORMER_ENSEMBLE"))
    final = {"completed": True, "scope": "offline held-out-instance single-step HV contribution; not final-search HV improvement",
             "cross_repeat_pass": True, "training_performed": True,
             "comparison_design": {"state_count": len(rows), "instance_count": 6, "trajectory_count": 12,
                 "replicates_per_legal_action": 3, "scenario_count": 100, "ccp_confidence": 0.9, "population_size": 20,
                 "source_search_budget": 400, "outer_folds": 6, "training_instances_per_fold": 4,
                 "inner_validation_instances_per_fold": 1, "test_instances_per_fold": 1},
             "new_objective_evaluations": 0, "new_ppo_updates": 0,
             "source_costs": {k: v for k, v in quality.items() if "evaluations" in k},
             "methods": methods, "transformer_seed_sensitivity": seed_sensitivity,
             "elapsed_seconds_this_invocation": time.monotonic() - start,
             "offline_screen_pass": any_pass,
             "next_action": "review candidate for paired closed-loop search; no automatic PPO" if any_pass else
                            "inspect held-out predictions and labels; do not expand PPO"}
    if any(digest(Path(path)) != expected for path, expected in plan["inputs_sha256"].items()):
        raise ValueError("inputs or source files changed during offline fitting")
    write_json(root / "QUICK_RESULT.json", final)
    write_json(root / "STATUS.json", {"stage": "complete", "folds_complete": 6, "offline_screen_pass": any_pass})
    write_json(root / "OFFLINE_COMPLETE", {"completed": True, "offline_screen_pass": any_pass})
    log(f"COMPLETE offline_screen_pass={any_pass}; this is not a final-search performance claim")
    return final


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--threads", type=int, choices=(1, 2), default=2)
    args = parser.parse_args(argv)
    if args.epochs < 1 or args.patience < 1:
        parser.error("epochs and patience must be positive")
    root, collection = args.out.resolve(), args.collection_root.resolve()
    if root == collection or collection in root.parents:
        parser.error("offline outputs must be outside the immutable collection directory")
    root.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    with (root / "offline.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.exit(75, "another offline comparison is running in this output directory\n")
        try:
            execute(collection, root, args.epochs, args.patience)
        except Exception as error:
            write_json(root / "STATUS.json", {"stage": "failed", "error": str(error)})
            raise


if __name__ == "__main__":
    main()
