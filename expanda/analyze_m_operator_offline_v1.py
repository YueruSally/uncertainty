#!/usr/bin/env python3
"""Explain frozen offline predictions; never retrain, retune or admit a policy.

The two ungated variants are post-hoc diagnostics on already exposed outer folds.
They must not be reported as a new independent validation or used to pick a new
threshold on these folds. All original predictions and decisions remain intact.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from m_operator_offline import (
    BALANCED, choose, decisions, digest, gain_metrics, load_panel,
    predict_ridge, read_json, transform, write_json,
)

VARIANTS = ("mean_prediction_zero_threshold", "minus_disagreement_zero_threshold", "frozen_policy")


def policy_report(rows, prediction, mask, target, threshold):
    selected = choose(prediction, mask, threshold)
    records = decisions(rows, selected, target, prediction)
    metrics = gain_metrics(records)
    metrics["switch_count"] = int((selected != BALANCED).sum())
    return {"metrics": metrics, "records": records}, selected


def decompose(rows, predictions, mask, target, threshold):
    """Keep model, data, labels and original threshold fixed; remove layers only."""
    mean = np.mean(predictions, axis=0)
    conservative = mean - np.std(predictions, axis=0)
    conservative[:, BALANCED] = 0.0
    reports, selections = {}, {}
    for name, scores, cutoff in zip(VARIANTS, (mean, conservative, conservative), (0.0, 0.0, threshold)):
        reports[name], selections[name] = policy_report(rows, scores, mask, target, cutoff)
    raw, penalized, frozen = (selections[name] for name in VARIANTS)
    reports["layer_effects"] = {
        "switches_removed_by_disagreement": int(((raw != BALANCED) & (penalized == BALANCED)).sum()),
        "actions_changed_by_disagreement": int((raw != penalized).sum()),
        "switches_removed_by_threshold": int(((penalized != BALANCED) & (frozen == BALANCED)).sum()),
        "always_balanced_threshold_selected": threshold is None,
    }
    return reports


def gate_diagnostic(prediction, mask, target, threshold, scale):
    """Replay the existing inner-validation grid, without selecting a new value."""
    grid = []
    for cutoff in (None, 0.0, 0.1 * scale, 0.25 * scale, 0.5 * scale, scale):
        selected = choose(prediction, mask, cutoff)
        gains = target[np.arange(len(target)), selected]
        grid.append({"threshold": cutoff, "switches": int((selected != BALANCED).sum()),
                     "mean_net_gain_all_states": float(gains.mean())})
    positive = [r for r in grid if r["threshold"] is not None and r["mean_net_gain_all_states"] > 1e-12]
    return {"frozen_threshold": threshold, "always_balanced": threshold is None,
            "positive_inner_validation_grid_choices": len(positive), "inner_validation_grid": grid}


def assert_frozen_decisions(computed, saved):
    def key(row):
        return row["run"], row["state_id"]
    actual = {key(row): row for row in computed}
    expected = {key(row): row for row in saved}
    if len(actual) != len(computed) or set(actual) != set(expected):
        raise ValueError("saved predictions and original decisions have different state identities")
    for state, row in actual.items():
        if row["selected_operator"] != expected[state]["selected_operator"] or not np.isclose(
                row["mean_heldout_gain"], expected[state]["mean_heldout_gain"], rtol=1e-10, atol=1e-14):
            raise ValueError("cannot reproduce original frozen decisions; stop diagnosis")


def inspect(collection, offline):
    if not (offline / "OFFLINE_COMPLETE").exists():
        raise ValueError("offline comparison is incomplete")
    plan = read_json(offline / "PLAN.json")
    original = read_json(offline / "QUICK_RESULT.json")
    if not original.get("training_performed"):
        raise ValueError("comparison stopped before training; inspect REPEAT_CHECK.json instead")
    # A different dataset order would silently associate predictions with the
    # wrong state. Verify the frozen dataset and feature code before using them.
    checks = {str(p.resolve()): p for p in (
        collection / "PLAN.json", collection / "BATCH_2_COMPLETE",
        collection / "batch_2/DATASET.json", collection / "batch_2/QUALITY.json",
        Path(__file__).resolve().parent / "m_operator_offline.py")}
    for name, path in checks.items():
        if digest(path) != plan["inputs_sha256"].get(name):
            raise ValueError(f"input or feature code differs from frozen run: {path}")
    rows, x, mask, y, _, _ = load_panel(collection)
    names = np.array([row["instance"] for row in rows])
    results, provenance = [], {str(p): digest(p) for p in checks.values()}
    for path in (offline / "PLAN.json", offline / "QUICK_RESULT.json", Path(__file__).resolve()):
        provenance[str(path)] = digest(path)
    for fold in plan["folds"]:
        folder = offline / fold["test"]
        saved = read_json(folder / "FOLD_COMPLETE.json")
        provenance[str(folder / "FOLD_COMPLETE.json")] = digest(folder / "FOLD_COMPLETE.json")
        if saved["split"] != fold:
            raise ValueError("fold split changed")
        train = np.flatnonzero(np.isin(names, fold["training"]))
        validation = np.flatnonzero(names == fold["validation"])
        test = np.flatnonzero(names == fold["test"])
        scaler = saved["preprocessing"]
        scale = scaler["target_scale"]
        yy = y / scale
        predictions, checkpoints = [], []
        for seed in plan["training_seeds"]:
            path = folder / f"seed_{seed}.json"
            fit = read_json(path)
            provenance[str(path)] = digest(path)
            pred = np.asarray(fit["predictions"])
            if pred.shape != (len(rows), 5) or not np.isfinite(pred).all():
                raise ValueError("invalid saved prediction array")
            predictions.append(pred)
            history = {r["epoch"]: r for r in fit["history"]}
            epoch = fit["selected_epoch"]
            checkpoints.append({"seed": seed, "selected_epoch": epoch,
                "initial_validation_mse": history[0]["validation_mse"],
                "selected_training_mse": history[epoch]["train_mse"],
                "selected_validation_mse": history[epoch]["validation_mse"],
                "validation_mse_reduction_from_initial": history[0]["validation_mse"] - history[epoch]["validation_mse"]})
        predictions = np.asarray(predictions)
        threshold = saved["methods"]["TRANSFORMER_ENSEMBLE"]["threshold"]
        partitions = {}
        for label, indices in (("training", train), ("inner_validation", validation), ("outer_test", test)):
            partitions[label] = decompose([rows[i] for i in indices], predictions[:, indices], mask[indices], y[indices], threshold)
        assert_frozen_decisions(partitions["outer_test"]["frozen_policy"]["records"],
                                saved["methods"]["TRANSFORMER_ENSEMBLE"]["decisions"])
        conservative = predictions.mean(0) - predictions.std(0)
        conservative[:, BALANCED] = 0.0
        ridge_path, phase_path = folder / "RIDGE_MODEL.json", folder / "PHASE_MODEL.json"
        provenance.update({str(p): digest(p) for p in (ridge_path, phase_path)})
        ridge = predict_ridge(read_json(ridge_path)["coefficients"], transform(x, scaler)) * scale
        phases = read_json(phase_path)
        phase_pred = np.array([phases[row["phase"]] for row in rows])
        gates = {}
        for method, pred in (("PHASE_RULE", phase_pred), ("RIDGE", ridge), ("TRANSFORMER_ENSEMBLE", conservative)):
            frozen_threshold = saved["methods"][method]["threshold"]
            gates[method] = gate_diagnostic(pred[validation], mask[validation], y[validation], frozen_threshold, scale)
            reproduced, _ = policy_report([rows[i] for i in test], pred[test], mask[test], y[test], frozen_threshold)
            assert_frozen_decisions(reproduced["records"], saved["methods"][method]["decisions"])
            gates[method]["outer_predicted_positive_states_before_threshold"] = int((choose(pred[test], mask[test], 0) != BALANCED).sum())
        # Test loss is diagnostic only; it never feeds back into checkpoint choice.
        error_baselines = {}
        alternative_mask = mask.copy()
        alternative_mask[:, BALANCED] = False
        constants = np.array([np.mean(yy[train, a][mask[train, a]]) if mask[train, a].any() else 0 for a in range(5)])
        for label, indices in (("training", train), ("inner_validation", validation), ("outer_test", test)):
            legal = alternative_mask[indices]
            targets = yy[indices][legal]
            const = np.broadcast_to(constants, yy[indices].shape)[legal]
            fitted = predictions.mean(0)[indices][legal] / scale
            error_baselines[label] = {"zero_predictor_mse": float(np.mean(targets ** 2)),
                "training_operator_mean_mse": float(np.mean((targets - const) ** 2)),
                "transformer_mean_mse": float(np.mean((targets - fitted) ** 2))}
        results.append({"test_instance": fold["test"], "inner_validation_instance": fold["validation"],
                        "checkpoints": checkpoints, "gates": gates, "error_baselines": error_baselines,
                        "partitions": partitions})
        print(f"[M-OFFLINE-DIAG] checked {len(results)}/{len(plan['folds'])}: {fold['test']}", flush=True)
    all_records = {name: [r for f in results for r in f["partitions"]["outer_test"][name]["records"]] for name in VARIANTS}
    variant_metrics = {}
    for name, records in all_records.items():
        per_instance = [gain_metrics(f["partitions"]["outer_test"][name]["records"]) for f in results]
        variant_metrics[name] = {**gain_metrics(records),
            "switch_count": sum(r["selected_operator"] != "replace" for r in records),
            "positive_instances": sum(m["mean_net_gain_all_states"] > 1e-12 for m in per_instance)}
    checkpoints = [p for f in results for p in f["checkpoints"]]
    summary = {"posthoc_diagnostic_only": True, "new_training_steps": 0, "new_objective_evaluations": 0,
        "original_offline_screen_pass": original["offline_screen_pass"],
        "saved_decisions_reproduced_exactly": True,
        "always_balanced_gate_folds": {method: sum(f["gates"][method]["always_balanced"] for f in results)
            for method in ("PHASE_RULE", "RIDGE", "TRANSFORMER_ENSEMBLE")},
        "checkpoint_count": len(checkpoints), "epoch_zero_checkpoint_count": sum(p["selected_epoch"] == 0 for p in checkpoints),
        "transformer_outer_variants": variant_metrics,
        "per_instance": [{"instance": f["test_instance"], "selected_epochs": [c["selected_epoch"] for c in f["checkpoints"]],
            "layer_effects": f["partitions"]["outer_test"]["layer_effects"],
            "outer_metrics": {name: {key: f["partitions"]["outer_test"][name]["metrics"][key]
                for key in ("mean_net_gain_all_states", "switch_count")} for name in VARIANTS}} for f in results],
        "qualification": "Ungated variants are post-hoc failure analysis, not new validated policies. No thresholds or checkpoints were changed."}
    if any(digest(Path(path)) != expected for path, expected in provenance.items()):
        raise ValueError("inputs changed during diagnosis")
    return {"summary": summary, "folds": results, "inputs_sha256": provenance}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection-root", type=Path, required=True)
    parser.add_argument("--offline-root", type=Path, required=True)
    args = parser.parse_args(argv)
    result = inspect(args.collection_root.resolve(), args.offline_root.resolve())
    report = args.offline_root.resolve() / "diagnosis_v1"
    write_json(report / "DIAGNOSIS.json", result)
    write_json(report / "QUICK_RESULT.json", result["summary"])
    print(f"[M-OFFLINE-DIAG] COMPLETE: {report / 'QUICK_RESULT.json'}", flush=True)


if __name__ == "__main__":
    main()
