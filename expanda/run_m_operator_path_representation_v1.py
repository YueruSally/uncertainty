#!/usr/bin/env python3
"""Paired SUMMARY vs PATH_TOKEN study on the frozen, verified CCP100 panel."""
from __future__ import annotations

import argparse
import fcntl
from pathlib import Path
import time

import numpy as np
import torch

from evaluate_m_operator_offline_v1 import evaluate_prediction
from m_operator_evidence import decision_quality, structure_report
from m_operator_offline import (
    BALANCED, digest, fit_scaler, gain_metrics, grouped_folds, load_panel,
    read_json, transform, write_json,
)
from m_operator_path_representation import (
    ARMS, MatchedPathTransformer, build_parent_inputs, fit, fit_parent_scaler,
    parent_arrays, predict,
)


def log(message):
    print(f"[M-PATH] {message}", flush=True)


def validate_saved_fit(folder, arm, seed):
    filename = folder / arm / f"seed_{seed}.json"
    record = read_json(filename)
    checkpoint = filename.with_suffix(".pt")
    if not checkpoint.exists() or digest(checkpoint) != record["checkpoint_sha256"]:
        raise ValueError(f"saved fit checkpoint missing/changed: {checkpoint}")
    return record


def fit_fold(root, fold, rows, x, mask, y, parents, groups, seeds, max_updates, status):
    folder = root / fold["test"]
    folder.mkdir(parents=True, exist_ok=True)
    complete = folder / "FOLD_COMPLETE.json"
    if complete.exists():
        for arm in ARMS:
            for seed in seeds:
                validate_saved_fit(folder, arm, seed)
        result = read_json(complete)
        for path, expected in result["fit_file_sha256"].items():
            if digest(root / path) != expected:
                raise ValueError("saved fit metadata changed")
        status["fits_complete"] += len(ARMS) * len(seeds)
        log(f"fold={fold['test']} reuse all paired fits")
        return result
    names = np.array([r["instance"] for r in rows])
    train = np.flatnonzero(np.isin(names, fold["training"]))
    validation, test = np.flatnonzero(names == fold["validation"]), np.flatnonzero(names == fold["test"])
    common = fit_scaler(x[train], mask[train], y[train])
    ps = fit_parent_scaler(parents, train)
    xx, scale = transform(x, common), common["target_scale"]
    yy = (y / scale).astype(np.float32)
    inputs = {arm: build_parent_inputs(parents, groups, ps, arm) for arm in ARMS}
    preprocessing = {"common_candidate_context": common, "shared_parent_columns": ps}
    write_json(folder / "SPLIT.json", {"split": fold, "preprocessing": preprocessing})
    predictions, training = {arm: [] for arm in ARMS}, {arm: {} for arm in ARMS}
    fit_files = {}
    for si, seed in enumerate(seeds):
        paired = {}
        # Alternate arm order for timing; data permutations/initial weights are
        # reset from the same seed and independent of wall-clock execution order.
        for arm in (ARMS if si % 2 == 0 else ARMS[::-1]):
            destination = folder / arm
            destination.mkdir(exist_ok=True)
            saved = destination / f"seed_{seed}.json"
            if saved.exists():
                fitted = validate_saved_fit(folder, arm, seed)
                log(f"fold={fold['test']} arm={arm} seed={seed} reuse")
            else:
                log(f"fold={fold['test']} arm={arm} seed={seed} start fits={status['fits_complete']}/{status['fit_count']}")
                model, fitted = fit(xx, mask, yy, inputs[arm], train, validation, seed, max_updates,
                    progress=lambda msg: log(f"fold={fold['test']} arm={arm} seed={seed} {msg}"))
                fitted["predictions"] = (predict(model, xx, mask, inputs[arm]) * scale).tolist()
                temporary = saved.with_suffix(".pt.tmp")
                torch.save({"model": model.state_dict(), "feature_count": xx.shape[-1], "arm": arm,
                            "preprocessing": preprocessing, "split": fold, "seed": seed,
                            "selected_update": fitted["selected_update"]}, temporary)
                temporary.replace(saved.with_suffix(".pt"))
                fitted["checkpoint_sha256"] = digest(saved.with_suffix(".pt"))
                write_json(saved, fitted)
            pred = np.asarray(fitted["predictions"], dtype=float)
            if pred.shape != y.shape or not np.isfinite(pred).all():
                raise ValueError("invalid saved representation predictions")
            predictions[arm].append(pred)
            scored = evaluate_prediction(rows, y, mask, pred, validation, test, scale)
            training[arm][str(seed)] = {**scored,
                **{k: fitted[k] for k in ("selected_update", "optimizer_updates", "parameter_count",
                                          "initial_weights_sha256", "elapsed_seconds")}}
            paired[arm] = fitted
            fit_files[str(saved.relative_to(root))] = digest(saved)
            status["fits_complete"] += 1
            write_json(root / "STATUS.json", {**status, "stage": "training", "fold": fold["test"],
                                              "arm": arm, "seed": seed})
            log(f"fit_complete={status['fits_complete']}/{status['fit_count']} fold={fold['test']} arm={arm} seed={seed}")
        if any(paired[ARMS[0]][k] != paired[ARMS[1]][k] for k in ("parameter_count", "initial_weights_sha256")):
            raise ValueError("paired arms do not have identical initial weights/parameter counts")
    methods = {}
    for arm in ARMS:
        mean, sd = np.mean(predictions[arm], axis=0), np.std(predictions[arm], axis=0)
        conservative = mean - sd
        conservative[:, BALANCED] = 0.
        methods[arm] = {**evaluate_prediction(rows, y, mask, conservative, validation, test, scale),
            "auxiliary": decision_quality(mean[test], y[test], mask[test], y[train], mask[train], scale)}
    result = {"split": fold, "preprocessing": preprocessing, "state_counts": {
        "training": len(train), "validation": len(validation), "test": len(test)},
        "methods": methods, "training_seeds": training, "paired_initialization_exact": True,
        "fit_file_sha256": fit_files}
    write_json(complete, result)
    return result


def average_defined(values):
    values = [v for v in values if v is not None]
    return float(np.mean(values)) if values else None


def summarize(folds, seeds, old_result):
    methods = {}
    aux = ("mse_ratio_to_training_operator_mean", "weighted_sign_accuracy",
           "always_nonpositive_weighted_accuracy", "positive_gain_weighted_recall")
    for arm in ARMS:
        records = [r for f in folds for r in f["methods"][arm]["decisions"]]
        per_instance = [{"instance": f["split"]["test"], **f["methods"][arm]["metrics"],
                         **f["methods"][arm]["auxiliary"]} for f in folds]
        seed_results = {}
        for seed in seeds:
            seeded = [r for f in folds for r in f["training_seeds"][arm][str(seed)]["decisions"]]
            seed_results[str(seed)] = gain_metrics(seeded)
        methods[arm] = {**gain_metrics(records), "per_instance": per_instance,
            "per_trajectory": [{"run": run, **gain_metrics([r for r in records if r["run"] == run])}
                               for run in sorted({r["run"] for r in records})],
            "positive_instances": sum(r["mean_net_gain_all_states"] > 1e-12 for r in per_instance),
            **{k: average_defined([r[k] for r in per_instance]) for k in aux},
            "per_training_seed": seed_results,
            "std_net_gain_across_training_seeds": float(np.std(
                [r["mean_net_gain_all_states"] for r in seed_results.values()], ddof=1)) if len(seeds) > 1 else None,
            "total_training_seconds": sum(f["training_seeds"][arm][str(s)]["elapsed_seconds"] for f in folds for s in seeds)}
    paired = [{"instance": f["split"]["test"],
        "path_minus_summary_net_gain": f["methods"]["PATH_TOKEN"]["metrics"]["mean_net_gain_all_states"] -
                                       f["methods"]["SUMMARY"]["metrics"]["mean_net_gain_all_states"]} for f in folds]
    difference = methods["PATH_TOKEN"]["mean_net_gain_all_states"] - methods["SUMMARY"]["mean_net_gain_all_states"]
    improving = sum(r["path_minus_summary_net_gain"] > 1e-12 for r in paired)
    passed = (methods["PATH_TOKEN"]["mean_net_gain_all_states"] > 1e-12 and difference > 1e-12
              and methods["PATH_TOKEN"]["positive_instances"] >= 4 and improving >= 4)
    old = old_result["methods"]["TRANSFORMER_ENSEMBLE"]
    return {"methods": methods,
            "paired_comparison": {"path_minus_summary_mean_net_gain": difference,
                                  "improved_instances": improving, "per_instance": paired},
            "historical_summary_reference": {
                "mean_net_gain_all_states": old["mean_net_gain_all_states"],
                "switch_fraction": old["switch_fraction"],
                "role": "old smaller architecture; contextual reference only, not the matched control"},
            "screening_criteria": {"path_net_gain_positive": True, "path_better_than_summary": True,
                "minimum_positive_instances": 4, "minimum_improved_instances": 4, "zero_tolerance": 1e-12},
            "representation_screen_pass": bool(passed),
            "next_action": "review path policy for CCP100 paired closed-loop small search; no automatic PPO" if passed else
                           "stop this representation experiment; inspect paired decisions before any further change"}


def execute(collection, offline, root):
    started = time.monotonic()
    original, old_result = read_json(offline / "PLAN.json"), read_json(offline / "QUICK_RESULT.json")
    if not (offline / "OFFLINE_COMPLETE").exists() or not old_result.get("training_performed"):
        raise ValueError("requires the completed original offline experiment")
    rows, x, mask, y, _, quality = load_panel(collection)
    source = Path(__file__).resolve().parent
    inputs = [collection / name for name in ("PLAN.json", "batch_2/QUALITY.json", "batch_2/DATASET.json", "BATCH_2_COMPLETE")]
    for path in inputs + [source / "m_operator_offline.py"]:
        if digest(path) != original["inputs_sha256"].get(str(path.resolve())):
            raise ValueError("original collection/features changed; cannot call this a paired representation study")
    if original["torch"] != torch.__version__ or original["numpy"] != np.__version__:
        raise ValueError("use the same torch/numpy environment as the original offline run")
    folds, seeds = original["folds"], original["training_seeds"]
    if folds != grouped_folds([r["instance"] for r in rows]) or original["patience"] != 8:
        raise ValueError("requires original six instance folds and eight-validation patience")
    max_updates = int(original["maximum_epochs"]) * 8
    inputs += [offline / name for name in ("PLAN.json", "QUICK_RESULT.json", "OFFLINE_COMPLETE")]
    inputs += [source / name for name in ("m_operator_offline.py", "m_operator_evidence.py",
        "m_operator_path_representation.py", "run_m_operator_path_representation_v1.py", "evaluate_m_operator_offline_v1.py")]
    parameter_count = sum(p.numel() for p in MatchedPathTransformer(x.shape[-1]).parameters())
    plan = {"protocol": "operator-path-representation-v1", "inputs_sha256": {str(p.resolve()): digest(p) for p in inputs},
        "folds": folds, "seeds": seeds, "arms": list(ARMS), "max_optimizer_updates": max_updates,
        "torch": torch.__version__, "numpy": np.__version__, "threads": torch.get_num_threads(),
        "parameter_count_each_arm": parameter_count,
        "paired_controls": {"same_architecture": True, "same_parameter_count": True,
            "same_initial_weights_per_seed": True, "same_candidate_global_summary_vectors": True,
            "same_train_only_primitive_normalization": True, "same_effective_batch_size": 32,
            "same_microbatch_size": 4, "same_minibatch_order_per_seed": True,
            "same_update_cap": True, "same_inner_validation_stopping_rule": True,
            "same_legal_action_MSE": True, "same_gate_grid_and_calibration_rule": True},
        "architecture": "width32; within-path attention/pooling; operator-to-path cross-attention; operator self-attention; all four heads; no dropout",
        "inputs": {"SUMMARY": "existing candidate/context vectors plus mean/std of transformed parent rows as one two-token block",
                   "PATH_TOKEN": "same candidate/context vectors plus recorded path headers and ordered arcs, grouped by path"},
        "scope": "tests recorded path structure with a matched new summary control; no explicit batch/global-path identity; no outcome inputs",
        "optimizer": {"name": "AdamW", "learning_rate": 1e-3, "weight_decay": 1e-4, "gradient_clip": 5.},
        "validation_every_updates": 8, "early_stop_after_stale_validations": 8,
        "checkpoint": "lowest inner-validation MSE including update0; outer labels never select checkpoints",
        "gate": "mean minus one initialization std; old inner-validation threshold grid; NOT calibrated confidence bound",
        "source_design": {"scenario_count": 100, "population_size": 20, "evaluation_budget": 400, "repeats": 3},
        "screen": "positive net gain on all states and >=4/6 instances; better than matched summary overall and >=4/6 instances",
        "new_objective_evaluations": 0, "new_ppo_updates": 0, "automatic_closed_loop": False,
        "qualification": "previously inspected development instances; no independent final-search claim; seed variation is not a statistical confidence interval"}
    saved_plan = root / "PLAN.json"
    if saved_plan.exists() and read_json(saved_plan) != plan:
        raise ValueError("inputs/code/plan changed; preserve this directory and use a new output root")
    write_json(saved_plan, plan)
    if (root / "REPRESENTATION_COMPLETE").exists():
        log("COMPLETE already present; preserving results")
        return read_json(root / "QUICK_RESULT.json")
    structure = structure_report(rows)
    write_json(root / "STRUCTURE_AUDIT.json", structure)
    if not structure["path_grouping_exact"] or structure["path_groups_recovered_state_count"] != len(rows):
        raise ValueError("path boundaries/order cannot be recovered exactly")
    parents, groups = parent_arrays(rows)
    write_json(root / "INPUT_SIZE.json", {
        "parent_tokens_max": max(map(len, parents)), "path_count_max": max(map(len, groups)),
        "path_tokens_max": max(1 + len(g["arc_indices"]) for gs in groups for g in gs),
        "truncated_tokens": 0, "state_microbatch": 4,
        "qualification": "hierarchical local attention; no full-parent quadratic self-attention"})
    log(f"states={len(rows)} folds={len(folds)} seeds={len(seeds)} fits={len(folds)*len(seeds)*2} params_per_arm={parameter_count}; no evaluations/PPO")
    status = {"fits_complete": 0, "fit_count": len(folds) * len(seeds) * 2}
    completed = []
    for i, fold in enumerate(folds, 1):
        completed.append(fit_fold(root, fold, rows, x, mask, y, parents, groups, seeds, max_updates, status))
        write_json(root / "STATUS.json", {**status, "stage": "training", "folds_complete": i,
                                         "elapsed_seconds": time.monotonic() - started})
        log(f"fold_complete={i}/{len(folds)} fits={status['fits_complete']}/{status['fit_count']} elapsed={time.monotonic()-started:.1f}s")
    result = summarize(completed, seeds, old_result)
    result.update({"completed": True, "new_objective_evaluations": 0, "new_ppo_updates": 0,
        "parameter_count_each_arm": parameter_count, "paired_initialization_exact": all(f["paired_initialization_exact"] for f in completed),
        "comparison_design": {"state_count": len(rows), "instances": 6, "trajectories": 12,
            "training_instances": 4, "validation_instances": 1, "outer_test_instances": 1, "fold_count": 6,
            "training_seed_count": len(seeds), "scenario_count": 100, "source_search_budget": 400},
        "elapsed_seconds_this_invocation": time.monotonic() - started,
        "synthetic_test_fixture": bool(read_json(collection / "PLAN.json").get("synthetic_test_fixture")),
        "qualification": "offline one-step selection gain in absolute HV units; not percent improvement or final search HV; inspected development instances"})
    if any(digest(Path(path)) != sha for path, sha in plan["inputs_sha256"].items()):
        raise ValueError("input/source changed during fitting")
    write_json(root / "COMPLETE.json", result)
    quick = {k: v for k, v in result.items() if k != "methods"}
    quick["methods"] = {arm: {k: value for k, value in method.items() if k not in ("per_trajectory", "per_training_seed")}
                        for arm, method in result["methods"].items()}
    write_json(root / "QUICK_RESULT.json", quick)
    write_json(root / "STATUS.json", {**status, "stage": "complete", "representation_screen_pass": result["representation_screen_pass"]})
    write_json(root / "REPRESENTATION_COMPLETE", {"completed": True, "representation_screen_pass": result["representation_screen_pass"]})
    log(f"COMPLETE representation_screen_pass={result['representation_screen_pass']}; no automatic search/PPO")
    return quick


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection-root", type=Path, required=True)
    parser.add_argument("--offline-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    collection, offline, root = (p.resolve() for p in (args.collection_root, args.offline_root, args.out))
    if any(root == p or p in root.parents or root in p.parents for p in (collection, offline)):
        parser.error("use a separate output directory; preserve original collection and offline results")
    root.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    with (root / "representation.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.exit(75, "another representation job is already running here\n")
        try:
            execute(collection, offline, root)
        except Exception as error:
            write_json(root / "STATUS.json", {"stage": "failed", "error": str(error)})
            raise


if __name__ == "__main__":
    main()
