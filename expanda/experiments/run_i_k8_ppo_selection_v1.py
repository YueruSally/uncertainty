#!/usr/bin/env python3
"""Select and freeze an I-K8 PPO checkpoint on 20 validation instances."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import fcntl
import math
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import threading
import time

SOURCE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE))

import torch

from i_k8_scale_policy import FrozenIInitializationRanker
from m_operator_offline import digest, read_json, write_json
from run_learning_components_deterministic import _normalizer
from transformer_ppo.episode import run_episode
from transformer_ppo.initialization_k8 import InitializationK8Controller
from transformer_ppo.instance_catalog import canonical_digest, read_instance


REFERENCE_UPDATE = 0
PPO_UPDATES = (2, 4, 8)
ALL_UPDATES = (REFERENCE_UPDATE, *PPO_UPDATES)


def method(update):
    return f"I_K8_TRANSFORMER_PPO_U{update:03d}"


def log(message):
    print(f"[I-K8-PPO-SELECT] {message}", flush=True)


def validate_run(row, expected_method, seed, plan_sha):
    if (row.get("completed") is not True
            or row.get("strategy") != expected_method
            or row.get("algorithm_seed") != seed
            or row.get("plan_sha256") != plan_sha
            or row.get("real_objective_evaluation_count") != 400
            or row.get("evaluation_budget") != 400
            or row.get("initialization_k8_selection_mode") != "transformer"
            or row.get("initialization_k8_pool_events") != 20
            or row.get("initialization_k8_candidate_objective_evaluations") != 20
            or row.get("initialization_k8_hidden_candidate_objective_evaluations") != 0
            or row.get("ppo_updates", 0)
            or row.get("training_feedback_evaluations", 0)
            or row.get("counterfactual_evaluation_count", 0)
            or not row.get("scenario_digest")
            or not math.isfinite(float(
                row.get("final_hypervolume", math.nan)))
            or float(row.get("final_hypervolume", 0.0)) <= 0.0):
        raise ValueError("invalid I-K8 PPO checkpoint validation search")


def run_worker(a):
    torch.set_num_threads(1)
    info, _ = read_instance(a.instance)
    predictor = FrozenIInitializationRanker(
        read_json(a.policy_bundle), digest)
    holders = []

    def factory(*args, **kwargs):
        controller = InitializationK8Controller(
            *args, selection_mode="transformer", pool_size=8,
            selection_policy=predictor, **kwargs)
        holders.append(controller)
        return controller

    result = run_episode(
        a.data, a.instance, a.out, None, _normalizer(a.normalization),
        algorithm_seed=a.seed, evaluation_budget=400, pop_size=20,
        generations=1000, evaluation_mode="ccp", component="BASE",
        scenario_count=100, ccp_confidence=.9,
        uncertainty_profile="transport_border_saa_v1",
        inference_controller_factories={"BASE": factory},
        method_override=a.method)
    if len(holders) != 1:
        raise RuntimeError("I-K8 PPO validation controller was not installed once")
    result.update({
        "strategy": a.method,
        "instance": info["instance_id"],
        "algorithm_seed": a.seed,
        "plan_sha256": a.plan_sha,
        "policy_bundle_sha256": digest(a.policy_bundle),
        "scenario_digest": read_json(a.out / "configuration.json")[
            "scenario_digest"],
        "new_training_steps": 0,
        "new_ppo_updates": 0,
        "learning_enabled": False,
        "role": "frozen I-K8 PPO checkpoint selection on validation instance",
    })
    validate_run(result, a.method, a.seed, a.plan_sha)
    write_json(a.out / "RUN_COMPLETE.json", result)


def validation_summary(rows, validation_records):
    keyed = {(row["instance"], row["strategy"]): row for row in rows}
    expected = {(record["instance_id"], method(update))
                for record in validation_records for update in ALL_UPDATES}
    if len(keyed) != len(rows) or set(keyed) != expected:
        raise ValueError("I-K8 validation checkpoint grid incomplete or duplicated")
    candidates = []
    for update in ALL_UPDATES:
        label = method(update)
        values = []
        for record in validation_records:
            name = record["instance_id"]
            baseline = keyed[name, method(REFERENCE_UPDATE)]
            row = keyed[name, label]
            if (row["scenario_digest"] != baseline["scenario_digest"]
                    or row["algorithm_seed"] != baseline["algorithm_seed"]):
                raise ValueError("I-K8 validation pairing changed")
            values.append(100.0 * (
                row["final_hypervolume"] / baseline["final_hypervolume"] - 1.0))
        candidates.append({
            "update": update,
            "method": label,
            "validation_instances": len(values),
            "mean_paired_percent_vs_supervised": statistics.fmean(values),
            "positive_instances": sum(value > 1e-12 for value in values),
            "ties": sum(abs(value) <= 1e-12 for value in values),
            "negative_instances": sum(value < -1e-12 for value in values),
            "paired_percent": values,
        })
    ppo_candidates = [
        row for row in candidates if row["update"] in PPO_UPDATES]
    selected = max(ppo_candidates, key=lambda row: (
        row["mean_paired_percent_vs_supervised"],
        row["positive_instances"], -row["update"]))
    return {
        "checkpoint_candidates": candidates,
        "selection_rule": (
            "among true PPO updates 2/4/8, maximize mean paired final-HV "
            "percent versus frozen supervised update0 over all 20 validation "
            "instances; then more positive instances; then earlier update"),
        "selected_update": selected["update"],
        "selected_method": selected["method"],
        "selected_validation_mean_percent_vs_supervised": selected[
            "mean_paired_percent_vs_supervised"],
        "selected_validation_positive_instances": selected[
            "positive_instances"],
        "supervised_reference_update": REFERENCE_UPDATE,
    }


def load_catalog(generated_instances):
    catalog = read_json(generated_instances / "catalog.json")
    identity = dict(catalog)
    claimed = identity.pop("catalog_digest", None)
    if (canonical_digest(identity) != claimed
            or catalog.get("train_count") != 200
            or catalog.get("validation_count") != 20
            or catalog.get("final_test_instances_used") is not False):
        raise ValueError("I-K8 generated instance catalog changed")
    records = [row for row in catalog["instances"]
               if row["development_role"] == "validation"]
    if len(records) != 20:
        raise ValueError("I-K8 PPO selection requires 20 validation instances")
    return records


def frozen_plan(a, records, progress):
    checkpoint_map = {row["update"]: row for row in progress["checkpoints"]}
    if tuple(sorted(checkpoint_map)) != ALL_UPDATES:
        raise ValueError("I-K8 PPO checkpoint candidates changed")
    inputs = [
        a.data, a.normalization, a.ppo_root / "PLAN.json",
        a.ppo_root / "TRAINING_PROGRESS.json",
        a.generated_instances / "catalog.json",
        a.curve_root / "BUNDLE.json",
        a.development_root / "QUICK_RESULT.json",
        Path(__file__).resolve(), SOURCE / "i_k8_scale_policy.py",
        SOURCE / "transformer_ppo/initialization_k8.py",
    ]
    inputs.extend(Path(row["bundle"]) for row in checkpoint_map.values())
    inputs.extend(a.generated_instances / f"{row['instance_id']}.json"
                  for row in records)
    return {
        "protocol": "i-k8-ppo-selection-v1",
        "checkpoint_updates": list(ALL_UPDATES),
        "ppo_checkpoint_candidates": list(PPO_UPDATES),
        "supervised_reference_update": REFERENCE_UPDATE,
        "validation_instances": [row["instance_id"] for row in records],
        "validation_algorithm_seeds": {
            row["instance_id"]: row["algorithm_seed"] for row in records},
        "validation_search_count": len(records) * len(ALL_UPDATES),
        "scenario_count": 100,
        "ccp_confidence": .9,
        "population": 20,
        "budget": 400,
        "candidate_pool": 8,
        "checkpoint_selection": (
            "among updates 2/4/8, maximum mean paired final-HV percent "
            "versus update0 on all 20 validation instances; tie by positive "
            "instance count then earlier update"),
        "gradient_instances": 200,
        "validation_instances_used_for_gradient": 0,
        "workers": a.workers,
        "threads_per_worker": 1,
        "formal_methods_after_freeze": [
            "BASE", "I_DIRECTED_BALANCED",
            "frozen supervised I-K8 Transformer",
            "frozen validation-selected I-K8 Transformer-PPO"],
        "development_only": True,
        "opens_locked_final_test": False,
        "checkpoint_bundles": {
            str(update): checkpoint_map[update]["bundle"]
            for update in ALL_UPDATES},
        "input_sha256": {
            str(Path(path).resolve()): digest(path) for path in inputs},
    }


def execute(a):
    root = a.out.resolve()
    progress = read_json(a.ppo_root / "TRAINING_PROGRESS.json")
    development = read_json(a.development_root / "QUICK_RESULT.json")
    if (not (a.ppo_root / "TRAINING_COMPLETE").exists()
            or progress.get("ppo_updates") != 8
            or progress.get("training_instances") != 200
            or progress.get("validation_instances_used_for_gradient") != 0
            or development.get("development_search_pass") is not True):
        raise ValueError("I-K8 bounded PPO training or admission is incomplete")
    records = load_catalog(a.generated_instances)
    plan = frozen_plan(a, records, progress)
    root.mkdir(parents=True, exist_ok=True)
    if (root / "PLAN.json").exists() and read_json(root / "PLAN.json") != plan:
        raise ValueError("I-K8 PPO selection plan changed")
    write_json(root / "PLAN.json", plan)
    if (root / "PIPELINE_COMPLETE").exists():
        log("already complete; see QUICK_RESULT.json")
        return
    plan_sha = digest(root / "PLAN.json")
    checkpoint_map = {
        row["update"]: Path(row["bundle"]) for row in progress["checkpoints"]}
    (root / "logs").mkdir(exist_ok=True)
    started = time.monotonic()
    lock = threading.Lock()
    completed = 0
    environment = {
        **os.environ,
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "PYTHONUNBUFFERED": "1",
    }

    def run_search(folder, instance_path, seed, update):
        nonlocal completed
        label = method(update)
        reused = (folder / "RUN_COMPLETE.json").exists()
        if not reused:
            if folder.exists():
                quarantine = root / "interrupted_runs" / (
                    f"{instance_path.stem}_{label}_{time.time_ns()}")
                quarantine.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(folder, quarantine)
            logfile = root / "logs" / (
                f"{instance_path.stem}_{label}_{time.time_ns()}.log")
            command = [
                sys.executable, "-u", str(Path(__file__).resolve()),
                "--worker", "--data", str(a.data.resolve()),
                "--generated-instances", str(a.generated_instances.resolve()),
                "--curve-root", str(a.curve_root.resolve()),
                "--development-root", str(a.development_root.resolve()),
                "--normalization", str(a.normalization.resolve()),
                "--ppo-root", str(a.ppo_root.resolve()),
                "--out", str(folder), "--instance", str(instance_path.resolve()),
                "--seed", str(seed), "--method", label,
                "--policy-bundle", str(checkpoint_map[update]),
                "--plan-sha", plan_sha,
            ]
            with logfile.open("w") as handle:
                result = subprocess.run(
                    command, cwd=SOURCE, env=environment,
                    stdout=handle, stderr=subprocess.STDOUT, timeout=300)
            if result.returncode:
                tail = "\n".join(logfile.read_text(
                    errors="replace").splitlines()[-25:])
                raise RuntimeError(f"worker failed: {logfile}\n{tail}")
        row = read_json(folder / "RUN_COMPLETE.json")
        validate_run(row, label, seed, plan_sha)
        with lock:
            completed += 1
            write_json(root / "STATUS.json", {
                "stage": "validation_selection",
                "completed_searches": completed,
                "total_searches": len(records) * len(ALL_UPDATES),
                "elapsed_seconds": time.monotonic() - started,
            })
            log(f"{completed}/{len(records) * len(ALL_UPDATES)} "
                f"{'reused' if reused else 'complete'} "
                f"{instance_path.stem}/{label}")
        return row

    rows = []
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        futures = []
        for record in records:
            path = a.generated_instances / f"{record['instance_id']}.json"
            for update in ALL_UPDATES:
                folder = root / "validation" / record["instance_id"] / method(update)
                futures.append(pool.submit(
                    run_search, folder, path, record["algorithm_seed"], update))
        for future in as_completed(futures):
            rows.append(future.result())

    selection = validation_summary(rows, records)
    selected_update = selection["selected_update"]
    selected_checkpoint = next(
        row for row in progress["checkpoints"]
        if row["update"] == selected_update)
    supervised_bundle = a.curve_root / "BUNDLE.json"
    supervised_model = read_json(supervised_bundle)["models"]["200"]
    freeze = {
        "protocol": "i-k8-formal-method-freeze-v1",
        "reference_method": "BASE",
        "heuristic_method": "I_DIRECTED_BALANCED",
        "supervised": {
            "method": "I_K8_TRANSFORMER",
            "bundle": str(supervised_bundle.resolve()),
            "bundle_sha256": digest(supervised_bundle),
            "checkpoint_sha256": supervised_model["checkpoint_sha256"],
        },
        "transformer_ppo": {
            "method": method(selected_update),
            "ppo_update": selected_update,
            "bundle": selected_checkpoint["bundle"],
            "bundle_sha256": selected_checkpoint["bundle_sha256"],
            "checkpoint_sha256": selected_checkpoint["checkpoint_sha256"],
        },
        "selection": selection,
        "training_plan_sha256": digest(a.ppo_root / "PLAN.json"),
        "selection_plan_sha256": plan_sha,
        "development_result_sha256": digest(
            a.development_root / "QUICK_RESULT.json"),
        "training_instances": 200,
        "selection_instances": 20,
        "final_test_opened": False,
    }
    freeze["manifest_digest"] = canonical_digest(freeze)
    write_json(root / "FROZEN_FORMAL_METHODS.json", freeze)
    result = {
        "completed": True,
        "validation": selection,
        "frozen_methods": freeze,
        "formal_methods_ready": True,
        "new_ppo_updates": 8,
        "new_validation_searches": len(rows),
        "new_objective_evaluations": len(rows) * 400,
        "final_test_opened": False,
        "elapsed_seconds_this_invocation": time.monotonic() - started,
        "next_action": (
            "use BASE, directed heuristic, frozen supervised Transformer and "
            "validation-selected Transformer-PPO in the predeclared formal experiment"),
    }
    write_json(root / "COMPLETE.json", {
        **result, "validation_runs": rows})
    write_json(root / "QUICK_RESULT.json", result)
    write_json(root / "STATUS.json", {
        "stage": "complete",
        "selected_update": selected_update,
        "completed_searches": completed,
        "final_test_opened": False,
    })
    (root / "PIPELINE_COMPLETE").write_text(
        f"selected_ppo_update={selected_update}\nfinal_test_opened=false\n")
    log(f"COMPLETE selected PPO update={selected_update}; "
        "final test remains locked")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path,
        default=SOURCE / "data/data_expanded.xlsx")
    for name in ("generated-instances", "curve-root", "development-root",
                 "normalization", "ppo-root", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--workers", choices=(1, 2, 3, 4), type=int, default=3)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--instance", type=Path)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--method")
    parser.add_argument("--policy-bundle", type=Path)
    parser.add_argument("--plan-sha")
    a = parser.parse_args()
    a.out = a.out.resolve()
    if a.worker:
        if any(getattr(a, name) is None for name in
               ("instance", "seed", "method", "policy_bundle", "plan_sha")):
            parser.error("worker arguments incomplete")
        a.out.mkdir(parents=True, exist_ok=False)
        run_worker(a)
        return
    a.out.mkdir(parents=True, exist_ok=True)
    with (a.out / ".lock").open("w") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            execute(a)
        except Exception as exc:
            previous = (read_json(a.out / "STATUS.json")
                        if (a.out / "STATUS.json").exists() else {})
            write_json(a.out / "STATUS.json", {
                **previous, "stage": "stopped", "error": str(exc)})
            log(f"STOPPED {exc}")
            raise


if __name__ == "__main__":
    main()
