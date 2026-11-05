#!/usr/bin/env python3
"""Bounded collection -> matched fits -> 120 searches. CCP100/pop20/B400."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

import numpy as np
import torch

from m_candidate_information import (CandidatePolicy, CandidateController, METHODS,
    TRAIN_INSTANCES, VALID_INSTANCE, TEST_INSTANCES, DATA_SEEDS, TRAIN_SEEDS, TEST_SEEDS,
    THRESHOLD_SCALED, MAX_STATES)
from m_operator_offline import digest, read_json, write_json
from m_operator_fixed_mix import bootstrap
from run_learning_components_deterministic import _normalizer
from run_m_operator_closed_loop_v1 import load_run
from transformer_ppo.instance_catalog import read_instance
from transformer_ppo.episode import run_episode

SOURCE = Path(__file__).resolve().parent


def log(message):
    print(f"[M-CAND] {message}", flush=True)


def worker(a):
    torch.set_num_threads(1)
    info, _ = read_instance(a.instance)
    policy = CandidatePolicy(info["instance_id"], a.seed, a.method, a.bundle, a.collect)
    holders = []

    def factory(*args, **kwargs):
        controller = CandidateController(*args, **kwargs)
        holders.append(controller)
        return controller

    result = run_episode(a.data, a.instance, a.out, policy, _normalizer(a.normalization), a.seed,
        evaluation_budget=400, pop_size=20, generations=1000, evaluation_mode="ccp", component="M",
        scenario_count=100, ccp_confidence=.9, uncertainty_profile="transport_border_saa_v1",
        inference_controller_factories={"M": factory})
    if result["real_objective_evaluation_count"] != 400 or result["ppo_updates"] or result["counterfactual_evaluation_count"]:
        raise RuntimeError("requires exactly B400 and no PPO; training feedback is counted separately")
    if not a.collect and result["training_feedback_evaluations"]:
        raise RuntimeError("search must evaluate only the selected candidate")
    if not result["selected_candidate_execution_checks"]:
        raise RuntimeError("no candidate was executed after G10")
    result.update(strategy=a.method, algorithm_seed=a.seed, plan_sha256=a.plan_sha,
                  model_bundle_sha256=digest(a.bundle) if a.bundle else None)
    if a.collect:
        rows = holders[0].feedback_rows
        if not rows or len(rows) > MAX_STATES:
            raise RuntimeError("no usable feedback or collection cap exceeded")
        write_json(a.out / "FEEDBACK.json", {"rows": rows, "feedback_evaluations": result["training_feedback_evaluations"]})
        result["feedback_sha256"] = digest(a.out / "FEEDBACK.json")
    result["total_objective_evaluations_including_training_feedback"] = (
        result["real_objective_evaluation_count"] + result["training_feedback_evaluations"])
    write_json(a.out / "RUN_COMPLETE.json", result)


def frozen_plan(a):
    inputs = [a.data.resolve(), a.normalization.resolve(), SOURCE / "data/border_crossing_events.csv"]
    for name in TRAIN_INSTANCES + (VALID_INSTANCE,) + TEST_INSTANCES:
        path = (a.instances / f"{name}.json").resolve()
        record, _ = read_instance(path)
        if record["instance_id"] != name or record["source_data_sha256"] != digest(a.data):
            raise ValueError(f"instance identity/workbook mismatch: {name}")
        inputs.append(path)
    _normalizer(a.normalization)
    inputs += sorted(SOURCE.glob("*.py")) + sorted((SOURCE / "transformer_ppo").glob("*.py"))
    return {"protocol": "candidate-information-v1", "role": "bounded development diagnosis; not formal/OOS evidence",
        "scenario_count": 100, "ccp_confidence": .9, "population": 20, "evaluation_budget": 400,
        "minimum_control_generation": 10, "methods": METHODS,
        "training_instances": TRAIN_INSTANCES, "validation_instance": VALID_INSTANCE,
        "data_seeds": DATA_SEEDS, "collection_runs": 10, "maximum_feedback_states_per_run": MAX_STATES,
        "maximum_feedback_evaluations": 10*MAX_STATES*5, "collection_state_selection": "first at most 32 supported G10+ mutation events per trajectory, without looking at outcomes",
        "training_seeds": TRAIN_SEEDS, "epochs": 20, "threshold_scaled": THRESHOLD_SCALED,
        "test_instances": TEST_INSTANCES, "test_seeds": TEST_SEEDS, "search_runs": 120,
        "workers": a.workers, "threads_per_worker": 1, "per_process_timeout_seconds": 300,
        "invocation_timeout_seconds": 1800, "automatic_retry_count": 0, "new_ppo_updates": 0,
        "candidate_contract": "concrete-proposal-v1; each operator retains its own legal-target distribution; freeze full repair-inclusive encoding before choice; selected encoding is executed without redraw",
        "search_information": "nominal encoding/path attributes only; no CCP evaluation of unselected candidates",
        "control": "same proposals, labels, splits, scaler, network shape, initial weights and optimizer schedule; STATE_ONLY zeros the specific-feature columns after scaling",
        "gate": "fixed positive threshold 0.02*training-target-RMS; no tuning; numerical correctness only, no offline performance gate for this bounded diagnostic",
        "primary_metric": "paired percent (STATE_PLUS_CANDIDATE HV / STATE_ONLY HV - 1), separately per instance; additionally compare both with Balanced",
        "decision": "all two instance mean effects >0 AND pooled equal-instance bootstrap lower >0 vs both STATE_ONLY and Balanced -> promising for new-instance confirmation; otherwise stop this branch; no automatic extension/PPO",
        "bootstrap": "10000 fixed-seed (897900) resamples of 20 paired seeds within each fixed instance; conditional development intervals only",
        "timing": "all candidate construction and inference included; two concurrent workers, not isolated formal timing",
        "source_sha256": {str(p): digest(p) for p in inputs}, "torch": str(torch.__version__), "numpy": str(np.__version__)}


def summarize(runs):
    groups = {}
    for row in runs:
        key = row["instance"], row["algorithm_seed"]
        methods = groups.setdefault(key, {})
        if row["strategy"] in methods:
            raise ValueError("duplicate search run")
        methods[row["strategy"]] = row
    if set(groups) != {(name, seed) for name in TEST_INSTANCES for seed in TEST_SEEDS}:
        raise ValueError("missing predeclared test pairs")
    rng = np.random.default_rng(897900)
    per_instance, pair_records, pooled = [], [], {"candidate_vs_state_only": [], "candidate_vs_balanced": []}
    for instance in TEST_INSTANCES:
        by_method = {m: [] for m in METHODS}
        comparisons = {"candidate_vs_state_only": [], "candidate_vs_balanced": [], "state_only_vs_balanced": []}
        for seed in TEST_SEEDS:
            methods = groups[instance, seed]
            if set(methods) != set(METHODS):
                raise ValueError("each comparison requires all three methods")
            h, s, c = [methods[m] for m in METHODS]
            for row in (h, s, c):
                if (row["pairing"] != h["pairing"] or row["real_objective_evaluation_count"] != 400
                    or row["ppo_updates"] or row["training_feedback_evaluations"]
                    or row["candidate_contract"] != "concrete-proposal-v1"
                    or row["minimum_control_generation"] != 10
                    or row["model_bundle_sha256"] != h["model_bundle_sha256"]):
                    raise ValueError("search pairing/budget/model/no-feedback contract violated")
                if not np.isfinite(row["final_hypervolume"]) or row["final_hypervolume"] <= 0:
                    raise ValueError("relative HV requires positive finite baseline HV")
                by_method[row["strategy"]].append(row)
            effects = {"candidate_vs_state_only": 100*(c["final_hypervolume"]/s["final_hypervolume"]-1),
                "candidate_vs_balanced": 100*(c["final_hypervolume"]/h["final_hypervolume"]-1),
                "state_only_vs_balanced": 100*(s["final_hypervolume"]/h["final_hypervolume"]-1)}
            for key, value in effects.items():
                comparisons[key].append(value)
            pair_records.append({"instance": instance, "seed": seed, **effects})
        indices = rng.integers(0, len(TEST_SEEDS), size=(10000, len(TEST_SEEDS)))
        report = {"instance": instance, "paired_seeds": len(TEST_SEEDS), "methods": {}}
        for method, rows in by_method.items():
            report["methods"][method] = {"mean_hv": float(np.mean([r["final_hypervolume"] for r in rows])),
                "mean_seconds": float(np.mean([r["optimisation_seconds"] for r in rows])),
                "mean_candidate_construction_seconds": float(np.mean([r["candidate_construction_seconds"] for r in rows])),
                "switches": sum(r["closed_loop_switches"] for r in rows)}
        for key, values in comparisons.items():
            report[key] = bootstrap(values, indices)
            if key in pooled:
                pooled[key].append(np.asarray(values)[indices].mean(1))
        per_instance.append(report)
    intervals = {key: {"mean": float(np.mean([r[key]["mean"] for r in per_instance])),
                      "low": float(np.quantile(np.mean(values, axis=0), .025)),
                      "high": float(np.quantile(np.mean(values, axis=0), .975))} for key, values in pooled.items()}
    promising = all(intervals[key]["low"] > 0 and all(r[key]["mean"] > 0 for r in per_instance) for key in pooled)
    return {"per_instance": per_instance, "pooled_equal_instance_comparison": intervals,
        "paired_runs": pair_records, "candidate_information_promising": promising,
        "conclusion": "PROMISING_REQUIRES_NEW_INSTANCES" if promising else "STOP_THIS_BRANCH_NO_RELIABLE_GAIN_ESTABLISHED",
        "qualification": "Failure is not proof of no effect. Seed variation is conditional on these two development instances and frozen scenarios. No claim of PPO benefit.",
        "automatic_extension": False, "automatic_ppo": False}


def execute(a):
    root = a.out.resolve()
    # JSON round trip normalizes tuples before equality checks on resume.
    import json
    plan = json.loads(json.dumps(frozen_plan(a)))
    if (root / "PLAN.json").exists() and read_json(root / "PLAN.json") != plan:
        raise ValueError("frozen inputs/code/design changed; use a new output root")
    write_json(root / "PLAN.json", plan)
    plan_sha = digest(root / "PLAN.json")
    if (root / "PIPELINE_COMPLETE").exists():
        log("already complete; inspect QUICK_RESULT.json")
        return
    deadline = time.monotonic()+1800
    stop, status_lock = threading.Event(), threading.Lock()
    env = {**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    (root / "logs").mkdir(exist_ok=True)

    def launch(command, label):
        if stop.is_set() or time.monotonic() >= deadline:
            raise RuntimeError("invocation stopped/time cap reached")
        path = root / "logs" / f"{label}_{time.time_ns()}.log"
        log(f"start {label}; log={path}")
        with path.open("w") as handle, subprocess.Popen(command, cwd=SOURCE, env=env,
                stdout=handle, stderr=subprocess.STDOUT) as process:
            start = last = time.monotonic()
            try:
                while process.poll() is None:
                    now = time.monotonic()
                    if stop.is_set() or now >= deadline or now-start > 300:
                        raise TimeoutError(f"bounded run stopped; log={path}")
                    if now-last >= 15:
                        log(f"running {label} pid={process.pid} elapsed={now-start:.0f}s")
                        last = now
                    time.sleep(.2)
            finally:
                if process.poll() is None:
                    process.kill()
                    process.wait()
            if process.returncode:
                raise RuntimeError(f"{label} exit={process.returncode}; log={path}\n" + "\n".join(path.read_text(errors="replace").splitlines()[-20:]))

    def run_job(instance, seed, method, collecting=False):
        folder = root / ("collection" if collecting else "search") / instance / f"seed_{seed}" / method
        bundle = root / "training/BUNDLE.json"
        if not (folder / "RUN_COMPLETE.json").exists():
            if folder.exists() and any(folder.iterdir()):
                archived = root / "interrupted_runs" / f"{instance}_{seed}_{method}_{time.time_ns()}"
                archived.parent.mkdir(exist_ok=True)
                folder.rename(archived)
            command = [sys.executable, "-u", str(Path(__file__).resolve()), "--worker", "--instance",
                str((a.instances / f"{instance}.json").resolve()), "--data", str(a.data.resolve()),
                "--normalization", str(a.normalization.resolve()), "--out", str(folder),
                "--seed", str(seed), "--method", method, "--plan-sha", plan_sha]
            command += ["--collect"] if collecting else ["--bundle", str(bundle)]
            launch(command, f"{'data' if collecting else 'search'}_{instance}_{seed}_{method}")
        row = load_run(folder, instance, plan_sha)
        if row["strategy"] != method or row["algorithm_seed"] != seed:
            raise ValueError("saved run identity mismatch")
        if not collecting and row["model_bundle_sha256"] != digest(bundle):
            raise ValueError("frozen model bundle changed")
        if collecting and row.get("feedback_sha256") != digest(folder / "FEEDBACK.json"):
            raise ValueError("saved training feedback changed")
        return row, folder

    def parallel(jobs, stage):
        done = []
        with ThreadPoolExecutor(max_workers=a.workers) as pool:
            futures = [pool.submit(run_job, *job) for job in jobs]
            try:
                for future in as_completed(futures):
                    value = future.result()
                    done.append(value)
                    with status_lock:
                        write_json(root / "STATUS.json", {"stage": stage, "completed": len(done), "total": len(jobs)})
                    log(f"{stage} {len(done)}/{len(jobs)} complete {value[0]['instance']}/{value[0]['algorithm_seed']}/{value[0]['strategy']}")
            except BaseException:
                stop.set()
                for future in futures:
                    future.cancel()
                raise
        return done

    log("STAGE 1: at most 320 states / 1600 extra training evaluations; no learning")
    data_runs = parallel([(name, seed, METHODS[0], True) for name in TRAIN_INSTANCES+(VALID_INSTANCE,) for seed in DATA_SEEDS], "collection")
    rows, feedback = [], 0
    for result, folder in sorted(data_runs, key=lambda item: (item[0]["instance"], item[0]["algorithm_seed"])):
        value = read_json(folder / "FEEDBACK.json")
        if (not 1 <= len(value["rows"]) <= MAX_STATES
            or value["feedback_evaluations"] != result["training_feedback_evaluations"]
            or value["feedback_evaluations"] != sum(sum(r["eligible"]) for r in value["rows"])
            or any(r["instance"] != result["instance"] or r["algorithm_seed"] != result["algorithm_seed"] for r in value["rows"])):
            raise ValueError("training feedback identity/completeness mismatch")
        rows.extend(value["rows"])
        feedback += value["feedback_evaluations"]
    dataset = root / "DATASET.json"
    write_json(dataset, {"protocol": "candidate-information-v1", "quality_pass": True,
        "plan_sha256": plan_sha, "rows": rows, "training_feedback_evaluations": feedback,
        "collection_search_evaluations": sum(row["real_objective_evaluation_count"] for row, _ in data_runs)})
    log(f"STAGE 2: same {len(rows)} states / two information arms / 3 matched seeds / 20 epochs")
    bundle = root / "training/BUNDLE.json"
    if bundle.exists():
        if read_json(bundle)["dataset_sha256"] != digest(dataset):
            raise ValueError("training data changed since frozen fit")
    else:
        launch([sys.executable, "-u", str(SOURCE / "train_m_candidate_information.py"), "--dataset", str(dataset),
                "--out", str(root / "training")], "training")
    log("STAGE 3: 120 complete searches; no offline-performance gate, no online training")
    jobs = []
    for index, seed in enumerate(TEST_SEEDS):
        order = METHODS[index % 3:] + METHODS[:index % 3]
        for name in TEST_INSTANCES:
            jobs.extend((name, seed, method, False) for method in order)
    runs = [row for row, _ in parallel(jobs, "search")]
    result = summarize(runs)
    pairs = result.pop("paired_runs")
    training = read_json(bundle)
    report = {"completed": True, "plan_sha256": plan_sha, "design": plan,
        "training_feedback_states": len(rows), "training_feedback_evaluations": feedback,
        "collection_search_evaluations": sum(row["real_objective_evaluation_count"] for row, _ in data_runs),
        "test_search_evaluations": sum(row["real_objective_evaluation_count"] for row in runs),
        "training_summary": {"training_states": training["training_states"],
            "validation_states": training["validation_states"], "paired_initialization_exact": training["paired_initialization_exact"],
            "threshold_scaled": training["threshold_scaled"], "threshold_raw": training["threshold_raw"],
            "selected_epochs": {m: [h["selected_epoch"] for h in hs] for m, hs in training["histories"].items()},
            "offline_validation": training["offline"], "ppo_updates": 0},
        "model_bundle_sha256": digest(bundle), **result}
    write_json(root / "COMPLETE.json", {**report, "runs": runs, "paired_runs": pairs})
    write_json(root / "QUICK_RESULT.json", report)
    table = []
    for item in report["per_instance"]:
        for method, stats in item["methods"].items():
            table.append({"instance": item["instance"], "method": method,
                "mean_hv": stats["mean_hv"], "mean_seconds": stats["mean_seconds"],
                "paired_percent_vs_balanced": 0. if method == METHODS[0] else item[
                    "state_only_vs_balanced" if method == METHODS[1] else "candidate_vs_balanced"]["mean"]})
    with (root / "RESULT_TABLE.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    write_json(root / "STATUS.json", {"stage": "complete", "conclusion": report["conclusion"]})
    (root / "PIPELINE_COMPLETE").write_text("bounded candidate-information experiment complete; no automatic PPO\n")
    log("COMPLETE " + report["conclusion"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--collect", action="store_true")
    for key in ("instances", "instance", "bundle"):
        parser.add_argument("--"+key, type=Path)
    parser.add_argument("--data", type=Path, default=SOURCE / "data/data_expanded.xlsx")
    parser.add_argument("--normalization", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--method", choices=METHODS)
    parser.add_argument("--plan-sha")
    parser.add_argument("--workers", type=int, choices=(1, 2), default=2)
    a = parser.parse_args()
    if a.worker:
        if a.instance is None or a.seed is None or a.method is None or not a.plan_sha:
            parser.error("worker requires instance, seed, method, plan-sha")
        worker(a)
        return
    if a.instances is None:
        parser.error("coordinator requires --instances")
    a.out.mkdir(parents=True, exist_ok=True)
    with (a.out / ".lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            execute(a)
        except Exception as exc:
            write_json(a.out / "STATUS.json", {"stage": "failed", "error": str(exc)})
            log(f"FAILED {exc}")
            raise


if __name__ == "__main__":
    main()
