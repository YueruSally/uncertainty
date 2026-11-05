#!/usr/bin/env python3
"""Bounded 18-run development comparison; frozen policy, CCP100/pop20/B400."""
from __future__ import annotations

import argparse
import csv
import fcntl
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

import numpy as np
import torch

from m_operator_closed_loop import export_frozen_policy
from m_operator_offline import digest, read_json, write_json
from run_m_frozen_operator_episode import METHODS

INSTANCES = ("train-S0-021", "train-S0-022")
SEEDS = (895001, 895002, 895003)


def log(message):
    print(f"[M-CLOSED] {message}", flush=True)


def paired_results(runs):
    """Validate pairing before making any performance claim."""
    groups = {}
    for row in runs:
        key = row["instance"], row["algorithm_seed"]
        if row["strategy"] in groups.setdefault(key, {}):
            raise ValueError("duplicate paired run")
        groups[key][row["strategy"]] = row
    for key, methods in groups.items():
        if set(methods) != set(METHODS):
            raise ValueError(f"incomplete methods for {key}")
        baseline = methods["BASE"]
        if baseline["final_hypervolume"] <= 0:
            raise ValueError("cannot report relative improvement against zero Base HV")
        for row in methods.values():
            if row["pairing"] != baseline["pairing"]:
                raise ValueError(f"scenario/configuration/initial-population pairing mismatch: {key}")
            if row["evaluation_budget"] != 400 or not 396 <= row["real_objective_evaluation_count"] <= 400:
                raise ValueError("evaluation budget mismatch")
            if row["ppo_updates"] or row.get("counterfactual_evaluation_count", 0):
                raise ValueError("unexpected online training or counterfactual evaluation")
            if not math.isfinite(row["final_hypervolume"]):
                raise ValueError("non-finite final HV")
            row["paired_percent_vs_base"] = 100 * (row["final_hypervolume"] / baseline["final_hypervolume"] - 1)
    summaries = [{"method": method, "runs": len(groups),
        "mean_final_hv": statistics.fmean(g[method]["final_hypervolume"] for g in groups.values()),
        "mean_paired_percent_vs_base": statistics.fmean(g[method]["paired_percent_vs_base"] for g in groups.values()),
        "mean_optimisation_seconds": statistics.fmean(g[method]["optimisation_seconds"] for g in groups.values()),
        "mean_worker_seconds_including_policy_load": statistics.fmean(g[method]["worker_seconds"] for g in groups.values())}
        for method in METHODS]
    per_instance = []
    for instance in sorted({key[0] for key in groups}):
        rows = [g for key, g in groups.items() if key[0] == instance]
        differences = [g["TRANSFORMER_PRETRAINED"]["paired_percent_vs_base"] - g["HEURISTIC_BALANCED"]["paired_percent_vs_base"] for g in rows]
        per_instance.append({"instance": instance,
            "transformer_minus_heuristic_percentage_points": statistics.fmean(differences),
            "transformer_switches": sum(g["TRANSFORMER_PRETRAINED"].get("closed_loop_switches", 0) for g in rows)})
    positive = sum(r["transformer_minus_heuristic_percentage_points"] > 1e-8 for r in per_instance)
    negative = sum(r["transformer_minus_heuristic_percentage_points"] < -1e-8 for r in per_instance)
    switches = sum(r["transformer_switches"] for r in per_instance)
    passed = positive == len(per_instance) and switches > 0
    next_action = ("bounded development extension may be considered; no automatic PPO" if passed else
        "pause current frozen policy: no switches" if switches == 0 else
        "pause current frozen policy: both instances worse" if negative == len(per_instance) else
        "mixed/tied: at most one predeclared extension, with unchanged model; do not train automatically")
    return {"methods": summaries, "per_instance": per_instance, "screen_pass": passed,
            "next_action": next_action, "total_transformer_switches": switches}


def load_run(folder, instance, plan_sha):
    row = read_json(folder / "RUN_COMPLETE.json")
    if row["plan_sha256"] != plan_sha:
        raise ValueError("run belongs to a different frozen plan")
    config = read_json(folder / "configuration.json")
    expected = {"evaluation_mode": "ccp", "population": 20, "training_size": 100,
                "evaluation_budget": 400, "alpha": .9}
    if any(config.get(k) != v for k, v in expected.items()):
        raise ValueError("run is not CCP100/pop20/B400/q90")
    row["instance"] = instance
    row["pairing"] = {key: config[key] for key in (
        "instance_digest", "population", "generations", "evaluation_budget", "training_size",
        "alpha", "scenario_digest", "normalization", "uncertainty", "operator_probabilities",
        "crossover_rate", "mutation_rate")}
    row["pairing"]["algorithm_seed"] = config["seeds"]["algorithm"]
    row["pairing"]["initial_population"] = row["initial_population_fingerprints"]
    return row


def execute(a):
    root, source = a.out.resolve(), Path(__file__).resolve().parent
    bundle = root / "FROZEN_POLICY.json"
    frozen = export_frozen_policy(a.path_root.resolve(), a.collection_root.resolve(), bundle)
    if set(INSTANCES) & set(frozen["source_instances"]):
        raise ValueError("closed-loop instances overlap offline model data")
    collection_plan = read_json(a.collection_root / "PLAN.json")
    for name, path in (("normalization", a.normalization), ("workbook", a.data)):
        if digest(path) != collection_plan["inputs"][name]["sha256"]:
            raise ValueError(f"{name} differs from the frozen model's source experiment")
    files = [a.data.resolve(), a.normalization.resolve(), bundle]
    files += [(a.instances / (name + ".json")).resolve() for name in INSTANCES]
    files += sorted(source.glob("*.py"))
    files += sorted((source / "transformer_ppo").glob("*.py"))
    files += [source / "data/border_crossing_events.csv"]
    plan = {"protocol": "frozen-m-closed-loop-v1", "instances": list(INSTANCES), "seeds": list(SEEDS),
        "methods": list(METHODS), "run_count": 18, "scenario_count": 100, "ccp_confidence": .9,
        "population_size": 20, "evaluation_budget": 400, "maximum_generations": 1000,
        "minimum_control_generation": 10, "execution_contract": "operator_shared_v1",
        "workers": 1, "threads_per_worker": 1, "per_run_timeout_seconds": 600,
        "invocation_time_limit_seconds": 3600, "automatic_retry_count": 0,
        "new_training_steps": 0, "counterfactual_evaluations": 0,
        "torch": str(torch.__version__), "numpy": str(np.__version__),
        "inputs_sha256": {str(p): digest(p) for p in files},
        "pass_rule": "both instance means of paired Transformer-minus-heuristic points > 1e-8, with at least one switch",
        "role": "development screen, not a formal or independent test; no model selection on search outcomes",
        "baseline_note": "all methods rerun; heuristic uses same current shared operator executor as Transformer",
        "time_note": "serial single-thread workers; optimisation includes setup, worker time additionally includes policy loading; offline training excluded"}
    if (root / "PLAN.json").exists() and read_json(root / "PLAN.json") != plan:
        raise ValueError("inputs/code/plan changed: preserve results and use a new output directory")
    write_json(root / "PLAN.json", plan)
    plan_sha = digest(root / "PLAN.json")
    if (root / "PIPELINE_COMPLETE").exists():
        log("already complete; see QUICK_RESULT.json")
        return
    log(f"frozen fold={frozen['selected']['source_fold']} threshold={frozen['selected']['frozen_threshold']}; 18 runs, no training; serial/1 thread")
    started, runs = time.monotonic(), []
    env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "PYTHONUNBUFFERED": "1"}
    for instance in INSTANCES:
        for si, seed in enumerate(SEEDS):
            # Rotate resource/timing order, never select/reorder based on outcomes.
            order = METHODS[si:] + METHODS[:si]
            for method in order:
                folder = root / instance / f"seed_{seed}" / method
                if not (folder / "RUN_COMPLETE.json").exists():
                    if folder.exists() and any(folder.iterdir()):
                        quarantine = root / "interrupted_runs" / instance / f"seed_{seed}" / f"{method}_{time.time_ns()}"
                        quarantine.parent.mkdir(parents=True, exist_ok=True)
                        folder.rename(quarantine)
                        log(f"preserved incomplete output: {quarantine}")
                    remaining = 3600 - (time.monotonic() - started)
                    if remaining <= 0:
                        raise TimeoutError("one-hour run limit reached; completed searches are retained")
                    filename = root / "logs" / f"{instance}_{seed}_{method}_{time.time_ns()}.log"
                    filename.parent.mkdir(exist_ok=True)
                    command = [sys.executable, "-u", str(source / "run_m_frozen_operator_episode.py"),
                        "--data", str(a.data.resolve()), "--instance", str((a.instances / (instance + ".json")).resolve()),
                        "--normalization", str(a.normalization.resolve()), "--bundle", str(bundle),
                        "--out", str(folder), "--seed", str(seed), "--method", method, "--plan-sha", plan_sha]
                    log(f"start {len(runs)+1}/18 {instance} seed={seed} {method} log={filename}")
                    with filename.open("w") as handle:
                        with subprocess.Popen(command, cwd=source, env=env, stdout=handle, stderr=subprocess.STDOUT) as proc:
                            run_started, last_heartbeat = time.monotonic(), time.monotonic()
                            while proc.poll() is None:
                                if time.monotonic() - run_started >= min(600, remaining):
                                    proc.kill()
                                    proc.wait()
                                    raise TimeoutError(f"run timeout; log={filename}")
                                if time.monotonic() - last_heartbeat >= 20:
                                    log(f"running {len(runs)+1}/18 pid={proc.pid} seconds={time.monotonic()-run_started:.0f}")
                                    last_heartbeat = time.monotonic()
                                time.sleep(.2)
                            if proc.returncode:
                                raise RuntimeError(f"worker exit={proc.returncode}; log={filename}")
                row = load_run(folder, instance, plan_sha)
                runs.append(row)
                write_json(root / "STATUS.json", {"stage": "search", "completed_runs": len(runs), "total_runs": 18,
                    "last_run": str(folder), "elapsed_seconds": time.monotonic() - started})
                log(f"complete {len(runs)}/18 {method} HV={row['final_hypervolume']:.7g} seconds={row['optimisation_seconds']:.1f} switches={row.get('closed_loop_switches',0)}")
                if method == "TRANSFORMER_PRETRAINED" and not (root / "TIMING_FIRST_RUN.json").exists():
                    write_json(root / "TIMING_FIRST_RUN.json", {"run": str(folder), "part_of_18_runs": True,
                        "policy_loading_seconds": row["policy_loading_seconds"], **row["timing_seconds"]})
                    log(f"timing {row['timing_seconds']}")
            paired_results([r for r in runs if r["instance"] == instance and r["algorithm_seed"] == seed])
            log(f"pair verified {instance} seed={seed}: identical initial population/scenarios; common B400")
    report = {"completed": True, "design": plan, "pairing_exact": True,
        "new_training_steps": 0, "counterfactual_evaluations": 0,
        "scope": "complete searches under small B400 budget; no PPO and no formal generalization claim",
        **paired_results(runs), "elapsed_seconds_this_invocation": time.monotonic() - started}
    write_json(root / "COMPLETE.json", {"runs": runs, **report})
    write_json(root / "QUICK_RESULT.json", report)
    with (root / "RESULT_TABLE.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(report["methods"][0]))
        writer.writeheader()
        writer.writerows(report["methods"])
    write_json(root / "STATUS.json", {"stage": "complete", "completed_runs": 18, "total_runs": 18,
        "screen_pass": report["screen_pass"]})
    (root / "PIPELINE_COMPLETE").write_text("completed all 18 searches; see screen_pass separately\n")
    log("COMPLETE screen=" + ("PASSED" if report["screen_pass"] else "DID NOT PASS"))
    for row in report["methods"]:
        log(f"{row['method']}: meanHV={row['mean_final_hv']:.7g} paired={row['mean_paired_percent_vs_base']:+.3f}% time={row['mean_optimisation_seconds']:.2f}s")
    log(report["next_action"])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("instances", "normalization", "path-root", "collection-root", "out"):
        p.add_argument("--" + key, type=Path, required=True)
    p.add_argument("--data", type=Path, default=Path(__file__).resolve().parent / "data/data_expanded.xlsx")
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    with (a.out / ".lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            execute(a)
        except Exception as exc:
            previous = read_json(a.out / "STATUS.json") if (a.out / "STATUS.json").exists() else {}
            write_json(a.out / "STATUS.json", {**previous, "stage": "failed", "error": str(exc)})
            log(f"FAILED {exc}")
            raise


if __name__ == "__main__":
    main()
