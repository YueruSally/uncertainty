#!/usr/bin/env python3
"""Locked four-method K8 formal test with independent OOS5000 re-evaluation."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import csv
import fcntl
import hashlib
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import threading
import time

SOURCE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE))

import numpy as np
import torch

import baseline_uncertainty as base
from evaluate_m_directed_balanced_oos import (
    _evaluate_front, _normalizer as oos_normalizer, _uncertainty_profile,
    derive_common_oos_reference)
from m_operator_closed_loop import ProfiledBase
from m_operator_offline import digest, read_json, write_json
from m_prescreen_scale_policy import (FrozenScaleRanker,
    ScaleSearchController, ScaleSearchPolicy)
from run_ev_ccp_oos_pilot import scenario_digest
from run_learning_components_deterministic import _normalizer
from run_m_operator_closed_loop_v1 import load_run
from transformer_ppo.archive import exact_hv_3d
from transformer_ppo.episode import run_episode
from transformer_ppo.instance_catalog import canonical_digest, read_instance


INSTANCES = tuple(f"final-test-S0-{index:03d}" for index in range(1, 11))
SEEDS = tuple(range(910201, 910211))
BASE = "BASE"
BALANCED = "HEURISTIC_BALANCED"
SUPERVISED = "TRANSFORMER_K8_SIZE200"
OOS_SCENARIO_COUNT = 5000


def log(message):
    print(f"[M-K8-FORMAL] {message}", flush=True)


def frozen_methods(manifest):
    ppo = manifest["transformer_ppo"]["method"]
    if (manifest.get("protocol") != "m-prescreen-formal-method-freeze-v1"
            or manifest.get("final_test_opened") is not False
            or manifest.get("training_instances") != 200
            or manifest.get("selection_instances") != 20
            or manifest["transformer_ppo"].get("ppo_update") not in (2, 4, 8)
            or not ppo.startswith("TRANSFORMER_K8_PPO_U")):
        raise ValueError("formal method freeze is invalid")
    identity = dict(manifest); claimed = identity.pop("manifest_digest", None)
    if canonical_digest(identity) != claimed:
        raise ValueError("formal method freeze digest mismatch")
    if manifest["supervised"]["method"] != SUPERVISED:
        raise ValueError("supervised formal method changed")
    return (BASE, BALANCED, SUPERVISED, ppo)


def validate_run(row, expected_method):
    if (row.get("strategy") != expected_method
            or row.get("evaluation_budget") != 400
            or not 396 <= row.get("real_objective_evaluation_count", -1) <= 400
            or row.get("ppo_updates", 0)
            or row.get("training_feedback_evaluations", 0)
            or row.get("counterfactual_evaluation_count", 0)
            or row.get("candidate_audit_evaluations", 0)
            or row.get("stop_reason") != "evaluation_budget"
            or not math.isfinite(row.get("final_hypervolume", math.nan))
            or row.get("final_hypervolume", 0) <= 0):
        raise ValueError("invalid frozen formal search")
    if expected_method != BASE:
        audit = row.get("k8_scale_search", {})
        if (audit.get("method") != expected_method
                or audit.get("hidden_candidate_objective_evaluations") != 0
                or audit.get("pool_events", 0) < 1):
            raise ValueError("invalid K8 formal method audit")


def run_worker(a):
    torch.set_num_threads(1)
    info, _ = read_instance(a.instance)
    holders = []
    if a.method == BASE:
        policy = None
        factories = {"BASE": ProfiledBase}
        component = "BASE"
    else:
        predictor = None
        if a.method != BALANCED:
            predictor = FrozenScaleRanker(read_json(a.policy_bundle), digest)
        policy = ScaleSearchPolicy(info["instance_id"], a.seed, a.method,
                                   predictor=predictor)

        def factory(*args, **kwargs):
            controller = ScaleSearchController(*args, **kwargs)
            holders.append(controller)
            return controller

        factories = {"M": factory}; component = "M"
    result = run_episode(a.data, a.instance, a.out, policy,
        _normalizer(a.normalization), a.seed, evaluation_budget=400,
        pop_size=20, generations=1000, evaluation_mode="ccp",
        component=component, scenario_count=100, ccp_confidence=.9,
        uncertainty_profile="transport_border_saa_v1",
        inference_controller_factories=factories)
    if a.method != BASE and len(holders) != 1:
        raise RuntimeError("formal K8 controller was not installed once")
    result.update(strategy=a.method, algorithm_seed=a.seed,
        plan_sha256=a.plan_sha, new_training_steps=0, new_ppo_updates=0,
        learning_enabled=False, role="locked final formal comparison")
    if a.policy_bundle is not None:
        result["policy_bundle_sha256"] = digest(a.policy_bundle)
    validate_run(result, a.method)
    write_json(a.out/"RUN_COMPLETE.json", result)


def hierarchical_interval(values, seed=98300000):
    names = sorted(values)
    arrays = [np.asarray(values[name], dtype=float) for name in names]
    lengths = {len(array) for array in arrays}
    if len(lengths) != 1 or not arrays:
        raise ValueError("hierarchical bootstrap requires a balanced grid")
    matrix = np.stack(arrays); rng = np.random.default_rng(seed)
    instance_indices = rng.integers(0, len(names), (50_000, len(names)))
    run_count = next(iter(lengths))
    run_indices = rng.integers(
        0, run_count, (50_000, len(names), run_count))
    sampled = matrix[instance_indices[:, :, None], run_indices]
    # Preserve both levels of the balanced design: resample instances, then
    # resample all paired algorithm-seed runs within each sampled instance.
    means = sampled.mean((1, 2))
    return [float(np.quantile(means, .025)),
            float(np.quantile(means, .975))]


def paired_summary(records, value_key, seed):
    grouped = {}
    for row in records:
        grouped.setdefault(row["instance"], []).append(float(row[value_key]))
    values = [value for rows in grouped.values() for value in rows]
    interval = hierarchical_interval(grouped, seed)
    return {"paired_runs": len(values),
        "equal_instance_mean_percent": statistics.fmean(
            statistics.fmean(rows) for rows in grouped.values()),
        "hierarchical_bootstrap_95ci_low": interval[0],
        "hierarchical_bootstrap_95ci_high": interval[1],
        "wins": sum(value > 1e-12 for value in values),
        "ties": sum(abs(value) <= 1e-12 for value in values),
        "losses": sum(value < -1e-12 for value in values),
        "per_instance": [{"instance": name, "runs": len(grouped[name]),
            "mean_percent": statistics.fmean(grouped[name]),
            "wins": sum(value > 1e-12 for value in grouped[name]),
            "ties": sum(abs(value) <= 1e-12 for value in grouped[name]),
            "losses": sum(value < -1e-12 for value in grouped[name])}
            for name in sorted(grouped)]}


def summarize_optimization(rows, methods):
    keyed = {(row["instance"], row["algorithm_seed"], row["strategy"]): row
             for row in rows}
    expected = {(name, seed, method) for name in INSTANCES for seed in SEEDS
                for method in methods}
    if len(keyed) != len(rows) or set(keyed) != expected:
        raise ValueError("formal optimization grid incomplete or duplicated")
    paired = []
    for name in INSTANCES:
        for seed in SEEDS:
            base_row = keyed[name, seed, BASE]
            balanced = keyed[name, seed, BALANCED]
            for method in methods:
                row = keyed[name, seed, method]
                validate_run(row, method)
                if row["pairing"] != base_row["pairing"]:
                    raise ValueError("formal initial population/scenario pairing changed")
                paired.append({"instance": name, "algorithm_seed": seed,
                    "method": method,
                    "final_hypervolume": row["final_hypervolume"],
                    "percent_vs_base": 100*(row["final_hypervolume"] /
                        base_row["final_hypervolume"]-1),
                    "percent_vs_balanced": 100*(row["final_hypervolume"] /
                        balanced["final_hypervolume"]-1)})
    comparisons = {}
    for method_index, method in enumerate(methods):
        selected = [row for row in paired if row["method"] == method]
        comparisons[method] = {
            "vs_base": paired_summary(selected, "percent_vs_base",
                                      98301000+method_index),
            "vs_balanced": paired_summary(selected, "percent_vs_balanced",
                                          98302000+method_index)}
    ppo = methods[-1]
    direct = []
    for name in INSTANCES:
        for seed in SEEDS:
            supervised = keyed[name, seed, SUPERVISED]["final_hypervolume"]
            ppo_hv = keyed[name, seed, ppo]["final_hypervolume"]
            direct.append({"instance": name, "algorithm_seed": seed,
                "ppo_minus_supervised_percent": 100*(ppo_hv/supervised-1)})
    table = []
    for method in methods:
        selected = [row for row in rows if row["strategy"] == method]
        table.append({"method": method, "runs": len(selected),
            "mean_hv": statistics.fmean(
                row["final_hypervolume"] for row in selected),
            "mean_seconds": statistics.fmean(
                row["optimisation_seconds"] for row in selected),
            "equal_instance_mean_percent_vs_base": comparisons[method]
                ["vs_base"]["equal_instance_mean_percent"],
            "equal_instance_mean_percent_vs_balanced": comparisons[method]
                ["vs_balanced"]["equal_instance_mean_percent"]})
    return {"completed": True, "pairing_exact": True,
        "methods": list(methods), "comparisons": comparisons,
        "ppo_minus_supervised": paired_summary(
            direct, "ppo_minus_supervised_percent", 98303000),
        "table": table, "paired_runs": paired}


def evaluate_oos_instance(task):
    started = time.perf_counter(); instance_path = Path(task["instance_path"])
    data = Path(task["data"]); normalizer_path = Path(task["normalization"])
    out = Path(task["out"]); instance_name = task["instance"]
    methods = tuple(task["methods"]); seeds = tuple(task["seeds"])
    instance, batches = read_instance(instance_path)
    if hashlib.sha256(data.read_bytes()).hexdigest() != instance["source_data_sha256"]:
        raise ValueError("OOS workbook differs from final instance")
    base.BORDER_EVENT_DEFINITIONS = base.load_border_event_definitions(
        base.DEFAULT_BORDER_EVENT_DATA_FILE)
    network = base.load_network_from_extended(str(data))
    (_nodes, _regions, _hold, _proc, _trans_cost, arcs, _timetables,
     _source_batches, _wait_cost, _wait_emission, _carbon,
     _emission_factors, _speeds, _trans_map, _border_delay, _theta, _) = network
    profile = _uncertainty_profile("transport_border_saa_v1",
                                   OOS_SCENARIO_COUNT)
    active_border_delay = base.load_border_delay_map_from_workbook(
        data, include_customs=profile["border_include_customs"])
    scenarios = base.build_scenario_set(arcs, active_border_delay,
        OOS_SCENARIO_COUNT, int(instance["oos5000_seed"]), stochastic=True,
        border_event_definitions=base.BORDER_EVENT_DEFINITIONS,
        mode_time_cv=profile["mode_time_cv"],
        mode_time_max_factor=profile["mode_time_max_factor"],
        border_delay_cv=profile["border_delay_cv"],
        border_delay_max_factor=profile["border_delay_max_factor"],
        border_mean_source=profile["border_mean_source"])
    scenario_sha = scenario_digest(scenarios)
    base.ACTIVE_SCENARIO_SET = scenarios; base._PATH_SCENARIO_CACHE = {}
    base.RISK_METRIC = "ccp"
    base.CONFIDENCE_COST = base.CONFIDENCE_EMISSION = base.CONFIDENCE_TIME = .9
    divisor = (max(1., sum(float(batch.quantity) for batch in batches)),
        max(1., sum(float(batch.quantity) for batch in batches)),
        max(1., max(float(batch.LT) for batch in batches)-
            min(float(batch.ET) for batch in batches)))
    instance_normalizer = oos_normalizer(normalizer_path).with_divisor(divisor)
    rows = []
    for seed in seeds:
        target = out/"raw"/instance_name/f"seed_{seed}.json"
        if target.is_file():
            value = read_json(target)
            if (value.get("oos_scenario_digest") == scenario_sha
                    and set(value.get("methods", {})) == set(methods)):
                rows.append(value); continue
        evaluated = {method: _evaluate_front(
            Path(task["run_paths"][str(seed)][method]), instance, batches,
            network, scenarios, active_border_delay, instance_normalizer,
            .9, profile) for method in methods}
        value = {"completed": True, "instance": instance_name,
            "algorithm_seed": seed, "oos_scenario_count": scenarios.size,
            "oos_seed": scenarios.seed, "oos_scenario_digest": scenario_sha,
            "uncertainty_profile": profile,
            "normalization_divisor": list(divisor), "methods": evaluated}
        write_json(target, value); rows.append(value)
    return {"instance": instance_name, "rows": rows,
        "scenario_digest": scenario_sha,
        "elapsed_seconds": time.perf_counter()-started}


def run_oos(a, methods):
    tasks = []
    for name in INSTANCES:
        paths = {str(seed): {method: str(
            a.out/"runs"/name/f"seed_{seed}"/method) for method in methods}
            for seed in SEEDS}
        tasks.append({"instance": name,
            "instance_path": str(a.final_instances/f"{name}.json"),
            "data": str(a.data), "normalization": str(a.normalization),
            "out": str(a.out/"oos"), "run_paths": paths,
            "methods": list(methods), "seeds": list(SEEDS)})
    rows = []
    with ProcessPoolExecutor(max_workers=min(3, a.workers)) as pool:
        futures = [pool.submit(evaluate_oos_instance, task) for task in tasks]
        for index, future in enumerate(as_completed(futures), 1):
            result = future.result(); rows.extend(result["rows"])
            log(f"OOS {index}/{len(tasks)} {result['instance']} "
                f"elapsed={result['elapsed_seconds']:.1f}s")
    if len(rows) != len(INSTANCES)*len(SEEDS):
        raise RuntimeError("formal OOS grid incomplete")
    training_digests = {value["training_scenario_digest"] for row in rows
        for value in row["methods"].values()}
    oos_digests = {row["oos_scenario_digest"] for row in rows}
    if len(oos_digests) != len(INSTANCES) or oos_digests & training_digests:
        raise RuntimeError("formal OOS scenarios are not independent per instance")
    reference_info = derive_common_oos_reference(
        rows, oos_normalizer(a.normalization).reference)
    reference = reference_info["reference"]; paired = []
    candidate_scenarios = 0
    for row in rows:
        hvs = {method: exact_hv_3d(
            row["methods"][method]["oos_normalized_nondominated_points"],
            reference) for method in methods}
        if any(value <= 0 for value in hvs.values()):
            raise RuntimeError("nonpositive formal OOS HV")
        row["oos_hypervolumes"] = hvs
        for method in methods:
            candidate_scenarios += (row["methods"][method]["source_front_size"]
                                    * OOS_SCENARIO_COUNT)
            paired.append({"instance": row["instance"],
                "algorithm_seed": row["algorithm_seed"], "method": method,
                "percent_vs_balanced": 100*(hvs[method]/hvs[BALANCED]-1)})
        write_json(a.out/"oos"/"runs"/row["instance"]/
                   f"seed_{row['algorithm_seed']}.json", row)
    comparisons = {}
    for index, method in enumerate(methods):
        comparisons[method] = paired_summary(
            [row for row in paired if row["method"] == method],
            "percent_vs_balanced", 98310000+index)
    ppo = methods[-1]; direct = []
    for row in rows:
        direct.append({"instance": row["instance"],
            "algorithm_seed": row["algorithm_seed"],
            "ppo_minus_supervised_percent": 100*(
                row["oos_hypervolumes"][ppo]/
                row["oos_hypervolumes"][SUPERVISED]-1)})
    result = {"completed": True, "scenario_count": OOS_SCENARIO_COUNT,
        "per_instance_scenario_seed": True,
        "scenario_digests": sorted(oos_digests),
        "distinct_from_all_training_scenarios": True,
        "one_common_four_method_reference": True,
        "hv_reference": reference_info,
        "candidate_scenario_evaluations": candidate_scenarios,
        "vs_balanced_percent": comparisons,
        "ppo_minus_supervised": paired_summary(
            direct, "ppo_minus_supervised_percent", 98320000)}
    write_json(a.out/"oos"/"QUICK_RESULT.json", result)
    return result


def validate_final_catalog(a, protocol):
    catalog = read_json(a.final_instances/"catalog.json")
    identity = dict(catalog); claimed = identity.pop("catalog_digest", None)
    expected = protocol["final_test"]
    if (canonical_digest(identity) != claimed
            or claimed != expected["catalog_digest"]
            or catalog.get("source_catalog_digest") != expected[
                "source_catalog_digest"]
            or catalog.get("instance_count") != 10
            or catalog.get("master_seed") != expected["master_seed"]
            or catalog.get("ccp_seed_base") != expected["ccp_seed_base"]
            or catalog.get("oos_seed_base") != expected["oos_seed_base"]
            or catalog.get("optimisation_outcomes_used") is not False
            or catalog.get("model_selection_outcomes_used") is not False):
        raise ValueError("locked final-test catalog changed")
    if [row["instance_id"] for row in catalog["instances"]] != list(INSTANCES):
        raise ValueError("final-test instance identities changed")
    return catalog


def frozen_plan(a, manifest, methods):
    protocol = read_json(SOURCE/"experiments/m_prescreen_protocol_v1.json")
    design = protocol["final_test"]; catalog = validate_final_catalog(a, protocol)
    if (design["algorithm_seeds"] != list(SEEDS)
            or design.get("methods") != list(methods)
            or design.get("population") != 20
            or design.get("evaluation_budget") != 400
            or design["independent_oos_scenarios_per_instance"] !=
            OOS_SCENARIO_COUNT
            or digest(SOURCE/design["generator"]) != design["generator_sha256"]
            or digest(a.data) != design["network_workbook_sha256"]):
        raise ValueError("formal protocol or frozen generator changed")
    bundle_paths = [Path(manifest["supervised"]["bundle"]),
                    Path(manifest["transformer_ppo"]["bundle"])]
    inputs = [a.data, a.normalization, a.final_instances/"catalog.json",
        a.freeze_root/"FROZEN_FORMAL_METHODS.json", Path(__file__).resolve(),
        SOURCE/"m_prescreen_scale_policy.py",
        SOURCE/"evaluate_m_directed_balanced_oos.py",
        SOURCE/"experiments/m_prescreen_protocol_v1.json", *bundle_paths]
    inputs.extend(a.final_instances/f"{name}.json" for name in INSTANCES)
    return {"protocol": "m-prescreen-locked-formal-v1",
        "instances": list(INSTANCES), "algorithm_seeds": list(SEEDS),
        "methods": list(methods), "run_count": 400,
        "scenario_count": 100, "confidence": .9,
        "population": 20, "evaluation_budget": 400,
        "maximum_generations": 1000,
        "independent_oos_scenarios_per_instance": OOS_SCENARIO_COUNT,
        "workers": a.workers, "threads_per_worker": 1,
        "model_selection_after_opening": False,
        "simple_k8_included": False, "base_included": True,
        "formal_method_manifest_digest": manifest["manifest_digest"],
        "final_catalog_digest": catalog["catalog_digest"],
        "final_test_opened": True,
        "input_sha256": {str(Path(path).resolve()): digest(path)
                         for path in inputs}}


def execute(a):
    root = a.out.resolve(); manifest = read_json(
        a.freeze_root/"FROZEN_FORMAL_METHODS.json")
    methods = frozen_methods(manifest); plan = frozen_plan(a, manifest, methods)
    if (root/"PLAN.json").exists() and read_json(root/"PLAN.json") != plan:
        raise ValueError("formal plan changed after final test was opened")
    write_json(root/"PLAN.json", plan)
    (root/"FINAL_TEST_OPENED").write_text(
        f"catalog_digest={plan['final_catalog_digest']}\n"
        f"plan_sha256={digest(root/'PLAN.json')}\n")
    if (root/"PIPELINE_COMPLETE").exists():
        log("already complete; see QUICK_RESULT.json")
        return
    plan_sha = digest(root/"PLAN.json")
    bundles = {SUPERVISED: Path(manifest["supervised"]["bundle"]),
               methods[-1]: Path(manifest["transformer_ppo"]["bundle"])}
    (root/"logs").mkdir(exist_ok=True)
    started = time.monotonic(); lock = threading.Lock(); completed = 0
    env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
           "OPENBLAS_NUM_THREADS": "1", "PYTHONUNBUFFERED": "1"}

    def group(name, seed):
        nonlocal completed
        rows = []; offset = SEEDS.index(seed) % len(methods)
        order = methods[offset:]+methods[:offset]
        for method in order:
            folder = root/"runs"/name/f"seed_{seed}"/method
            reused = (folder/"RUN_COMPLETE.json").exists()
            if not reused:
                if folder.exists():
                    quarantine = root/"interrupted_runs"/(
                        f"{name}_{seed}_{method}_{time.time_ns()}")
                    quarantine.parent.mkdir(parents=True, exist_ok=True)
                    folder.rename(quarantine)
                logfile = root/"logs"/f"{name}_{seed}_{method}_{time.time_ns()}.log"
                command = [sys.executable, "-u", str(Path(__file__).resolve()),
                    "--worker", "--data", str(a.data.resolve()),
                    "--final-instances", str(a.final_instances.resolve()),
                    "--normalization", str(a.normalization.resolve()),
                    "--freeze-root", str(a.freeze_root.resolve()),
                    "--out", str(folder),
                    "--instance", str((a.final_instances/f"{name}.json").resolve()),
                    "--seed", str(seed), "--method", method,
                    "--plan-sha", plan_sha]
                if method in bundles:
                    command.extend(["--policy-bundle", str(bundles[method])])
                with logfile.open("w") as handle:
                    result = subprocess.run(command, cwd=SOURCE, env=env,
                        stdout=handle, stderr=subprocess.STDOUT, timeout=300)
                if result.returncode:
                    tail = "\n".join(logfile.read_text(
                        errors="replace").splitlines()[-20:])
                    raise RuntimeError(f"worker failed: {logfile}\n{tail}")
            row = load_run(folder, name, plan_sha); validate_run(row, method)
            rows.append(row)
            with lock:
                completed += 1
                write_json(root/"STATUS.json", {"stage": "optimization",
                    "completed_searches": completed, "total_searches": 400,
                    "elapsed_seconds": time.monotonic()-started})
                log(f"{completed}/400 {'reused' if reused else 'complete'} "
                    f"{name}/{seed}/{method}")
        return rows

    rows = []
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        futures = [pool.submit(group, name, seed) for name in INSTANCES
                   for seed in SEEDS]
        for future in as_completed(futures):
            rows.extend(future.result())
    optimization = summarize_optimization(rows, methods)
    write_json(root/"OPTIMIZATION_RESULT.json", optimization)
    with (root/"RESULT_TABLE.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle,
            fieldnames=list(optimization["table"][0]))
        writer.writeheader(); writer.writerows(optimization["table"])
    write_json(root/"STATUS.json", {"stage": "independent_oos5000",
        "completed_searches": completed, "total_searches": 400})
    oos = run_oos(a, methods)
    result = {"completed": True, "methods": list(methods),
        "formal_method_manifest": manifest,
        "optimization": optimization, "independent_oos5000": oos,
        "searches": 400, "search_objective_evaluations": 160_000,
        "new_training_steps": 0, "new_ppo_updates": 0,
        "final_test_opened": True,
        "elapsed_seconds_this_invocation": time.monotonic()-started,
        "qualification": (
            "single locked final test; no model, threshold or checkpoint "
            "selection is permitted from these outcomes")}
    write_json(root/"COMPLETE.json", {**result, "optimization_runs": rows})
    write_json(root/"QUICK_RESULT.json", result)
    write_json(root/"STATUS.json", {"stage": "complete",
        "completed_searches": completed, "total_searches": 400})
    (root/"PIPELINE_COMPLETE").write_text("LOCKED_FORMAL_COMPLETE\n")
    log("COMPLETE locked four-method formal experiment")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path,
        default=SOURCE/"data/data_expanded.xlsx")
    for name in ("final-instances", "normalization", "freeze-root", "out"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--workers", choices=(1, 2, 3), type=int, default=3)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--instance", type=Path)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--method")
    parser.add_argument("--policy-bundle", type=Path)
    parser.add_argument("--plan-sha")
    a = parser.parse_args(); a.out = a.out.resolve()
    if a.worker:
        if any(getattr(a, name) is None for name in
               ("instance", "seed", "method", "plan_sha")):
            parser.error("worker arguments incomplete")
        a.out.mkdir(parents=True, exist_ok=False)
        run_worker(a)
        return
    a.out.mkdir(parents=True, exist_ok=True)
    with (a.out/".lock").open("w") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            execute(a)
        except Exception as exc:
            previous = (read_json(a.out/"STATUS.json")
                        if (a.out/"STATUS.json").exists() else {})
            write_json(a.out/"STATUS.json", {
                **previous, "stage": "stopped", "error": str(exc)})
            log(f"STOPPED {exc}")
            raise


if __name__ == "__main__":
    main()
