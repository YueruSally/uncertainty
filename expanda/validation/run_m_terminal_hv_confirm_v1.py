#!/usr/bin/env python3
"""Frozen final confirmation: new instances, CCP100 search, independent OOS.

This script is separate from the development source manifest and never trains
or selects a model. Run only after the development trial has finished/passed.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

SOURCE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE))

import numpy as np
import torch
import baseline_uncertainty as base
from m_candidate_information import CandidatePolicy, CandidateController
from m_terminal_hv_ppo import (TerminalPolicy, SingleInterventionPolicy,
    SingleInterventionBaseline, load_checkpoint)
from m_operator_closed_loop import ProfiledBase
from m_operator_offline import read_json, write_json, digest
from evaluate_m_directed_balanced_oos import _uncertainty_profile, _nondominated
from run_ccp_candidate_pool import decision_signature
from run_formal_ev_vs_ccp_s30_30runs import restore_individual
from run_learning_components_deterministic import _normalizer
from run_ev_ccp_oos_pilot import scenario_digest
from run_m_operator_closed_loop_v1 import load_run
from transformer_ppo.instance_catalog import read_instance, canonical_digest
from transformer_ppo.episode import run_episode
from transformer_ppo.archive import exact_hv_3d

METHODS = ("BASE", "HEURISTIC_BALANCED", "TRANSFORMER_PPO")
INSTANCE_COUNT = 10
SEEDS = tuple(range(899201, 899211))
MASTER_SEED = 92_000_000
SEARCH_SEED_BASE = 93_000_000
OOS_SEED_BASE = 94_000_000
OOS_COUNT = 5000
DEVELOPMENT_MODES = {"terminal-hv-ppo-v1": "repeated-decisions",
    "single-intervention-ppo-v2": "single-intervention",
    "complete-terminal-feedback-v3": "single-intervention"}


def log(message):
    print(f"[M-CONFIRM] {message}", flush=True)


def freeze_development(development):
    plan = read_json(development / "PLAN.json")
    result = read_json(development / "QUICK_RESULT.json")
    selection = read_json(development / "SELECTED_POLICY.json")
    if (not (development / "PIPELINE_COMPLETE").exists() or not result.get("completed")
            or not result.get("development_pass") or selection["update"] <= 0
            or selection != result["selected"] or plan["protocol"] not in DEVELOPMENT_MODES):
        raise ValueError("development has not established PPO benefit; no final confirmation started")
    mode = DEVELOPMENT_MODES[plan["protocol"]]
    if bool(plan.get("single_intervention")) != (mode == "single-intervention"):
        raise ValueError("development execution mode is inconsistent")
    for path, sha in plan["input_sha256"].items():
        if digest(path) != sha:
            raise ValueError(f"development input/code changed: {path}")
    if digest(selection["checkpoint"]) != selection["sha256"]:
        raise ValueError("selected checkpoint changed")
    bundles = [p for p in plan["input_sha256"] if Path(p).name == "BUNDLE.json"]
    if len(bundles) != 1:
        raise ValueError("ambiguous source bundle")
    checkpoint = load_checkpoint(selection["checkpoint"])
    if (checkpoint.get("protocol") != "terminal-hv-ppo-v1"
            or checkpoint.get("execution_mode", "repeated-decisions") != mode
            or checkpoint["update"] != selection["update"]
            or checkpoint["bundle_sha256"] != digest(bundles[0])):
        raise ValueError("selected checkpoint execution mode/update/bundle differs from development")
    return plan, selection, Path(bundles[0])


def known_batch_signatures(source_instances, output):
    """Exclude all saved prior instance catalogs, not only the original split."""
    known = set()
    for path in source_instances.glob("*.json"):
        record = read_json(path)
        if record.get("batches"):
            known.add(canonical_digest(record["batches"]))
        if record.get("batch_signature"):
            known.add(record["batch_signature"])
    diagnostics = SOURCE / "transformer_ppo_runs"
    for path in diagnostics.rglob("catalog.json"):
        if output == path.resolve() or output in path.resolve().parents:
            continue
        catalog = read_json(path)
        records = catalog.get("instances", [])
        if isinstance(records, dict):
            records = records.values()
        for record in records:
            if isinstance(record, dict) and record.get("batch_signature"):
                known.add(record["batch_signature"])
    return known


def check_pair(rows):
    if set(rows) != set(METHODS):
        raise ValueError("all three methods required")
    first = rows[METHODS[0]]
    for method, row in rows.items():
        if (row["pairing"] != first["pairing"] or row["real_objective_evaluation_count"] != 400
                or row.get("ppo_updates", 0) or row.get("counterfactual_evaluation_count", 0)
                or row.get("training_feedback_evaluations", 0)):
            raise ValueError("different initial population/scenarios/budget or evaluation-time learning")
        if method != "BASE" and row["candidate_contract"] != "concrete-proposal-v1":
            raise ValueError("candidate execution changed")
    if any(row.get("single_intervention") for row in rows.values()):
        if not all(row.get("single_intervention") for row in rows.values()):
            raise ValueError("mixed single-intervention execution modes")
        heuristic, learned = rows["HEURISTIC_BALANCED"], rows["TRANSFORMER_PPO"]
        if (not heuristic.get("first_supported_input_sha256")
                or heuristic["first_supported_input_sha256"] != learned.get("first_supported_input_sha256")):
            raise ValueError("single-intervention state differs from paired Balanced")
        for method, row in rows.items():
            if (row.get("learned_decisions") != int(method == "TRANSFORMER_PPO")
                    or row.get("sampling") != (method == "TRANSFORMER_PPO")
                    or row.get("collect_for_training") is not False):
                raise ValueError("confirmation must preserve one frozen stochastic decision without training")
        if heuristic["closed_loop_switches"] or not 0 <= learned["closed_loop_switches"] <= 1:
            raise ValueError("more than one learned switch in confirmation")
        if not learned["closed_loop_switches"] and not np.isclose(
                learned["final_hypervolume"], heuristic["final_hypervolume"], rtol=0, atol=1e-12):
            raise ValueError("no-switch confirmation changed the Balanced result")


def search_worker(a, plan):
    torch.set_num_threads(1)
    instance_path = a.out / "instances" / (a.instance+".json")
    instance, _ = read_instance(instance_path)
    target = a.out / "search" / a.instance / f"seed_{a.seed}" / a.method
    single = bool(plan.get("single_intervention"))
    if a.method == "BASE":
        policy, component, factories = None, "BASE", {"BASE": ProfiledBase}
    elif a.method == "HEURISTIC_BALANCED":
        policy = (SingleInterventionBaseline(a.instance, a.seed) if single else
            CandidatePolicy(a.instance, a.seed, a.method))
        component, factories = "M", {"M": CandidateController}
    else:
        policy_class = SingleInterventionPolicy if single else TerminalPolicy
        policy = policy_class(a.instance, a.seed, plan["bundle"], plan["selected"]["checkpoint"], plan["selected"]["sha256"])
        component, factories = "M", {"M": CandidateController}
    row = run_episode(Path(plan["data"]), instance_path, target, policy,
        _normalizer(Path(plan["normalization"])), a.seed, evaluation_budget=400, pop_size=20,
        generations=1000, evaluation_mode="ccp", component=component, scenario_count=100,
        ccp_confidence=.9, uncertainty_profile="transport_border_saa_v1", inference_controller_factories=factories)
    row.update(strategy=a.method, algorithm_seed=a.seed, plan_sha256=digest(a.out / "PLAN.json"),
        policy_sha256=plan["selected"]["sha256"] if a.method == "TRANSFORMER_PPO" else None,
        final_front_sha256=digest(target / "final_feasible_nondominated.json"))
    if single:
        row.update(single_intervention=True, sampling=a.method == "TRANSFORMER_PPO", collect_for_training=False,
            learned_decisions=policy.model_evaluations if policy else 0,
            first_supported_input_sha256=policy.first_supported_input_sha256 if policy else None)
    if row["real_objective_evaluation_count"] != 400 or row.get("ppo_updates", 0) or row.get("training_feedback_evaluations", 0):
        raise ValueError("confirmation violated B400/frozen-policy contract")
    write_json(target / "RUN_COMPLETE.json", row)


def evaluate_fixed_front(folder, instance, batches, network, scenarios, delay, normalizer):
    config = read_json(folder / "configuration.json")
    if config["instance_digest"] != instance["instance_digest"] or config["scenario_digest"] == scenario_digest(scenarios):
        raise ValueError("wrong instance or reused optimization scenarios")
    (nodes, regions, hold, proc, trans_cost, arcs, timetables, _, wait_cost, wait_emission,
     carbon, _, _, trans_map, _, theta, _) = network
    tt, lookup = base.build_timetable_dict(timetables), base.build_arc_lookup(arcs)
    points, details = [], []
    source = read_json(folder / "final_feasible_nondominated.json")
    for row in source:
        individual = restore_individual(row, {}, tt, lookup)
        before = decision_signature(individual)
        base.evaluate_individual(individual, batches, arcs, tt, wait_cost, wait_emission,
            node_hold_cost=hold, node_proc_cost=proc, carbon_tax_map=carbon, trans_map=trans_map,
            border_delay_map=delay, theta_rm=theta, node_trans_cost=trans_cost)
        if before != decision_signature(individual) or before != row["decision_fingerprint"]:
            raise ValueError("OOS evaluation changed the frozen decision")
        arrays = [np.asarray(v) for v in (individual.cost_s, individual.emission_s, individual.makespan_s)]
        if any(v.shape != (scenarios.size,) or not np.isfinite(v).all() for v in arrays):
            raise ValueError("invalid OOS objective array")
        objectives = [float(base.empirical_ccp_quantile(v, .9)) for v in arrays]
        if individual.feasible:
            points.append(objectives)
        details.append({"fingerprint": before, "feasible": bool(individual.feasible), "q90": objectives})
    normalized = [list(normalizer.transform(p)) for p in _nondominated(points)]
    return {"hv": exact_hv_3d(normalized, normalizer.reference), "normalized_points": normalized,
        "outside_reference": sum(any(p[i] >= normalizer.reference[i] for i in range(3)) for p in normalized),
        "evaluations": len(source), "infeasible": sum(not r["feasible"] for r in details), "decisions": details,
        "source_front_sha256": digest(folder / "final_feasible_nondominated.json"),
        "search_scenario_digest": config["scenario_digest"]}


def oos_worker(a, plan):
    started = time.perf_counter()
    instance, batches = read_instance(a.out / "instances" / (a.instance+".json"))
    base.BORDER_EVENT_DEFINITIONS = base.load_border_event_definitions(base.DEFAULT_BORDER_EVENT_DATA_FILE)
    network = base.load_network_from_extended(plan["data"])
    arcs = network[5]
    profile = _uncertainty_profile("transport_border_saa_v1", plan["oos_count"])
    delay = base.load_border_delay_map_from_workbook(Path(plan["data"]), include_customs=False)
    scenarios = base.build_scenario_set(arcs, delay, plan["oos_count"], instance["oos5000_seed"], stochastic=True,
        border_event_definitions=base.BORDER_EVENT_DEFINITIONS, mode_time_cv=profile["mode_time_cv"],
        mode_time_max_factor=profile["mode_time_max_factor"], border_delay_cv=profile["border_delay_cv"],
        border_delay_max_factor=profile["border_delay_max_factor"], border_mean_source=profile["border_mean_source"])
    base.ACTIVE_SCENARIO_SET, base._PATH_SCENARIO_CACHE = scenarios, {}
    base.RISK_METRIC = "ccp"
    base.CONFIDENCE_COST = base.CONFIDENCE_EMISSION = base.CONFIDENCE_TIME = .9
    divisor = (max(1., sum(b.quantity for b in batches)), max(1., sum(b.quantity for b in batches)),
        max(1., max(b.LT for b in batches)-min(b.ET for b in batches)))
    normalizer = _normalizer(Path(plan["normalization"])).with_divisor(divisor)
    rows = []
    for seed in plan["seeds"]:
        methods = {m: evaluate_fixed_front(a.out / "search" / a.instance / f"seed_{seed}" / m,
            instance, batches, network, scenarios, delay, normalizer) for m in METHODS}
        rows.append({"instance": a.instance, "seed": seed, "methods": methods})
    write_json(a.out / "oos" / (a.instance+".json"), {"completed": True, "rows": rows,
        "plan_sha256": digest(a.out / "PLAN.json"), "scenario_digest": scenario_digest(scenarios),
        "scenario_seed": scenarios.seed, "scenario_count": scenarios.size, "reference": list(normalizer.reference),
        "elapsed_seconds": time.perf_counter()-started})


def hierarchical_summary(rows, instances, seeds, resamples=10000):
    indexed = {(r["instance"], r["seed"]): r for r in rows}
    if len(indexed) != len(rows) or set(indexed) != {(n, s) for n in instances for s in seeds}:
        raise ValueError("missing or duplicate predeclared pairs")
    effects = {"ppo_vs_balanced": [], "ppo_vs_base": [], "heuristic_vs_base": []}
    per_instance = []
    for name in instances:
        values = {k: [] for k in effects}
        for seed in seeds:
            hv = {m: float(indexed[name, seed]["methods"][m]["hv"]) for m in METHODS}
            if not np.isfinite(list(hv.values())).all() or hv["BASE"] <= 0 or hv["HEURISTIC_BALANCED"] <= 0:
                raise ValueError("nonpositive baseline HV: relative metric invalid; reference will not be moved after seeing results")
            values["ppo_vs_balanced"].append(100*(hv["TRANSFORMER_PPO"]/hv["HEURISTIC_BALANCED"]-1))
            values["ppo_vs_base"].append(100*(hv["TRANSFORMER_PPO"]/hv["BASE"]-1))
            values["heuristic_vs_base"].append(100*(hv["HEURISTIC_BALANCED"]/hv["BASE"]-1))
        per_instance.append({"instance": name, **{k: float(np.mean(v)) for k, v in values.items()}})
        for key in effects:
            effects[key].append(values[key])
    rng = np.random.default_rng(899900)
    # Outer resampling unit is the instance; inner seeds are paired across methods.
    instance_draws = rng.integers(len(instances), size=(resamples, len(instances)))
    seed_draws = rng.integers(len(seeds), size=(resamples, len(instances), len(seeds)))
    pooled = {}
    for key, value in effects.items():
        matrix = np.asarray(value)
        boot = matrix[instance_draws[:, :, None], seed_draws].mean(axis=(1,2))
        pooled[key] = {"mean": float(matrix.mean()), "hierarchical_95ci_low": float(np.quantile(boot,.025)),
            "hierarchical_95ci_high": float(np.quantile(boot,.975)),
            "positive_instances": int((matrix.mean(1)>0).sum())}
    return {"per_instance": per_instance, "pooled_equal_instance": pooled}


def run(a):
    development_plan, selection, bundle = freeze_development(a.development)
    for path in (a.data, a.normalization):
        if development_plan["input_sha256"].get(str(path.resolve())) != digest(path):
            raise ValueError("confirmation workbook/normalization differ from development")
    root = a.out.resolve()
    source_files = [Path(p) for p in development_plan["input_sha256"]] + [Path(__file__),
        SOURCE / "generate_s0_final_test_instances.py", a.source_instances / "catalog.json"]
    single = bool(development_plan.get("single_intervention"))
    plan = {"protocol": "single-intervention-confirm-v2" if single else "terminal-hv-confirm-v1",
        "development_protocol": development_plan["protocol"], "single_intervention": single,
        "execution_mode": DEVELOPMENT_MODES[development_plan["protocol"]],
        "evaluation_policy": "one frozen softmax sample, then unchanged Balanced" if single else "frozen argmax at each supported G10+ event",
        "selected": selection, "bundle": str(bundle),
        "development_sha256": digest(a.development / "QUICK_RESULT.json"), "data": str(a.data.resolve()),
        "normalization": str(a.normalization.resolve()), "source_instances": str(a.source_instances.resolve()),
        "instance_count": INSTANCE_COUNT, "master_seed": MASTER_SEED, "search_seed_base": SEARCH_SEED_BASE,
        "oos_seed_base": OOS_SEED_BASE, "seeds": list(SEEDS), "methods": list(METHODS),
        "search_ccp_count": 100, "ccp_confidence": .9, "population": 20, "budget": 400,
        "search_runs": INSTANCE_COUNT*len(SEEDS)*3, "oos_count": OOS_COUNT,
        "reference_policy": "unchanged training-only normalization and HV reference; never expand based on test results",
        "primary": "OOS PPO versus Balanced equal-instance mean paired percent; resample instances then paired seeds",
        "pass_rule": "OOS mean positive and hierarchical 95% lower bound >0, with a majority of instance means positive",
        "no_model_selection": True, "new_training_updates": 0, "workers": a.workers, "threads": 1,
        "source_sha256": {str(p.resolve()): digest(p) for p in source_files}}
    if (root / "PLAN.json").exists() and read_json(root / "PLAN.json") != plan:
        raise ValueError("confirmation plan changed; do not reuse or silently reselect")
    write_json(root / "PLAN.json", plan)
    sha = digest(root / "PLAN.json")
    if (root / "CONFIRMATION_COMPLETE").exists():
        log("already complete; no new searches")
        return
    stop = threading.Event()
    deadline = time.monotonic()+3600
    env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}

    def launch(command, label, timeout=600):
        if stop.is_set() or time.monotonic() >= deadline:
            raise TimeoutError("invocation cap reached; completed outputs retained")
        logfile = root / "logs" / f"{label}_{time.time_ns()}.log"
        logfile.parent.mkdir(exist_ok=True)
        with logfile.open("w") as handle, subprocess.Popen(command, cwd=SOURCE, env=env,
                stdout=handle, stderr=subprocess.STDOUT) as proc:
            start = last = time.monotonic()
            try:
                while proc.poll() is None:
                    now = time.monotonic()
                    if stop.is_set() or now >= deadline or now-start > timeout:
                        raise TimeoutError(f"timed out: {logfile}")
                    if now-last >= 15:
                        log(f"running {label} pid={proc.pid} elapsed={now-start:.0f}s")
                        last = now
                    time.sleep(.2)
            finally:
                if proc.poll() is None:
                    proc.kill()
                    proc.wait()
            if proc.returncode:
                raise RuntimeError(f"failed: {logfile}\n"+"\n".join(logfile.read_text(errors="replace").splitlines()[-20:]))

    generated = root / "instances"
    if not (generated / "catalog.json").exists():
        if generated.exists() and any(generated.iterdir()):
            generated.rename(root / f"interrupted_generation_{time.time_ns()}")
        launch([sys.executable, "-u", str(SOURCE / "generate_s0_final_test_instances.py"), "--data", plan["data"],
            "--source-instances", plan["source_instances"], "--out", str(generated), "--count", str(INSTANCE_COUNT),
            "--master-seed", str(MASTER_SEED), "--ccp-seed-base", str(SEARCH_SEED_BASE), "--oos-seed-base", str(OOS_SEED_BASE)], "generate")
    catalog = read_json(generated / "catalog.json")
    if (catalog["master_seed"] != MASTER_SEED or catalog["instance_count"] != INSTANCE_COUNT
            or catalog["ccp_seed_base"] != SEARCH_SEED_BASE or catalog["oos_seed_base"] != OOS_SEED_BASE):
        raise ValueError("generated instances violate frozen confirmation plan")
    payload = dict(catalog)
    if payload.pop("catalog_digest") != canonical_digest(payload):
        raise ValueError("generated catalog changed")
    instances = [r["instance_id"] for r in catalog["instances"]]
    previous_signatures = known_batch_signatures(a.source_instances, root)
    new_signatures = set()
    for record in catalog["instances"]:
        instance, _ = read_instance(generated / (record["instance_id"]+".json"))
        if instance["instance_digest"] != record["instance_digest"]:
            raise ValueError("generated instance changed")
        signature = canonical_digest(instance["batches"])
        if signature != record["batch_signature"] or signature in previous_signatures or signature in new_signatures:
            raise ValueError("new confirmation repeats a previously stored batch set")
        new_signatures.add(signature)
    hashes = {str(p): digest(p) for p in sorted(generated.glob("*.json"))}
    receipt = root / "FROZEN_INSTANCES.json"
    if receipt.exists() and read_json(receipt) != hashes:
        raise ValueError("test manifests changed after freezing")
    write_json(receipt, hashes)

    def task(name, seed=None, method=None):
        target = root / "oos" / (name+".json") if seed is None else root / "search" / name / f"seed_{seed}" / method / "RUN_COMPLETE.json"
        if not target.exists():
            if seed is not None and target.parent.exists() and any(target.parent.iterdir()):
                quarantine = root / "interrupted_runs" / f"{name}_{seed}_{method}_{time.time_ns()}"
                quarantine.parent.mkdir(exist_ok=True)
                target.parent.rename(quarantine)
            command = [sys.executable, "-u", str(Path(__file__).resolve()), "--worker", "oos" if seed is None else "search",
                       "--out", str(root), "--instance", name]
            if seed is not None:
                command += ["--seed", str(seed), "--method", method]
            launch(command, f"{name}_{seed}_{method}")
        value = read_json(target)
        if value["plan_sha256"] != sha:
            raise ValueError("saved output is from another plan")
        if seed is not None:
            value = load_run(target.parent, name, sha)
            if (value["strategy"] != method or value["algorithm_seed"] != seed
                    or value["final_front_sha256"] != digest(target.parent / "final_feasible_nondominated.json")
                    or value["policy_sha256"] != (selection["sha256"] if method == "TRANSFORMER_PPO" else None)):
                raise ValueError("saved search/front/policy changed")
        else:
            for row in value["rows"]:
                for m, result in row["methods"].items():
                    front = root / "search" / name / f"seed_{row['seed']}" / m / "final_feasible_nondominated.json"
                    if result["source_front_sha256"] != digest(front):
                        raise ValueError("OOS source front changed")
        return value

    def parallel(jobs, stage):
        values = []
        with ThreadPoolExecutor(max_workers=a.workers) as pool:
            futures = [pool.submit(task, *j) for j in jobs]
            try:
                for future in as_completed(futures):
                    values.append(future.result())
                    log(f"{stage} {len(values)}/{len(jobs)} complete")
                    write_json(root / "STATUS.json", {"stage": stage, "completed": len(values), "total": len(jobs)})
            except BaseException:
                stop.set()
                for f in futures:
                    f.cancel()
                raise
        return values

    jobs = [(n,s,m) for n in instances for i,s in enumerate(SEEDS) for m in METHODS[i%3:]+METHODS[:i%3]]
    runs = parallel(jobs, "CCP100 search")
    grouped = {}
    for row in runs:
        grouped.setdefault((row["instance"],row["algorithm_seed"]), {})[row["strategy"]] = row
    for rows in grouped.values():
        check_pair(rows)
    # Always re-evaluate every method and seed, regardless of online performance.
    oos = parallel([(n,) for n in instances], "independent OOS5000")
    oos_rows = [r for value in oos for r in value["rows"]]
    online = [{"instance": n, "seed": s, "methods": {m: {"hv": r["final_hypervolume"]} for m,r in group.items()}}
              for (n,s),group in grouped.items()]
    oos_report = hierarchical_summary(oos_rows, instances, SEEDS)
    primary = oos_report["pooled_equal_instance"]["ppo_vs_balanced"]
    passed = primary["hierarchical_95ci_low"] > 0 and primary["positive_instances"] > INSTANCE_COUNT/2
    summary = {"completed": True, "independent_confirmation_pass": passed, "selected_policy": selection,
        "comparison_design": {"instances": INSTANCE_COUNT, "paired_seeds": len(SEEDS), "methods": METHODS,
            "search_scenarios": 100, "population":20, "evaluation_budget":400, "oos_scenarios":OOS_COUNT,
            "execution_mode": plan["execution_mode"], "evaluation_policy": plan["evaluation_policy"]},
        "online": hierarchical_summary(online, instances, SEEDS), "oos": oos_report,
        "methods": {m: {"mean_optimisation_seconds": float(np.mean([r["optimisation_seconds"] for r in runs if r["strategy"] == m]))} for m in METHODS},
        "cost": {"new_training_updates":0, "search_objective_evaluations":400*len(runs),
            "oos_objective_evaluations":sum(v["evaluations"] for r in oos_rows for v in r["methods"].values()),
            "oos_scenario_candidate_evaluations":OOS_COUNT*sum(v["evaluations"] for r in oos_rows for v in r["methods"].values()),
            "oos_summed_worker_seconds":sum(v["elapsed_seconds"] for v in oos),
            "interrupted_runs":len(list((root / "interrupted_runs").glob("*"))),
            "cost_scope":"completed searches and OOS evaluations; any uncommitted interrupted work is not recoverable from complete receipts",
            "prior_training":read_json(a.development / "QUICK_RESULT.json")["training_summary"]},
        "reference_frozen":True, "initial_population_and_search_scenarios_paired":True,
        "qualification":"Fixed-policy confirmation; one independent OOS set per new instance. No model selection, threshold adjustment, or selective method/seed exclusion."}
    write_json(root / "QUICK_RESULT.json", summary)
    write_json(root / "COMPLETE.json", {**summary, "search_runs":runs, "oos_records":oos})
    with (root / "RESULT_TABLE.csv").open("w") as handle:
        writer=csv.writer(handle)
        writer.writerow(["method","mean_optimisation_seconds","OOS_mean_percent_vs_base"])
        for m in METHODS:
            effect=0. if m=="BASE" else oos_report["pooled_equal_instance"]["heuristic_vs_base" if m=="HEURISTIC_BALANCED" else "ppo_vs_base"]["mean"]
            writer.writerow([m,summary["methods"][m]["mean_optimisation_seconds"],effect])
    (root / "CONFIRMATION_COMPLETE").write_text("complete; inspect independent_confirmation_pass\n")
    write_json(root / "STATUS.json",{"stage":"complete","independent_confirmation_pass":passed})
    log(f"COMPLETE independent_confirmation_pass={passed}")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--development",type=Path)
    p.add_argument("--source-instances",type=Path)
    p.add_argument("--data",type=Path,default=SOURCE / "data/data_expanded.xlsx")
    p.add_argument("--normalization",type=Path)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--workers",type=int,choices=(1,2),default=2)
    p.add_argument("--worker",choices=("search","oos"))
    p.add_argument("--instance")
    p.add_argument("--seed",type=int)
    p.add_argument("--method",choices=METHODS)
    a=p.parse_args()
    if a.worker:
        plan=read_json(a.out / "PLAN.json")
        (search_worker if a.worker=="search" else oos_worker)(a,plan)
        return
    if not all((a.development,a.source_instances,a.normalization)):
        p.error("requires development, source-instances, normalization")
    a.out.mkdir(parents=True,exist_ok=True)
    with (a.out / ".lock").open("w") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            run(a)
        except Exception as exc:
            write_json(a.out / "STOPPED.json",{"reason":str(exc),"completed_outputs_retained":True})
            raise


if __name__=="__main__":
    main()
