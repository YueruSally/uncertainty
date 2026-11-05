#!/usr/bin/env python3
"""One bounded terminal-HV PPO trial. No automated hyperparameter search."""
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

import numpy as np
import torch

from m_candidate_information import CandidatePolicy, CandidateController
from m_operator_offline import read_json, write_json, digest
from m_operator_fixed_mix import bootstrap
from m_terminal_hv_ppo import (METHODS, initialize, TerminalPolicy, checkpoint_model,
    optimizer_for, save_checkpoint, update_ppo, SingleInterventionPolicy, SingleInterventionBaseline)
from run_learning_components_deterministic import _normalizer
from run_m_operator_closed_loop_v1 import load_run
from transformer_ppo.instance_catalog import read_instance
from transformer_ppo.episode import run_episode

SOURCE = Path(__file__).resolve().parent
TRAIN_INSTANCES = tuple("train-S0-"+n for n in ("002", "005", "006", "007", "008", "009", "014", "016"))
TRAIN_SEEDS = tuple(range(898001, 898005))
VALID_INSTANCES = ("train-S0-011", "train-S0-015")
VALID_SEEDS = (898101, 898102)
TEST_INSTANCES = ("train-S0-021", "train-S0-022")
TEST_SEEDS = tuple(range(898201, 898221))
EPISODES_PER_UPDATE = 4
SINGLE_TRAIN_SEEDS = tuple(range(900001, 900009))
SINGLE_VALID_SEEDS = tuple(range(900101, 900111))
SINGLE_TEST_SEEDS = tuple(range(900201, 900221))
SINGLE_EPISODES_PER_UPDATE = 8
SINGLE_VALIDATION_UPDATES = (0, 4, 8)


def design(a):
    single = bool(getattr(a, "single_intervention", False))
    return {"single": single,
        "train_seeds": SINGLE_TRAIN_SEEDS if single else TRAIN_SEEDS,
        "valid_seeds": SINGLE_VALID_SEEDS if single else VALID_SEEDS,
        "test_seeds": SINGLE_TEST_SEEDS if single else TEST_SEEDS,
        "episodes_per_update": SINGLE_EPISODES_PER_UPDATE if single else EPISODES_PER_UPDATE}


def log(text):
    print(f"[M-TERMINAL] {text}", flush=True)


def worker(a):
    torch.set_num_threads(1)
    info, _ = read_instance(a.instance)
    single = bool(getattr(a, "single_intervention", False))
    if single:
        policy = (SingleInterventionBaseline(info["instance_id"], a.seed) if a.method == METHODS[0] else
            SingleInterventionPolicy(info["instance_id"], a.seed, a.bundle, a.checkpoint, a.checkpoint_sha, sample=a.sample))
    elif a.method == METHODS[0]:
        policy = CandidatePolicy(info["instance_id"], a.seed, "HEURISTIC_BALANCED")
    elif a.method == METHODS[1]:
        policy = CandidatePolicy(info["instance_id"], a.seed, "STATE_PLUS_CANDIDATE", a.bundle)
    else:
        policy = TerminalPolicy(info["instance_id"], a.seed, a.bundle, a.checkpoint,
                                a.checkpoint_sha, sample=a.sample)
    if a.sample and a.method != METHODS[2]:
        raise ValueError("only PPO policy may produce on-policy samples")
    result = run_episode(a.data, a.instance, a.out, policy, _normalizer(a.normalization), a.seed,
        evaluation_budget=400, pop_size=20, generations=1000, evaluation_mode="ccp", component="M",
        scenario_count=100, ccp_confidence=.9, uncertainty_profile="transport_border_saa_v1",
        inference_controller_factories={"M": CandidateController})
    if (result["real_objective_evaluation_count"] != 400 or result["ppo_updates"]
            or result["counterfactual_evaluation_count"] or result["training_feedback_evaluations"]):
        raise RuntimeError("executor must use exactly B400, no internal updates or extra evaluations")
    result.update(strategy=a.method, algorithm_seed=a.seed, plan_sha256=a.plan_sha,
        source_bundle_sha256=digest(a.bundle), policy_sha256=a.checkpoint_sha,
        sampling=bool(a.sample or (single and a.method != METHODS[0])), collect_for_training=a.sample,
        training_role="on-policy episode for external PPO" if a.sample else "frozen evaluation")
    if single:
        result.update(single_intervention=True, learned_decisions=policy.model_evaluations,
            first_supported_input_sha256=policy.first_supported_input_sha256)
        if policy.model_evaluations != int(a.method != METHODS[0]) or policy.switches > 1:
            raise RuntimeError("single-intervention protocol requires exactly one learned choice and at most one switch")
    if a.sample:
        if not policy.transitions:
            raise RuntimeError("no on-policy actions collected")
        write_json(a.out / "TRAJECTORY.json", {"policy_sha256": a.checkpoint_sha,
            "transitions": policy.transitions})
        result["trajectory_sha256"] = digest(a.out / "TRAJECTORY.json")
    write_json(a.out / "RUN_COMPLETE.json", result)


def make_plan(a):
    d = design(a)
    train_seeds, valid_seeds, test_seeds = d["train_seeds"], d["valid_seeds"], d["test_seeds"]
    episode_batch = d["episodes_per_update"]
    info = read_json(a.bundle)
    if info["protocol"] != "candidate-information-v1":
        raise ValueError("requires the existing concrete-candidate pretraining bundle")
    if set(TRAIN_INSTANCES) & set(VALID_INSTANCES) or (set(TRAIN_INSTANCES) | set(VALID_INSTANCES)) & set(TEST_INSTANCES):
        raise ValueError("PPO splits overlap")
    if set(TEST_INSTANCES) & (set(info["training_instances"]) | {info["validation_instance"]}):
        raise ValueError("development search overlaps pretraining instances")
    if len(TRAIN_INSTANCES)*len(train_seeds) % episode_batch:
        raise ValueError("whole rollout blocks required")
    files = [a.data.resolve(), a.normalization.resolve(), a.bundle.resolve(), SOURCE / "data/border_crossing_events.csv"]
    for p, expected in info["checkpoint_sha256"].items():
        if digest(p) != expected:
            raise ValueError("pretraining checkpoint changed")
        files.append(Path(p))
    for name in TRAIN_INSTANCES+VALID_INSTANCES+TEST_INSTANCES:
        path = (a.instances / (name+".json")).resolve()
        record, _ = read_instance(path)
        if record["instance_id"] != name or record["source_data_sha256"] != digest(a.data):
            raise ValueError(f"instance/workbook mismatch: {name}")
        files.append(path)
    _normalizer(a.normalization)
    files += sorted(SOURCE.glob("*.py"))+sorted((SOURCE / "transformer_ppo").glob("*.py"))
    blocks = len(TRAIN_INSTANCES)*len(train_seeds)//episode_batch
    validation_updates = list(SINGLE_VALIDATION_UPDATES) if d["single"] else list(range(blocks+1))
    if not validation_updates or validation_updates[0] != 0 or validation_updates[-1] != blocks:
        raise ValueError("validation schedule must include initial and final checkpoint")
    total = 2*len(TRAIN_INSTANCES)*len(train_seeds) + (len(validation_updates)+1)*len(VALID_INSTANCES)*len(valid_seeds) + 3*len(TEST_INSTANCES)*len(test_seeds)
    return json.loads(json.dumps({"protocol": "single-intervention-ppo-v2" if d["single"] else "terminal-hv-ppo-v1", "role": "bounded development; these search instances have been used for prior development",
        "ccp_scenarios": 100, "ccp_confidence": .9, "population": 20, "evaluation_budget": 400,
        "minimum_control_generation": 10, "methods": METHODS, "train_instances": TRAIN_INSTANCES,
        "train_seeds": train_seeds, "validation_instances": VALID_INSTANCES, "validation_seeds": valid_seeds,
        "test_instances": TEST_INSTANCES, "test_seeds": test_seeds, "single_intervention": d["single"],
        "intervention_rule": "first G10+ state with legal Balanced and at least one legal alternative; one sampled choice even if Balanced is selected; all remaining choices use unchanged Balanced" if d["single"] else "every supported G10+ event",
        "validation_updates": validation_updates,
        "training_episode_count": len(TRAIN_INSTANCES)*len(train_seeds), "episodes_per_update": episode_batch,
        "maximum_ppo_updates": blocks, "maximum_search_runs": total, "maximum_new_objective_evaluations": total*400,
        "gamma": 1, "reward": "terminal final_HV / matched_Balanced_final_HV - 1; no clipping/shaping",
        "optimizer": {"learning_rate": 1e-4, "epochs": 4, "minibatch": 64, "clip": .2,
            "entropy_coefficient": .005, "value_coefficient": .5, "gradient_norm": 1., "target_kl": .03},
        "actor": "same three small pretrained Transformers; mean scores, offset 0.02 for alternatives, temperature 0.25; " + ("one softmax sample in both training and evaluation; no best-of sampling" if d["single"] else "softmax sampling in training, argmax in evaluation"),
        "checkpoint_selection": "highest mean final paired HV gain on fixed validation searches; ties prefer earlier update, including update 0",
        "decision": "PPO improves on both Balanced and unchanged pretrained model in each development instance; equal-instance bootstrap lower bound >0 against both; selected update >0. Otherwise this trial stops, with all results retained.",
        "bootstrap": "10000 resamples of paired seeds within fixed instances; development only, not new-instance generalization",
        "cost": "matched Balanced reward searches counted as training cost; pretraining is prior sunk cost; no unselected candidate CCP evaluation at inference",
        "timing": "candidate construction and inference included; concurrent 1-thread workers, not isolated formal timing",
        "workers": a.workers, "threads_per_worker": 1, "run_timeout_seconds": 300,
        "invocation_timeout_seconds": 2700 if d["single"] else 1800, "automatic_retries": 0,
        "input_sha256": {str(p): digest(p) for p in files}, "torch": str(torch.__version__), "numpy": str(np.__version__)}))


def verify_pair(rows):
    first = rows[0]
    for row in rows:
        if (row["pairing"] != first["pairing"] or row["real_objective_evaluation_count"] != 400
                or row["ppo_updates"] or row["counterfactual_evaluation_count"] or row["training_feedback_evaluations"]
                or row["source_bundle_sha256"] != first["source_bundle_sha256"]
                or row["candidate_contract"] != "concrete-proposal-v1" or row["minimum_control_generation"] != 10):
            raise ValueError("paired initial population/scenarios/budget/execution mismatch")
        if not np.isfinite(row["final_hypervolume"]) or row["final_hypervolume"] <= 0:
            raise ValueError("paired percent requires positive finite HV")
    if any(r.get("single_intervention") for r in rows):
        if (not all(r.get("single_intervention") for r in rows)
                or not first.get("first_supported_input_sha256")
                or any(r["first_supported_input_sha256"] != first["first_supported_input_sha256"] for r in rows)):
            raise ValueError("single-decision inputs differ from the paired Balanced state")
        if any(r["learned_decisions"] != int(r["strategy"] != METHODS[0]) or r["closed_loop_switches"] > 1 for r in rows):
            raise ValueError("single-intervention count exceeded")
        baseline = next((r for r in rows if r["strategy"] == METHODS[0]), None)
        if baseline is not None and any(
                r["closed_loop_switches"] == 0
                and not np.isclose(r["final_hypervolume"], baseline["final_hypervolume"], rtol=0, atol=1e-12)
                for r in rows):
            raise ValueError("no-switch policy changed the paired Balanced result")


def report_search(runs, plan):
    pairs = {}
    for row in runs:
        key = row["instance"], row["algorithm_seed"]
        if row["strategy"] in pairs.setdefault(key, {}):
            raise ValueError("duplicate run")
        pairs[key][row["strategy"]] = row
    if set(pairs) != {(n, s) for n in plan["test_instances"] for s in plan["test_seeds"]}:
        raise ValueError("incomplete development pairs")
    rng = np.random.default_rng(898900)
    reports, values, samples = [], [], {"ppo_vs_balanced": [], "ppo_vs_pretrained": []}
    for name in plan["test_instances"]:
        grouped = {m: [] for m in METHODS}
        effects = {"ppo_vs_balanced": [], "ppo_vs_pretrained": [], "pretrained_vs_balanced": []}
        for seed in plan["test_seeds"]:
            group = pairs[name, seed]
            if set(group) != set(METHODS):
                raise ValueError("requires all three frozen methods")
            rows = [group[m] for m in METHODS]
            verify_pair(rows)
            if plan.get("single_intervention"):
                if any(r["sampling"] != (r["strategy"] != METHODS[0]) or r.get("collect_for_training") for r in rows):
                    raise ValueError("single-intervention evaluation must use one frozen sample, without training")
            elif any(r["sampling"] for r in rows):
                raise ValueError("development must use frozen argmax policies")
            h, pre, ppo = [r["final_hypervolume"] for r in rows]
            effect = {"ppo_vs_balanced": 100*(ppo/h-1), "ppo_vs_pretrained": 100*(ppo/pre-1),
                      "pretrained_vs_balanced": 100*(pre/h-1)}
            values.append({"instance": name, "seed": seed, **effect})
            for k, v in effect.items():
                effects[k].append(v)
            for row in rows:
                grouped[row["strategy"]].append(row)
        indices = rng.integers(0, len(plan["test_seeds"]), (10000, len(plan["test_seeds"])))
        item = {"instance": name, "paired_seeds": len(plan["test_seeds"]), "methods": {}}
        for method, group in grouped.items():
            item["methods"][method] = {"mean_hv": float(np.mean([r["final_hypervolume"] for r in group])),
                "mean_seconds": float(np.mean([r["optimisation_seconds"] for r in group])),
                "switches": sum(r["closed_loop_switches"] for r in group)}
        for key, v in effects.items():
            item[key] = bootstrap(v, indices)
            if key in samples:
                samples[key].append(np.asarray(v)[indices].mean(1))
        reports.append(item)
    pooled = {k: {"mean": float(np.mean([r[k]["mean"] for r in reports])),
        "low": float(np.quantile(np.mean(v, axis=0), .025)),
        "high": float(np.quantile(np.mean(v, axis=0), .975))} for k, v in samples.items()}
    passed = all(v["low"] > 0 and all(r[k]["mean"] > 0 for r in reports) for k, v in pooled.items())
    return {"per_instance": reports, "paired_runs": values, "pooled_equal_instance_comparison": pooled,
            "development_gain_established": passed}


def execute(a):
    torch.set_num_threads(1)
    root = a.out.resolve()
    plan = make_plan(a)
    if (root / "PLAN.json").exists() and read_json(root / "PLAN.json") != plan:
        raise ValueError("frozen inputs/config/code changed: use a new output directory")
    write_json(root / "PLAN.json", plan)
    plan_sha = digest(root / "PLAN.json")
    if (root / "PIPELINE_COMPLETE").exists():
        log("already complete; retained QUICK_RESULT.json")
        return
    deadline = time.monotonic()+plan["invocation_timeout_seconds"]
    stop = threading.Event()
    env = {**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    (root / "logs").mkdir(exist_ok=True)

    def run_job(stage, name, seed, method, checkpoint=None, sample=False):
        sha = digest(checkpoint) if checkpoint else None
        folder = root / stage / name / f"seed_{seed}" / method
        if not (folder / "RUN_COMPLETE.json").exists():
            if stop.is_set() or time.monotonic() >= deadline:
                raise TimeoutError("invocation cap reached; complete runs/checkpoints retained for resume")
            if folder.exists() and any(folder.iterdir()):
                archived = root / "interrupted_runs" / f"{name}_{seed}_{method}_{time.time_ns()}"
                archived.parent.mkdir(exist_ok=True)
                folder.rename(archived)
            command = [sys.executable, "-u", str(Path(__file__).resolve()), "--worker", "--instance",
                str((a.instances / (name+".json")).resolve()), "--seed", str(seed), "--method", method,
                "--data", str(a.data.resolve()), "--normalization", str(a.normalization.resolve()),
                "--bundle", str(a.bundle.resolve()), "--out", str(folder), "--plan-sha", plan_sha]
            if checkpoint:
                command += ["--checkpoint", str(checkpoint), "--checkpoint-sha", sha]
            if sample:
                command += ["--sample"]
            if plan["single_intervention"]:
                command += ["--single-intervention"]
            logfile = root / "logs" / f"{stage.replace('/', '_')}_{name}_{seed}_{method}_{time.time_ns()}.log"
            log(f"start {stage} {name}/{seed}/{method}")
            with logfile.open("w") as handle, subprocess.Popen(command, cwd=SOURCE, env=env,
                    stdout=handle, stderr=subprocess.STDOUT) as process:
                started = last = time.monotonic()
                try:
                    while process.poll() is None:
                        now = time.monotonic()
                        if stop.is_set() or now >= deadline or now-started > plan["run_timeout_seconds"]:
                            raise TimeoutError(f"bounded process stopped; results retained; log={logfile}")
                        if now-last >= 15:
                            log(f"running {stage} {name}/{method} pid={process.pid} elapsed={now-started:.0f}s")
                            last = now
                        time.sleep(.2)
                finally:
                    if process.poll() is None:
                        process.kill()
                        process.wait()
                if process.returncode:
                    raise RuntimeError(f"run failed; log={logfile}\n"+"\n".join(logfile.read_text(errors="replace").splitlines()[-20:]))
        row = load_run(folder, name, plan_sha)
        expected_sampling = bool(sample or (plan["single_intervention"] and method != METHODS[0]))
        if (row["policy_sha256"] != sha or row["sampling"] != expected_sampling
                or row.get("collect_for_training", row["sampling"]) != sample or row["strategy"] != method
                or row["algorithm_seed"] != seed or row["source_bundle_sha256"] != digest(a.bundle)):
            raise ValueError("saved run policy/identity differs")
        if sample:
            if row["trajectory_sha256"] != digest(folder / "TRAJECTORY.json"):
                raise ValueError("trajectory changed")
            trajectory = read_json(folder / "TRAJECTORY.json")
            if trajectory["policy_sha256"] != sha:
                raise ValueError("off-policy trajectory")
            row["transitions"] = trajectory["transitions"]
        return row

    def parallel(jobs, label):
        result = []
        with ThreadPoolExecutor(max_workers=a.workers) as pool:
            futures = [pool.submit(run_job, *job) for job in jobs]
            try:
                for f in as_completed(futures):
                    row = f.result()
                    result.append(row)
                    write_json(root / "STATUS.json", {"stage": label, "completed": len(result), "total": len(jobs)})
                    log(f"{label} {len(result)}/{len(jobs)} complete {row['instance']}/{row['strategy']} HV={row['final_hypervolume']:.7g}")
            except BaseException:
                stop.set()
                for f in futures:
                    f.cancel()
                raise
        return result

    def save_model(path, model, optimizer, index, metrics):
        save_checkpoint(path, {"protocol": "terminal-hv-ppo-v1", "plan_sha256": plan_sha,
            "bundle_sha256": digest(a.bundle), "update": index,
            "execution_mode": "single-intervention" if plan["single_intervention"] else "repeated-decisions",
            "feature_count": len(read_json(a.bundle)["scaler"]["mean"]), "ensemble_count": len(model.actors),
            "model": model.state_dict(), "optimizer": optimizer.state_dict(), "metrics": metrics})

    initial = root / "checkpoints/update_000.pt"
    if not initial.exists():
        model = initialize(a.bundle)
        save_model(initial, model, optimizer_for(model), 0, {})
    valid_baselines = parallel([("validation_baselines", n, s, METHODS[0]) for n in plan["validation_instances"] for s in plan["validation_seeds"]], "validation baselines")
    hval = {(r["instance"], r["algorithm_seed"]): r for r in valid_baselines}
    selected, validations, training_metrics = None, [], []
    train_pairs = [(n, s) for s in plan["train_seeds"] for n in plan["train_instances"]]
    current = initial
    for update in range(plan["maximum_ppo_updates"]+1):
        model, saved = checkpoint_model(current, digest(current))
        if saved["plan_sha256"] != plan_sha or saved["update"] != update:
            raise ValueError("checkpoint belongs to another frozen run")
        optimizer = optimizer_for(model)
        optimizer.load_state_dict(saved["optimizer"])
        if update:
            training_metrics.append(saved["metrics"])
        if update in plan["validation_updates"]:
            validation = parallel([(f"validation/update_{update:03d}", n, s, METHODS[2], current)
                for n in plan["validation_instances"] for s in plan["validation_seeds"]], f"validation update={update}")
            gains, instance_gains = [], {}
            # Canonical order is required for deterministic selection on resume.
            for row in sorted(validation, key=lambda r: (r["instance"], r["algorithm_seed"])):
                baseline = hval[row["instance"], row["algorithm_seed"]]
                verify_pair([baseline, row])
                gain = 100*(row["final_hypervolume"]/baseline["final_hypervolume"]-1)
                gains.append(gain)
                instance_gains.setdefault(row["instance"], []).append(gain)
            record = {"update": update, "checkpoint": str(current), "sha256": digest(current),
                "mean_final_paired_percent": float(np.mean(gains)), "paired_percent": gains,
                "per_instance_mean": {n: float(np.mean(v)) for n,v in instance_gains.items()}}
            validations.append(record)
            if selected is None or record["mean_final_paired_percent"] > selected["mean_final_paired_percent"]+1e-12:
                selected = record
            log(f"VALIDATION update={update}/{plan['maximum_ppo_updates']} final_HV_gain={record['mean_final_paired_percent']:+.3f}% selected_update={selected['update']}")
        write_json(root / "TRAINING_PROGRESS.json", {"completed_ppo_updates": update,
            "selected": selected, "validation": validations, "ppo": training_metrics})
        if update == plan["maximum_ppo_updates"]:
            break
        next_checkpoint = root / f"checkpoints/update_{update+1:03d}.pt"
        if not next_checkpoint.exists():
            block = train_pairs[update*plan["episodes_per_update"]:(update+1)*plan["episodes_per_update"]]
            # All complete trajectories use the SAME actor, before one update.
            jobs = []
            for n, s in block:
                jobs.extend([("training_baselines", n, s, METHODS[0]),
                    (f"rollouts/update_{update:03d}", n, s, METHODS[2], current, True)])
            rows = parallel(jobs, f"on-policy block={update+1}")
            grouped = {}
            for row in rows:
                grouped.setdefault((row["instance"], row["algorithm_seed"]), {})[row["strategy"]] = row
            episodes = []
            for pair in block:
                baseline, rollout = grouped[pair][METHODS[0]], grouped[pair][METHODS[2]]
                verify_pair([baseline, rollout])
                if plan["single_intervention"] and len(rollout["transitions"]) != 1:
                    raise ValueError("single-intervention return must be assigned to exactly one learned choice")
                episodes.append({"final_hv": rollout["final_hypervolume"], "balanced_hv": baseline["final_hypervolume"],
                    "transitions": rollout["transitions"]})
            metrics = update_ppo(model, optimizer, episodes, update+1)
            save_model(next_checkpoint, model, optimizer, update+1, metrics)
            log(f"PPO COMMITTED update={update+1} actions={metrics['actions']} mean_terminal_reward={metrics['mean_terminal_reward']:+.6f}")
        current = next_checkpoint
    # Model selection is finished BEFORE any development search outcome is read.
    selection_path = root / "SELECTED_POLICY.json"
    if selection_path.exists() and read_json(selection_path) != selected:
        raise ValueError("selected policy changed on resume")
    write_json(selection_path, selected)
    jobs = []
    for n in plan["test_instances"]:
        for i, s in enumerate(plan["test_seeds"]):
            for method in METHODS[i % 3:]+METHODS[:i % 3]:
                cp = Path(selected["checkpoint"]) if method == METHODS[2] else (
                    initial if plan["single_intervention"] and method == METHODS[1] else None)
                jobs.append(("development", n, s, method, cp))
    log("FROZEN THREE-METHOD DEVELOPMENT: CCP100 / POP20 / B400")
    runs = parallel(jobs, "development")
    report = report_search(runs, plan)
    passed = report["development_gain_established"] and selected["update"] > 0
    # Include every executed objective evaluation, including matched reward baselines.
    receipts = [read_json(p) for p in root.glob("**/RUN_COMPLETE.json") if "interrupted_runs" not in p.parts]
    interrupted = list((root / "interrupted_runs").glob("*")) if (root / "interrupted_runs").exists() else []
    complete = {"completed": True, **report, "selected": selected, "training_summary": {
        "completed_ppo_updates": plan["maximum_ppo_updates"], "episodes": plan["training_episode_count"],
        "optimizer_steps": sum(m["optimizer_steps"] for m in training_metrics),
        "training_and_reward_baseline_evaluations": 2*400*plan["training_episode_count"],
        "completed_run_objective_evaluations": sum(r["real_objective_evaluation_count"] for r in receipts),
        "completed_search_runs": len(receipts), "new_counterfactual_evaluations": 0,
        "interrupted_runs": len(interrupted), "uncommitted_interrupted_cost_not_in_total": bool(interrupted),
        "pretraining_reused": True, "prior_pretraining_cost_not_in_new_evaluation_total": True},
        "development_pass": passed, "conclusion": "PROMISING_REQUIRES_NEW_INSTANCES_AND_OOS" if passed else "BOUNDED_PPO_TRIAL_NO_RELIABLE_GAIN_ESTABLISHED",
        "protocol": plan["protocol"], "single_intervention": plan["single_intervention"],
        "validation_runs_per_checkpoint": len(plan["validation_instances"])*len(plan["validation_seeds"]),
        "validation_checkpoint_candidates": plan["validation_updates"],
        "qualification": "These two development instances cannot establish independent generalization. Conditional seed intervals are not population-of-instance intervals. No automatic retraining or new-instance selection.",
        "equal_test_parameters": True, "scenario_exact": True}
    write_json(root / "COMPLETE.json", {**complete, "runs": runs, "ppo_updates": training_metrics, "validation": validations})
    write_json(root / "QUICK_RESULT.json", complete)
    write_json(root / "STATUS.json", {"stage": "complete", "conclusion": complete["conclusion"]})
    with (root / "RESULT_TABLE.csv").open("w") as handle:
        writer = csv.writer(handle)
        writer.writerow(["instance", "method", "mean_hv", "mean_seconds", "paired_percent_vs_balanced"])
        for item in report["per_instance"]:
            for method in METHODS:
                stats = item["methods"][method]
                effect = 0. if method == METHODS[0] else item["pretrained_vs_balanced" if method == METHODS[1] else "ppo_vs_balanced"]["mean"]
                writer.writerow([item["instance"], method, stats["mean_hv"], stats["mean_seconds"], effect])
    (root / "PIPELINE_COMPLETE").write_text("complete; inspect development_pass, not this marker, for performance\n")
    log(f"COMPLETE development_pass={passed}; selected_update={selected['update']}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=SOURCE / "data/data_expanded.xlsx")
    p.add_argument("--instances", type=Path)
    p.add_argument("--normalization", type=Path, required=True)
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--workers", type=int, choices=(1, 2), default=2)
    p.add_argument("--worker", action="store_true")
    p.add_argument("--instance", type=Path)
    p.add_argument("--seed", type=int)
    p.add_argument("--method", choices=METHODS)
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--checkpoint-sha")
    p.add_argument("--sample", action="store_true")
    p.add_argument("--single-intervention", action="store_true",
        help="v2: one learned choice per search; identical sampling policy in training and evaluation")
    p.add_argument("--plan-sha")
    a = p.parse_args()
    if a.worker:
        if a.instance is None or a.seed is None or a.method is None or not a.plan_sha:
            p.error("worker requires instance, seed, method and plan-sha")
        worker(a)
        return
    if a.instances is None:
        p.error("coordinator requires --instances")
    a.out.mkdir(parents=True, exist_ok=True)
    with (a.out / ".pipeline.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            execute(a)
        except Exception as exc:
            write_json(a.out / "STOPPED.json", {"reason": str(exc), "resume": "same command and unchanged inputs reuse completed searches and committed PPO updates"})
            log(f"STOPPED {exc}")
            raise


if __name__ == "__main__":
    main()
