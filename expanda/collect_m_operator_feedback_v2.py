#!/usr/bin/env python3
"""Capacity-aware, outcome-blind state selection followed by verified replay."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import uuid

import collect_m_operator_feedback_v1 as common
from analyze_m_operator_oracle_audit import analyze_runs, phase, read_jsonl
from inspect_m_operator_sampling_v1 import iter_jsonl
from transformer_ppo.audit_sampling import PHASES, select_phase_states


def freeze_json(path, value):
    if path.exists() and common.read_json(path) != value:
        raise ValueError(f"immutable record changed; preserve outputs and use a new root: {path}")
    if not path.exists():
        common.write_json(path, value)


def validate_episode(run, instance, seed, plan):
    config = common.read_json(run / "configuration.json")
    result = common.read_json(run / "COMPLETE.json")
    expected = {"component": "M", "instance_id": instance, "evaluation_mode": "ccp",
                "population": 20, "generations": 1000, "evaluation_budget": 400,
                "training_size": 100, "alpha": .9, "policy_training": False}
    for name, value in expected.items():
        if config.get(name) != value:
            raise ValueError(f"{run}: wrong {name}: {config.get(name)!r}")
    if (config["seeds"]["algorithm"] != seed
            or config["uncertainty"]["name"] != "transport_border_saa_v1"
            or config["uncertainty"]["scenario_count"] != 100
            or config["policy_checkpoint"]["checkpoint_sha256"] != plan["inputs"]["checkpoint"]["sha256"]
            or config["policy_inference"]["action_mode"] != "argmax"
            or config["policy_inference"]["policy_seed"] != 68101):
        raise ValueError(f"seed, checkpoint, inference or CCP settings differ: {run}")
    if (result.get("completed") is not True or result.get("ppo_updates") != 0
            or result.get("mutation_control") != "operator"
            or result.get("stop_reason") != "evaluation_budget"
            or not 20 <= result.get("real_objective_evaluation_count", -1) <= 400):
        raise ValueError(f"incomplete/incorrect frozen B400 run: {run}")
    return config, result


def inventory(run, instance, seed, plan):
    """Read only pre-action states and run metadata; never read audit rewards."""
    config, result = validate_episode(run, instance, seed, plan)
    opportunities = {name: [] for name in PHASES}
    signatures, ids = {}, []
    trace = hashlib.sha256()
    for row in iter_jsonl(run / "mutation_policy_states.jsonl"):
        state_id = row["policy_state_id"]
        if row["instance_id"] != instance or state_id != len(ids) + 1:
            raise ValueError(f"nonsequential policy states or wrong instance: {run}")
        mask, candidates = row["observation"]["eligible"], row["candidates"]
        if len(mask) != 5 or len(candidates) != 5 or any(
                bool(mask[index]) != bool(candidate["eligible"]) for index, candidate in enumerate(candidates)):
            raise ValueError(f"five-operator policy mask mismatch: {run}: {state_id}")
        ids.append(state_id)
        signature = common.json_hash(row)
        trace.update(signature.encode())
        if sum(bool(value) for value in mask) >= 2:
            opportunities[phase(row["evaluation"] / 400)].append(state_id)
            signatures[str(state_id)] = common.json_hash({
                key: row[key] for key in ("instance_id", "generation", "evaluation", "observation")})
    selection = select_phase_states(opportunities)
    return {
        "instance": instance, "seed": seed, "source_run": str(run),
        "policy_state_count": len(ids), "trace_sha256": trace.hexdigest(),
        "scenario_digest": config["scenario_digest"],
        "search_evaluations": result["real_objective_evaluation_count"],
        "source_counterfactual_evaluations": result.get("candidate_audit_evaluation_count", 0),
        "final_hypervolume": result["final_hypervolume"],
        "front_sha256": common.digest(run / "final_archive_lineage.json"),
        "selected_signatures": {str(key): signatures[str(key)] for key in selection["policy_state_ids"]},
        **selection,
    }


def launch(run, command, log_root, attempts):
    """Retry process failures, preserving all logs. Never retry a quality failure."""
    if (run / "COMPLETE.json").exists():
        return
    for attempt in range(1, attempts + 1):
        if run.exists():
            archive = log_root.parent / "interrupted_runs" / f"{run.name}_{uuid.uuid4().hex[:8]}"
            archive.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(run), str(archive))
        run.parent.mkdir(parents=True, exist_ok=True)
        log_root.mkdir(parents=True, exist_ok=True)
        log = log_root / f"{run.name}_attempt{attempt}_{uuid.uuid4().hex[:8]}.log"
        print(f"[M-DATA-V2] start {run.name} attempt={attempt}/{attempts} log={log}", flush=True)
        with log.open("w") as handle:
            process = subprocess.run(command, cwd=common.SCRIPT_ROOT,
                                     env={**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"},
                                     stdout=handle, stderr=subprocess.STDOUT)
        if process.returncode == 0 and (run / "COMPLETE.json").exists():
            return
        if attempt == attempts:
            raise RuntimeError(f"process exit={process.returncode}; see {log}")


def discovery(root, source_root, instance, seed, plan, attempts):
    if source_root is not None:
        source = source_root / "runs" / f"{instance}_seed_{seed}"
        if (source / "COMPLETE.json").exists():
            old_plan = common.read_json(source_root / "PLAN.json")
            request = common.read_json(source_root / "requests" / (source.name + ".json"))
            if request != {"plan_sha256": common.json_hash(old_plan), "instance": instance, "seed": seed}:
                raise ValueError(f"source run provenance mismatch: {source}")
            for name, value in plan["inputs"].items():
                if old_plan["inputs"][name]["sha256"] != value["sha256"]:
                    raise ValueError(f"source input changed: {name}")
            result = inventory(source, instance, seed, plan)
            result["discovery_source"] = "existing_trajectory"
            return result
    run = root / "discovery" / f"{instance}_seed_{seed}"
    command = common.command_for(plan, instance, seed, run)
    command[command.index("--candidate-audit-states") + 1] = "0"
    launch(run, command, root / "logs/discovery", attempts)
    config, result = validate_episode(run, instance, seed, plan)
    if config.get("candidate_audit") is not None or result.get("candidate_audit_evaluation_count") != 0:
        raise ValueError(f"discovery must not evaluate counterfactual children: {run}")
    value = inventory(run, instance, seed, plan)
    value["discovery_source"] = "new_frozen_trajectory_no_audit"
    return value


def validate_replay(run, selection, plan):
    config, result = validate_episode(run, selection["instance"], selection["seed"], plan)
    if config.get("candidate_audit") != {
            "states": 32, "candidates": 5, "replicates": 3, "policy_state_ids": selection["policy_state_ids"]}:
        raise ValueError(f"audit state-id plan differs: {run}")
    replay = inventory(run, selection["instance"], selection["seed"], plan)
    for key in ("trace_sha256", "scenario_digest", "search_evaluations", "final_hypervolume", "front_sha256"):
        if replay[key] != selection[key]:
            raise ValueError(f"replay changed {key}; no approval or automatic resampling: {run}")
    raw = read_jsonl(run / "mutation_candidate_audit_states.jsonl")
    if [row.get("policy_state_id") for row in raw] != selection["policy_state_ids"]:
        raise ValueError(f"requested distinct audit states were not all executed: {run}")
    for row in raw:
        actual = common.json_hash({key: row[key] for key in ("instance_id", "generation", "evaluation", "observation")})
        if actual != selection["selected_signatures"][str(row["policy_state_id"])]:
            raise ValueError(f"audited pre-action snapshot differs from source policy state: {run}")
    summary, states = analyze_runs([run], expected_replicates=3, expected_state_count=32)
    outcomes = read_jsonl(run / "mutation_candidate_audit_outcomes.jsonl")
    if (not summary["completeness_pass"] or summary["phase_counts"] != selection["quotas"]
            or result.get("candidate_audit_state_count") != 32
            or result.get("candidate_audit_evaluation_count") != len(outcomes)
            or any(row.get("execution_contract") != common.CONTRACT for row in outcomes)):
        raise ValueError(f"incomplete legal feedback, wrong quotas or execution contract: {run}")
    return {"instance": selection["instance"], "seed": selection["seed"], "run": str(run),
            "states": 32, "phase_counts": summary["phase_counts"],
            "available_by_phase": selection["available_by_phase"],
            "capacity_limited_phases": selection["capacity_limited_phases"],
            "replay_exact": True, "scenario_digest": config["scenario_digest"],
            "search_evaluations": result["real_objective_evaluation_count"],
            "counterfactual_evaluations": len(outcomes), "ppo_updates": 0}


def replay_one(root, selection, plan, attempts):
    run = root / "runs" / f"{selection['instance']}_seed_{selection['seed']}"
    command = common.command_for(plan, selection["instance"], selection["seed"], run)
    command += ["--candidate-audit-policy-state-ids", *map(str, selection["policy_state_ids"])]
    launch(run, command, root / "logs/replay", attempts)
    return validate_replay(run, selection, plan)


def parallel_stage(root, batch, label, items, work, workers):
    completed, failures = [], []
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = {pool.submit(work, item): item for item in items}
        while pending:
            done, _ = wait(pending, timeout=15, return_when=FIRST_COMPLETED)
            for future in done:
                pending.pop(future)
                try:
                    value = future.result()
                    completed.append(value)
                    print(f"[M-DATA-V2] {label} {len(completed)}/{len(items)} complete: {value['instance']}", flush=True)
                except Exception as error:
                    failures.append(str(error))
                    print(f"[M-DATA-V2] FAILED {label}: {error}", flush=True)
            status = {"batch": batch, "stage": label, "completed_tasks": len(completed),
                      "tasks": len(items), "failures": failures, "elapsed_seconds": time.monotonic() - started}
            common.write_json(root / "STATUS.json", status)
            print(f"[M-DATA-V2] {label} progress={len(completed)}/{len(items)} elapsed={status['elapsed_seconds']:.0f}s failures={len(failures)}", flush=True)
    if failures:
        raise RuntimeError(f"{label} failed: {failures[0]}")
    return sorted(completed, key=lambda value: value["instance"])


def finish(root, batch, selections, plan):
    if batch == 2:
        selections = common.read_json(root / "batch_1/SAMPLING_PLAN.json")["trajectories"] + selections
    checked = [validate_replay(root / "runs" / f"{s['instance']}_seed_{s['seed']}", s, plan) for s in selections]
    for instance in common.INSTANCE_IDS:
        if len({row["scenario_digest"] for row in checked if row["instance"] == instance}) != 1:
            raise ValueError(f"CCP scenario digest differs across algorithm seeds: {instance}")
    summary, states = analyze_runs([row["run"] for row in checked], expected_replicates=3, expected_state_count=192 * batch)
    if not summary["completeness_pass"]:
        raise ValueError("aggregate feedback incomplete")
    source_cost = sum(row["search_evaluations"] for row in selections)
    new_discovery_cost = sum(row["search_evaluations"] for row in selections if row["discovery_source"] != "existing_trajectory")
    quality = {"completed": True, "data_quality_pass": True, "protocol": plan["protocol"],
               "plan_sha256": common.json_hash(plan), "batch": batch, "state_count": len(states),
               "trajectory_count": len(checked), "instance_count": 6,
               "scenario_count": 100, "population_size": 20, "evaluation_budget_per_trajectory": 400,
               "phase_counts": summary["phase_counts"], "replay_exact": True, "new_training_steps": 0,
               "discovery_search_evaluations_including_reused": source_cost,
               "historical_source_counterfactual_evaluations": sum(row["source_counterfactual_evaluations"] for row in selections),
               "new_discovery_search_evaluations": new_discovery_cost,
               "replay_search_evaluations": sum(row["search_evaluations"] for row in checked),
               "counterfactual_evaluations": summary["counterfactual_evaluation_count"],
               "checked_runs": checked,
               "qualification": "capacity-aware protocol amendment; quality only, not model performance",
               "next_action": "collect fixed batch 2 regardless of gain" if batch == 1 else "grouped offline comparison; no automatic PPO"}
    # Preserve trajectory identities and repeats, including capacity-limited phases.
    common.write_json(root / f"batch_{batch}/DATASET.json", {"plan_sha256": common.json_hash(plan),
                     "split_unit": "instance then trajectory; keep all repeats together", "states": states})
    common.write_json(root / f"batch_{batch}/QUALITY.json", quality)
    common.write_json(root / f"BATCH_{batch}_COMPLETE", {"plan_sha256": common.json_hash(plan), "states": len(states)})
    common.write_json(root / "STATUS.json", {"batch": batch, "stage": "complete", "state_count": len(states)})
    return quality


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instances", type=Path, required=True)
    parser.add_argument("--normalization", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=common.SCRIPT_ROOT / "data/data_expanded.xlsx")
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--batch", type=int, choices=(1, 2), default=1)
    parser.add_argument("--workers", type=int, choices=(1, 2, 3), default=2)
    parser.add_argument("--max-attempts", type=int, choices=(1, 2, 3), default=2)
    args = parser.parse_args(argv)
    root = args.out.resolve()
    source_root = args.source_root.resolve() if args.source_root else None
    if root == source_root:
        parser.error("use a new output root; the original collection must remain unchanged")
    root.mkdir(parents=True, exist_ok=True)
    with (root / "collection.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.exit(75, "Another collection is running in this directory.\n")
        try:
            if source_root is not None and not (source_root / "PLAN.json").is_file():
                raise ValueError(f"source collection PLAN.json is missing: {source_root}")
            plan = common.make_plan(args.instances.resolve(), args.normalization.resolve(), args.checkpoint.resolve(), args.data.resolve())
            plan.update(protocol="m-shared-operator-feedback-v2-capacity-replay",
                        amendment_reason="v1 required eight early states when one logged trajectory had only six; threshold sampling also missed available states; revised using opportunity counts, not gains",
                        source_root=str(source_root) if source_root else None,
                        phase_quality_rule="10/11/11 capped by available pre-action opportunities; redistribute shortages cyclically; exact selected ids replayed",
                        sampling_rule="midpoint quantiles within each phase; no outcomes used; total 32 distinct states; at least one per phase",
                        verification="full pre-action policy trace, selected observations, final front and HV equal discovery",
                        cost_accounting="discovery and replay each use B400; audit evaluation costs reported separately")
            common.freeze_plan(root, plan)
            if args.batch == 2:
                prior = common.read_json(root / "batch_1/QUALITY.json")
                if prior.get("data_quality_pass") is not True or prior.get("plan_sha256") != common.json_hash(plan):
                    raise ValueError("batch 2 requires matching batch 1 quality completion")
            seed = common.ALGORITHM_SEEDS[args.batch - 1]
            print(f"===== M-DATA-V2 BATCH {args.batch}: INVENTORY / CCP100 / POP20 / B400 =====", flush=True)
            selections = parallel_stage(root, args.batch, "inventory", common.INSTANCE_IDS,
                lambda name: discovery(root, source_root, name, seed, plan, args.max_attempts), args.workers)
            freeze_json(root / f"batch_{args.batch}/SAMPLING_PLAN.json",
                        {"plan_sha256": common.json_hash(plan), "trajectories": selections})
            for selection in selections:
                print(f"[M-DATA-V2] quota {selection['instance']}: available={selection['available_by_phase']} selected={selection['quotas']}", flush=True)
            print("===== M-DATA-V2 REPLAY PRESELECTED STATES / NO TRAINING =====", flush=True)
            parallel_stage(root, args.batch, "replay", selections,
                           lambda selection: replay_one(root, selection, plan, args.max_attempts), args.workers)
            # An update while running invalidates the collection, never mixes versions.
            current = common.make_plan(args.instances.resolve(), args.normalization.resolve(), args.checkpoint.resolve(), args.data.resolve())
            if current["source_hashes"] != plan["source_hashes"] or current["inputs"] != plan["inputs"]:
                raise ValueError("code or inputs changed during collection")
            quality = finish(root, args.batch, selections, plan)
            print(json.dumps({key: quality[key] for key in ("data_quality_pass", "state_count", "phase_counts", "replay_exact",
                             "new_discovery_search_evaluations", "replay_search_evaluations", "counterfactual_evaluations", "next_action")}, indent=2), flush=True)
            print(f"===== M-DATA-V2 BATCH {args.batch} COMPLETE =====", flush=True)
        except Exception as error:
            common.write_json(root / "STATUS.json", {"batch": args.batch, "stage": "failed", "error": str(error)})
            parser.exit(1, f"[M-DATA-V2] STOPPED: {error}\n")


if __name__ == "__main__":
    main()
