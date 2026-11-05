#!/usr/bin/env python3
"""Collect a predeclared 192+192 state panel; no fitting or policy updates."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import uuid

from analyze_m_operator_oracle_audit import analyze_runs, read_jsonl


INSTANCE_IDS = ("train-S0-002", "train-S0-005", "train-S0-006",
                "train-S0-008", "train-S0-009", "train-S0-011")
ALGORITHM_SEEDS = (891001, 891002)
STATES_PER_RUN = 32
CONTRACT = "operator_shared_v1"
SCRIPT_ROOT = Path(__file__).resolve().parent


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def make_plan(instances, normalization, checkpoint, data):
    if read_json(normalization).get("source") != "training-only":
        raise ValueError("normalization must declare source=training-only")
    files = {"normalization": normalization, "checkpoint": checkpoint, "workbook": data,
             "border_events": SCRIPT_ROOT / "data/border_crossing_events.csv"}
    workbook_digest = digest(data)
    for name in INSTANCE_IDS:
        path = instances / (name + ".json")
        manifest = read_json(path)
        if manifest["instance_id"] != name:
            raise ValueError(f"instance identity differs from filename: {path}")
        if manifest["source_data_sha256"] != workbook_digest:
            raise ValueError(f"workbook differs from instance source: {path}")
        files[name] = path
    sources = sorted(set(SCRIPT_ROOT.glob("*.py")) | set((SCRIPT_ROOT / "transformer_ppo").rglob("*.py")))
    return {
        "protocol": "m-shared-operator-feedback-v1", "execution_contract": CONTRACT,
        "instance_ids": list(INSTANCE_IDS), "algorithm_seeds": list(ALGORITHM_SEEDS),
        "states_per_run": STATES_PER_RUN, "replicates_per_legal_operator": 3,
        "batches": {"1": {"seed": ALGORITHM_SEEDS[0], "states": 192},
                    "2": {"seed": ALGORITHM_SEEDS[1], "additional_states": 192, "cumulative_states": 384}},
        "scenario_count": 100, "ccp_confidence": .9, "population_size": 20,
        "evaluation_budget": 400, "generation_cap": 1000,
        "uncertainty_profile": "transport_border_saa_v1",
        "sampling_policy": "frozen V6 operator Transformer; argmax; no PPO updates",
        "phase_quality_rule": "each trajectory: 8 to 14 states in each of early/middle/late evaluation-budget thirds",
        "second_batch_rule": "same instances; fixed second seed; data-quality checks only; never select by gain",
        "split_unit": "instance; retain trajectory and all repeats of each state together",
        "inputs": {name: {"path": str(path.resolve()), "sha256": digest(path)} for name, path in files.items()},
        "source_hashes": {str(path.relative_to(SCRIPT_ROOT)): digest(path) for path in sources},
    }


def freeze_plan(root, plan):
    target = root / "PLAN.json"
    if target.exists():
        if read_json(target) != plan:
            raise ValueError("PLAN/input/code mismatch: preserve this collection and use a new COLLECTION_ROOT")
    else:
        write_json(target, plan)
    return json_hash(plan)


def command_for(plan, instance, seed, run):
    inputs = plan["inputs"]
    return [sys.executable, "-u", str(SCRIPT_ROOT / "run_learning_components_deterministic.py"),
            "--data", inputs["workbook"]["path"], "--instance", inputs[instance]["path"],
            "--normalization", inputs["normalization"]["path"], "--out", str(run),
            "--component", "M", "--checkpoint", inputs["checkpoint"]["path"],
            "--inference-action-mode", "argmax", "--inference-policy-seed", "68101",
            "--model-choice-probability", "1", "--algorithm-seed", str(seed),
            "--evaluation-mode", "ccp", "--uncertainty-profile", "transport_border_saa_v1",
            "--scenario-count", "100", "--ccp-confidence", "0.90",
            "--evaluation-budget", "400", "--pop", "20", "--gens", "1000",
            "--candidate-audit-states", "32", "--candidate-audit-candidates", "5",
            "--candidate-audit-replicates", "3", "--device", "cpu"]


def validate_run(run, instance, seed, plan, plan_hash):
    request = read_json(run.parent.parent / "requests" / (run.name + ".json"))
    if request != {"plan_sha256": plan_hash, "instance": instance, "seed": seed}:
        raise ValueError(f"run provenance mismatch: {run}")
    result, config = read_json(run / "COMPLETE.json"), read_json(run / "configuration.json")
    expected = {"component": "M", "evaluation_mode": "ccp", "instance_id": instance,
                "population": 20, "evaluation_budget": 400, "training_size": 100,
                "generations": 1000, "alpha": .9, "policy_training": False,
                "candidate_audit": {"states": 32, "candidates": 5, "replicates": 3}}
    for name, value in expected.items():
        if config.get(name) != value:
            raise ValueError(f"wrong {name} at {run}: {config.get(name)!r}")
    if (config.get("seeds", {}).get("algorithm") != seed
            or config.get("uncertainty", {}).get("name") != "transport_border_saa_v1"
            or config.get("uncertainty", {}).get("scenario_count") != 100
            or not config.get("scenario_digest")
            or (config.get("policy_checkpoint") or {}).get("checkpoint_sha256") != plan["inputs"]["checkpoint"]["sha256"]
            or (config.get("policy_inference") or {}).get("action_mode") != "argmax"
            or (config.get("policy_inference") or {}).get("policy_seed") != 68101):
        raise ValueError(f"seed/scenario/checkpoint mismatch: {run}")
    evaluations = result.get("real_objective_evaluation_count", -1)
    if (result.get("completed") is not True or result.get("ppo_updates") != 0
            or result.get("mutation_control") != "operator"
            or result.get("stop_reason") != "evaluation_budget"
            or not 20 <= evaluations <= 400):
        raise ValueError(f"run incomplete, wrong control/budget, or policy was trained: {run}")
    summary, states = analyze_runs([run], expected_replicates=3, expected_state_count=32)
    if not summary["completeness_pass"]:
        raise ValueError(f"incomplete legal feedback: {run}: {summary['excluded_states'][:2]}")
    if any(not 8 <= summary["phase_counts"].get(phase, 0) <= 14 for phase in ("early", "middle", "late")):
        raise ValueError(f"phase coverage outside frozen 8..14 rule: {run}: {summary['phase_counts']}")
    outcomes = read_jsonl(run / "mutation_candidate_audit_outcomes.jsonl")
    if any(row.get("execution_contract") != CONTRACT for row in outcomes):
        raise ValueError(f"unverified legacy execution labels: {run}")
    if any(row["instance"] != instance or not 0 <= row["evaluation"] <= evaluations for row in states):
        raise ValueError(f"state identity/evaluation mismatch: {run}")
    if result.get("candidate_audit_evaluation_count") != len(outcomes):
        raise ValueError(f"counterfactual evaluation accounting mismatch: {run}")
    return {"instance": instance, "seed": seed, "run": str(run), "states": len(states),
            "phase_counts": summary["phase_counts"], "legal_operator_count_distribution": summary["legal_operator_count_distribution"],
            "search_objective_evaluations": evaluations, "counterfactual_evaluations": len(outcomes),
            "scenario_digest": config["scenario_digest"], "ppo_updates": 0,
            "optimisation_seconds_including_audit": result.get("optimisation_seconds"),
            "data_sha256": {name: digest(run / name) for name in (
                "mutation_candidate_audit_states.jsonl", "mutation_candidate_audit_outcomes.jsonl")}}


def collect_one(root, instance, seed, plan, plan_hash, max_attempts):
    run = root / "runs" / f"{instance}_seed_{seed}"
    if (run / "COMPLETE.json").exists():
        return {**validate_run(run, instance, seed, plan, plan_hash), "status": "existing"}
    request_path = root / "requests" / (run.name + ".json")
    request = {"plan_sha256": plan_hash, "instance": instance, "seed": seed}
    if request_path.exists() and read_json(request_path) != request:
        raise ValueError(f"incomplete run provenance mismatch: {run}")
    # The episode runner requires an empty output directory. Keep launch
    # provenance outside it, including if the process stops before completion.
    write_json(request_path, request)
    run.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, max_attempts + 1):
        if run.exists():
            archived = root / "interrupted_runs" / f"{run.name}_{uuid.uuid4().hex[:8]}"
            archived.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(run), str(archived))
        log = root / "logs" / f"{instance}_seed_{seed}_attempt_{attempt}_{uuid.uuid4().hex[:8]}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        print(f"[M-DATA] start {instance} seed={seed} attempt={attempt}/{max_attempts} log={log}", flush=True)
        with log.open("w", encoding="utf-8") as handle:
            process = subprocess.run(command_for(plan, instance, seed, run), cwd=SCRIPT_ROOT,
                                     env={**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"},
                                     stdout=handle, stderr=subprocess.STDOUT)
        if process.returncode == 0 and (run / "COMPLETE.json").exists():
            # A completed but low-quality trajectory is not resampled to obtain
            # a favourable outcome. Only interrupted/failed processes retry.
            return {**validate_run(run, instance, seed, plan, plan_hash), "status": "complete"}
        if attempt == max_attempts:
            raise RuntimeError(f"rollout exit={process.returncode}; see {log}")
    raise AssertionError("unreachable")


def line_count(path):
    try:
        with path.open("rb") as handle:
            return sum(chunk.count(b"\n") for chunk in iter(lambda: handle.read(1024 * 1024), b""))
    except FileNotFoundError:
        return 0


def finish_batch(root, batch, plan, plan_hash):
    selected = [(instance, seed) for seed in ALGORITHM_SEEDS[:batch] for instance in INSTANCE_IDS]
    verified = [validate_run(root / "runs" / f"{instance}_seed_{seed}", instance, seed, plan, plan_hash)
                for instance, seed in selected]
    if batch == 2:
        for instance in INSTANCE_IDS:
            digests = {row["scenario_digest"] for row in verified if row["instance"] == instance}
            if len(digests) != 1:
                raise ValueError(f"CCP scenarios changed between algorithm seeds: {instance}")
    summary, states = analyze_runs([row["run"] for row in verified], expected_state_count=192 * batch)
    if not summary["completeness_pass"]:
        raise ValueError("aggregate legal-action feedback completeness check failed")
    target = root / f"batch_{batch}"
    quality = {"completed": True, "data_quality_pass": True, "batch": batch,
               "state_count": len(states), "instance_count": 6, "trajectory_count": len(selected),
               "scenario_count": 100, "ccp_confidence": .9, "population_size": 20,
               "evaluation_budget_per_trajectory": 400, "phase_counts": summary["phase_counts"],
               "search_objective_evaluations": sum(row["search_objective_evaluations"] for row in verified),
               "counterfactual_evaluations": sum(row["counterfactual_evaluations"] for row in verified),
               "new_training_steps": 0, "sampling_policy": plan["sampling_policy"],
               "plan_sha256": plan_hash, "checked_runs": verified,
               "qualification": "collection quality only; no claim of learnability or final HV improvement",
               "next_action": "review data quality, then collect fixed batch 2 regardless of gain" if batch == 1
                              else "grouped offline model comparison; no automatic PPO"}
    write_json(target / "QUALITY.json", quality)
    write_json(target / "DATASET.json", {"plan_sha256": plan_hash, "execution_contract": CONTRACT,
                                         "split_unit": plan["split_unit"], "states": states})
    write_json(root / f"BATCH_{batch}_COMPLETE", {"plan_sha256": plan_hash, "states": len(states)})
    return quality


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instances", type=Path, required=True)
    parser.add_argument("--normalization", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=SCRIPT_ROOT / "data/data_expanded.xlsx")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--batch", type=int, choices=(1, 2), default=1)
    parser.add_argument("--workers", type=int, choices=(1, 2, 3), default=2)
    parser.add_argument("--max-attempts", type=int, choices=(1, 2, 3), default=2)
    args = parser.parse_args(argv)
    root = args.out.resolve()
    root.mkdir(parents=True, exist_ok=True)
    with (root / "collection.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.exit(75, "Another collection owns this output directory.\n")
        started = time.monotonic()
        status = {"batch": args.batch, "state": "preflight", "completed_tasks": 0, "tasks": 6, "failures": []}
        try:
            plan = make_plan(args.instances.resolve(), args.normalization.resolve(), args.checkpoint.resolve(), args.data.resolve())
            plan_hash = freeze_plan(root, plan)
            if args.batch == 2:
                prior = read_json(root / "batch_1/QUALITY.json")
                if prior.get("data_quality_pass") is not True or prior.get("plan_sha256") != plan_hash:
                    raise ValueError("batch 2 requires matching, completed batch 1 data quality checks")
            status["state"] = "collecting"
            write_json(root / "STATUS.json", status)
            seed = ALGORITHM_SEEDS[args.batch - 1]
            print(f"[M-DATA] batch={args.batch} jobs=6 workers={args.workers} seed={seed} new_states=192 CCP100/pop20/B400", flush=True)
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                pending = {pool.submit(collect_one, root, instance, seed, plan, plan_hash, args.max_attempts): instance
                           for instance in INSTANCE_IDS}
                while pending:
                    done, _ = wait(pending, timeout=15, return_when=FIRST_COMPLETED)
                    for future in done:
                        instance = pending.pop(future)
                        try:
                            result = future.result()
                            status["completed_tasks"] += 1
                            print(f"[M-DATA] {status['completed_tasks']}/6 {result['status']}: {instance} states=32 phases={result['phase_counts']}", flush=True)
                        except Exception as error:
                            status["failures"].append({"instance": instance, "error": str(error)})
                            print(f"[M-DATA] FAILED {instance}: {error}", flush=True)
                    status["observed_state_lines"] = sum(line_count(root / "runs" / f"{name}_seed_{seed}" / "mutation_candidate_audit_states.jsonl") for name in INSTANCE_IDS)
                    status["elapsed_seconds"] = time.monotonic() - started
                    status["updated_utc"] = datetime.now(timezone.utc).isoformat()
                    write_json(root / "STATUS.json", status)
                    print(f"[M-DATA] progress tasks={status['completed_tasks']}/6 states={status['observed_state_lines']}/192 elapsed={status['elapsed_seconds']:.0f}s failures={len(status['failures'])}", flush=True)
            if status["failures"]:
                raise RuntimeError("some trajectories failed; see STATUS.json; valid completed runs are reusable")
            # Changes during collection invalidate the panel instead of silently
            # mixing versions; successful trajectory outputs remain preserved.
            freeze_plan(root, make_plan(args.instances.resolve(), args.normalization.resolve(), args.checkpoint.resolve(), args.data.resolve()))
            status["state"] = "checking_quality"
            write_json(root / "STATUS.json", status)
            quality = finish_batch(root, args.batch, plan, plan_hash)
            status.update(state="complete", state_count=quality["state_count"], elapsed_seconds=time.monotonic() - started)
            write_json(root / "STATUS.json", status)
            print(json.dumps({key: quality[key] for key in ("data_quality_pass", "state_count", "phase_counts", "search_objective_evaluations", "counterfactual_evaluations", "new_training_steps", "next_action")}, indent=2), flush=True)
            print(f"===== M-DATA BATCH {args.batch} COMPLETE =====", flush=True)
        except Exception as error:
            status.update(state="failed", error=str(error), elapsed_seconds=time.monotonic() - started)
            write_json(root / "STATUS.json", status)
            parser.exit(1, f"[M-DATA] STOPPED: {error}\n")


if __name__ == "__main__":
    main()
