#!/usr/bin/env python3
"""Audit unique one-shot M candidates through the final CCP search budget.

The collector first runs an exact Base control and an M controller with zero
model probability.  It spreads target events across each Base trajectory,
probes the frozen M observation at each target, and then deterministically
executes distinct eligible mutation candidates.  Every later mutation returns
to exact Base control.  Frozen CCP scenarios and algorithm seeds make final-HV
differences attributable to the single forced mutation intervention.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys


def _read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl(path: Path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _sha256(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _short_id(instance_id: str):
    return instance_id.removeprefix("train-").replace("-", "_")


def _command(args, instance_path, run, seed, component,
             model_choice_probability=None, model_choice_event=None,
             forced_candidate_index=None, policy_seed=0):
    command = [
        sys.executable, "-u", str(args.runner),
        "--instance", str(instance_path),
        "--normalization", str(args.normalization),
        "--out", str(run),
        "--component", component,
        "--algorithm-seed", str(seed),
        "--evaluation-mode", args.evaluation_mode,
        "--uncertainty-profile", args.uncertainty_profile,
        "--scenario-count", str(args.scenario_count),
        "--ccp-confidence", str(args.ccp_confidence),
        "--evaluation-budget", str(args.evaluation_budget),
        "--pop", str(args.pop),
        "--gens", str(args.gens),
        "--device", args.device,
    ]
    if component == "M":
        command.extend([
            "--checkpoint", str(args.checkpoint),
            "--inference-action-mode", "argmax",
            "--inference-policy-seed", str(policy_seed),
        ])
    if model_choice_probability is not None:
        command.extend([
            "--model-choice-probability", str(model_choice_probability)])
    if model_choice_event is not None:
        command.extend(["--model-choice-event", str(model_choice_event)])
    if forced_candidate_index is not None:
        command.extend([
            "--m-forced-candidate-index", str(forced_candidate_index)])
    return command


def _target_outcome(run: Path, event: int):
    rows = [
        row for row in _jsonl(run / "mutation_outcomes.jsonl")
        if row.get("mixture_event_index") == event
    ]
    if len(rows) != 1:
        raise RuntimeError(
            f"expected one mutation outcome for event {event}: {run}")
    return rows[0]


def _target_policy_state(run: Path, event: int):
    rows = [
        row for row in _jsonl(run / "mutation_policy_states.jsonl")
        if row.get("mixture_event_index") == event
        and row.get("decision_stage", "location") == "location"
    ]
    if len(rows) != 1:
        raise RuntimeError(
            f"expected one M location state for event {event}: {run}")
    return rows[0]


def _valid_complete(run: Path, seed: int, component: str,
                    event=None, forced_candidate_index=None):
    marker_path = run / "COMPLETE.json"
    config_path = run / "configuration.json"
    if not marker_path.is_file() or not config_path.is_file():
        return False
    try:
        marker = _read(marker_path)
        config = _read(config_path)
        valid = (
            marker.get("completed") is True
            and marker.get("component") == component
            and config.get("component") == component
            and config.get("seeds", {}).get("algorithm") == seed
        )
        if component == "M":
            outcomes = _jsonl(run / "mutation_outcomes.jsonl")
            model_rows = [
                row for row in outcomes
                if row.get("decision_kind") == "model_choice"
            ]
            if event is None:
                valid = valid and not model_rows
            else:
                target = _target_outcome(run, event)
                valid = valid and len(model_rows) == 1
                _target_policy_state(run, event)
                if forced_candidate_index is not None:
                    valid = valid and (
                        int(target.get("selected_candidate_index", -1))
                        == int(forced_candidate_index)
                        and int(target.get("policy_metadata", {}).get(
                            "forced_action_index", -1))
                        == int(forced_candidate_index)
                    )
        return bool(valid)
    except (OSError, TypeError, ValueError, RuntimeError):
        return False


def _run_job(job):
    run, log, command, seed, component, event, forced = job
    if _valid_complete(run, seed, component, event, forced):
        return run, "existing"
    if run.exists():
        raise RuntimeError(
            f"incomplete output already exists; use a fresh root: {run}")
    run.parent.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment.setdefault("OMP_NUM_THREADS", "1")
    environment.setdefault("MKL_NUM_THREADS", "1")
    with log.open("w", encoding="utf-8") as handle:
        completed = subprocess.run(
            command, cwd=Path(__file__).resolve().parent,
            stdout=handle, stderr=subprocess.STDOUT,
            env=environment, check=False)
    if completed.returncode != 0 or not _valid_complete(
            run, seed, component, event, forced):
        raise RuntimeError(
            f"run failed with exit {completed.returncode}; inspect {log}")
    return run, "complete"


def _execute(jobs, workers, stage):
    print(f"[{stage}] jobs={len(jobs)} workers={workers}", flush=True)
    counts = {"existing": 0, "complete": 0}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(_run_job, job) for job in jobs]
        for completed_count, future in enumerate(as_completed(futures), 1):
            run, status = future.result()
            counts[status] += 1
            print(
                f"[{stage}] {completed_count}/{len(jobs)} "
                f"{status}: {run}", flush=True)
    return counts


def _spread_events(base_run: Path, requested: int, maximum_generation=None):
    rows = [
        row for row in _jsonl(base_run / "mutation_outcomes.jsonl")
        if row.get("decision_kind") == "base_mixture_choice"
        and row.get("operator") == "replace"
        and row.get("mixture_event_index") is not None
        and (maximum_generation is None
             or int(row.get("generation", -1)) <= maximum_generation)
    ]
    rows.sort(key=lambda row: int(row["mixture_event_index"]))
    if not rows:
        raise ValueError(f"{base_run} has no usable replace mutation events")
    count = min(int(requested), len(rows))
    if count == 1:
        return [rows[len(rows) // 2]]
    indices = [
        round(offset * (len(rows) - 1) / (count - 1))
        for offset in range(count)
    ]
    if len(set(indices)) != count:
        raise ValueError(f"could not select {count} distinct events: {base_run}")
    return [rows[index] for index in indices]


def _spread_indices(indices, requested):
    values = sorted(set(int(index) for index in indices))
    if not values:
        return []
    count = min(int(requested), len(values))
    if count == 1:
        return [values[len(values) // 2]]
    offsets = [
        round(offset * (len(values) - 1) / (count - 1))
        for offset in range(count)
    ]
    return [values[offset] for offset in offsets]


def _action_signature(outcome):
    names = (
        "operator", "batch_id", "path_index", "arc_index", "new_mode",
        "replacement_path_id", "from_node", "to_node",
    )
    return ":".join(str(outcome.get(name)) for name in names)


def _analyse(args, manifest, counts):
    states = []
    scenario_mismatches = []
    p000_mismatches = []
    duplicate_consistency_failures = []
    forced_execution_failures = []
    for instance in manifest["instances"]:
        short = instance["short_id"]
        for seed_row in instance["seeds"]:
            seed = int(seed_row["algorithm_seed"])
            root = args.out / short / f"seed_{seed}"
            pure_base = root / "BASE"
            p000 = root / "P000"
            base_marker = _read(pure_base / "COMPLETE.json")
            base_hv = float(base_marker["final_hypervolume"])
            base_hash = _sha256(pure_base / "final_archive_lineage.json")
            p000_marker = _read(p000 / "COMPLETE.json")
            p000_hash = _sha256(p000 / "final_archive_lineage.json")
            if (not math.isclose(
                    base_hv, float(p000_marker["final_hypervolume"]),
                    abs_tol=1e-15) or base_hash != p000_hash):
                p000_mismatches.append(f"{short}:{seed}")
            base_digest = _read(
                pure_base / "configuration.json")["scenario_digest"]
            for state in seed_row["states"]:
                event = int(state["event"])
                probe = root / f"event_{event}" / "PROBE"
                if _read(probe / "configuration.json")[
                        "scenario_digest"] != base_digest:
                    scenario_mismatches.append(str(probe))
                records = []
                for candidate_index in state["forced_candidate_indices"]:
                    run = root / f"event_{event}" / f"candidate_{candidate_index}"
                    marker = _read(run / "COMPLETE.json")
                    config = _read(run / "configuration.json")
                    if config["scenario_digest"] != base_digest:
                        scenario_mismatches.append(str(run))
                    outcome = _target_outcome(run, event)
                    final_hv = float(marker["final_hypervolume"])
                    selected = int(outcome.get("selected_candidate_index", -1))
                    if selected != int(candidate_index):
                        forced_execution_failures.append(str(run))
                    record = {
                        "candidate_index": int(candidate_index),
                        "action_signature": _action_signature(outcome),
                        "operator": outcome.get("operator"),
                        "batch_id": outcome.get("batch_id"),
                        "replacement_path_id": outcome.get(
                            "replacement_path_id"),
                        "selected_replacement_executed": outcome.get(
                            "selected_replacement_executed"),
                        "effective_mutation": bool(outcome.get(
                            "effective_mutation")),
                        "survived_selection": outcome.get(
                            "survived_environmental_selection"),
                        "immediate_archive_delta": float(
                            outcome.get("reward_hv_delta", 0.0)),
                        "final_hv": final_hv,
                        "final_percent": 100.0 * (
                            final_hv - base_hv) / base_hv,
                        "final_archive_sha256": _sha256(
                            run / "final_archive_lineage.json"),
                    }
                    records.append(record)
                grouped = {}
                for record in records:
                    grouped.setdefault(
                        record["action_signature"], []).append(record)
                unique = []
                for signature, duplicates in grouped.items():
                    hvs = {round(item["final_hv"], 15) for item in duplicates}
                    hashes = {
                        item["final_archive_sha256"] for item in duplicates}
                    if len(hvs) != 1 or len(hashes) != 1:
                        duplicate_consistency_failures.append(
                            f"{short}:{seed}:{event}:{signature}")
                    unique.append(duplicates[0])
                base_record = {
                    "candidate_index": None,
                    "action_signature": "BASE",
                    "operator": "base_passthrough",
                    "final_hv": base_hv,
                    "final_percent": 0.0,
                    "immediate_archive_delta": 0.0,
                }
                candidates = [base_record, *unique]
                best = max(candidates, key=lambda row: row["final_hv"])
                worst = min(unique, key=lambda row: row["final_hv"])
                states.append({
                    "instance": short,
                    "algorithm_seed": seed,
                    "event": event,
                    "generation": int(state["generation"]),
                    "raw_candidate_count": int(state["candidate_count"]),
                    "eligible_candidate_count": int(
                        state["eligible_candidate_count"]),
                    "audited_candidate_count": len(unique),
                    "base_final_hv": base_hv,
                    "audited_oracle_final_hv": best["final_hv"],
                    "audited_oracle_percent": best["final_percent"],
                    "best_action_signature": best["action_signature"],
                    "worst_candidate_percent": worst["final_percent"],
                    "positive_candidate_fraction": sum(
                        row["final_percent"] > 0.0 for row in unique
                    ) / len(unique),
                    "actions": sorted(
                        unique, key=lambda row: row["final_percent"],
                        reverse=True),
                })
    oracle_values = [row["audited_oracle_percent"] for row in states]
    action_values = [
        action["final_percent"]
        for state in states for action in state["actions"]
    ]
    summary = {
        "completed": True,
        "definition": (
            "one forced M candidate followed by exact Base mutation control "
            "to the same final CCP search budget"),
        "algorithm_seeds": list(args.algorithm_seeds),
        "instance_count": len(manifest["instances"]),
        "state_count": len(states),
        "probe_run_count": len(states),
        "branch_run_count": len(action_values),
        "distinct_action_count": sum(
            row["audited_candidate_count"] for row in states),
        "mean_audited_oracle_percent": statistics.fmean(oracle_values),
        "median_audited_oracle_percent": statistics.median(oracle_values),
        "maximum_audited_oracle_percent": max(oracle_values),
        "states_with_positive_oracle": sum(
            value > 0.0 for value in oracle_values),
        "states_with_oracle_above_5_percent": sum(
            value >= 5.0 for value in oracle_values),
        "states_with_oracle_above_10_percent": sum(
            value >= 10.0 for value in oracle_values),
        "positive_action_count": sum(value > 0.0 for value in action_values),
        "negative_action_count": sum(value < 0.0 for value in action_values),
        "zero_action_count": sum(value == 0.0 for value in action_values),
        "mean_action_percent": statistics.fmean(action_values),
        "p000_exact_base": not p000_mismatches,
        "scenario_exact": not scenario_mismatches,
        "duplicate_replay_consistent": not duplicate_consistency_failures,
        "forced_execution_exact": not forced_execution_failures,
        "p000_mismatches": p000_mismatches,
        "scenario_mismatches": scenario_mismatches,
        "duplicate_consistency_failures": duplicate_consistency_failures,
        "forced_execution_failures": forced_execution_failures,
        "control_status_counts": counts["control"],
        "probe_status_counts": counts["probe"],
        "branch_status_counts": counts["branch"],
        "states": states,
    }
    if (p000_mismatches or scenario_mismatches
            or duplicate_consistency_failures or forced_execution_failures):
        raise RuntimeError(
            "M counterfactual replay consistency failed; inspect summary")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instances", type=Path, required=True)
    parser.add_argument("--normalization", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--instance-ids", nargs="+", default=(
        "train-S0-002", "train-S0-005", "train-S0-006"))
    parser.add_argument("--algorithm-seeds", nargs="+", type=int,
                        default=(777001, 777002, 777003))
    parser.add_argument("--states-per-instance", type=int, default=4)
    parser.add_argument("--actions-per-state", type=int, default=12)
    parser.add_argument(
        "--maximum-generation", type=int,
        help="Optionally restrict target mutation states to this generation.")
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--evaluation-budget", type=int, default=400)
    parser.add_argument("--evaluation-mode", choices=("ccp", "ccp100"),
                        default="ccp")
    parser.add_argument("--uncertainty-profile",
                        default="transport_border_saa_v1")
    parser.add_argument("--scenario-count", type=int, default=200)
    parser.add_argument("--ccp-confidence", type=float, default=0.90)
    parser.add_argument("--pop", type=int, default=20)
    parser.add_argument("--gens", type=int, default=1000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--runner", type=Path,
        default=Path(__file__).resolve().with_name(
            "run_learning_components_deterministic.py"))
    args = parser.parse_args(argv)
    if (args.states_per_instance < 1 or args.actions_per_state < 2
            or args.max_workers < 1 or not args.algorithm_seeds):
        parser.error("states, actions, workers, and seeds must be positive")
    if args.maximum_generation is not None and args.maximum_generation < 0:
        parser.error("maximum generation cannot be negative")
    for path, label in (
            (args.normalization, "normalization"),
            (args.checkpoint, "checkpoint"), (args.runner, "runner")):
        if not path.is_file():
            parser.error(f"{label} is missing: {path}")
    args.out.mkdir(parents=True, exist_ok=True)
    completion = args.out / "AUDIT_COMPLETE.json"
    completion.unlink(missing_ok=True)
    specifications = []
    for instance_id in args.instance_ids:
        path = args.instances / f"{instance_id}.json"
        if not path.is_file():
            parser.error(f"instance is missing: {path}")
        specifications.append((instance_id, _short_id(instance_id), path))

    control_jobs = []
    for _, short, instance_path in specifications:
        for seed in args.algorithm_seeds:
            root = args.out / short / f"seed_{seed}"
            for component, name, probability in (
                    ("BASE", "BASE", None), ("M", "P000", 0.0)):
                run = root / name
                command = _command(
                    args, instance_path, run, seed, component,
                    model_choice_probability=probability,
                    policy_seed=seed + 101)
                control_jobs.append((
                    run, run.with_suffix(".nohup.log"), command,
                    seed, component, None, None))
    control_counts = _execute(
        control_jobs, min(args.max_workers, len(control_jobs)), "CONTROL")

    manifest = {
        "schema_version": 1,
        "algorithm_seeds": list(args.algorithm_seeds),
        "states_per_instance_requested": args.states_per_instance,
        "actions_per_state_requested": args.actions_per_state,
        "instances": [],
    }
    probe_jobs = []
    for instance_id, short, instance_path in specifications:
        instance_row = {
            "instance_id": instance_id, "short_id": short, "seeds": []}
        for seed in args.algorithm_seeds:
            root = args.out / short / f"seed_{seed}"
            selected = _spread_events(
                root / "P000", args.states_per_instance,
                maximum_generation=args.maximum_generation)
            seed_row = {"algorithm_seed": seed, "states": []}
            for row in selected:
                event = int(row["mixture_event_index"])
                run = root / f"event_{event}" / "PROBE"
                command = _command(
                    args, instance_path, run, seed, "M",
                    model_choice_event=event, policy_seed=seed + 202)
                probe_jobs.append((
                    run, run.with_suffix(".nohup.log"), command,
                    seed, "M", event, None))
                seed_row["states"].append({
                    "event": event,
                    "generation": int(row.get("generation", -1)),
                })
            instance_row["seeds"].append(seed_row)
        manifest["instances"].append(instance_row)
    probe_counts = _execute(
        probe_jobs, min(args.max_workers, len(probe_jobs)), "PROBE")

    branch_jobs = []
    for instance, (_, _, instance_path) in zip(
            manifest["instances"], specifications):
        short = instance["short_id"]
        for seed_row in instance["seeds"]:
            seed = int(seed_row["algorithm_seed"])
            root = args.out / short / f"seed_{seed}"
            for state in seed_row["states"]:
                event = int(state["event"])
                policy_state = _target_policy_state(
                    root / f"event_{event}" / "PROBE", event)
                observation = policy_state["observation"]
                eligible = [
                    index for index, value in enumerate(observation["eligible"])
                    if value
                ]
                selected_indices = _spread_indices(
                    eligible, args.actions_per_state)
                if len(selected_indices) < 2:
                    raise RuntimeError(
                        f"M state has fewer than two eligible candidates: "
                        f"{short}:{seed}:{event}")
                state.update({
                    "candidate_count": len(observation["eligible"]),
                    "eligible_candidate_count": len(eligible),
                    "forced_candidate_indices": selected_indices,
                })
                for candidate_index in selected_indices:
                    run = (
                        root / f"event_{event}"
                        / f"candidate_{candidate_index}")
                    command = _command(
                        args, instance_path, run, seed, "M",
                        model_choice_event=event,
                        forced_candidate_index=candidate_index,
                        policy_seed=seed + 303)
                    branch_jobs.append((
                        run, run.with_suffix(".nohup.log"), command,
                        seed, "M", event, candidate_index))
    manifest_path = args.out / "collection_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    branch_counts = _execute(
        branch_jobs, min(args.max_workers, len(branch_jobs)), "BRANCH")
    counts = {
        "control": control_counts, "probe": probe_counts,
        "branch": branch_counts,
    }
    summary = _analyse(args, manifest, counts)
    completion.write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
