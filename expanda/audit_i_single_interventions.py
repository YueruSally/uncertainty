#!/usr/bin/env python3
"""Audit one forced I path decision followed by exact Base search control."""
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


def _command(args, instance_path, run, seed, component, *,
             model_choice_event=None, forced_candidate_index=None,
             policy_seed=0):
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
    if component == "I":
        command.extend([
            "--checkpoint", str(args.checkpoint),
            "--inference-action-mode", "argmax",
            "--inference-policy-seed", str(policy_seed),
        ])
        if model_choice_event is not None:
            command.extend([
                "--model-choice-event", str(model_choice_event)])
        if forced_candidate_index is not None:
            command.extend([
                "--i-forced-candidate-index", str(forced_candidate_index)])
    return command


def _target_policy_state(run: Path, event: int):
    rows = [
        row for row in _jsonl(run / "initialization_policy_states.jsonl")
        if int(row.get("decision_event_index", -1)) == int(event)
    ]
    if len(rows) != 1:
        raise RuntimeError(
            f"expected one I policy state for event {event}: {run}")
    return rows[0]


def _target_action(run: Path, event: int):
    matches = []
    for outcome in _jsonl(run / "initialization_outcomes.jsonl"):
        for row in outcome.get("selected_paths", []):
            if int(row.get("decision_event_index") or -1) == int(event):
                matches.append({**row, "event_id": outcome["event_id"]})
    if len(matches) != 1:
        raise RuntimeError(
            f"expected one I selected action for event {event}: {run}")
    return matches[0]


def _valid_complete(run: Path, seed: int, component: str, event=None,
                    forced_candidate_index=None):
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
        if component == "I":
            target_rows = []
            for outcome in _jsonl(run / "initialization_outcomes.jsonl"):
                target_rows.extend(
                    row for row in outcome.get("selected_paths", [])
                    if row.get("target_model_choice") is True)
            if event is None:
                valid = valid and not target_rows
            else:
                action = _target_action(run, event)
                valid = valid and len(target_rows) == 1
                _target_policy_state(run, event)
                if forced_candidate_index is not None:
                    valid = valid and (
                        int(action.get("selected_candidate_index", -1))
                        == int(forced_candidate_index)
                        and int(action.get("forced_candidate_index", -1))
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


def _initialization_events(run: Path):
    grouped = {}
    for row in _jsonl(run / "initialization_candidates.jsonl"):
        event = row.get("decision_event_index")
        if event is None or row.get("decision_stage") != "path":
            continue
        grouped.setdefault(int(event), []).append(row)
    events = []
    for event, rows in sorted(grouped.items()):
        base_rows = [row for row in rows if row.get("shadow_base_chosen")]
        chosen_rows = [row for row in rows if row.get("chosen")]
        if len(rows) < 2 or len(base_rows) != 1 or len(chosen_rows) != 1:
            continue
        events.append({
            "event": event,
            "event_id": rows[0]["event_id"],
            "batch_id": rows[0]["batch_id"],
            "selection_step": rows[0]["selection_step"],
            "required_path_count": rows[0]["required_path_count"],
            "candidate_count": len(rows),
            "base_candidate_index": int(base_rows[0]["candidate_id"]),
        })
    return events


def _spread_events(run: Path, requested: int):
    rows = _initialization_events(run)
    if not rows:
        raise ValueError(f"{run} has no usable I path decisions")
    count = min(int(requested), len(rows))
    if count == 1:
        return [rows[len(rows) // 2]]
    indices = [
        round(offset * (len(rows) - 1) / (count - 1))
        for offset in range(count)
    ]
    return [rows[index] for index in indices]


def _spread_indices_with_base(eligible, requested, base_index):
    values = sorted(set(int(index) for index in eligible))
    base_index = int(base_index)
    if base_index not in values:
        raise ValueError("Base candidate is not eligible")
    count = min(int(requested), len(values))
    if count == len(values):
        return values
    others = [value for value in values if value != base_index]
    needed = count - 1
    if needed <= 0:
        return [base_index]
    if needed == 1:
        selected = [others[len(others) // 2]]
    else:
        offsets = [
            round(offset * (len(others) - 1) / (needed - 1))
            for offset in range(needed)
        ]
        selected = [others[offset] for offset in offsets]
    return sorted([base_index, *selected])


def _analyse(args, manifest, counts):
    states = []
    p000_mismatches = []
    scenario_mismatches = []
    replay_failures = []
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
                probe_state = _target_policy_state(probe, event)
                records = []
                for candidate_index in state["forced_candidate_indices"]:
                    run = (
                        root / f"event_{event}"
                        / f"candidate_{candidate_index}")
                    marker = _read(run / "COMPLETE.json")
                    config = _read(run / "configuration.json")
                    if config["scenario_digest"] != base_digest:
                        scenario_mismatches.append(str(run))
                    policy_state = _target_policy_state(run, event)
                    if (policy_state["candidate_path_ids"]
                            != probe_state["candidate_path_ids"]
                            or policy_state["observation"]
                            != probe_state["observation"]):
                        replay_failures.append(str(run))
                    action = _target_action(run, event)
                    selected = int(action["selected_candidate_index"])
                    if selected != int(candidate_index):
                        forced_execution_failures.append(str(run))
                    final_hv = float(marker["final_hypervolume"])
                    records.append({
                        "candidate_index": int(candidate_index),
                        "path_id": str(action["chosen_path_id"]),
                        "is_base_candidate": (
                            int(candidate_index)
                            == int(state["base_candidate_index"])),
                        "final_hv": final_hv,
                        "final_percent": 100.0 * (
                            final_hv - base_hv) / base_hv,
                        "final_archive_sha256": _sha256(
                            run / "final_archive_lineage.json"),
                    })
                base_records = [
                    row for row in records if row["is_base_candidate"]]
                if len(base_records) != 1 or (
                        not math.isclose(
                            base_records[0]["final_hv"], base_hv,
                            abs_tol=1e-15)
                        or base_records[0]["final_archive_sha256"] != base_hash):
                    replay_failures.append(
                        f"{short}:{seed}:{event}:base-candidate")
                best = max(
                    [{"candidate_index": None, "path_id": "BASE",
                      "is_base_candidate": True, "final_hv": base_hv,
                      "final_percent": 0.0}, *records],
                    key=lambda row: row["final_hv"])
                states.append({
                    "instance": short,
                    "algorithm_seed": seed,
                    **{key: state[key] for key in (
                        "event", "event_id", "batch_id", "selection_step",
                        "required_path_count", "candidate_count",
                        "base_candidate_index")},
                    "audited_candidate_count": len(records),
                    "base_final_hv": base_hv,
                    "audited_oracle_final_hv": best["final_hv"],
                    "audited_oracle_percent": best["final_percent"],
                    "best_candidate_index": best["candidate_index"],
                    "best_path_id": best["path_id"],
                    "positive_candidate_fraction": sum(
                        row["final_percent"] > 0.0 for row in records
                    ) / len(records),
                    "actions": sorted(
                        records, key=lambda row: row["final_percent"],
                        reverse=True),
                })
    oracle_values = [row["audited_oracle_percent"] for row in states]
    action_values = [
        action["final_percent"]
        for state in states for action in state["actions"]
        if not action["is_base_candidate"]
    ]
    summary = {
        "completed": True,
        "definition": (
            "one forced I path identity followed by exact Base initialization, "
            "crossover, and mutation to the same final CCP search budget"),
        "algorithm_seeds": list(args.algorithm_seeds),
        "instance_count": len(manifest["instances"]),
        "state_count": len(states),
        "probe_run_count": len(states),
        "branch_run_count": sum(
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
        "mean_nonbase_action_percent": statistics.fmean(action_values),
        "p000_exact_base": not p000_mismatches,
        "scenario_exact": not scenario_mismatches,
        "state_replay_exact": not replay_failures,
        "forced_execution_exact": not forced_execution_failures,
        "p000_mismatches": p000_mismatches,
        "scenario_mismatches": scenario_mismatches,
        "state_replay_failures": replay_failures,
        "forced_execution_failures": forced_execution_failures,
        "control_status_counts": counts["control"],
        "probe_status_counts": counts["probe"],
        "branch_status_counts": counts["branch"],
        "states": states,
    }
    if (p000_mismatches or scenario_mismatches or replay_failures
            or forced_execution_failures):
        raise RuntimeError(
            "I counterfactual replay consistency failed; inspect summary")
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
    parser.add_argument("--actions-per-state", type=int, default=10)
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
            base_run = root / "BASE"
            control_jobs.append((
                base_run, base_run.with_suffix(".nohup.log"),
                _command(args, instance_path, base_run, seed, "BASE"),
                seed, "BASE", None, None))
            p000 = root / "P000"
            control_jobs.append((
                p000, p000.with_suffix(".nohup.log"),
                _command(
                    args, instance_path, p000, seed, "I",
                    model_choice_event=1_000_000_000,
                    policy_seed=seed + 101),
                seed, "I", None, None))
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
                root / "P000", args.states_per_instance)
            seed_row = {"algorithm_seed": seed, "states": selected}
            for state in selected:
                event = int(state["event"])
                run = root / f"event_{event}" / "PROBE"
                probe_jobs.append((
                    run, run.with_suffix(".nohup.log"),
                    _command(
                        args, instance_path, run, seed, "I",
                        model_choice_event=event,
                        policy_seed=seed + 202),
                    seed, "I", event, None))
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
                base_index = int(
                    policy_state["shadow_base_candidate_index"])
                selected_indices = _spread_indices_with_base(
                    eligible, args.actions_per_state, base_index)
                if len(selected_indices) < 2:
                    raise RuntimeError(
                        f"I state has fewer than two eligible candidates: "
                        f"{short}:{seed}:{event}")
                state.update({
                    "eligible_candidate_count": len(eligible),
                    "base_candidate_index": base_index,
                    "forced_candidate_indices": selected_indices,
                })
                for candidate_index in selected_indices:
                    run = (
                        root / f"event_{event}"
                        / f"candidate_{candidate_index}")
                    branch_jobs.append((
                        run, run.with_suffix(".nohup.log"),
                        _command(
                            args, instance_path, run, seed, "I",
                            model_choice_event=event,
                            forced_candidate_index=candidate_index,
                            policy_seed=seed + 303),
                        seed, "I", event, candidate_index))
    manifest_path = args.out / "collection_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, allow_nan=False), encoding="utf-8")
    branch_counts = _execute(
        branch_jobs, min(args.max_workers, len(branch_jobs)), "BRANCH")
    counts = {
        "control": control_counts,
        "probe": probe_counts,
        "branch": branch_counts,
    }
    summary = _analyse(args, manifest, counts)
    completion.write_text(
        json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
