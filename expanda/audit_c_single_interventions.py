#!/usr/bin/env python3
"""Collect and summarize one-shot C interventions through the final budget.

For every selected Base crossover state, several private policy seeds sample
one joint C action.  Every other crossover executes the exact Base plan.  The
algorithm seed and frozen CCP scenarios remain fixed, so final-HV differences
are attributable to that single intervention and its downstream consequences.
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


def _command(args, instance_path, run, seed, component, policy_seed=0,
             model_choice_probability=None, model_choice_event=None):
    result = [
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
    if component == "C":
        result.extend([
            "--checkpoint", str(args.checkpoint),
            "--inference-action-mode", "sample",
            "--inference-policy-seed", str(policy_seed),
        ])
    if model_choice_probability is not None:
        result.extend([
            "--model-choice-probability", str(model_choice_probability)])
    if model_choice_event is not None:
        result.extend(["--model-choice-event", str(model_choice_event)])
    return result


def _target_outcome(run: Path, event: int):
    rows = [
        row for row in _jsonl(run / "crossover_outcomes.jsonl")
        if row.get("crossover_event_index") == event
    ]
    if len(rows) != 1:
        raise RuntimeError(
            f"expected one crossover outcome for event {event}: {run}")
    return rows[0]


def _valid_complete(run: Path, seed: int, component: str,
                    model_choice_event=None):
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
        if component == "C":
            outcomes = _jsonl(run / "crossover_outcomes.jsonl")
            model_rows = [row for row in outcomes if row.get("model_decisions")]
            if model_choice_event is None:
                valid = valid and not model_rows
            else:
                target = _target_outcome(run, model_choice_event)
                valid = (
                    valid and target.get("model_decisions") == 1
                    and len(model_rows) == 1
                )
        return bool(valid)
    except (OSError, TypeError, ValueError, RuntimeError):
        return False


def _run_job(job):
    run, log, command, seed, component, event = job
    if _valid_complete(run, seed, component, event):
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
            run, seed, component, event):
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


def _spread_events(base_run: Path, count: int):
    rows = [
        row for row in _jsonl(base_run / "crossover_outcomes.jsonl")
        if int(row.get("joint_candidate_count", 0)) >= 2
        and row.get("crossover_event_index") is not None
    ]
    rows.sort(key=lambda row: int(row["crossover_event_index"]))
    if len(rows) < count:
        raise ValueError(
            f"{base_run} has only {len(rows)} usable C events; requested {count}")
    if count == 1:
        return [rows[len(rows) // 2]]
    indices = [
        round(offset * (len(rows) - 1) / (count - 1))
        for offset in range(count)
    ]
    if len(set(indices)) != count:
        raise ValueError(f"could not select {count} distinct events: {base_run}")
    return [rows[index] for index in indices]


def _action_signature(outcome):
    action = outcome.get("selected_joint_action")
    if action == "base_passthrough":
        return "base_passthrough"
    changed = [
        row for row in outcome.get("selected_cuts", [])
        if row.get("different_from_shadow_base") is True
    ]
    if action != "single_cut_override" or len(changed) != 1:
        raise RuntimeError("one-shot C branch has an invalid selected action")
    row = changed[0]
    return ":".join(map(str, (
        "override", row.get("cut_index"), row.get("batch_id"),
        row.get("chosen_index"), row.get("join_node"),
    )))


def _analyse(args, manifest):
    states = []
    exact_passthroughs = 0
    passthroughs = 0
    duplicate_consistency_failures = []
    scenario_mismatches = []
    p000_mismatches = []
    for instance in manifest["instances"]:
        short = instance["short_id"]
        root = args.out / short / f"seed_{args.algorithm_seed}"
        pure_base = root / "BASE"
        p000 = root / "P000"
        base_marker = _read(pure_base / "COMPLETE.json")
        base_hv = float(base_marker["final_hypervolume"])
        base_hash = _sha256(pure_base / "final_archive_lineage.json")
        p000_marker = _read(p000 / "COMPLETE.json")
        p000_hash = _sha256(p000 / "final_archive_lineage.json")
        if (not math.isclose(
                base_hv, float(p000_marker["final_hypervolume"]), abs_tol=1e-15)
                or base_hash != p000_hash):
            p000_mismatches.append(short)
        base_digest = _read(pure_base / "configuration.json")["scenario_digest"]
        for state in instance["states"]:
            event = int(state["event"])
            records = []
            for policy_seed in args.policy_seeds:
                run = root / f"event_{event}" / f"policy_{policy_seed}"
                marker = _read(run / "COMPLETE.json")
                config = _read(run / "configuration.json")
                if config["scenario_digest"] != base_digest:
                    scenario_mismatches.append(str(run))
                outcome = _target_outcome(run, event)
                final_hv = float(marker["final_hypervolume"])
                signature = _action_signature(outcome)
                record = {
                    "policy_seed": policy_seed,
                    "action": outcome.get("selected_joint_action"),
                    "action_signature": signature,
                    "final_hv": final_hv,
                    "final_percent": 100.0 * (final_hv - base_hv) / base_hv,
                    "immediate_archive_delta": float(
                        outcome.get("reward_hv_delta", 0.0)),
                    "child_survived": bool(
                        outcome.get("child_1_survived")
                        or outcome.get("child_2_survived")),
                    "final_archive_sha256": _sha256(
                        run / "final_archive_lineage.json"),
                }
                records.append(record)
                if signature == "base_passthrough":
                    passthroughs += 1
                    if (math.isclose(final_hv, base_hv, abs_tol=1e-15)
                            and record["final_archive_sha256"] == base_hash):
                        exact_passthroughs += 1
            grouped = {}
            for record in records:
                grouped.setdefault(record["action_signature"], []).append(record)
            unique = []
            for signature, duplicates in grouped.items():
                hvs = {round(item["final_hv"], 15) for item in duplicates}
                hashes = {item["final_archive_sha256"] for item in duplicates}
                if len(hvs) != 1 or len(hashes) != 1:
                    duplicate_consistency_failures.append(
                        f"{short}:{event}:{signature}")
                unique.append(duplicates[0])
            base_record = {
                "action": "base_passthrough", "action_signature": "BASE",
                "final_hv": base_hv, "final_percent": 0.0,
                "immediate_archive_delta": 0.0,
            }
            overrides = [
                record for record in unique
                if record["action"] == "single_cut_override"
            ]
            candidates = [base_record, *overrides]
            best = max(candidates, key=lambda row: row["final_hv"])
            states.append({
                "instance": short,
                "event": event,
                "generation": state["generation"],
                "sampled_runs": len(records),
                "distinct_sampled_actions": len(unique),
                "distinct_overrides": len(overrides),
                "base_final_hv": base_hv,
                "audited_oracle_final_hv": best["final_hv"],
                "audited_oracle_percent": best["final_percent"],
                "best_action_signature": best["action_signature"],
                "best_override_percent": (
                    None if not overrides else
                    max(row["final_percent"] for row in overrides)),
                "worst_override_percent": (
                    None if not overrides else
                    min(row["final_percent"] for row in overrides)),
                "positive_override_fraction": (
                    None if not overrides else
                    sum(row["final_percent"] > 0.0 for row in overrides)
                    / len(overrides)),
                "actions": sorted(
                    unique, key=lambda row: row["final_percent"], reverse=True),
            })
    oracle_values = [row["audited_oracle_percent"] for row in states]
    override_values = [
        row["best_override_percent"] for row in states
        if row["best_override_percent"] is not None
    ]
    summary = {
        "completed": True,
        "definition": (
            "sampled one-shot C action followed by exact Base control to the "
            "same final search budget"),
        "instance_count": len(manifest["instances"]),
        "state_count": len(states),
        "branch_run_count": len(states) * len(args.policy_seeds),
        "distinct_override_count": sum(
            row["distinct_overrides"] for row in states),
        "mean_audited_oracle_percent": statistics.fmean(oracle_values),
        "median_audited_oracle_percent": statistics.median(oracle_values),
        "maximum_audited_oracle_percent": max(oracle_values),
        "states_with_positive_oracle": sum(value > 0.0 for value in oracle_values),
        "states_with_oracle_above_5_percent": sum(
            value >= 5.0 for value in oracle_values),
        "states_with_oracle_above_10_percent": sum(
            value >= 10.0 for value in oracle_values),
        "mean_best_override_percent": (
            None if not override_values else statistics.fmean(override_values)),
        "base_passthrough_runs": passthroughs,
        "exact_base_passthrough_runs": exact_passthroughs,
        "p000_exact_base": not p000_mismatches,
        "scenario_exact": not scenario_mismatches,
        "duplicate_replay_consistent": not duplicate_consistency_failures,
        "p000_mismatches": p000_mismatches,
        "scenario_mismatches": scenario_mismatches,
        "duplicate_consistency_failures": duplicate_consistency_failures,
        "states": states,
    }
    if (p000_mismatches or scenario_mismatches
            or duplicate_consistency_failures
            or exact_passthroughs != passthroughs):
        raise RuntimeError(
            "counterfactual replay consistency failed; inspect partial summary")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instances", type=Path, required=True)
    parser.add_argument("--normalization", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--instance-ids", nargs="+", default=(
        "train-S0-002", "train-S0-005", "train-S0-006"))
    parser.add_argument("--algorithm-seed", type=int, default=777001)
    parser.add_argument("--policy-seeds", nargs="+", type=int, default=tuple(
        range(29101, 29117)))
    parser.add_argument("--states-per-instance", type=int, default=4)
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--evaluation-budget", type=int, default=400)
    parser.add_argument("--evaluation-mode", choices=("ccp", "ccp100"),
                        default="ccp")
    parser.add_argument("--uncertainty-profile", default="transport_border_saa_v1")
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
    if (args.states_per_instance < 1 or args.max_workers < 1
            or not args.policy_seeds):
        parser.error("states, workers, and policy seeds must be positive")
    for path, label in ((args.normalization, "normalization"),
                        (args.checkpoint, "checkpoint"),
                        (args.runner, "runner")):
        if not path.is_file():
            parser.error(f"{label} is missing: {path}")
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
        root = args.out / short / f"seed_{args.algorithm_seed}"
        for component, name, probability in (
                ("BASE", "BASE", None), ("C", "P000", 0.0)):
            run = root / name
            command = _command(
                args, instance_path, run, args.algorithm_seed, component,
                policy_seed=args.policy_seeds[0],
                model_choice_probability=probability)
            control_jobs.append((
                run, run.with_suffix(".nohup.log"), command,
                args.algorithm_seed, component, None))
    control_counts = _execute(
        control_jobs, min(args.max_workers, len(control_jobs)), "CONTROL")

    manifest = {
        "schema_version": 1,
        "algorithm_seed": args.algorithm_seed,
        "policy_seeds": args.policy_seeds,
        "states_per_instance": args.states_per_instance,
        "instances": [],
    }
    branch_jobs = []
    for instance_id, short, instance_path in specifications:
        root = args.out / short / f"seed_{args.algorithm_seed}"
        selected = _spread_events(root / "P000", args.states_per_instance)
        state_rows = []
        for row in selected:
            event = int(row["crossover_event_index"])
            state_rows.append({
                "event": event,
                "generation": int(row.get("generation", -1)),
                "joint_candidate_count": int(row["joint_candidate_count"]),
            })
            for policy_seed in args.policy_seeds:
                run = root / f"event_{event}" / f"policy_{policy_seed}"
                command = _command(
                    args, instance_path, run, args.algorithm_seed, "C",
                    policy_seed=policy_seed, model_choice_event=event)
                branch_jobs.append((
                    run, run.with_suffix(".nohup.log"), command,
                    args.algorithm_seed, "C", event))
        manifest["instances"].append({
            "instance_id": instance_id, "short_id": short,
            "states": state_rows,
        })
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "audit_manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n",
        encoding="utf-8")
    branch_counts = _execute(branch_jobs, args.max_workers, "BRANCH")
    summary = _analyse(args, manifest)
    summary["control_status_counts"] = control_counts
    summary["branch_status_counts"] = branch_counts
    completion.write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n",
        encoding="utf-8")
    print(json.dumps({
        key: value for key, value in summary.items() if key != "states"
    }, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
