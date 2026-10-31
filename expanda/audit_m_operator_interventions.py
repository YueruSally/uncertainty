#!/usr/bin/env python3
"""Audit downstream effects of forcing only the M mutation operator."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys


EPSILON = 1e-12
OPERATORS = ("add", "del", "mod", "mode", "replace")


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _jsonl(path):
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _paired_percent(reference, candidate):
    reference = float(reference)
    if abs(reference) <= EPSILON:
        return None
    return 100.0 * (float(candidate) - reference) / reference


def _spread_events(rows, requested, minimum_generation):
    usable = sorted(
        (
            row for row in rows
            if row.get("decision_kind") == "base_mixture_choice"
            and int(row.get("generation", -1)) >= minimum_generation
            and row.get("parent_id")
        ),
        key=lambda row: int(row["mixture_event_index"]),
    )
    if len(usable) < requested:
        raise ValueError(
            f"only {len(usable)} usable M events at generation >= "
            f"{minimum_generation}; requested {requested}")
    if requested == 1:
        return [usable[len(usable) // 2]]
    positions = [
        round(index * (len(usable) - 1) / (requested - 1))
        for index in range(requested)
    ]
    if len(set(positions)) != requested:
        raise RuntimeError("spread event positions overlap")
    return [usable[position] for position in positions]


def _base_command(args, instance, out, algorithm_seed):
    return [
        sys.executable, "-u", str(args.runner),
        "--instance", str(instance),
        "--normalization", str(args.normalization),
        "--out", str(out),
        "--component", "M",
        "--checkpoint", str(args.checkpoint),
        "--algorithm-seed", str(algorithm_seed),
        "--inference-action-mode", "argmax",
        "--inference-policy-seed", "44000",
        "--model-choice-probability", "0",
        "--evaluation-mode", args.evaluation_mode,
        "--uncertainty-profile", args.uncertainty_profile,
        "--scenario-count", str(args.scenario_count),
        "--ccp-confidence", str(args.ccp_confidence),
        "--evaluation-budget", str(args.evaluation_budget),
        "--pop", str(args.pop),
        "--gens", str(args.gens),
        "--device", args.device,
    ]


def _branch_command(args, instance, out, algorithm_seed, event,
                    continuation_seed, operator=None):
    command = _base_command(args, instance, out, algorithm_seed)
    probability_position = command.index("--model-choice-probability")
    del command[probability_position:probability_position + 2]
    command.extend([
        "--m-continuation-event", str(event),
        "--m-continuation-seed", str(continuation_seed),
    ])
    if operator is None:
        command.extend(["--model-choice-probability", "0"])
    else:
        command.extend([
            "--model-choice-event", str(event),
            "--m-forced-operator", operator,
        ])
    return command


def _target_row(run, event):
    rows = [
        row for row in _jsonl(Path(run) / "mutation_outcomes.jsonl")
        if int(row.get("mixture_event_index", -1)) == event
    ]
    if len(rows) != 1:
        raise RuntimeError(
            f"expected one mutation outcome at event {event}; "
            f"found {len(rows)}: {run}")
    return rows[0]


def _valid(run, algorithm_seed, event=None, continuation_seed=None,
           operator=None):
    run = Path(run)
    if not (run / "COMPLETE.json").is_file():
        return False
    try:
        complete = _read(run / "COMPLETE.json")
        config = _read(run / "configuration.json")
        control = config["learning_control"]
        valid = (
            complete.get("completed") is True
            and complete.get("component") == "M"
            and config.get("component") == "M"
            and config.get("seeds", {}).get("algorithm") == algorithm_seed
        )
        if event is None:
            return bool(
                valid
                and control.get("model_choice_probabilities", {}).get("M") == 0
                and control.get("m_forced_operator") is None)
        valid = bool(
            valid
            and control.get("m_continuation_event") == event
            and control.get("m_continuation_seed") == continuation_seed
            and complete.get("continuation_reseed_count") == 1
            and control.get("m_forced_operator") == operator)
        target = _target_row(run, event)
        if operator is None:
            valid = valid and target.get("decision_kind") == "base_mixture_choice"
        else:
            valid = bool(
                valid
                and target.get("decision_kind") == "forced_operator_choice"
                and target.get("forced_operator") == operator
                and target.get("operator") == operator)
        return bool(valid)
    except (KeyError, OSError, TypeError, ValueError, RuntimeError):
        return False


def _run(job):
    run, log, command, seed, event, continuation_seed, operator = job
    if _valid(run, seed, event, continuation_seed, operator):
        return run, "existing"
    if Path(run).exists():
        raise RuntimeError(f"incomplete output exists; use a fresh root: {run}")
    Path(run).parent.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment.setdefault("OMP_NUM_THREADS", "1")
    environment.setdefault("MKL_NUM_THREADS", "1")
    with Path(log).open("w", encoding="utf-8") as handle:
        result = subprocess.run(
            command, cwd=Path(__file__).resolve().parent,
            stdout=handle, stderr=subprocess.STDOUT,
            env=environment, check=False)
    if result.returncode != 0 or not _valid(
            run, seed, event, continuation_seed, operator):
        raise RuntimeError(f"run failed with exit {result.returncode}: {log}")
    return run, "complete"


def _execute(jobs, workers, label):
    counts = Counter()
    print(f"[{label}] jobs={len(jobs)} workers={workers}", flush=True)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(_run, job) for job in jobs]
        for index, future in enumerate(as_completed(futures), 1):
            run, status = future.result()
            counts[status] += 1
            print(
                f"[{label}] {index}/{len(jobs)} {status}: {run}",
                flush=True)
    return dict(counts)


def _analyse(args, states, status_counts):
    records = []
    scenario_failures = []
    replay_failures = []
    forced_failures = []
    shadow_failures = []
    state_snapshots = {}
    for state in states:
        short = state["instance"]
        seed = int(state["algorithm_seed"])
        event = int(state["event"])
        state_key = f"{short}:{seed}:{event}"
        for continuation_seed in args.continuation_seeds:
            root = (
                args.out / short / f"seed_{seed}" / f"event_{event}" /
                f"continuation_{continuation_seed}")
            reference = root / "REFERENCE"
            reference_complete = _read(reference / "COMPLETE.json")
            reference_config = _read(reference / "configuration.json")
            reference_target = _target_row(reference, event)
            reference_hash = _sha256(reference / "final_archive_lineage.json")
            parent_ids = {reference_target["parent_id"]}
            for operator in OPERATORS:
                run = root / f"operator_{operator}"
                complete = _read(run / "COMPLETE.json")
                config = _read(run / "configuration.json")
                target = _target_row(run, event)
                parent_ids.add(target["parent_id"])
                if config["scenario_digest"] != reference_config["scenario_digest"]:
                    scenario_failures.append(str(run))
                if target.get("operator") != operator:
                    forced_failures.append(str(run))
                run_hash = _sha256(run / "final_archive_lineage.json")
                shadow_match = operator == reference_target["operator"]
                if shadow_match and run_hash != reference_hash:
                    shadow_failures.append(str(run))
                states_logged = _jsonl(
                    run / "mutation_operator_intervention_states.jsonl")
                target_states = [
                    row for row in states_logged
                    if int(row["mixture_event_index"]) == event
                ]
                if len(target_states) != 1:
                    raise RuntimeError(
                        f"expected one operator state at {state_key}: {run}")
                snapshot = target_states[0]
                previous = state_snapshots.setdefault(state_key, snapshot)
                if previous["parent_id"] != snapshot["parent_id"]:
                    replay_failures.append(str(run))
                records.append({
                    "instance": short,
                    "algorithm_seed": seed,
                    "event": event,
                    "generation": int(target["generation"]),
                    "continuation_seed": int(continuation_seed),
                    "operator": operator,
                    "shadow_base_operator": reference_target["operator"],
                    "shadow_operator_match": shadow_match,
                    "mutation_success": bool(target["mutation_success"]),
                    "decision_changed": bool(target["decision_changed"]),
                    "reference_final_hv": float(
                        reference_complete["final_hypervolume"]),
                    "operator_final_hv": float(complete["final_hypervolume"]),
                    "final_percent": _paired_percent(
                        reference_complete["final_hypervolume"],
                        complete["final_hypervolume"]),
                    "reference_archive_sha256": reference_hash,
                    "operator_archive_sha256": run_hash,
                })
            if len(parent_ids) != 1:
                replay_failures.append(
                    f"{state_key}:continuation_{continuation_seed}")

    grouped = defaultdict(list)
    for record in records:
        grouped[(
            record["instance"], record["algorithm_seed"],
            record["event"], record["operator"],
        )].append(record)
    actions = []
    for key, rows in sorted(grouped.items()):
        values = [float(row["final_percent"]) for row in rows]
        mean = statistics.fmean(values)
        deviation = statistics.pstdev(values)
        actions.append({
            "instance": key[0],
            "algorithm_seed": key[1],
            "event": key[2],
            "operator": key[3],
            "continuation_count": len(values),
            "mean_final_percent": mean,
            "standard_deviation_final_percent": deviation,
            "safe_score": mean - deviation,
            "minimum_final_percent": min(values),
            "maximum_final_percent": max(values),
            "positive_fraction": sum(value > EPSILON for value in values) / len(values),
            "negative_fraction": sum(value < -EPSILON for value in values) / len(values),
            "mutation_success_fraction": sum(
                row["mutation_success"] for row in rows) / len(rows),
        })

    by_state = defaultdict(list)
    for action in actions:
        by_state[(
            action["instance"], action["algorithm_seed"], action["event"]
        )].append(action)
    state_rows = []
    for key, rows in sorted(by_state.items()):
        best = max(rows, key=lambda row: row["safe_score"])
        state_rows.append({
            "instance": key[0],
            "algorithm_seed": key[1],
            "event": key[2],
            "generation": next(
                int(item["generation"]) for item in records
                if (item["instance"], item["algorithm_seed"], item["event"])
                == key),
            "best_operator": best["operator"],
            "safe_oracle_percent": best["safe_score"],
            "mean_oracle_percent": max(
                row["mean_final_percent"] for row in rows),
        })

    safe_oracles = [row["safe_oracle_percent"] for row in state_rows]
    per_operator = []
    for operator in OPERATORS:
        rows = [row for row in actions if row["operator"] == operator]
        per_operator.append({
            "operator": operator,
            "states": len(rows),
            "mean_final_percent": statistics.fmean(
                row["mean_final_percent"] for row in rows),
            "mean_safe_score": statistics.fmean(row["safe_score"] for row in rows),
            "positive_safe_states": sum(row["safe_score"] > EPSILON for row in rows),
        })
    summary = {
        "completed": True,
        "definition": (
            "forced mutation operator with Base batch/internal target selection, "
            "matched Base reference, frozen CCP scenarios, and shared "
            "post-operation continuation seeds"),
        "instance_count": len({row["instance"] for row in state_rows}),
        "state_count": len(state_rows),
        "operator_action_count": len(actions),
        "continuation_seed_count": len(args.continuation_seeds),
        "branch_record_count": len(records),
        "total_run_count": sum(status_counts.values()),
        "mean_safe_oracle_percent": statistics.fmean(safe_oracles),
        "median_safe_oracle_percent": statistics.median(safe_oracles),
        "maximum_safe_oracle_percent": max(safe_oracles),
        "states_with_positive_safe_operator": sum(
            value > EPSILON for value in safe_oracles),
        "states_with_safe_oracle_above_5_percent": sum(
            value >= 5.0 for value in safe_oracles),
        "states_with_safe_oracle_above_10_percent": sum(
            value >= 10.0 for value in safe_oracles),
        "scenario_exact": not scenario_failures,
        "pre_intervention_state_replay_exact": not replay_failures,
        "forced_operator_exact": not forced_failures,
        "shadow_base_operator_exact": not shadow_failures,
        "scenario_failures": scenario_failures,
        "replay_failures": replay_failures,
        "forced_failures": forced_failures,
        "shadow_failures": shadow_failures,
        "per_operator": per_operator,
        "states": state_rows,
        "actions": actions,
        "records": records,
        "admission_thresholds": {
            "minimum_positive_safe_states": 3,
            "minimum_mean_safe_oracle_percent": 2.0,
            "minimum_states_above_5_percent": 2,
            "minimum_states_above_10_percent": 1,
        },
    }
    thresholds = summary["admission_thresholds"]
    summary["admission_pass"] = bool(
        summary["instance_count"] == 3
        and summary["state_count"] == 6
        and summary["operator_action_count"] == 30
        and summary["continuation_seed_count"] == 4
        and summary["branch_record_count"] == 120
        and summary["total_run_count"] == 144
        and summary["scenario_exact"]
        and summary["pre_intervention_state_replay_exact"]
        and summary["forced_operator_exact"]
        and summary["shadow_base_operator_exact"]
        and summary["states_with_positive_safe_operator"] >=
        thresholds["minimum_positive_safe_states"]
        and summary["mean_safe_oracle_percent"] >=
        thresholds["minimum_mean_safe_oracle_percent"]
        and summary["states_with_safe_oracle_above_5_percent"] >=
        thresholds["minimum_states_above_5_percent"]
        and summary["states_with_safe_oracle_above_10_percent"] >=
        thresholds["minimum_states_above_10_percent"]
    )
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instances", type=Path, required=True)
    parser.add_argument("--normalization", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--instance-ids", nargs="+", required=True)
    parser.add_argument("--algorithm-seed", type=int, default=777001)
    parser.add_argument("--continuation-seeds", nargs="+", type=int,
                        required=True)
    parser.add_argument("--states-per-instance", type=int, default=2)
    parser.add_argument("--minimum-generation", type=int, default=10)
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
    if (len(args.instance_ids) != 3 or args.states_per_instance != 2
            or len(args.continuation_seeds) != 4):
        parser.error("smoke requires 3 instances, 2 states, and 4 continuations")
    if len(set(args.continuation_seeds)) != 4:
        parser.error("continuation seeds must be unique")
    for path, name in (
            (args.normalization, "normalization"),
            (args.checkpoint, "checkpoint"), (args.runner, "runner")):
        if not path.is_file():
            parser.error(f"missing {name}: {path}")

    args.out.mkdir(parents=True, exist_ok=True)
    control_jobs = []
    instance_paths = {}
    for instance_id in args.instance_ids:
        short = instance_id.replace("train-", "").replace("-", "_")
        instance = args.instances / f"{instance_id}.json"
        if not instance.is_file():
            parser.error(f"missing instance: {instance}")
        instance_paths[short] = instance
        run = args.out / short / f"seed_{args.algorithm_seed}" / "P000"
        control_jobs.append((
            run, run.with_suffix(".nohup.log"),
            _base_command(args, instance, run, args.algorithm_seed),
            args.algorithm_seed, None, None, None))
    control_counts = _execute(
        control_jobs, min(args.max_workers, len(control_jobs)), "CONTROL")

    states = []
    for short, instance in sorted(instance_paths.items()):
        run = args.out / short / f"seed_{args.algorithm_seed}" / "P000"
        selected = _spread_events(
            _jsonl(run / "mutation_outcomes.jsonl"),
            args.states_per_instance, args.minimum_generation)
        for row in selected:
            states.append({
                "instance": short,
                "algorithm_seed": args.algorithm_seed,
                "event": int(row["mixture_event_index"]),
                "generation": int(row["generation"]),
                "parent_id": row["parent_id"],
            })
    manifest = {
        "schema_version": 1,
        "selection": {
            "outcome_independent": True,
            "rule": "evenly spread Base events at or after minimum generation",
            "minimum_generation": args.minimum_generation,
            "states_per_instance": args.states_per_instance,
        },
        "operators": list(OPERATORS),
        "continuation_seeds": list(args.continuation_seeds),
        "states": states,
    }
    (args.out / "selection_manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n",
        encoding="utf-8")

    jobs = []
    for state in states:
        short = state["instance"]
        seed = int(state["algorithm_seed"])
        event = int(state["event"])
        instance = instance_paths[short]
        for continuation_seed in args.continuation_seeds:
            root = (
                args.out / short / f"seed_{seed}" / f"event_{event}" /
                f"continuation_{continuation_seed}")
            reference = root / "REFERENCE"
            jobs.append((
                reference, reference.with_suffix(".nohup.log"),
                _branch_command(
                    args, instance, reference, seed, event,
                    continuation_seed, None),
                seed, event, continuation_seed, None))
            for operator in OPERATORS:
                run = root / f"operator_{operator}"
                jobs.append((
                    run, run.with_suffix(".nohup.log"),
                    _branch_command(
                        args, instance, run, seed, event,
                        continuation_seed, operator),
                    seed, event, continuation_seed, operator))
    branch_counts = _execute(jobs, args.max_workers, "OPERATOR")
    summary = _analyse(args, states, branch_counts)
    (args.out / "AUDIT_COMPLETE.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n",
        encoding="utf-8")
    quick_fields = (
        "completed", "instance_count", "state_count", "operator_action_count",
        "continuation_seed_count", "branch_record_count", "total_run_count",
        "mean_safe_oracle_percent", "median_safe_oracle_percent",
        "maximum_safe_oracle_percent", "states_with_positive_safe_operator",
        "states_with_safe_oracle_above_5_percent",
        "states_with_safe_oracle_above_10_percent", "scenario_exact",
        "pre_intervention_state_replay_exact", "forced_operator_exact",
        "shadow_base_operator_exact", "per_operator", "states",
        "admission_thresholds", "admission_pass",
    )
    quick = {field: summary[field] for field in quick_fields}
    (args.out / "QUICK_RESULT.json").write_text(
        json.dumps(quick, indent=2, allow_nan=False) + "\n",
        encoding="utf-8")
    print(json.dumps(quick, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
