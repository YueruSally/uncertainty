#!/usr/bin/env python3
"""Measure whether high-potential one-shot M actions survive new continuations.

The input is a completed unique-action M audit.  For a small, predeclared set
of high-oracle states, this collector replays the same pre-intervention search
state, executes either exact Base mutation or a forced M candidate, and then
restarts both branches from the same post-operation Python/NumPy random seed.
It reports immediate, post-selection, +1-generation, +3-generation, and final
paired HV effects without training or changing the CCP scenario set.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys


EPSILON = 1e-12
HORIZONS = (
    "immediate_archive_delta",
    "post_selection_archive_percent",
    "plus_1_generation_archive_percent",
    "plus_3_generations_archive_percent",
    "final_percent",
)


def _read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl(path: Path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _sha256_json(value):
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sign(value):
    if value is None:
        return None
    if value > EPSILON:
        return 1
    if value < -EPSILON:
        return -1
    return 0


def _mean(values):
    values = [value for value in values if value is not None]
    return statistics.fmean(values) if values else None


def _pearson(xs, ys):
    pairs = [
        (float(x), float(y)) for x, y in zip(xs, ys)
        if x is not None and y is not None
    ]
    if len(pairs) < 2:
        return None
    xs, ys = zip(*pairs)
    x_mean, y_mean = statistics.fmean(xs), statistics.fmean(ys)
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in pairs)
    denominator = math.sqrt(
        sum((x - x_mean) ** 2 for x in xs)
        * sum((y - y_mean) ** 2 for y in ys)
    )
    return numerator / denominator if denominator > EPSILON else None


def _paired_percent(reference, candidate):
    reference = float(reference)
    candidate = float(candidate)
    if abs(reference) <= EPSILON:
        return None
    return 100.0 * (candidate - reference) / reference


def _select_states(audit, requested, minimum_oracle, max_per_instance):
    eligible = [
        state for state in audit["states"]
        if float(state["audited_oracle_percent"]) >= minimum_oracle
        and state.get("actions")
    ]
    eligible.sort(
        key=lambda row: (
            -float(row["audited_oracle_percent"]),
            row["instance"], int(row["algorithm_seed"]), int(row["event"])))
    selected = []
    counts = Counter()
    for state in eligible:
        if counts[state["instance"]] >= max_per_instance:
            continue
        selected.append(state)
        counts[state["instance"]] += 1
        if len(selected) == requested:
            break
    if len(selected) < requested:
        raise ValueError(
            f"only {len(selected)} states satisfy oracle >= {minimum_oracle}% "
            f"with at most {max_per_instance} per instance; requested {requested}")
    return selected


def _select_actions(state, requested):
    actions = [
        action for action in state["actions"]
        if action.get("candidate_index") is not None
        and action.get("selected_replacement_executed") is True
    ]
    actions.sort(key=lambda row: float(row["final_percent"]), reverse=True)
    if len(actions) < requested:
        raise ValueError(
            f"state {state['instance']}:{state['algorithm_seed']}:"
            f"{state['event']} has {len(actions)} executable actions; "
            f"requested {requested}")
    top_count = max(1, requested - 1)
    selected = actions[:top_count]
    if requested > top_count:
        selected.extend(actions[-(requested - top_count):])
    deduplicated = []
    seen = set()
    for action in selected:
        index = int(action["candidate_index"])
        if index not in seen:
            deduplicated.append(action)
            seen.add(index)
    if len(deduplicated) != requested:
        raise ValueError("top/bottom action selection produced duplicates")
    return deduplicated


def _command(args, instance_path, run, algorithm_seed, event,
             continuation_seed, candidate_index=None):
    command = [
        sys.executable, "-u", str(args.runner),
        "--instance", str(instance_path),
        "--normalization", str(args.normalization),
        "--out", str(run),
        "--component", "M",
        "--checkpoint", str(args.checkpoint),
        "--algorithm-seed", str(algorithm_seed),
        "--inference-action-mode", "argmax",
        "--inference-policy-seed", str(continuation_seed + 100_000),
        "--m-continuation-event", str(event),
        "--m-continuation-seed", str(continuation_seed),
        "--evaluation-mode", args.evaluation_mode,
        "--uncertainty-profile", args.uncertainty_profile,
        "--scenario-count", str(args.scenario_count),
        "--ccp-confidence", str(args.ccp_confidence),
        "--evaluation-budget", str(args.evaluation_budget),
        "--pop", str(args.pop),
        "--gens", str(args.gens),
        "--device", args.device,
    ]
    if candidate_index is None:
        command.extend(["--model-choice-probability", "0"])
    else:
        command.extend([
            "--model-choice-event", str(event),
            "--m-forced-candidate-index", str(candidate_index),
        ])
    return command


def _target_model_row(run, event):
    rows = [
        row for row in _jsonl(run / "mutation_outcomes.jsonl")
        if row.get("mixture_event_index") == event
        and row.get("decision_kind") == "model_choice"
    ]
    if len(rows) != 1:
        raise RuntimeError(
            f"expected one model action at event {event}, found {len(rows)}: {run}")
    return rows[0]


def _target_state(run, event):
    rows = [
        row for row in _jsonl(run / "mutation_policy_states.jsonl")
        if row.get("mixture_event_index") == event
        and row.get("decision_stage", "location") == "location"
    ]
    if len(rows) != 1:
        raise RuntimeError(
            f"expected one policy state at event {event}, found {len(rows)}: {run}")
    return rows[0]


def _valid_complete(run, algorithm_seed, event, continuation_seed,
                    candidate_index):
    complete_path = run / "COMPLETE.json"
    config_path = run / "configuration.json"
    if not complete_path.is_file() or not config_path.is_file():
        return False
    try:
        complete = _read(complete_path)
        config = _read(config_path)
        control = config["learning_control"]
        valid = (
            complete.get("completed") is True
            and complete.get("component") == "M"
            and config.get("component") == "M"
            and config.get("seeds", {}).get("algorithm") == algorithm_seed
            and control.get("m_continuation_event") == event
            and control.get("m_continuation_seed") == continuation_seed
            and complete.get("continuation_reseed_count") == 1
        )
        model_rows = [
            row for row in _jsonl(run / "mutation_outcomes.jsonl")
            if row.get("decision_kind") == "model_choice"
        ]
        if candidate_index is None:
            valid = valid and not model_rows
        else:
            target = _target_model_row(run, event)
            _target_state(run, event)
            valid = valid and len(model_rows) == 1
            valid = valid and int(
                target.get("selected_candidate_index", -1)) == candidate_index
            valid = valid and int(target.get("policy_metadata", {}).get(
                "forced_action_index", -1)) == candidate_index
        return bool(valid)
    except (KeyError, OSError, TypeError, ValueError, RuntimeError):
        return False


def _run_job(job):
    (run, log, command, algorithm_seed, event,
     continuation_seed, candidate_index) = job
    if _valid_complete(
            run, algorithm_seed, event, continuation_seed, candidate_index):
        return run, "existing"
    if run.exists():
        raise RuntimeError(f"incomplete output exists; use a fresh root: {run}")
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
            run, algorithm_seed, event, continuation_seed, candidate_index):
        raise RuntimeError(
            f"run failed with exit {completed.returncode}; inspect {log}")
    return run, "complete"


def _execute(jobs, workers):
    counts = {"existing": 0, "complete": 0}
    print(f"[CONTINUATION] jobs={len(jobs)} workers={workers}", flush=True)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(_run_job, job) for job in jobs]
        for completed_count, future in enumerate(as_completed(futures), 1):
            run, status = future.result()
            counts[status] += 1
            print(
                f"[CONTINUATION] {completed_count}/{len(jobs)} "
                f"{status}: {run}", flush=True)
    return counts


def _progress(run):
    rows = [
        row for row in _jsonl(run / "search_progress.jsonl")
        if row.get("checkpoint") == "environmental_selection"
    ]
    return {int(row["generation"]): row for row in rows}


def _progress_percent(reference, candidate, generation, field):
    if generation not in reference or generation not in candidate:
        return None
    left = reference[generation].get(field)
    right = candidate[generation].get(field)
    if left is None or right is None:
        return None
    return _paired_percent(left, right)


def _horizon_summary(records, horizon):
    selected = [
        record for record in records
        if record.get(horizon) is not None
    ]
    sign_rows = [
        record for record in selected
        if _sign(record[horizon]) is not None
    ]
    return {
        "sample_count": len(selected),
        "sign_accuracy_with_final": (
            _mean([
                _sign(record[horizon]) == _sign(record["final_percent"])
                for record in sign_rows
            ]) if sign_rows else None),
        "pearson_with_final": _pearson(
            [record[horizon] for record in selected],
            [record["final_percent"] for record in selected]),
    }


def _ranking_summary(records, horizon):
    grouped = defaultdict(list)
    for record in records:
        grouped[(
            record["instance"], record["algorithm_seed"],
            record["event"], record["continuation_seed"]
        )].append(record)
    top_matches = []
    comparable = correct = 0
    for rows in grouped.values():
        usable = [row for row in rows if row.get(horizon) is not None]
        if len(usable) < 2:
            continue
        final_span = max(row["final_percent"] for row in usable) - min(
            row["final_percent"] for row in usable)
        if final_span <= EPSILON:
            continue
        predicted = max(usable, key=lambda row: row[horizon])
        best_final = max(row["final_percent"] for row in usable)
        top_matches.append(
            abs(predicted["final_percent"] - best_final) <= EPSILON)
        for left_index, left in enumerate(usable):
            for right in usable[left_index + 1:]:
                x_sign = _sign(left[horizon] - right[horizon])
                y_sign = _sign(
                    left["final_percent"] - right["final_percent"])
                if x_sign == 0 or y_sign == 0:
                    continue
                comparable += 1
                correct += x_sign == y_sign
    return {
        "informative_state_continuations": len(top_matches),
        "top_1_accuracy": _mean(top_matches),
        "pairwise_accuracy": correct / comparable if comparable else None,
        "comparable_pairs": comparable,
    }


def _analyse(args, manifest, status_counts):
    records = []
    scenario_failures = []
    forced_failures = []
    state_hashes = defaultdict(set)
    for state in manifest["states"]:
        short = state["instance"]
        algorithm_seed = int(state["algorithm_seed"])
        event = int(state["event"])
        for continuation_seed in manifest["continuation_seeds"]:
            root = (
                args.out / short / f"seed_{algorithm_seed}" /
                f"event_{event}" / f"continuation_{continuation_seed}")
            reference_run = root / "REFERENCE"
            reference_complete = _read(reference_run / "COMPLETE.json")
            reference_config = _read(reference_run / "configuration.json")
            reference_progress = _progress(reference_run)
            for action in state["actions"]:
                candidate_index = int(action["candidate_index"])
                run = root / f"candidate_{candidate_index}"
                complete = _read(run / "COMPLETE.json")
                config = _read(run / "configuration.json")
                if config["scenario_digest"] != reference_config["scenario_digest"]:
                    scenario_failures.append(str(run))
                outcome = _target_model_row(run, event)
                if int(outcome.get("selected_candidate_index", -1)) != candidate_index:
                    forced_failures.append(str(run))
                state_row = _target_state(run, event)
                state_hashes[(short, algorithm_seed, event)].add(
                    _sha256_json(state_row["observation"]))
                generation = int(outcome["generation"])
                progress = _progress(run)
                final_percent = _paired_percent(
                    reference_complete["final_hypervolume"],
                    complete["final_hypervolume"])
                records.append({
                    "instance": short,
                    "algorithm_seed": algorithm_seed,
                    "event": event,
                    "generation": generation,
                    "continuation_seed": int(continuation_seed),
                    "candidate_index": candidate_index,
                    "action_signature": action["action_signature"],
                    "selection_role": action["selection_role"],
                    "original_final_percent": float(action["final_percent"]),
                    "immediate_archive_delta": float(
                        outcome.get("reward_hv_delta", 0.0)),
                    "post_selection_archive_percent": _progress_percent(
                        reference_progress, progress, generation, "archive_hv"),
                    "plus_1_generation_archive_percent": _progress_percent(
                        reference_progress, progress, generation + 1, "archive_hv"),
                    "plus_3_generations_archive_percent": _progress_percent(
                        reference_progress, progress, generation + 3, "archive_hv"),
                    "final_percent": final_percent,
                    "reference_final_hv": float(
                        reference_complete["final_hypervolume"]),
                    "candidate_final_hv": float(complete["final_hypervolume"]),
                    "survived_selection": bool(outcome.get(
                        "survived_environmental_selection")),
                    "run": str(run),
                })

    grouped = defaultdict(list)
    for record in records:
        grouped[(
            record["instance"], record["algorithm_seed"],
            record["event"], record["candidate_index"]
        )].append(record)
    action_stability = []
    for key, rows in sorted(grouped.items()):
        values = [row["final_percent"] for row in rows]
        signs = Counter(_sign(value) for value in values)
        action_stability.append({
            "instance": key[0],
            "algorithm_seed": key[1],
            "event": key[2],
            "candidate_index": key[3],
            "selection_role": rows[0]["selection_role"],
            "original_final_percent": rows[0]["original_final_percent"],
            "continuation_count": len(values),
            "final_percent_each": values,
            "mean_final_percent": statistics.fmean(values),
            "median_final_percent": statistics.median(values),
            "minimum_final_percent": min(values),
            "maximum_final_percent": max(values),
            "positive_fraction": signs[1] / len(values),
            "negative_fraction": signs[-1] / len(values),
            "tie_fraction": signs[0] / len(values),
            "majority_sign_agreement": max(signs.values()) / len(values),
            "stable_positive": signs[1] / len(values) >= 0.75,
        })
    top_actions = [
        row for row in action_stability if row["selection_role"] == "top_1"
    ]
    horizon_summaries = {
        horizon: {
            **_horizon_summary(records, horizon),
            **_ranking_summary(records, horizon),
        }
        for horizon in HORIZONS
    }
    eligible_horizons = [
        (name, metrics) for name, metrics in horizon_summaries.items()
        if name != "final_percent"
        and metrics["sign_accuracy_with_final"] is not None
    ]
    best_horizon = max(
        eligible_horizons,
        key=lambda item: (
            item[1]["sign_accuracy_with_final"],
            item[1]["pearson_with_final"]
            if item[1]["pearson_with_final"] is not None else -2.0),
    )[0]
    state_replay_failures = [
        ":".join(map(str, key))
        for key, hashes in state_hashes.items() if len(hashes) != 1
    ]
    top_stable_fraction = _mean([
        row["stable_positive"] for row in top_actions])
    mean_top_positive_fraction = _mean([
        row["positive_fraction"] for row in top_actions])
    summary = {
        "completed": True,
        "definition": (
            "same pre-intervention M state/action with matched Base reference "
            "and shared post-operation continuation seed under frozen CCP"),
        "source_audit": str(args.audit_root),
        "selected_state_count": len(manifest["states"]),
        "selected_action_count": len(action_stability),
        "continuation_seed_count": len(manifest["continuation_seeds"]),
        "branch_record_count": len(records),
        "total_run_count": status_counts["existing"] + status_counts["complete"],
        "scenario_exact": not scenario_failures,
        "forced_execution_exact": not forced_failures,
        "pre_intervention_state_replay_exact": not state_replay_failures,
        "scenario_failures": scenario_failures,
        "forced_execution_failures": forced_failures,
        "state_replay_failures": state_replay_failures,
        "top_action_stable_positive_states": sum(
            row["stable_positive"] for row in top_actions),
        "top_action_stable_positive_state_fraction": top_stable_fraction,
        "mean_top_action_positive_continuation_fraction": (
            mean_top_positive_fraction),
        "mean_all_action_majority_sign_agreement": _mean([
            row["majority_sign_agreement"] for row in action_stability]),
        "best_short_horizon": best_horizon,
        "best_short_horizon_metrics": horizon_summaries[best_horizon],
        "horizon_summaries": horizon_summaries,
        "action_stability": action_stability,
        "records": records,
        "admission_pass": bool(
            not scenario_failures and not forced_failures
            and not state_replay_failures
            and top_stable_fraction is not None
            and top_stable_fraction >= 0.70
            and mean_top_positive_fraction is not None
            and mean_top_positive_fraction >= 0.70
            and horizon_summaries[best_horizon][
                "sign_accuracy_with_final"] >= 0.70),
    }
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-root", type=Path, required=True)
    parser.add_argument("--instances", type=Path, required=True)
    parser.add_argument("--normalization", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--continuation-seeds", nargs="+", type=int,
                        default=(31001, 31002, 31003, 31004))
    parser.add_argument("--state-count", type=int, default=12)
    parser.add_argument("--actions-per-state", type=int, default=3)
    parser.add_argument("--minimum-oracle-percent", type=float, default=5.0)
    parser.add_argument("--max-states-per-instance", type=int, default=4)
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

    for path, name in (
            (args.audit_root / "AUDIT_COMPLETE.json", "source audit"),
            (args.normalization, "normalization"),
            (args.checkpoint, "checkpoint"), (args.runner, "runner")):
        if not path.is_file():
            parser.error(f"missing {name}: {path}")
    if (args.state_count < 1 or args.actions_per_state < 2
            or args.max_states_per_instance < 1 or args.max_workers < 1
            or not args.continuation_seeds):
        parser.error("states, actions, workers, and continuation seeds are invalid")
    if len(set(args.continuation_seeds)) != len(args.continuation_seeds):
        parser.error("continuation seeds must be unique")

    audit = _read(args.audit_root / "AUDIT_COMPLETE.json")
    for field in (
            "completed", "p000_exact_base", "scenario_exact",
            "duplicate_replay_consistent", "forced_execution_exact"):
        if audit.get(field) is not True:
            parser.error(f"source audit failed prerequisite: {field}")
    states = _select_states(
        audit, args.state_count, args.minimum_oracle_percent,
        args.max_states_per_instance)
    manifest_states = []
    for state in states:
        actions = _select_actions(state, args.actions_per_state)
        action_rows = []
        for index, action in enumerate(actions):
            row = dict(action)
            row["selection_role"] = (
                "top_1" if index == 0 else
                f"top_{index + 1}" if index < args.actions_per_state - 1
                else "bottom_1")
            action_rows.append(row)
        manifest_states.append({
            "instance": state["instance"],
            "algorithm_seed": int(state["algorithm_seed"]),
            "event": int(state["event"]),
            "generation": int(state["generation"]),
            "audited_oracle_percent": float(state["audited_oracle_percent"]),
            "actions": action_rows,
        })
    manifest = {
        "schema_version": 1,
        "source_audit": str(args.audit_root),
        "selection": {
            "state_count": args.state_count,
            "minimum_oracle_percent": args.minimum_oracle_percent,
            "max_states_per_instance": args.max_states_per_instance,
            "actions_per_state": args.actions_per_state,
            "action_rule": "top actions plus one bottom contrast action",
        },
        "continuation_seeds": list(args.continuation_seeds),
        "states": manifest_states,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "selection_manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n",
        encoding="utf-8")

    jobs = []
    for state in manifest_states:
        short = state["instance"]
        algorithm_seed = int(state["algorithm_seed"])
        event = int(state["event"])
        instance_id = short.replace("_", "-")
        instance_path = args.instances / f"train-{instance_id}.json"
        if not instance_path.is_file():
            parser.error(f"missing instance: {instance_path}")
        for continuation_seed in args.continuation_seeds:
            root = (
                args.out / short / f"seed_{algorithm_seed}" /
                f"event_{event}" / f"continuation_{continuation_seed}")
            reference = root / "REFERENCE"
            jobs.append((
                reference, reference.with_suffix(".nohup.log"),
                _command(
                    args, instance_path, reference, algorithm_seed, event,
                    continuation_seed, None),
                algorithm_seed, event, continuation_seed, None))
            for action in state["actions"]:
                candidate_index = int(action["candidate_index"])
                run = root / f"candidate_{candidate_index}"
                jobs.append((
                    run, run.with_suffix(".nohup.log"),
                    _command(
                        args, instance_path, run, algorithm_seed, event,
                        continuation_seed, candidate_index),
                    algorithm_seed, event, continuation_seed, candidate_index))

    counts = _execute(jobs, min(args.max_workers, len(jobs)))
    summary = _analyse(args, manifest, counts)
    (args.out / "STABILITY_COMPLETE.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n",
        encoding="utf-8")
    print(json.dumps({
        key: summary[key] for key in (
            "completed", "selected_state_count", "selected_action_count",
            "continuation_seed_count", "branch_record_count",
            "top_action_stable_positive_state_fraction",
            "mean_top_action_positive_continuation_fraction",
            "mean_all_action_majority_sign_agreement", "best_short_horizon",
            "best_short_horizon_metrics", "scenario_exact",
            "forced_execution_exact", "pre_intervention_state_replay_exact",
            "admission_pass")
    }, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
