#!/usr/bin/env python3
"""Collect matched one-shot mutation branches for downstream supervision."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
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


def _short_id(instance_id: str):
    value = instance_id.removeprefix("train-")
    return value.replace("-", "_")


def _valid_complete(run: Path, seed: int, require_policy_state: bool):
    marker = run / "COMPLETE.json"
    config_path = run / "configuration.json"
    if not marker.is_file() or not config_path.is_file():
        return False
    try:
        marker_data = _read(marker)
        config = _read(config_path)
    except (OSError, ValueError):
        return False
    valid = (
        marker_data.get("completed") is True
        and marker_data.get("component") == "M"
        and config.get("component") == "M"
        and config.get("seeds", {}).get("algorithm") == seed
    )
    if require_policy_state:
        valid = valid and (run / "mutation_policy_states.jsonl").is_file()
    return bool(valid)


def _command(args, instance_path, checkpoint, run, seed, policy_seed,
             model_choice_probability=None, model_choice_event=None):
    result = [
        sys.executable, "-u", str(args.runner),
        "--instance", str(instance_path),
        "--normalization", str(args.normalization),
        "--out", str(run),
        "--component", "M",
        "--checkpoint", str(checkpoint),
        "--algorithm-seed", str(seed),
        "--inference-action-mode", "sample",
        "--inference-policy-seed", str(policy_seed),
        "--evaluation-budget", str(args.evaluation_budget),
        "--pop", str(args.pop),
        "--gens", str(args.gens),
        "--device", args.device,
    ]
    if model_choice_probability is not None:
        result.extend([
            "--model-choice-probability", str(model_choice_probability)])
    if model_choice_event is not None:
        result.extend(["--model-choice-event", str(model_choice_event)])
    return result


def _run_job(job):
    run, log, command, seed, require_policy_state = job
    if _valid_complete(run, seed, require_policy_state):
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
            run, seed, require_policy_state):
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
    progress_generations = {
        int(row["generation"])
        for row in _jsonl(base_run / "search_progress.jsonl")
        if row.get("checkpoint") == "environmental_selection"
    }
    rows = [
        row for row in _jsonl(base_run / "mutation_outcomes.jsonl")
        if row.get("decision_kind") == "base_mixture_choice"
        and row.get("operator") == "replace"
        and row.get("mixture_event_index") is not None
        and int(row.get("generation", -1)) + 1 in progress_generations
    ]
    rows.sort(key=lambda row: int(row["mixture_event_index"]))
    if len(rows) < count:
        raise ValueError(
            f"{base_run} has only {len(rows)} usable replace events; "
            f"requested {count}")
    if count == 1:
        return [rows[len(rows) // 2]]
    indices = [round(offset * (len(rows) - 1) / (count - 1))
               for offset in range(count)]
    if len(set(indices)) != count:
        raise ValueError(f"could not select {count} distinct events: {base_run}")
    return [rows[index] for index in indices]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instances", type=Path, required=True)
    parser.add_argument("--normalization", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--instance-ids", nargs="+", default=[
        f"train-S0-{index:03d}" for index in range(1, 11)])
    parser.add_argument("--algorithm-seed", type=int, default=777001)
    parser.add_argument("--policy-seeds", nargs="+", type=int,
                        default=(28101, 28102, 28103, 28104))
    parser.add_argument("--states-per-instance", type=int, default=4)
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--evaluation-budget", type=int, default=400)
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
    if not args.normalization.is_file():
        parser.error(f"normalization file is missing: {args.normalization}")

    # A resumed/expanded collection must not expose a stale success marker
    # while new state/action branches are still running.
    completion_marker = args.out / "COLLECTION_COMPLETE.json"
    completion_marker.unlink(missing_ok=True)

    specifications = []
    for instance_id in args.instance_ids:
        short = _short_id(instance_id)
        instance_path = args.instances / f"{instance_id}.json"
        checkpoint = args.checkpoint_root / f"heldout_{short}.pt"
        if not instance_path.is_file():
            parser.error(f"instance is missing: {instance_path}")
        if not checkpoint.is_file():
            parser.error(f"checkpoint is missing: {checkpoint}")
        specifications.append((instance_id, short, instance_path, checkpoint))

    base_jobs = []
    for _, short, instance_path, checkpoint in specifications:
        run = args.out / short / f"seed_{args.algorithm_seed}" / "P000"
        log = run.with_suffix(".nohup.log")
        command = _command(
            args, instance_path, checkpoint, run, args.algorithm_seed,
            args.policy_seeds[0], model_choice_probability=0.0)
        base_jobs.append((run, log, command, args.algorithm_seed, False))
    base_counts = _execute(base_jobs, args.max_workers, "BASE")

    manifest = {
        "schema_version": 1,
        "algorithm_seed": args.algorithm_seed,
        "policy_seeds": args.policy_seeds,
        "states_per_instance": args.states_per_instance,
        "evaluation_budget": args.evaluation_budget,
        "population": args.pop,
        "instances": [],
    }
    branch_jobs = []
    for instance_id, short, instance_path, checkpoint in specifications:
        base_run = args.out / short / f"seed_{args.algorithm_seed}" / "P000"
        selected = _spread_events(base_run, args.states_per_instance)
        state_rows = []
        for row in selected:
            event = int(row["mixture_event_index"])
            generation = int(row["generation"])
            state_rows.append({"event": event, "generation": generation})
            state_root = (
                args.out / short / f"seed_{args.algorithm_seed}"
                / f"event_{event}")
            for policy_seed in args.policy_seeds:
                run = state_root / f"policy_{policy_seed}"
                log = run.with_suffix(".nohup.log")
                command = _command(
                    args, instance_path, checkpoint, run,
                    args.algorithm_seed, policy_seed,
                    model_choice_event=event)
                branch_jobs.append(
                    (run, log, command, args.algorithm_seed, True))
        manifest["instances"].append({
            "instance_id": instance_id,
            "short_id": short,
            "checkpoint": str(checkpoint.resolve()),
            "states": state_rows,
        })

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "collection_manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n",
        encoding="utf-8")
    branch_counts = _execute(branch_jobs, args.max_workers, "BRANCH")
    summary = {
        "completed": True,
        "base_run_count": len(base_jobs),
        "branch_run_count": len(branch_jobs),
        "state_count": sum(
            len(instance["states"]) for instance in manifest["instances"]),
        "base_status_counts": base_counts,
        "branch_status_counts": branch_counts,
        "manifest": str((args.out / "collection_manifest.json").resolve()),
    }
    completion_marker.write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n",
        encoding="utf-8")
    print(json.dumps(summary, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
