#!/usr/bin/env python3
"""Run resumable Base convergence curves on three selected S0 training instances."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parent
RUNNER = ROOT / "run_learning_components_deterministic.py"
ROLES = ("low", "median", "high")


def _read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _atomic_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n",
        encoding="utf-8")
    temporary.replace(path)


def selected_records(selection):
    if (selection.get("configuration") != "S0"
            or selection.get("instance_count") != 50
            or selection.get("test_instances_used") is not False
            or selection.get("optimisation_outcomes_used") is not False):
        raise ValueError("selection is not the frozen training-only S0 artifact")
    rows = selection.get("selected", [])
    by_role = {row.get("role"): row for row in rows}
    if len(rows) != 3 or set(by_role) != set(ROLES):
        raise ValueError("selection must contain exactly low, median, and high")
    ordered = [by_role[role] for role in ROLES]
    if len({row.get("instance_id") for row in ordered}) != 3:
        raise ValueError("selected instance identifiers must be distinct")
    if any(not str(row.get("instance_id", "")).startswith("train-S0-")
           for row in ordered):
        raise ValueError("pilot selection contains a non-training S0 instance")
    return ordered


def _runtime_environment():
    result = {
        "python": platform.python_version(),
        "python_executable": str(Path(sys.executable).resolve()),
        "platform": platform.platform(),
    }
    try:
        import numpy
        result["numpy"] = numpy.__version__
    except ImportError:
        result["numpy"] = None
    try:
        import torch
        result.update({
            "torch": torch.__version__,
            "torch_cuda_build": torch.version.cuda,
            "torch_cuda_available": torch.cuda.is_available(),
        })
    except ImportError:
        result.update({
            "torch": None, "torch_cuda_build": None,
            "torch_cuda_available": False,
        })
    return result


def _git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
            stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _progress(path: Path):
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _completed(run: Path, record, repeat, seed, args, normalization):
    marker = run / "COMPLETE.json"
    config_path = run / "configuration.json"
    progress_path = run / "search_progress.jsonl"
    if not marker.is_file() or not config_path.is_file() or not progress_path.is_file():
        return False
    summary, config = _read(marker), _read(config_path)
    rows = _progress(progress_path)
    archive = [row["archive_hv"] for row in rows if row["archive_hv"] is not None]
    finite_hv = all(
        isinstance(row.get(name), (int, float)) and math.isfinite(row[name])
        for row in rows for name in ("population_hv", "archive_hv")
        if row.get(name) is not None)
    if (summary.get("completed") is not True
            or summary.get("component") != "BASE"
            or summary.get("stop_reason") != "evaluation_budget"
            or summary.get("real_objective_evaluation_count") != args.evaluation_budget
            or summary.get("evaluation_budget") != args.evaluation_budget
            or config.get("instance_id") != record["instance_id"]
            or config.get("instance_digest") != record["instance_digest"]
            or config.get("population") != args.pop
            or config.get("generations") != args.gens
            or config.get("seeds", {}).get("algorithm") != seed
            or config.get("evaluation_mode") != "deterministic"
            or config.get("policy_training") is not False
            or config.get("policy_checkpoint") is not None
            or not rows or rows[0].get("real_objective_evaluation_count") != args.pop
            or rows[-1].get("real_objective_evaluation_count") != args.evaluation_budget
            or not finite_hv
            or any(b < a - 1e-12 for a, b in zip(archive, archive[1:]))
            or tuple(config.get("normalization", {}).get("minimum", ()))
            != tuple(normalization["minimum"])
            or tuple(config.get("normalization", {}).get("maximum", ()))
            != tuple(normalization["maximum"])
            or tuple(config.get("normalization", {}).get("reference", ()))
            != tuple(normalization["reference"])):
        raise ValueError(
            f"completed convergence run has mismatched metadata: {run} "
            f"(role={record['role']}, repeat={repeat})")
    return True


def _quarantine(path: Path, root: Path):
    if not path.exists() or not any(path.iterdir()):
        return
    destination_root = root / "interrupted_runs"
    destination_root.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    destination = destination_root / f"{path.name}_{stamp}"
    suffix = 1
    while destination.exists():
        destination = destination_root / f"{path.name}_{stamp}_{suffix}"
        suffix += 1
    shutil.move(str(path), str(destination))
    print(f"[RESUME] moved incomplete output to {destination}", flush=True)


def _run(instance: Path, out: Path, log: Path, seed: int, args):
    command = [
        sys.executable, "-u", str(RUNNER),
        "--data", str(args.data),
        "--instance", str(instance),
        "--normalization", str(args.normalization),
        "--component", "BASE",
        "--algorithm-seed", str(seed),
        "--evaluation-budget", str(args.evaluation_budget),
        "--pop", str(args.pop),
        "--gens", str(args.gens),
        "--out", str(out),
    ]
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as handle:
        process = subprocess.Popen(
            command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT,
            text=True)
        run_started = time.perf_counter()
        while True:
            try:
                returncode = process.wait(timeout=args.heartbeat_seconds)
                break
            except subprocess.TimeoutExpired:
                progress_path = out / "search_progress.jsonl"
                if progress_path.is_file():
                    rows = _progress(progress_path)
                    if rows:
                        last = rows[-1]
                        done = int(last["real_objective_evaluation_count"])
                        elapsed = time.perf_counter() - run_started
                        remaining = (
                            elapsed / max(done, 1)
                            * max(0, args.evaluation_budget - done))
                        print(
                            f"[HEARTBEAT] evaluations={done}/"
                            f"{args.evaluation_budget} "
                            f"percent={100.0 * done / args.evaluation_budget:.1f} "
                            f"elapsed_minutes={elapsed / 60.0:.1f} "
                            f"eta_minutes={remaining / 60.0:.1f}",
                            flush=True)
                        continue
                print(
                    f"[HEARTBEAT] preparing initial population "
                    f"elapsed_minutes={(time.perf_counter() - run_started) / 60.0:.1f}",
                    flush=True)
    if returncode:
        raise RuntimeError(
            f"Base convergence run failed with exit {returncode}; "
            f"inspect {log}")


def _write_progress_csv(path: Path, records, args):
    fields = [
        "role", "complexity_rank", "instance_id", "repeat",
        "algorithm_seed", "checkpoint", "generation",
        "real_objective_evaluation_count", "population_hv", "archive_hv",
        "runtime_seconds", "objective_evaluation_seconds_cumulative",
        "feature_construction_seconds_cumulative",
        "inference_seconds_cumulative", "ppo_update_seconds_cumulative",
    ]
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for instance_index, record in enumerate(records):
            for repeat in range(args.repeats):
                seed = args.algorithm_seed_base + instance_index * 100 + repeat
                run = args.out / "runs" / record["role"] / f"repeat_{repeat + 1:02d}"
                for row in _progress(run / "search_progress.jsonl"):
                    writer.writerow({
                        "role": record["role"],
                        "complexity_rank": record["rank"],
                        "instance_id": record["instance_id"],
                        "repeat": repeat + 1,
                        "algorithm_seed": seed,
                        **{name: row.get(name) for name in fields[5:]},
                    })
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path,
                        default=ROOT / "data/data_expanded.xlsx")
    parser.add_argument("--instances", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--normalization", type=Path, required=True)
    parser.add_argument("--algorithm-seed-base", type=int, default=89_200_000)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--evaluation-budget", type=int, default=50_000)
    parser.add_argument("--pop", type=int, default=250)
    parser.add_argument("--gens", type=int, default=250)
    parser.add_argument("--heartbeat-seconds", type=float, default=60.0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.data = args.data.resolve()
    args.instances = args.instances.resolve()
    args.selection = args.selection.resolve()
    args.normalization = args.normalization.resolve()
    args.out = args.out.resolve()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    if args.heartbeat_seconds <= 0:
        parser.error("--heartbeat-seconds must be positive")
    if args.pop < 2 or args.pop % 2:
        parser.error("--pop must be even and at least two")
    if args.evaluation_budget < args.pop + 4:
        parser.error("--evaluation-budget must exceed the initial population")
    for label, path in (
            ("network workbook", args.data),
            ("selection", args.selection),
            ("normalization", args.normalization)):
        if not path.is_file():
            parser.error(f"{label} is missing: {path}")
    if not args.instances.is_dir():
        parser.error(f"instance directory is missing: {args.instances}")
    try:
        selection = _read(args.selection)
        records = selected_records(selection)
        normalization = _read(args.normalization)
        if (normalization.get("source") != "training-only"
                or normalization.get("configuration") != "S0"
                or normalization.get("test_instances_used") is not False
                or normalization.get("validation_instances_used") is not False):
            raise ValueError("normalization is not training-only S0")
        for record in records:
            manifest = args.instances / f"{record['instance_id']}.json"
            value = _read(manifest)
            if (value.get("instance_digest") != record.get("instance_digest")
                    or value.get("split") != "train"
                    or value.get("configuration") != "S0"):
                raise ValueError(f"selection/manifest mismatch: {manifest}")
    except (KeyError, OSError, TypeError, ValueError,
            json.JSONDecodeError) as error:
        parser.error(str(error))

    config = {
        "schema_version": 1,
        "stage": "training-only S0 Base convergence pilot",
        "formal": False,
        "purpose": "freeze the final evaluation budget before formal training",
        "component": "BASE",
        "configuration": "S0",
        "test_instances_used": False,
        "validation_set_used": False,
        "selected_instances": [{
            "role": row["role"], "rank": row["rank"],
            "instance_id": row["instance_id"],
            "instance_digest": row["instance_digest"],
            "candidate_path_token_count": row["candidate_path_token_count"],
        } for row in records],
        "selection_digest": selection["selection_digest"],
        "selection_sha256": _sha256(args.selection),
        "normalization_digest": normalization["normalization_digest"],
        "normalization_sha256": _sha256(args.normalization),
        "source_data_sha256": _sha256(args.data),
        "population": args.pop,
        "evaluation_budget": args.evaluation_budget,
        "generations": args.gens,
        "repeats": args.repeats,
        "algorithm_seed_base": args.algorithm_seed_base,
        "seed_formula": "base + selected_instance_index*100 + repeat_index",
        "heartbeat_seconds": args.heartbeat_seconds,
        "git_commit": _git_commit(),
        "runtime_environment": _runtime_environment(),
        "implementation_sha256": {
            Path(__file__).name: _sha256(Path(__file__)),
            RUNNER.name: _sha256(RUNNER),
            "episode.py": _sha256(ROOT / "transformer_ppo/episode.py"),
        },
    }
    config_path = args.out / "pilot_configuration.json"
    if config_path.exists():
        if _read(config_path) != config:
            parser.error("existing pilot configuration differs; use a new output")
    elif args.out.exists() and any(args.out.iterdir()):
        parser.error("non-empty output has no pilot_configuration.json")
    else:
        args.out.mkdir(parents=True, exist_ok=True)
        _atomic_json(config_path, config)

    tasks = [
        (instance_index, record, repeat)
        for instance_index, record in enumerate(records)
        for repeat in range(args.repeats)
    ]
    started = time.perf_counter()
    newly_completed = 0
    for task_index, (instance_index, record, repeat) in enumerate(tasks):
        seed = args.algorithm_seed_base + instance_index * 100 + repeat
        run = args.out / "runs" / record["role"] / f"repeat_{repeat + 1:02d}"
        if _completed(run, record, repeat, seed, args, normalization):
            print(
                f"[RESUME] {task_index + 1}/{len(tasks)} {record['role']} "
                f"{record['instance_id']} repeat={repeat + 1}", flush=True)
            continue
        _quarantine(run, args.out)
        print(
            f"[BASE] {task_index + 1}/{len(tasks)} {record['role']} "
            f"{record['instance_id']} repeat={repeat + 1} seed={seed}",
            flush=True)
        one_started = time.perf_counter()
        _run(
            args.instances / f"{record['instance_id']}.json", run,
            args.out / "logs" / f"{record['role']}_repeat_{repeat + 1:02d}.log",
            seed, args)
        if not _completed(run, record, repeat, seed, args, normalization):
            raise RuntimeError(f"run exited without a valid completion marker: {run}")
        newly_completed += 1
        elapsed = time.perf_counter() - started
        remaining = len(tasks) - task_index - 1
        eta = elapsed / newly_completed * remaining / 60.0
        print(
            f"[BASE] completed seconds={time.perf_counter() - one_started:.1f} "
            f"eta_minutes={eta:.1f}", flush=True)

    _write_progress_csv(args.out / "convergence_progress.csv", records, args)
    summaries = []
    for instance_index, record in enumerate(records):
        for repeat in range(args.repeats):
            run = args.out / "runs" / record["role"] / f"repeat_{repeat + 1:02d}"
            summary = _read(run / "COMPLETE.json")
            summaries.append({
                "role": record["role"],
                "rank": record["rank"],
                "instance_id": record["instance_id"],
                "repeat": repeat + 1,
                "algorithm_seed": args.algorithm_seed_base + instance_index * 100 + repeat,
                "evaluations": summary["real_objective_evaluation_count"],
                "final_hypervolume": summary["final_hypervolume"],
                "final_front_size": summary["final_front_size"],
                "optimisation_seconds": summary["optimisation_seconds"],
            })
    complete = {
        "completed": True,
        "run_count": len(tasks),
        "total_actual_evaluations": sum(row["evaluations"] for row in summaries),
        "total_optimisation_seconds": sum(
            row["optimisation_seconds"] for row in summaries),
        "wall_seconds_this_invocation": time.perf_counter() - started,
        "progress_csv": "convergence_progress.csv",
        "runs": summaries,
    }
    _atomic_json(args.out / "COMPLETE.json", complete)
    print(json.dumps(complete, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
