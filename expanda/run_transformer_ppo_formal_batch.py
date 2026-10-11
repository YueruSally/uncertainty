#!/usr/bin/env python3
"""Run one resumable paired formal-test block for four frozen CCP100 methods."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

from transformer_ppo.config import TEST_CONFIGS
from transformer_ppo.statistics import formal_report


ROOT = Path(__file__).resolve().parent
CONTROL = ROOT / "run_transformer_ppo_control_ccp100.py"
POLICY = ROOT / "run_ppo_policy_ccp100.py"
OOS = ROOT / "run_transformer_ppo_oos.py"
HV = ROOT / "compute_transformer_ppo_oos_hv.py"
METHODS = ("random", "rule", "transformer", "mlp")
METHOD_NAMES = {
    "random": "Random-CCP100",
    "rule": "Rule-CCP100",
    "transformer": "transformer-ppo-masked",
    "mlp": "mlp-ppo-masked",
}


def _read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT,
            text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _atomic_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _checkpoint_info(path: Path, architecture: str, replication: int,
                     catalog_digest: str, normalization_digest: str):
    path = path.resolve()
    root = path.parent
    config = _read(root / "training_configuration.json")
    summary = _read(root / "training_summary.json")
    expected_path = (root / summary["checkpoint"]).resolve()
    actual_sha = _sha256(path)
    if (expected_path != path
            or summary.get("completed") is not True
            or len(summary.get("episodes", [])) != 450
            or summary.get("checkpoint_sha256") != actual_sha
            or config.get("architecture") != architecture
            or config.get("policy_seed") != replication
            or config.get("episode_count") != 450
            or config.get("catalog_digest") != catalog_digest
            or config.get("normalization_digest") != normalization_digest):
        raise ValueError(f"invalid frozen {architecture} checkpoint provenance: {path}")
    return {
        "path": str(path),
        "sha256": actual_sha,
        "training_configuration_sha256": _sha256(root / "training_configuration.json"),
        "training_summary_sha256": _sha256(root / "training_summary.json"),
        "policy_seed": replication,
        "architecture": architecture,
    }


def _seeds(args, index):
    offset = (args.replication - 1) * 1_000_000 + index
    return args.algorithm_seed_base + offset, args.policy_seed_base + offset


def _paths(out: Path, instance_id: str, method: str):
    root = out / "instances" / instance_id / method
    return root / "run", root / "oos"


def _completed_run(run_dir: Path, record: dict, method: str, args,
                   algorithm_seed: int, policy_seed: int, checkpoint_sha=None):
    complete_path = run_dir / "COMPLETE.json"
    config_path = run_dir / "configuration.json"
    front_path = run_dir / "final_feasible_nondominated.json"
    if not complete_path.exists() or not config_path.exists() or not front_path.exists():
        return False
    complete, config = _read(complete_path), _read(config_path)
    expected_method = METHOD_NAMES[method]
    seeds = config.get("seeds", {})
    if (complete.get("completed") is not True
            or complete.get("method") != expected_method
            or complete.get("stop_reason") != "evaluation_budget"
            or complete.get("evaluation_budget") != args.evaluation_budget
            or not args.evaluation_budget - 3 <= int(
                complete.get("ccp_evaluation_count", -1)) <= args.evaluation_budget
            or config.get("method") != expected_method
            or config.get("instance_id") != record["instance_id"]
            or config.get("instance_digest") != record["instance_digest"]
            or config.get("evaluation_budget") != args.evaluation_budget
            or config.get("population") != args.pop
            or config.get("generations") != args.gens
            or seeds.get("algorithm") != algorithm_seed
            or seeds.get("training") != record["ccp100_seed"]
            or seeds.get("path") != record["path_seed"]):
        raise ValueError(f"completed formal run metadata mismatch: {run_dir}")
    if method in ("random", "rule"):
        if seeds.get("policy") != policy_seed:
            raise ValueError(f"control policy seed mismatch: {run_dir}")
    else:
        provenance = config.get("policy_checkpoint") or {}
        if (config.get("policy_training") is not False
                or provenance.get("checkpoint_sha256") != checkpoint_sha
                or provenance.get("policy_seed") != args.replication
                or provenance.get("catalog_digest") != args.catalog_digest
                or provenance.get("normalization_digest") != args.normalization_digest):
            raise ValueError(f"PPO checkpoint provenance mismatch: {run_dir}")
    front = _read(front_path)
    if not isinstance(front, list):
        raise ValueError(f"formal front is not a JSON list: {front_path}")
    return True


def _completed_oos(oos_dir: Path, run_dir: Path, record: dict):
    json_path = oos_dir / "oos_5000.json"
    npz_path = oos_dir / "oos_5000.npz"
    if not json_path.exists() or not npz_path.exists():
        return False
    result = _read(json_path)
    source_path = run_dir / "final_feasible_nondominated.json"
    source = _read(source_path)
    if (result.get("instance_id") != record["instance_id"]
            or result.get("instance_digest") != record["instance_digest"]
            or result.get("size") != 5000
            or result.get("oos_seed") != record["oos5000_seed"]
            or result.get("source_sha256") != _sha256(source_path)
            or result.get("evaluated_decisions") != len(source)
            or len(result.get("candidates", [])) != len(source)):
        raise ValueError(f"OOS result metadata mismatch: {oos_dir}")
    return True


def _completed_hv(path: Path, record: dict):
    if not path.exists():
        return False
    rows = _read(path)
    if (not isinstance(rows, list) or len(rows) != 4
            or {row.get("method") for row in rows} != set(METHODS)
            or any(row.get("instance_id") != record["instance_id"]
                   or row.get("configuration") != record["configuration"]
                   for row in rows)
            or any(not math.isfinite(float(row.get("oos_hv", float("nan"))))
                   or float(row["oos_hv"]) < 0.0 for row in rows)
            or len({row.get("oos_master_digest") for row in rows}) != 1):
        raise ValueError(f"OOS-HV result metadata mismatch: {path}")
    return True


def _completed_instance(index, record, args, checkpoints):
    algorithm_seed, policy_seed = _seeds(args, index)
    scenario_digests, oos_digests = set(), set()
    for method in METHODS:
        run_dir, oos_dir = _paths(args.out, record["instance_id"], method)
        checkpoint = checkpoints.get(method)
        checkpoint_sha = checkpoint["sha256"] if checkpoint else None
        if not _completed_run(
                run_dir, record, method, args, algorithm_seed, policy_seed,
                checkpoint_sha=checkpoint_sha):
            return False
        if not _completed_oos(oos_dir, run_dir, record):
            return False
        scenario_digests.add(_read(run_dir / "configuration.json")["scenario_digest"])
        oos_digests.add(_read(oos_dir / "oos_5000.json")["oos_master_digest"])
    if len(scenario_digests) != 1 or len(oos_digests) != 1:
        raise ValueError(
            f"completed methods do not share paired scenarios: {record['instance_id']}")
    return _completed_hv(
        args.out / "instances" / record["instance_id"] / "oos_hv.json", record)


def _quarantine(path: Path, out: Path, label: str):
    if not path.exists():
        return
    interrupted = out / "interrupted"
    interrupted.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    destination = interrupted / f"{label}_{stamp}"
    suffix = 1
    while destination.exists():
        destination = interrupted / f"{label}_{stamp}_{suffix}"
        suffix += 1
    path.rename(destination)
    print(f"[FORMAL] quarantined incomplete output: {destination}", flush=True)


def _run_command(command, log_path: Path, environment: dict):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write("\n[COMMAND] " + " ".join(command) + "\n")
        handle.flush()
        result = subprocess.run(
            command, stdout=handle, stderr=subprocess.STDOUT,
            cwd=ROOT.parent, env=environment, check=False)
    if result.returncode:
        raise RuntimeError(
            f"command failed with exit code {result.returncode}; see {log_path}")


def _run_instance(task, args, checkpoints):
    index, record = task
    instance_id = record["instance_id"]
    instance_path = args.instances / f"{instance_id}.json"
    algorithm_seed, policy_seed = _seeds(args, index)
    device = args.devices[index % len(args.devices)]
    environment = os.environ.copy()
    environment.update({
        "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1",
        "PYTHONUNBUFFERED": "1",
    })
    started = time.perf_counter()
    scenario_digests = set()
    oos_digests = set()
    for method in METHODS:
        run_dir, oos_dir = _paths(args.out, instance_id, method)
        checkpoint = checkpoints.get(method)
        checkpoint_sha = checkpoint["sha256"] if checkpoint else None
        if not _completed_run(
                run_dir, record, method, args, algorithm_seed, policy_seed,
                checkpoint_sha=checkpoint_sha):
            _quarantine(run_dir, args.out, f"{instance_id}_{method}_run")
            log = args.out / "logs" / instance_id / f"{method}_run.log"
            if method in ("random", "rule"):
                command = [
                    sys.executable, "-u", str(CONTROL),
                    "--data", str(args.data), "--instance", str(instance_path),
                    "--out", str(run_dir), "--policy", method,
                    "--algorithm-seed", str(algorithm_seed),
                    "--policy-seed", str(policy_seed),
                    "--evaluation-budget", str(args.evaluation_budget),
                    "--pop", str(args.pop), "--gens", str(args.gens),
                ]
                method_environment = dict(environment, CUDA_VISIBLE_DEVICES="")
            else:
                command = [
                    sys.executable, "-u", str(POLICY),
                    "--data", str(args.data), "--instance", str(instance_path),
                    "--checkpoint", checkpoint["path"],
                    "--normalization", str(args.normalization),
                    "--out", str(run_dir),
                    "--algorithm-seed", str(algorithm_seed),
                    "--evaluation-budget", str(args.evaluation_budget),
                    "--pop", str(args.pop), "--gens", str(args.gens),
                    "--device", "cuda",
                ]
                method_environment = dict(environment, CUDA_VISIBLE_DEVICES=device)
            _run_command(command, log, method_environment)
        if not _completed_run(
                run_dir, record, method, args, algorithm_seed, policy_seed,
                checkpoint_sha=checkpoint_sha):
            raise RuntimeError(f"formal method exited without valid completion: {run_dir}")
        scenario_digests.add(_read(run_dir / "configuration.json")["scenario_digest"])

        if not _completed_oos(oos_dir, run_dir, record):
            _quarantine(oos_dir, args.out, f"{instance_id}_{method}_oos")
            command = [
                sys.executable, "-u", str(OOS),
                "--data", str(args.data), "--instance", str(instance_path),
                "--run", str(run_dir), "--out", str(oos_dir),
            ]
            log = args.out / "logs" / instance_id / f"{method}_oos.log"
            _run_command(command, log, dict(environment, CUDA_VISIBLE_DEVICES=""))
        if not _completed_oos(oos_dir, run_dir, record):
            raise RuntimeError(f"OOS stage exited without valid completion: {oos_dir}")
        oos_digests.add(_read(oos_dir / "oos_5000.json")["oos_master_digest"])

    if len(scenario_digests) != 1:
        raise ValueError(f"four methods did not share CCP100 scenarios: {instance_id}")
    if len(oos_digests) != 1:
        raise ValueError(f"four methods did not share OOS-5000 scenarios: {instance_id}")

    hv_path = args.out / "instances" / instance_id / "oos_hv.json"
    if not _completed_hv(hv_path, record):
        _quarantine(hv_path, args.out, f"{instance_id}_oos_hv")
        command = [
            sys.executable, "-u", str(HV),
            "--instance", str(instance_path),
            "--normalization", str(args.normalization),
        ]
        for method in METHODS:
            _, oos_dir = _paths(args.out, instance_id, method)
            command.extend(["--oos", f"{method}={oos_dir / 'oos_5000.json'}"])
        command.extend(["--out", str(hv_path)])
        log = args.out / "logs" / instance_id / "oos_hv.log"
        _run_command(command, log, dict(environment, CUDA_VISIBLE_DEVICES=""))
    if not _completed_hv(hv_path, record):
        raise RuntimeError(f"HV stage exited without valid completion: {hv_path}")
    return instance_id, time.perf_counter() - started


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path,
                        default=ROOT / "data/data_expanded.xlsx")
    parser.add_argument("--instances", type=Path, required=True)
    parser.add_argument("--normalization", type=Path, required=True)
    parser.add_argument("--transformer-checkpoint", type=Path, required=True)
    parser.add_argument("--mlp-checkpoint", type=Path, required=True)
    parser.add_argument("--replication", type=int, choices=(1, 2, 3), required=True)
    parser.add_argument("--catalog-digest", required=True)
    parser.add_argument("--normalization-digest", required=True)
    parser.add_argument("--evaluation-budget", type=int, default=15_000)
    parser.add_argument("--pop", type=int, default=100)
    parser.add_argument("--gens", type=int, default=1_000)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--devices", default="0,1")
    parser.add_argument("--algorithm-seed-base", type=int, default=80_000_000)
    parser.add_argument("--policy-seed-base", type=int, default=90_000_000)
    parser.add_argument("--development-max-instances", type=int,
                        help="Smoke-only cap; forbidden for a formal completion artifact.")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.data = args.data.resolve()
    args.instances = args.instances.resolve()
    args.normalization = args.normalization.resolve()
    args.out = args.out.resolve()
    args.devices = tuple(value.strip() for value in args.devices.split(",") if value.strip())
    if not 1 <= args.workers <= 8 or not args.devices:
        parser.error("workers must be 1-8 and at least one CUDA device is required")
    if args.development_max_instances is None and args.evaluation_budget != 15_000:
        parser.error("formal comparison requires the frozen 15,000-evaluation budget")
    if (args.development_max_instances is not None
            and not 1 <= args.development_max_instances <= 120):
        parser.error("development-max-instances must be between 1 and 120")

    catalog = _read(args.instances / "catalog.json")
    normalization = _read(args.normalization)
    if catalog.get("catalog_digest") != args.catalog_digest:
        parser.error("catalog digest does not match the frozen command")
    if normalization.get("normalization_digest") != args.normalization_digest:
        parser.error("normalization digest does not match the frozen command")
    records = [row for row in catalog["instances"] if row["split"] == "test"]
    counts = Counter(row["configuration"] for row in records)
    if (len(records) != 120
            or counts != Counter({configuration: 10 for configuration in TEST_CONFIGS})):
        parser.error(f"expected 120 test manifests (10 per S0-S11), found {counts}")
    records.sort(key=lambda row: (int(row["configuration"][1:]), row["instance_id"]))
    if args.development_max_instances is not None:
        records = records[:args.development_max_instances]
    selected_counts = Counter(row["configuration"] for row in records)

    try:
        checkpoints = {
            "transformer": _checkpoint_info(
                args.transformer_checkpoint, "transformer", args.replication,
                args.catalog_digest, args.normalization_digest),
            "mlp": _checkpoint_info(
                args.mlp_checkpoint, "mlp", args.replication,
                args.catalog_digest, args.normalization_digest),
        }
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))

    batch_config = {
        "schema_version": 1,
        "stage": ("paired four-method CCP100 formal test with OOS-5000"
                  if args.development_max_instances is None
                  else "development smoke of paired formal batch"),
        "formal": args.development_max_instances is None,
        "replication": args.replication,
        "catalog_digest": args.catalog_digest,
        "normalization_digest": args.normalization_digest,
        "normalization_sha256": _sha256(args.normalization),
        "source_data_sha256": _sha256(args.data),
        "test_instance_count": len(records),
        "configuration_counts": dict(sorted(selected_counts.items())),
        "methods": list(METHODS),
        "evaluation_budget": args.evaluation_budget,
        "population": args.pop,
        "generations": args.gens,
        "algorithm_seed_base": args.algorithm_seed_base,
        "policy_seed_base": args.policy_seed_base,
        "seed_formula": "base + (replication-1)*1000000 + sorted_test_index",
        "checkpoints": checkpoints,
        "workers": args.workers,
        "devices": list(args.devices),
        "threads_per_worker": 1,
        "oos_size": 5000,
        "oos_seed_source": "frozen test manifest",
        "development_max_instances": args.development_max_instances,
        "git_commit": _git_commit(),
        "implementation_sha256": {
            path.name: _sha256(path) for path in (
                Path(__file__), CONTROL, POLICY, OOS, HV,
                ROOT / "transformer_ppo/episode.py",
                ROOT / "transformer_ppo/ppo.py",
            )
        },
    }
    args.out.mkdir(parents=True, exist_ok=True)
    config_path = args.out / "batch_configuration.json"
    if config_path.exists():
        if _read(config_path) != batch_config:
            parser.error("existing batch configuration differs; use a new output directory")
    else:
        _atomic_json(config_path, batch_config)

    tasks = list(enumerate(records))
    total = len(tasks)
    already_complete = sum(
        _completed_instance(index, record, args, checkpoints)
        for index, record in tasks)
    print(
        f"[FORMAL] replication={args.replication} total={total} "
        f"already_complete={already_complete} workers={args.workers} "
        f"devices={args.devices} budget={args.evaluation_budget}", flush=True)
    started = time.perf_counter()
    completed = already_complete
    durations = []
    pending = [task for task in tasks if not _completed_instance(
        task[0], task[1], args, checkpoints)]
    executor = ThreadPoolExecutor(max_workers=args.workers)
    futures = {}
    try:
        futures = {
            executor.submit(_run_instance, task, args, checkpoints): task
            for task in pending
        }
        for future in as_completed(futures):
            instance_id, duration = future.result()
            completed += 1
            durations.append(duration)
            new_count = max(1, completed - already_complete)
            elapsed = time.perf_counter() - started
            eta_hours = elapsed / new_count * (total - completed) / 3600.0
            print(
                f"[FORMAL] {completed:03d}/{total} {instance_id} "
                f"seconds={duration:.1f} eta_hours={eta_hours:.2f}", flush=True)
    except BaseException:
        for future in futures:
            future.cancel()
        executor.shutdown(wait=True, cancel_futures=True)
        raise
    else:
        executor.shutdown(wait=True)

    rows = []
    for index, record in tasks:
        if not _completed_instance(index, record, args, checkpoints):
            raise RuntimeError(f"incomplete formal instance: {record['instance_id']}")
        path = args.out / "instances" / record["instance_id"] / "oos_hv.json"
        rows.extend(_read(path))
    if len(rows) != 4 * total:
        raise RuntimeError(
            f"expected {4 * total} method-instance OOS-HV rows, found {len(rows)}")
    rows_path = args.out / "oos_hv_rows.json"
    statistics_path = args.out / "formal_statistics.json"
    _atomic_json(rows_path, rows)
    if args.development_max_instances is None:
        _atomic_json(statistics_path, formal_report(rows))
    summary = {
        "completed": True,
        "completed_instance_count": total,
        "method_instance_count": 4 * total,
        "resumed_completed_count": already_complete,
        "new_completed_count": len(durations),
        "wall_seconds": time.perf_counter() - started,
        "mean_instance_seconds": (
            sum(durations) / len(durations) if durations else 0.0),
        "oos_hv_rows_sha256": _sha256(rows_path),
        "formal_statistics_sha256": (
            _sha256(statistics_path) if statistics_path.exists() else None),
        **batch_config,
    }
    _atomic_json(args.out / "BATCH_COMPLETE.json", summary)
    print(
        f"[FORMAL] replication={args.replication} complete "
        f"wall_hours={summary['wall_seconds'] / 3600:.2f}", flush=True)


if __name__ == "__main__":
    main()
