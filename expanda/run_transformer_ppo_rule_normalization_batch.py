#!/usr/bin/env python3
"""Run the 450 frozen training-instance Rule fronts used for normalization."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from multi_instance_logging import write_json


ROOT = Path(__file__).resolve().parent
RUNNER = ROOT / "run_transformer_ppo_control_ccp100.py"


def _completed(run_dir: Path, instance_id: str, evaluation_budget: int) -> bool:
    complete_path = run_dir / "COMPLETE.json"
    config_path = run_dir / "configuration.json"
    if not complete_path.exists() or not config_path.exists():
        return False
    complete = json.loads(complete_path.read_text(encoding="utf-8"))
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if (complete.get("completed") is not True
            or complete.get("method") != "Rule-CCP100"
            or complete.get("evaluation_budget") != evaluation_budget
            or config.get("instance_id") != instance_id
            or config.get("method") != "Rule-CCP100"):
        raise ValueError(f"completed run metadata mismatch: {run_dir}")
    if not (run_dir / "final_feasible_nondominated.json").exists():
        raise ValueError(f"completed run is missing its final front: {run_dir}")
    return True


def _run_one(task, args, out: Path):
    index, record = task
    instance_id = record["instance_id"]
    run_dir = out / "runs" / instance_id
    if _completed(run_dir, instance_id, args.evaluation_budget):
        return instance_id, "skipped", 0.0
    started = time.perf_counter()
    log_path = out / "logs" / f"{instance_id}.log"
    command = [
        sys.executable, "-u", str(RUNNER),
        "--data", str(args.data),
        "--instance", str(args.instances / f"{instance_id}.json"),
        "--out", str(run_dir), "--policy", "rule",
        "--algorithm-seed", str(args.algorithm_seed_base + index),
        "--policy-seed", str(args.policy_seed_base + index),
        "--evaluation-budget", str(args.evaluation_budget),
        "--pop", str(args.pop), "--gens", str(args.gens),
    ]
    environment = os.environ.copy()
    environment.update({
        "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1",
        "PYTHONUNBUFFERED": "1",
    })
    with log_path.open("x", encoding="utf-8") as handle:
        result = subprocess.run(
            command, stdout=handle, stderr=subprocess.STDOUT,
            cwd=ROOT.parent, env=environment, check=False)
    if result.returncode != 0:
        raise RuntimeError(
            f"{instance_id} failed with exit code {result.returncode}; see {log_path}")
    if not _completed(run_dir, instance_id, args.evaluation_budget):
        raise RuntimeError(f"{instance_id} exited without a valid COMPLETE.json")
    return instance_id, "completed", time.perf_counter() - started


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path,
                        default=ROOT / "data/data_expanded.xlsx")
    parser.add_argument("--instances", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--catalog-digest", required=True)
    parser.add_argument("--evaluation-budget", type=int, required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--pop", type=int, default=100)
    parser.add_argument("--gens", type=int, default=1_000)
    parser.add_argument("--algorithm-seed-base", type=int, default=60_000_000)
    parser.add_argument("--policy-seed-base", type=int, default=70_000_000)
    args = parser.parse_args()
    args.data = args.data.resolve()
    args.instances = args.instances.resolve()
    args.out = args.out.resolve()
    if args.workers < 1 or args.workers > 16:
        parser.error("workers must be between 1 and 16")
    catalog = json.loads((args.instances / "catalog.json").read_text(encoding="utf-8"))
    if catalog.get("catalog_digest") != args.catalog_digest:
        parser.error("catalog digest does not match the frozen command")
    records = [row for row in catalog["instances"] if row["split"] == "train"]
    counts = Counter(row["configuration"] for row in records)
    if (len(records) != 450
            or counts != Counter({f"S{i}": 50 for i in range(9)})):
        parser.error(f"expected 450 training manifests (50 per S0-S8), found {counts}")
    records.sort(key=lambda row: (int(row["configuration"][1:]), row["instance_id"]))
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "runs").mkdir(exist_ok=True)
    (args.out / "logs").mkdir(exist_ok=True)
    batch_config = {
        "schema_version": 1, "stage": "training-only Rule normalization fronts",
        "catalog_digest": args.catalog_digest,
        "instance_count": len(records), "workers": args.workers,
        "evaluation_budget": args.evaluation_budget, "population": args.pop,
        "generations": args.gens, "algorithm_seed_base": args.algorithm_seed_base,
        "policy_seed_base": args.policy_seed_base,
        "thread_limits_per_worker": 1,
    }
    config_path = args.out / "batch_configuration.json"
    if config_path.exists():
        existing = json.loads(config_path.read_text(encoding="utf-8"))
        if existing != batch_config:
            parser.error("existing batch configuration differs; use a new output directory")
    else:
        write_json(config_path, batch_config)
    tasks = list(enumerate(records))
    incomplete = []
    already_complete = 0
    for index, record in tasks:
        run_dir = args.out / "runs" / record["instance_id"]
        if _completed(run_dir, record["instance_id"], args.evaluation_budget):
            already_complete += 1
        elif run_dir.exists() and any(run_dir.iterdir()):
            incomplete.append(str(run_dir))
    if incomplete:
        raise RuntimeError(
            "incomplete non-empty run directories require inspection before resume:\n"
            + "\n".join(incomplete))
    print(
        f"[RULE-BATCH] total=450 already_complete={already_complete} "
        f"workers={args.workers} budget={args.evaluation_budget}", flush=True)
    started = time.perf_counter()
    completed = already_complete
    durations = []
    pending = [task for task in tasks if not _completed(
        args.out / "runs" / task[1]["instance_id"],
        task[1]["instance_id"], args.evaluation_budget)]
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(_run_one, task, args, args.out): task for task in pending}
        for future in as_completed(futures):
            instance_id, status, duration = future.result()
            completed += 1
            if status == "completed":
                durations.append(duration)
            elapsed = time.perf_counter() - started
            rate = max(1, completed - already_complete)
            remaining = max(0.0, elapsed / rate * (450 - completed) / 3600)
            print(
                f"[RULE-BATCH] {completed:03d}/450 {instance_id} {status} "
                f"run_seconds={duration:.1f} eta_hours={remaining:.2f}",
                flush=True,
            )
    summary = {
        "completed": True, "instance_count": 450,
        "resumed_completed_count": already_complete,
        "new_completed_count": len(durations),
        "wall_seconds": time.perf_counter() - started,
        "mean_run_seconds": sum(durations) / len(durations) if durations else 0.0,
        **batch_config,
    }
    write_json(args.out / "BATCH_COMPLETE.json", summary)
    print(f"[RULE-BATCH] complete wall_seconds={summary['wall_seconds']:.1f}", flush=True)


if __name__ == "__main__":
    main()
