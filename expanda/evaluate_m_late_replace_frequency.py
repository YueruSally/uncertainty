#!/usr/bin/env python3
"""Compare late-stage replace schedules with matched CCP Base runs."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys


EPSILON = 1e-12
STRATEGIES = {
    "BASE": (None, False),
    "REPLACE_LATE_P050": (0.50, False),
    "REPLACE_LATE_P100": (1.00, False),
    "REPLACE_LATE_POSITIVE_PROBE": (1.00, True),
}


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _command(args, instance, out, seed, probability, positive_probe):
    command = [
        sys.executable, "-u", str(args.runner),
        "--instance", str(instance),
        "--normalization", str(args.normalization),
        "--out", str(out),
        "--component", "M",
        "--checkpoint", str(args.checkpoint),
        "--algorithm-seed", str(seed),
        "--inference-action-mode", "argmax",
        "--inference-policy-seed", "45000",
        "--model-choice-probability", "0",
        "--evaluation-mode", "ccp",
        "--uncertainty-profile", "transport_border_saa_v1",
        "--scenario-count", str(args.scenario_count),
        "--ccp-confidence", str(args.ccp_confidence),
        "--evaluation-budget", str(args.evaluation_budget),
        "--pop", str(args.pop),
        "--gens", str(args.gens),
        "--device", args.device,
    ]
    if probability is not None:
        command.extend([
            "--m-forced-operator", "replace",
            "--m-forced-operator-min-generation",
            str(args.minimum_generation),
            "--m-forced-operator-probability", str(probability),
        ])
        if positive_probe:
            command.append("--m-forced-operator-positive-probe")
    return command


def _valid(run, seed, probability, positive_probe, minimum_generation):
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
            and config.get("seeds", {}).get("algorithm") == seed
        )
        if probability is None:
            return bool(
                valid
                and control.get("m_forced_operator") is None
                and control.get("model_choice_probabilities", {}).get("M") == 0)
        return bool(
            valid
            and control.get("m_forced_operator") == "replace"
            and control.get("m_forced_operator_min_generation")
            == minimum_generation
            and abs(float(control.get("m_forced_operator_probability"))
                    - probability) <= EPSILON
            and bool(control.get("m_forced_operator_positive_probe"))
            == positive_probe
            and (not positive_probe
                 or complete.get("forced_operator_probe_completed") is True))
    except (KeyError, OSError, TypeError, ValueError):
        return False


def _run(job):
    (run, log, command, seed, probability, positive_probe,
     minimum_generation) = job
    if _valid(run, seed, probability, positive_probe, minimum_generation):
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
            run, seed, probability, positive_probe, minimum_generation):
        raise RuntimeError(f"run failed with exit {result.returncode}: {log}")
    return run, "complete"


def _strategy_run(args, instance, seed, strategy):
    if (args.reuse_root is not None
            and strategy != "REPLACE_LATE_POSITIVE_PROBE"):
        return args.reuse_root / instance / f"seed_{seed}" / strategy
    return args.out / instance / f"seed_{seed}" / strategy


def _summary(args, status_counts, instance_names):
    runs = []
    scenario_failures = []
    for instance in instance_names:
        for seed in args.algorithm_seeds:
            base_run = _strategy_run(args, instance, seed, "BASE")
            base_complete = _read(base_run / "COMPLETE.json")
            base_config = _read(base_run / "configuration.json")
            base_hv = float(base_complete["final_hypervolume"])
            for strategy in STRATEGIES:
                run = _strategy_run(args, instance, seed, strategy)
                complete = _read(run / "COMPLETE.json")
                config = _read(run / "configuration.json")
                if config["scenario_digest"] != base_config["scenario_digest"]:
                    scenario_failures.append(
                        f"{instance}:seed_{seed}:{strategy}")
                hv = float(complete["final_hypervolume"])
                percent = 100.0 * (hv - base_hv) / base_hv
                runs.append({
                    "instance": instance,
                    "algorithm_seed": seed,
                    "strategy": strategy,
                    "base_hv": base_hv,
                    "final_hv": hv,
                    "paired_percent": percent,
                    "model_control_events": int(
                        complete.get("model_control_events", 0)),
                    "base_control_events": int(
                        complete.get("base_control_events", 0)),
                    "final_front_size": int(complete["final_front_size"]),
                    "operator_probe_positive": complete.get(
                        "forced_operator_probe_positive"),
                    "operator_probe_delta_hv": complete.get(
                        "forced_operator_probe_delta_hv"),
                })
    strategies = []
    for strategy in STRATEGIES:
        rows = [row for row in runs if row["strategy"] == strategy]
        values = [row["paired_percent"] for row in rows]
        strategies.append({
            "strategy": strategy,
            "runs": len(rows),
            "mean_paired_percent": statistics.fmean(values),
            "median_paired_percent": statistics.median(values),
            "minimum_paired_percent": min(values),
            "maximum_paired_percent": max(values),
            "wins": sum(value > EPSILON for value in values),
            "ties": sum(abs(value) <= EPSILON for value in values),
            "losses": sum(value < -EPSILON for value in values),
            "runs_above_5_percent": sum(value >= 5.0 for value in values),
            "runs_above_10_percent": sum(value >= 10.0 for value in values),
            "mean_forced_events": statistics.fmean(
                row["model_control_events"] for row in rows),
            "positive_probes": sum(
                row["operator_probe_positive"] is True for row in rows),
        })
    alternatives = [row for row in strategies if row["strategy"] != "BASE"]
    best = max(alternatives, key=lambda row: row["mean_paired_percent"])
    result = {
        "completed": True,
        "definition": (
            "matched 400-evaluation frozen-CCP comparison of Base against "
            "50%, 100%, and immediate-positive-probe replace schedules from "
            "generation 15"),
        "instance_count": len(instance_names),
        "algorithm_seed_count": len(args.algorithm_seeds),
        "run_count": len(runs),
        "status_counts": status_counts,
        "reuse_root": (
            None if args.reuse_root is None else str(args.reuse_root)),
        "scenario_exact": not scenario_failures,
        "scenario_failures": scenario_failures,
        "strategies": strategies,
        "best_nonbase_strategy": best,
        "screening_thresholds": {
            "minimum_mean_paired_percent": 2.0,
            "minimum_wins": 6,
            "minimum_paired_percent": -5.0,
            "minimum_runs_above_5_percent": 1,
        },
        "runs": runs,
    }
    thresholds = result["screening_thresholds"]
    result["screening_pass"] = bool(
        result["scenario_exact"]
        and best["mean_paired_percent"] >=
        thresholds["minimum_mean_paired_percent"]
        and best["wins"] >= thresholds["minimum_wins"]
        and best["minimum_paired_percent"] >=
        thresholds["minimum_paired_percent"]
        and best["runs_above_5_percent"] >=
        thresholds["minimum_runs_above_5_percent"])
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instances", type=Path, required=True)
    parser.add_argument("--normalization", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--reuse-root", type=Path,
        help=("Reuse completed BASE/P050/P100 runs from this compatible root; "
              "only the positive-probe strategy is written to --out."))
    parser.add_argument("--instance-ids", nargs="+", required=True)
    parser.add_argument("--algorithm-seeds", nargs="+", type=int,
                        required=True)
    parser.add_argument("--minimum-generation", type=int, default=15)
    parser.add_argument("--max-workers", type=int, default=9)
    parser.add_argument("--evaluation-budget", type=int, default=400)
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
    if len(args.instance_ids) != 3 or len(args.algorithm_seeds) != 3:
        parser.error("frequency screen requires exactly 3 instances and 3 seeds")
    for path, name in (
            (args.normalization, "normalization"),
            (args.checkpoint, "checkpoint"), (args.runner, "runner")):
        if not path.is_file():
            parser.error(f"missing {name}: {path}")
    args.out.mkdir(parents=True, exist_ok=True)
    instance_paths = {}
    for instance_id in args.instance_ids:
        path = args.instances / f"{instance_id}.json"
        if not path.is_file():
            parser.error(f"missing instance: {path}")
        instance_paths[instance_id.replace("train-", "").replace("-", "_")] = path

    jobs = []
    for instance, instance_path in sorted(instance_paths.items()):
        for seed in args.algorithm_seeds:
            for strategy, specification in STRATEGIES.items():
                probability, positive_probe = specification
                run = _strategy_run(args, instance, seed, strategy)
                jobs.append((
                    run, run.with_suffix(".nohup.log"),
                    _command(
                        args, instance_path, run, seed, probability,
                        positive_probe),
                    seed, probability, positive_probe,
                    args.minimum_generation))
    counts = Counter()
    print(f"[FREQUENCY] jobs={len(jobs)} workers={args.max_workers}", flush=True)
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = [executor.submit(_run, job) for job in jobs]
        for index, future in enumerate(as_completed(futures), 1):
            run, status = future.result()
            counts[status] += 1
            print(
                f"[FREQUENCY] {index}/{len(jobs)} {status}: {run}",
                flush=True)
    result = _summary(args, dict(counts), sorted(instance_paths))
    (args.out / "COMPLETE.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n",
        encoding="utf-8")
    quick = {key: result[key] for key in (
        "completed", "definition", "instance_count", "algorithm_seed_count",
        "run_count", "status_counts", "scenario_exact", "strategies",
        "best_nonbase_strategy", "screening_thresholds", "screening_pass")}
    (args.out / "QUICK_RESULT.json").write_text(
        json.dumps(quick, indent=2, allow_nan=False) + "\n",
        encoding="utf-8")
    print(json.dumps(quick, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
