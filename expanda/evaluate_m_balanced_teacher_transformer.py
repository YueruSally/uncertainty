#!/usr/bin/env python3
"""Matched CCP comparison: Base vs Balanced heuristic vs M Transformer."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys


EPSILON = 1e-12
def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _valid(run, role, seed, minimum_generation):
    run = Path(run)
    if not (run / "COMPLETE.json").is_file():
        return False
    try:
        complete = _read(run / "COMPLETE.json")
        config = _read(run / "configuration.json")
        control = config["learning_control"]
        common = bool(
            complete.get("completed") is True
            and config.get("seeds", {}).get("algorithm") == seed)
        if role == "BASE":
            return common and complete.get("component") == "BASE"
        if role == "HEURISTIC":
            return bool(
                common and complete.get("component") == "M"
                and control.get("m_forced_operator") == "replace"
                and control.get("m_forced_operator_min_generation")
                == minimum_generation
                and control.get("m_forced_replace_selector")
                == "joint_balanced")
        return bool(
            common and complete.get("component") == "M"
            and control.get("m_forced_operator") == "replace"
            and control.get("m_forced_operator_min_generation")
            == minimum_generation
            and control.get("m_forced_replace_selector") == "model"
            and control.get("model_choice_probabilities", {}).get("M") == 1.0)
    except (KeyError, OSError, TypeError, ValueError):
        return False


def _quarantine_incomplete_run(run):
    """Move an incomplete run aside so an interrupted evaluation can resume."""
    run = Path(run)
    if not run.exists():
        return None
    if len(run.parents) < 3:
        raise ValueError(f"run path is too shallow to quarantine safely: {run}")
    quarantine = run.parents[2] / "interrupted_runs"
    relative = Path(run.parents[1].name) / run.parent.name / run.name
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    destination = quarantine / relative.parent / f"{relative.name}_{stamp}"
    suffix = 1
    while destination.exists():
        destination = quarantine / relative.parent / (
            f"{relative.name}_{stamp}_{suffix}")
        suffix += 1
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(run), str(destination))
    print(
        f"[M-TRANSFORMER-PILOT] quarantined incomplete run: "
        f"{run} -> {destination}",
        flush=True,
    )
    return destination


def _run(job):
    run, log, command, role, seed, minimum_generation = job
    if _valid(run, role, seed, minimum_generation):
        return str(run), "existing"
    if run.exists():
        _quarantine_incomplete_run(run)
    run.parent.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment.setdefault("OMP_NUM_THREADS", "1")
    environment.setdefault("MKL_NUM_THREADS", "1")
    with log.open("w", encoding="utf-8") as handle:
        result = subprocess.run(
            command, cwd=Path(__file__).resolve().parent,
            stdout=handle, stderr=subprocess.STDOUT,
            env=environment, check=False)
    if result.returncode != 0 or not _valid(
            run, role, seed, minimum_generation):
        raise RuntimeError(f"pilot run failed with exit {result.returncode}: {log}")
    return str(run), "complete"


def _command(args, instance, run, role, seed):
    command = [
        sys.executable, "-u", str(args.runner),
        "--instance", str(instance),
        "--normalization", str(args.normalization),
        "--out", str(run),
        "--component", "BASE" if role == "BASE" else "M",
        "--algorithm-seed", str(seed),
        "--evaluation-mode", "ccp",
        "--uncertainty-profile", "transport_border_saa_v1",
        "--scenario-count", str(args.scenario_count),
        "--ccp-confidence", str(args.ccp_confidence),
        "--evaluation-budget", str(args.evaluation_budget),
        "--pop", str(args.pop), "--gens", str(args.gens),
        "--device", args.device,
    ]
    if role == "HEURISTIC":
        command.extend([
            "--checkpoint", str(args.bootstrap_checkpoint),
            "--inference-action-mode", "argmax",
            "--model-choice-probability", "0",
            "--m-forced-operator", "replace",
            "--m-forced-operator-min-generation", str(args.minimum_generation),
            "--m-forced-operator-probability", "1",
            "--m-forced-replace-selector", "joint_balanced",
        ])
    elif role == "TRANSFORMER":
        command.extend([
            "--checkpoint", str(args.transformer_checkpoint),
            "--inference-action-mode", "argmax",
            "--inference-policy-seed", "63001",
            "--model-choice-probability", "1",
            "--m-forced-operator", "replace",
            "--m-forced-operator-min-generation", str(args.minimum_generation),
            "--m-forced-operator-probability", "1",
            "--m-forced-replace-selector", "model",
        ])
    return command


def _strategy_summary(name, rows):
    values = [row["paired_percent"] for row in rows]
    mean = statistics.fmean(values)
    standard_deviation = statistics.stdev(values) if len(values) >= 2 else 0.0
    standard_error = standard_deviation / math.sqrt(len(values))
    return {
        "strategy": name,
        "runs": len(rows),
        "mean_paired_percent": mean,
        "median_paired_percent": statistics.median(values),
        "standard_deviation_paired_percent": standard_deviation,
        "mean_95ci_low": mean - 1.96 * standard_error,
        "mean_95ci_high": mean + 1.96 * standard_error,
        "minimum_paired_percent": min(values),
        "maximum_paired_percent": max(values),
        "wins": sum(value > EPSILON for value in values),
        "ties": sum(abs(value) <= EPSILON for value in values),
        "losses": sum(value < -EPSILON for value in values),
        "runs_above_5_percent": sum(value >= 5.0 for value in values),
        "runs_above_10_percent": sum(value >= 10.0 for value in values),
        "mean_optimisation_seconds": statistics.fmean(
            row["optimisation_seconds"] for row in rows),
        "mean_model_control_events": statistics.fmean(
            row["model_control_events"] for row in rows),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instances", type=Path, required=True)
    parser.add_argument("--normalization", type=Path, required=True)
    parser.add_argument("--bootstrap-checkpoint", type=Path, required=True)
    parser.add_argument("--transformer-checkpoint", type=Path, required=True)
    parser.add_argument("--training-report", type=Path, required=True)
    parser.add_argument(
        "--heuristic-result-name", default="TEACHER_BALANCED")
    parser.add_argument(
        "--transformer-result-name", default="TRANSFORMER_BC")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--instance-ids", nargs="+", required=True)
    parser.add_argument("--algorithm-seeds", nargs="+", type=int, required=True)
    parser.add_argument("--minimum-generation", type=int, default=10)
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
    strategy_specs = (
        ("BASE", "BASE"),
        (args.heuristic_result_name, "HEURISTIC"),
        (args.transformer_result_name, "TRANSFORMER"),
    )
    if len({name for name, _ in strategy_specs}) != len(strategy_specs):
        parser.error("three result strategy names must be unique")
    for path, name in (
            (args.normalization, "normalization"),
            (args.bootstrap_checkpoint, "bootstrap checkpoint"),
            (args.transformer_checkpoint, "Transformer checkpoint"),
            (args.training_report, "training report"),
            (args.runner, "runner")):
        if not path.is_file():
            parser.error(f"missing {name}: {path}")
    args.out.mkdir(parents=True, exist_ok=True)
    jobs = []
    normalized_instances = {}
    for requested in args.instance_ids:
        instance_path = args.instances / f"{requested}.json"
        if not instance_path.is_file():
            parser.error(f"missing instance: {instance_path}")
        instance = requested.replace("train-", "").replace("-", "_")
        normalized_instances[instance] = instance_path
        for seed in args.algorithm_seeds:
            for strategy, role in strategy_specs:
                run = args.out / instance / f"seed_{seed}" / strategy
                log = args.out / "logs" / f"{instance}_seed_{seed}_{strategy}.log"
                log.parent.mkdir(parents=True, exist_ok=True)
                jobs.append((
                    run, log,
                    _command(args, instance_path, run, role, seed),
                    role, seed, args.minimum_generation))
    statuses = Counter()
    print(f"[M-TRANSFORMER-PILOT] jobs={len(jobs)} workers={args.max_workers}",
          flush=True)
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {executor.submit(_run, job): job for job in jobs}
        for index, future in enumerate(as_completed(futures), 1):
            run, status = future.result()
            statuses[status] += 1
            print(f"[M-TRANSFORMER-PILOT] {index}/{len(jobs)} "
                  f"{status}: {run}", flush=True)

    rows = []
    scenario_failures = []
    for instance in normalized_instances:
        for seed in args.algorithm_seeds:
            base_run = args.out / instance / f"seed_{seed}" / "BASE"
            base = _read(base_run / "COMPLETE.json")
            base_config = _read(base_run / "configuration.json")
            base_hv = float(base["final_hypervolume"])
            if base_hv <= 0.0:
                raise ValueError(f"nonpositive Base HV: {base_run}")
            for strategy, _ in strategy_specs:
                run = args.out / instance / f"seed_{seed}" / strategy
                complete = _read(run / "COMPLETE.json")
                config = _read(run / "configuration.json")
                if config["scenario_digest"] != base_config["scenario_digest"]:
                    scenario_failures.append(f"{instance}:seed_{seed}:{strategy}")
                hv = float(complete["final_hypervolume"])
                rows.append({
                    "instance": instance,
                    "algorithm_seed": seed,
                    "strategy": strategy,
                    "base_hv": base_hv,
                    "final_hv": hv,
                    "paired_percent": 100.0 * (hv - base_hv) / base_hv,
                    "final_front_size": int(complete["final_front_size"]),
                    "model_control_events": int(
                        complete.get("model_control_events", 0)),
                    "base_control_events": int(
                        complete.get("base_control_events", 0)),
                    "inference_seconds": float(
                        complete.get("inference_seconds", 0.0)),
                    "optimisation_seconds": float(
                        complete["optimisation_seconds"]),
                })
    strategies = [
        _strategy_summary(
            strategy,
            [row for row in rows if row["strategy"] == strategy])
        for strategy, _ in strategy_specs
    ]
    by_name = {row["strategy"]: row for row in strategies}
    training = _read(args.training_report)
    transformer = by_name[args.transformer_result_name]
    teacher_imitation = None
    ppo_training = None
    if "training" in training and "validation" in training:
        teacher_imitation = {
            "training_top_1": training["training"]["top_1_accuracy"],
            "validation_top_1": training["validation"]["top_1_accuracy"],
            "selected_epoch": training["selected_epoch"],
            "training_states": training["training"]["states"],
            "validation_states": training["validation"]["states"],
            "soft_label_temperature": training.get("soft_label_temperature"),
        }
    elif training.get("completed") is True and training.get("component") == "M":
        episodes = training.get("episodes", [])
        ppo_training = {
            "episode_count": len(episodes),
            "ppo_updates": sum(int(row.get("ppo_updates", 0)) for row in episodes),
            "reward_total": sum(float(row.get("reward_total", 0.0)) for row in episodes),
            "positive_reward_actions": sum(
                int(row.get("positive_reward_actions", 0)) for row in episodes),
            "real_objective_evaluations": int(
                training.get("total_actual_evaluations", 0)),
            "counterfactual_evaluations": int(
                training.get("total_counterfactual_evaluations", 0)),
            "elapsed_seconds": float(training.get("elapsed_seconds", 0.0)),
        }
    else:
        parser.error("training report is neither a BC nor an M PPO report")
    result = {
        "completed": True,
        "definition": (
            "matched frozen-CCP comparison of exact Base, deterministic "
            "Balanced heuristic M, and path-v4 Transformer M"),
        "instance_count": len(normalized_instances),
        "algorithm_seed_count": len(args.algorithm_seeds),
        "run_count": len(rows),
        "status_counts": dict(statuses),
        "scenario_exact": not scenario_failures,
        "scenario_failures": scenario_failures,
        "teacher_imitation": teacher_imitation,
        "ppo_training": ppo_training,
        "strategies": strategies,
        "pilot_thresholds": {
            "minimum_validation_top_1": (
                0.50 if teacher_imitation is not None else None),
            "minimum_ppo_updates": (
                1 if ppo_training is not None else None),
            "minimum_transformer_mean_paired_percent": 0.0,
            "minimum_transformer_wins": math.ceil(
                0.5 * len(normalized_instances) * len(args.algorithm_seeds)),
        },
        "runs": rows,
    }
    thresholds = result["pilot_thresholds"]
    result["pilot_pass"] = bool(
        result["scenario_exact"]
        and (teacher_imitation is None
             or teacher_imitation["validation_top_1"]
             >= thresholds["minimum_validation_top_1"])
        and (ppo_training is None
             or ppo_training["ppo_updates"]
             >= thresholds["minimum_ppo_updates"])
        and transformer["mean_paired_percent"]
        > thresholds["minimum_transformer_mean_paired_percent"]
        and transformer["wins"] >= thresholds["minimum_transformer_wins"])
    (args.out / "COMPLETE.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n",
        encoding="utf-8")
    quick = {key: result[key] for key in (
        "completed", "definition", "instance_count", "algorithm_seed_count",
        "run_count", "scenario_exact", "teacher_imitation", "ppo_training",
        "strategies", "pilot_thresholds", "pilot_pass")}
    (args.out / "QUICK_RESULT.json").write_text(
        json.dumps(quick, indent=2, allow_nan=False) + "\n",
        encoding="utf-8")
    print(json.dumps(quick, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
