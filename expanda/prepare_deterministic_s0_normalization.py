#!/usr/bin/env python3
"""Build frozen S0 deterministic reward-HV bounds from training Base fronts."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from multi_instance_logging import write_json
from transformer_ppo.archive import ObjectiveNormalizer
from transformer_ppo.config import SCENARIOS
from transformer_ppo.episode import run_episode
from transformer_ppo.instance_catalog import read_instance


OBJECTIVES = ("cost", "emission", "makespan")


def _read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _atomic_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _manifests(instances: Path, configuration: str, max_instances):
    result = []
    for path in sorted(instances.glob(f"train-{configuration}-*.json")):
        record = _read(path)
        if (record.get("split") == "train"
                and record.get("configuration") == configuration):
            result.append(path)
    if not result:
        raise ValueError(
            f"no train-{configuration} manifests found in {instances.resolve()}")
    if max_instances is None and len(result) != 50:
        raise ValueError(
            f"formal preparation requires exactly 50 train-{configuration} "
            f"manifests, found {len(result)}")
    if max_instances is not None:
        if not 1 <= max_instances <= len(result):
            raise ValueError("--max-instances is outside the available manifest count")
        result = result[:max_instances]
    return result


def _run_complete(run: Path, instance: dict, args, algorithm_seed: int):
    marker = run / "COMPLETE.json"
    config_path = run / "configuration.json"
    front = run / "final_feasible_nondominated.json"
    if not marker.exists() or not config_path.exists() or not front.exists():
        return False
    summary, config = _read(marker), _read(config_path)
    if (summary.get("completed") is not True
            or summary.get("component") != "BASE"
            or summary.get("stop_reason") != "evaluation_budget"
            or summary.get("real_objective_evaluation_count") != args.evaluation_budget
            or config.get("evaluation_mode") != "deterministic"
            or config.get("instance_id") != instance["instance_id"]
            or config.get("instance_digest") != instance["instance_digest"]
            or config.get("seeds", {}).get("algorithm") != algorithm_seed
            or config.get("evaluation_budget") != args.evaluation_budget
            or config.get("population") != args.pop
            or config.get("generations") != args.gens):
        raise ValueError(f"completed Base preparation run has mismatched metadata: {run}")
    return True


def _scaled_front(front, batches):
    total_teu = max(1.0, sum(float(batch.quantity) for batch in batches))
    horizon = max(
        1.0,
        max(float(batch.LT) for batch in batches)
        - min(float(batch.ET) for batch in batches),
    )
    divisor = (total_teu, total_teu, horizon)
    points = []
    for row in front:
        values = row["optimisation_objectives"]
        points.append(tuple(
            float(values[name]) / divisor[index]
            for index, name in enumerate(OBJECTIVES)
        ))
    return points


def _freeze(points, lower_percentile, upper_percentile, pad_fraction):
    values = np.asarray(points, dtype=float)
    if values.ndim != 2 or values.shape[1] != 3 or not len(values):
        raise ValueError("at least one three-objective training point is required")
    if not np.all(np.isfinite(values)):
        raise ValueError("normalization source contains nonfinite objectives")
    lower = np.percentile(values, lower_percentile, axis=0)
    upper = np.percentile(values, upper_percentile, axis=0)
    span = upper - lower
    pad = pad_fraction * np.where(
        span > 0.0, span, np.maximum(np.abs(lower), 1.0))
    minimum, maximum = lower - pad, upper + pad
    if np.any(maximum <= minimum):
        raise ValueError("failed to construct positive normalization ranges")
    return minimum.tolist(), maximum.tolist()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path,
                        default=Path(__file__).parent / "data/data_expanded.xlsx")
    parser.add_argument("--instances", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--configuration", choices=tuple(SCENARIOS), default="S0")
    parser.add_argument("--algorithm-seed-base", type=int, default=70_000_000)
    parser.add_argument("--evaluation-budget", type=int, required=True)
    parser.add_argument("--pop", type=int, default=20)
    parser.add_argument("--gens", type=int, default=1_000)
    parser.add_argument("--lower-percentile", type=float, default=1.0)
    parser.add_argument("--upper-percentile", type=float, default=99.0)
    parser.add_argument("--pad-fraction", type=float, default=0.10)
    parser.add_argument("--max-instances", type=int,
                        help="Development only; omit for the formal 50-instance pass.")
    args = parser.parse_args()
    if not args.data.is_file():
        parser.error(f"network workbook does not exist: {args.data}")
    if not args.instances.is_dir():
        parser.error(f"instance directory does not exist: {args.instances}")
    if not 0.0 <= args.lower_percentile < args.upper_percentile <= 100.0:
        parser.error("percentiles must satisfy 0 <= lower < upper <= 100")
    if args.pad_fraction < 0.0:
        parser.error("--pad-fraction must be non-negative")
    try:
        manifests = _manifests(
            args.instances, args.configuration, args.max_instances)
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))

    run_config = {
        "schema_version": 1,
        "purpose": "training-only deterministic normalization preparation",
        "configuration": args.configuration,
        "instance_count": len(manifests),
        "instances": [path.stem for path in manifests],
        "source_data_sha256": _sha256(args.data),
        "algorithm_seed_base": args.algorithm_seed_base,
        "evaluation_budget": args.evaluation_budget,
        "population": args.pop,
        "generations": args.gens,
        "lower_percentile": args.lower_percentile,
        "upper_percentile": args.upper_percentile,
        "pad_fraction": args.pad_fraction,
        "test_instances_used": False,
        "validation_instances_used": False,
    }
    config_path = args.out / "preparation_configuration.json"
    if config_path.exists():
        if _read(config_path) != run_config:
            parser.error("existing preparation configuration does not match arguments")
    elif args.out.exists() and any(args.out.iterdir()):
        parser.error("non-empty output has no valid preparation configuration")
    else:
        args.out.mkdir(parents=True, exist_ok=True)
        _atomic_json(config_path, run_config)

    # Base does not consume reward normalization.  This finite placeholder is
    # supplied only because the shared episode writer reports a diagnostic HV;
    # raw final objectives below are the sole source of the frozen bounds.
    placeholder = ObjectiveNormalizer(
        (0.0, 0.0, 0.0), (1.0, 1.0, 1.0), reference=(1.1, 1.1, 1.1))
    started = time.perf_counter()
    source_files, points = [], []
    for index, manifest in enumerate(manifests):
        instance, batches = read_instance(manifest)
        run = args.out / "base_runs" / manifest.stem
        algorithm_seed = args.algorithm_seed_base + index
        if _run_complete(run, instance, args, algorithm_seed):
            print(f"[RESUME] {index + 1}/{len(manifests)} {manifest.stem}", flush=True)
        else:
            if run.exists() and any(run.iterdir()):
                parser.error(f"incomplete non-empty Base run requires inspection: {run}")
            print(f"[BASE] {index + 1}/{len(manifests)} {manifest.stem}", flush=True)
            run_episode(
                args.data, manifest, run, None, placeholder,
                algorithm_seed=algorithm_seed,
                evaluation_budget=args.evaluation_budget, pop_size=args.pop,
                generations=args.gens, evaluation_mode="deterministic",
                component="BASE")
        front_path = run / "final_feasible_nondominated.json"
        front = _read(front_path)
        if not front:
            parser.error(
                f"Base produced no feasible final nondominated solutions: {manifest.stem}")
        points.extend(_scaled_front(front, batches))
        source_files.append({
            "instance_id": instance["instance_id"],
            "instance_digest": instance["instance_digest"],
            "front": str(front_path.relative_to(args.out)),
            "front_sha256": _sha256(front_path),
            "front_size": len(front),
        })

    try:
        minimum, maximum = _freeze(
            points, args.lower_percentile, args.upper_percentile,
            args.pad_fraction)
    except ValueError as error:
        parser.error(str(error))
    normalization = {
        "schema_version": 1,
        "source": "training-only",
        "source_method": "base-nsga2-deterministic",
        "configuration": args.configuration,
        "definition": "cost/total_teu, emission/total_teu, makespan/instance_horizon",
        "minimum": minimum,
        "maximum": maximum,
        "reference": [1.1, 1.1, 1.1],
        "percentiles": [args.lower_percentile, args.upper_percentile],
        "pad_fraction": args.pad_fraction,
        "point_count": len(points),
        "instance_count": len(manifests),
        "test_instances_used": False,
        "validation_instances_used": False,
        "source_files": source_files,
        "preparation_configuration_sha256": _sha256(config_path),
    }
    normalization["normalization_digest"] = hashlib.sha256(
        json.dumps(normalization, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    normalization_path = args.out / "reward_normalization.json"
    _atomic_json(normalization_path, normalization)
    complete = {
        "completed": True,
        "normalization": normalization_path.name,
        "normalization_sha256": _sha256(normalization_path),
        "normalization_digest": normalization["normalization_digest"],
        "instance_count": len(manifests),
        "point_count": len(points),
        "elapsed_seconds": time.perf_counter() - started,
    }
    write_json(args.out / "COMPLETE.json", complete)
    print(json.dumps(complete, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
