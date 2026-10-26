#!/usr/bin/env python3
"""Validate every frozen input and write the S0 deterministic final-test plan."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from s0_deterministic_test_plan import (
    ALGORITHM_REPEATS,
    ALGORITHM_SEED_BASE,
    ALGORITHM_SEED_STRIDE,
    AUXILIARY_METHODS,
    LEARNED_METHODS,
    POLICY_SEEDS,
    PRIMARY_METHODS,
    build_schedule,
    canonical_digest,
)


ROOT = Path(__file__).resolve().parent


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


def _git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
            stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _portable(path: Path):
    path = path.resolve()
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _verified_digest(value, field, label):
    expected = value.get(field)
    payload = dict(value)
    payload.pop(field, None)
    if not expected or canonical_digest(payload) != expected:
        raise ValueError(f"{label} digest mismatch")
    return expected


def _parse_checkpoints(values, component):
    result = {}
    for value in values:
        try:
            seed_text, path_text = value.split("=", 1)
            seed = int(seed_text)
        except (TypeError, ValueError):
            raise ValueError(
                f"{component} checkpoint must use POLICY_SEED=PATH")
        if seed in result:
            raise ValueError(f"duplicate {component} checkpoint seed: {seed}")
        result[seed] = Path(path_text).resolve()
    if set(result) != set(POLICY_SEEDS):
        raise ValueError(
            f"{component} checkpoints must contain seeds {list(POLICY_SEEDS)}")
    return result


def _validate_final_instances(directory: Path, data_sha: str):
    catalog_path = directory / "catalog.json"
    catalog = _read(catalog_path)
    digest = _verified_digest(catalog, "catalog_digest", "final-test catalog")
    if (catalog.get("configuration") != "S0"
            or catalog.get("split") != "test"
            or catalog.get("test_role") != "final"
            or catalog.get("instance_count") != 10
            or catalog.get("source_data_sha256") != data_sha
            or catalog.get("training_instances_used") is not False
            or catalog.get("development_test_instances_used") is not False
            or catalog.get("optimisation_outcomes_used") is not False
            or catalog.get("model_selection_outcomes_used") is not False):
        raise ValueError("final-test catalog provenance is invalid")
    records = catalog.get("instances", [])
    if len(records) != 10:
        raise ValueError("final-test catalog must list exactly ten instances")
    validated = []
    signatures = set()
    for record in records:
        instance_id = record.get("instance_id")
        path = directory / f"{instance_id}.json"
        manifest = _read(path)
        instance_digest = _verified_digest(
            manifest, "instance_digest", f"final-test manifest {instance_id}")
        signature = manifest.get("batch_signature")
        if (manifest.get("instance_id") != instance_id
                or manifest.get("split") != "test"
                or manifest.get("test_role") != "final"
                or manifest.get("configuration") != "S0"
                or manifest.get("source_data_sha256") != data_sha
                or instance_digest != record.get("instance_digest")
                or signature != record.get("batch_signature")):
            raise ValueError(f"final-test catalog/manifest mismatch: {path}")
        if signature in signatures:
            raise ValueError("duplicate final-test batch signature")
        signatures.add(signature)
        validated.append({
            "instance_id": instance_id,
            "instance_digest": instance_digest,
            "batch_signature": signature,
            "instance_seed": manifest["instance_seed"],
            "path_seed": manifest["path_seed"],
        })
    return catalog, digest, validated


def _validate_checkpoint(path, component, seed, training_plan,
                         normalization, normalization_path, data_sha):
    if not path.is_file():
        raise ValueError(f"missing {component}{seed} checkpoint: {path}")
    root = path.parent
    config_path = root / "training_configuration.json"
    summary_path = root / "training_summary.json"
    config = _read(config_path)
    summary = _read(summary_path)
    checkpoint_sha = _sha256(path)
    expected_path = (root / summary.get("checkpoint", "")).resolve()
    episodes = summary.get("episodes", [])
    expected_ids = training_plan["training"][
        "ordered_instance_ids_by_policy_seed"][str(seed)]
    expected_records = {
        row["instance_id"]: row["instance_digest"]
        for row in training_plan["training"]["instances"]
    }
    ordered = config.get("ordered_instances", [])
    if (expected_path != path
            or summary.get("completed") is not True
            or summary.get("component") != component
            or summary.get("checkpoint_sha256") != checkpoint_sha
            or summary.get("total_actual_evaluations") != 300_000
            or len(episodes) != 10
            or config.get("component") != component
            or config.get("architecture") != "transformer"
            or config.get("policy_seed") != seed
            or config.get("episode_count") != 10
            or config.get("evaluation_mode") != "deterministic"
            or config.get("configuration") != "S0"
            or config.get("evaluation_budget_per_episode") != 30_000
            or config.get("population") != 250
            or config.get("generations") != 250
            or config.get("normalization_digest")
            != normalization.get("normalization_digest")
            or config.get("normalization_sha256") != _sha256(normalization_path)
            or config.get("source_data_sha256") != data_sha
            or config.get("oos_used_for_training") is not False
            or [row.get("instance_id") for row in ordered] != expected_ids
            or [row.get("instance") for row in episodes] != expected_ids
            or any(row.get("instance_digest")
                   != expected_records.get(row.get("instance_id"))
                   for row in ordered)
            or any(row.get("real_objective_evaluation_count") != 30_000
                   or row.get("stop_reason") != "evaluation_budget"
                   for row in episodes)):
        raise ValueError(f"invalid frozen {component}{seed} checkpoint provenance")
    floor = training_plan["training"][
        "expected_minimum_ppo_updates_for_audit"][component]
    updates = episodes[-1].get("ppo_updates_total")
    if not isinstance(updates, int) or updates < floor:
        raise ValueError(f"{component}{seed} PPO update floor was not met")
    return {
        "path": _portable(path),
        "sha256": checkpoint_sha,
        "policy_seed": seed,
        "component": component,
        "training_episode_count": len(episodes),
        "training_evaluations": summary["total_actual_evaluations"],
        "ppo_updates_total": updates,
        "training_configuration_sha256": _sha256(config_path),
        "training_summary_sha256": _sha256(summary_path),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path,
                        default=ROOT / "data/data_expanded.xlsx")
    parser.add_argument("--training-plan", type=Path, required=True)
    parser.add_argument("--final-instances", type=Path, required=True)
    parser.add_argument("--normalization", type=Path, required=True)
    parser.add_argument("--rule-normalization", type=Path, required=True)
    parser.add_argument("--i-checkpoints", nargs=3, required=True)
    parser.add_argument("--c-checkpoints", nargs=3, required=True)
    parser.add_argument("--m-checkpoints", nargs=3, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error(f"output already exists: {args.out}")
    try:
        for label, path in (
                ("network workbook", args.data),
                ("training plan", args.training_plan),
                ("final instances", args.final_instances),
                ("normalization", args.normalization),
                ("Rule-I normalization", args.rule_normalization)):
            if not path.exists():
                raise ValueError(f"{label} is missing: {path}")
        data_sha = _sha256(args.data)
        training_plan = _read(args.training_plan)
        training_plan_digest = _verified_digest(
            training_plan, "plan_digest", "training plan")
        if (training_plan.get("configuration") != "S0"
                or training_plan.get("population") != 250
                or training_plan.get("evaluation_budget") != 30_000
                or training_plan.get("generation_cap") != 250
                or training_plan.get("policy_seeds") != list(POLICY_SEEDS)
                or training_plan.get("validation_instances_used") is not False):
            raise ValueError("training plan does not match the frozen S0 design")
        normalization = _read(args.normalization)
        if (normalization.get("normalization_digest")
                != training_plan["normalization"]["normalization_digest"]
                or _sha256(args.normalization)
                != training_plan["normalization"]["sha256"]
                or normalization.get("source") != "training-only"
                or normalization.get("test_instances_used") is not False
                or normalization.get("validation_instances_used") is not False):
            raise ValueError("objective normalization differs from training plan")
        rule = _read(args.rule_normalization)
        rule_digest = _verified_digest(
            rule, "normalization_digest", "Rule-I normalization")
        if (rule.get("source") != "training-only"
                or rule.get("configuration") != "S0"
                or rule.get("instance_count") != 50
                or rule.get("test_instances_used") is not False
                or rule.get("validation_instances_used") is not False
                or rule.get("source_data_sha256") != data_sha):
            raise ValueError("Rule-I normalization provenance is invalid")
        final_catalog, final_catalog_digest, instances = (
            _validate_final_instances(args.final_instances, data_sha))
        checkpoint_args = {
            "I": _parse_checkpoints(args.i_checkpoints, "I"),
            "C": _parse_checkpoints(args.c_checkpoints, "C"),
            "M": _parse_checkpoints(args.m_checkpoints, "M"),
        }
        checkpoints = {}
        for component, by_seed in checkpoint_args.items():
            checkpoints[component] = {
                str(seed): _validate_checkpoint(
                    path, component, seed, training_plan, normalization,
                    args.normalization, data_sha)
                for seed, path in sorted(by_seed.items())
            }
        schedule, shard_loads = build_schedule(
            [row["instance_id"] for row in instances])
    except (KeyError, OSError, TypeError, ValueError,
            json.JSONDecodeError) as error:
        parser.error(str(error))

    plan = {
        "schema_version": 1,
        "stage": "single-scene deterministic S0 final test",
        "formal": True,
        "configuration": "S0",
        "uncertainty": False,
        "population": 250,
        "evaluation_budget": 30_000,
        "generation_cap": 250,
        "primary_methods": list(PRIMARY_METHODS),
        "auxiliary_methods": list(AUXILIARY_METHODS),
        "policy_seeds": list(POLICY_SEEDS),
        "algorithm_repeats_per_instance": ALGORITHM_REPEATS,
        "algorithm_seed_base": ALGORITHM_SEED_BASE,
        "algorithm_seed_stride": ALGORITHM_SEED_STRIDE,
        "algorithm_seed_formula": (
            "base + (algorithm_repeat-1)*stride + sorted_final_instance_index"),
        "checkpoint_rule": "frozen final checkpoint for each policy seed",
        "training_plan": {
            "path": _portable(args.training_plan),
            "sha256": _sha256(args.training_plan),
            "plan_digest": training_plan_digest,
            "legacy_development_test_records_ignored": True,
        },
        "final_test_instances": {
            "directory": _portable(args.final_instances),
            "catalog_sha256": _sha256(args.final_instances / "catalog.json"),
            "catalog_digest": final_catalog_digest,
            "source_catalog_digest": final_catalog["source_catalog_digest"],
            "instances": instances,
            "training_instances_used": False,
            "development_test_instances_used": False,
            "optimisation_outcomes_used": False,
            "model_selection_outcomes_used": False,
        },
        "normalization": {
            "path": _portable(args.normalization),
            "sha256": _sha256(args.normalization),
            "normalization_digest": normalization["normalization_digest"],
            "reference": normalization["reference"],
        },
        "rule_initialization_normalization": {
            "path": _portable(args.rule_normalization),
            "sha256": _sha256(args.rule_normalization),
            "normalization_digest": rule_digest,
            "features": rule["features"],
            "weights": rule["weights"],
            "score": rule["score"],
        },
        "checkpoints": checkpoints,
        "schedule": schedule,
        "schedule_digest": canonical_digest(schedule),
        "execution": {
            "shard_count": 2,
            "estimated_runtime_weights": shard_loads,
            "sharding_role": "operational only; excluded from analysis",
        },
        "run_counts": {
            "BASE": 30,
            "RULE-I": 30,
            "I": 90,
            "C": 90,
            "M": 90,
            "ICM": 90,
            "total": 420,
        },
        "analysis": {
            "independent_unit": "final-test instance",
            "learned_method_aggregation": (
                "mean over 3 policy seeds and 3 algorithm seeds within instance"),
            "control_aggregation": (
                "mean over 3 algorithm seeds within instance"),
            "primary_family": [
                "I vs BASE", "C vs BASE", "M vs BASE", "ICM vs BASE"],
            "primary_multiplicity": "Holm correction over four comparisons",
            "secondary_predeclared_contrast": "I vs RULE-I",
            "statistics": [
                "mean HV", "wins-ties-losses", "bootstrap 95% interval",
                "exact sign-flip p-value", "Holm-adjusted p-value"],
        },
        "source_data_sha256": data_sha,
        "git_commit": _git_commit(),
        "implementation_sha256": {
            Path(__file__).name: _sha256(Path(__file__)),
            "s0_deterministic_test_plan.py": _sha256(
                ROOT / "s0_deterministic_test_plan.py"),
            "run_learning_components_deterministic.py": _sha256(
                ROOT / "run_learning_components_deterministic.py"),
        },
    }
    plan["plan_digest"] = canonical_digest(plan)
    _atomic_json(args.out, plan)
    print(json.dumps({
        "completed": True,
        "out": str(args.out.resolve()),
        "plan_digest": plan["plan_digest"],
        "final_test_catalog_digest": final_catalog_digest,
        "task_count": len(schedule),
        "run_counts": plan["run_counts"],
        "estimated_runtime_weights_by_shard": shard_loads,
    }, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

