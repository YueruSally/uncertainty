#!/usr/bin/env python3
"""Generate unseen instances and freeze the formal CCP Base/I/M plan."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from im_ccp_formal_plan import (
    ALGORITHM_SEEDS,
    CCP_CONFIDENCE,
    EVALUATION_BUDGET,
    FINAL_INSTANCE_COUNT,
    FINAL_INSTANCE_CCP_SEED_BASE,
    FINAL_INSTANCE_MASTER_SEED,
    FINAL_INSTANCE_OOS_SEED_BASE,
    GENERATIONS,
    I_PROFILE,
    METHODS,
    M_MINIMUM_GENERATION,
    M_OPERATOR,
    M_PROBABILITY,
    M_SELECTOR,
    OOS_SCENARIO_COUNT,
    POPULATION,
    SHARD_COUNT,
    TASKS_PER_SHARD,
    TRAINING_SCENARIO_COUNT,
    UNCERTAINTY_PROFILE,
    build_schedule,
    canonical_digest,
    expected_rule_i_attempts,
)


ROOT = Path(__file__).resolve().parent
GENERATOR = ROOT / "generate_s0_final_test_instances.py"
IMPLEMENTATION_FILES = (
    "baseline_uncertainty.py",
    "evaluate_m_directed_balanced_oos.py",
    "generate_s0_final_test_instances.py",
    "im_ccp_formal_plan.py",
    "multi_instance_logging.py",
    "run_ccp_candidate_pool.py",
    "run_ev_ccp_oos_pilot.py",
    "run_formal_ev_vs_ccp_s30_30runs.py",
    "run_learning_components_deterministic.py",
    "transformer_ppo/archive.py",
    "transformer_ppo/comparison_controllers.py",
    "transformer_ppo/controller.py",
    "transformer_ppo/episode.py",
    "transformer_ppo/initialization_rule.py",
    "transformer_ppo/instance_catalog.py",
    "transformer_ppo/ppo.py",
    "run_im_ccp_formal_shard_v1.py",
    "analyze_im_ccp_formal_oos_v1.py",
)


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n",
        encoding="utf-8")
    temporary.replace(path)


def _verify_catalog(instances, data):
    catalog_path = instances / "catalog.json"
    if not catalog_path.is_file():
        raise ValueError(f"missing final-instance catalog: {catalog_path}")
    catalog = _read(catalog_path)
    payload = dict(catalog)
    digest = payload.pop("catalog_digest", None)
    if not digest or canonical_digest(payload) != digest:
        raise ValueError("final-instance catalog digest mismatch")
    records = catalog.get("instances", [])
    if (catalog.get("instance_count") != FINAL_INSTANCE_COUNT
            or len(records) != FINAL_INSTANCE_COUNT
            or catalog.get("master_seed") != FINAL_INSTANCE_MASTER_SEED
            or catalog.get("ccp_seed_base") != FINAL_INSTANCE_CCP_SEED_BASE
            or catalog.get("oos_seed_base") != FINAL_INSTANCE_OOS_SEED_BASE
            or catalog.get("source_data_sha256") != _sha256(data)):
        raise ValueError("final-instance catalog is not the frozen formal set")
    for index, record in enumerate(records):
        path = instances / f"{record['instance_id']}.json"
        value = _read(path)
        if (value.get("instance_digest") != record.get("instance_digest")
                or value.get("test_role") != "final"
                or value.get("ccp100_seed")
                != FINAL_INSTANCE_CCP_SEED_BASE + index
                or value.get("oos5000_seed")
                != FINAL_INSTANCE_OOS_SEED_BASE + index):
            raise ValueError(f"invalid frozen final instance: {path}")
    return catalog


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data", type=Path,
        default=ROOT / "data/data_expanded.xlsx")
    parser.add_argument("--source-instances", type=Path, required=True)
    parser.add_argument("--normalization", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    args.data = args.data.resolve()
    args.source_instances = args.source_instances.resolve()
    args.normalization = args.normalization.resolve()
    args.checkpoint = args.checkpoint.resolve()
    args.out = args.out.resolve()
    for path, label in (
            (args.data, "network workbook"),
            (args.source_instances / "catalog.json", "source catalog"),
            (args.normalization, "training-only normalization"),
            (args.checkpoint, "M infrastructure checkpoint")):
        if not path.is_file():
            parser.error(f"missing {label}: {path}")
    normalization = _read(args.normalization)
    if normalization.get("source") != "training-only":
        parser.error("normalization must declare source=training-only")

    instances = args.out / "final_test_instances"
    plan_path = args.out / "FORMAL_PLAN.json"
    if plan_path.is_file():
        existing = _read(plan_path)
        payload = dict(existing)
        digest = payload.pop("plan_digest", None)
        if not digest or canonical_digest(payload) != digest:
            parser.error("existing formal plan has an invalid digest")
        if (existing.get("population") != POPULATION
                or existing.get("generations") != GENERATIONS
                or existing.get("evaluation_budget") != EVALUATION_BUDGET
                or existing.get("algorithm_seeds") != list(ALGORITHM_SEEDS)
                or existing.get("final_test_instances", {}).get(
                    "instance_count") != FINAL_INSTANCE_COUNT
                or existing.get("m_rule", {}).get("minimum_generation")
                != M_MINIMUM_GENERATION):
            parser.error("existing formal plan uses a different protocol")
        for relative, expected in existing.get(
                "implementation_sha256", {}).items():
            path = ROOT / relative
            if not path.is_file() or _sha256(path) != expected:
                parser.error(
                    "existing formal plan was frozen with different code; "
                    "use a new FORMAL_ROOT")
        print(json.dumps({
            "completed": True,
            "status": "existing_frozen_plan",
            "plan": str(plan_path),
            "plan_digest": digest,
        }, indent=2))
        return
    if instances.exists() and any(instances.iterdir()):
        try:
            catalog = _verify_catalog(instances, args.data)
        except (KeyError, OSError, TypeError, ValueError,
                json.JSONDecodeError) as error:
            parser.error(str(error))
    else:
        instances.parent.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable, "-u", str(GENERATOR),
            "--data", str(args.data),
            "--source-instances", str(args.source_instances),
            "--out", str(instances),
            "--count", str(FINAL_INSTANCE_COUNT),
            "--master-seed", str(FINAL_INSTANCE_MASTER_SEED),
            "--ccp-seed-base", str(FINAL_INSTANCE_CCP_SEED_BASE),
            "--oos-seed-base", str(FINAL_INSTANCE_OOS_SEED_BASE),
        ]
        subprocess.run(command, cwd=ROOT, check=True)
        catalog = _verify_catalog(instances, args.data)

    records = catalog["instances"]
    instance_ids = [row["instance_id"] for row in records]
    schedule = build_schedule(instance_ids)
    implementation = {}
    for relative in IMPLEMENTATION_FILES:
        path = ROOT / relative
        if not path.is_file():
            parser.error(f"formal implementation file is missing: {path}")
        implementation[relative] = _sha256(path)
    plan = {
        "schema_version": 1,
        "formal": True,
        "status": "frozen_before_execution",
        "claim_boundary": (
            "deterministic directed I and M teachers; not learned PPO"),
        "methods": list(METHODS),
        "population": POPULATION,
        "generations": GENERATIONS,
        "evaluation_budget": EVALUATION_BUDGET,
        "evaluation_budget_definition": "population * (generations + 1)",
        "evaluation_mode": "ccp",
        "uncertainty_profile": UNCERTAINTY_PROFILE,
        "training_scenario_count": TRAINING_SCENARIO_COUNT,
        "ccp_confidence": CCP_CONFIDENCE,
        "oos_scenario_count": OOS_SCENARIO_COUNT,
        "oos_hv_reference_rule": (
            "componentwise maximum of the training-only normalization "
            "reference and the pooled Base+I+M OOS nadir plus "
            "max(0.05, 10% pooled axis range); frozen before results"),
        "algorithm_seeds": list(ALGORITHM_SEEDS),
        "independent_unit": "previously unseen final-test instance",
        "inference_replication_rule": (
            f"average {len(ALGORITHM_SEEDS)} algorithm seeds within each of "
            f"{FINAL_INSTANCE_COUNT} instances"),
        "i_rule": {
            "component": "RULE-I",
            "profile": I_PROFILE,
            "expected_controlled_initializations": expected_rule_i_attempts(),
            "extra_objective_evaluations": 0,
        },
        "m_rule": {
            "component": "M",
            "model_choice_probability": 0.0,
            "operator": M_OPERATOR,
            "minimum_generation": M_MINIMUM_GENERATION,
            "probability": M_PROBABILITY,
            "selector": M_SELECTOR,
            "extra_objective_evaluations_per_action": 0,
        },
        "network": {
            "path": str(args.data),
            "sha256": _sha256(args.data),
        },
        "border_event_data": {
            "path": str(ROOT / "data/border_crossing_events.csv"),
            "sha256": _sha256(ROOT / "data/border_crossing_events.csv"),
        },
        "normalization": {
            "path": str(args.normalization),
            "sha256": _sha256(args.normalization),
            "source": normalization["source"],
        },
        "m_infrastructure_checkpoint": {
            "path": str(args.checkpoint),
            "sha256": _sha256(args.checkpoint),
            "scientific_role": (
                "loads the generic M controller only; model choice probability "
                "is fixed to zero"),
        },
        "final_test_instances": {
            "instance_count": FINAL_INSTANCE_COUNT,
            "directory": str(instances),
            "catalog_path": str(instances / "catalog.json"),
            "catalog_sha256": _sha256(instances / "catalog.json"),
            "catalog_digest": catalog["catalog_digest"],
            "instances": records,
        },
        "implementation_sha256": implementation,
        "execution": {
            "shard_count": SHARD_COUNT,
            "tasks_per_shard": TASKS_PER_SHARD,
            "recommended_workers_per_shard": 2,
        },
        "schedule": schedule,
        "schedule_digest": canonical_digest(schedule),
    }
    plan["plan_digest"] = canonical_digest(plan)
    _atomic_json(plan_path, plan)
    print(json.dumps({
        "completed": True,
        "status": "new_frozen_plan",
        "plan": str(plan_path),
        "plan_digest": plan["plan_digest"],
        "instance_count": len(records),
        "task_count": len(schedule),
        "tasks_per_shard": TASKS_PER_SHARD,
    }, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
