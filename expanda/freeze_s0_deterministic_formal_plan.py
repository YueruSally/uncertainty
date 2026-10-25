#!/usr/bin/env python3
"""Freeze the single-scene deterministic training and test configuration."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from s0_deterministic_formal_plan import (
    ALGORITHM_SEED_BASE,
    EVALUATION_BUDGET,
    EXPECTED_MINIMUM_PPO_UPDATES,
    GENERATION_CAP,
    POLICY_SEEDS,
    POPULATION,
    TRAINING_INSTANCE_COUNT,
    canonical_digest,
    evenly_spaced_ranks,
    ordered_ids_by_seed,
)


def _read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _manifest_record(path: Path, split: str):
    value = _read(path)
    if (value.get("split") != split
            or value.get("configuration") != "S0"
            or value.get("instance_id") != path.stem):
        raise ValueError(f"unexpected S0 {split} manifest: {path}")
    return {
        "instance_id": value["instance_id"],
        "instance_digest": value["instance_digest"],
    }


def build_plan(instances: Path, selection_path: Path,
               normalization_path: Path):
    selection = _read(selection_path)
    normalization = _read(normalization_path)
    if (selection.get("configuration") != "S0"
            or selection.get("instance_count") != 50
            or selection.get("test_instances_used") is not False
            or selection.get("optimisation_outcomes_used") is not False):
        raise ValueError("selection is not the frozen training-only S0 ranking")
    ranking = selection.get("ranking", [])
    if len(ranking) != 50 or [row.get("rank") for row in ranking] != list(range(1, 51)):
        raise ValueError("selection ranking must contain ranks 1 through 50")
    if (normalization.get("source") != "training-only"
            or normalization.get("configuration") != "S0"
            or normalization.get("test_instances_used") is not False
            or normalization.get("validation_instances_used") is not False):
        raise ValueError("normalization is not the frozen training-only S0 artifact")

    ranks = evenly_spaced_ranks(len(ranking), TRAINING_INSTANCE_COUNT)
    selected = [ranking[rank - 1] for rank in ranks]
    selected_records = []
    for row in selected:
        manifest = instances / f"{row['instance_id']}.json"
        record = _manifest_record(manifest, "train")
        if record["instance_digest"] != row["instance_digest"]:
            raise ValueError(f"selection/manifest digest mismatch: {manifest}")
        selected_records.append({
            "complexity_rank": row["rank"],
            "candidate_path_token_count": row["candidate_path_token_count"],
            **record,
        })

    test_paths = sorted(instances.glob("test-S0-*.json"))
    if len(test_paths) != 10:
        raise ValueError(f"expected exactly 10 S0 test manifests; found {len(test_paths)}")
    test_records = [_manifest_record(path, "test") for path in test_paths]
    training_ids = [row["instance_id"] for row in selected_records]

    plan = {
        "schema_version": 1,
        "stage": "single-scene deterministic S0 formal component experiment",
        "configuration": "S0",
        "uncertainty": False,
        "population": POPULATION,
        "evaluation_budget": EVALUATION_BUDGET,
        "generation_cap": GENERATION_CAP,
        "methods": ["BASE", "I", "C", "M", "ICM"],
        "policy_seeds": list(POLICY_SEEDS),
        "algorithm_seed_base": ALGORITHM_SEED_BASE,
        "checkpoint_rule": "last checkpoint after the final predeclared episode",
        "validation_instances_used": False,
        "training": {
            "selection_rule": (
                "ten equally spaced ranks, including both endpoints, from the "
                "training-only static candidate-path-token complexity ranking"),
            "same_instance_set_for_all_components_and_policy_seeds": True,
            "instance_count_per_component_seed": len(selected_records),
            "instances": selected_records,
            "ordered_instance_ids_by_policy_seed": ordered_ids_by_seed(training_ids),
            "expected_minimum_ppo_updates_for_audit":
                EXPECTED_MINIMUM_PPO_UPDATES,
            "update_floor_role": (
                "post-run integrity check, not an adaptive stopping rule"),
        },
        "testing": {
            "instance_count": len(test_records),
            "instances": test_records,
            "training_instances_used": False,
        },
        "normalization": {
            "path": str(normalization_path.resolve()),
            "sha256": _sha256(normalization_path),
            "normalization_digest": normalization["normalization_digest"],
            "reference": normalization["reference"],
            "source": normalization["source"],
        },
        "selection": {
            "path": str(selection_path.resolve()),
            "sha256": _sha256(selection_path),
            "selection_digest": selection["selection_digest"],
        },
        "test_instances_used_for_plan_selection": False,
        "optimisation_outcomes_used_for_training_instance_selection": False,
    }
    plan["plan_digest"] = canonical_digest(plan)
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instances", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--normalization", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    for label, path in (
        ("instances", args.instances),
        ("selection", args.selection),
        ("normalization", args.normalization),
    ):
        if not path.exists():
            parser.error(f"{label} does not exist: {path}")
    if args.out.exists():
        parser.error(f"output already exists: {args.out}")
    try:
        plan = build_plan(args.instances, args.selection, args.normalization)
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))
    _atomic_json(args.out, plan)
    print(json.dumps({
        "completed": True,
        "out": str(args.out.resolve()),
        "plan_digest": plan["plan_digest"],
        "training_instances": [
            {"rank": row["complexity_rank"], "instance_id": row["instance_id"]}
            for row in plan["training"]["instances"]
        ],
        "test_instance_count": plan["testing"]["instance_count"],
    }, indent=2))


if __name__ == "__main__":
    main()
