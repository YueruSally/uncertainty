#!/usr/bin/env python3
"""Freeze the training-only S0 V2 initialization top-k ablation."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from s0_deterministic_formal_plan import canonical_digest


FIT_RANKS = (1, 25, 50)
VALIDATION_RANKS = (2, 26, 49)
ROLES = ("low", "median", "high")
SETTINGS = (
    {"name": "top4", "top_k": 4},
    {"name": "top8", "top_k": 8},
    {"name": "top16", "top_k": 16},
    {"name": "all", "top_k": None},
)


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


def _ranked_rows(selection):
    if (selection.get("configuration") != "S0"
            or selection.get("instance_count") != 50
            or selection.get("test_instances_used") is not False
            or selection.get("optimisation_outcomes_used") is not False):
        raise ValueError("selection is not the training-only S0 ranking")
    ranking = selection.get("ranking", [])
    if (len(ranking) != 50
            or [row.get("rank") for row in ranking] != list(range(1, 51))):
        raise ValueError("selection must contain the complete ranks 1 through 50")
    return ranking


def _records(ranking, ranks, instances: Path):
    records = []
    for role, rank in zip(ROLES, ranks):
        source = ranking[rank - 1]
        manifest = instances / f"{source['instance_id']}.json"
        value = _read(manifest)
        if (value.get("split") != "train"
                or value.get("configuration") != "S0"
                or value.get("instance_id") != source["instance_id"]
                or value.get("instance_digest") != source["instance_digest"]):
            raise ValueError(f"selection/manifest mismatch: {manifest}")
        records.append({
            "role": role,
            "rank": rank,
            "instance_id": source["instance_id"],
            "instance_digest": source["instance_digest"],
            "candidate_path_count": source["candidate_path_count"],
            "candidate_path_token_count": source[
                "candidate_path_token_count"],
        })
    return records


def build_plan(instances: Path, selection_path: Path,
               normalization_path: Path, rule_normalization_path: Path):
    selection = _read(selection_path)
    normalization = _read(normalization_path)
    rule_normalization = _read(rule_normalization_path)
    ranking = _ranked_rows(selection)
    if (normalization.get("source") != "training-only"
            or normalization.get("configuration") != "S0"
            or normalization.get("test_instances_used") is not False
            or normalization.get("validation_instances_used") is not False):
        raise ValueError("objective normalization is not training-only S0")
    if not rule_normalization.get("normalization_digest"):
        raise ValueError("Rule-I normalization has no normalization_digest")

    fitting = _records(ranking, FIT_RANKS, instances)
    validation = _records(ranking, VALIDATION_RANKS, instances)
    fit_ids = {row["instance_id"] for row in fitting}
    validation_ids = {row["instance_id"] for row in validation}
    if not fit_ids.isdisjoint(validation_ids):
        raise ValueError("fitting and validation instances overlap")

    validation_selection = {
        "configuration": "S0",
        "instance_count": 50,
        "test_instances_used": False,
        "optimisation_outcomes_used": False,
        "selected": validation,
    }
    validation_selection["selection_digest"] = canonical_digest(
        validation_selection)
    plan = {
        "schema_version": 1,
        "stage": "training-only S0 V2 initialization top-k ablation",
        "formal": False,
        "configuration": "S0",
        "test_instances_used": False,
        "optimisation_outcomes_used": False,
        "selection_rule": {
            "fitting_ranks": list(FIT_RANKS),
            "validation_ranks": list(VALIDATION_RANKS),
            "basis": "static candidate-path-token complexity only",
            "sets_are_disjoint": True,
        },
        "fitting_instances": fitting,
        "validation_instances": validation,
        # Compatible with run_s0_base_convergence_pilot.py.
        **validation_selection,
        "settings": [dict(row) for row in SETTINGS],
        "training": {
            "policy_seed": 10_901,
            "population": 250,
            "evaluation_budget": 5_000,
            "generation_cap": 250,
            "one_policy_seed_for_development_screening": True,
        },
        "validation": {
            "algorithm_seed_base": 89_400_000,
            "repeats_per_instance": 3,
            "population": 250,
            "evaluation_budget": 5_000,
            "generation_cap": 250,
            "pairing": "same instance and algorithm seed for every setting",
        },
        "selection": {
            "path": str(selection_path.resolve()),
            "sha256": _sha256(selection_path),
            "source_selection_digest": selection["selection_digest"],
        },
        "normalization": {
            "path": str(normalization_path.resolve()),
            "sha256": _sha256(normalization_path),
            "normalization_digest": normalization["normalization_digest"],
        },
        "rule_normalization": {
            "path": str(rule_normalization_path.resolve()),
            "sha256": _sha256(rule_normalization_path),
            "normalization_digest": rule_normalization[
                "normalization_digest"],
        },
        "selection_criterion": {
            "primary": "highest mean paired validation HV",
            "near_tie_relative_fraction": 0.002,
            "near_tie_rule": (
                "within 0.2% of best: lower paired-delta standard deviation; "
                "then smaller top_k; all is ordered after finite top_k"),
            "failure_rule": (
                "if no top-k is stable against all/base, do not force top8"),
        },
    }
    plan["plan_digest"] = canonical_digest(plan)
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instances", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--normalization", type=Path, required=True)
    parser.add_argument("--rule-normalization", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    for label, path in (
            ("instances", args.instances),
            ("selection", args.selection),
            ("normalization", args.normalization),
            ("rule normalization", args.rule_normalization)):
        if not path.exists():
            parser.error(f"{label} does not exist: {path}")
    if args.out.exists():
        parser.error(f"output already exists: {args.out}")
    try:
        plan = build_plan(
            args.instances, args.selection, args.normalization,
            args.rule_normalization)
    except (KeyError, OSError, TypeError, ValueError,
            json.JSONDecodeError) as error:
        parser.error(str(error))
    _atomic_json(args.out, plan)
    print(json.dumps({
        "completed": True,
        "out": str(args.out.resolve()),
        "plan_digest": plan["plan_digest"],
        "fitting_instances": plan["fitting_instances"],
        "validation_instances": plan["validation_instances"],
        "settings": plan["settings"],
    }, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
