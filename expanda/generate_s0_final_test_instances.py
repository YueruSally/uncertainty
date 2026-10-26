#!/usr/bin/env python3
"""Generate a fresh S0 final-test set from the frozen test-side data domain."""
from __future__ import annotations

from dataclasses import asdict
import argparse
import hashlib
import json
from pathlib import Path
import random

import baseline_uncertainty as base
from transformer_ppo.config import SCENARIOS
from transformer_ppo.instance_catalog import (
    _batches,
    _find_path_seed,
    canonical_digest,
)


DEFAULT_MASTER_SEED = 62_000_000


def _read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _atomic_json(path, value):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path,
                        default=Path(__file__).parent / "data/data_expanded.xlsx")
    parser.add_argument("--source-instances", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--master-seed", type=int, default=DEFAULT_MASTER_SEED)
    args = parser.parse_args()
    if args.count < 1:
        parser.error("--count must be positive")
    if args.out.exists() and any(args.out.iterdir()):
        parser.error(f"output directory must be empty: {args.out}")
    source_catalog_path = args.source_instances / "catalog.json"
    if not source_catalog_path.is_file():
        parser.error(f"source catalog is missing: {source_catalog_path}")
    source_catalog = _read(source_catalog_path)
    source_digest = source_catalog.get("catalog_digest")
    digest_payload = dict(source_catalog)
    digest_payload.pop("catalog_digest", None)
    if canonical_digest(digest_payload) != source_digest:
        parser.error("source catalog digest mismatch")
    data_sha = _sha256(args.data)
    if source_catalog.get("source_data_sha256") != data_sha:
        parser.error("network workbook differs from the frozen source catalog")
    test_od_pool = [tuple(pair) for pair in source_catalog.get("test_od_pool", ())]
    train_od_pool = {tuple(pair) for pair in source_catalog.get("train_od_pool", ())}
    if not test_od_pool or train_od_pool.intersection(test_od_pool):
        parser.error("source catalog does not contain disjoint train/test OD pools")

    development_paths = sorted(args.source_instances.glob("test-S0-*.json"))
    if len(development_paths) != 10:
        parser.error(
            f"expected 10 previously used S0 development-test manifests; "
            f"found {len(development_paths)}")
    sample = _read(development_paths[0])
    release = sample["release_time_definition"]
    release_max = float(release["maximum_h"])
    release_step = float(release["step_h"])

    existing_signatures = set()
    for path in args.source_instances.glob("*.json"):
        if path.name == "catalog.json":
            continue
        value = _read(path)
        signature = value.get("batch_signature")
        if signature:
            existing_signatures.add(signature)

    base.BORDER_EVENT_DEFINITIONS = base.load_border_event_definitions(
        base.DEFAULT_BORDER_EVENT_DATA_FILE)
    network = base.load_network_from_extended(str(args.data))
    nodes, regions, _, _, _, arcs, timetables, raw, *_ = network
    timetable = base.build_timetable_dict(timetables)
    lookup = base.build_arc_lookup(arcs)
    k, alpha = SCENARIOS["S0"]
    args.out.mkdir(parents=True, exist_ok=True)
    records = []
    random_state = random.getstate()
    try:
        for index in range(args.count):
            instance_seed = args.master_seed + index * 100
            batches = _batches(
                raw, test_od_pool, k, alpha, instance_seed, "test",
                release_max, release_step)
            signature = canonical_digest([asdict(batch) for batch in batches])
            if signature in existing_signatures:
                raise ValueError(
                    "new final-test instance duplicates a frozen source instance")
            existing_signatures.add(signature)
            path_seed = _find_path_seed(
                instance_seed + 1_000_000, nodes, regions, arcs, batches,
                timetable, lookup)
            instance_id = f"final-test-S0-{index + 1:03d}"
            record = {
                "schema_version": 2,
                "instance_id": instance_id,
                "split": "test",
                "test_role": "final",
                "configuration": "S0",
                "K": k,
                "deadline_window_alpha": alpha,
                "alpha_definition": (
                    "LT=ET+max(alpha*sampled_source_window,1 hour)"),
                "release_time_definition": release,
                "quantity_definition": sample["quantity_definition"],
                "od_pool": "test",
                "instance_seed": instance_seed,
                "path_seed": path_seed,
                "ccp100_seed": 63_000_000 + index,
                "oos5000_seed": 64_000_000 + index,
                "source_data_sha256": data_sha,
                "source_catalog_digest": source_digest,
                "batch_signature": signature,
                "batches": [asdict(batch) for batch in batches],
            }
            record["instance_digest"] = canonical_digest(record)
            _atomic_json(args.out / f"{instance_id}.json", record)
            records.append({key: record[key] for key in (
                "instance_id", "split", "test_role", "configuration", "K",
                "deadline_window_alpha", "instance_seed", "path_seed",
                "ccp100_seed", "oos5000_seed", "batch_signature",
                "instance_digest")})
            print(
                f"[FINAL-TEST] {index + 1:02d}/{args.count} "
                f"{instance_id} path_seed={path_seed}", flush=True)
    finally:
        random.setstate(random_state)

    catalog = {
        "schema_version": 1,
        "purpose": "previously unseen S0 final-test instances",
        "configuration": "S0",
        "split": "test",
        "test_role": "final",
        "instance_count": len(records),
        "master_seed": args.master_seed,
        "source_data_sha256": data_sha,
        "source_catalog_file": source_catalog_path.name,
        "source_instances_directory_name": args.source_instances.name,
        "source_catalog_digest": source_digest,
        "source_test_od_pool_digest": canonical_digest(test_od_pool),
        "development_test_instances_excluded": [
            path.stem for path in development_paths],
        "training_instances_used": False,
        "development_test_instances_used": False,
        "optimisation_outcomes_used": False,
        "model_selection_outcomes_used": False,
        "instances": records,
    }
    catalog["catalog_digest"] = canonical_digest(catalog)
    _atomic_json(args.out / "catalog.json", catalog)
    print(json.dumps({
        "completed": True,
        "out": str(args.out.resolve()),
        "instance_count": len(records),
        "catalog_digest": catalog["catalog_digest"],
        "source_catalog_digest": source_digest,
        "training_instances_used": False,
        "development_test_instances_used": False,
        "optimisation_outcomes_used": False,
        "model_selection_outcomes_used": False,
    }, indent=2))


if __name__ == "__main__":
    main()
