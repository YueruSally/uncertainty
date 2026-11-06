#!/usr/bin/env python3
"""Generate three untouched training-domain instances for bounded K8 search."""
from __future__ import annotations

from dataclasses import asdict
import argparse
import json
from pathlib import Path
import random

import baseline_uncertainty as base
from generate_m_prescreen_scale_instances import (atomic_json, digest,
    read_json)
from transformer_ppo.config import SCENARIOS
from transformer_ppo.instance_catalog import (_batches, _find_path_seed,
                                               canonical_digest)


COUNT = 3
MASTER_SEED = 97_000_000
PATH_SEED_OFFSET = 400_000
CCP_SEED_BASE = 97_500_000
MANIFEST_ALGORITHM_SEED_BASE = 97_600_000
LOCKED_FINAL_TEST_MASTER_SEED = 98_000_000


def generate(data, source_instances, scale_instances, out):
    source_catalog = read_json(source_instances/"catalog.json")
    source_identity = dict(source_catalog)
    source_claimed = source_identity.pop("catalog_digest", None)
    scale_catalog = read_json(scale_instances/"catalog.json")
    scale_identity = dict(scale_catalog)
    scale_claimed = scale_identity.pop("catalog_digest", None)
    data_sha = digest(data)
    if (canonical_digest(source_identity) != source_claimed
            or source_catalog.get("source_data_sha256") != data_sha):
        raise ValueError("source catalog or workbook changed")
    if (canonical_digest(scale_identity) != scale_claimed
            or scale_catalog.get("source_catalog_digest") != source_claimed
            or scale_catalog.get("train_count") != 200
            or scale_catalog.get("validation_count") != 20):
        raise ValueError("scale instance catalog changed")
    train_od = [tuple(pair) for pair in source_catalog.get("train_od_pool", ())]
    test_od = {tuple(pair) for pair in source_catalog.get("test_od_pool", ())}
    if not train_od or set(train_od) & test_od:
        raise ValueError("source catalog lacks disjoint train/test OD pools")
    sample_paths = sorted(source_instances.glob("train-S0-*.json"))
    if not sample_paths:
        raise ValueError("source catalog has no train-S0 instance")
    sample = read_json(sample_paths[0])
    release = sample["release_time_definition"]
    plan = {
        "protocol": "m-prescreen-bounded-search-instance-generation-v1",
        "count": COUNT, "master_seed": MASTER_SEED,
        "path_seed_offset": PATH_SEED_OFFSET,
        "ccp_seed_base": CCP_SEED_BASE,
        "manifest_algorithm_seed_base": MANIFEST_ALGORITHM_SEED_BASE,
        "locked_final_test_master_seed": LOCKED_FINAL_TEST_MASTER_SEED,
        "source_catalog_digest": source_claimed,
        "scale_catalog_digest": scale_claimed,
        "source_data_sha256": data_sha,
        "data": str(data.resolve()),
        "source_instances": str(source_instances.resolve()),
        "scale_instances": str(scale_instances.resolve()),
    }
    out.mkdir(parents=True, exist_ok=True)
    plan_path = out/"GENERATION_PLAN.json"
    if plan_path.exists() and read_json(plan_path) != plan:
        raise ValueError("bounded-search generation plan changed")
    if any(out.iterdir()) and not plan_path.exists():
        raise ValueError("nonempty search-instance directory lacks plan")
    atomic_json(plan_path, plan)
    completed_path = out/"catalog.json"
    if completed_path.exists():
        completed = read_json(completed_path)
        identity = dict(completed)
        claimed = identity.pop("catalog_digest", None)
        if canonical_digest(identity) != claimed or completed.get("count") != COUNT:
            raise ValueError("completed bounded-search catalog is invalid")
        return completed
    existing = set()
    for root in (source_instances, scale_instances):
        for path in root.glob("*.json"):
            if path.name == "catalog.json":
                continue
            signature = read_json(path).get("batch_signature")
            if signature:
                existing.add(signature)
    base.BORDER_EVENT_DEFINITIONS = base.load_border_event_definitions(
        base.DEFAULT_BORDER_EVENT_DATA_FILE)
    network = base.load_network_from_extended(str(data))
    nodes, regions, _, _, _, arcs, timetables, raw, *_ = network
    timetable = base.build_timetable_dict(timetables)
    lookup = base.build_arc_lookup(arcs)
    k, alpha = SCENARIOS["S0"]
    records = []
    state = random.getstate()
    try:
        for index in range(COUNT):
            instance_seed = MASTER_SEED+index*100
            batches = _batches(raw, train_od, k, alpha, instance_seed,
                "train", float(release["maximum_h"]),
                float(release["step_h"]))
            signature = canonical_digest([asdict(batch) for batch in batches])
            if signature in existing:
                raise ValueError("bounded-search instance duplicates prior data")
            existing.add(signature)
            path_seed = _find_path_seed(instance_seed+PATH_SEED_OFFSET,
                nodes, regions, arcs, batches, timetable, lookup)
            name = f"prescreen-search-S0-{index+1:03d}"
            record = {"schema_version": 2, "instance_id": name,
                "split": "train", "development_role": "bounded_search",
                "configuration": "S0", "K": k,
                "deadline_window_alpha": alpha,
                "alpha_definition": sample["alpha_definition"],
                "release_time_definition": release,
                "quantity_definition": sample["quantity_definition"],
                "od_pool": "train", "instance_seed": instance_seed,
                "path_seed": path_seed, "ccp100_seed": CCP_SEED_BASE+index,
                "oos5000_seed": None,
                "algorithm_seed": MANIFEST_ALGORITHM_SEED_BASE+index,
                "source_data_sha256": data_sha,
                "source_catalog_digest": source_claimed,
                "scale_catalog_digest": scale_claimed,
                "batch_signature": signature,
                "batches": [asdict(batch) for batch in batches]}
            record["instance_digest"] = canonical_digest(record)
            path = out/f"{name}.json"
            if path.exists() and read_json(path) != record:
                raise ValueError(f"existing search instance changed: {name}")
            if not path.exists():
                atomic_json(path, record)
            records.append({key: record[key] for key in (
                "instance_id", "instance_seed", "path_seed", "ccp100_seed",
                "algorithm_seed", "batch_signature", "instance_digest")})
            print(f"[M-K8-SEARCH-INSTANCES] {index+1}/{COUNT} {name}",
                  flush=True)
    finally:
        random.setstate(state)
    catalog = {"schema_version": 1,
        "purpose": "untouched training-domain bounded K8 final-HV search",
        "configuration": "S0", "od_pool": "train", "count": COUNT,
        "master_seed": MASTER_SEED, "path_seed_offset": PATH_SEED_OFFSET,
        "ccp_seed_base": CCP_SEED_BASE,
        "manifest_algorithm_seed_base": MANIFEST_ALGORITHM_SEED_BASE,
        "locked_final_test_master_seed": LOCKED_FINAL_TEST_MASTER_SEED,
        "source_data_sha256": data_sha,
        "source_catalog_digest": source_claimed,
        "scale_catalog_digest": scale_claimed,
        "training_instances_used": False,
        "validation_instances_used": False,
        "final_test_instances_used": False,
        "instances": records}
    catalog["catalog_digest"] = canonical_digest(catalog)
    atomic_json(out/"catalog.json", catalog)
    return catalog


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path,
        default=Path(__file__).parent/"data/data_expanded.xlsx")
    parser.add_argument("--source-instances", type=Path, required=True)
    parser.add_argument("--scale-instances", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = generate(args.data.resolve(), args.source_instances.resolve(),
        args.scale_instances.resolve(), args.out.resolve())
    print(json.dumps({key: result[key] for key in (
        "count", "catalog_digest", "final_test_instances_used")}, indent=2))


if __name__ == "__main__":
    main()
