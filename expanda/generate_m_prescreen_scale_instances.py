#!/usr/bin/env python3
"""Generate frozen training-domain S0 instances for the K8 learning curve."""
from __future__ import annotations

from dataclasses import asdict
import argparse
import hashlib
import json
from pathlib import Path
import random

import baseline_uncertainty as base
from transformer_ppo.config import SCENARIOS
from transformer_ppo.instance_catalog import (_batches, _find_path_seed,
                                               canonical_digest)


TRAIN_COUNT = 200
VALIDATION_COUNT = 20
MASTER_SEED = 93_000_000
PATH_SEED_OFFSET = 1_000_000
CCP_SEED_BASE = 95_000_000
ALGORITHM_SEED_BASE = 96_000_000
FINAL_TEST_MASTER_SEED = 98_000_000


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path, value):
    temporary = path.with_name(path.name+".tmp")
    temporary.write_text(json.dumps(value, indent=2)+"\n", encoding="utf-8")
    temporary.replace(path)


def generate(data, source_instances, out):
    catalog_path = source_instances / "catalog.json"
    catalog = read_json(catalog_path)
    claimed = catalog.get("catalog_digest")
    payload = dict(catalog); payload.pop("catalog_digest", None)
    if canonical_digest(payload) != claimed:
        raise ValueError("source catalog digest mismatch")
    data_sha = digest(data)
    if catalog.get("source_data_sha256") != data_sha:
        raise ValueError("network workbook differs from source catalog")
    train_od = [tuple(pair) for pair in catalog.get("train_od_pool", ())]
    test_od = {tuple(pair) for pair in catalog.get("test_od_pool", ())}
    if not train_od or set(train_od) & test_od:
        raise ValueError("source catalog lacks disjoint train/test OD pools")
    samples = sorted(source_instances.glob("train-S0-*.json"))
    if not samples:
        raise ValueError("source catalog has no train-S0 instance")
    sample = read_json(samples[0])
    release = sample["release_time_definition"]
    plan = {"protocol": "m-prescreen-scale-instance-generation-v1",
        "train_count": TRAIN_COUNT, "validation_count": VALIDATION_COUNT,
        "master_seed": MASTER_SEED, "path_seed_offset": PATH_SEED_OFFSET,
        "ccp_seed_base": CCP_SEED_BASE,
        "algorithm_seed_base": ALGORITHM_SEED_BASE,
        "locked_final_test_master_seed": FINAL_TEST_MASTER_SEED,
        "source_catalog_digest": claimed,
        "source_data_sha256": data_sha,
        "data": str(data.resolve()),
        "source_instances": str(source_instances.resolve())}
    out.mkdir(parents=True, exist_ok=True)
    plan_path = out/"GENERATION_PLAN.json"
    if plan_path.exists() and read_json(plan_path) != plan:
        raise ValueError("scale generation plan changed; retain old output")
    if any(out.iterdir()) and not plan_path.exists():
        raise ValueError("nonempty scale instance directory lacks generation plan")
    atomic_json(plan_path, plan)
    completed_catalog = out/"catalog.json"
    if completed_catalog.exists():
        completed = read_json(completed_catalog)
        identity = dict(completed); digest_value = identity.pop("catalog_digest", None)
        if (canonical_digest(identity) != digest_value
                or completed.get("train_count") != TRAIN_COUNT
                or completed.get("validation_count") != VALIDATION_COUNT):
            raise ValueError("completed scale catalog is invalid")
        return completed
    existing = set()
    for path in source_instances.glob("*.json"):
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
    total = TRAIN_COUNT + VALIDATION_COUNT
    if MASTER_SEED + total*100 >= FINAL_TEST_MASTER_SEED:
        raise ValueError("generated namespace overlaps locked final-test namespace")
    records = []
    state = random.getstate()
    try:
        for index in range(total):
            role = "train" if index < TRAIN_COUNT else "validation"
            role_index = index+1 if role == "train" else index-TRAIN_COUNT+1
            instance_seed = MASTER_SEED + index*100
            batches = _batches(raw, train_od, k, alpha, instance_seed,
                "train", float(release["maximum_h"]),
                float(release["step_h"]))
            signature = canonical_digest([asdict(batch) for batch in batches])
            if signature in existing:
                raise ValueError("generated instance duplicates an existing batch set")
            existing.add(signature)
            path_seed = _find_path_seed(instance_seed+PATH_SEED_OFFSET,
                nodes, regions, arcs, batches, timetable, lookup)
            name = f"prescreen-{role}-S0-{role_index:03d}"
            record = {"schema_version": 2, "instance_id": name,
                "split": "train", "development_role": role,
                "configuration": "S0", "K": k,
                "deadline_window_alpha": alpha,
                "alpha_definition": sample["alpha_definition"],
                "release_time_definition": release,
                "quantity_definition": sample["quantity_definition"],
                "od_pool": "train", "instance_seed": instance_seed,
                "path_seed": path_seed,
                "ccp100_seed": CCP_SEED_BASE+index,
                "oos5000_seed": None,
                "algorithm_seed": ALGORITHM_SEED_BASE+index,
                "source_data_sha256": data_sha,
                "source_catalog_digest": claimed,
                "batch_signature": signature,
                "batches": [asdict(batch) for batch in batches]}
            record["instance_digest"] = canonical_digest(record)
            path = out/f"{name}.json"
            if path.exists():
                if read_json(path) != record:
                    raise ValueError(f"existing generated instance changed: {name}")
            else:
                atomic_json(path, record)
            records.append({key: record[key] for key in (
                "instance_id", "development_role", "instance_seed",
                "path_seed", "ccp100_seed", "algorithm_seed",
                "batch_signature", "instance_digest")})
            print(f"[M-K8-INSTANCES] {index+1}/{total} {name}", flush=True)
            atomic_json(out/"GENERATION_STATUS.json", {
                "completed_instances": index+1, "total_instances": total,
                "last_instance": name})
    finally:
        random.setstate(state)
    generated = {"schema_version": 1,
        "purpose": "K8 prescreen nested instance-count learning curve",
        "configuration": "S0", "od_pool": "train",
        "train_count": TRAIN_COUNT, "validation_count": VALIDATION_COUNT,
        "master_seed": MASTER_SEED, "path_seed_offset": PATH_SEED_OFFSET,
        "ccp_seed_base": CCP_SEED_BASE,
        "algorithm_seed_base": ALGORITHM_SEED_BASE,
        "locked_final_test_master_seed": FINAL_TEST_MASTER_SEED,
        "source_data_sha256": data_sha,
        "source_catalog_digest": claimed,
        "training_outcomes_used": False,
        "validation_outcomes_used": False,
        "final_test_instances_used": False,
        "instances": records}
    generated["catalog_digest"] = canonical_digest(generated)
    atomic_json(out/"catalog.json", generated)
    atomic_json(out/"GENERATION_STATUS.json", {
        "completed_instances": total, "total_instances": total,
        "catalog_digest": generated["catalog_digest"], "stage": "complete"})
    return generated


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path,
        default=Path(__file__).parent/"data/data_expanded.xlsx")
    parser.add_argument("--source-instances", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = generate(args.data.resolve(), args.source_instances.resolve(),
                      args.out.resolve())
    print(json.dumps({key: result[key] for key in (
        "train_count", "validation_count", "catalog_digest",
        "final_test_instances_used")}, indent=2))


if __name__ == "__main__":
    main()
