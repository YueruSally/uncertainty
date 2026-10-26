#!/usr/bin/env python3
"""Freeze Rule-I path-feature ranges from all 50 S0 training instances."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random

import baseline_uncertainty as base
from transformer_ppo.initialization_rule import FEATURES
from transformer_ppo.instance_catalog import canonical_digest, read_instance


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path,
                        default=Path(__file__).parent / "data/data_expanded.xlsx")
    parser.add_argument("--instances", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error(f"output already exists: {args.out}")
    manifests = sorted(args.instances.glob("train-S0-*.json"))
    if len(manifests) != 50:
        parser.error(f"expected exactly 50 train-S0 manifests; found {len(manifests)}")
    source_sha = _sha256(args.data)
    base.BORDER_EVENT_DEFINITIONS = base.load_border_event_definitions(
        base.DEFAULT_BORDER_EVENT_DATA_FILE)
    network = base.load_network_from_extended(str(args.data))
    nodes, regions, _, _, _, arcs, timetables, _, *_ = network
    timetable = base.build_timetable_dict(timetables)
    lookup = base.build_arc_lookup(arcs)
    minimum = [float("inf")] * 3
    maximum = [float("-inf")] * 3
    point_count = 0
    random_state = random.getstate()
    try:
        for index, manifest in enumerate(manifests, start=1):
            record, batches = read_instance(manifest)
            if (record.get("split") != "train"
                    or record.get("configuration") != "S0"
                    or record.get("source_data_sha256") != source_sha):
                parser.error(f"unexpected training manifest: {manifest}")
            random.seed(int(record["path_seed"]))
            library = base.build_path_library(
                nodes, regions, arcs, batches, timetable, lookup)
            for batch in batches:
                candidates = library.get((batch.origin, batch.destination), ())
                if not candidates:
                    parser.error(f"empty candidate set: {manifest} Batch {batch.batch_id}")
                for path in candidates:
                    values = [float(getattr(path, feature)) for feature in FEATURES]
                    minimum = [min(old, value) for old, value in zip(minimum, values)]
                    maximum = [max(old, value) for old, value in zip(maximum, values)]
                    point_count += 1
            print(f"[RULE-I] {index:02d}/50 {manifest.stem}", flush=True)
    finally:
        random.setstate(random_state)
    if any(high <= low for low, high in zip(minimum, maximum)):
        parser.error("non-positive Rule-I feature range")
    result = {
        "schema_version": 1,
        "source": "training-only",
        "source_method": "all S0 training candidate paths",
        "configuration": "S0",
        "instance_count": 50,
        "point_count": point_count,
        "features": list(FEATURES),
        "minimum": minimum,
        "maximum": maximum,
        "score": "sum of three un-clipped min-max normalized features",
        "weights": [1.0, 1.0, 1.0],
        "test_instances_used": False,
        "validation_instances_used": False,
        "source_data_sha256": source_sha,
    }
    result["normalization_digest"] = canonical_digest(result)
    _atomic_json(args.out, result)
    print(json.dumps({
        "completed": True,
        "out": str(args.out.resolve()),
        "instance_count": 50,
        "point_count": point_count,
        "minimum": minimum,
        "maximum": maximum,
        "normalization_digest": result["normalization_digest"],
    }, indent=2))


if __name__ == "__main__":
    main()
