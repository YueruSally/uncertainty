#!/usr/bin/env python3
"""Select low/median/high S0 training instances using static path complexity.

The selection is deliberately independent of optimisation outcomes.  It uses
only the frozen training manifests, network workbook, and reproducible path
libraries.  The resulting three instances are intended for resource and
convergence pilots, not for final testing.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random

import baseline_uncertainty as base
from resource_pilot_selection import METRIC_NAME, SELECTION_RULE, select_records
from transformer_ppo.instance_catalog import canonical_digest, read_instance


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


def _complexity(record, batches, path_library, manifest: Path):
    counts = []
    path_node_tokens = 0
    path_mode_tokens = 0
    for batch in batches:
        paths = path_library.get((batch.origin, batch.destination), ())
        if not paths:
            raise ValueError(
                f"reproducible path library is empty for Batch {batch.batch_id} "
                f"in {manifest.name}")
        counts.append(len(paths))
        path_node_tokens += sum(len(path.nodes) for path in paths)
        path_mode_tokens += sum(len(path.modes) for path in paths)
    # One candidate marker plus its node and mode sequence approximates the
    # amount of variable-length structure the Transformer must encode.
    candidate_count = sum(counts)
    token_count = candidate_count + path_node_tokens + path_mode_tokens
    return {
        "instance_id": record["instance_id"],
        "manifest": str(manifest.resolve()),
        "instance_digest": record["instance_digest"],
        "path_seed": int(record["path_seed"]),
        "batch_count": len(batches),
        "unique_od_count": len({
            (batch.origin, batch.destination) for batch in batches
        }),
        "candidate_path_count": candidate_count,
        "mean_candidate_paths_per_batch": candidate_count / len(batches),
        "maximum_candidate_paths_per_batch": max(counts),
        "candidate_path_node_count": path_node_tokens,
        "candidate_path_mode_count": path_mode_tokens,
        METRIC_NAME: token_count,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data", type=Path,
        default=Path(__file__).parent / "data/data_expanded.xlsx")
    parser.add_argument("--instances", type=Path, required=True)
    parser.add_argument("--configuration", default="S0")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if not args.data.is_file():
        parser.error(f"network workbook does not exist: {args.data}")
    if not args.instances.is_dir():
        parser.error(f"instance directory does not exist: {args.instances}")
    if args.out.exists():
        parser.error(f"output already exists: {args.out}")

    manifests = sorted(
        args.instances.glob(f"train-{args.configuration}-*.json"))
    if len(manifests) != 50:
        parser.error(
            f"expected exactly 50 train-{args.configuration} manifests; "
            f"found {len(manifests)}")

    source_sha = _sha256(args.data)
    base.BORDER_EVENT_DEFINITIONS = base.load_border_event_definitions(
        base.DEFAULT_BORDER_EVENT_DATA_FILE)
    network = base.load_network_from_extended(str(args.data))
    nodes, regions, _, _, _, arcs, timetables, _, *_ = network
    timetable = base.build_timetable_dict(timetables)
    lookup = base.build_arc_lookup(arcs)

    records = []
    random_state = random.getstate()
    try:
        for index, manifest in enumerate(manifests, start=1):
            record, batches = read_instance(manifest)
            if (record.get("split") != "train"
                    or record.get("configuration") != args.configuration):
                parser.error(f"unexpected manifest metadata: {manifest}")
            if record.get("source_data_sha256") != source_sha:
                parser.error(f"source workbook digest mismatch: {manifest}")
            random.seed(int(record["path_seed"]))
            paths = base.build_path_library(
                nodes, regions, arcs, batches, timetable, lookup)
            row = _complexity(record, batches, paths, manifest)
            records.append(row)
            print(
                f"[SELECT] {index:02d}/{len(manifests)} "
                f"{row['instance_id']} {METRIC_NAME}={row[METRIC_NAME]}",
                flush=True,
            )
    finally:
        random.setstate(random_state)

    ranking, selected = select_records(records)
    result = {
        "schema_version": 1,
        "purpose": "training-only S0 resource and convergence pilot selection",
        "configuration": args.configuration,
        "source_data": str(args.data.resolve()),
        "source_data_sha256": source_sha,
        "instance_directory": str(args.instances.resolve()),
        "instance_count": len(records),
        "test_instances_used": False,
        "optimisation_outcomes_used": False,
        "primary_metric": METRIC_NAME,
        "primary_metric_definition": (
            "sum over Batch-specific candidate paths of "
            "1 + number of path nodes + number of path modes"),
        "selection_rule": SELECTION_RULE,
        "selected": selected,
        "ranking": [
            {"rank": index, **row}
            for index, row in enumerate(ranking, start=1)
        ],
    }
    result["selection_digest"] = canonical_digest(result)
    _atomic_json(args.out, result)
    print(json.dumps({
        "completed": True,
        "out": str(args.out.resolve()),
        "selection_digest": result["selection_digest"],
        "selected": [
            {
                "role": row["role"],
                "rank": row["rank"],
                "instance_id": row["instance_id"],
                METRIC_NAME: row[METRIC_NAME],
            }
            for row in selected
        ],
    }, indent=2))


if __name__ == "__main__":
    main()
