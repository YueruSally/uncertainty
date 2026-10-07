#!/usr/bin/env python3
"""Compute common normalized OOS-HV for four paired runs of one test instance."""
import argparse
import json
from pathlib import Path

from multi_instance_logging import write_json
from transformer_ppo.archive import ObjectiveNormalizer, exact_hv_3d, nondominated
from transformer_ppo.instance_catalog import read_instance


METHODS = ("random", "rule", "transformer", "mlp")


def parse_method_path(value):
    method, separator, path = value.partition("=")
    if not separator or method not in METHODS or not path:
        raise argparse.ArgumentTypeError(
            "expected METHOD=PATH where METHOD is random, rule, transformer, or mlp")
    return method, Path(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instance", type=Path, required=True)
    parser.add_argument("--normalization", type=Path, required=True)
    parser.add_argument("--oos", type=parse_method_path, action="append", required=True,
                        help="Repeat exactly once per method, e.g. transformer=.../oos_5000.json")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    supplied = dict(args.oos)
    if len(args.oos) != len(METHODS) or set(supplied) != set(METHODS):
        parser.error("--oos must provide each of random, rule, transformer and mlp exactly once")
    if args.out.exists():
        parser.error("output already exists")
    instance, batches = read_instance(args.instance)
    if instance["split"] != "test":
        parser.error("OOS-HV can only be computed for a frozen test manifest")
    values = json.loads(args.normalization.read_text(encoding="utf-8"))
    if values.get("source") != "training-only":
        parser.error("normalization must declare source=training-only")
    divisor = (
        max(1.0, sum(float(batch.quantity) for batch in batches)),
        max(1.0, sum(float(batch.quantity) for batch in batches)),
        max(1.0, max(float(batch.LT) for batch in batches)
            - min(float(batch.ET) for batch in batches)),
    )
    normalizer = ObjectiveNormalizer(
        tuple(values["minimum"]), tuple(values["maximum"]),
        tuple(values.get("reference", (1.1, 1.1, 1.1))), divisor)
    rows, master_digest = [], None
    for method in METHODS:
        result = json.loads(supplied[method].read_text(encoding="utf-8"))
        if (result.get("size") != 5000
                or result.get("instance_digest") != instance["instance_digest"]
                or result.get("oos_seed") != instance["oos5000_seed"]):
            raise ValueError(f"{method} OOS result does not match the frozen test manifest")
        if master_digest is None:
            master_digest = result["oos_master_digest"]
        elif result["oos_master_digest"] != master_digest:
            raise ValueError("methods were not evaluated on the same OOS scenario master")
        points = []
        for candidate in result["candidates"]:
            objectives = tuple(float(candidate[name]["q90"])
                               for name in ("cost", "emission", "makespan"))
            points.append(normalizer.transform(objectives))
        front = nondominated(points)
        rows.append({
            "configuration": instance["configuration"],
            "instance_id": instance["instance_id"], "method": method,
            "oos_hv": exact_hv_3d(front, normalizer.reference),
            "oos_candidate_count": len(points), "oos_front_size": len(front),
            "oos_master_digest": master_digest,
        })
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.out, rows)
    print(rows)


if __name__ == "__main__":
    main()
