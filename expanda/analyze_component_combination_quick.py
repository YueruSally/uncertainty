#!/usr/bin/env python3
"""Summarise a paired component and combination quick evaluation directory."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


METHODS = ("BASE", "I", "C", "M", "IC", "ICM")


def _read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _mean(values):
    return sum(values) / len(values)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--instances", nargs="+", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    args = parser.parse_args(argv)

    rows = []
    for instance in args.instances:
        for seed in args.seeds:
            cell = {}
            for method in METHODS:
                path = (
                    args.root / instance / f"seed_{seed}" / method
                    / "COMPLETE.json"
                )
                config_path = path.with_name("configuration.json")
                if not path.is_file():
                    parser.error(f"missing result: {path}")
                if not config_path.is_file():
                    parser.error(f"missing configuration: {config_path}")
                result = _read(path)
                config = _read(config_path)
                value = result.get("final_hypervolume")
                if (result.get("completed") is not True
                        or result.get("component") != method
                        or config.get("component") != method
                        or config.get("seeds", {}).get("algorithm") != seed
                        or not isinstance(value, (int, float))
                        or not math.isfinite(value)):
                    parser.error(f"invalid result: {path}")
                cell[method] = float(value)
            base = cell["BASE"]
            for method in METHODS:
                rows.append({
                    "instance": instance,
                    "seed": seed,
                    "method": method,
                    "final_hypervolume": cell[method],
                    "delta_vs_base": cell[method] - base,
                    "percent_vs_base": 100.0 * (cell[method] - base) / base,
                })

    aggregate = {}
    for method in METHODS:
        selected = [row for row in rows if row["method"] == method]
        aggregate[method] = {
            "runs": len(selected),
            "mean_hv": _mean([row["final_hypervolume"] for row in selected]),
            "mean_paired_delta": _mean(
                [row["delta_vs_base"] for row in selected]),
            "mean_paired_percent": _mean(
                [row["percent_vs_base"] for row in selected]),
            "wins": sum(row["delta_vs_base"] > 1e-12 for row in selected),
            "ties": sum(abs(row["delta_vs_base"]) <= 1e-12 for row in selected),
            "losses": sum(row["delta_vs_base"] < -1e-12 for row in selected),
        }

    per_instance = []
    for instance in args.instances:
        instance_rows = [row for row in rows if row["instance"] == instance]
        values = {
            method: [
                row for row in instance_rows if row["method"] == method
            ] for method in METHODS
        }
        per_instance.append({
            "instance": instance,
            "base_mean_hv": _mean([
                row["final_hypervolume"] for row in values["BASE"]
            ]),
            "mean_paired_percent": {
                method: _mean([
                    row["percent_vs_base"] for row in values[method]
                ]) for method in METHODS[1:]
            },
        })

    print(json.dumps({
        "root": str(args.root),
        "instances": args.instances,
        "algorithm_seeds": args.seeds,
        "paired_run_count_per_method": len(args.instances) * len(args.seeds),
        "aggregate": aggregate,
        "per_instance": per_instance,
    }, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
