#!/usr/bin/env python3
"""Analyze frozen per-instance OOS-HV JSON files or one equivalent CSV."""
import argparse
import csv
import json
from pathlib import Path

from multi_instance_logging import write_json
from transformer_ppo.statistics import formal_report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input-csv", type=Path,
                        help="CSV columns: configuration,instance_id,method,oos_hv")
    source.add_argument("--input-root", type=Path,
                        help="Root containing per-instance oos_hv.json outputs")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("output already exists")
    if args.input_csv:
        with args.input_csv.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    else:
        rows = []
        for path in sorted(args.input_root.rglob("oos_hv.json")):
            value = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(value, list):
                raise ValueError(f"expected a JSON list: {path}")
            rows.extend(value)
    result = formal_report(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.out, result)
    print({name: len(entries) for name, entries in result["families"].items()})
