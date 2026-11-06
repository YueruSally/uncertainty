#!/usr/bin/env python3
"""Recompute the locked formal report with the corrected two-level bootstrap."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

SOURCE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE))

from m_operator_offline import digest, read_json, write_json
from experiments.run_m_prescreen_formal_v1 import (
    BALANCED, INSTANCES, SEEDS, SUPERVISED, paired_summary,
    summarize_optimization)


def execute(root, out):
    root = root.resolve(); out = out.resolve(); out.mkdir(parents=True, exist_ok=True)
    complete = read_json(root/"COMPLETE.json")
    if (not (root/"PIPELINE_COMPLETE").exists()
            or complete.get("completed") is not True
            or complete.get("final_test_opened") is not True
            or complete.get("searches") != 400):
        raise ValueError("requires one completed locked formal experiment")
    methods = tuple(complete["methods"]); ppo = methods[-1]
    optimization = summarize_optimization(
        complete["optimization_runs"], methods)

    rows = []
    for name in INSTANCES:
        for seed in SEEDS:
            path = root/"oos"/"runs"/name/f"seed_{seed}.json"
            row = read_json(path)
            if (row.get("instance") != name
                    or row.get("algorithm_seed") != seed
                    or set(row.get("oos_hypervolumes", {})) != set(methods)):
                raise ValueError(f"invalid saved OOS row: {path}")
            rows.append(row)
    paired = []
    direct = []
    for row in rows:
        hvs = row["oos_hypervolumes"]
        for method in methods:
            paired.append({"instance": row["instance"],
                "algorithm_seed": row["algorithm_seed"], "method": method,
                "percent_vs_balanced": 100*(hvs[method]/hvs[BALANCED]-1)})
        direct.append({"instance": row["instance"],
            "algorithm_seed": row["algorithm_seed"],
            "ppo_minus_supervised_percent": 100*(
                hvs[ppo]/hvs[SUPERVISED]-1)})
    oos = {"vs_balanced_percent": {
        method: paired_summary(
            [row for row in paired if row["method"] == method],
            "percent_vs_balanced", 98310000+index)
        for index, method in enumerate(methods)},
        "ppo_minus_supervised": paired_summary(
            direct, "ppo_minus_supervised_percent", 98320000)}
    result = {"completed": True,
        "definition": (
            "report-only correction: resample 10 final-test instances with "
            "replacement and, within every sampled instance, resample all "
            "10 paired algorithm-seed runs before computing the grand mean"),
        "raw_searches_reused": 400,
        "raw_oos_rows_reused": 100,
        "new_searches": 0, "new_objective_evaluations": 0,
        "new_oos_evaluations": 0, "new_training_steps": 0,
        "methods": list(methods),
        "optimization": optimization,
        "independent_oos5000": oos,
        "source_sha256": {
            str((root/"COMPLETE.json").resolve()): digest(root/"COMPLETE.json"),
            str((root/"QUICK_RESULT.json").resolve()): digest(root/"QUICK_RESULT.json"),
            str(Path(__file__).resolve()): digest(Path(__file__).resolve())},
        "unchanged_from_original": {
            "means": True, "wins_ties_losses": True,
            "raw_hypervolumes": True, "only_confidence_intervals_changed": True},
        "original_report_preserved": str((root/"QUICK_RESULT.json").resolve())}
    write_json(out/"QUICK_RESULT.json", result)
    (out/"RECOMPUTE_COMPLETE").write_text(digest(out/"QUICK_RESULT.json")+"\n")
    print((out/"QUICK_RESULT.json").read_text(), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(); execute(args.root, args.out)


if __name__ == "__main__":
    main()
