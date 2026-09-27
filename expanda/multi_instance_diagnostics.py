#!/usr/bin/env python3
"""Inspect validation model action before selecting a label by OOS HV."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np

from multi_instance_logging import write_json


def rows(path):
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            yield json.loads(line)


def diagnose(root, out):
    if out.exists():
        raise ValueError("diagnostic file exists")
    report = {}
    for label in ("selection_survivor", "first_front_member", "parent_relation"):
        paths = sorted((root / "validation_labels").glob(f"S*/{label}_eps0.1/run*"))
        if len(paths) != 10:
            raise ValueError(f"expected ten validation runs for {label}")
        scores, tv, fallback, different, argmax = [], [], Counter(), [], []
        candidate_counts, operators = Counter(), Counter()
        for run in paths:
            by_event = defaultdict(int)
            for row in rows(run / "mutation_candidates.jsonl"):
                if row["eligible"]:
                    by_event[row["event_id"]] += 1
                    if row["policy_score"] is not None:
                        scores.append(row["policy_score"])
            candidate_counts.update(by_event.values())
            for row in rows(run / "mutation_outcomes.jsonl"):
                operators[row["operator"]] += 1
                meta = row["policy_metadata"]
                tv.append(meta.get("tv_distance", 0.))
                fallback[str(meta.get("fallback"))] += 1
                if "different_from_shadow_rule" in meta:
                    different.append(meta["different_from_shadow_rule"])
                if "different_argmax" in meta:
                    argmax.append(meta["different_argmax"])
        report[label] = dict(candidate_scores=dict(n=len(scores),
            minimum=float(np.min(scores)) if scores else None,
            maximum=float(np.max(scores)) if scores else None,
            standard_deviation=float(np.std(scores)) if scores else None,
            distinct_rounded_4dp=int(np.unique(np.round(scores, 4)).size)),
            eligible_count_distribution=dict(candidate_counts),
            operator_attempts=dict(operators), fallback_counts=dict(fallback),
            mean_tv=float(np.mean(tv)) if tv else None,
            different_from_shadow_rule_rate=float(np.mean(different)) if different else None,
            different_argmax_rate=float(np.mean(argmax)) if argmax else None)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_json(out, report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(diagnose(args.root, args.out))
