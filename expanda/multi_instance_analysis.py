#!/usr/bin/env python3
"""Frozen HV bounds, validation label choice, and paired test summaries."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

from multi_instance_catalog import read_instance
from multi_instance_logging import write_json
from postprocess_formal_ev_vs_ccp import exact_hv_3d

REF = (1.1, 1.1, 1.1)
LABELS = ("selection_survivor", "first_front_member", "parent_relation")


def _json(path):
    return json.loads(path.read_text())


def _points(run, view, size):
    if view == "in_sample":
        return [tuple(row["optimisation_objectives"][x] for x in
                      ("cost", "emission", "makespan"))
                for row in _json(run / "final_feasible_nondominated.json")]
    results = _json(run / f"oos_{size}" / f"oos_{size}.json")
    return [tuple(row[x]["q90"] for x in ("cost", "emission", "makespan"))
            for row in results["candidates"]]


def freeze_bounds(root, instances_dir, split, out):
    if out.exists():
        raise ValueError("normalisation already frozen")
    size = 1000 if split == "validation" else 5000
    stage = "validation_rule" if split == "validation" else "test_controls"
    bounds = dict(schema_version=1, reference=list(REF), source="Rule-only frozen fronts",
                  pad_fraction=.10, view_size=size, instances={})
    for path in sorted(instances_dir.glob("*.json")):
        if path.name == "catalogue.json":
            continue
        instance, _ = read_instance(path)
        if instance["split"] != split:
            continue
        sid = instance["instance_id"]
        runs = sorted((root / stage / sid / "rule").glob("run*/COMPLETE.json"))
        required = 5 if split == "validation" else 10
        if len(runs) != required:
            raise ValueError(f"{sid}: require {required} complete Rule runs")
        bounds["instances"][sid] = {}
        for view in ("in_sample", "oos"):
            points = [p for complete in runs for p in _points(complete.parent, view, size)]
            if not points:
                raise ValueError(f"{sid}: empty Rule front")
            a = np.asarray(points, dtype=float)
            span = np.ptp(a, axis=0)
            pad = .10 * np.where(span > 0, span, np.maximum(np.abs(a[0]), 1.))
            bounds["instances"][sid][view] = dict(
                minimum=(a.min(axis=0)-pad).tolist(), maximum=(a.max(axis=0)+pad).tolist())
    bounds["bounds_digest"] = hashlib.sha256(json.dumps(bounds, sort_keys=True).encode()).hexdigest()
    out.parent.mkdir(parents=True, exist_ok=True)
    write_json(out, bounds)
    return bounds


def hv(run, bounds, sid, view="oos", size=None):
    size = size or bounds["view_size"]
    points = np.asarray(_points(run, view, size), dtype=float).reshape((-1, 3))
    if not len(points):
        return 0.0
    b = bounds["instances"][sid][view]
    minimum, maximum = np.asarray(b["minimum"]), np.asarray(b["maximum"])
    normal = (points-minimum) / (maximum-minimum)
    return exact_hv_3d(normal, reference=REF)


def paired_values(root, bounds, label, size):
    values = []
    for sid in ("S8", "S9"):
        for repeat in range(1, 6):
            rule = root / "validation_rule" / sid / "rule" / f"run{repeat:02d}"
            learned = root / "validation_labels" / sid / f"{label}_eps0.1" / f"run{repeat:02d}"
            for path in (rule, learned):
                if not (path / f"oos_{size}" / f"oos_{size}.json").exists():
                    raise ValueError(f"missing frozen OOS evaluation: {path}")
            r = hv(rule, bounds, sid, size=size)
            l = hv(learned, bounds, sid, size=size)
            if r <= 0:
                raise ValueError("Rule HV is zero: relative 1% threshold undefined")
            values.append((l-r)/r)
    return values


def choose_label(root, bounds_file, models, out, size=1000, initial=None):
    if out.exists():
        raise ValueError("label choice already exists")
    bounds = _json(bounds_file)
    if size not in (1000, 5000):
        raise ValueError("validation size must be 1000 or 5000")
    labels = (_json(initial)["top_two"] if initial else LABELS)
    if size == 5000 and not initial:
        raise ValueError("5000-scenario decision needs frozen first-stage top two")
    comparison = {label: float(np.mean(paired_values(root, bounds, label, size)))
                  for label in labels}
    ordered = sorted(LABELS, key=lambda label: comparison[label], reverse=True)
    gap = comparison[ordered[0]] - comparison[ordered[1]]
    if gap < .01 and size == 1000:
        # This file identifies precisely which two fixed fronts to extend.
        result = dict(status="needs_5000_extension", top_two=ordered[:2],
                      paired_relative_hv_improvement=comparison,
                      threshold="0.01 difference in relative improvement over Rule")
    else:
        if gap < .01:
            # Tie-break by fixed in-sample convergence integral per evaluation.
            auc = {}
            for label in ordered[:2]:
                curves = []
                for sid in ("S8", "S9"):
                    for r in range(1, 6):
                        path = root / "validation_labels" / sid / f"{label}_eps0.1" / f"run{r:02d}"
                        generations = [json.loads(line) for line in
                            (path / "generation_summary.jsonl").read_text().splitlines()]
                        b = bounds["instances"][sid]["in_sample"]
                        lo, hi = np.asarray(b["minimum"]), np.asarray(b["maximum"])
                        xy = [(0, 0.)]
                        for row in generations:
                            front = np.asarray(row["training_front"], dtype=float).reshape((-1, 3))
                            xy.append((row["ccp_evaluation_count"],
                                exact_hv_3d((front-lo)/(hi-lo), reference=REF) if len(front) else 0.))
                        curves.append(float(np.trapz([v for _, v in xy], [x for x, _ in xy])
                                            / max(1, xy[-1][0])))
                auc[label] = float(np.mean(curves))
            if abs(auc[ordered[0]] - auc[ordered[1]]) < .01 * max(auc.values()):
                # When quality and convergence are close, choose simpler/faster.
                def timing(label):
                    return np.mean([_json(root / "validation_labels" / sid /
                        f"{label}_eps0.1" / f"run{r:02d}" / "COMPLETE.json")["total_seconds"]
                        for sid in ("S8", "S9") for r in range(1, 6)])
                winner = min(ordered[:2], key=lambda label: (timing(label), LABELS.index(label)))
            else:
                winner = max(ordered[:2], key=lambda label: auc[label])
        else:
            auc = None
            winner = ordered[0]
        model = models / f"{winner}.joblib"
        result = dict(status="label_frozen", label=winner,
                      model_sha256=hashlib.sha256(model.read_bytes()).hexdigest(),
                      paired_relative_hv_improvement=comparison,
                      convergence_auc=auc, comparison_size=size,
                      threshold="0.01 difference in relative improvement over Rule",
                      bounds_digest=bounds["bounds_digest"])
    out.parent.mkdir(parents=True, exist_ok=True)
    write_json(out, result)
    return result


def freeze_epsilon(root, bounds_file, label_choice, out):
    if out.exists():
        raise ValueError("policy lock already exists")
    bounds, choice = _json(bounds_file), _json(label_choice)
    if choice["status"] != "label_frozen":
        raise ValueError("first complete the label comparison")
    label = choice["label"]
    means = {}
    for epsilon, stage in ((.10, "validation_labels"), (0, "epsilon_zero")):
        differences = []
        for sid in ("S8", "S9"):
            for repeat in range(1, 6):
                learned = root / stage / sid / f"{label}_eps{epsilon:g}" / f"run{repeat:02d}"
                rule = root / "validation_rule" / sid / "rule" / f"run{repeat:02d}"
                r = hv(rule, bounds, sid)
                differences.append((hv(learned, bounds, sid)-r)/r)
        means[str(epsilon)] = float(np.mean(differences))
    epsilon = 0 if means["0"] > means["0.1"] else .10
    lock = dict(status="frozen", label=label, epsilon=epsilon,
                model_sha256=choice["model_sha256"],
                bounds_digest=bounds["bounds_digest"], validation_improvement=means)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_json(out, lock)
    return lock


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    f = sub.add_parser("freeze-bounds")
    f.add_argument("--root", type=Path, required=True)
    f.add_argument("--instances", type=Path, required=True)
    f.add_argument("--split", choices=("validation", "test"), required=True)
    f.add_argument("--out", type=Path, required=True)
    c = sub.add_parser("choose-label")
    c.add_argument("--root", type=Path, required=True)
    c.add_argument("--bounds", type=Path, required=True)
    c.add_argument("--models", type=Path, required=True)
    c.add_argument("--size", type=int, default=1000)
    c.add_argument("--initial", type=Path,
                   help="First-stage needs_5000_extension JSON; required for --size 5000")
    c.add_argument("--out", type=Path, required=True)
    e = sub.add_parser("freeze-epsilon")
    e.add_argument("--root", type=Path, required=True)
    e.add_argument("--bounds", type=Path, required=True)
    e.add_argument("--label-choice", type=Path, required=True)
    e.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = (freeze_bounds(args.root, args.instances, args.split, args.out)
              if args.command == "freeze-bounds" else
              choose_label(args.root, args.bounds, args.models, args.out,
                           args.size, args.initial)
              if args.command == "choose-label" else
              freeze_epsilon(args.root, args.bounds, args.label_choice, args.out))
    print({k: v for k, v in result.items() if k != "instances"})
