#!/usr/bin/env python3
"""Freeze an offline-validated segment model for CCP100 pilot use."""

import argparse
import hashlib
import json
from pathlib import Path

import joblib

from mutation_logging import write_json


OBJECTIVE_HEADS = ["cost_improved", "emission_improved", "makespan_improved"]
RISK_HEAD = "capacity_worsened"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        parser.error("refusing to overwrite output model")
    artifact = joblib.load(args.model)
    report = json.loads(args.metrics.read_text())
    if artifact.get("schema_version") != 2:
        raise ValueError("Expected segment model schema 2")
    if report.get("oos_used") is not False:
        raise ValueError("OOS data must not be used for model promotion")
    if set(report["train_runs"]) & set(report["validation_runs"]):
        raise ValueError("Training and validation runs overlap")
    required = OBJECTIVE_HEADS + [RISK_HEAD]
    for target in required:
        if target not in artifact.get("pipelines", {}):
            raise ValueError(f"Missing fitted pipeline for {target}")
        if report["targets"].get(target, {}).get("status") != "trained":
            raise ValueError(f"Target {target} was not successfully trained")
    source_sha = sha256(args.model)
    artifact.update({
        "deployment_status": "pilot_only",
        "source_offline_model_sha256": source_sha,
        "policy": {
            "name": "learning-segment-objective-mixture",
            "guided_operator": "mode",
            "objective_heads": OBJECTIVE_HEADS,
            "objective_head_sampling": "uniform-per-mode-event",
            "risk_head": RISK_HEAD,
            "score_formula": "P(objective improvement) * (1 - P(capacity worsened))",
            "non_mode_policy": "rule-eligible",
            "oos_used": False,
        },
    })
    args.out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, args.out)
    manifest = {
        "schema_version": 1, "promoted_model": str(args.out),
        "promoted_model_sha256": sha256(args.out),
        "source_offline_model_sha256": source_sha,
        "metrics_sha256": sha256(args.metrics), "policy": artifact["policy"],
        "train_runs": report["train_runs"],
        "validation_runs": report["validation_runs"],
    }
    write_json(args.out.with_suffix(".manifest.json"), manifest)
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
