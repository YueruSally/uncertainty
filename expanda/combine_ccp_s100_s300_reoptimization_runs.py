#!/usr/bin/env python3
"""Combine the completed 3-run and 2-run CCP100/CCP300 pilots.

The script performs no optimisation and no scenario evaluation. It joins the
saved per-candidate OOS summaries and recomputes one global OOS reference front,
normalisation, HV and IGD+ across all five paired replicates.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from types import SimpleNamespace

import run_ccp_s100_s300_reoptimization_pilot as pilot


ROOT = Path(__file__).resolve().parent
DEFAULT_FIRST = ROOT / "ccp_s100_s300_reoptimization_pilot_3runs"
DEFAULT_SECOND = ROOT / "ccp_s100_s300_reoptimization_additional_runs_04_05"
DEFAULT_OUT = ROOT / "ccp_s100_s300_reoptimization_pilot_5runs_combined"


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_candidate_csv(path: Path) -> list[dict]:
    rows = []
    with path.open(newline="", encoding="utf-8") as handle:
        for source in csv.DictReader(handle):
            row = {
                "replicate": int(source["replicate"]),
                "method": source["method"],
                "training_sample_size": int(source["training_sample_size"]),
                "source_solution_id": source["source_solution_id"],
                "decision_fingerprint": source["decision_fingerprint"],
            }
            for objective in pilot.OBJECTIVES:
                row[objective] = {}
                for field in (
                    "mean",
                    "q90",
                    "coverage",
                    "exceedance_rate",
                    "coverage_signed_gap_pp",
                    "coverage_absolute_gap_pp",
                    "training_q90",
                    "q90_signed_error_pct",
                    "q90_absolute_error_pct",
                ):
                    raw = source[f"{objective}_{field}"]
                    row[objective][field] = None if raw == "" else float(raw)
            rows.append(row)
    if not rows:
        raise RuntimeError(f"No candidate rows in {path}")
    return rows


def core_input_hashes(config: dict) -> list[str]:
    """Return model/data hashes while allowing the driver script to change."""
    hashes = config.get("input_sha256", {})
    if not isinstance(hashes, dict):
        raise RuntimeError("configuration input_sha256 must be a mapping")
    return sorted(
        str(value)
        for key, value in hashes.items()
        if not str(key).endswith("run_ccp_s100_s300_reoptimization_pilot.py")
    )


def validate_sources(first: Path, second: Path) -> tuple[list[dict], dict, dict]:
    complete_a = read_json(first / "COMPLETE.json")
    complete_b = read_json(second / "COMPLETE.json")
    config_a = read_json(first / "configuration.json")
    config_b = read_json(second / "configuration.json")
    for label, complete in (("first", complete_a), ("second", complete_b)):
        if not complete.get("completed"):
            raise RuntimeError(f"{label} experiment is not complete")
        if complete.get("post_oos_candidate_filtering") is not False:
            raise RuntimeError(f"{label} experiment used post-OOS filtering")
        if complete.get("all_decision_fingerprints_unchanged") is not True:
            raise RuntimeError(f"{label} experiment failed decision fingerprint audit")
    frozen_fields = (
        "population",
        "generations",
        "alpha",
        "training_sizes",
        "validation_size",
        "validation_seed",
        "validation_digest",
        "nested_training_prefixes",
        "paired_algorithm_seeds",
        "path_library_seed",
        "common_oos_for_all_candidates",
        "post_oos_candidate_filtering",
        "scenario_level_results_written",
        "operators",
        "waiting",
    )
    mismatches = [
        field for field in frozen_fields if config_a.get(field) != config_b.get(field)
    ]
    if mismatches:
        raise RuntimeError(f"Pilot configurations differ: {mismatches}")
    if core_input_hashes(config_a) != core_input_hashes(config_b):
        raise RuntimeError("Pilot configurations use different model/data input hashes")

    rows_a = load_candidate_csv(first / "validation/per_candidate_summary.csv")
    rows_b = load_candidate_csv(second / "validation/per_candidate_summary.csv")
    rows = rows_a + rows_b
    replicate_ids = sorted({row["replicate"] for row in rows})
    if replicate_ids != [1, 2, 3, 4, 5]:
        raise RuntimeError(f"Expected exact replicate IDs 1..5, got {replicate_ids}")
    keys = [pilot.validation_key(row) for row in rows]
    if len(keys) != len(set(keys)):
        raise RuntimeError("Duplicate candidate key across the two source experiments")
    for replicate in replicate_ids:
        if {row["method"] for row in rows if row["replicate"] == replicate} != set(
            pilot.METHODS
        ):
            raise RuntimeError(f"Replicate {replicate} does not contain both methods")
    if len(rows) != complete_a["candidate_count"] + complete_b["candidate_count"]:
        raise RuntimeError("Combined candidate count differs from completion markers")
    return rows, config_a, {
        "first": complete_a,
        "second": complete_b,
        "replicate_ids": replicate_ids,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first", type=Path, default=DEFAULT_FIRST)
    parser.add_argument("--second", type=Path, default=DEFAULT_SECOND)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output: {args.out}")
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "metrics").mkdir()

    summaries, config, audit = validate_sources(args.first, args.second)
    runtimes = {}
    for source in (args.first, args.second):
        source_runtime = read_json(source / "runtime_summary.json")
        for key, value in source_runtime.items():
            if key.endswith("_optimisation_seconds"):
                if key in runtimes:
                    raise RuntimeError(f"Duplicate runtime key: {key}")
                runtimes[key] = float(value)

    quality, scaling = pilot.validated_quality(summaries)
    replicate_ids = audit["replicate_ids"]
    run_rows = pilot.build_run_rows(
        summaries, quality, runtimes, replicate_ids, float(config["alpha"])
    )
    paired_rows = pilot.build_paired_rows(run_rows, replicate_ids)
    aggregate = pilot.aggregate_results(run_rows, paired_rows)

    pilot.write_csv(args.out / "metrics/per_run_summary.csv", run_rows)
    pilot.write_csv(args.out / "metrics/paired_comparison.csv", paired_rows)
    pilot.atomic_json(args.out / "metrics/aggregate_summary.json", aggregate)
    pilot.atomic_json(args.out / "metrics/normalisation.json", scaling)
    pilot.plot_paired_metrics(args.out, run_rows)
    pilot.plot_coverage(args.out, run_rows, float(config["alpha"]))
    report_args = SimpleNamespace(
        replicates=5,
        pop=int(config["population"]),
        gens=int(config["generations"]),
        validation_scenarios=int(config["validation_size"]),
    )
    pilot.write_report(args.out, run_rows, aggregate, report_args)
    pilot.atomic_json(
        args.out / "SOURCE_AUDIT.json",
        {
            "first_source": str(args.first),
            "second_source": str(args.second),
            "first_complete": audit["first"],
            "second_complete": audit["second"],
            "replicate_ids": replicate_ids,
            "combined_candidate_count": len(summaries),
            "common_validation_seed": config["validation_seed"],
            "common_validation_digest": config["validation_digest"],
            "global_metrics_recomputed": True,
            "new_optimisation": False,
            "new_scenario_evaluation": False,
        },
    )
    pilot.atomic_json(
        args.out / "COMPLETE.json",
        {
            "replicates": 5,
            "replicate_ids": replicate_ids,
            "methods": list(pilot.METHODS),
            "combined_candidate_count": len(summaries),
            "one_global_oos_reference_front_per_view": True,
            "one_global_normalisation_per_view": True,
            "post_oos_candidate_filtering": False,
            "descriptive_not_inferential": True,
            "completed": True,
        },
    )
    print((args.out / "REPORT.md").read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
