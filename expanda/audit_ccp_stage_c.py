#!/usr/bin/env python3
"""Create a non-destructive semantic supplement from the frozen EV/CCP pilot."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

import baseline_uncertainty as base


OBJECTIVES = ("cost", "emission", "makespan")
METHODS = ("EV", "CCP30", "CCP50")


def objective_summary(values: list[float], alpha: float) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
        "q90_empirical_order_statistic": base.empirical_ccp_quantile(
            array, alpha),
    }


def delta(validation: float, training: float) -> dict[str, float | None]:
    return {
        "absolute": float(validation - training),
        "relative_fraction": (
            float((validation - training) / training)
            if abs(training) > 1e-15 else None),
    }


def load_deadlines(workbook: Path, batch_ids: set[int]) -> dict[str, float]:
    frame = pd.read_excel(workbook, sheet_name="Batches")
    result = {}
    for _, row in frame.iterrows():
        batch_id = int(row["BatchID"])
        if batch_id not in batch_ids:
            continue
        latest_arrival = float(row["LT"])
        if not math.isfinite(latest_arrival) or latest_arrival <= 0.0:
            raise ValueError(
                f"batch {batch_id} has no genuine positive finite LT")
        result[str(batch_id)] = latest_arrival
    missing = batch_ids - {int(key) for key in result}
    if missing:
        raise ValueError(f"missing LT for batch IDs {sorted(missing)}")
    return result


def build_supplement(pilot: Path, out: Path, alpha: float = .90) -> dict[str, Any]:
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {out}")
    out.mkdir(parents=True, exist_ok=True)
    config = json.loads((pilot / "configuration.json").read_text())
    provenance = json.loads((pilot / "candidate_provenance.json").read_text())
    old_validation = json.loads(
        (pilot / "validation/per_candidate_summary.json").read_text())
    old_by_id = {row["source_solution_id"]: row for row in old_validation}

    realised: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {objective: [] for objective in OBJECTIVES})
    scenario_ids: dict[str, list[int]] = defaultdict(list)
    with (pilot / "validation/scenario_level_results.csv").open(
            newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            solution_id = row["source_solution_id"]
            scenario_ids[solution_id].append(int(row["scenario_id"]))
            for objective in OBJECTIVES:
                realised[solution_id][objective].append(float(row[objective]))

    batch_ids = {
        int(batch_id) for row in old_validation
        for batch_id in row["batch_on_time_probability"]
    }
    deadlines = load_deadlines(Path(config["data_file"]), batch_ids)
    candidate_results = []
    for candidate in provenance:
        solution_id = candidate["source_solution_id"]
        if scenario_ids[solution_id] != list(range(config["validation_size"])):
            raise ValueError(f"incomplete/non-common scenarios for {solution_id}")
        oos = {objective: objective_summary(
            realised[solution_id][objective], alpha)
               for objective in OBJECTIVES}
        training = candidate["optimisation_objectives"]
        comparison_basis = "OOS mean minus direct-expected-input objective"
        comparison = {
            objective: delta(oos[objective]["mean"], training[objective])
            for objective in OBJECTIVES}
        if candidate["method"].startswith("CCP"):
            comparison_basis = "OOS q90 minus training q90 objective"
            comparison = {
                objective: delta(
                    oos[objective]["q90_empirical_order_statistic"],
                    training[objective])
                for objective in OBJECTIVES}
        diagnostic = old_by_id[solution_id]["batch_on_time_probability"]
        emission_values = realised[solution_id]["emission"]
        emission_scale = max(1.0, max(abs(value) for value in emission_values))
        emission_constant = (
            max(emission_values) - min(emission_values)
            <= 1e-12 * emission_scale)
        candidate_results.append({
            "method": candidate["method"],
            "source_run": candidate["source_run"],
            "source_solution_id": solution_id,
            "decision_fingerprint": candidate["decision_fingerprint"],
            "training_objectives": training,
            "training_objective_semantics": (
                "direct expected-input objectives" if candidate["method"] == "EV"
                else "separate empirical q90 objective order statistics"),
            "oos_objectives": oos,
            "training_to_oos_comparison_basis": comparison_basis,
            "training_to_oos_delta": comparison,
            "punctuality_diagnostic": {
                "not_ccp_feasibility": True,
                "event": "realised batch completion time <= Batches.LT",
                "latest_arrival_by_batch_h": deadlines,
                "batch_on_time_probability": diagnostic,
                "minimum_batch_on_time_probability": min(diagnostic.values()),
                "all_batches_at_least_0_90": min(diagnostic.values()) >= alpha,
            },
            "emission_constant": emission_constant,
            "emission_identity_holds": emission_constant and bool(np.allclose(
                [oos["emission"]["mean"],
                 oos["emission"]["q90_empirical_order_statistic"],
                 oos["emission"]["minimum"], oos["emission"]["maximum"]],
                oos["emission"]["mean"], rtol=1e-12, atol=1e-6)),
        })

    method_summary = {}
    fingerprint_sets = {}
    for method in METHODS:
        rows = [row for row in candidate_results if row["method"] == method]
        fingerprint_sets[method] = {row["decision_fingerprint"] for row in rows}
        method_summary[method] = {
            "candidate_count": len(rows),
            "unique_decision_fingerprints": len(fingerprint_sets[method]),
            "candidates": [{
                "source_solution_id": row["source_solution_id"],
                "decision_fingerprint": row["decision_fingerprint"],
                "training_objectives": row["training_objectives"],
                "oos_mean": {o: row["oos_objectives"][o]["mean"]
                             for o in OBJECTIVES},
                "oos_q90": {
                    o: row["oos_objectives"][o][
                        "q90_empirical_order_statistic"]
                    for o in OBJECTIVES},
                "primary_generalisation_delta": row["training_to_oos_delta"],
                "punctuality_diagnostic_minimum": row[
                    "punctuality_diagnostic"][
                        "minimum_batch_on_time_probability"],
            } for row in rows],
        }
    cross_method = {
        f"{left}_vs_{right}_fingerprint_sets_differ": (
            fingerprint_sets[left] != fingerprint_sets[right])
        for index, left in enumerate(METHODS)
        for right in METHODS[index + 1:]
    }
    summary = {
        "status": "semantic correction supplement; frozen pilot not modified",
        "alpha": alpha,
        "q90_definition": "sorted(values)[ceil(alpha*S)-1]",
        "ccp_semantics": (
            "quantile-based objective optimisation; no punctuality chance "
            "constraint in the current executable model"),
        "method_summary": method_summary,
        "cross_method_decision_comparison": cross_method,
        "punctuality_is_diagnostic_not_feasibility": True,
        "all_candidates_have_constant_emission": all(
            row["emission_identity_holds"] for row in candidate_results),
    }
    (out / "per_candidate_corrected_validation.json").write_text(
        json.dumps(candidate_results, indent=2), encoding="utf-8")
    (out / "corrected_method_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    (out / "README.md").write_text(
        "# Frozen pilot semantic-audit supplement\n\n"
        "This directory was computed from the already frozen pilot CSV and "
        "candidate provenance. No optimisation or candidate mutation was run, "
        "and the original pilot directory was not modified.\n\n"
        "The current executable CCP minimises three separate empirical "
        "order-statistic objectives. For objective samples z_1,...,z_S and "
        "alpha=0.90, q_alpha = sorted(z)[ceil(alpha*S)-1]. Alpha does not "
        "currently define a punctuality feasibility constraint.\n\n"
        "`Batches.LT` is a genuine latest-arrival input for every pilot batch. "
        "The retained frequency of completion <= LT is reported only as a "
        "punctuality diagnostic. It is not CCP feasibility.\n\n"
        "The preserved S200 run used population 300, generations 500, S=200, "
        "and one pilot run. No full or formal run was launched here.\n",
        encoding="utf-8")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", default="pilot_ev_ccp_oos")
    parser.add_argument("--out", default="pilot_ev_ccp_oos_semantic_audit")
    parser.add_argument("--alpha", type=float, default=.90)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = build_supplement(Path(args.pilot), Path(args.out), args.alpha)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
