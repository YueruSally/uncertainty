import csv
import json
from pathlib import Path

import pytest

import combine_ccp_s100_s300_reoptimization_runs as combine


def _write_source(root: Path, replicate_ids, digest="same"):
    (root / "validation").mkdir(parents=True)
    config = {
        "population": 100,
        "generations": 100,
        "alpha": 0.9,
        "training_sizes": [100, 300],
        "validation_size": 5000,
        "validation_seed": 980999,
        "validation_digest": digest,
        "nested_training_prefixes": True,
        "paired_algorithm_seeds": True,
        "path_library_seed": 0,
        "common_oos_for_all_candidates": True,
        "post_oos_candidate_filtering": False,
        "scenario_level_results_written": False,
        "operators": {"crossover_rate": 0.9, "mutation_rate": 0.15},
        "waiting": {"value": 1},
    }
    (root / "configuration.json").write_text(json.dumps(config))
    count = len(replicate_ids) * 2
    complete = {
        "completed": True,
        "post_oos_candidate_filtering": False,
        "all_decision_fingerprints_unchanged": True,
        "candidate_count": count,
    }
    (root / "COMPLETE.json").write_text(json.dumps(complete))
    fields = ["replicate", "method", "training_sample_size", "source_solution_id",
              "decision_fingerprint"]
    for objective in ("cost", "emission", "makespan"):
        fields += [f"{objective}_{field}" for field in (
            "mean", "q90", "coverage", "exceedance_rate",
            "coverage_signed_gap_pp", "coverage_absolute_gap_pp",
            "training_q90", "q90_signed_error_pct", "q90_absolute_error_pct")]
    with (root / "validation/per_candidate_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for replicate in replicate_ids:
            for method, size in (("CCP100", 100), ("CCP300", 300)):
                row = {
                    "replicate": replicate,
                    "method": method,
                    "training_sample_size": size,
                    "source_solution_id": f"r{replicate}-{method}",
                    "decision_fingerprint": f"f{replicate}-{method}",
                }
                for objective in ("cost", "emission", "makespan"):
                    row.update({
                        f"{objective}_mean": 1,
                        f"{objective}_q90": 2,
                        f"{objective}_coverage": .9,
                        f"{objective}_exceedance_rate": .1,
                        f"{objective}_coverage_signed_gap_pp": 0,
                        f"{objective}_coverage_absolute_gap_pp": 0,
                        f"{objective}_training_q90": 2,
                        f"{objective}_q90_signed_error_pct": 0,
                        f"{objective}_q90_absolute_error_pct": 0,
                    })
                writer.writerow(row)


def test_validate_sources_accepts_disjoint_complete_five_runs(tmp_path):
    first, second = tmp_path / "first", tmp_path / "second"
    _write_source(first, [1, 2, 3])
    _write_source(second, [4, 5])
    rows, config, audit = combine.validate_sources(first, second)
    assert len(rows) == 10
    assert audit["replicate_ids"] == [1, 2, 3, 4, 5]
    assert config["validation_digest"] == "same"


def test_validate_sources_rejects_different_oos_digest(tmp_path):
    first, second = tmp_path / "first", tmp_path / "second"
    _write_source(first, [1, 2, 3], digest="a")
    _write_source(second, [4, 5], digest="b")
    with pytest.raises(RuntimeError, match="configurations differ"):
        combine.validate_sources(first, second)

