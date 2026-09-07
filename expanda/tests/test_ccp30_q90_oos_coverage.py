import numpy as np

import run_ccp30_q90_oos_coverage as coverage


def test_coverage_uses_inclusive_threshold_and_strict_exceedance(monkeypatch):
    monkeypatch.setattr(coverage, "OOS_SCENARIOS", 5)
    source = {"source_solution_id": "s1", "optimisation_objectives": {
        "cost": 2, "emission": 2, "makespan": 2}}
    row = coverage.coverage_row(source, ([1, 2, 2, 3, 4],) * 3)
    for objective in coverage.OBJECTIVES:
        assert row[f"oos_coverage_{objective}"] == .6
        assert row[f"oos_exceedance_{objective}"] == .4
        assert row[f"oos_coverage_{objective}"] + row[
            f"oos_exceedance_{objective}"] == 1


def test_summary_counts_below_above_and_exact_target():
    rows = []
    for value in (.89, .90, .91):
        row = {"solution_id": str(value)}
        for objective in coverage.OBJECTIVES:
            row[f"oos_coverage_{objective}"] = value
            row[f"oos_exceedance_{objective}"] = 1-value
        rows.append(row)
    result = coverage.summarise(rows)
    assert result["cost"]["coverage_below_0_90_count"] == 1
    assert result["cost"]["coverage_above_0_90_count"] == 1
    assert result["cost"]["coverage_exactly_0_90_count"] == 1
    assert np.isclose(result["cost"]["mean_oos_coverage"], .9)
