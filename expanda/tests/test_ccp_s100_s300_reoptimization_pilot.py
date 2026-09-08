import numpy as np

import run_ccp_s100_s300_reoptimization_pilot as pilot


def test_exact_hypervolume_single_origin_point():
    assert np.isclose(pilot.exact_hypervolume_3d([[0.0, 0.0, 0.0]]), 1.2 ** 3)


def test_coverage_diagnostic_exact_ninety_percent():
    values = np.arange(1.0, 11.0)
    result = pilot.coverage_diagnostic(values, training_q90=9.0, alpha=0.9)
    assert np.isclose(result["coverage"], 0.9)
    assert np.isclose(result["exceedance_rate"], 0.1)
    assert np.isclose(result["coverage_signed_gap_pp"], 0.0)
    assert np.isclose(result["coverage_absolute_gap_pp"], 0.0)


def test_coverage_diagnostic_optimistic_threshold():
    values = np.arange(1.0, 11.0)
    result = pilot.coverage_diagnostic(values, training_q90=8.0, alpha=0.9)
    assert np.isclose(result["coverage"], 0.8)
    assert np.isclose(result["coverage_signed_gap_pp"], -10.0)
    assert np.isclose(result["coverage_absolute_gap_pp"], 10.0)


def test_numeric_summary_ignores_missing_values():
    result = pilot.numeric_summary([1.0, None, 3.0])
    assert np.isclose(result["mean"], 2.0)
    assert np.isclose(result["median"], 2.0)
    assert np.isclose(result["minimum"], 1.0)
    assert np.isclose(result["maximum"], 3.0)


def test_build_paired_rows_uses_s300_minus_s100_direction():
    shared = {
        "candidate_count": 10,
        "oos_mean_igd_plus": 0.2,
        "oos_q90_hypervolume": 0.6,
        "oos_q90_igd_plus": 0.3,
        "all_objectives_mean_absolute_coverage_gap_pp": 2.0,
        "cost_mean_coverage": 0.89,
        "emission_mean_coverage": 0.90,
        "makespan_mean_coverage": 0.91,
    }
    left = {
        **shared,
        "replicate": 1,
        "method": "CCP100",
        "sample_size": 100,
        "optimisation_runtime_seconds": 10.0,
        "oos_mean_hypervolume": 0.5,
    }
    right = {
        **shared,
        "replicate": 1,
        "method": "CCP300",
        "sample_size": 300,
        "optimisation_runtime_seconds": 30.0,
        "oos_mean_hypervolume": 0.8,
    }
    row = pilot.build_paired_rows([left, right], 1)[0]
    assert np.isclose(row["oos_mean_hypervolume_S300_minus_S100"], 0.3)
    assert np.isclose(row["optimisation_runtime_seconds_S300_minus_S100"], 20.0)

