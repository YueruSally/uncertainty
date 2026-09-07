import numpy as np

from run_ccp_s30_s50_selection import (
    DEFAULT_PAIRS, DEFAULT_VALIDATION_SEED, exact_hypervolume_3d,
    percent_errors, validated_quality,
)


def _summary(method, training, validation):
    return {
        "method": method,
        "training_objectives": dict(zip(("cost", "emission", "makespan"), training)),
        "cost": {"q90": validation[0]},
        "emission": {"q90": validation[1]},
        "makespan": {"q90": validation[2]},
    }


def test_predeclared_seeds_are_unique_and_paired_by_design():
    seeds = [x for pair in DEFAULT_PAIRS for x in pair] + [DEFAULT_VALIDATION_SEED]
    assert len(DEFAULT_PAIRS) == 5
    assert len(seeds) == len(set(seeds))


def test_exact_hypervolume_3d_known_boxes():
    assert np.isclose(exact_hypervolume_3d([(0.2, 0.2, 0.2)]), 1.0)
    # Two boxes whose union has volume 0.048 + 0.048 - 0.008.
    assert np.isclose(exact_hypervolume_3d([(0, 1, 1), (1, 0, 1)]), 0.088)


def test_absolute_percentage_error_robust_summaries():
    rows = [_summary("CCP30", (100, 2, 50), (110, 2, 55)),
            _summary("CCP30", (100, 2, 50), (120, 2, 60))]
    result = percent_errors(rows, "cost")
    assert result["median_absolute_percentage_error"] == 15.0
    assert result["mean_absolute_percentage_error"] == 15.0
    assert result["maximum_absolute_percentage_error"] == 20.0


def test_quality_uses_one_common_scaling_and_pooled_reference():
    rows = {
        "CCP30": [_summary("CCP30", (1, 1, 1), (1, 3, 3)),
                  _summary("CCP30", (1, 1, 1), (3, 1, 3))],
        "CCP50": [_summary("CCP50", (1, 1, 1), (3, 3, 1))],
    }
    quality, scaling = validated_quality(rows)
    assert scaling["ideal"] == [1.0, 1.0, 1.0]
    assert scaling["nadir"] == [3.0, 3.0, 3.0]
    assert scaling["pooled_reference_size"] == 3
    assert quality["CCP30"]["validated_nondominated_size"] == 2
    assert quality["CCP50"]["validated_nondominated_size"] == 1
