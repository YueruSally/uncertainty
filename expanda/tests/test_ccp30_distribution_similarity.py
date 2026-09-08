import numpy as np

import run_ccp30_distribution_similarity as similarity


def test_identical_empirical_distributions_have_zero_primary_distances():
    values = np.array([1.0, 2.0, 2.0, 4.0, 7.0])
    result = similarity.distribution_metrics(values, values, bins=5, smoothing=0.5)
    assert np.isclose(result["js_distance"], 0.0)
    assert np.isclose(result["wasserstein"], 0.0)
    assert np.isclose(result["wasserstein_normalized"], 0.0)
    assert np.isclose(result["ks_statistic"], 0.0)
    assert np.isclose(result["kl_sample_to_oos"], 0.0)
    assert np.isclose(result["kl_oos_to_sample"], 0.0)


def test_shifted_distributions_produce_positive_finite_distances():
    sample = np.array([0.0, 0.0, 0.0, 0.0])
    reference = np.array([1.0, 1.0, 1.0, 1.0])
    result = similarity.distribution_metrics(sample, reference, bins=4, smoothing=0.5)
    for metric in (
        "js_distance",
        "wasserstein",
        "wasserstein_normalized",
        "ks_statistic",
        "kl_sample_to_oos",
        "kl_oos_to_sample",
        "symmetric_kl",
    ):
        assert np.isfinite(result[metric])
        assert result[metric] > 0
    assert np.isclose(result["wasserstein"], 1.0)
    assert np.isclose(result["ks_statistic"], 1.0)


def test_histogram_edges_include_training_values_outside_oos_range():
    reference = np.linspace(0.0, 1.0, 101)
    sample = np.array([-10.0, 0.5, 10.0])
    result = similarity.distribution_metrics(sample, reference, bins=10, smoothing=0.5)
    assert np.isfinite(result["js_distance"])
    assert np.isfinite(result["symmetric_kl"])
    assert 0 <= result["js_distance"] <= np.sqrt(np.log(2.0))
