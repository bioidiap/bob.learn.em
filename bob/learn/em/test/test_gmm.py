#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
# Thu Feb 16 17:57:10 2012 +0200
#
# Copyright (C) 2011-2014 Idiap Research Institute, Martigny, Switzerland

"""Tests the GMM machine and the GMMStats container
"""

import contextlib
import os
import tempfile

from copy import deepcopy

import numpy as np

from dask.distributed import Client
from h5py import File as HDF5File
from pkg_resources import resource_filename

from bob.learn.em import GMMMachine, GMMStats, KMeansMachine
from bob.learn.em import gmm as gmm_module

from .test_kmeans import to_dask_array, to_numpy


def load_array(filename):
    with HDF5File(filename, "r") as f:
        array = np.array(f[list(f.keys())[0]])
    # remove dimensions with size 1
    return np.squeeze(array)


@contextlib.contextmanager
def multiprocess_dask_client():
    try:
        client = Client()
        with client.as_current():
            yield client
    finally:
        client.close()


def loadGMM():
    gmm = GMMMachine(n_gaussians=2)

    gmm.weights = load_array(
        resource_filename("bob.learn.em", "data/gmm.init_weights.hdf5")
    )
    gmm.means = load_array(
        resource_filename("bob.learn.em", "data/gmm.init_means.hdf5")
    )
    gmm.variances = load_array(
        resource_filename("bob.learn.em", "data/gmm.init_variances.hdf5")
    )

    return gmm


def assert_gmm_equal(gmm1, gmm2):
    """Asserts that two GMMs are equal"""
    np.testing.assert_almost_equal(gmm1.weights, gmm2.weights)
    np.testing.assert_almost_equal(gmm1.means, gmm2.means)
    np.testing.assert_almost_equal(gmm1.variances, gmm2.variances)
    np.testing.assert_almost_equal(
        gmm1.variance_thresholds, gmm2.variance_thresholds
    )
    assert gmm1 == gmm2


def test_GMMStats():
    # Test a GMMStats
    # Initializes a GMMStats
    n_gaussians = 2
    n_features = 3
    gs = GMMStats(n_gaussians, n_features)
    log_likelihood = -3.0
    T = 57
    n = np.array([4.37, 5.31], "float64")
    sumpx = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], "float64")
    sumpxx = np.array([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]], "float64")
    gs.log_likelihood = log_likelihood
    gs.t = T
    gs.n = n
    gs.sum_px = sumpx
    gs.sum_pxx = sumpxx
    np.testing.assert_equal(gs.log_likelihood, log_likelihood)
    np.testing.assert_equal(gs.t, T)
    np.testing.assert_equal(gs.n, n)
    np.testing.assert_equal(gs.sum_px, sumpx)
    np.testing.assert_equal(gs.sum_pxx, sumpxx)
    np.testing.assert_equal(gs.shape, (n_gaussians, n_features))

    # Saves and reads from file using `from_hdf5`
    filename = str(tempfile.mkstemp(".hdf5")[1])
    gs.save(HDF5File(filename, "w"))
    gs_loaded = GMMStats.from_hdf5(HDF5File(filename, "r"))
    assert gs == gs_loaded
    assert (gs != gs_loaded) is False
    assert gs.is_similar_to(gs_loaded)

    assert type(gs_loaded.n_gaussians) is np.int64
    assert type(gs_loaded.n_features) is np.int64
    assert type(gs_loaded.log_likelihood) is np.float64

    # Saves and load from file using `load`
    filename = str(tempfile.mkstemp(".hdf5")[1])
    gs.save(hdf5=HDF5File(filename, "w"))
    gs_loaded = GMMStats(n_gaussians, n_features)
    gs_loaded.load(HDF5File(filename, "r"))
    assert gs == gs_loaded
    assert (gs != gs_loaded) is False
    assert gs.is_similar_to(gs_loaded)

    # Makes them different
    gs_loaded.t = 58
    assert (gs == gs_loaded) is False
    assert gs != gs_loaded
    assert not (gs.is_similar_to(gs_loaded))

    # Accumulates from another GMMStats
    gs2 = GMMStats(n_gaussians, n_features)
    gs2.log_likelihood = log_likelihood
    gs2.t = T
    gs2.n = n.copy()
    gs2.sum_px = sumpx.copy()
    gs2.sum_pxx = sumpxx.copy()
    gs2 += gs
    np.testing.assert_equal(gs2.log_likelihood, 2 * log_likelihood)
    np.testing.assert_equal(gs2.t, 2 * T)
    np.testing.assert_almost_equal(gs2.n, 2 * n, decimal=8)
    np.testing.assert_almost_equal(gs2.sum_px, 2 * sumpx, decimal=8)
    np.testing.assert_almost_equal(gs2.sum_pxx, 2 * sumpxx, decimal=8)

    # Re-init and checks for zeros
    gs_loaded.init_fields()
    np.testing.assert_equal(gs_loaded.log_likelihood, 0)
    np.testing.assert_equal(gs_loaded.t, 0)
    np.testing.assert_equal(gs_loaded.n, np.zeros((n_gaussians,)))
    np.testing.assert_equal(
        gs_loaded.sum_px, np.zeros((n_gaussians, n_features))
    )
    np.testing.assert_equal(
        gs_loaded.sum_pxx, np.zeros((n_gaussians, n_features))
    )
    # Resize and checks size
    assert gs_loaded.shape == (n_gaussians, n_features)
    gs_loaded.resize(4, 5)
    assert gs_loaded.shape == (4, 5)
    assert gs_loaded.sum_px.shape[0] == 4
    assert gs_loaded.sum_px.shape[1] == 5

    # Clean-up
    os.unlink(filename)


def test_GMMMachine():
    # Test a GMMMachine basic features

    weights = np.array([0.5, 0.5], "float64")
    weights2 = np.array([0.6, 0.4], "float64")
    means = np.array([[3, 70, 0], [4, 72, 0]], "float64")
    means2 = np.array([[3, 7, 0], [4, 72, 0]], "float64")
    variances = np.array([[1, 10, 1], [2, 5, 2]], "float64")
    variances2 = np.array([[10, 10, 1], [2, 5, 2]], "float64")
    varianceThresholds = np.array([[0, 0, 0], [0, 0, 0]], "float64")
    varianceThresholds2 = np.array(
        [[0.0005, 0.0005, 0.0005], [0, 0, 0]], "float64"
    )

    # Initializes a GMMMachine
    gmm = GMMMachine(n_gaussians=2)
    # Sets the weights, means, variances and varianceThresholds and
    # Checks correctness
    gmm.weights = weights
    gmm.means = means
    gmm.variances = variances
    gmm.variance_thresholds = varianceThresholds
    assert gmm.shape == (2, 3)
    np.testing.assert_equal(gmm.weights, weights)
    np.testing.assert_equal(gmm.means, means)
    np.testing.assert_equal(gmm.variances, variances)
    np.testing.assert_equal(gmm.variance_thresholds, varianceThresholds)

    newMeans = np.array([[3, 70, 2], [4, 72, 2]], "float64")
    newVariances = np.array([[1, 1, 1], [2, 2, 2]], "float64")

    # Checks particular varianceThresholds-related methods
    varianceThresholds1D = np.array([0.3, 1, 0.5], "float64")
    gmm.variance_thresholds = varianceThresholds1D
    np.testing.assert_equal(gmm.variance_thresholds, varianceThresholds1D)

    gmm.variance_thresholds = 0.005
    np.testing.assert_equal(gmm.variance_thresholds, 0.005)

    gmm.means = newMeans
    gmm.variances = newVariances
    np.testing.assert_equal(gmm.means, newMeans)
    np.testing.assert_equal(gmm.variances, newVariances)

    # Checks comparison
    gmm2 = deepcopy(gmm)
    gmm3 = GMMMachine(n_gaussians=2)
    gmm3.weights = weights2
    gmm3.means = means
    gmm3.variances = variances
    gmm3.variance_thresholds = varianceThresholds
    gmm4 = GMMMachine(n_gaussians=2)
    gmm4.weights = weights
    gmm4.means = means2
    gmm4.variances = variances
    gmm4.variance_thresholds = varianceThresholds
    gmm5 = GMMMachine(n_gaussians=2)
    gmm5.weights = weights
    gmm5.means = means
    gmm5.variances = variances2
    gmm5.variance_thresholds = varianceThresholds
    gmm6 = GMMMachine(n_gaussians=2)
    gmm6.weights = weights
    gmm6.means = means
    gmm6.variances = variances
    gmm6.variance_thresholds = varianceThresholds2

    assert_gmm_equal(gmm, gmm2)
    assert (gmm != gmm2) is False
    assert gmm.is_similar_to(gmm2)
    assert gmm != gmm3
    assert gmm.is_similar_to(gmm3) is False
    assert gmm != gmm4
    assert gmm.is_similar_to(gmm4) is False
    assert gmm != gmm5
    assert gmm.is_similar_to(gmm5) is False
    assert gmm != gmm6
    assert gmm.is_similar_to(gmm6) is False

    # Saving and loading
    with tempfile.NamedTemporaryFile(suffix=".hdf5") as f:
        filename = f.name
        gmm.save(HDF5File(filename, "w"))
        # Using from_hdf5
        gmm1 = GMMMachine.from_hdf5(HDF5File(filename, "r"))
        assert type(gmm1.n_gaussians) is np.int64
        assert type(gmm1.update_means) is np.bool_
        assert type(gmm1.update_variances) is np.bool_
        assert type(gmm1.update_weights) is np.bool_
        assert type(gmm1.trainer) is str
        assert gmm1.ubm is None
        assert_gmm_equal(gmm, gmm1)
        # Using load
        gmm1 = GMMMachine(n_gaussians=gmm.n_gaussians)
        gmm1.load(HDF5File(filename, "r"))
        assert type(gmm1.n_gaussians) is np.int64
        assert type(gmm1.update_means) is np.bool_
        assert type(gmm1.update_variances) is np.bool_
        assert type(gmm1.update_weights) is np.bool_
        assert type(gmm1.trainer) is str
        assert gmm1.ubm is None
        assert_gmm_equal(gmm, gmm1)

    with tempfile.NamedTemporaryFile(suffix=".hdf5") as f:
        filename = f.name
        gmm.save(filename)
        gmm1 = GMMMachine.from_hdf5(filename)
        assert_gmm_equal(gmm, gmm1)

    # Weights
    n_gaussians = 5
    machine = GMMMachine(n_gaussians)

    default_weights = np.full(
        shape=(n_gaussians,), fill_value=1.0 / n_gaussians
    )
    default_log_weights = np.full(
        shape=(n_gaussians,), fill_value=np.log(1.0 / n_gaussians)
    )

    # Test weights getting and setting
    np.testing.assert_almost_equal(machine.weights, default_weights)
    np.testing.assert_almost_equal(machine.log_weights, default_log_weights)

    modified_weights = default_weights
    modified_weights[: n_gaussians // 2] = (1 / n_gaussians) / 2
    modified_weights[n_gaussians // 2 + n_gaussians % 2 :] = (
        1 / n_gaussians
    ) * 1.5

    # Ensure setter works (log_weights is updated correctly)
    machine.weights = modified_weights
    np.testing.assert_almost_equal(machine.weights, modified_weights)
    np.testing.assert_almost_equal(
        machine.log_weights, np.log(modified_weights)
    )


def test_GMMMachine_stats():
    """Tests a GMMMachine (statistics)"""

    arrayset = load_array(
        resource_filename("bob.learn.em", "data/faithful.torch3_f64.hdf5")
    )
    gmm = GMMMachine(n_gaussians=2)
    gmm.weights = np.array([0.5, 0.5], "float64")
    gmm.means = np.array([[3, 70], [4, 72]], "float64")
    gmm.variances = np.array([[1, 10], [2, 5]], "float64")
    gmm.variance_thresholds = np.array([[0, 0], [0, 0]], "float64")

    stats = gmm_module.e_step(
        arrayset,
        gmm,
    )

    stats_ref = GMMStats(n_gaussians=2, n_features=2)
    stats_ref.load(
        HDF5File(resource_filename("bob.learn.em", "data/stats.hdf5"), "r")
    )

    np.testing.assert_equal(stats.t, stats_ref.t)
    np.testing.assert_almost_equal(stats.n, stats_ref.n, decimal=10)
    # np.testing.assert_equal(stats.sum_px, stats_ref.sum_px)
    # Note AA: precision error above
    np.testing.assert_almost_equal(stats.sum_px, stats_ref.sum_px, decimal=10)
    np.testing.assert_almost_equal(stats.sum_pxx, stats_ref.sum_pxx, decimal=10)


def test_GMMMachine_ll_computation():
    """Test a GMMMachine (log-likelihood computation)"""

    data = load_array(resource_filename("bob.learn.em", "data/data.hdf5"))
    gmm = GMMMachine(n_gaussians=2)
    gmm.weights = load_array(
        resource_filename("bob.learn.em", "data/weights.hdf5")
    )
    gmm.means = load_array(resource_filename("bob.learn.em", "data/means.hdf5"))
    gmm.variances = load_array(
        resource_filename("bob.learn.em", "data/variances.hdf5")
    )

    # Compare the log-likelihood with the one obtained using Chris Matlab implementation
    matlab_ll_ref = -2.361583051672024e02
    np.testing.assert_almost_equal(
        gmm.log_likelihood(data), matlab_ll_ref, decimal=10
    )


def test_GMMMachine_single_ll_vs_multiple():

    np.random.seed(3)  # FIXING A SEED

    data = np.random.rand(
        100, 50
    )  # Doesn't matter if it is random. The average of 1D array (in python) MUST output the same result for the 2D array (in C++)

    gmm = GMMMachine(n_gaussians=2)
    gmm.weights = load_array(
        resource_filename("bob.learn.em", "data/weights.hdf5")
    )
    gmm.means = load_array(resource_filename("bob.learn.em", "data/means.hdf5"))
    gmm.variances = load_array(
        resource_filename("bob.learn.em", "data/variances.hdf5")
    )

    ll = 0
    for i in range(data.shape[0]):
        ll += gmm.log_likelihood(data[i, :])
    ll /= data.shape[0]

    assert np.isclose(ll, gmm.log_likelihood(data).mean())


def test_GMMStats_operations():
    """Test a GMMStats."""
    # Initializing a GMMStats
    data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [7, 8, 9]])
    n_gaussians = 2
    n_features = data.shape[-1]
    machine = GMMMachine(n_gaussians)

    machine.means = np.array([[0, 0, 0], [8, 8, 8]])
    machine.variances = np.ones_like(machine.means)

    # Populate the GMMStats
    stats = gmm_module.e_step(
        data,
        machine,
    )

    # Check shapes
    assert stats.n.shape == (n_gaussians,), stats.n.shape
    assert stats.sum_px.shape == (n_gaussians, n_features), stats.sum_px.shape
    assert stats.sum_pxx.shape == (n_gaussians, n_features), stats.sum_pxx.shape

    # Check values
    expected_ll = -37.2998511206581
    expected_n = np.array([1, 3])
    expected_sumPx = np.array([[1, 2, 3], [18, 21, 24]])
    expected_sumPxx = np.array([[1, 4, 9], [114, 153, 198]])

    np.testing.assert_almost_equal(stats.log_likelihood, expected_ll)
    assert stats.t == data.shape[0]
    np.testing.assert_almost_equal(stats.n, expected_n)
    np.testing.assert_almost_equal(stats.sum_px, expected_sumPx)
    np.testing.assert_almost_equal(stats.sum_pxx, expected_sumPxx)

    # Adding Statistics
    new_stats = stats + stats

    new_expected_ll = expected_ll * 2
    new_expected_n = expected_n * 2
    new_expected_sumPx = expected_sumPx * 2
    new_expected_sumPxx = expected_sumPxx * 2

    assert new_stats.n.shape == (n_gaussians,), new_stats.n.shape
    assert new_stats.sum_px.shape == (
        n_gaussians,
        n_features,
    ), new_stats.sum_px.shape
    assert new_stats.sum_pxx.shape == (
        n_gaussians,
        n_features,
    ), new_stats.sum_pxx.shape

    np.testing.assert_almost_equal(new_stats.log_likelihood, new_expected_ll)
    assert new_stats.t == data.shape[0] * 2
    np.testing.assert_almost_equal(new_stats.n, new_expected_n)
    np.testing.assert_almost_equal(new_stats.sum_px, new_expected_sumPx)
    np.testing.assert_almost_equal(new_stats.sum_pxx, new_expected_sumPxx)

    # In-place adding of Statistics
    new_stats += stats

    new_expected_ll += expected_ll
    new_expected_n += expected_n
    new_expected_sumPx += expected_sumPx
    new_expected_sumPxx += expected_sumPxx

    assert new_stats.n.shape == (n_gaussians,), new_stats.n.shape
    assert new_stats.sum_px.shape == (
        n_gaussians,
        n_features,
    ), new_stats.sum_px.shape
    assert new_stats.sum_pxx.shape == (
        n_gaussians,
        n_features,
    ), new_stats.sum_pxx.shape

    np.testing.assert_almost_equal(new_stats.log_likelihood, new_expected_ll)
    assert new_stats.t == data.shape[0] * 3
    np.testing.assert_almost_equal(new_stats.n, new_expected_n)
    np.testing.assert_almost_equal(new_stats.sum_px, new_expected_sumPx)
    np.testing.assert_almost_equal(new_stats.sum_pxx, new_expected_sumPxx)


def test_machine_parameters():
    n_gaussians = 3
    n_features = 2
    machine = GMMMachine(n_gaussians)
    machine.means = np.repeat([[0], [1], [-1]], n_features, 1)
    machine.variances = np.ones_like(machine.means)
    np.testing.assert_equal(
        machine.means, np.repeat([[0], [1], [-1]], n_features, 1)
    )
    np.testing.assert_equal(
        machine.variances, np.ones((n_gaussians, n_features))
    )

    # Setters
    new_means = np.repeat([[1], [2], [3]], n_features, axis=1)
    machine.means = new_means
    assert machine.means.shape == (n_gaussians, n_features)
    np.testing.assert_equal(machine.means, new_means)
    new_variances = np.repeat([[0.2], [1.1], [1]], n_features, axis=1)
    machine.variances = new_variances
    assert machine.variances.shape == (n_gaussians, n_features)
    np.testing.assert_equal(machine.variances, new_variances)


def test_gmm_kmeans_plusplus_init():
    n_gaussians = 3
    machine = GMMMachine(
        n_gaussians,
        k_means_trainer=KMeansMachine(
            n_clusters=n_gaussians, init_method="k-means++"
        ),
    )
    data = np.array(
        [[1.5, 1], [1, 1.5], [-1, 0.5], [-1.5, 0], [2, 2], [2.5, 2.5]]
    )
    for transform in (to_numpy, to_dask_array):
        data = transform(data)
        machine = machine.fit(data)
        expected_means = np.array([[2.25, 2.25], [-1.25, 0.25], [1.25, 1.25]])
        expected_variances = np.array(
            [[1 / 16, 1 / 16], [1 / 16, 1 / 16], [1 / 16, 1 / 16]]
        )
        np.testing.assert_almost_equal(machine.means, expected_means, decimal=3)
        np.testing.assert_almost_equal(machine.variances, expected_variances)


def test_gmm_kmeans_parallel_init():
    n_gaussians = 3
    machine = GMMMachine(
        n_gaussians,
        k_means_trainer=KMeansMachine(
            n_clusters=n_gaussians, init_method="k-means||"
        ),
    )
    data = np.array(
        [[1.5, 1], [1, 1.5], [-1, 0.5], [-1.5, 0], [2, 2], [2.5, 2.5]]
    )
    with multiprocess_dask_client():
        for transform in (to_numpy, to_dask_array):
            data = transform(data)
            machine = machine.fit(data)
            expected_means = np.array(
                [[1.25, 1.25], [-1.25, 0.25], [2.25, 2.25]]
            )
            expected_variances = np.array(
                [[1 / 16, 1 / 16], [1 / 16, 1 / 16], [1 / 16, 1 / 16]]
            )
            np.testing.assert_almost_equal(
                machine.means, expected_means, decimal=3
            )
            np.testing.assert_almost_equal(
                machine.variances, expected_variances
            )


def test_likelihood():
    data = np.array([[1, 1, 1], [-1, 0, 0], [0, 0, 1], [2, 2, 2]])
    n_gaussians = 3
    machine = GMMMachine(n_gaussians)
    machine.means = np.repeat([[0], [1], [-1]], 3, 1)
    machine.variances = np.ones_like(machine.means)
    for transform in (to_numpy, to_dask_array):
        data = transform(data)
        log_likelihood = machine.log_likelihood(data)
        expected_ll = np.array(
            [
                -3.6519900964986527,
                -3.83151883210222,
                -3.83151883210222,
                -5.344374066745753,
            ]
        )
        np.testing.assert_almost_equal(log_likelihood, expected_ll)


def test_likelihood_variance():
    data = np.array([[1, 1, 1], [-1, 0, 0], [0, 0, 1], [2, 2, 2]])
    n_gaussians = 3
    machine = GMMMachine(n_gaussians)
    machine.means = np.repeat([[0], [1], [-1]], 3, 1)
    machine.variances = np.array(
        [
            [1.1, 1.2, 0.8],
            [0.2, 0.4, 0.5],
            [1, 1, 1],
        ]
    )
    for transform in (to_numpy, to_dask_array):
        data = transform(data)
        log_likelihood = machine.log_likelihood(data)
        expected_ll = np.array(
            [
                -2.202846959440514,
                -3.8699524542323793,
                -4.229029034375473,
                -6.940892214952679,
            ]
        )
        np.testing.assert_almost_equal(log_likelihood, expected_ll)


def test_likelihood_weight():
    data = np.array([[1, 1, 1], [-1, 0, 0], [0, 0, 1], [2, 2, 2]])
    n_gaussians = 3
    machine = GMMMachine(n_gaussians)
    machine.means = np.repeat([[0], [1], [-1]], 3, 1)
    machine.variances = np.ones_like(machine.means)
    machine.weights = [0.6, 0.1, 0.3]
    for transform in (to_numpy, to_dask_array):
        data = transform(data)
        log_likelihood = machine.log_likelihood(data)
        expected_ll = np.array(
            [
                -4.206596356117164,
                -3.492325679996329,
                -3.634745457950943,
                -6.49485678536014,
            ]
        )
        np.testing.assert_almost_equal(log_likelihood, expected_ll)


def test_GMMMachine_object():
    n_gaussians = 5
    machine = GMMMachine(n_gaussians)

    default_weights = np.full(
        shape=(n_gaussians,), fill_value=1.0 / n_gaussians
    )
    default_log_weights = np.full(
        shape=(n_gaussians,), fill_value=np.log(1.0 / n_gaussians)
    )

    # Test weights getting and setting
    np.testing.assert_almost_equal(machine.weights, default_weights)
    np.testing.assert_almost_equal(machine.log_weights, default_log_weights)

    modified_weights = default_weights
    modified_weights[: n_gaussians // 2] = (1 / n_gaussians) / 2
    modified_weights[n_gaussians // 2 + n_gaussians % 2 :] = (
        1 / n_gaussians
    ) * 1.5

    # Ensure setter works (log_weights is updated correctly)
    machine.weights = modified_weights
    np.testing.assert_almost_equal(machine.weights, modified_weights)
    np.testing.assert_almost_equal(
        machine.log_weights, np.log(modified_weights)
    )


def test_ml_em():
    # Simple GMM test
    data = np.array([[1, 2, 2], [2, 1, 2], [7, 8, 9], [7, 7, 8], [7, 9, 7]])
    n_gaussians = 2
    n_features = data.shape[-1]

    machine = GMMMachine(
        n_gaussians,
        update_means=True,
        update_variances=True,
        update_weights=True,
    )
    machine.means = np.repeat([[2], [8]], n_features, 1)
    machine.variances = np.ones_like(machine.means)

    stats = gmm_module.e_step(
        data,
        machine,
    )
    gmm_module.m_step(
        [stats],
        machine,
    )

    expected_means = np.array([[1.5, 1.5, 2.0], [7.0, 8.0, 8.0]])
    np.testing.assert_almost_equal(machine.means, expected_means)
    expected_weights = np.array([2 / 5, 3 / 5])
    np.testing.assert_almost_equal(machine.weights, expected_weights)
    eps = np.finfo(float).eps
    expected_variances = np.array([[1 / 4, 1 / 4, eps], [eps, 2 / 3, 2 / 3]])
    np.testing.assert_almost_equal(machine.variances, expected_variances)


def test_map_em():
    n_gaussians = 2
    prior_machine = GMMMachine(n_gaussians)
    prior_machine.means = np.array([[2, 2, 2], [8, 8, 8]])
    prior_machine.variances = np.ones_like(prior_machine.means)
    prior_machine.weights = np.array([0.5, 0.5])

    machine = GMMMachine(
        n_gaussians,
        trainer="map",
        ubm=prior_machine,
        update_means=True,
        update_variances=True,
        update_weights=True,
    )

    post_data = np.array(
        [[1, 2, 2], [2, 1, 2], [7, 8, 9], [7, 7, 8], [7, 9, 7]]
    )

    machine.initialize_gaussians(None)

    # Machine equals to priors before fitting
    np.testing.assert_equal(machine.means, prior_machine.means)
    np.testing.assert_equal(machine.variances, prior_machine.variances)
    np.testing.assert_equal(machine.weights, prior_machine.weights)

    stats = gmm_module.e_step(
        post_data,
        machine,
    )
    gmm_module.m_step(
        [stats],
        machine,
    )

    expected_means = np.array(
        [[1.83333333, 1.83333333, 2.0], [7.57142857, 8, 8]]
    )
    np.testing.assert_almost_equal(machine.means, expected_means)
    eps = np.finfo(float).eps
    expected_vars = np.array([[eps, eps, eps], [eps, eps, eps]])
    np.testing.assert_almost_equal(machine.variances, expected_vars)
    expected_weights = np.array([0.46226415, 0.53773585])
    np.testing.assert_almost_equal(machine.weights, expected_weights)


def test_ml_transformer():
    data = np.array([[1, 2, 2], [2, 1, 2], [7, 8, 9], [7, 7, 8], [7, 9, 7]])
    test_data = np.array([[1, 1, 1], [1, 1, 2], [8, 9, 9], [8, 8, 8]])
    n_gaussians = 2
    n_features = 3

    machine = GMMMachine(
        n_gaussians,
        update_means=True,
        update_variances=True,
        update_weights=True,
    )
    machine.means = np.array([[2, 2, 2], [8, 8, 8]])
    machine.variances = np.ones_like(machine.means)

    for transform in (to_numpy, to_dask_array):
        data = transform(data)
        machine = machine.fit(data)

        expected_means = np.array([[1.5, 1.5, 2.0], [7.0, 8.0, 8.0]])
        np.testing.assert_almost_equal(machine.means, expected_means)
        expected_weights = np.array([2 / 5, 3 / 5])
        np.testing.assert_almost_equal(machine.weights, expected_weights)
        eps = np.finfo(float).eps
        expected_variances = np.array(
            [[1 / 4, 1 / 4, eps], [eps, 2 / 3, 2 / 3]]
        )
        np.testing.assert_almost_equal(machine.variances, expected_variances)

        stats = machine.acc_stats(test_data)

        expected_stats = GMMStats(n_gaussians, n_features)
        expected_stats.init_fields(
            log_likelihood=-6755399441055685.0,
            t=test_data.shape[0],
            n=np.array([2, 2], dtype=float),
            sum_px=np.array([[2, 2, 3], [16, 17, 17]], dtype=float),
            sum_pxx=np.array([[2, 2, 5], [128, 145, 145]], dtype=float),
        )
        assert stats.is_similar_to(expected_stats)


def test_map_transformer():
    post_data = np.array(
        [[1, 2, 2], [2, 1, 2], [7, 8, 9], [7, 7, 8], [7, 9, 7]]
    )
    test_data = np.array([[1, 1, 1], [1, 1, 2], [8, 9, 9], [8, 8, 8]])
    n_gaussians = 2
    n_features = 3
    prior_machine = GMMMachine(n_gaussians)
    prior_machine.means = np.array([[2, 2, 2], [8, 8, 8]])
    prior_machine.variances = np.ones_like(prior_machine.means)
    prior_machine.weights = np.array([0.5, 0.5])

    machine = GMMMachine(
        n_gaussians,
        trainer="map",
        ubm=prior_machine,
        update_means=True,
        update_variances=True,
        update_weights=True,
    )

    for transform in (to_numpy, to_dask_array):
        post_data = transform(post_data)
        machine = machine.fit(post_data)

        expected_means = np.array(
            [[1.83333333, 1.83333333, 2.0], [7.57142857, 8, 8]]
        )
        np.testing.assert_almost_equal(machine.means, expected_means)
        eps = np.finfo(float).eps
        expected_vars = np.array([[eps, eps, eps], [eps, eps, eps]])
        np.testing.assert_almost_equal(machine.variances, expected_vars)
        expected_weights = np.array([0.46226415, 0.53773585])
        np.testing.assert_almost_equal(machine.weights, expected_weights)

        stats = machine.acc_stats(test_data)

        expected_stats = GMMStats(n_gaussians, n_features)
        expected_stats.init_fields(
            log_likelihood=-1.3837590691807108e16,
            t=test_data.shape[0],
            n=np.array([2, 2], dtype=float),
            sum_px=np.array([[2, 2, 3], [16, 17, 17]], dtype=float),
            sum_pxx=np.array([[2, 2, 5], [128, 145, 145]], dtype=float),
        )
        assert stats.is_similar_to(expected_stats)


# Tests from `test_em.py`


def test_gmm_ML_1():
    """Trains a GMMMachine with ML_GMMTrainer"""
    ar = load_array(
        resource_filename("bob.learn.em", "data/faithful.torch3_f64.hdf5")
    )
    gmm_ref = GMMMachine.from_hdf5(
        HDF5File(resource_filename("bob.learn.em", "data/gmm_ML.hdf5"), "r")
    )

    for transform in (to_numpy, to_dask_array):
        ar = transform(ar)

        gmm = loadGMM()

        # test rng handling
        gmm.convergence_threshold = 0.001
        gmm.update_means = True
        gmm.update_variances = True
        gmm.update_weights = True
        gmm.random_state = np.random.RandomState(seed=12345)
        gmm = gmm.fit(ar)

        gmm = loadGMM()
        gmm.convergence_threshold = 0.001
        gmm.update_means = True
        gmm.update_variances = True
        gmm.update_weights = True
        # Generate reference
        # gmm.save(HDF5File(resource_filename("bob.learn.em", "data/gmm_ML.hdf5"), "w"))

        gmm = gmm.fit(ar)

        assert_gmm_equal(gmm, gmm_ref)


def test_gmm_ML_2():
    # Trains a GMMMachine with ML_GMMTrainer; compares to a reference
    ar = load_array(
        resource_filename("bob.learn.em", "data/dataNormalized.hdf5")
    )

    # Test results
    # Load torch3vision reference
    meansML_ref = load_array(
        resource_filename("bob.learn.em", "data/meansAfterML.hdf5")
    )
    variancesML_ref = load_array(
        resource_filename("bob.learn.em", "data/variancesAfterML.hdf5")
    )
    weightsML_ref = load_array(
        resource_filename("bob.learn.em", "data/weightsAfterML.hdf5")
    )

    for transform in (to_numpy, to_dask_array):
        ar = transform(ar)
        # Initialize GMMMachine
        gmm = GMMMachine(n_gaussians=5)
        gmm.means = load_array(
            resource_filename("bob.learn.em", "data/meansAfterKMeans.hdf5")
        ).astype("float64")
        gmm.variances = load_array(
            resource_filename("bob.learn.em", "data/variancesAfterKMeans.hdf5")
        ).astype("float64")
        gmm.weights = np.exp(
            load_array(
                resource_filename(
                    "bob.learn.em", "data/weightsAfterKMeans.hdf5"
                )
            ).astype("float64")
        )

        threshold = 0.001
        gmm.variance_thresholds = threshold

        # Initialize ML Trainer
        gmm.mean_var_update_threshold = 0.001
        gmm.max_fitting_steps = 25
        gmm.convergence_threshold = 0.000001
        gmm.update_means = True
        gmm.update_variances = True
        gmm.update_weights = True

        # Run ML
        gmm = gmm.fit(ar)

        # Compare to current results
        np.testing.assert_allclose(gmm.means, meansML_ref, atol=3e-3)
        np.testing.assert_allclose(gmm.variances, variancesML_ref, atol=3e-3)
        np.testing.assert_allclose(gmm.weights, weightsML_ref, atol=1e-4)


def test_gmm_MAP_1():
    # Train a GMMMachine with MAP_GMMTrainer
    ar = load_array(
        resource_filename("bob.learn.em", "data/faithful.torch3_f64.hdf5")
    )

    # test with rng
    gmmprior = GMMMachine.from_hdf5(
        HDF5File(resource_filename("bob.learn.em", "data/gmm_ML.hdf5"), "r")
    )
    gmm = GMMMachine.from_hdf5(
        HDF5File(resource_filename("bob.learn.em", "data/gmm_ML.hdf5"), "r"),
        ubm=gmmprior,
    )
    gmm.update_means = True
    gmm.update_variances = False
    gmm.update_weights = False
    rng = np.random.RandomState(seed=12345)
    gmm.random_state = rng
    gmm = gmm.fit(ar)

    gmmprior = GMMMachine.from_hdf5(
        HDF5File(resource_filename("bob.learn.em", "data/gmm_ML.hdf5"), "r")
    )
    gmm = GMMMachine.from_hdf5(
        HDF5File(resource_filename("bob.learn.em", "data/gmm_ML.hdf5"), "r"),
        ubm=gmmprior,
    )
    gmm.update_means = True
    gmm.update_variances = False
    gmm.update_weights = False

    # Generate reference
    # gmm.save(HDF5File(resource_filename("bob.learn.em", "data/gmm_MAP.hdf5"), "w"))

    gmm_ref = GMMMachine.from_hdf5(
        HDF5File(resource_filename("bob.learn.em", "data/gmm_MAP.hdf5"), "r")
    )

    for transform in (to_numpy, to_dask_array):
        ar = transform(ar)
        gmm = gmm.fit(ar)

        np.testing.assert_almost_equal(gmm.means, gmm_ref.means, decimal=3)
        np.testing.assert_almost_equal(
            gmm.variances, gmm_ref.variances, decimal=3
        )
        np.testing.assert_almost_equal(gmm.weights, gmm_ref.weights, decimal=3)


def test_gmm_MAP_2():
    # Train a GMMMachine with MAP_GMMTrainer and compare with matlab reference

    data = load_array(resource_filename("bob.learn.em", "data/data.hdf5"))
    data = data.reshape((1, -1))  # make a 2D array out of it
    means = load_array(resource_filename("bob.learn.em", "data/means.hdf5"))
    variances = load_array(
        resource_filename("bob.learn.em", "data/variances.hdf5")
    )
    weights = load_array(resource_filename("bob.learn.em", "data/weights.hdf5"))

    gmm = GMMMachine(n_gaussians=2)
    gmm.means = means
    gmm.variances = variances
    gmm.weights = weights

    gmm_adapted = GMMMachine(
        n_gaussians=2,
        trainer="map",
        ubm=gmm,
        max_fitting_steps=1,
        update_means=True,
        update_variances=False,
        update_weights=False,
        mean_var_update_threshold=0.0,
    )
    gmm_adapted.means = means
    gmm_adapted.variances = variances
    gmm_adapted.weights = weights

    new_means = load_array(
        resource_filename("bob.learn.em", "data/new_adapted_mean.hdf5")
    )

    for transform in (to_numpy, to_dask_array):
        data = transform(data)
        gmm_adapted = gmm_adapted.fit(data)

        # Compare to matlab reference
        np.testing.assert_allclose(new_means.T, gmm_adapted.means, rtol=1e-4)


def test_gmm_MAP_3():
    # Train a GMMMachine with MAP_GMMTrainer; compares to old reference
    ar = load_array(resource_filename("bob.learn.em", "data/dataforMAP.hdf5"))

    # Initialize GMMMachine
    n_gaussians = 5
    prior_gmm = GMMMachine(n_gaussians)
    prior_gmm.means = load_array(
        resource_filename("bob.learn.em", "data/meansAfterML.hdf5")
    )
    prior_gmm.variances = load_array(
        resource_filename("bob.learn.em", "data/variancesAfterML.hdf5")
    )
    prior_gmm.weights = load_array(
        resource_filename("bob.learn.em", "data/weightsAfterML.hdf5")
    )

    threshold = 0.001
    prior_gmm.variance_thresholds = threshold

    # Initialize MAP Trainer
    prior = 0.001
    accuracy = 0.00001
    gmm = GMMMachine(
        n_gaussians,
        trainer="map",
        ubm=prior_gmm,
        convergence_threshold=prior,
        max_fitting_steps=1,
        update_means=True,
        update_variances=False,
        update_weights=False,
        mean_var_update_threshold=accuracy,
        map_relevance_factor=None,
    )
    gmm.variance_thresholds = threshold

    # Test results
    # Load torch3vision reference
    meansMAP_ref = load_array(
        resource_filename("bob.learn.em", "data/meansAfterMAP.hdf5")
    )
    variancesMAP_ref = load_array(
        resource_filename("bob.learn.em", "data/variancesAfterMAP.hdf5")
    )
    weightsMAP_ref = load_array(
        resource_filename("bob.learn.em", "data/weightsAfterMAP.hdf5")
    )

    for transform in (to_numpy, to_dask_array):
        ar = transform(ar)
        # Train
        gmm = gmm.fit(ar)

        # Compare to current results
        # Gaps are quite large. This might be explained by the fact that there is no
        # adaptation of a given Gaussian in torch3 when the corresponding responsibilities
        # are below the responsibilities threshold
        np.testing.assert_allclose(gmm.means, meansMAP_ref, atol=2e-1)
        np.testing.assert_allclose(gmm.variances, variancesMAP_ref, atol=1e-4)
        np.testing.assert_allclose(gmm.weights, weightsMAP_ref, atol=1e-4)


def test_gmm_test():
    # Tests a GMMMachine by computing scores against a model and comparing to a reference

    ar = load_array(resource_filename("bob.learn.em", "data/dataforMAP.hdf5"))

    # Initialize GMMMachine
    n_gaussians = 5
    gmm = GMMMachine(n_gaussians)
    gmm.means = load_array(
        resource_filename("bob.learn.em", "data/meansAfterML.hdf5")
    )
    gmm.variances = load_array(
        resource_filename("bob.learn.em", "data/variancesAfterML.hdf5")
    )
    gmm.weights = load_array(
        resource_filename("bob.learn.em", "data/weightsAfterML.hdf5")
    )

    threshold = 0.001
    gmm.variance_thresholds = threshold

    # Test against the model
    score_mean_ref = -1.50379e06
    for transform in (to_numpy, to_dask_array):
        ar = transform(ar)
        score = gmm.log_likelihood(ar).sum()
        score /= len(ar)

        # Compare current results to torch3vision
        assert abs(score - score_mean_ref) / score_mean_ref < 1e-4
