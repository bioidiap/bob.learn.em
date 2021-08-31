#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
# Thu Feb 16 17:57:10 2012 +0200
#
# Copyright (C) 2011-2014 Idiap Research Institute, Martigny, Switzerland

"""Tests the GMM machine and the GMMStats container
"""

import os
import numpy
import tempfile

import bob.io.base
from bob.io.base.test_utils import datafile

from bob.learn.em.mixture import GMMMachine
from bob.learn.em.mixture import MLGMMTrainer
from bob.learn.em.mixture import Statistics
from bob.learn.em.mixture import Gaussian


def test_GMMStats():
    """Test a GMMStats."""
    # Initializing a GMMStats
    data = numpy.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [7, 8, 9]])
    n_gaussians = 2
    n_features = data.shape[-1]
    machine = GMMMachine(n_gaussians)
    trainer = MLGMMTrainer()

    machine.gaussians_ = numpy.array([Gaussian([0, 0, 0]), Gaussian([8, 8, 8])])

    # Populate the Statistics
    trainer.e_step(machine, data)

    stats = trainer.last_step_stats

    # Check shapes
    assert stats.n.shape == (n_gaussians,), stats.n.shape
    assert stats.sumPx.shape == (n_gaussians, n_features), stats.sumPx.shape
    assert stats.sumPxx.shape == (n_gaussians, n_features), stats.sumPxx.shape

    # Check values
    expected_ll = -37.2998511206581
    expected_n = numpy.array([1, 3])
    expected_sumPx = numpy.array([[1, 2, 3], [18, 21, 24]])
    expected_sumPxx = numpy.array([[1, 4, 9], [114, 153, 198]])

    assert numpy.isclose(stats.log_likelihood, expected_ll)
    assert stats.T == data.shape[0]
    assert numpy.allclose(stats.n, expected_n)
    assert numpy.allclose(stats.sumPx, expected_sumPx)
    assert numpy.allclose(stats.sumPxx, expected_sumPxx)

    # Adding Statistics
    new_stats = stats + stats

    new_expected_ll = expected_ll * 2
    new_expected_n = expected_n * 2
    new_expected_sumPx = expected_sumPx * 2
    new_expected_sumPxx = expected_sumPxx * 2

    assert new_stats.n.shape == (n_gaussians,), new_stats.n.shape
    assert new_stats.sumPx.shape == (n_gaussians, n_features), new_stats.sumPx.shape
    assert new_stats.sumPxx.shape == (n_gaussians, n_features), new_stats.sumPxx.shape

    assert numpy.isclose(new_stats.log_likelihood, new_expected_ll)
    assert new_stats.T == data.shape[0] * 2
    assert numpy.allclose(new_stats.n, new_expected_n)
    assert numpy.allclose(new_stats.sumPx, new_expected_sumPx)
    assert numpy.allclose(new_stats.sumPxx, new_expected_sumPxx)

    # In-place adding of Statistics
    new_stats += stats

    new_expected_ll += expected_ll
    new_expected_n += expected_n
    new_expected_sumPx += expected_sumPx
    new_expected_sumPxx += expected_sumPxx

    assert new_stats.n.shape == (n_gaussians,), new_stats.n.shape
    assert new_stats.sumPx.shape == (n_gaussians, n_features), new_stats.sumPx.shape
    assert new_stats.sumPxx.shape == (n_gaussians, n_features), new_stats.sumPxx.shape

    assert numpy.isclose(new_stats.log_likelihood, new_expected_ll)
    assert new_stats.T == data.shape[0] * 3
    assert numpy.allclose(new_stats.n, new_expected_n)
    assert numpy.allclose(new_stats.sumPx, new_expected_sumPx)
    assert numpy.allclose(new_stats.sumPxx, new_expected_sumPxx)


def test_GMMMachine_object():
    n_gaussians = 5
    machine = GMMMachine(n_gaussians)

    default_weights = numpy.full(shape=(n_gaussians,), fill_value=1.0/n_gaussians)
    default_log_weights = numpy.full(shape=(n_gaussians,), fill_value=numpy.log(1.0/n_gaussians))

    # Test weights getting and setting
    assert numpy.allclose(machine.weights, default_weights)
    assert numpy.allclose(machine.log_weights, default_log_weights)

    modified_weights = default_weights
    modified_weights[:n_gaussians//2] = (1/n_gaussians)/2
    modified_weights[n_gaussians//2+n_gaussians%2:] = (1/n_gaussians)*1.5

    # Ensure setter works (log_weights is updated correctly)
    machine.weights = modified_weights
    assert numpy.allclose(machine.weights, modified_weights)
    assert numpy.allclose(machine.log_weights, numpy.log(modified_weights))


def test_MLTrainer():
    data = numpy.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [7, 8, 9]])
    n_gaussians = 2
    n_features = data.shape[-1]
    machine = GMMMachine(n_gaussians)
    trainer = MLGMMTrainer()
    trainer.initialize(machine, data)

    trainer.e_step(machine, data)
    trainer.m_step(machine, data)


def OLD_test_GMMMachine_1():
    # Test a GMMMachine basic features

    weights = numpy.array([0.5, 0.5], "float64")
    weights2 = numpy.array([0.6, 0.4], "float64")
    means = numpy.array([[3, 70, 0], [4, 72, 0]], "float64")
    means2 = numpy.array([[3, 7, 0], [4, 72, 0]], "float64")
    variances = numpy.array([[1, 10, 1], [2, 5, 2]], "float64")
    variances2 = numpy.array([[10, 10, 1], [2, 5, 2]], "float64")
    varianceThresholds = numpy.array([[0, 0, 0], [0, 0, 0]], "float64")
    varianceThresholds2 = numpy.array([[0.0005, 0.0005, 0.0005], [0, 0, 0]], "float64")

    # Initializes a GMMMachine
    gmm = GMMMachine(2, 3)
    # Sets the weights, means, variances and varianceThresholds and
    # Checks correctness
    gmm.weights = weights
    gmm.means = means
    gmm.variances = variances
    gmm.variance_thresholds = varianceThresholds
    assert gmm.shape == (2, 3)
    assert (gmm.weights == weights).all()
    assert (gmm.means == means).all()
    assert (gmm.variances == variances).all()
    assert (gmm.variance_thresholds == varianceThresholds).all()

    # Checks supervector-like accesses
    assert (gmm.mean_supervector == means.reshape(means.size)).all()
    assert (gmm.variance_supervector == variances.reshape(variances.size)).all()
    newMeans = numpy.array([[3, 70, 2], [4, 72, 2]], "float64")
    newVariances = numpy.array([[1, 1, 1], [2, 2, 2]], "float64")

    # Checks particular varianceThresholds-related methods
    varianceThresholds1D = numpy.array([0.3, 1, 0.5], "float64")
    gmm.set_variance_thresholds(varianceThresholds1D)
    assert (gmm.variance_thresholds[0, :] == varianceThresholds1D).all()
    assert (gmm.variance_thresholds[1, :] == varianceThresholds1D).all()

    gmm.set_variance_thresholds(0.005)
    assert (gmm.variance_thresholds == 0.005).all()

    # Checks Gaussians access
    gmm.means = newMeans
    gmm.variances = newVariances
    assert (gmm.get_gaussian(0).mean == newMeans[0, :]).all()
    assert (gmm.get_gaussian(1).mean == newMeans[1, :]).all()
    assert (gmm.get_gaussian(0).variance == newVariances[0, :]).all()
    assert (gmm.get_gaussian(1).variance == newVariances[1, :]).all()

    # Checks resize
    gmm.resize(4, 5)
    assert gmm.shape == (4, 5)

    # Checks comparison
    gmm2 = GMMMachine(gmm)
    gmm3 = GMMMachine(2, 3)
    gmm3.weights = weights2
    gmm3.means = means
    gmm3.variances = variances
    # gmm3.varianceThresholds = varianceThresholds
    gmm4 = GMMMachine(2, 3)
    gmm4.weights = weights
    gmm4.means = means2
    gmm4.variances = variances
    # gmm4.varianceThresholds = varianceThresholds
    gmm5 = GMMMachine(2, 3)
    gmm5.weights = weights
    gmm5.means = means
    gmm5.variances = variances2
    # gmm5.varianceThresholds = varianceThresholds
    gmm6 = GMMMachine(2, 3)
    gmm6.weights = weights
    gmm6.means = means
    gmm6.variances = variances
    # gmm6.varianceThresholds = varianceThresholds2

    assert gmm == gmm2
    assert (gmm != gmm2) is False
    assert gmm.is_similar_to(gmm2)
    assert gmm != gmm3
    assert (gmm == gmm3) is False
    assert gmm.is_similar_to(gmm3) is False
    assert gmm != gmm4
    assert (gmm == gmm4) is False
    assert gmm.is_similar_to(gmm4) is False
    assert gmm != gmm5
    assert (gmm == gmm5) is False
    assert gmm.is_similar_to(gmm5) is False
    assert gmm != gmm6
    assert (gmm == gmm6) is False
    assert gmm.is_similar_to(gmm6) is False


def test_GMMMachine_2():
    # Test a GMMMachine (statistics)

    arrayset = bob.io.base.load(
        datafile("faithful.torch3_f64.hdf5", __name__, path="../data/")
    )
    gmm = GMMMachine(2, 2)
    gmm.weights = numpy.array([0.5, 0.5], "float64")
    gmm.means = numpy.array([[3, 70], [4, 72]], "float64")
    gmm.variances = numpy.array([[1, 10], [2, 5]], "float64")
    gmm.variance_thresholds = numpy.array([[0, 0], [0, 0]], "float64")

    stats = GMMStats(2, 2)
    gmm.acc_statistics(arrayset, stats)

    stats_ref = GMMStats(
        bob.io.base.HDF5File(datafile("stats.hdf5", __name__, path="../data/"))
    )

    assert stats.t == stats_ref.t
    assert numpy.allclose(stats.n, stats_ref.n, atol=1e-10)
    # assert numpy.array_equal(stats.sumPx, stats_ref.sumPx)
    # Note AA: precision error above
    assert numpy.allclose(stats.sum_px, stats_ref.sum_px, atol=1e-10)
    assert numpy.allclose(stats.sum_pxx, stats_ref.sum_pxx, atol=1e-10)


def test_GMMMachine_3():
    # Test a GMMMachine (log-likelihood computation)

    data = bob.io.base.load(datafile("data.hdf5", __name__, path="../data/"))
    gmm = GMMMachine(2, 50)
    gmm.weights = bob.io.base.load(datafile("weights.hdf5", __name__, path="../data/"))
    gmm.means = bob.io.base.load(datafile("means.hdf5", __name__, path="../data/"))
    gmm.variances = bob.io.base.load(
        datafile("variances.hdf5", __name__, path="../data/")
    )

    # Compare the log-likelihood with the one obtained using Chris Matlab
    # implementation
    matlab_ll_ref = -2.361583051672024e02
    assert abs(gmm(data) - matlab_ll_ref) < 1e-10


def test_GMMMachine_4():

    import numpy

    numpy.random.seed(3)  # FIXING A SEED

    data = numpy.random.rand(
        100, 50
    )  # Doesn't matter if it is ramdom. The average of 1D array (in python) MUST output the same result for the 2D array (in C++)

    gmm = GMMMachine(2, 50)
    gmm.weights = bob.io.base.load(datafile("weights.hdf5", __name__, path="../data/"))
    gmm.means = bob.io.base.load(datafile("means.hdf5", __name__, path="../data/"))
    gmm.variances = bob.io.base.load(
        datafile("variances.hdf5", __name__, path="../data/")
    )

    ll = 0
    for i in range(data.shape[0]):
        ll += gmm(data[i, :])
    ll /= data.shape[0]

    assert ll == gmm(data)
