#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
# Thu Feb 16 17:57:10 2012 +0200
#
# Copyright (C) 2011-2014 Idiap Research Institute, Martigny, Switzerland

"""Tests the GMM machine and the GMMStats container
"""

import numpy as np

import bob.io.base
from bob.io.base.test_utils import datafile

from bob.learn.em.mixture import GMMMachine
from bob.learn.em.mixture import MLGMMTrainer
from bob.learn.em.mixture import MAPGMMTrainer
from bob.learn.em.mixture import Statistics
from bob.learn.em.mixture import Gaussian
from bob.learn.em.mixture import MultiGaussian


def test_GMMMachine_1():
  # Test a GMMMachine basic features

  weights   = np.array([0.5, 0.5], 'float64')
  weights2   = np.array([0.6, 0.4], 'float64')
  means     = np.array([[3, 70, 0], [4, 72, 0]], 'float64')
  means2     = np.array([[3, 7, 0], [4, 72, 0]], 'float64')
  variances = np.array([[1, 10, 1], [2, 5, 2]], 'float64')
  variances2 = np.array([[10, 10, 1], [2, 5, 2]], 'float64')
  varianceThresholds = np.array([[0, 0, 0], [0, 0, 0]], 'float64')
  varianceThresholds2 = np.array([[0.0005, 0.0005, 0.0005], [0, 0, 0]], 'float64')

  # Initializes a GMMMachine
  gmm = GMMMachine(n_gaussians=2)
  # Sets the weights, means, variances and varianceThresholds and
  # Checks correctness
  gmm.weights = weights
  gmm.means = means
  gmm.variances = variances
  gmm.variance_thresholds = varianceThresholds
  assert gmm.shape == (2,3)
  assert (gmm.weights == weights).all()
  assert (gmm.means == means).all()
  assert (gmm.variances == variances).all()
  assert (gmm.variance_thresholds == varianceThresholds).all()

  # Checks supervector-like accesses
  assert (gmm.means == means).all()
  assert (gmm.variances == variances).all()
  newMeans = np.array([[3, 70, 2], [4, 72, 2]], 'float64')
  newVariances = np.array([[1, 1, 1], [2, 2, 2]], 'float64')


  # Checks particular varianceThresholds-related methods
  varianceThresholds1D = np.array([0.3, 1, 0.5], 'float64')
  gmm.variance_thresholds = varianceThresholds1D
  assert (gmm.variance_thresholds[0,:] == varianceThresholds1D).all()
  assert (gmm.variance_thresholds[1,:] == varianceThresholds1D).all()

  gmm.variance_thresholds = 0.005
  assert (gmm.variance_thresholds == 0.005).all()

  # Checks Gaussians access
  gmm.means     = newMeans
  gmm.variances = newVariances
  assert (gmm.gaussians_[0]["mean"] == newMeans[0,:]).all()
  assert (gmm.gaussians_[1]["mean"] == newMeans[1,:]).all()
  assert (gmm.gaussians_[0]["variance"] == newVariances[0,:]).all()
  assert (gmm.gaussians_[1]["variance"] == newVariances[1,:]).all()

  # Checks comparison
  gmm2 = gmm.copy()
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

  arrayset = bob.io.base.load(datafile("faithful.torch3_f64.hdf5", __name__, path="../data/"))
  gmm = GMMMachine(n_gaussians=2)
  gmm.weights   = np.array([0.5, 0.5], 'float64')
  gmm.means     = np.array([[3, 70], [4, 72]], 'float64')
  gmm.variances = np.array([[1, 10], [2, 5]], 'float64')
  gmm.variance_thresholds = np.array([[0, 0], [0, 0]], 'float64')

  trainer = MLGMMTrainer()
  trainer.e_step(gmm, arrayset)
  stats = trainer.last_step_stats

  stats_ref = Statistics(n_gaussians=2, n_features=2)
  stats_ref.from_file(bob.io.base.HDF5File(datafile("stats.hdf5",__name__, path="../data/")))

  assert stats.t == stats_ref.t
  assert np.allclose(stats.n, stats_ref.n, atol=1e-10)
  #assert np.array_equal(stats.sumPx, stats_ref.sumPx)
  #Note AA: precision error above
  assert np.allclose(stats.sumPx, stats_ref.sumPx, atol=1e-10)
  assert np.allclose(stats.sumPxx, stats_ref.sumPxx, atol=1e-10)


def test_GMMMachine_3():
  # Test a GMMMachine (log-likelihood computation)

  data = bob.io.base.load(datafile('data.hdf5', __name__, path="../data/"))
  gmm = GMMMachine(n_gaussians=2)
  gmm.weights   = bob.io.base.load(datafile('weights.hdf5', __name__, path="../data/"))
  gmm.means     = bob.io.base.load(datafile('means.hdf5', __name__, path="../data/"))
  gmm.variances = bob.io.base.load(datafile('variances.hdf5', __name__, path="../data/"))

  # Compare the log-likelihood with the one obtained using Chris Matlab
  # implementation
  matlab_ll_ref = -2.361583051672024e+02
  assert abs(gmm.log_likelihood(data) - matlab_ll_ref) < 1e-10


def test_GMMMachine_4():

  np.random.seed(3) # FIXING A SEED

  data = np.random.rand(100,50) # Doesn't matter if it is random. The average of 1D array (in python) MUST output the same result for the 2D array (in C++)

  gmm = GMMMachine(n_gaussians=2)
  gmm.weights   = bob.io.base.load(datafile('weights.hdf5', __name__, path="../data/"))
  gmm.means     = bob.io.base.load(datafile('means.hdf5', __name__, path="../data/"))
  gmm.variances = bob.io.base.load(datafile('variances.hdf5', __name__, path="../data/"))


  ll = 0
  for i in range(data.shape[0]):
    ll += gmm.log_likelihood(data[i,:])
  ll /= data.shape[0]

  assert np.isclose(ll, gmm.log_likelihood(data).mean())


def test_GMMStats():
    """Test a GMMStats."""
    # Initializing a GMMStats
    data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [7, 8, 9]])
    n_gaussians = 2
    n_features = data.shape[-1]
    machine = GMMMachine(n_gaussians)
    trainer = MLGMMTrainer()

    machine.gaussians_ = MultiGaussian(means=np.array([[0, 0, 0], [8, 8, 8]]))

    # Populate the Statistics
    trainer.e_step(machine, data)

    stats = trainer.last_step_stats

    # Check shapes
    assert stats.n.shape == (n_gaussians,), stats.n.shape
    assert stats.sumPx.shape == (n_gaussians, n_features), stats.sumPx.shape
    assert stats.sumPxx.shape == (n_gaussians, n_features), stats.sumPxx.shape

    # Check values
    expected_ll = -37.2998511206581
    expected_n = np.array([1, 3])
    expected_sumPx = np.array([[1, 2, 3], [18, 21, 24]])
    expected_sumPxx = np.array([[1, 4, 9], [114, 153, 198]])

    np.testing.assert_almost_equal(stats.log_likelihood, expected_ll)
    assert stats.t == data.shape[0]
    np.testing.assert_almost_equal(stats.n, expected_n)
    np.testing.assert_almost_equal(stats.sumPx, expected_sumPx)
    np.testing.assert_almost_equal(stats.sumPxx, expected_sumPxx)

    # Adding Statistics
    new_stats = stats + stats

    new_expected_ll = expected_ll * 2
    new_expected_n = expected_n * 2
    new_expected_sumPx = expected_sumPx * 2
    new_expected_sumPxx = expected_sumPxx * 2

    assert new_stats.n.shape == (n_gaussians,), new_stats.n.shape
    assert new_stats.sumPx.shape == (n_gaussians, n_features), new_stats.sumPx.shape
    assert new_stats.sumPxx.shape == (n_gaussians, n_features), new_stats.sumPxx.shape

    np.testing.assert_almost_equal(new_stats.log_likelihood, new_expected_ll)
    assert new_stats.t == data.shape[0] * 2
    np.testing.assert_almost_equal(new_stats.n, new_expected_n)
    np.testing.assert_almost_equal(new_stats.sumPx, new_expected_sumPx)
    np.testing.assert_almost_equal(new_stats.sumPxx, new_expected_sumPxx)

    # In-place adding of Statistics
    new_stats += stats

    new_expected_ll += expected_ll
    new_expected_n += expected_n
    new_expected_sumPx += expected_sumPx
    new_expected_sumPxx += expected_sumPxx

    assert new_stats.n.shape == (n_gaussians,), new_stats.n.shape
    assert new_stats.sumPx.shape == (n_gaussians, n_features), new_stats.sumPx.shape
    assert new_stats.sumPxx.shape == (n_gaussians, n_features), new_stats.sumPxx.shape

    np.testing.assert_almost_equal(new_stats.log_likelihood, new_expected_ll)
    assert new_stats.t == data.shape[0] * 3
    np.testing.assert_almost_equal(new_stats.n, new_expected_n)
    np.testing.assert_almost_equal(new_stats.sumPx, new_expected_sumPx)
    np.testing.assert_almost_equal(new_stats.sumPxx, new_expected_sumPxx)


def test_machine_parameters():
    n_gaussians = 3
    n_features = 2
    machine = GMMMachine(n_gaussians)
    machine.gaussians_ = MultiGaussian(
        means=np.repeat([[0], [1], [-1]], n_features, 1)
    )
    assert machine.means.shape == (n_gaussians, n_features)
    np.testing.assert_almost_equal(
        machine.means, np.repeat([[0], [1], [-1]], n_features, 1)
    )
    assert machine.variances.shape == (n_gaussians, n_features)
    np.testing.assert_almost_equal(machine.variances, 1)

    # Setters

    new_means = np.repeat([[1], [2], [3]], n_features, axis=1)
    machine.means = new_means
    np.testing.assert_almost_equal(machine.gaussians_["mean"], new_means)
    assert machine.means.shape == (n_gaussians, n_features)
    np.testing.assert_almost_equal(machine.means, new_means)
    new_variances = np.repeat([[0.2], [1.1], [1]], n_features, axis=1)
    machine.variances = new_variances
    np.testing.assert_almost_equal(machine.gaussians_["variance"], new_variances)
    assert machine.variances.shape == (n_gaussians, n_features)
    np.testing.assert_almost_equal(machine.variances, new_variances)


def test_likelihood():
    data = np.array([[1, 1, 1], [-1, 0, 0], [0, 0, 1], [2, 2, 2]])
    n_gaussians = 3
    machine = GMMMachine(n_gaussians)
    machine.gaussians_ = MultiGaussian(
        means=np.repeat([[0], [1], [-1]], 3, 1)
    )
    log_likelihood = machine.log_likelihood(data)
    expected_ll = np.array(
        [-3.6519900964986527, -3.83151883210222, -3.83151883210222, -5.344374066745753]
    )
    np.testing.assert_almost_equal(log_likelihood, expected_ll)


def test_likelihood_variance():
    data = np.array([[1, 1, 1], [-1, 0, 0], [0, 0, 1], [2, 2, 2]])
    n_gaussians = 3
    machine = GMMMachine(n_gaussians)
    machine.gaussians_ = MultiGaussian(
        means=np.repeat([[0], [1], [-1]], 3, 1),
        variances=np.array([
            [1.1, 1.2, 0.8],
            [0.2, 0.4, 0.5],
            [1, 1, 1],
        ])
    )
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
    machine.gaussians_ = MultiGaussian(
        means=np.repeat([[0], [1], [-1]], 3, 1)
    )
    machine.weights = [0.6, 0.1, 0.3]
    log_likelihood = machine.log_likelihood(data)
    expected_ll = np.array(
        [-4.206596356117164, -3.492325679996329, -3.634745457950943, -6.49485678536014]
    )
    np.testing.assert_almost_equal(log_likelihood, expected_ll)


def test_GMMMachine_object():
    n_gaussians = 5
    machine = GMMMachine(n_gaussians)

    default_weights = np.full(shape=(n_gaussians,), fill_value=1.0 / n_gaussians)
    default_log_weights = np.full(
        shape=(n_gaussians,), fill_value=np.log(1.0 / n_gaussians)
    )

    # Test weights getting and setting
    np.testing.assert_almost_equal(machine.weights, default_weights)
    np.testing.assert_almost_equal(machine.log_weights, default_log_weights)

    modified_weights = default_weights
    modified_weights[: n_gaussians // 2] = (1 / n_gaussians) / 2
    modified_weights[n_gaussians // 2 + n_gaussians % 2 :] = (1 / n_gaussians) * 1.5

    # Ensure setter works (log_weights is updated correctly)
    machine.weights = modified_weights
    np.testing.assert_almost_equal(machine.weights, modified_weights)
    np.testing.assert_almost_equal(machine.log_weights, np.log(modified_weights))


def test_MLTrainer():
    data = np.array([[1, 2, 2], [2, 1, 2], [7, 8, 9], [7, 7, 8]])
    n_gaussians = 2
    n_features = data.shape[-1]
    machine = GMMMachine(n_gaussians)
    gaussians_init = np.array([Gaussian([m] * n_features) for m in range(n_gaussians)])
    trainer = MLGMMTrainer(init_method=gaussians_init)
    trainer.initialize(machine, data)

    trainer.e_step(machine, data)
    trainer.m_step(machine, data)
    # TODO
    raise NotImplementedError("TODO")


def test_MAPTrainer():
    n_gaussians = 2
    machine = GMMMachine(n_gaussians)
    prior_machine = GMMMachine(n_gaussians)
    prior_machine.gaussians_ = MultiGaussian(
        means=np.repeat([[2], [8]], 3, 1)
    )
    prior_machine.weights = np.array([0.5, 0.5])

    post_data = np.array([[1, 2, 2], [2, 1, 2], [7, 8, 9], [7, 7, 9]])
    trainer = MAPGMMTrainer(prior_gmm=prior_machine, update_means=True, update_variances=True, update_weights=True, relevance_factor=4,alpha=0.5)
    trainer.initialize(machine, post_data)
    trainer.e_step(machine, post_data)
    trainer.m_step(machine, post_data)
    expected_means = np.array([[1.83333333, 1.83333333, 2.],[7.66666667, 7.83333333, 8.33333333]])
    np.testing.assert_almost_equal(machine.means, expected_means)
    expected_vars = np.array([[1, 1, 1], [7, 8, 9]])  # TODO take the values from C++
    np.testing.assert_almost_equal(
        machine.gaussians_["variances"], expected_vars, err_msg="TODO"  # TODO
    )
