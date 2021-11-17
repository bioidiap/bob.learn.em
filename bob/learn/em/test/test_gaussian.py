#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
# Thu Feb 16 16:54:45 2012 +0200
#
# Copyright (C) 2011-2013 Idiap Research Institute, Martigny, Switzerland

"""Tests the Gaussian class
"""

import tempfile

import numpy as np

from h5py import File as HDF5File

from bob.learn.em.mixture import Gaussians


def equals(x, y, epsilon):
    return abs(x - y) < epsilon


def test_GaussiansObject():
  """Tests a Gaussians object creation and manipulation."""

  # Initializes a Gaussians with zero mean and unit variance
  g = Gaussians(means=np.zeros((3,)))
  np.testing.assert_equal(g["means"], np.zeros((1,3)))
  np.testing.assert_equal(g["variances"], np.ones((1,3)))
  np.testing.assert_equal(g.shape, (1,3))

  # Set and check mean, variance, variance thresholds
  mean     = np.array([[0, 1, 2]], 'float64')
  variance = np.array([[3, 2, 1]], 'float64')
  g["means"]     = mean
  g["variances"] = variance
  g["variance_thresholds"] = 0.0005
  np.testing.assert_equal(g["means"], mean)
  np.testing.assert_equal(g["variances"], variance)
  np.testing.assert_equal(g["variance_thresholds"], np.full((1, 3), 0.0005))

  # Save and read from file
  filename = str(tempfile.mkstemp(".hdf5")[1])
  g.save(HDF5File(filename, 'w'))
  g_loaded = Gaussians.from_hdf5(HDF5File(filename, "r"))
  assert g == g_loaded
  assert (g != g_loaded) is False
  assert g.is_similar_to(g_loaded)

  # Save and read from file using `from_hdf5`
  filename = str(tempfile.mkstemp(".hdf5")[1])
  g.save(hdf5=HDF5File(filename, 'w'))
  g_loaded = Gaussians.from_hdf5(hdf5=HDF5File(filename, "r"))
  assert g == g_loaded
  assert (g != g_loaded) is False
  assert g.is_similar_to(g_loaded)

  # Save and load from file using `load` into an existing Gaussians
  filename = str(tempfile.mkstemp(".hdf5")[1])
  g.save(HDF5File(filename, 'w'))
  g_loaded = Gaussians(np.zeros(shape=(3,)))  # Dummy fill, same size
  g_loaded.load(HDF5File(filename, "r"))
  assert g == g_loaded
  assert (g != g_loaded) is False
  assert g.is_similar_to(g_loaded)

  # Make them different
  g2 = g.copy()
  g["variance_thresholds"] = 0.001
  assert (g == g2) is False
  assert g != g2
  assert not g.is_similar_to(g_loaded)

  # Check likelihood computation
  sample1 = np.array([0, 1, 2], 'float64')
  sample2 = np.array([1, 2, 3], 'float64')
  sample3 = np.array([2, 3, 4], 'float64')
  ref1 = -3.652695334228046
  ref2 = -4.569362000894712
  ref3 = -7.319362000894712
  np.testing.assert_almost_equal(g.log_likelihood(sample1), ref1, decimal=10)
  np.testing.assert_almost_equal(g.log_likelihood(sample2), ref2, decimal=10)
  np.testing.assert_almost_equal(g.log_likelihood(sample3), ref3, decimal=10)


def test_gaussian_variance_threshold():
    # Creating a Gaussian
    gaussian = Gaussians(means=[1, 2], variances=[1, 1], variance_thresholds=[1e-5, 1e-5])

    # Testing variance threshold application
    gaussian["variances"] = np.array([1e-8, 1e-4])
    np.testing.assert_equal(gaussian["variances"], np.array([[1e-5, 1e-4]]))
    gaussian["variance_thresholds"] = np.array([1e-7, 1e-3])
    np.testing.assert_equal(gaussian["variances"], np.array([[1e-5, 1e-3]]))
    gaussian["variances"] = np.array([0, 1e-8])
    np.testing.assert_equal(gaussian["variances"], np.array([[1e-7, 1e-3]]))
    new_gaussian = Gaussians(means=[0,0,0], variances=0, variance_thresholds=1e-5)
    np.testing.assert_equal(new_gaussian["variances"], np.full((1,3), 1e-5))


def test_likelihood():
    """Tests the likelihood computation of a simple normal Gaussian."""
    gaussian = Gaussians(means=[0, 0], variances=[1, 1], variance_thresholds=[1e-5, 1e-5])
    log_likelihood = gaussian.log_likelihood(np.array([[0.4, 0.2]], "float64"))
    np.testing.assert_almost_equal(log_likelihood, [[-1.93787706641]], decimal=10)

    multi_log_likelihood = gaussian.log_likelihood(
        np.array([[0.4, 0.2], [0.1, 0.3]], "float64")
    )
    expected = np.array([[-1.93787706641, -1.88787706640]])
    np.testing.assert_almost_equal(multi_log_likelihood, expected, decimal=10)

    # Default settings
    gaussian_def = Gaussians(means=[1, 2, 3])
    np.testing.assert_equal(gaussian_def["variances"], np.array([[1.0, 1.0, 1.0]]))
    eps = np.finfo(float).eps
    np.testing.assert_equal(
        gaussian_def["variance_thresholds"], np.array([[eps, eps, eps]])
    )
    assert hasattr(gaussian, "log_likelihood")
    log_likelihood = gaussian.log_likelihood(np.array([1.0, 2.0]))
    np.testing.assert_almost_equal(log_likelihood, np.array([[-4.33787706641]]), decimal=10)


def test_multiple_gaussian():
    """Tests the capacity to store and work with multiple gaussians."""
    gaussians = Gaussians(
        means=np.array([[0, 0],[3, 3]])
    )
    expected_means = np.array([[0.0, 0.0], [3.0, 3.0]])
    expected_variances = np.array([[1.0, 1.0], [1.0, 1.0]])
    eps = np.finfo(float).eps
    expected_var_thresholds = np.array([[eps, eps], [eps, eps]])
    np.testing.assert_equal(gaussians["means"], expected_means)
    np.testing.assert_equal(gaussians["variances"], expected_variances)
    np.testing.assert_equal(gaussians["variance_thresholds"], expected_var_thresholds)
    test_samples = np.array([[0.0, 0.0], [1.0, 2.0], [3.0, 3.0]])
    log_likelihoods = gaussians.log_likelihood(test_samples)
    expected_likelihoods = np.array([
        [-1.8378770664, -4.3378770664, -10.8378770664],  # samples ll On gaussian 0
        [-10.8378770664, -4.3378770664, -1.8378770664],  # samples ll On gaussian 1
    ])
    np.testing.assert_almost_equal(log_likelihoods, expected_likelihoods, decimal=10)

    # Variances threshold application
    gaussians["variance_thresholds"] = np.array([[1e-5, 1e-5], [1e-3, 1e-3]])
    expected_variances = np.array([[1, 1], [1, 1]])
    np.testing.assert_equal(gaussians["variances"], expected_variances)
    gaussians["variances"] = np.array([[1e-8, 1e-4], [1e-2, 1e-8]])
    expected_variances = np.array([[1e-5, 1e-4], [1e-2, 1e-3]])
    np.testing.assert_equal(gaussians["variances"], expected_variances)
    gaussians["variance_thresholds"] = np.array([[1e-3, 1e-3], [1e-3, 1e-3]])
    expected_variances = np.array([[1e-3, 1e-3], [1e-2, 1e-3]])
    np.testing.assert_equal(gaussians["variances"], expected_variances)

    # Initializing with different shapes
    gaussians = Gaussians(means=np.zeros(shape=(2,3)), variances=0, variance_thresholds=1e-4)
    np.testing.assert_equal(gaussians["variances"], np.full(shape=(2,3), fill_value=1e-4))


def test_gaussians_attributes_copy():
    """Ensures that the input arrays are correctly copied."""
    means = np.array([[0, 0], [1, 1], [2, 2]], dtype=float)
    variances = np.ones_like(means, dtype=float)
    variance_thresholds = np.full_like(means, 1e-5, dtype=float)
    gaussians = Gaussians(means, variances, variance_thresholds)

    means[0,0] = np.nan
    expected_means = np.array([[0, 0], [1, 1], [2, 2]])
    np.testing.assert_equal(gaussians["means"], expected_means)

    variances[0,0] = np.nan
    expected_variances = np.ones_like(means, dtype=float)
    np.testing.assert_equal(gaussians["variances"], expected_variances)

    variance_thresholds[0,0] = np.nan
    expected_thresholds = np.full_like(means, 1e-5, dtype=float)
    np.testing.assert_equal(gaussians["variance_thresholds"], expected_thresholds)
