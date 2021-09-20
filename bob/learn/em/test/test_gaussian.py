#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
# Thu Feb 16 16:54:45 2012 +0200
#
# Copyright (C) 2011-2013 Idiap Research Institute, Martigny, Switzerland

"""Tests the Gaussian class
"""

import numpy as np

from bob.learn.em.mixture import Gaussian
from bob.learn.em.mixture import MultiGaussian


def equals(x, y, epsilon):
    return abs(x - y) < epsilon


def test_GaussianMachine():
  # Test a GaussianMachine more thoroughly

  # Initializes a Gaussian with zero mean and unit variance
  g = Gaussian(mean=np.zeros((3,)))
  np.testing.assert_equal(g["mean"], np.zeros((3,)))
  np.testing.assert_equal(g["variance"], np.ones((3,)))
  np.testing.assert_equal(g.shape, (3,))

  # Set and check mean, variance, variance thresholds
  mean     = np.array([0, 1, 2], 'float64')
  variance = np.array([3, 2, 1], 'float64')
  g["mean"]     = mean
  g["variance"] = variance
  g["variance_threshold"] = 0.0005
  np.testing.assert_equal(g["mean"], mean)
  np.testing.assert_equal(g["variance"], variance)
  np.testing.assert_equal(g["variance_threshold"], np.full((3,), 0.0005))

  # Make them different
  g2 = g.copy()
  g["variance_threshold"] = 0.001
  assert (g == g2) is False
  assert g != g2

  # Check likelihood computation
  sample1 = np.array([0, 1, 2], 'float64')
  sample2 = np.array([1, 2, 3], 'float64')
  sample3 = np.array([2, 3, 4], 'float64')
  ref1 = -3.652695334228046
  ref2 = -4.569362000894712
  ref3 = -7.319362000894712
  eps = 1e-10
  assert equals(g.log_likelihood(sample1), ref1, eps)
  assert equals(g.log_likelihood(sample2), ref2, eps)
  assert equals(g.log_likelihood(sample3), ref3, eps)


def test_gaussian_variance_threshold():
    # Creating a Gaussian
    gaussian = Gaussian(mean=[1, 2], variance=[1, 1], variance_threshold=[1e-5, 1e-5])

    # Testing variance threshold application
    gaussian["variance"] = np.array([1e-8, 1e-4])
    np.testing.assert_equal(gaussian["variance"], np.array([1e-5, 1e-4]))
    gaussian["variance_threshold"] = np.array([1e-7, 1e-3])
    np.testing.assert_equal(gaussian["variance"], np.array([1e-5, 1e-3]))
    gaussian["variance"] = np.array([0, 1e-8])
    np.testing.assert_equal(gaussian["variance"], np.array([1e-7, 1e-3]))
    new_gaussian = Gaussian(mean=[0,0,0], variance=0, variance_threshold=1e-5)
    np.testing.assert_equal(new_gaussian["variance"], np.full((3,), 1e-5))


def test_likelihood():
    # Test the likelihood computation of a simple normal Gaussian
    gaussian = Gaussian(mean=[0, 0], variance=[1, 1], variance_threshold=[1e-5, 1e-5])
    log_likelihood = gaussian.log_likelihood(np.array([0.4, 0.2], "float64"))
    np.testing.assert_almost_equal(log_likelihood.compute(), -1.93787706641, decimal=10)

    multi_log_likelihood = gaussian.log_likelihood(
        np.array([[0.4, 0.2], [0.1, 0.3]], "float64")
    )
    expected = np.array([-1.93787706641, -1.88787706640])
    np.testing.assert_almost_equal(multi_log_likelihood, expected, decimal=10)

    # Default settings
    gaussian_def = Gaussian(mean=[1, 2, 3])
    np.testing.assert_equal(gaussian_def["variance"], np.array([1.0, 1.0, 1.0]))
    eps = np.finfo(float).eps
    np.testing.assert_equal(
        gaussian_def["variance_threshold"], np.array([eps, eps, eps])
    )
    assert hasattr(gaussian, "log_likelihood")
    log_likelihood = gaussian.log_likelihood(np.array([1.0, 2.0]))
    np.testing.assert_almost_equal(log_likelihood, -4.33787706641, decimal=10)


def test_multi_gaussian():
    gaussians = MultiGaussian(
        means=np.array([[0, 0],[3, 3]])
    )
    expected_means = np.array([[0.0, 0.0], [3.0, 3.0]])
    expected_variances = np.array([[1.0, 1.0], [1.0, 1.0]])
    eps = np.finfo(float).eps
    expected_variance_thresholds = np.array([[eps, eps], [eps, eps]])
    np.testing.assert_equal(gaussians["mean"], expected_means)
    np.testing.assert_equal(gaussians["variance"], expected_variances)
    np.testing.assert_equal(
        gaussians["variance_threshold"], expected_variance_thresholds
    )
    assert hasattr(gaussians[0], "log_likelihood")
    gaussians[0].log_likelihood(np.array([1.0, 2.0]))

    # Variances threshold application
    gaussians["variance_threshold"] = np.array([[1e-5, 1e-5], [1e-3, 1e-3]])
    expected_variances = np.array([[1, 1], [1, 1]])
    np.testing.assert_equal(gaussians["variance"], expected_variances)
    gaussians["variance"] = np.array([[1e-8, 1e-4], [1e-2, 1e-8]])
    expected_variances = np.array([[1e-5, 1e-4], [1e-2, 1e-3]])
    np.testing.assert_equal(gaussians["variance"], expected_variances)
    gaussians["variance_threshold"] = np.array([[1e-3, 1e-3], [1e-3, 1e-3]])
    expected_variances = np.array([[1e-3, 1e-3], [1e-2, 1e-3]])
    np.testing.assert_equal(gaussians["variance"], expected_variances)

    # Initializing with different shapes
    gaussians = MultiGaussian(means=np.zeros(shape=(2,3)), variances=0, variance_thresholds=1e-4)
    np.testing.assert_equal(gaussians["variance"], np.full(shape=(2,3), fill_value=1e-4))
