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


def test_gaussian():
    # Setting values
    gaussian = Gaussian(mean=[1, 2], variance=[1, 1], variance_threshold=[1e-5, 1e-5])

    np.testing.assert_equal(gaussian["mean"], np.array([1, 2]))

    # Mean
    gaussian["mean"] = np.array([0, 0])

    np.testing.assert_equal(gaussian["mean"], np.array([0, 0]))

    # Variance
    gaussian["variance"] = np.array([1e-3, 1e-4])
    np.testing.assert_equal(gaussian["variance"], np.array([1e-3, 1e-4]))

    # Testing variance threshold application
    gaussian["variance"] = np.array([1e-8, 1e-4])
    np.testing.assert_equal(gaussian["variance"], np.array([1e-5, 1e-4]))
    gaussian["variance_threshold"] = np.array([1e-7, 1e-3])
    np.testing.assert_equal(gaussian["variance"], np.array([1e-5, 1e-3]))
    gaussian["variance"] = np.array([1e-8, 1e-8])
    np.testing.assert_equal(gaussian["variance"], np.array([1e-7, 1e-3]))

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
    np.testing.assert_equal(
        gaussian_def["variance_threshold"], np.array([1.0e-5, 1.0e-5, 1.0e-5])
    )
    assert hasattr(gaussian, "log_likelihood")
    log_likelihood = gaussian.log_likelihood(np.array([1.0, 2.0]))
    np.testing.assert_almost_equal(log_likelihood, -4.33787706641, decimal=10)


def test_multi_gaussian():
    gaussians = MultiGaussian(
        means=np.array([
            [0, 0],
            [3, 3],
        ])
    )
    expected_means = np.array([[0.0, 0.0], [3.0, 3.0]])
    expected_variances = np.array([[1.0, 1.0], [1.0, 1.0]])
    expected_variance_thresholds = np.array([[1e-5, 1e-5], [1e-5, 1e-5]])
    np.testing.assert_equal(gaussians["mean"], expected_means)
    np.testing.assert_equal(gaussians["variance"], expected_variances)
    np.testing.assert_equal(
        gaussians["variance_threshold"], expected_variance_thresholds
    )
    assert hasattr(gaussians[0], "log_likelihood")
    gaussians[0].log_likelihood(np.array([1.0, 2.0]))

    # Variances threshold application
    gaussians["variance_threshold"] = np.array([[1e-5, 1e-5], [1e-3, 1e-3]])
    gaussians["variance"] = np.array([[1e-8, 1e-4], [1e-4, 1e-8]])
    expected_variances = np.array([[1e-5, 1e-4], [1e-3, 1e-3]])
    np.testing.assert_equal(gaussians["variance"], expected_variances)
