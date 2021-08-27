#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
# Thu Feb 16 16:54:45 2012 +0200
#
# Copyright (C) 2011-2013 Idiap Research Institute, Martigny, Switzerland

"""Tests the Gaussian class
"""

import numpy

from bob.learn.em.mixture import Gaussian

def equals(x, y, epsilon):
    return (abs(x - y) < epsilon)

def test_gaussian():
    gaussian = Gaussian(mean=[1,2], variance=[1,1], variance_threshold=[1e-5, 1e-5])

    assert numpy.array_equal(gaussian["m"], numpy.array([1,2])), gaussian["m"]
    assert numpy.array_equal(gaussian.m, numpy.array([1,2])), gaussian.m

    gaussian["m"] = numpy.array([0,0])

    assert numpy.array_equal(gaussian["m"], numpy.array([0,0])), gaussian["m"]
    assert numpy.array_equal(gaussian.m, numpy.array([0,0])), gaussian.m

    # Test the likelihood computation of a simple normal Gaussian
    log_likelihood = gaussian.log_likelihood(numpy.array([0.4, 0.2], 'float64')).compute()
    assert equals(log_likelihood, -1.93787706641, 1e-10)

    multi_log_likelihood = gaussian.log_likelihood(numpy.array([[0.4, 0.2],[0.1,0.3]], 'float64'))
    assert multi_log_likelihood.shape == (2,), multi_log_likelihood.shape
    expected = numpy.array([-1.93787706641, -1.88787706640])
    assert numpy.allclose(multi_log_likelihood, expected, atol=1e-10)

    gaussian_def = Gaussian(mean=[1,2,3])
    assert numpy.array_equal(gaussian_def.variance, numpy.array([1.,1.,1.])), gaussian_def.variance
    assert numpy.array_equal(gaussian_def.variance_threshold, numpy.array([1.e-5,1.e-5,1.e-5])), gaussian_def.variance_threshold


def test_multi_gaussian():
    gaussians = numpy.array(
        [
            Gaussian([0,0]),
            Gaussian([1,1]),
        ]
    )
    expected = numpy.array([[0.,0.],[1.,1.]])
    assert numpy.array_equal(gaussians["m"], expected), gaussians["m"]


def test_gaussian_str():
    gaussian = Gaussian(mean=[0,0], variance=[1,1], variance_threshold=[1e-5, 1e-5])
    assert str(gaussian) == "Gaussian:\n\tmean: [0. 0.]\n\tvar:  [1. 1.]\n\tvar_threshold: [1.e-05 1.e-05]", str(gaussian)
