#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
# Thu Feb 16 16:54:45 2012 +0200
#
# Copyright (C) 2011-2013 Idiap Research Institute, Martigny, Switzerland

"""Tests the Gaussian machine
"""

import os
import numpy
import tempfile

import bob.io.base

from bob.learn.em import Gaussian

def equals(x, y, epsilon):
  return (abs(x - y) < epsilon)

def test_GaussianNormal():
  # Test the likelihood computation of a simple normal Gaussian
  gaussian = Gaussian(2)
  # By default, initialized with zero mean and unit variance
  logLH = gaussian.log_likelihood(numpy.array([0.4, 0.2], 'float64'))
  assert equals(logLH, -1.93787706641, 1e-10)

def test_GaussianMachine():
  # Test a GaussianMachine more thoroughly

  # Initializes a Gaussian with zero mean and unit variance
  g = Gaussian(3)
  assert (g.mean == 0.0).all()
  assert (g.variance == 1.0).all()
  assert g.shape == (3,)

  # Set and check mean, variance, variance thresholds
  mean     = numpy.array([0, 1, 2], 'float64')
  variance = numpy.array([3, 2, 1], 'float64')
  g.mean     = mean
  g.variance = variance
  g.set_variance_thresholds(0.0005)
  assert (g.mean == mean).all()
  assert (g.variance == variance).all()
  assert (g.variance_thresholds == 0.0005).all()

  # Save and read from file
  filename = str(tempfile.mkstemp(".hdf5")[1])
  g.save(bob.io.base.HDF5File(filename, 'w'))
  g_loaded = Gaussian(bob.io.base.HDF5File(filename))
  assert g == g_loaded
  assert (g != g_loaded ) is False
  assert g.is_similar_to(g_loaded)
  
  # Save and read from file using the keyword argument
  filename = str(tempfile.mkstemp(".hdf5")[1])
  g.save(hdf5=bob.io.base.HDF5File(filename, 'w'))
  g_loaded = Gaussian(hdf5=bob.io.base.HDF5File(filename))
  assert g == g_loaded
  assert (g != g_loaded ) is False
  assert g.is_similar_to(g_loaded)

  # Save and loading from file using the keyword argument
  filename = str(tempfile.mkstemp(".hdf5")[1])
  g.save(bob.io.base.HDF5File(filename, 'w'))
  g_loaded = bob.learn.em.Gaussian()
  g_loaded.load(bob.io.base.HDF5File(filename))
  assert g == g_loaded
  assert (g != g_loaded ) is False
  assert g.is_similar_to(g_loaded)

  # Save and loading from file using the keyword argument
  filename = str(tempfile.mkstemp(".hdf5")[1])
  g.save(bob.io.base.HDF5File(filename, 'w'))
  g_loaded = bob.learn.em.Gaussian()
  g_loaded.load(hdf5=bob.io.base.HDF5File(filename))
  assert g == g_loaded
  assert (g != g_loaded ) is False
  assert g.is_similar_to(g_loaded)


  # Make them different
  g_loaded.set_variance_thresholds(0.001)
  assert (g == g_loaded ) is False
  assert g != g_loaded

  # Check likelihood computation
  sample1 = numpy.array([0, 1, 2], 'float64')
  sample2 = numpy.array([1, 2, 3], 'float64')
  sample3 = numpy.array([2, 3, 4], 'float64')
  ref1 = -3.652695334228046
  ref2 = -4.569362000894712
  ref3 = -7.319362000894712
  eps = 1e-10
  assert equals(g.log_likelihood(sample1), ref1, eps)
  assert equals(g.log_likelihood(sample2), ref2, eps)
  assert equals(g.log_likelihood(sample3), ref3, eps)

  # Check resize and assignment
  g.resize(5)
  assert g.shape == (5,)
  g2 = Gaussian()
  g2 = g
  assert g == g2
  assert (g != g2 ) is False
  g3 = Gaussian(g)
  assert g == g3
  assert (g != g3 ) is False

  # Clean-up
  os.unlink(filename)
