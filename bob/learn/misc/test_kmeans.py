#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
# Thu Feb 16 17:57:10 2012 +0200
#
# Copyright (C) 2011-2014 Idiap Research Institute, Martigny, Switzerland

"""Tests the KMeans machine
"""

import os
import numpy
import tempfile

import bob.io.base
from . import KMeansMachine

def equals(x, y, epsilon):
  return (abs(x - y) < epsilon)

def test_KMeansMachine():
  # Test a KMeansMachine

  means = numpy.array([[3, 70, 0], [4, 72, 0]], 'float64')
  mean  = numpy.array([3,70,1], 'float64')

  # Initializes a KMeansMachine
  km = KMeansMachine(2,3)
  km.means = means
  assert km.dim_c == 2
  assert km.dim_d == 3

  # Sets and gets
  assert (km.means == means).all()
  assert (km.get_mean(0) == means[0,:]).all()
  assert (km.get_mean(1) == means[1,:]).all()
  km.set_mean(0, mean)
  assert (km.get_mean(0) == mean).all()

  # Distance and closest mean
  eps = 1e-10
  assert equals( km.get_distance_from_mean(mean, 0), 0, eps)
  assert equals( km.get_distance_from_mean(mean, 1), 6, eps)
  (index, dist) = km.get_closest_mean(mean)
  assert index == 0
  assert equals( dist, 0, eps)
  assert equals( km.get_min_distance(mean), 0, eps)

  # Loads and saves
  filename = str(tempfile.mkstemp(".hdf5")[1])
  km.save(bob.io.base.HDF5File(filename, 'w'))
  km_loaded = KMeansMachine(bob.io.base.HDF5File(filename))
  assert km == km_loaded

  # Resize
  km.resize(4,5)
  assert km.dim_c == 4
  assert km.dim_d == 5

  # Copy constructor and comparison operators
  km.resize(2,3)
  km2 = KMeansMachine(km)
  assert km2 == km
  assert (km2 != km) is False
  assert km2.is_similar_to(km)
  means2 = numpy.array([[3, 70, 0], [4, 72, 2]], 'float64')
  km2.means = means2
  assert (km2 == km) is False
  assert km2 != km
  assert (km2.is_similar_to(km)) is False

  # Clean-up
  os.unlink(filename)
