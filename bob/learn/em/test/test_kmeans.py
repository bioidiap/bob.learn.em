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
from bob.learn.em import KMeansMachine

def equals(x, y, epsilon):
  return (abs(x - y) < epsilon)

def test_KMeansMachine():
  # Test a KMeansMachine

  means = numpy.array([[3, 70, 0], [4, 72, 0]], 'float64')
  mean  = numpy.array([3,70,1], 'float64')

  # Initializes a KMeansMachine
  km = KMeansMachine(2,3)
  km.means = means
  assert km.shape == (2,3)

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
  assert km.shape == (4,5)

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
  
  
def test_KMeansMachine2():
  kmeans             = bob.learn.em.KMeansMachine(2,2)
  kmeans.means       = numpy.array([[1.2,1.3],[0.2,-0.3]])

  data               = numpy.array([
                                  [1.,1],
                                  [1.2, 3],
                                  [0,0],
                                  [0.3,0.2],
                                  [0.2,0]
                                 ])
  variances, weights = kmeans.get_variances_and_weights_for_each_cluster(data)

  variances_result = numpy.array([[ 0.01,1.],
                                  [ 0.01555556, 0.00888889]])
  weights_result = numpy.array([ 0.4, 0.6])
  
  assert equals(weights_result,weights, 1e-3).all()
  assert equals(variances_result,variances,1e-3).all()
 
