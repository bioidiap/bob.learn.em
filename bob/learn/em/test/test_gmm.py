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

from bob.learn.em import GMMStats, GMMMachine

def test_GMMStats():
  # Test a GMMStats
  # Initializes a GMMStats
  gs = GMMStats(2,3)
  log_likelihood = -3.
  T = 57
  n = numpy.array([4.37, 5.31], 'float64')
  sumpx = numpy.array([[1., 2., 3.], [4., 5., 6.]], 'float64')
  sumpxx = numpy.array([[10., 20., 30.], [40., 50., 60.]], 'float64')
  gs.log_likelihood = log_likelihood
  gs.t = T
  gs.n = n
  gs.sum_px = sumpx
  gs.sum_pxx = sumpxx
  assert gs.log_likelihood == log_likelihood
  assert gs.t == T
  assert (gs.n == n).all()
  assert (gs.sum_px == sumpx).all()
  assert (gs.sum_pxx == sumpxx).all()
  assert gs.shape==(2,3)

  # Saves and reads from file
  filename = str(tempfile.mkstemp(".hdf5")[1])
  gs.save(bob.io.base.HDF5File(filename, 'w'))
  gs_loaded = GMMStats(bob.io.base.HDF5File(filename))
  assert gs == gs_loaded
  assert (gs != gs_loaded ) is False
  assert gs.is_similar_to(gs_loaded)
  
  # Saves and reads from file using the keyword argument
  filename = str(tempfile.mkstemp(".hdf5")[1])
  gs.save(hdf5=bob.io.base.HDF5File(filename, 'w'))
  gs_loaded = GMMStats(bob.io.base.HDF5File(filename))
  assert gs == gs_loaded
  assert (gs != gs_loaded ) is False
  assert gs.is_similar_to(gs_loaded)

  # Saves and load from file using the keyword argument
  filename = str(tempfile.mkstemp(".hdf5")[1])
  gs.save(hdf5=bob.io.base.HDF5File(filename, 'w'))
  gs_loaded = GMMStats()
  gs_loaded.load(bob.io.base.HDF5File(filename))
  assert gs == gs_loaded
  assert (gs != gs_loaded ) is False
  assert gs.is_similar_to(gs_loaded)

  # Saves and load from file using the keyword argument
  filename = str(tempfile.mkstemp(".hdf5")[1])
  gs.save(hdf5=bob.io.base.HDF5File(filename, 'w'))
  gs_loaded = GMMStats()
  gs_loaded.load(hdf5=bob.io.base.HDF5File(filename))
  assert gs == gs_loaded
  assert (gs != gs_loaded ) is False
  assert gs.is_similar_to(gs_loaded)
  
  
  # Makes them different
  gs_loaded.t = 58
  assert (gs == gs_loaded ) is False
  assert gs != gs_loaded
  assert (gs.is_similar_to(gs_loaded)) is False
  # Accumulates from another GMMStats
  gs2 = GMMStats(2,3)
  gs2.log_likelihood = log_likelihood
  gs2.t = T
  gs2.n = n
  gs2.sum_px = sumpx
  gs2.sum_pxx = sumpxx
  gs2 += gs
  eps = 1e-8
  assert gs2.log_likelihood == 2*log_likelihood
  assert gs2.t == 2*T
  assert numpy.allclose(gs2.n, 2*n, eps)
  assert numpy.allclose(gs2.sum_px, 2*sumpx, eps)
  assert numpy.allclose(gs2.sum_pxx, 2*sumpxx, eps)

  # Reinit and checks for zeros
  gs_loaded.init()
  assert gs_loaded.log_likelihood == 0
  assert gs_loaded.t == 0
  assert (gs_loaded.n == 0).all()
  assert (gs_loaded.sum_px == 0).all()
  assert (gs_loaded.sum_pxx == 0).all()
  # Resize and checks size
  assert  gs_loaded.shape==(2,3)
  gs_loaded.resize(4,5)  
  assert  gs_loaded.shape==(4,5)
  assert gs_loaded.sum_px.shape[0] == 4
  assert gs_loaded.sum_px.shape[1] == 5

  # Clean-up
  os.unlink(filename)

def test_GMMMachine_1():
  # Test a GMMMachine basic features

  weights   = numpy.array([0.5, 0.5], 'float64')
  weights2   = numpy.array([0.6, 0.4], 'float64')
  means     = numpy.array([[3, 70, 0], [4, 72, 0]], 'float64')
  means2     = numpy.array([[3, 7, 0], [4, 72, 0]], 'float64')
  variances = numpy.array([[1, 10, 1], [2, 5, 2]], 'float64')
  variances2 = numpy.array([[10, 10, 1], [2, 5, 2]], 'float64')
  varianceThresholds = numpy.array([[0, 0, 0], [0, 0, 0]], 'float64')
  varianceThresholds2 = numpy.array([[0.0005, 0.0005, 0.0005], [0, 0, 0]], 'float64')

  # Initializes a GMMMachine
  gmm = GMMMachine(2,3)
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
  assert (gmm.mean_supervector == means.reshape(means.size)).all()
  assert (gmm.variance_supervector == variances.reshape(variances.size)).all()
  newMeans = numpy.array([[3, 70, 2], [4, 72, 2]], 'float64')
  newVariances = numpy.array([[1, 1, 1], [2, 2, 2]], 'float64')


  # Checks particular varianceThresholds-related methods
  varianceThresholds1D = numpy.array([0.3, 1, 0.5], 'float64')
  gmm.set_variance_thresholds(varianceThresholds1D)
  assert (gmm.variance_thresholds[0,:] == varianceThresholds1D).all()
  assert (gmm.variance_thresholds[1,:] == varianceThresholds1D).all()

  gmm.set_variance_thresholds(0.005)
  assert (gmm.variance_thresholds == 0.005).all()

  # Checks Gaussians access
  gmm.means     = newMeans
  gmm.variances = newVariances
  assert (gmm.get_gaussian(0).mean == newMeans[0,:]).all()
  assert (gmm.get_gaussian(1).mean == newMeans[1,:]).all()
  assert (gmm.get_gaussian(0).variance == newVariances[0,:]).all()
  assert (gmm.get_gaussian(1).variance == newVariances[1,:]).all()

  # Checks resize
  gmm.resize(4,5)
  assert gmm.shape == (4,5)

  # Checks comparison
  gmm2 = GMMMachine(gmm)
  gmm3 = GMMMachine(2,3)
  gmm3.weights = weights2
  gmm3.means = means
  gmm3.variances = variances
  #gmm3.varianceThresholds = varianceThresholds
  gmm4 = GMMMachine(2,3)
  gmm4.weights = weights
  gmm4.means = means2
  gmm4.variances = variances
  #gmm4.varianceThresholds = varianceThresholds
  gmm5 = GMMMachine(2,3)
  gmm5.weights = weights
  gmm5.means = means
  gmm5.variances = variances2
  #gmm5.varianceThresholds = varianceThresholds
  gmm6 = GMMMachine(2,3)
  gmm6.weights = weights
  gmm6.means = means
  gmm6.variances = variances
  #gmm6.varianceThresholds = varianceThresholds2

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
  gmm = GMMMachine(2, 2)
  gmm.weights   = numpy.array([0.5, 0.5], 'float64')
  gmm.means     = numpy.array([[3, 70], [4, 72]], 'float64')
  gmm.variances = numpy.array([[1, 10], [2, 5]], 'float64')
  gmm.variance_thresholds = numpy.array([[0, 0], [0, 0]], 'float64')

  stats = GMMStats(2, 2)
  gmm.acc_statistics(arrayset, stats)

  stats_ref = GMMStats(bob.io.base.HDF5File(datafile("stats.hdf5",__name__, path="../data/")))

  assert stats.t == stats_ref.t
  assert numpy.allclose(stats.n, stats_ref.n, atol=1e-10)
  #assert numpy.array_equal(stats.sumPx, stats_ref.sumPx)
  #Note AA: precision error above
  assert numpy.allclose(stats.sum_px, stats_ref.sum_px, atol=1e-10)
  assert numpy.allclose(stats.sum_pxx, stats_ref.sum_pxx, atol=1e-10)

def test_GMMMachine_3():
  # Test a GMMMachine (log-likelihood computation)

  data = bob.io.base.load(datafile('data.hdf5', __name__, path="../data/"))
  gmm = GMMMachine(2, 50)
  gmm.weights   = bob.io.base.load(datafile('weights.hdf5', __name__, path="../data/"))
  gmm.means     = bob.io.base.load(datafile('means.hdf5', __name__, path="../data/"))
  gmm.variances = bob.io.base.load(datafile('variances.hdf5', __name__, path="../data/"))

  # Compare the log-likelihood with the one obtained using Chris Matlab
  # implementation
  matlab_ll_ref = -2.361583051672024e+02
  assert abs(gmm(data) - matlab_ll_ref) < 1e-10
  
  
def test_GMMMachine_4():

  import numpy
  numpy.random.seed(3) # FIXING A SEED

  data = numpy.random.rand(100,50) #Doesn't matter if it is ramdom. The average of 1D array (in python) MUST output the same result for the 2D array (in C++)
  
  gmm = GMMMachine(2, 50)
  gmm.weights   = bob.io.base.load(datafile('weights.hdf5', __name__, path="../data/"))
  gmm.means     = bob.io.base.load(datafile('means.hdf5', __name__, path="../data/"))
  gmm.variances = bob.io.base.load(datafile('variances.hdf5', __name__, path="../data/"))


  ll = 0
  for i in range(data.shape[0]):
    ll += gmm(data[i,:])
  ll /= data.shape[0]
  
  assert ll==gmm(data)
  
  
