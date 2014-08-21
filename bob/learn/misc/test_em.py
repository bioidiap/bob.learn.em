#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Francois Moulin <Francois.Moulin@idiap.ch>
# Tue May 10 11:35:58 2011 +0200
#
# Copyright (C) 2011-2013 Idiap Research Institute, Martigny, Switzerland

"""Test trainer package
"""
import unittest
import numpy

import bob.io.base
from bob.io.base.test_utils import datafile

from . import KMeansMachine, GMMMachine, KMeansTrainer, \
    ML_GMMTrainer, MAP_GMMTrainer

def loadGMM():
  gmm = GMMMachine(2, 2)

  gmm.weights = bob.io.base.load(datafile('gmm.init_weights.hdf5', __name__))
  gmm.means = bob.io.base.load(datafile('gmm.init_means.hdf5', __name__))
  gmm.variances = bob.io.base.load(datafile('gmm.init_variances.hdf5', __name__))
  gmm.variance_threshold = numpy.array([0.001, 0.001], 'float64')

  return gmm

def equals(x, y, epsilon):
  return (abs(x - y) < epsilon).all()

class MyTrainer1(KMeansTrainer):
  """Simple example of python trainer: """

  def __init__(self):
    KMeansTrainer.__init__(self)

  def train(self, machine, data):
    a = numpy.ndarray((2, 2), 'float64')
    a[0, :] = data[1]
    a[1, :] = data[2]
    machine.means = a

def test_gmm_ML_1():

  # Trains a GMMMachine with ML_GMMTrainer

  ar = bob.io.base.load(datafile("faithful.torch3_f64.hdf5", __name__))

  gmm = loadGMM()

  ml_gmmtrainer = ML_GMMTrainer(True, True, True)
  ml_gmmtrainer.train(gmm, ar)

  #config = bob.io.base.HDF5File(datafile('gmm_ML.hdf5", __name__), 'w')
  #gmm.save(config)

  gmm_ref = GMMMachine(bob.io.base.HDF5File(datafile('gmm_ML.hdf5', __name__)))
  gmm_ref_32bit_debug = GMMMachine(bob.io.base.HDF5File(datafile('gmm_ML_32bit_debug.hdf5', __name__)))
  gmm_ref_32bit_release = GMMMachine(bob.io.base.HDF5File(datafile('gmm_ML_32bit_release.hdf5', __name__)))

  assert (gmm == gmm_ref) or (gmm == gmm_ref_32bit_release) or (gmm == gmm_ref_32bit_debug)

def test_gmm_ML_2():

  # Trains a GMMMachine with ML_GMMTrainer; compares to an old reference

  ar = bob.io.base.load(datafile('dataNormalized.hdf5', __name__))

  # Initialize GMMMachine
  gmm = GMMMachine(5, 45)
  gmm.means = bob.io.base.load(datafile('meansAfterKMeans.hdf5', __name__)).astype('float64')
  gmm.variances = bob.io.base.load(datafile('variancesAfterKMeans.hdf5', __name__)).astype('float64')
  gmm.weights = numpy.exp(bob.io.base.load(datafile('weightsAfterKMeans.hdf5', __name__)).astype('float64'))

  threshold = 0.001
  gmm.set_variance_thresholds(threshold)

  # Initialize ML Trainer
  prior = 0.001
  max_iter_gmm = 25
  accuracy = 0.00001
  ml_gmmtrainer = ML_GMMTrainer(True, True, True, prior)
  ml_gmmtrainer.max_iterations = max_iter_gmm
  ml_gmmtrainer.convergence_threshold = accuracy

  # Run ML
  ml_gmmtrainer.train(gmm, ar)

  # Test results
  # Load torch3vision reference
  meansML_ref = bob.io.base.load(datafile('meansAfterML.hdf5', __name__))
  variancesML_ref = bob.io.base.load(datafile('variancesAfterML.hdf5', __name__))
  weightsML_ref = bob.io.base.load(datafile('weightsAfterML.hdf5', __name__))

  # Compare to current results
  assert equals(gmm.means, meansML_ref, 3e-3)
  assert equals(gmm.variances, variancesML_ref, 3e-3)
  assert equals(gmm.weights, weightsML_ref, 1e-4)

def test_gmm_MAP_1():

  # Train a GMMMachine with MAP_GMMTrainer

  ar = bob.io.base.load(datafile('faithful.torch3_f64.hdf5', __name__))

  gmm = GMMMachine(bob.io.base.HDF5File(datafile("gmm_ML.hdf5", __name__)))
  gmmprior = GMMMachine(bob.io.base.HDF5File(datafile("gmm_ML.hdf5", __name__)))

  map_gmmtrainer = MAP_GMMTrainer(16)
  map_gmmtrainer.set_prior_gmm(gmmprior)
  map_gmmtrainer.train(gmm, ar)

  #config = bob.io.base.HDF5File(datafile('gmm_MAP.hdf5", 'w', __name__))
  #gmm.save(config)

  gmm_ref = GMMMachine(bob.io.base.HDF5File(datafile('gmm_MAP.hdf5', __name__)))
  #gmm_ref_32bit_release = GMMMachine(bob.io.base.HDF5File(datafile('gmm_MAP_32bit_release.hdf5', __name__)))

  assert (equals(gmm.means,gmm_ref.means,1e-3) and equals(gmm.variances,gmm_ref.variances,1e-3) and equals(gmm.weights,gmm_ref.weights,1e-3))

def test_gmm_MAP_2():

  # Train a GMMMachine with MAP_GMMTrainer and compare with matlab reference

  map_adapt = MAP_GMMTrainer(4., True, False, False, 0.)
  data = bob.io.base.load(datafile('data.hdf5', __name__))
  data = data.reshape((1, data.shape[0])) # make a 2D array out of it
  means = bob.io.base.load(datafile('means.hdf5', __name__))
  variances = bob.io.base.load(datafile('variances.hdf5', __name__))
  weights = bob.io.base.load(datafile('weights.hdf5', __name__))

  gmm = GMMMachine(2,50)
  gmm.means = means
  gmm.variances = variances
  gmm.weights = weights

  map_adapt.set_prior_gmm(gmm)

  gmm_adapted = GMMMachine(2,50)
  gmm_adapted.means = means
  gmm_adapted.variances = variances
  gmm_adapted.weights = weights

  map_adapt.max_iterations = 1
  map_adapt.train(gmm_adapted, data)

  new_means = bob.io.base.load(datafile('new_adapted_mean.hdf5', __name__))

  # Compare to matlab reference
  assert equals(new_means[0,:], gmm_adapted.means[:,0], 1e-4)
  assert equals(new_means[1,:], gmm_adapted.means[:,1], 1e-4)

def test_gmm_MAP_3():

  # Train a GMMMachine with MAP_GMMTrainer; compares to old reference

  ar = bob.io.base.load(datafile('dataforMAP.hdf5', __name__))

  # Initialize GMMMachine
  n_gaussians = 5
  n_inputs = 45
  prior_gmm = GMMMachine(n_gaussians, n_inputs)
  prior_gmm.means = bob.io.base.load(datafile('meansAfterML.hdf5', __name__))
  prior_gmm.variances = bob.io.base.load(datafile('variancesAfterML.hdf5', __name__))
  prior_gmm.weights = bob.io.base.load(datafile('weightsAfterML.hdf5', __name__))

  threshold = 0.001
  prior_gmm.set_variance_thresholds(threshold)

  # Initialize MAP Trainer
  relevance_factor = 0.1
  prior = 0.001
  max_iter_gmm = 1
  accuracy = 0.00001
  map_factor = 0.5
  map_gmmtrainer = MAP_GMMTrainer(relevance_factor, True, False, False, prior)
  map_gmmtrainer.max_iterations = max_iter_gmm
  map_gmmtrainer.convergence_threshold = accuracy
  map_gmmtrainer.set_prior_gmm(prior_gmm)
  map_gmmtrainer.set_t3_map(map_factor);

  gmm = GMMMachine(n_gaussians, n_inputs)
  gmm.set_variance_thresholds(threshold)

  # Train
  map_gmmtrainer.train(gmm, ar)

  # Test results
  # Load torch3vision reference
  meansMAP_ref = bob.io.base.load(datafile('meansAfterMAP.hdf5', __name__))
  variancesMAP_ref = bob.io.base.load(datafile('variancesAfterMAP.hdf5', __name__))
  weightsMAP_ref = bob.io.base.load(datafile('weightsAfterMAP.hdf5', __name__))

  # Compare to current results
  # Gaps are quite large. This might be explained by the fact that there is no
  # adaptation of a given Gaussian in torch3 when the corresponding responsibilities
  # are below the responsibilities threshold
  assert equals(gmm.means, meansMAP_ref, 2e-1)
  assert equals(gmm.variances, variancesMAP_ref, 1e-4)
  assert equals(gmm.weights, weightsMAP_ref, 1e-4)

def test_gmm_test():

  # Tests a GMMMachine by computing scores against a model and compare to
  # an old reference

  ar = bob.io.base.load(datafile('dataforMAP.hdf5', __name__))

  # Initialize GMMMachine
  n_gaussians = 5
  n_inputs = 45
  gmm = GMMMachine(n_gaussians, n_inputs)
  gmm.means = bob.io.base.load(datafile('meansAfterML.hdf5', __name__))
  gmm.variances = bob.io.base.load(datafile('variancesAfterML.hdf5', __name__))
  gmm.weights = bob.io.base.load(datafile('weightsAfterML.hdf5', __name__))

  threshold = 0.001
  gmm.set_variance_thresholds(threshold)

  # Test against the model
  score_mean_ref = -1.50379e+06
  score = 0.
  for v in ar: score += gmm.forward(v)
  score /= len(ar)

  # Compare current results to torch3vision
  assert abs(score-score_mean_ref)/score_mean_ref<1e-4

def test_custom_trainer():

  # Custom python trainer

  ar = bob.io.base.load(datafile("faithful.torch3_f64.hdf5", __name__))

  mytrainer = MyTrainer1()

  machine = KMeansMachine(2, 2)
  mytrainer.train(machine, ar)

  for i in range(0, 2):
    assert (ar[i+1] == machine.means[i, :]).all()
