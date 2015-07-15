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

from bob.learn.em import KMeansMachine, GMMMachine, KMeansTrainer, \
    ML_GMMTrainer, MAP_GMMTrainer

import bob.learn.em

import bob.core
bob.core.log.setup("bob.learn.em")

#, MAP_GMMTrainer

def loadGMM():
  gmm = GMMMachine(2, 2)

  gmm.weights = bob.io.base.load(datafile('gmm.init_weights.hdf5', __name__, path="../data/"))
  gmm.means = bob.io.base.load(datafile('gmm.init_means.hdf5', __name__, path="../data/"))
  gmm.variances = bob.io.base.load(datafile('gmm.init_variances.hdf5', __name__, path="../data/"))
  #gmm.variance_thresholds = numpy.array([[0.001, 0.001],[0.001, 0.001]], 'float64')

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

  ar = bob.io.base.load(datafile("faithful.torch3_f64.hdf5", __name__, path="../data/"))
  gmm = loadGMM()

  ml_gmmtrainer = ML_GMMTrainer(True, True, True)
  #ml_gmmtrainer.train(gmm, ar)
  bob.learn.em.train(ml_gmmtrainer, gmm, ar, convergence_threshold=0.001)

  #config = bob.io.base.HDF5File(datafile('gmm_ML.hdf5", __name__), 'w')
  #gmm.save(config)

  gmm_ref = GMMMachine(bob.io.base.HDF5File(datafile('gmm_ML.hdf5', __name__, path="../data/")))
  gmm_ref_32bit_debug = GMMMachine(bob.io.base.HDF5File(datafile('gmm_ML_32bit_debug.hdf5', __name__, path="../data/")))
  gmm_ref_32bit_release = GMMMachine(bob.io.base.HDF5File(datafile('gmm_ML_32bit_release.hdf5', __name__, path="../data/")))

  assert (gmm == gmm_ref) or (gmm == gmm_ref_32bit_release) or (gmm == gmm_ref_32bit_release)


def test_gmm_ML_2():

  # Trains a GMMMachine with ML_GMMTrainer; compares to an old reference

  ar = bob.io.base.load(datafile('dataNormalized.hdf5', __name__, path="../data/"))

  # Initialize GMMMachine
  gmm = GMMMachine(5, 45)
  gmm.means = bob.io.base.load(datafile('meansAfterKMeans.hdf5', __name__, path="../data/")).astype('float64')
  gmm.variances = bob.io.base.load(datafile('variancesAfterKMeans.hdf5', __name__, path="../data/")).astype('float64')
  gmm.weights = numpy.exp(bob.io.base.load(datafile('weightsAfterKMeans.hdf5', __name__, path="../data/")).astype('float64'))

  threshold = 0.001
  gmm.set_variance_thresholds(threshold)

  # Initialize ML Trainer
  prior = 0.001
  max_iter_gmm = 25
  accuracy = 0.00001
  ml_gmmtrainer = ML_GMMTrainer(True, True, True, prior)

  # Run ML
  #ml_gmmtrainer.train(gmm, ar)
  bob.learn.em.train(ml_gmmtrainer, gmm, ar, max_iterations = max_iter_gmm, convergence_threshold=accuracy)

  # Test results
  # Load torch3vision reference
  meansML_ref = bob.io.base.load(datafile('meansAfterML.hdf5', __name__, path="../data/"))
  variancesML_ref = bob.io.base.load(datafile('variancesAfterML.hdf5', __name__, path="../data/"))
  weightsML_ref = bob.io.base.load(datafile('weightsAfterML.hdf5', __name__, path="../data/"))


  # Compare to current results
  assert equals(gmm.means, meansML_ref, 3e-3)
  assert equals(gmm.variances, variancesML_ref, 3e-3)
  assert equals(gmm.weights, weightsML_ref, 1e-4)



def test_gmm_MAP_1():

  # Train a GMMMachine with MAP_GMMTrainer

  ar = bob.io.base.load(datafile('faithful.torch3_f64.hdf5', __name__, path="../data/"))

  gmm = GMMMachine(bob.io.base.HDF5File(datafile("gmm_ML.hdf5", __name__, path="../data/")))
  gmmprior = GMMMachine(bob.io.base.HDF5File(datafile("gmm_ML.hdf5", __name__, path="../data/")))

  map_gmmtrainer = MAP_GMMTrainer(update_means=True, update_variances=False, update_weights=False, prior_gmm=gmmprior, relevance_factor=4.)

  #map_gmmtrainer.train(gmm, ar)
  bob.learn.em.train(map_gmmtrainer, gmm, ar)

  gmm_ref = GMMMachine(bob.io.base.HDF5File(datafile('gmm_MAP.hdf5', __name__, path="../data/")))

  assert (equals(gmm.means,gmm_ref.means,1e-3) and equals(gmm.variances,gmm_ref.variances,1e-3) and equals(gmm.weights,gmm_ref.weights,1e-3))


def test_gmm_MAP_2():

  # Train a GMMMachine with MAP_GMMTrainer and compare with matlab reference

  data = bob.io.base.load(datafile('data.hdf5', __name__, path="../data/"))
  data = data.reshape((1, data.shape[0])) # make a 2D array out of it
  means = bob.io.base.load(datafile('means.hdf5', __name__, path="../data/"))
  variances = bob.io.base.load(datafile('variances.hdf5', __name__, path="../data/"))
  weights = bob.io.base.load(datafile('weights.hdf5', __name__, path="../data/"))

  gmm = GMMMachine(2,50)
  gmm.means = means
  gmm.variances = variances
  gmm.weights = weights

  map_adapt = MAP_GMMTrainer(update_means=True, update_variances=False, update_weights=False, mean_var_update_responsibilities_threshold=0.,prior_gmm=gmm, relevance_factor=4.)

  gmm_adapted = GMMMachine(2,50)
  gmm_adapted.means = means
  gmm_adapted.variances = variances
  gmm_adapted.weights = weights

  #map_adapt.max_iterations = 1
  #map_adapt.train(gmm_adapted, data)
  bob.learn.em.train(map_adapt, gmm_adapted, data, max_iterations = 1)

  new_means = bob.io.base.load(datafile('new_adapted_mean.hdf5', __name__, path="../data/"))

 # print new_means[0,:]
 # print gmm_adapted.means[:,0]

  # Compare to matlab reference
  assert equals(new_means[0,:], gmm_adapted.means[:,0], 1e-4)
  assert equals(new_means[1,:], gmm_adapted.means[:,1], 1e-4)


def test_gmm_MAP_3():

  # Train a GMMMachine with MAP_GMMTrainer; compares to old reference

  ar = bob.io.base.load(datafile('dataforMAP.hdf5', __name__, path="../data/"))

  # Initialize GMMMachine
  n_gaussians = 5
  n_inputs = 45
  prior_gmm = GMMMachine(n_gaussians, n_inputs)
  prior_gmm.means = bob.io.base.load(datafile('meansAfterML.hdf5', __name__, path="../data/"))
  prior_gmm.variances = bob.io.base.load(datafile('variancesAfterML.hdf5', __name__, path="../data/"))
  prior_gmm.weights = bob.io.base.load(datafile('weightsAfterML.hdf5', __name__, path="../data/"))

  threshold = 0.001
  prior_gmm.set_variance_thresholds(threshold)

  # Initialize MAP Trainer
  relevance_factor = 0.1
  prior = 0.001
  max_iter_gmm = 1
  accuracy = 0.00001
  map_factor = 0.5
  map_gmmtrainer = MAP_GMMTrainer(prior_gmm, alpha=map_factor, update_means=True, update_variances=False, update_weights=False, mean_var_update_responsibilities_threshold=accuracy)
  #map_gmmtrainer.max_iterations = max_iter_gmm
  #map_gmmtrainer.convergence_threshold = accuracy

  gmm = GMMMachine(n_gaussians, n_inputs)
  gmm.set_variance_thresholds(threshold)

  # Train
  #map_gmmtrainer.train(gmm, ar)
  bob.learn.em.train(map_gmmtrainer, gmm, ar, max_iterations = max_iter_gmm, convergence_threshold=prior)

  # Test results
  # Load torch3vision reference
  meansMAP_ref = bob.io.base.load(datafile('meansAfterMAP.hdf5', __name__, path="../data/"))
  variancesMAP_ref = bob.io.base.load(datafile('variancesAfterMAP.hdf5', __name__, path="../data/"))
  weightsMAP_ref = bob.io.base.load(datafile('weightsAfterMAP.hdf5', __name__, path="../data/"))

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

  ar = bob.io.base.load(datafile('dataforMAP.hdf5', __name__, path="../data/"))

  # Initialize GMMMachine
  n_gaussians = 5
  n_inputs = 45
  gmm = GMMMachine(n_gaussians, n_inputs)
  gmm.means = bob.io.base.load(datafile('meansAfterML.hdf5', __name__, path="../data/"))
  gmm.variances = bob.io.base.load(datafile('variancesAfterML.hdf5', __name__, path="../data/"))
  gmm.weights = bob.io.base.load(datafile('weightsAfterML.hdf5', __name__, path="../data/"))

  threshold = 0.001
  gmm.set_variance_thresholds(threshold)

  # Test against the model
  score_mean_ref = -1.50379e+06
  score = 0.
  for v in ar: score += gmm(v)
  score /= len(ar)

  # Compare current results to torch3vision
  assert abs(score-score_mean_ref)/score_mean_ref<1e-4


def test_custom_trainer():

  # Custom python trainer

  ar = bob.io.base.load(datafile("faithful.torch3_f64.hdf5", __name__, path="../data/"))

  mytrainer = MyTrainer1()

  machine = KMeansMachine(2, 2)
  mytrainer.train(machine, ar)

  for i in range(0, 2):
    assert (ar[i+1] == machine.means[i, :]).all()
    
    
    
def test_EMPCA():

  # Tests our Probabilistic PCA trainer for linear machines for a simple
  # problem:
  ar=numpy.array([
    [1, 2, 3],
    [2, 4, 19],
    [3, 6, 5],
    [4, 8, 13],
    ], dtype='float64')

  # Expected llh 1 and 2 (Reference values)
  exp_llh1 =  -32.8443
  exp_llh2 =  -30.8559

  # Do two iterations of EM to check the training procedure
  T = bob.learn.em.EMPCATrainer()
  m = bob.learn.linear.Machine(3,2)
  # Initialization of the trainer
  T.initialize(m, ar)
  # Sets ('random') initialization values for test purposes
  w_init = numpy.array([1.62945, 0.270954, 1.81158, 1.67002, 0.253974,
    1.93774], 'float64').reshape(3,2)
  sigma2_init = 1.82675
  m.weights = w_init
  T.sigma2 = sigma2_init
  # Checks that the log likehood matches the reference one
  # This should be sufficient to check everything as it requires to use
  # the new value of W and sigma2
  # This does an E-Step, M-Step, computes the likelihood, and compares it to
  # the reference value obtained using matlab
  T.e_step(m, ar)
  T.m_step(m, ar)
  llh1 = T.compute_likelihood(m)
  assert abs(exp_llh1 - llh1) < 2e-4
  T.e_step(m, ar)
  T.m_step(m, ar)
  llh2 = T.compute_likelihood(m)
  assert abs(exp_llh2 - llh2) < 2e-4    
    
