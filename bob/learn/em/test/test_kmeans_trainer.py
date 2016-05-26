#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
# Fri Jan 18 12:46:00 2013 +0200
#
# Copyright (C) 2011-2014 Idiap Research Institute, Martigny, Switzerland

"""Test K-Means algorithm
"""
import numpy

import bob.core
import bob.io
from bob.io.base.test_utils import datafile

from bob.learn.em import KMeansMachine, KMeansTrainer

def equals(x, y, epsilon):
  return (abs(x - y) < epsilon).all()

def kmeans_plus_plus(machine, data, seed):
  """Python implementation of K-Means++ (initialization)"""
  n_data = data.shape[0]
  rng = bob.core.random.mt19937(seed)
  u = bob.core.random.uniform('int32', 0, n_data-1)
  index = u(rng)
  machine.set_mean(0, data[index,:])
  weights = numpy.zeros(shape=(n_data,), dtype=numpy.float64)

  for m in range(1,machine.dim_c):
    for s in range(n_data):
      s_cur = data[s,:]
      w_cur = machine.get_distance_from_mean(s_cur, 0)
      for i in range(m):
        w_cur = min(machine.get_distance_from_mean(s_cur, i), w_cur)
      weights[s] = w_cur
    weights *= weights
    weights /= numpy.sum(weights)
    d = bob.core.random.discrete('int32', weights)
    index = d(rng)
    machine.set_mean(m, data[index,:])


def NormalizeStdArray(path):
  array = bob.io.base.load(path).astype('float64')
  std = array.std(axis=0)
  return (array/std, std)

def multiplyVectorsByFactors(matrix, vector):
  for i in range(0, matrix.shape[0]):
    for j in range(0, matrix.shape[1]):
      matrix[i, j] *= vector[j]

def flipRows(array):
  if len(array.shape) == 2:
    return numpy.array([numpy.array(array[1, :]), numpy.array(array[0, :])], 'float64')
  elif len(array.shape) == 1:
    return numpy.array([array[1], array[0]], 'float64')
  else:
    raise Exception('Input type not supportd by flipRows')

if hasattr(KMeansTrainer, 'KMEANS_PLUS_PLUS'):
  def test_kmeans_plus_plus():

    # Tests the K-Means++ initialization
    dim_c = 5
    dim_d = 7
    n_samples = 150
    data = numpy.random.randn(n_samples,dim_d)
    seed = 0

    # C++ implementation
    machine = KMeansMachine(dim_c, dim_d)
    trainer = KMeansTrainer()
    trainer.rng = bob.core.random.mt19937(seed)
    trainer.initialization_method = 'KMEANS_PLUS_PLUS'
    trainer.initialize(machine, data)

    # Python implementation
    py_machine = KMeansMachine(dim_c, dim_d)
    kmeans_plus_plus(py_machine, data, seed)
    assert equals(machine.means, py_machine.means, 1e-8)

def test_kmeans_noduplicate():
  # Data/dimensions
  dim_c = 2
  dim_d = 3
  seed = 0
  data = numpy.array([[1,2,3],[1,2,3],[1,2,3],[4,5,6.]])
  # Defines machine and trainer
  machine = KMeansMachine(dim_c, dim_d)
  trainer = KMeansTrainer()
  rng = bob.core.random.mt19937(seed)
  trainer.initialization_method = 'RANDOM_NO_DUPLICATE'
  trainer.initialize(machine, data, rng)
  # Makes sure that the two initial mean vectors selected are different
  assert equals(machine.get_mean(0), machine.get_mean(1), 1e-8) == False


def test_kmeans_a():

  # Trains a KMeansMachine
  # This files contains draws from two 1D Gaussian distributions:
  #   * 100 samples from N(-10,1)
  #   * 100 samples from N(10,1)
  data = bob.io.base.load(datafile("samplesFrom2G_f64.hdf5", __name__, path="../data/"))

  machine = KMeansMachine(2, 1)

  trainer = KMeansTrainer()
  #trainer.train(machine, data)
  bob.learn.em.train(trainer,machine,data)

  [variances, weights] = machine.get_variances_and_weights_for_each_cluster(data)
  variances_b = numpy.ndarray(shape=(2,1), dtype=numpy.float64)
  weights_b = numpy.ndarray(shape=(2,), dtype=numpy.float64)
  machine.__get_variances_and_weights_for_each_cluster_init__(variances_b, weights_b)
  machine.__get_variances_and_weights_for_each_cluster_acc__(data, variances_b, weights_b)
  machine.__get_variances_and_weights_for_each_cluster_fin__(variances_b, weights_b)
  m1 = machine.get_mean(0)
  m2 = machine.get_mean(1)

  ## Check means [-10,10] / variances [1,1] / weights [0.5,0.5]
  if(m1<m2): means=numpy.array(([m1[0],m2[0]]), 'float64')
  else: means=numpy.array(([m2[0],m1[0]]), 'float64')
  assert equals(means, numpy.array([-10.,10.]), 2e-1)
  assert equals(variances, numpy.array([1.,1.]), 2e-1)
  assert equals(weights, numpy.array([0.5,0.5]), 1e-3)

  assert equals(variances, variances_b, 1e-8)
  assert equals(weights, weights_b, 1e-8)



def test_kmeans_b():

  # Trains a KMeansMachine
  (arStd,std) = NormalizeStdArray(datafile("faithful.torch3.hdf5", __name__, path="../data/"))

  machine = KMeansMachine(2, 2)

  trainer = KMeansTrainer()
  #trainer.seed = 1337
  bob.learn.em.train(trainer,machine, arStd, convergence_threshold=0.001)

  [variances, weights] = machine.get_variances_and_weights_for_each_cluster(arStd)

  means = numpy.array(machine.means)
  variances = numpy.array(variances)

  multiplyVectorsByFactors(means, std)
  multiplyVectorsByFactors(variances, std ** 2)

  gmmWeights = bob.io.base.load(datafile('gmm.init_weights.hdf5', __name__, path="../data/"))
  gmmMeans = bob.io.base.load(datafile('gmm.init_means.hdf5', __name__, path="../data/"))
  gmmVariances = bob.io.base.load(datafile('gmm.init_variances.hdf5', __name__, path="../data/"))

  if (means[0, 0] < means[1, 0]):
    means = flipRows(means)
    variances = flipRows(variances)
    weights = flipRows(weights)

  assert equals(means, gmmMeans, 1e-3)
  assert equals(weights, gmmWeights, 1e-3)
  assert equals(variances, gmmVariances, 1e-3)

  # Check that there is no duplicate means during initialization
  machine = KMeansMachine(2, 1)
  trainer = KMeansTrainer()
  trainer.initialization_method = 'RANDOM_NO_DUPLICATE'
  data = numpy.array([[1.], [1.], [1.], [1.], [1.], [1.], [2.], [3.]])
  bob.learn.em.train(trainer, machine, data)
  assert (numpy.isnan(machine.means).any()) == False
