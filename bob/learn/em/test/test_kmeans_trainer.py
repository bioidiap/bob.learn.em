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

import dask.array as da
from dask_ml.cluster.k_means import k_init

from bob.learn.em.cluster import KMeansMachine, KMeansTrainer


def equals(x, y, epsilon):
    return (abs(x - y) < epsilon).all()


def NormalizeStdArray(path):
    array = bob.io.base.load(path).astype('float64')
    std = array.std(axis=0)
    return (array / std, std)


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
        raise Exception('Input type not supported by flipRows')


def test_kmeans_plus_plus():
    # Tests the K-Means++ initialization
    dim_c = 5
    dim_d = 7
    n_samples = 150
    data = da.random.random((n_samples, dim_d))
    seed = 0

    # C++ implementation
    machine = KMeansMachine(dim_c, dim_d)
    trainer = KMeansTrainer()
    trainer.initialize(machine, data, means_init="k-means++", random_state=seed)

    # Reference implementation
    py_machine = KMeansMachine(dim_c, dim_d)
    py_machine.means = k_init(data, dim_c, random_state=seed, init="k-means++")
    assert equals(machine.means, py_machine.means, 1e-8)


def test_kmeans_random():
    # Data/dimensions
    dim_c = 2
    dim_d = 3
    seed = 0
    data = da.array([[1, 2, 3], [1, 2, 3], [1, 2, 3], [4, 5, 6.]])
    # Defines machine and trainer
    machine = KMeansMachine(dim_c, dim_d)
    trainer = KMeansTrainer()
    trainer.initialize(machine, data, random_state=seed, means_init="random")
    # Makes sure that the two initial mean vectors selected are different
    assert equals(machine.get_mean(0), machine.get_mean(1), 1e-8) == False


def test_kmeans_a():
    # Trains a KMeansMachine
    # This files contains draws from two 1D Gaussian distributions:
    #   * 100 samples from N(-10,1)
    #   * 100 samples from N(10,1)
    data = bob.io.base.load(datafile("samplesFrom2G_f64.hdf5", __name__, path="../data/"))
    data = da.array(data)

    machine = KMeansMachine(2, 1)

    trainer = KMeansTrainer()
    # trainer.train(machine, data)
    bob.learn.em.train(trainer, machine, data)

    variances, weights = machine.get_variances_and_weights_for_each_cluster(data)
    m1 = machine.get_mean(0)
    m2 = machine.get_mean(1)

    ## Check means [-10,10] / variances [1,1] / weights [0.5,0.5]
    if (m1 < m2):
        means = numpy.array(([m1[0], m2[0]]), 'float64')
    else:
        means = numpy.array(([m2[0], m1[0]]), 'float64')
    assert equals(means, numpy.array([-10., 10.]), 2e-1)
    assert equals(variances, numpy.array([1., 1.]), 2e-1)
    assert equals(weights, numpy.array([0.5, 0.5]), 1e-3)



def test_kmeans_b():
    # Trains a KMeansMachine
    (arStd, std) = NormalizeStdArray(datafile("faithful.torch3.hdf5", __name__, path="../data/"))
    arStd = da.array(arStd)

    machine = KMeansMachine(2, 2)

    trainer = KMeansTrainer()
    # trainer.seed = 1337
    bob.learn.em.train(trainer, machine, arStd, convergence_threshold=0.001)

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


def test_kmeans_parallel():
    # Trains a KMeansMachine
    (arStd, std) = NormalizeStdArray(datafile("faithful.torch3.hdf5", __name__, path="../data/"))
    arStd = da.array(arStd)

    machine = KMeansMachine(2, 2)

    trainer = KMeansTrainer()
    # trainer.seed = 1337

    import multiprocessing.pool
    pool = multiprocessing.pool.ThreadPool(3)
    bob.learn.em.train(trainer, machine, arStd, convergence_threshold=0.001, pool = pool)

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


def test_trainer_exception():
    from nose.tools import assert_raises

    # Testing Inf
    machine = KMeansMachine(2, 2)
    data = numpy.array([[1.0, 2.0], [2, 3.], [1, 1.], [2, 5.], [numpy.inf, 1.0]])
    trainer = KMeansTrainer()
    assert_raises(ValueError, bob.learn.em.train, trainer, machine, data, 10)

    # Testing Nan
    machine = KMeansMachine(2, 2)
    data = numpy.array([[1.0, 2.0], [2, 3.], [1, numpy.nan], [2, 5.], [2.0, 1.0]])
    trainer = KMeansTrainer()
    assert_raises(ValueError, bob.learn.em.train, trainer, machine, data, 10)
