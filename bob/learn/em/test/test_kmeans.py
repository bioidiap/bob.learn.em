#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
# Thu Feb 16 17:57:10 2012 +0200
#
# Copyright (C) 2011-2014 Idiap Research Institute, Martigny, Switzerland

"""Tests the KMeans machine
"""

import numpy

from bob.learn.em.cluster import KMeansMachine
from bob.learn.em.cluster import KMeansTrainer

import dask.array as da


def equals(x, y, epsilon):
    return abs(x - y) < epsilon


def test_KMeansMachine():
    # Test a KMeansMachine

    means = numpy.array([[3, 70, 0], [4, 72, 0]], "float64")
    mean = numpy.array([3, 70, 1], "float64")

    # Initializes a KMeansMachine
    km = KMeansMachine(2)
    km.centroids_ = means

    # Distance and closest mean
    eps = 1e-10

    assert equals(km.transform(mean)[0], 1, eps), km.transform(mean)[0].compute()
    assert equals(km.transform(mean)[1], 6, eps), km.transform(mean)[1].compute()

    (index, dist) = km.get_closest_centroid(mean)

    assert index == 0
    assert equals(dist, 1, eps)
    assert equals(km.get_min_distance(mean), 1, eps)


def test_KMeansMachine_var_and_weight():
    kmeans = KMeansMachine(2)
    kmeans.centroids_ = numpy.array([[1.2, 1.3], [0.2, -0.3]])

    data = numpy.array([[1.0, 1], [1.2, 3], [0, 0], [0.3, 0.2], [0.2, 0]])
    variances, weights = kmeans.get_variances_and_weights_for_each_cluster(data)

    variances_result = numpy.array([[0.01, 1.0], [0.01555556, 0.00888889]])
    weights_result = numpy.array([0.4, 0.6])

    assert equals(weights_result, weights, 1e-3).all()
    assert equals(variances_result, variances, 1e-3).all()


def test_kmeans_fit():
    da.random.seed(0)
    data1 = da.random.normal(loc=1, size=(2000, 3))
    data2 = da.random.normal(loc=-1, size=(2000, 3))
    data = da.concatenate([data1, data2], axis=0)
    machine = KMeansMachine(2).fit(data)
    expected = da.array(
        [[-0.99262315, -1.05226141, -1.00525245], [1.00426431, 1.00359693, 1.05996704]]
    )
    assert da.isclose(machine.centroids_, expected).all(), machine.centroids_.compute()


def test_kmeans_fit_init_pp():
    da.random.seed(0)
    data1 = da.random.normal(loc=1, size=(2000, 3))
    data2 = da.random.normal(loc=-1, size=(2000, 3))
    data = da.concatenate([data1, data2], axis=0)
    trainer = KMeansTrainer(init_method="k-means++")
    machine = KMeansMachine(2).fit(data, trainer=trainer)
    expected = da.array(
        [[-0.99262315, -1.05226141, -1.00525245], [1.00426431, 1.00359693, 1.05996704]]
    )
    assert da.isclose(machine.centroids_, expected).all(), machine.centroids_.compute()


def test_kmeans_fit_init_random():
    da.random.seed(0)
    data1 = da.random.normal(loc=1, size=(2000, 3))
    data2 = da.random.normal(loc=-1, size=(2000, 3))
    data = da.concatenate([data1, data2], axis=0)
    trainer = KMeansTrainer(init_method="random", random_state=0)
    machine = KMeansMachine(2).fit(data, trainer=trainer)
    expected = da.array(
        [[-0.99433738, -1.05561588, -1.01236246], [0.99800688, 0.99873325, 1.05879539]]
    )
    assert da.isclose(machine.centroids_, expected).all(), machine.centroids_.compute()
