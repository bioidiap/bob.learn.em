#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
# Thu Feb 16 17:57:10 2012 +0200
#
# Copyright (C) 2011-2014 Idiap Research Institute, Martigny, Switzerland

"""Tests the KMeans machine
"""

import numpy as np

from bob.learn.em.cluster import KMeansMachine
from bob.learn.em.cluster import KMeansTrainer

import dask.array as da


def test_KMeansMachine():
    # Test a KMeansMachine

    means = np.array([[3, 70, 0], [4, 72, 0]], "float64")
    mean = np.array([3, 70, 1], "float64")

    # Initializes a KMeansMachine
    km = KMeansMachine(2)
    km.centroids_ = means

    # Distance and closest mean
    np.testing.assert_almost_equal(km.transform(mean)[0], 1)
    np.testing.assert_almost_equal(km.transform(mean)[1], 6)

    (index, dist) = km.get_closest_centroid(mean)

    assert index == 0, index
    np.testing.assert_almost_equal(dist, 1.0)
    np.testing.assert_almost_equal(km.get_min_distance(mean), 1)


def test_KMeansMachine_var_and_weight():
    kmeans = KMeansMachine(2)
    kmeans.centroids_ = np.array([[1.2, 1.3], [0.2, -0.3]])

    data = np.array([[1.0, 1], [1.2, 3], [0, 0], [0.3, 0.2], [0.2, 0]])
    variances, weights = kmeans.get_variances_and_weights_for_each_cluster(data)

    variances_result = np.array([[0.01, 1.0], [0.01555556, 0.00888889]])
    weights_result = np.array([0.4, 0.6])

    np.testing.assert_almost_equal(variances, variances_result)
    np.testing.assert_almost_equal(weights, weights_result)


def test_kmeans_fit():
    da.random.seed(0)
    data1 = da.random.normal(loc=1, size=(2000, 3))
    data2 = da.random.normal(loc=-1, size=(2000, 3))
    data = da.concatenate([data1, data2], axis=0)
    machine = KMeansMachine(2, random_state=0).fit(data)
    expected = [
        [-0.99262315, -1.05226141, -1.00525245],
        [1.00426431, 1.00359693, 1.05996704],
    ]
    np.testing.assert_almost_equal(machine.centroids_, expected)


def test_kmeans_fit_init_pp():
    da.random.seed(0)
    data1 = da.random.normal(loc=1, size=(2000, 3))
    data2 = da.random.normal(loc=-1, size=(2000, 3))
    data = da.concatenate([data1, data2], axis=0)
    trainer = KMeansTrainer(init_method="k-means++", random_state=0)
    machine = KMeansMachine(2).fit(data, trainer=trainer)
    centroids = machine.centroids_.compute()  # Silences `argsort not implemented`
    centroids = centroids[np.argsort(centroids[:,0])]
    expected = [
        [-0.99262315, -1.05226141, -1.00525245],
        [1.00426431, 1.00359693, 1.05996704],
    ]
    np.testing.assert_almost_equal(centroids, expected, decimal=7)


def test_kmeans_fit_init_random():
    da.random.seed(0)
    data1 = da.random.normal(loc=1, size=(2000, 3))
    data2 = da.random.normal(loc=-1, size=(2000, 3))
    data = da.concatenate([data1, data2], axis=0)
    trainer = KMeansTrainer(init_method="random", random_state=0)
    machine = KMeansMachine(2).fit(data, trainer=trainer)
    centroids = machine.centroids_.compute()  # Silences `argsort not implemented`
    centroids = centroids[np.argsort(centroids[:,0])]
    expected = [
        [-0.99433738, -1.05561588, -1.01236246],
        [0.99800688, 0.99873325, 1.05879539],
    ]
    np.testing.assert_almost_equal(centroids, expected, decimal=7)
