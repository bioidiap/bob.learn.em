#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
# Thu Feb 16 17:57:10 2012 +0200
#
# Copyright (C) 2011-2014 Idiap Research Institute, Martigny, Switzerland

"""Tests the KMeans machine
"""

import dask.array as da
import numpy as np

from bob.learn.em.cluster import KMeansMachine


def to_numpy(*args):
    result = []
    for x in args:
        result.append(np.array(x))
    if len(result) == 1:
        return result[0]
    return result


def to_dask_array(*args):
    result = []
    for x in args:
        result.append(da.from_array(np.array(x)))
    if len(result) == 1:
        return result[0]
    return result


def test_KMeansMachine():
    # Test a KMeansMachine

    means = np.array([[3, 70, 0], [4, 72, 0]], "float64")
    mean = np.array([3, 70, 1], "float64")

    for transform in (to_numpy, to_dask_array):
        means, mean = transform(means, mean)

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
    for transform in (to_numpy, to_dask_array):
        kmeans = KMeansMachine(2)
        kmeans.centroids_ = transform(np.array([[1.2, 1.3], [0.2, -0.3]]))

        data = np.array([[1.0, 1], [1.2, 3], [0, 0], [0.3, 0.2], [0.2, 0]])
        data = transform(data)
        variances, weights = kmeans.get_variances_and_weights_for_each_cluster(data)

        variances_result = np.array([[0.01, 1.0], [0.01555556, 0.00888889]])
        weights_result = np.array([0.4, 0.6])

        np.testing.assert_almost_equal(variances, variances_result)
        np.testing.assert_almost_equal(weights, weights_result)


np.set_printoptions(precision=9)


def test_kmeans_fit():
    np.random.seed(0)
    data1 = np.random.normal(loc=1, size=(2000, 3))
    data2 = np.random.normal(loc=-1, size=(2000, 3))
    data = np.concatenate([data1, data2], axis=0)

    for transform in (to_numpy, to_dask_array):
        data = transform(data)
        machine = KMeansMachine(2, random_state=0).fit(data)
        centroids = machine.centroids_[np.argsort(machine.centroids_[:, 0])]
        expected = [
            [-1.07173464, -1.06200356, -1.00724920],
            [0.99479125, 0.99665564, 0.97689017],
        ]
        np.testing.assert_almost_equal(centroids, expected)


def test_kmeans_fit_init_pp():
    np.random.seed(0)
    data1 = np.random.normal(loc=1, size=(2000, 3))
    data2 = np.random.normal(loc=-1, size=(2000, 3))
    data = np.concatenate([data1, data2], axis=0)

    for transform in (to_numpy, to_dask_array):
        data = transform(data)
        machine = KMeansMachine(2, init_method="k-means++", random_state=0).fit(data)
        centroids = machine.centroids_[np.argsort(machine.centroids_[:, 0])]
        expected = [
            [-1.07173464, -1.06200356, -1.00724920],
            [0.99479125, 0.99665564, 0.97689017],
        ]
        np.testing.assert_almost_equal(centroids, expected, decimal=7)


def test_kmeans_fit_init_random():
    np.random.seed(0)
    data1 = np.random.normal(loc=1, size=(2000, 3))
    data2 = np.random.normal(loc=-1, size=(2000, 3))
    data = np.concatenate([data1, data2], axis=0)
    for transform in (to_numpy, to_dask_array):
        data = transform(data)
        machine = KMeansMachine(2, init_method="random", random_state=0).fit(data)
        centroids = machine.centroids_[np.argsort(machine.centroids_[:, 0])]
        expected = [
            [-1.07329460, -1.06207104, -1.00714365],
            [0.99529015, 0.99570570, 0.97580858],
        ]
        np.testing.assert_almost_equal(centroids, expected, decimal=7)
