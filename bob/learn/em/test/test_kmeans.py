#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
# Thu Feb 16 17:57:10 2012 +0200
#
# Copyright (C) 2011-2014 Idiap Research Institute, Martigny, Switzerland

"""Tests the KMeans machine

Tries each test with a numpy array and the equivalent dask array.
"""

import copy

import dask.array as da
import numpy as np
import scipy.spatial.distance

from bob.learn.em import KMeansMachine, kmeans


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
        x = np.asarray(x)
        chunks = list(x.shape)
        chunks[0] = int(np.ceil(chunks[0] / 2))
        result.append(da.from_array(x, chunks=chunks))
    if len(result) == 1:
        return result[0]
    return result


def test_KMeansMachine():
    # Test a KMeansMachine

    means = np.array([[3, 70, 0], [4, 72, 0]], "float64")
    test_val = np.array([3, 70, 1], "float64")
    test_arr = np.array([[3, 70, 1], [5, 72, 0]], "float64")

    for transform in (to_numpy, to_dask_array):
        means, test_val, test_arr = transform(means, test_val, test_arr)

        # Initializes a KMeansMachine
        km = KMeansMachine(2)
        km.centroids_ = means

        # Distance and closest mean
        np.testing.assert_equal(km.transform(test_val)[0], np.array([1]))
        np.testing.assert_equal(km.transform(test_val)[1], np.array([6]))

        index = km.predict(test_val)
        assert index == 0

        indices = km.predict(test_arr)
        np.testing.assert_equal(indices, np.array([0, 1]))

        # Check __eq__ and is_similar_to
        km2 = KMeansMachine(2)
        assert km != km2
        assert not km.is_similar_to(km2)
        km2 = copy.deepcopy(km)
        assert km == km2
        assert km.is_similar_to(km2)
        km2.centroids_[0, 0] += 1
        assert km != km2
        assert not km.is_similar_to(km2)


def test_KMeansMachine_var_and_weight():
    for transform in (to_numpy, to_dask_array):
        kmeans = KMeansMachine(2)
        kmeans.centroids_ = transform(np.array([[1.2, 1.3], [0.2, -0.3]]))

        data = np.array([[1.0, 1], [1.2, 3], [0, 0], [0.3, 0.2], [0.2, 0]])
        data = transform(data)
        variances, weights = kmeans.get_variances_and_weights_for_each_cluster(
            data
        )

        variances_result = np.array([[0.01, 1.0], [0.01555556, 0.00888889]])
        weights_result = np.array([0.4, 0.6])

        np.testing.assert_almost_equal(variances, variances_result, decimal=7)
        np.testing.assert_equal(weights, weights_result)


def test_kmeans_fit():
    np.random.seed(0)
    data1 = np.random.normal(loc=1, size=(2000, 3))
    data2 = np.random.normal(loc=-1, size=(2000, 3))
    print(data1.min(), data1.max())
    print(data2.min(), data2.max())
    data = np.concatenate([data1, data2], axis=0)

    for transform in (to_numpy, to_dask_array):
        data = transform(data)
        machine = KMeansMachine(2, random_state=0).fit(data)
        centroids = machine.centroids_[np.argsort(machine.centroids_[:, 0])]
        expected = [
            [-1.07173464, -1.06200356, -1.00724920],
            [0.99479125, 0.99665564, 0.97689017],
        ]
        np.testing.assert_almost_equal(centroids, expected, decimal=7)

        # Early stop
        machine = KMeansMachine(2, max_iter=2)
        machine.fit(data)


def test_kmeans_fit_init_pp():
    np.random.seed(0)
    data1 = np.random.normal(loc=1, size=(2000, 3))
    data2 = np.random.normal(loc=-1, size=(2000, 3))
    data = np.concatenate([data1, data2], axis=0)

    for transform in (to_numpy, to_dask_array):
        data = transform(data)
        machine = KMeansMachine(2, init_method="k-means++", random_state=0).fit(
            data
        )
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
        machine = KMeansMachine(2, init_method="random", random_state=0).fit(
            data
        )
        centroids = machine.centroids_[np.argsort(machine.centroids_[:, 0])]
        expected = [
            [-1.07329460, -1.06207104, -1.00714365],
            [0.99529015, 0.99570570, 0.97580858],
        ]
        np.testing.assert_almost_equal(centroids, expected, decimal=7)


def test_kmeans_parameters():
    np.random.seed(0)
    data1 = np.random.normal(loc=1, size=(2000, 3))
    data2 = np.random.normal(loc=-1, size=(2000, 3))
    data = np.concatenate([data1, data2], axis=0)
    for transform in (to_numpy, to_dask_array):
        data = transform(data)
        machine = KMeansMachine(
            n_clusters=2,
            init_method="k-means||",
            convergence_threshold=1e-5,
            max_iter=5,
            random_state=0,
            init_max_iter=5,
        ).fit(data)
        centroids = machine.centroids_[np.argsort(machine.centroids_[:, 0])]
        expected = [
            [-1.07173464, -1.06200356, -1.00724920],
            [0.99479125, 0.99665564, 0.97689017],
        ]
        np.testing.assert_almost_equal(centroids, expected, decimal=7)


def test_get_centroids_distance():
    np.random.seed(0)
    n_features = 60
    n_samples = 240_000
    n_clusters = 256
    data = np.random.normal(loc=1, size=(n_samples, n_features))
    means = np.random.normal(loc=-1, size=(n_clusters, n_features))
    oracle = scipy.spatial.distance.cdist(means, data, metric="sqeuclidean")
    for transform in (to_numpy,):
        data, means = transform(data, means)
        dist = kmeans.get_centroids_distance(data, means)
        np.testing.assert_allclose(dist, oracle)
        assert type(data) is type(dist), (type(data), type(dist))
