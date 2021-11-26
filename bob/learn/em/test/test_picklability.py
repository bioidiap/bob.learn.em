#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Tiago de Freitas Pereira <tiago.pereira@idiap.ch>

import pickle

import numpy

from bob.learn.em import GMMMachine
from bob.learn.em import GMMStats
from bob.learn.em import KMeansMachine


def test_gmm_machine():
    gmm_machine = GMMMachine(3, 3)
    gmm_machine.means = numpy.arange(9).reshape(3, 3).astype("float")
    gmm_machine_after_pickle = pickle.loads(pickle.dumps(gmm_machine))

    assert numpy.allclose(
        gmm_machine_after_pickle.means, gmm_machine_after_pickle.means, 10e-3
    )
    assert numpy.allclose(
        gmm_machine_after_pickle.variances, gmm_machine_after_pickle.variances, 10e-3
    )
    assert numpy.allclose(
        gmm_machine_after_pickle.weights, gmm_machine_after_pickle.weights, 10e-3
    )


def test_kmeans_machine():
    # Test a KMeansMachine

    means = numpy.array([[3, 70, 0], [4, 72, 0]], "float64")
    mean = numpy.array([3, 70, 1], "float64")

    # Initializes a KMeansMachine
    kmeans_machine = KMeansMachine(2, 3)
    kmeans_machine.means = means

    kmeans_machine_after_pickle = pickle.loads(pickle.dumps(kmeans_machine))
    assert numpy.allclose(
        kmeans_machine_after_pickle.means, kmeans_machine.means, 10e-3
    )


def test_gmmstats():

    gs = GMMStats(2, 3)
    log_likelihood = -3.0
    T = 1
    n = numpy.array([0.4, 0.6], numpy.float64)
    sumpx = numpy.array([[1.0, 2.0, 3.0], [2.0, 4.0, 3.0]], numpy.float64)
    sumpxx = numpy.array([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]], numpy.float64)
    gs.log_likelihood = log_likelihood
    gs.t = T
    gs.n = n
    gs.sum_px = sumpx
    gs.sum_pxx = sumpxx

    gs_after_pickle = pickle.loads(pickle.dumps(gs))
    assert gs == gs_after_pickle
