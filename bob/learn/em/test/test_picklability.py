#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Tiago de Freitas Pereira <tiago.pereira@idiap.ch>

from bob.learn.em import (
    GMMMachine,
    ISVBase,
    ISVMachine,
    KMeansMachine,
    GMMStats,
    IVectorMachine,
)
import numpy
import pickle


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


def test_isv_base():
    ubm = GMMMachine(3, 3)
    ubm.means = numpy.arange(9).reshape(3, 3).astype("float")
    isv_base = ISVBase(ubm, 2)
    isv_base.u = numpy.arange(18).reshape(9, 2).astype("float")
    isv_base.d = numpy.arange(9).astype("float")

    isv_base_after_pickle = pickle.loads(pickle.dumps(isv_base))

    assert numpy.allclose(isv_base.u, isv_base_after_pickle.u, 10e-3)
    assert numpy.allclose(isv_base.d, isv_base_after_pickle.d, 10e-3)


def test_isv_machine():

    # Creates a UBM
    weights = numpy.array([0.4, 0.6], "float64")
    means = numpy.array([[1, 6, 2], [4, 3, 2]], "float64")
    variances = numpy.array([[1, 2, 1], [2, 1, 2]], "float64")
    ubm = GMMMachine(2, 3)
    ubm.weights = weights
    ubm.means = means
    ubm.variances = variances

    # Creates a ISVBaseMachine
    U = numpy.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]], "float64")
    # V = numpy.array([[0], [0], [0], [0], [0], [0]], 'float64')
    d = numpy.array([0, 1, 0, 1, 0, 1], "float64")
    base = ISVBase(ubm, 2)
    base.u = U
    base.d = d

    # Creates a ISVMachine
    z = numpy.array([3, 4, 1, 2, 0, 1], "float64")
    x = numpy.array([1, 2], "float64")
    isv_machine = ISVMachine(base)
    isv_machine.z = z
    isv_machine.x = x

    isv_machine_after_pickle = pickle.loads(pickle.dumps(isv_machine))
    assert numpy.allclose(
        isv_machine_after_pickle.isv_base.u, isv_machine.isv_base.u, 10e-3
    )
    assert numpy.allclose(
        isv_machine_after_pickle.isv_base.d, isv_machine.isv_base.d, 10e-3
    )
    assert numpy.allclose(isv_machine_after_pickle.x, isv_machine.x, 10e-3)
    assert numpy.allclose(isv_machine_after_pickle.z, isv_machine.z, 10e-3)


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


def test_ivector_machine():

    # Ubm
    ubm = GMMMachine(2, 3)
    ubm.weights = numpy.array([0.4, 0.6])
    ubm.means = numpy.array([[1.0, 7, 4], [4, 5, 3]])
    ubm.variances = numpy.array([[0.5, 1.0, 1.5], [1.0, 1.5, 2.0]])

    ivector_machine = IVectorMachine(ubm, 2)
    t = numpy.array([[1.0, 2], [4, 1], [0, 3], [5, 8], [7, 10], [11, 1]])
    sigma = numpy.array([1.0, 2.0, 1.0, 3.0, 2.0, 4.0])
    ivector_machine.t = t
    ivector_machine.sigma = sigma

    ivector_after_pickle = pickle.loads(pickle.dumps(ivector_machine))
    assert numpy.allclose(ivector_after_pickle.sigma, ivector_machine.sigma, 10e-3)
    assert numpy.allclose(ivector_after_pickle.t, ivector_machine.t, 10e-3)
    assert numpy.allclose(
        ivector_after_pickle.ubm.means, ivector_machine.ubm.means, 10e-3
    )
    assert numpy.allclose(
        ivector_after_pickle.ubm.variances, ivector_machine.ubm.variances, 10e-3
    )
    assert numpy.allclose(
        ivector_after_pickle.ubm.weights, ivector_machine.ubm.weights, 10e-3
    )
