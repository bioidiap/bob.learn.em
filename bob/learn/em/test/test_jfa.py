#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
# Tiago Freitas Pereira <tiago.pereira@idiap.ch>
# Tue Jul 19 12:16:17 2011 +0200
#
# Copyright (C) 2011-2014 Idiap Research Institute, Martigny, Switzerland

import numpy as np

from bob.learn.em import GMMMachine, GMMStats, ISVMachine, JFAMachine


def test_JFAMachine():

    eps = 1e-10

    # Creates a UBM
    ubm = GMMMachine(2, 3)
    ubm.weights = np.array([0.4, 0.6], "float64")
    ubm.means = np.array([[1, 6, 2], [4, 3, 2]], "float64")
    ubm.variances = np.array([[1, 2, 1], [2, 1, 2]], "float64")

    # Defines GMMStats
    gs = GMMStats(2, 3)
    gs.log_likelihood = -3.0
    gs.t = 1
    gs.n = np.array([0.4, 0.6], "float64")
    gs.sum_px = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], "float64")
    gs.sum_pxx = np.array([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]], "float64")

    # Creates a JFAMachine
    m = JFAMachine(ubm, 2, 2, em_iterations=10)
    m.U = np.array(
        [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]], "float64"
    )
    m.V = np.array([[6, 5], [4, 3], [2, 1], [1, 2], [3, 4], [5, 6]], "float64")
    m.D = np.array([0, 1, 0, 1, 0, 1], "float64")

    # Preparing the model
    y = np.array([1, 2], "float64")
    z = np.array([3, 4, 1, 2, 0, 1], "float64")
    model = [y, z]

    score_ref = -2.111577181208289
    score = m.score(model, gs)
    np.testing.assert_allclose(score, score_ref, atol=eps)

    # Scoring with numpy array
    np.random.seed(0)
    X = np.random.normal(loc=0.0, scale=1.0, size=(50, 3))
    score_ref = 2.028009315286946
    score = m.score_with_array(model, X)
    np.testing.assert_allclose(score, score_ref, atol=eps)


def test_ISVMachine():

    eps = 1e-10

    # Creates a UBM
    ubm = GMMMachine(2, 3)
    ubm.weights = np.array([0.4, 0.6], "float64")
    ubm.means = np.array([[1, 6, 2], [4, 3, 2]], "float64")
    ubm.variances = np.array([[1, 2, 1], [2, 1, 2]], "float64")

    # Creates a ISVMachine
    isv_machine = ISVMachine(ubm, r_U=2, em_iterations=10)
    isv_machine.U = np.array(
        [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]], "float64"
    )
    # base.v = numpy.array([[0], [0], [0], [0], [0], [0]], 'float64')
    isv_machine.D = np.array([0, 1, 0, 1, 0, 1], "float64")

    # Defines GMMStats
    gs = GMMStats(2, 3)
    gs.log_likelihood = -3.0
    gs.t = 1
    gs.n = np.array([0.4, 0.6], "float64")
    gs.sum_px = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], "float64")
    gs.sum_pxx = np.array([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]], "float64")

    # Enrolled model
    latent_z = np.array([3, 4, 1, 2, 0, 1], "float64")
    score = isv_machine.score(latent_z, gs)
    score_ref = -3.280498193082100
    np.testing.assert_allclose(score, score_ref, atol=eps)

    # Scoring with numpy array
    np.random.seed(0)
    X = np.random.normal(loc=0.0, scale=1.0, size=(50, 3))
    score_ref = -1.2343813195374242
    score = isv_machine.score_with_array(latent_z, X)
    np.testing.assert_allclose(score, score_ref, atol=eps)
