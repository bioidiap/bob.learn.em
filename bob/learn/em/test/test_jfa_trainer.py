#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
# Tiago Freitas Pereira <tiago.pereira@idiap.ch>
# Tue Jul 19 12:16:17 2011 +0200
#
# Copyright (C) 2011-2014 Idiap Research Institute, Martigny, Switzerland

import copy

import numpy as np

from bob.learn.em import GMMMachine, GMMStats, ISVMachine, JFAMachine

# Define Training set and initial values for tests
F1 = np.array(
    [
        0.3833,
        0.4516,
        0.6173,
        0.2277,
        0.5755,
        0.8044,
        0.5301,
        0.9861,
        0.2751,
        0.0300,
        0.2486,
        0.5357,
    ]
).reshape((6, 2))
F2 = np.array(
    [
        0.0871,
        0.6838,
        0.8021,
        0.7837,
        0.9891,
        0.5341,
        0.0669,
        0.8854,
        0.9394,
        0.8990,
        0.0182,
        0.6259,
    ]
).reshape((6, 2))
F = [F1, F2]

N1 = np.array([0.1379, 0.1821, 0.2178, 0.0418]).reshape((2, 2))
N2 = np.array([0.1069, 0.9397, 0.6164, 0.3545]).reshape((2, 2))
N = [N1, N2]

gs11 = GMMStats(2, 3)
gs11.n = N1[:, 0]
gs11.sum_px = F1[:, 0].reshape(2, 3)
gs12 = GMMStats(2, 3)
gs12.n = N1[:, 1]
gs12.sum_px = F1[:, 1].reshape(2, 3)

gs21 = GMMStats(2, 3)
gs21.n = N2[:, 0]
gs21.sum_px = F2[:, 0].reshape(2, 3)
gs22 = GMMStats(2, 3)
gs22.n = N2[:, 1]
gs22.sum_px = F2[:, 1].reshape(2, 3)

TRAINING_STATS_X = [gs11, gs12, gs21, gs22]
TRAINING_STATS_y = [0, 0, 1, 1]
UBM_MEAN = np.array([0.1806, 0.0451, 0.7232, 0.3474, 0.6606, 0.3839])
UBM_VAR = np.array([0.6273, 0.0216, 0.9106, 0.8006, 0.7458, 0.8131])
M_d = np.array([0.4106, 0.9843, 0.9456, 0.6766, 0.9883, 0.7668])
M_v = np.array(
    [
        0.3367,
        0.4116,
        0.6624,
        0.6026,
        0.2442,
        0.7505,
        0.2955,
        0.5835,
        0.6802,
        0.5518,
        0.5278,
        0.5836,
    ]
).reshape((6, 2))
M_u = np.array(
    [
        0.5118,
        0.3464,
        0.0826,
        0.8865,
        0.7196,
        0.4547,
        0.9962,
        0.4134,
        0.3545,
        0.2177,
        0.9713,
        0.1257,
    ]
).reshape((6, 2))

z1 = np.array([0.3089, 0.7261, 0.7829, 0.6938, 0.0098, 0.8432])
z2 = np.array([0.9223, 0.7710, 0.0427, 0.3782, 0.7043, 0.7295])
y1 = np.array([0.2243, 0.2691])
y2 = np.array([0.6730, 0.4775])
x1 = np.array([0.9976, 0.8116, 0.1375, 0.3900]).reshape((2, 2))
x2 = np.array([0.4857, 0.8944, 0.9274, 0.9175]).reshape((2, 2))
M_z = [z1, z2]
M_y = [y1, y2]
M_x = [x1, x2]


def test_JFATrainAndEnrol():
    # Train and enroll a JFAMachine

    # Calls the train function
    ubm = GMMMachine(2, 3)
    ubm.means = UBM_MEAN.reshape((2, 3))
    ubm.variances = UBM_VAR.reshape((2, 3))
    it = JFAMachine(ubm, 2, 2, em_iterations=10)

    it.U = copy.deepcopy(M_u)
    it.V = copy.deepcopy(M_v)
    it.D = copy.deepcopy(M_d)
    it.fit(TRAINING_STATS_X, TRAINING_STATS_y)

    v_ref = np.array(
        [
            [0.245364911936476, 0.978133261775424],
            [0.769646805052223, 0.940070736856596],
            [0.310779202800089, 1.456332053893072],
            [0.184760934399551, 2.265139705602147],
            [0.701987784039800, 0.081632150899400],
            [0.074344030229297, 1.090248340917255],
        ],
        "float64",
    )
    u_ref = np.array(
        [
            [0.049424652628448, 0.060480486336896],
            [0.178104127464007, 1.884873813495153],
            [1.204011484266777, 2.281351307871720],
            [7.278512126426286, -0.390966087173334],
            [-0.084424326581145, -0.081725474934414],
            [4.042143689831097, -0.262576386580701],
        ],
        "float64",
    )
    d_ref = np.array(
        [
            9.648467e-18,
            2.63720683155e-12,
            2.11822157653706e-10,
            9.1047243e-17,
            1.41163442535567e-10,
            3.30581e-19,
        ],
        "float64",
    )

    eps = 1e-10
    assert np.allclose(it.V, v_ref, eps)
    assert np.allclose(it.U, u_ref, eps)
    assert np.allclose(it.D, d_ref, eps)

    # Calls the enroll function

    Ne = np.array([0.1579, 0.9245, 0.1323, 0.2458]).reshape((2, 2))
    Fe = np.array(
        [
            0.1579,
            0.1925,
            0.3242,
            0.1234,
            0.2354,
            0.2734,
            0.2514,
            0.5874,
            0.3345,
            0.2463,
            0.4789,
            0.5236,
        ]
    ).reshape((6, 2))
    gse1 = GMMStats(2, 3)
    gse1.n = Ne[:, 0]
    gse1.sum_px = Fe[:, 0].reshape(2, 3)
    gse2 = GMMStats(2, 3)
    gse2.n = Ne[:, 1]
    gse2.sum_px = Fe[:, 1].reshape(2, 3)

    gse = [gse1, gse2]
    latent_y, latent_z = it.enroll(gse, 5)

    y_ref = np.array([0.555991469319657, 0.002773650670010], "float64")
    z_ref = np.array(
        [
            8.2228e-20,
            3.15216909492e-13,
            -1.48616735364395e-10,
            1.0625905e-17,
            3.7150503117895e-11,
            1.71104e-19,
        ],
        "float64",
    )

    assert np.allclose(latent_y, y_ref, eps)
    assert np.allclose(latent_z, z_ref, eps)


def test_ISVTrainAndEnrol():
    # Train and enroll an 'ISVMachine'

    eps = 1e-10
    d_ref = np.array(
        [
            0.39601136,
            0.07348469,
            0.47712682,
            0.44738127,
            0.43179856,
            0.45086029,
        ],
        "float64",
    )
    u_ref = np.array(
        [
            [0.855125642430777, 0.563104284748032],
            [-0.325497865404680, 1.923598985291687],
            [0.511575659503837, 1.964288663083095],
            [9.330165761678115, 1.073623827995043],
            [0.511099245664012, 0.278551249248978],
            [5.065578541930268, 0.509565618051587],
        ],
        "float64",
    )
    z_ref = np.array(
        [
            -0.079315777443826,
            0.092702428248543,
            -0.342488761656616,
            -0.059922635809136,
            0.133539981073604,
            0.213118695516570,
        ],
        "float64",
    )

    """
    Calls the train function
    """
    ubm = GMMMachine(2, 3)
    ubm.means = UBM_MEAN.reshape((2, 3))
    ubm.variances = UBM_VAR.reshape((2, 3))

    it = ISVMachine(
        ubm,
        r_U=2,
        relevance_factor=4.0,
        em_iterations=10,
    )

    it.U = copy.deepcopy(M_u)
    it = it.fit(TRAINING_STATS_X, TRAINING_STATS_y)

    assert np.allclose(it.D, d_ref, eps)
    assert np.allclose(it.U, u_ref, eps)

    """
    Calls the enroll function
    """

    Ne = np.array([0.1579, 0.9245, 0.1323, 0.2458]).reshape((2, 2))
    Fe = np.array(
        [
            0.1579,
            0.1925,
            0.3242,
            0.1234,
            0.2354,
            0.2734,
            0.2514,
            0.5874,
            0.3345,
            0.2463,
            0.4789,
            0.5236,
        ]
    ).reshape((6, 2))
    gse1 = GMMStats(2, 3)
    gse1.n = Ne[:, 0]
    gse1.sum_px = Fe[:, 0].reshape(2, 3)
    gse2 = GMMStats(2, 3)
    gse2.n = Ne[:, 1]
    gse2.sum_px = Fe[:, 1].reshape(2, 3)

    gse = [gse1, gse2]

    latent_z = it.enroll(gse, 5)
    assert np.allclose(latent_z, z_ref, eps)


def test_JFATrainInitialize():
    # Check that the initialization is consistent and using the rng (cf. issue #118)

    eps = 1e-10

    # UBM GMM
    ubm = GMMMachine(2, 3)
    ubm.means = UBM_MEAN.reshape((2, 3))
    ubm.variances = UBM_VAR.reshape((2, 3))

    # JFA
    it = JFAMachine(ubm, 2, 2, em_iterations=10)
    # first round

    it.initialize(TRAINING_STATS_X, TRAINING_STATS_y)
    u1 = it.U
    v1 = it.V
    d1 = it.D

    # second round
    it.initialize(TRAINING_STATS_X, TRAINING_STATS_y)
    u2 = it.U
    v2 = it.V
    d2 = it.D

    assert np.allclose(u1, u2, eps)
    assert np.allclose(v1, v2, eps)
    assert np.allclose(d1, d2, eps)


def test_ISVTrainInitialize():

    # Check that the initialization is consistent and using the rng (cf. issue #118)
    eps = 1e-10

    # UBM GMM
    ubm = GMMMachine(2, 3)
    ubm.means = UBM_MEAN.reshape((2, 3))
    ubm.variances = UBM_VAR.reshape((2, 3))

    # ISV
    it = ISVMachine(ubm, 2, em_iterations=10)
    # it.rng = rng

    it.initialize(TRAINING_STATS_X, TRAINING_STATS_y)
    u1 = copy.deepcopy(it.U)
    d1 = copy.deepcopy(it.D)

    # second round
    it.initialize(TRAINING_STATS_X, TRAINING_STATS_y)
    u2 = it.U
    d2 = it.D

    assert np.allclose(u1, u2, eps)
    assert np.allclose(d1, d2, eps)
