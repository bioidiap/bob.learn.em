#!/usr/bin/env python
# Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
# Tiago Freitas Pereira <tiago.pereira@idiap.ch>
# Amir Mohammadi <amir.mohammadi@idiap.ch>

import copy

import numpy as np

from bob.learn.em import GMMMachine, GMMStats, ISVMachine, JFAMachine

from .test_gmm import multiprocess_dask_client
from .test_kmeans import to_dask_array, to_numpy

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
    it = JFAMachine(2, 2, em_iterations=10, enroll_iterations=5, ubm=ubm)

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
    np.testing.assert_allclose(it.V, v_ref, rtol=eps, atol=1e-8)
    np.testing.assert_allclose(it.U, u_ref, rtol=eps, atol=1e-8)
    np.testing.assert_allclose(it.D, d_ref, rtol=eps, atol=1e-8)

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
    latent_y, latent_z = it.enroll(gse)

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

    np.testing.assert_allclose(latent_y, y_ref, rtol=eps, atol=1e-8)
    np.testing.assert_allclose(latent_z, z_ref, rtol=eps, atol=1e-8)


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
            [
                -0.079315777443826,
                0.092702428248543,
                -0.342488761656616,
                -0.059922635809136,
                0.133539981073604,
                0.213118695516570,
            ]
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
        ubm=ubm,
        r_U=2,
        relevance_factor=4.0,
        em_iterations=10,
        enroll_iterations=5,
    )

    it.U = copy.deepcopy(M_u)
    it = it.fit(TRAINING_STATS_X, TRAINING_STATS_y)

    np.testing.assert_allclose(it.D, d_ref, rtol=eps, atol=1e-8)
    np.testing.assert_allclose(it.U, u_ref, rtol=eps, atol=1e-8)

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
    latent_z = it.enroll(gse)
    np.testing.assert_allclose(latent_z, z_ref, rtol=eps, atol=1e-8)


def test_JFATrainInitialize():
    # Check that the initialization is consistent and using the rng (cf. issue #118)

    eps = 1e-10

    # UBM GMM
    ubm = GMMMachine(2, 3)
    ubm.means = UBM_MEAN.reshape((2, 3))
    ubm.variances = UBM_VAR.reshape((2, 3))

    # JFA
    it = JFAMachine(2, 2, em_iterations=10, ubm=ubm)
    # first round

    n_classes = it.estimate_number_of_classes(TRAINING_STATS_y)
    it.initialize(TRAINING_STATS_X, TRAINING_STATS_y, n_classes)
    u1 = it.U
    v1 = it.V
    d1 = it.D

    # second round
    it.initialize(TRAINING_STATS_X, TRAINING_STATS_y, n_classes)
    u2 = it.U
    v2 = it.V
    d2 = it.D

    np.testing.assert_allclose(u1, u2, rtol=eps, atol=1e-8)
    np.testing.assert_allclose(v1, v2, rtol=eps, atol=1e-8)
    np.testing.assert_allclose(d1, d2, rtol=eps, atol=1e-8)


def test_ISVTrainInitialize():

    # Check that the initialization is consistent and using the rng (cf. issue #118)
    eps = 1e-10

    # UBM GMM
    ubm = GMMMachine(2, 3)
    ubm.means = UBM_MEAN.reshape((2, 3))
    ubm.variances = UBM_VAR.reshape((2, 3))

    # ISV
    it = ISVMachine(2, em_iterations=10, ubm=ubm)
    # it.rng = rng

    n_classes = it.estimate_number_of_classes(TRAINING_STATS_y)
    it.initialize(TRAINING_STATS_X, TRAINING_STATS_y, n_classes)
    u1 = copy.deepcopy(it.U)
    d1 = copy.deepcopy(it.D)

    # second round
    it.initialize(TRAINING_STATS_X, TRAINING_STATS_y, n_classes)
    u2 = it.U
    d2 = it.D

    np.testing.assert_allclose(u1, u2, rtol=eps, atol=1e-8)
    np.testing.assert_allclose(d1, d2, rtol=eps, atol=1e-8)


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
    m = JFAMachine(2, 2, em_iterations=10, ubm=ubm)
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
    score = m.score_using_array(model, X)
    np.testing.assert_allclose(score, score_ref, atol=eps)


def test_ISVMachine():

    eps = 1e-10

    # Creates a UBM
    ubm = GMMMachine(2, 3)
    ubm.weights = np.array([0.4, 0.6], "float64")
    ubm.means = np.array([[1, 6, 2], [4, 3, 2]], "float64")
    ubm.variances = np.array([[1, 2, 1], [2, 1, 2]], "float64")

    # Creates a ISVMachine
    isv_machine = ISVMachine(ubm=ubm, r_U=2, em_iterations=10)
    isv_machine.U = np.array(
        [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]], "float64"
    )
    # base.v = np.array([[0], [0], [0], [0], [0], [0]], 'float64')
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
    score = isv_machine.score_using_array(latent_z, X)
    np.testing.assert_allclose(score, score_ref, atol=eps)


def _create_ubm_prior(means):
    # Creating a fake prior with 2 gaussians
    prior_gmm = GMMMachine(2)
    prior_gmm.means = means.copy()
    # All nice and round diagonal covariance
    prior_gmm.variances = np.ones((2, 3)) * 0.5
    prior_gmm.weights = np.array([0.3, 0.7])
    return prior_gmm


def test_ISV_JFA_fit():
    np.random.seed(10)
    data_class1 = np.random.normal(0, 0.5, (10, 3))
    data_class2 = np.random.normal(-0.2, 0.2, (10, 3))
    data = np.concatenate([data_class1, data_class2], axis=0)
    labels = [0] * 10 + [1] * 10
    means = np.vstack(
        (np.random.normal(0, 0.5, (1, 3)), np.random.normal(1, 0.5, (1, 3)))
    )
    prior_U = [
        [-0.150035, -0.44441],
        [-1.67812, 2.47621],
        [-0.52885, 0.659141],
        [-0.538446, 1.67376],
        [-0.111288, 2.06948],
        [1.39563, -1.65004],
    ]

    prior_V = [
        [0.732467, 0.281321],
        [0.543212, -0.512974],
        [1.04108, 0.835224],
        [-0.363719, -0.324688],
        [-1.21579, -0.905314],
        [-0.993204, -0.121991],
    ]

    prior_D = [
        0.943986,
        -0.0900599,
        -0.528103,
        0.541502,
        -0.717824,
        0.463729,
    ]

    for prior, machine_type, ref in [
        (
            None,
            "isv",
            0.0,
        ),
        (
            True,
            "isv",
            [
                [-0.01018673, -0.0266506],
                [-0.00160621, -0.00420217],
                [0.02811705, 0.07356008],
                [0.011624, 0.0304108],
                [0.03261831, 0.08533629],
                [0.04602191, 0.12040291],
            ],
        ),
        (
            None,
            "jfa",
            [
                [-0.05673845, -0.0543068],
                [-0.05302666, -0.05075409],
                [-0.02522509, -0.02414402],
                [-0.05723968, -0.05478655],
                [-0.05291602, -0.05064819],
                [-0.02463007, -0.0235745],
            ],
        ),
        (
            True,
            "jfa",
            [
                [0.002881, -0.00584225],
                [0.04143539, -0.08402497],
                [-0.26149924, 0.53028251],
                [-0.25156832, 0.51014406],
                [-0.38687765, 0.78453174],
                [-0.36015821, 0.73034858],
            ],
        ),
    ]:
        ref = np.asarray(ref)

        # Doing the training
        for transform in (to_numpy, to_dask_array):
            data, labels = transform(data, labels)

            if prior is None:
                ubm = None
                # we still provide an initial UBM because KMeans training is not
                # determenistic depending on inputting numpy or dask arrays
                ubm_kwargs = dict(n_gaussians=2, ubm=_create_ubm_prior(means))
            else:
                ubm = _create_ubm_prior(means)
                ubm_kwargs = None

            machine_kwargs = dict(
                ubm=ubm,
                relevance_factor=4,
                em_iterations=50,
                ubm_kwargs=ubm_kwargs,
                random_state=10,
            )

            if machine_type == "isv":
                machine = ISVMachine(2, **machine_kwargs)
                machine.U = prior_U
                test_attr = "U"
            else:
                machine = JFAMachine(2, 2, **machine_kwargs)
                machine.U = prior_U
                machine.V = prior_V
                machine.D = prior_D
                test_attr = "V"

            err_msg = f"Test failed with prior={prior} and machine_type={machine_type} and transform={transform}"
            with multiprocess_dask_client():
                machine.fit_using_array(data, labels)
            print(
                f"\nFinished training machine={machine_type} with prior={prior} and transform={transform}"
            )

            arr = getattr(machine, test_attr)
            np.testing.assert_allclose(
                arr,
                ref,
                atol=1e-7,
                err_msg=err_msg,
            )
