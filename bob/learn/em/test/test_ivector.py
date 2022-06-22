#!/usr/bin/env python
# @author: Yannick Dayer <yannick.dayer@idiap.ch>
# @date: Fri 06 May 2022 12:59:21 UTC+02

import copy

import numpy as np

from bob.learn.em import GMMMachine, GMMStats, IVectorMachine


def test_ivector_machine_base():
    dim_c, dim_d, dim_t = 2, 3, 4

    # Create the UBM and set its values manually
    ubm = GMMMachine(n_gaussians=dim_c)
    ubm.weights = np.array([0.4, 0.6], dtype=float)
    ubm.means = np.array([[1, 7, 4], [4, 5, 3]], dtype=float)
    ubm.variances = np.array([[0.5, 1.0, 1.5], [1.0, 1.5, 2.0]], dtype=float)

    machine = IVectorMachine(ubm=ubm, dim_t=dim_t)

    assert hasattr(machine, "ubm")
    assert hasattr(machine, "T")
    assert hasattr(machine, "sigma")

    assert machine.T.shape == (dim_c, dim_d, dim_t), machine.T.shape
    assert machine.sigma.shape == (dim_c, dim_d), machine.sigma.shape
    np.testing.assert_equal(machine.sigma, ubm.variances)


def test_ivector_machine_projection():

    # Create the UBM and set its values manually
    ubm = GMMMachine(n_gaussians=2)
    ubm.weights = np.array([0.4, 0.6], dtype=float)
    ubm.means = np.array([[1, 7, 4], [4, 5, 3]], dtype=float)
    ubm.variances = np.array([[0.5, 1.0, 1.5], [1.0, 1.5, 2.0]], dtype=float)

    machine = IVectorMachine(ubm=ubm, dim_t=2)
    machine.T = np.array(
        [[[1, 2], [4, 1], [0, 3]], [[5, 8], [7, 10], [11, 1]]], dtype=float
    )
    machine.sigma = np.array([[1, 2, 1], [3, 2, 4]], dtype=float)

    # Manually create a feature (usually projected with the UBM)
    gmm_projection = GMMStats(ubm.n_gaussians, ubm.means.shape[-1])
    gmm_projection.t = 1
    gmm_projection.n = np.array([0.4, 0.6], dtype=float)
    gmm_projection.sum_px = np.array([[1, 2, 3], [2, 4, 3]], dtype=float)
    gmm_projection.sum_pxx = np.array([[10, 20, 30], [40, 50, 60]], dtype=float)

    # Reference from C++ implementation
    ivector_projection_ref = np.array([-0.04213415, 0.21463343])
    ivector_projection = machine.project(gmm_projection)
    np.testing.assert_almost_equal(
        ivector_projection_ref, ivector_projection, decimal=7
    )


def test_ivector_machine_transformer():
    dim_t = 2
    ubm = GMMMachine(n_gaussians=2)
    ubm.means = np.array([[1, 7, 4], [4, 5, 3]], dtype=float)
    ubm.variances = np.array([[0.5, 1.0, 1.5], [1.0, 1.5, 2.0]], dtype=float)
    machine = IVectorMachine(ubm=ubm, dim_t=dim_t)
    machine.T = np.array(
        [[[1, 2], [4, 1], [0, 3]], [[5, 8], [7, 10], [11, 1]]], dtype=float
    )
    assert hasattr(machine, "fit")
    assert hasattr(machine, "transform")
    assert hasattr(machine, "enroll")
    assert hasattr(machine, "score")

    transformed = machine.transform([np.array([1, 2, 3])])[0]
    assert isinstance(transformed, np.ndarray)
    np.testing.assert_array_equal(
        transformed, np.array([-0.04213415, 0.21463343])
    )


def test_ivector_machine_training():

    # Define GMMStats
    gs1 = GMMStats(n_gaussians=2, n_features=3)
    gs1.log_likelihood = -3
    gs1.t = 1
    gs1.n = np.array([0.4, 0.6], dtype=float)
    gs1.sum_px = np.array([[1.0, 2.0, 3.0], [2.0, 4.0, 3.0]], dtype=float)
    gs1.sum_pxx = np.array(
        [[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]], dtype=float
    )

    gs2 = GMMStats(n_gaussians=2, n_features=3)
    gs2.log_likelihood = -4
    gs2.t = 1
    gs2.n = np.array([0.2, 0.8], dtype=float)
    gs2.sum_px = np.array([[2.0, 1.0, 3.0], [3.0, 4.1, 3.2]], dtype=float)
    gs2.sum_pxx = np.array(
        [[12.0, 15.0, 25.0], [39.0, 51.0, 62.0]], dtype=float
    )

    data = [gs1, gs2]

    # Define the ubm
    ubm = GMMMachine(n_gaussians=3)
    ubm.means = np.array([[1, 2, 3], [6, 7, 8], [10, 11, 12]])
    ubm.variances = np.ones((3, 3))

    machine = IVectorMachine(ubm=ubm, dim_t=2)
    machine.fit(data)

    assert False, "TODO: add tests (with projection)"


# Test class inspired by an implementation of Chris McCool
# Chris McCool (chris.mccool@nicta.com.au)
class IVectorTrainerPy:
    """An IVector extractor"""

    def __init__(
        self,
        convergence_threshold=0.001,
        max_iterations=10,
        compute_likelihood=False,
        sigma_update=False,
        variance_floor=1e-5,
    ):
        self.m_convergence_threshold = convergence_threshold
        self.m_max_iterations = max_iterations
        self.m_compute_likelihood = compute_likelihood
        self.m_sigma_update = sigma_update
        self.m_variance_floor = variance_floor
        self.T = None
        self.sigma = None

    def initialize(self, machine, data):
        ubm = machine.ubm
        self.m_dim_c = ubm.shape[0]
        self.m_dim_d = ubm.shape[1]
        self.m_dim_t = machine.T.shape[0]
        self.m_means = ubm.means
        t = np.random.randn(self.m_dim_c * self.m_dim_d, self.m_dim_t)
        machine.T = t
        machine.sigma = machine.ubm.variances

    def e_step(self, machine: IVectorMachine, data):
        n_samples = len(data)
        self.m_acc_Nij_Sigma_wij2 = np.zeros(
            shape=(self.m_dim_c, self.m_dim_t, self.m_dim_t)
        )
        self.m_acc_Fnorm_Sigma_wij = np.zeros(
            shape=(self.m_dim_c, self.m_dim_d, self.m_dim_t)
        )
        self.m_acc_Snorm = np.zeros(
            shape=(
                self.m_dim_c,
                self.m_dim_d,
            ),
            dtype=np.float64,
        )
        self.m_N = np.zeros(shape=(self.m_dim_c,), dtype=np.float64)

        for n in range(n_samples):
            Nij = data[n].n
            Fij = data[n].sum_px
            Sij = data[n].sum_pxx

            # Estimate latent variables
            from bob.learn.em.ivector import (
                compute_id_tt_sigma_inv_t,
                compute_tt_sigma_inv_fnorm,
            )

            TtSigmaInv_Fnorm = compute_tt_sigma_inv_fnorm(
                ubm_means=machine.ubm.means,
                stats=data[n],
                T=machine.T,
                sigma=machine.sigma,
            )
            I_TtSigmaInvNT = compute_id_tt_sigma_inv_t(
                stats=data[n], T=machine.T, sigma=machine.sigma
            )

            Fnorm = np.zeros(
                shape=(
                    self.m_dim_c,
                    self.m_dim_d,
                ),
                dtype=np.float64,
            )
            Snorm = np.zeros(
                shape=(
                    self.m_dim_c,
                    self.m_dim_d,
                ),
                dtype=np.float64,
            )

            # Compute normalized statistics
            for c in range(self.m_dim_c):
                # start = c * self.m_dim_d
                # end = (c + 1) * self.m_dim_d

                Fc = Fij[c, :]
                Sc = Sij[c, :]
                mc = self.m_means[c]

                Fc_mc = Fc * mc
                Nc_mc_mcT = Nij[c] * mc * mc

                Fnorm[c] = Fc - Nij[c] * mc
                Snorm[c] = Sc - (2 * Fc_mc) + Nc_mc_mcT

            # Latent variables
            I_TtSigmaInvNT_inv = np.linalg.inv(I_TtSigmaInvNT)
            E_w_ij = np.dot(I_TtSigmaInvNT_inv, TtSigmaInv_Fnorm)
            E_w_ij2 = I_TtSigmaInvNT_inv + np.outer(E_w_ij, E_w_ij)

            # Do the accumulation for each component
            self.m_acc_Snorm = self.m_acc_Snorm + Snorm  # (dim_c*dim_d)
            for c in range(self.m_dim_c):
                # start = c * self.m_dim_d
                # end = (c + 1) * self.m_dim_d
                current_Fnorm = Fnorm[c]  # (dim_d)
                self.m_acc_Nij_Sigma_wij2[c] = (
                    self.m_acc_Nij_Sigma_wij2[c] + Nij[c] * E_w_ij2
                )  # (dim_t, dim_t)
                self.m_acc_Fnorm_Sigma_wij[c] = self.m_acc_Fnorm_Sigma_wij[
                    c
                ] + np.outer(
                    current_Fnorm, E_w_ij
                )  # (dim_d, dim_t)
                self.m_N[c] = self.m_N[c] + Nij[c]

    def m_step(self, machine):
        A = self.m_acc_Nij_Sigma_wij2

        T = np.zeros(
            shape=(self.m_dim_c, self.m_dim_d, self.m_dim_t), dtype=np.float64
        )
        # Told = machine.T
        if self.m_sigma_update:
            sigma = np.zeros(shape=self.m_acc_Snorm.shape, dtype=np.float64)
        for c in range(self.m_dim_c):
            # start = c * self.m_dim_d
            # end = (c + 1) * self.m_dim_d
            # T update
            A = self.m_acc_Nij_Sigma_wij2[c].transpose()
            B = self.m_acc_Fnorm_Sigma_wij[c].transpose()
            if np.array_equal(A, np.zeros(A.shape)):
                X = np.zeros(
                    shape=(self.m_dim_t, self.m_dim_d), dtype=np.float64
                )
            else:
                X = np.linalg.solve(A, B)
            T[c, :] = X.transpose()
            # Sigma update
            if self.m_sigma_update:
                # Told_c = Told[start:end, :].transpose()
                # warning: Use of the new T estimate! (revert second next line if you don't want that)
                Fnorm_Ewij_Tt = np.diag(
                    np.dot(self.m_acc_Fnorm_Sigma_wij[c], X)
                )
                # Fnorm_Ewij_Tt = np.diag(np.dot(self.m_acc_Fnorm_Sigma_wij[c], Told_c))
                sigma[c] = (self.m_acc_Snorm[c] - Fnorm_Ewij_Tt) / self.m_N[c]

        machine.T = T
        if self.m_sigma_update:
            sigma[sigma < self.m_variance_floor] = self.m_variance_floor
            machine.sigma = sigma

    def finalize(self, machine, data):
        pass

    def train(self, machine, data):
        self.initialize(machine, data)
        self.e_step(machine, data)

        i = 0
        while True:
            self.m_step(machine, data)
            self.e_step(machine, data)
            i += 1
            if i >= self.m_max_iterations:
                break


def test_trainer_nosigma():
    # Ubm
    ubm = GMMMachine(2)
    ubm.means = np.array([[1.0, 7, 4], [4, 5, 3]])
    ubm.variances = np.array([[0.5, 1.0, 1.5], [1.0, 1.5, 2.0]])
    ubm.weights = np.array([0.4, 0.6])

    # Defines GMMStats
    gs1 = GMMStats(2, 3)
    log_likelihood1 = -3.0
    T1 = 1
    n1 = np.array([0.4, 0.6], np.float64)
    sumpx1 = np.array([[1.0, 2.0, 3.0], [2.0, 4.0, 3.0]], np.float64)
    sumpxx1 = np.array([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]], np.float64)
    gs1.log_likelihood = log_likelihood1
    gs1.t = T1
    gs1.n = n1
    gs1.sum_px = sumpx1
    gs1.sum_pxx = sumpxx1

    gs2 = GMMStats(2, 3)
    log_likelihood2 = -4.0
    T2 = 1
    n2 = np.array([0.2, 0.8], np.float64)
    sumpx2 = np.array([[2.0, 1.0, 3.0], [3.0, 4.1, 3.2]], np.float64)
    sumpxx2 = np.array([[12.0, 15.0, 25.0], [39.0, 51.0, 62.0]], np.float64)
    gs2.log_likelihood = log_likelihood2
    gs2.t = T2
    gs2.n = n2
    gs2.sum_px = sumpx2
    gs2.sum_pxx = sumpxx2

    data = [gs1, gs2]

    # Reference values TODO: load from hdf5 file
    acc_Nij_Sigma_wij2_ref1 = np.array(
        [
            [[0.03202305, -0.02947769], [-0.02947769, 0.0561132]],
            [[0.07953279, -0.07829414], [-0.07829414, 0.13814242]],
        ]
    )
    acc_Fnorm_Sigma_wij_ref1 = np.array(
        [
            [
                [-0.29622691, 0.61411796],
                [0.09391764, -0.27955961],
                [-0.39014455, 0.89367757],
            ],
            [
                [0.04695882, -0.13977981],
                [-0.05718673, 0.24159665],
                [-0.17098161, 0.47326585],
            ],
        ],
    )
    acc_Snorm_ref1 = np.array([[16.6, 22.4, 16.6], [61.4, 55.0, 97.4]])
    N_ref1 = np.array([0.6, 1.4])
    t_ref1 = np.array(
        [
            [
                [1.59543739, 11.78239235],
                [-3.20130371, -6.66379081],
                [4.79674111, 18.44618316],
            ],
            [
                [-0.91765407, -1.5319461],
                [2.26805901, 3.03434944],
                [2.76600031, 4.9935962],
            ],
        ]
    )

    acc_Nij_Sigma_wij2_ref2 = np.array(
        [
            [[0.37558389, -0.15405228], [-0.15405228, 0.1421269]],
            [[1.02076081, -0.57683953], [-0.57683953, 0.53912239]],
        ]
    )
    acc_Fnorm_Sigma_wij_ref2 = np.array(
        [
            [
                [-1.1261668, 1.46496753],
                [-0.03579289, -0.37875811],
                [-1.09037391, 1.84372565],
            ],
            [
                [-0.01789645, -0.18937906],
                [0.35221084, 0.15854126],
                [-0.10004552, 0.72559036],
            ],
        ]
    )
    acc_Snorm_ref2 = np.array([[16.6, 22.4, 16.6], [61.4, 55.0, 97.4]])
    N_ref2 = np.array([0.6, 1.4])
    t_ref2 = np.array(
        [
            [
                [2.2133685, 12.70654597],
                [-2.13959381, -4.98404887],
                [4.35296231, 17.69059484],
            ],
            [
                [-0.54644055, -0.93594252],
                [1.29308324, 1.67762053],
                [1.67583072, 3.13894546],
            ],
        ]
    )
    acc_Nij_Sigma_wij2_ref = [acc_Nij_Sigma_wij2_ref1, acc_Nij_Sigma_wij2_ref2]
    acc_Fnorm_Sigma_wij_ref = [
        acc_Fnorm_Sigma_wij_ref1,
        acc_Fnorm_Sigma_wij_ref2,
    ]
    acc_Snorm_ref = [acc_Snorm_ref1, acc_Snorm_ref2]
    N_ref = [N_ref1, N_ref2]
    t_ref = [t_ref1, t_ref2]

    assert acc_Fnorm_Sigma_wij_ref1.shape == (2, 3, 2)
    assert acc_Fnorm_Sigma_wij_ref2.shape == (2, 3, 2)
    assert acc_Nij_Sigma_wij2_ref1.shape == (2, 2, 2)
    assert acc_Nij_Sigma_wij2_ref2.shape == (2, 2, 2)

    # Reference implementation
    # Machine
    m = IVectorMachine(ubm, dim_t=2)
    t = np.array([[[1.0, 2], [4, 1], [0, 3]], [[5, 8], [7, 10], [11, 1]]])
    sigma = np.array([[1.0, 2.0, 1.0], [3.0, 2.0, 4.0]])

    # Initialization
    trainer = IVectorTrainerPy()
    trainer.initialize(m, data)
    m.T = t
    m.sigma = sigma
    for it in range(2):
        # E-Step
        trainer.e_step(m, data)
        np.testing.assert_almost_equal(
            acc_Nij_Sigma_wij2_ref[it], trainer.m_acc_Nij_Sigma_wij2, decimal=5
        )
        np.testing.assert_almost_equal(
            acc_Fnorm_Sigma_wij_ref[it],
            trainer.m_acc_Fnorm_Sigma_wij,
            decimal=5,
        )
        np.testing.assert_almost_equal(
            acc_Snorm_ref[it], trainer.m_acc_Snorm, decimal=5
        )
        np.testing.assert_almost_equal(N_ref[it], trainer.m_N, decimal=5)

        # M-Step
        trainer.m_step(m)
        np.testing.assert_almost_equal(t_ref[it], m.T, decimal=5)

    # New implementation
    # Machine
    m = IVectorMachine(ubm, dim_t=2, update_sigma=False)
    t = np.array([[[1.0, 2], [4, 1], [0, 3]], [[5, 8], [7, 10], [11, 1]]])
    sigma = np.array([[1.0, 2.0, 1.0], [3.0, 2.0, 4.0]])

    # Initialization
    m.T = t
    m.sigma = copy.deepcopy(sigma)
    stats = None
    for it in range(2):
        # E-Step
        stats = m.e_step(data)
        np.testing.assert_almost_equal(
            acc_Nij_Sigma_wij2_ref[it], stats.nij_sigma_wij2, decimal=5
        )
        np.testing.assert_almost_equal(
            acc_Fnorm_Sigma_wij_ref[it], stats.fnorm_sigma_wij, decimal=5
        )
        np.testing.assert_almost_equal(
            acc_Snorm_ref[it], stats.snormij, decimal=5
        )
        np.testing.assert_almost_equal(N_ref[it], stats.nij, decimal=5)

        # M-Step
        m.m_step(stats)
        np.testing.assert_almost_equal(t_ref[it], m.T, decimal=5)
        np.testing.assert_equal(sigma, m.sigma)  # sigma should not be updated


def test_trainer_update_sigma():
    # Ubm
    dim_c = 2
    dim_d = 3
    ubm = GMMMachine(dim_c, dim_d)
    ubm.weights = np.array([0.4, 0.6])
    ubm.means = np.array([[1.0, 7, 4], [4, 5, 3]])
    ubm.variances = np.array([[0.5, 1.0, 1.5], [1.0, 1.5, 2.0]])

    # Defines GMMStats
    gs1 = GMMStats(dim_c, dim_d)
    log_likelihood1 = -3.0
    T1 = 1
    n1 = np.array([0.4, 0.6], np.float64)
    sumpx1 = np.array([[1.0, 2.0, 3.0], [2.0, 4.0, 3.0]], np.float64)
    sumpxx1 = np.array([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]], np.float64)
    gs1.log_likelihood = log_likelihood1
    gs1.t = T1
    gs1.n = n1
    gs1.sum_px = sumpx1
    gs1.sum_pxx = sumpxx1

    gs2 = GMMStats(dim_c, dim_d)
    log_likelihood2 = -4.0
    T2 = 1
    n2 = np.array([0.2, 0.8], np.float64)
    sumpx2 = np.array([[2.0, 1.0, 3.0], [3.0, 4.1, 3.2]], np.float64)
    sumpxx2 = np.array([[12.0, 15.0, 25.0], [39.0, 51.0, 62.0]], np.float64)
    gs2.log_likelihood = log_likelihood2
    gs2.t = T2
    gs2.n = n2
    gs2.sum_px = sumpx2
    gs2.sum_pxx = sumpxx2

    data = [gs1, gs2]

    # Reference values TODO: load from hdf5 file
    acc_Nij_Sigma_wij2_ref1 = np.array(
        [
            [[0.03202305, -0.02947769], [-0.02947769, 0.0561132]],
            [[0.07953279, -0.07829414], [-0.07829414, 0.13814242]],
        ]
    )
    acc_Fnorm_Sigma_wij_ref1 = np.array(
        [
            [
                [-0.29622691, 0.61411796],
                [0.09391764, -0.27955961],
                [-0.39014455, 0.89367757],
            ],
            [
                [0.04695882, -0.13977981],
                [-0.05718673, 0.24159665],
                [-0.17098161, 0.47326585],
            ],
        ]
    )
    acc_Snorm_ref1 = np.array([[16.6, 22.4, 16.6], [61.4, 55.0, 97.4]])
    N_ref1 = np.array([0.6, 1.4])
    t_ref1 = np.array(
        [
            [
                [1.59543739, 11.78239235],
                [-3.20130371, -6.66379081],
                [4.79674111, 18.44618316],
            ],
            [
                [-0.91765407, -1.5319461],
                [2.26805901, 3.03434944],
                [2.76600031, 4.9935962],
            ],
        ]
    )
    sigma_ref1 = np.array(
        [
            [16.39472121, 34.72955353, 3.3108037],
            [43.73496916, 38.85472445, 68.22116903],
        ]
    )

    acc_Nij_Sigma_wij2_ref2 = np.array(
        [
            [[0.50807426, -0.11907756], [-0.11907756, 0.12336544]],
            [[1.18602399, -0.2835859], [-0.2835859, 0.39440498]],
        ]
    )
    acc_Fnorm_Sigma_wij_ref2 = np.array(
        [
            [
                [0.07221453, 1.1189786],
                [-0.08681275, -0.35396112],
                [0.15902728, 1.47293972],
            ],
            [
                [-0.04340637, -0.17698056],
                [0.10662127, 0.21484933],
                [0.13116645, 0.64474271],
            ],
        ]
    )
    acc_Snorm_ref2 = np.array([[16.6, 22.4, 16.6], [61.4, 55.0, 97.4]])
    N_ref2 = np.array([0.6, 1.4])
    t_ref2 = np.array(
        [
            [
                [2.93105054, 11.89961223],
                [-1.08988119, -3.92120757],
                [4.02093173, 15.82081981],
            ],
            [
                [-0.17376634, -0.57366984],
                [0.26585634, 0.73589952],
                [0.60557877, 2.07014704],
            ],
        ]
    )
    sigma_ref2 = np.array(
        [
            [5.12154025e00, 3.48623823e01, 1.00000000e-05],
            [4.37792350e01, 3.91525332e01, 6.85613258e01],
        ]
    )

    acc_Nij_Sigma_wij2_ref = [acc_Nij_Sigma_wij2_ref1, acc_Nij_Sigma_wij2_ref2]
    acc_Fnorm_Sigma_wij_ref = [
        acc_Fnorm_Sigma_wij_ref1,
        acc_Fnorm_Sigma_wij_ref2,
    ]
    acc_Snorm_ref = [acc_Snorm_ref1, acc_Snorm_ref2]
    N_ref = [N_ref1, N_ref2]
    t_ref = [t_ref1, t_ref2]
    sigma_ref = [sigma_ref1, sigma_ref2]

    # Reference implementation
    # Machine
    m = IVectorMachine(ubm, 2)
    t = np.array([[[1.0, 2], [4, 1], [0, 3]], [[5, 8], [7, 10], [11, 1]]])
    sigma = np.array([[1.0, 2.0, 1.0], [3.0, 2.0, 4.0]])

    # Initialization
    trainer = IVectorTrainerPy(sigma_update=True)
    trainer.initialize(m, data)
    m.T = t
    m.sigma = sigma

    for it in range(2):
        # E-Step
        trainer.e_step(m, data)
        np.testing.assert_almost_equal(
            acc_Nij_Sigma_wij2_ref[it], trainer.m_acc_Nij_Sigma_wij2, decimal=5
        )
        np.testing.assert_almost_equal(
            acc_Fnorm_Sigma_wij_ref[it],
            trainer.m_acc_Fnorm_Sigma_wij,
            decimal=5,
        )
        np.testing.assert_almost_equal(
            acc_Snorm_ref[it], trainer.m_acc_Snorm, decimal=5
        )
        np.testing.assert_almost_equal(N_ref[it], trainer.m_N, decimal=5)

        # M-Step
        trainer.m_step(m)
        np.testing.assert_almost_equal(t_ref[it], m.T, decimal=5)
        np.testing.assert_almost_equal(sigma_ref[it], m.sigma, decimal=5)

    # New implementation
    # Machine
    m = IVectorMachine(
        ubm, dim_t=2, variance_floor=1e-5
    )  # update_sigma is True by default

    # Manual Initialization
    m.T = t
    m.sigma = sigma
    for it in range(2):
        # E-Step
        stats = m.e_step(data)
        np.testing.assert_almost_equal(
            acc_Nij_Sigma_wij2_ref[it], stats.nij_sigma_wij2, decimal=5
        )
        np.testing.assert_almost_equal(
            acc_Fnorm_Sigma_wij_ref[it], stats.fnorm_sigma_wij, decimal=5
        )
        np.testing.assert_almost_equal(
            acc_Snorm_ref[it], stats.snormij, decimal=5
        )
        np.testing.assert_almost_equal(N_ref[it], stats.nij, decimal=5)

        # M-Step
        m.m_step(stats)
        np.testing.assert_almost_equal(t_ref[it], m.T, decimal=5)
        np.testing.assert_almost_equal(sigma_ref[it], m.sigma, decimal=5)


def test_trainer_update_sigma_parallel():
    # Ubm
    dim_c = 2
    dim_d = 3
    ubm = GMMMachine(dim_c, dim_d)
    ubm.weights = np.array([0.4, 0.6])
    ubm.means = np.array([[1.0, 7, 4], [4, 5, 3]])
    ubm.variances = np.array([[0.5, 1.0, 1.5], [1.0, 1.5, 2.0]])

    # Defines GMMStats
    gs1 = GMMStats(dim_c, dim_d)
    log_likelihood1 = -3.0
    T1 = 1
    n1 = np.array([0.4, 0.6], np.float64)
    sumpx1 = np.array([[1.0, 2.0, 3.0], [2.0, 4.0, 3.0]], np.float64)
    sumpxx1 = np.array([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]], np.float64)
    gs1.log_likelihood = log_likelihood1
    gs1.t = T1
    gs1.n = n1
    gs1.sum_px = sumpx1
    gs1.sum_pxx = sumpxx1

    gs2 = GMMStats(dim_c, dim_d)
    log_likelihood2 = -4.0
    T2 = 1
    n2 = np.array([0.2, 0.8], np.float64)
    sumpx2 = np.array([[2.0, 1.0, 3.0], [3.0, 4.1, 3.2]], np.float64)
    sumpxx2 = np.array([[12.0, 15.0, 25.0], [39.0, 51.0, 62.0]], np.float64)
    gs2.log_likelihood = log_likelihood2
    gs2.t = T2
    gs2.n = n2
    gs2.sum_px = sumpx2
    gs2.sum_pxx = sumpxx2

    # data = [gs1, gs2]

    # Reference values
    # acc_Nij_Sigma_wij2_ref1 = {
    #     0: np.array([[0.03202305, -0.02947769], [-0.02947769, 0.0561132]]),
    #     1: np.array([[0.07953279, -0.07829414], [-0.07829414, 0.13814242]]),
    # }
    # acc_Fnorm_Sigma_wij_ref1 = {
    #     0: np.array(
    #         [
    #             [-0.29622691, 0.61411796],
    #             [0.09391764, -0.27955961],
    #             [-0.39014455, 0.89367757],
    #         ]
    #     ),
    #     1: np.array(
    #         [
    #             [0.04695882, -0.13977981],
    #             [-0.05718673, 0.24159665],
    #             [-0.17098161, 0.47326585],
    #         ]
    #     ),
    # }
    # acc_Snorm_ref1 = np.array([16.6, 22.4, 16.6, 61.4, 55.0, 97.4])
    # N_ref1 = np.array([0.6, 1.4])
    # t_ref1 = np.array(
    #     [
    #         [1.59543739, 11.78239235],
    #         [-3.20130371, -6.66379081],
    #         [4.79674111, 18.44618316],
    #         [-0.91765407, -1.5319461],
    #         [2.26805901, 3.03434944],
    #         [2.76600031, 4.9935962],
    #     ]
    # )
    # sigma_ref1 = np.array(
    #     [
    #         16.39472121,
    #         34.72955353,
    #         3.3108037,
    #         43.73496916,
    #         38.85472445,
    #         68.22116903,
    #     ]
    # )

    # acc_Nij_Sigma_wij2_ref2 = {
    #     0: np.array([[0.50807426, -0.11907756], [-0.11907756, 0.12336544]]),
    #     1: np.array([[1.18602399, -0.2835859], [-0.2835859, 0.39440498]]),
    # }
    # acc_Fnorm_Sigma_wij_ref2 = {
    #     0: np.array(
    #         [
    #             [0.07221453, 1.1189786],
    #             [-0.08681275, -0.35396112],
    #             [0.15902728, 1.47293972],
    #         ]
    #     ),
    #     1: np.array(
    #         [
    #             [-0.04340637, -0.17698056],
    #             [0.10662127, 0.21484933],
    #             [0.13116645, 0.64474271],
    #         ]
    #     ),
    # }
    # acc_Snorm_ref2 = np.array([16.6, 22.4, 16.6, 61.4, 55.0, 97.4])
    # N_ref2 = np.array([0.6, 1.4])
    # t_ref2 = np.array(
    #     [
    #         [2.93105054, 11.89961223],
    #         [-1.08988119, -3.92120757],
    #         [4.02093173, 15.82081981],
    #         [-0.17376634, -0.57366984],
    #         [0.26585634, 0.73589952],
    #         [0.60557877, 2.07014704],
    #     ]
    # )
    # sigma_ref2 = np.array(
    #     [
    #         5.12154025e00,
    #         3.48623823e01,
    #         1.00000000e-05,
    #         4.37792350e01,
    #         3.91525332e01,
    #         6.85613258e01,
    #     ]
    # )

    # acc_Nij_Sigma_wij2_ref = [acc_Nij_Sigma_wij2_ref1, acc_Nij_Sigma_wij2_ref2]
    # acc_Fnorm_Sigma_wij_ref = [
    #     acc_Fnorm_Sigma_wij_ref1,
    #     acc_Fnorm_Sigma_wij_ref2,
    # ]
    # acc_Snorm_ref = [acc_Snorm_ref1, acc_Snorm_ref2]
    # N_ref = [N_ref1, N_ref2]
    # t_ref = [t_ref1, t_ref2]
    # sigma_ref = [sigma_ref1, sigma_ref2]

    # # Machine
    # t = np.array([[1.0, 2], [4, 1], [0, 3], [5, 8], [7, 10], [11, 1]])
    # sigma = np.array([1.0, 2.0, 1.0, 3.0, 2.0, 4.0])

    # C++ implementation TODO
    # Machine
    # serial_m = IVectorMachine(ubm, 2)
    # serial_m.variance_threshold = 1e-5

    # SERIAL TRAINER
    # serial_trainer = IVectorTrainer(update_sigma=True)
    # serial_m.t = t
    # serial_m.sigma = sigma

    # bob.learn.em.train(
    #     serial_trainer, serial_m, data, max_iterations=5, initialize=True
    # )

    # # PARALLEL TRAINER
    # parallel_m = IVectorMachine(ubm, 2)
    # parallel_m.variance_threshold = 1e-5

    # parallel_trainer = IVectorTrainer(update_sigma=True)
    # parallel_m.t = t
    # parallel_m.sigma = sigma

    # bob.learn.em.train(
    #     parallel_trainer,
    #     parallel_m,
    #     data,
    #     max_iterations=5,
    #     initialize=True,
    #     pool=2,
    # )

    # assert np.allclose(
    #     serial_trainer.acc_nij_wij2, parallel_trainer.acc_nij_wij2, 1e-5
    # )
    # assert np.allclose(
    #     serial_trainer.acc_fnormij_wij, parallel_trainer.acc_fnormij_wij, 1e-5
    # )
    # assert np.allclose(
    #     serial_trainer.acc_snormij, parallel_trainer.acc_snormij, 1e-5
    # )
    # assert np.allclose(serial_trainer.acc_nij, parallel_trainer.acc_nij, 1e-5)
