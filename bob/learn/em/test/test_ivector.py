#!/usr/bin/env python
# @author: Yannick Dayer <yannick.dayer@idiap.ch>
# @date: Fri 06 May 2022 12:59:21 UTC+02

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
    ubm = GMMMachine(n_gaussians=2)
    ubm.means = np.array([[1, 7, 4], [4, 5, 3]], dtype=float)
    ubm.variances = np.array([[0.5, 1.0, 1.5], [1.0, 1.5, 2.0]], dtype=float)
    machine = IVectorMachine(ubm=ubm, dim_t=2)
    assert hasattr(machine, "fit")
    assert hasattr(machine, "transform")
    assert hasattr(machine, "enroll")
    assert hasattr(machine, "score")

    transformed = machine.transform([np.ndarray([1, 2, 3])])[0]
    assert isinstance(transformed, IVectorMachine)
    nij_sigma_wij2_ref = np.array([[0.5, 1.0, 1.5], [1.0, 1.5, 2.0]])  # TODO
    nij_ref = np.array([[1, 2, 3], [4, 5, 6]])
    fnorm_sigma_wij_ref = np.array([[0.5, 1.0, 1.5], [1.0, 1.5, 2.0]])  # TODO
    snormij_ref = np.array([[1, 2, 3], [4, 5, 6]])
    np.testing.assert_almost_equal(
        transformed.acc_nij_sigma_wij2, nij_sigma_wij2_ref
    )
    np.testing.assert_almost_equal(transformed.acc_nij, nij_ref)
    np.testing.assert_almost_equal(
        transformed.acc_fnorm_sigma_wij, fnorm_sigma_wij_ref
    )
    np.testing.assert_almost_equal(transformed.acc_snormij, snormij_ref)


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
