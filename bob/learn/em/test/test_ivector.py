#!/usr/bin/env python
# @author: Yannick Dayer <yannick.dayer@idiap.ch>
# @date: Fri 06 May 2022 12:59:21 UTC+02

import numpy as np

from bob.learn.em import GMMMachine, GMMStats, IVectorMachine


def test_ivector_machine_projection():
    # Preset IVector Machine parameters
    t = np.array(
        [[1, 2], [4, 1], [0, 3], [5, 8], [7, 10], [11, 1]], dtype=float
    )
    sigma = np.array([1, 2, 1, 3, 2, 4], dtype=float)

    # Create the UBM and set its values manually
    ubm = GMMMachine(n_gaussians=2)
    ubm.weights = np.array([0.4, 0.6], dtype=float)
    ubm.means = np.array([[1, 7, 4], [4, 5, 3]], dtype=float)
    ubm.variances = np.array([[0.5, 1.0, 1.5], [1.0, 1.5, 2.0]], dtype=float)

    # Manually create a feature projected on the UBM
    gmm_projection = GMMStats(ubm.n_gaussians, ubm.means.shape[-1])
    gmm_projection.t = 1
    gmm_projection.n = np.array([0.4, 0.6], dtype=float)
    gmm_projection.sum_px = np.array([[1, 2, 3], [2, 4, 3]], dtype=float)
    gmm_projection.sum_pxx = np.array([[10, 20, 30], [40, 50, 60]], dtype=float)

    machine = IVectorMachine(ubm=ubm, dim_t=2)
    machine.t = t
    machine.sigma = sigma

    # Reference from C++ implementation
    ivector_projection_ref = np.array([-0.04213415, 0.21463343])
    ivector_projection = machine.project(gmm_projection)
    np.testing.assert_almost_equal(
        ivector_projection_ref, ivector_projection, decimal=5
    )


def test_ivector_machine_training():
    assert False, "Not implemented"
