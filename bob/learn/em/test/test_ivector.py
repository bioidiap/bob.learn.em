#!/usr/bin/env python
# @author: Yannick Dayer <yannick.dayer@idiap.ch>
# @date: Fri 06 May 2022 12:59:21 UTC+02

import contextlib
import copy

import dask.distributed
import numpy as np

from h5py import File as HDF5File
from pkg_resources import resource_filename

from bob.learn.em import GMMMachine, GMMStats, IVectorMachine

from .test_kmeans import to_dask_array, to_numpy


@contextlib.contextmanager
def _dask_distributed_context():
    try:
        client = dask.distributed.Client()
        with client.as_current():
            yield client
    finally:
        client.close()


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
    np.testing.assert_almost_equal(
        transformed, np.array([0.02774721, -0.35237828]), decimal=7
    )


def test_ivector_machine_training():

    gs1 = GMMStats.from_hdf5(
        resource_filename("bob.learn.em", "data/ivector_gs1.hdf5")
    )
    gs2 = GMMStats.from_hdf5(
        resource_filename("bob.learn.em", "data/ivector_gs2.hdf5")
    )

    data = [gs1, gs2]

    # Define the ubm
    ubm = GMMMachine(n_gaussians=3)
    ubm.means = np.array([[1, 2, 3], [6, 7, 8], [10, 11, 12]])
    ubm.variances = np.ones((3, 3))

    machine = IVectorMachine(ubm=ubm, dim_t=2)
    machine.fit(data)

    assert False, "TODO: add tests (with projection)"


def _load_references_from_file(filename):
    """Loads the IVectorStats references, T, and sigma for one step"""
    with HDF5File(filename, "r") as f:
        keys = (
            "nij_sigma_wij2",
            "fnorm_sigma_wij",
            "nij",
            "snormij",
            "T",
            "sigma",
        )
        ret = {k: f[k][()] for k in keys}
    return ret


def test_trainer_nosigma():
    # Ubm
    ubm = GMMMachine(2)
    ubm.means = np.array([[1.0, 7, 4], [4, 5, 3]])
    ubm.variances = np.array([[0.5, 1.0, 1.5], [1.0, 1.5, 2.0]])
    ubm.weights = np.array([0.4, 0.6])

    data = [
        GMMStats.from_hdf5(
            resource_filename("bob.learn.em", f"data/ivector_gs{i+1}.hdf5")
        )
        for i in range(2)
    ]

    references = [
        _load_references_from_file(
            resource_filename(
                "bob.learn.em", f"data/ivector_ref_nosigma_step{i+1}.hdf5"
            )
        )
        for i in range(2)
    ]

    # Machine
    m = IVectorMachine(ubm, dim_t=2, update_sigma=False)

    # Initialization
    m.T = np.array([[[1.0, 2], [4, 1], [0, 3]], [[5, 8], [7, 10], [11, 1]]])
    init_sigma = np.array([[1.0, 2.0, 1.0], [3.0, 2.0, 4.0]])
    m.sigma = copy.deepcopy(init_sigma)
    stats = None
    for it in range(2):
        # E-Step
        stats = m.e_step(data)
        np.testing.assert_almost_equal(
            references[it]["nij_sigma_wij2"], stats.nij_sigma_wij2, decimal=5
        )
        np.testing.assert_almost_equal(
            references[it]["fnorm_sigma_wij"], stats.fnorm_sigma_wij, decimal=5
        )
        np.testing.assert_almost_equal(
            references[it]["snormij"], stats.snormij, decimal=5
        )
        np.testing.assert_almost_equal(
            references[it]["nij"], stats.nij, decimal=5
        )

        # M-Step
        m.m_step(stats)
        np.testing.assert_almost_equal(references[it]["T"], m.T, decimal=5)
        np.testing.assert_equal(
            init_sigma, m.sigma
        )  # sigma should not be updated


def test_trainer_update_sigma():
    # Ubm
    ubm = GMMMachine(n_gaussians=2)
    ubm.weights = np.array([0.4, 0.6])
    ubm.means = np.array([[1.0, 7, 4], [4, 5, 3]])
    ubm.variances = np.array([[0.5, 1.0, 1.5], [1.0, 1.5, 2.0]])

    data = [
        GMMStats.from_hdf5(
            resource_filename("bob.learn.em", f"data/ivector_gs{i+1}.hdf5")
        )
        for i in range(2)
    ]

    references = [
        _load_references_from_file(
            resource_filename(
                "bob.learn.em", f"data/ivector_ref_step{i+1}.hdf5"
            )
        )
        for i in range(2)
    ]

    # Machine
    m = IVectorMachine(
        ubm, dim_t=2, variance_floor=1e-5
    )  # update_sigma is True by default

    # Manual Initialization
    m.T = np.array([[[1.0, 2], [4, 1], [0, 3]], [[5, 8], [7, 10], [11, 1]]])
    m.sigma = np.array([[1.0, 2.0, 1.0], [3.0, 2.0, 4.0]])

    for it in range(2):
        # E-Step
        stats = m.e_step(data)
        np.testing.assert_almost_equal(
            references[it]["nij_sigma_wij2"], stats.nij_sigma_wij2, decimal=5
        )
        np.testing.assert_almost_equal(
            references[it]["fnorm_sigma_wij"], stats.fnorm_sigma_wij, decimal=5
        )
        np.testing.assert_almost_equal(
            references[it]["snormij"], stats.snormij, decimal=5
        )
        np.testing.assert_almost_equal(
            references[it]["nij"], stats.nij, decimal=5
        )

        # M-Step
        m.m_step(stats)
        np.testing.assert_almost_equal(references[it]["T"], m.T, decimal=5)
        np.testing.assert_almost_equal(
            references[it]["sigma"], m.sigma, decimal=5
        )


def test_ivector_fit_parallel():
    # Ubm
    ubm = GMMMachine(n_gaussians=2)
    ubm.weights = np.array([0.4, 0.6])
    ubm.means = np.array([[1.0, 7, 4], [4, 5, 3]])
    ubm.variances = np.array([[0.5, 1.0, 1.5], [1.0, 1.5, 2.0]])

    fit_data = np.random.normal(
        loc=1, scale=1, size=(100, 3)
    )  # TODO save to ref file
    test_data = np.random.normal(
        loc=1, scale=1.2, size=(50, 3)
    )  # TODO save to ref file
    reference_result = np.array([10, 10] * 50)
    # TODO load ref files

    with _dask_distributed_context():
        for transform in to_numpy, to_dask_array:
            fit_data = transform(fit_data)
            projected_data = ubm.transform([d for d in fit_data])
            m = IVectorMachine(ubm=ubm, dim_t=2, max_iterations=2)
            m.fit([d for d in projected_data])
            result = m.transform([d for d in test_data])
            np.testing.assert_almost_equal(
                np.array(result), reference_result, decimal=5
            )

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
