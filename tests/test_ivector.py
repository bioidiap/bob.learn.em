#!/usr/bin/env python
# @author: Yannick Dayer <yannick.dayer@idiap.ch>
# @date: Fri 06 May 2022 12:59:21 UTC+02

import contextlib
import copy

import dask.bag
import dask.distributed
import numpy as np

from h5py import File as HDF5File
from pkg_resources import resource_filename

from bob.learn.em import GMMMachine, GMMStats, IVectorMachine
from bob.learn.em.ivector import e_step, m_step

from .test_kmeans import to_numpy


@contextlib.contextmanager
def _dask_distributed_context():
    try:
        client = dask.distributed.Client()
        with client.as_current():
            yield client
    finally:
        client.close()


def to_dask_bag(*args):
    """Converts all args into dask Bags."""
    result = []
    for x in args:
        x = np.asarray(x)
        result.append(dask.bag.from_sequence(x, npartitions=x.shape[0] * 2))
    if len(result) == 1:
        return result[0]
    return result


def test_ivector_machine_base():
    # Create the UBM and set its values manually
    ubm = GMMMachine(n_gaussians=2)
    ubm.weights = np.array([0.4, 0.6], dtype=float)
    ubm.means = np.array([[1, 7, 4], [4, 5, 3]], dtype=float)
    ubm.variances = np.array([[0.5, 1.0, 1.5], [1.0, 1.5, 2.0]], dtype=float)

    machine = IVectorMachine(ubm=ubm, dim_t=4)

    assert hasattr(machine, "ubm")
    assert hasattr(machine, "T")
    assert hasattr(machine, "sigma")

    assert machine.T is None
    assert machine.sigma is None


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
    machine.sigma = ubm.variances.copy()
    assert hasattr(machine, "fit")
    assert hasattr(machine, "transform")

    transformed = machine.transform(ubm.transform([np.array([1, 2, 3])]))[0]
    assert isinstance(transformed, np.ndarray)
    np.testing.assert_almost_equal(
        transformed, np.array([0.02774721, -0.35237828]), decimal=7
    )


def test_ivector_machine_training():
    gs1 = GMMStats.from_hdf5(
        resource_filename(__name__, "data/ivector_gs1.hdf5")
    )
    gs2 = GMMStats.from_hdf5(
        resource_filename(__name__, "data/ivector_gs2.hdf5")
    )

    data = [gs1, gs2]

    # Define the ubm
    ubm = GMMMachine(n_gaussians=2)
    ubm.means = np.array([[1, 2, 3], [6, 7, 8]])
    ubm.variances = np.ones((2, 3))

    np.random.seed(0)

    machine = IVectorMachine(ubm=ubm, dim_t=2)
    machine.fit(data)

    test_data = GMMStats(2, 3)
    test_data.t = 1
    test_data.log_likelihood = -0.5
    test_data.n = np.array([0.5, 0.5])
    test_data.sum_px = np.array([[8, 0, 4], [6, 6, 6]])
    test_data.sum_pxx = np.array([[10, 20, 30], [60, 70, 80]])
    projected = machine.project(test_data)

    proj_reference = np.array([0.94234370, -0.61558459])

    np.testing.assert_almost_equal(projected, proj_reference, decimal=4)


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
            resource_filename(__name__, f"data/ivector_gs{i+1}.hdf5")
        )
        for i in range(2)
    ]

    references = [
        _load_references_from_file(
            resource_filename(
                __name__, f"data/ivector_ref_nosigma_step{i+1}.hdf5"
            )
        )
        for i in range(2)
    ]

    # Machine
    m = IVectorMachine(ubm, dim_t=2, update_sigma=False)

    # Manual Initialization
    m.dim_c = ubm.n_gaussians
    m.dim_d = ubm.shape[-1]
    m.T = np.array([[[1.0, 2], [4, 1], [0, 3]], [[5, 8], [7, 10], [11, 1]]])
    init_sigma = np.array([[1.0, 2.0, 1.0], [3.0, 2.0, 4.0]])
    m.sigma = copy.deepcopy(init_sigma)
    stats = None
    for it in range(2):
        # E-Step
        stats = e_step(m, data)
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
        m_step(m, stats)
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
            resource_filename(__name__, f"data/ivector_gs{i+1}.hdf5")
        )
        for i in range(2)
    ]

    references = [
        _load_references_from_file(
            resource_filename(__name__, f"data/ivector_ref_step{i+1}.hdf5")
        )
        for i in range(2)
    ]

    # Machine
    m = IVectorMachine(
        ubm, dim_t=2, variance_floor=1e-5
    )  # update_sigma is True by default

    # Manual Initialization
    m.dim_c = ubm.n_gaussians
    m.dim_d = ubm.shape[-1]
    m.T = np.array([[[1.0, 2], [4, 1], [0, 3]], [[5, 8], [7, 10], [11, 1]]])
    m.sigma = np.array([[1.0, 2.0, 1.0], [3.0, 2.0, 4.0]])

    for it in range(2):
        # E-Step
        stats = e_step(m, data)
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
        m_step(m, stats)
        np.testing.assert_almost_equal(references[it]["T"], m.T, decimal=5)
        np.testing.assert_almost_equal(
            references[it]["sigma"], m.sigma, decimal=5
        )


def test_ivector_fit():
    # Ubm
    ubm = GMMMachine(n_gaussians=2)
    ubm.weights = np.array([0.4, 0.6])
    ubm.means = np.array([[1.0, 7, 4], [4, 5, 3]])
    ubm.variances = np.array([[0.5, 1.0, 1.5], [1.0, 1.5, 2.0]])

    fit_data_file = resource_filename(__name__, "data/ivector_fit_data.hdf5")
    with HDF5File(fit_data_file, "r") as f:
        fit_data = f["array"][()]

    test_data_file = resource_filename(__name__, "data/ivector_test_data.hdf5")
    with HDF5File(test_data_file, "r") as f:
        test_data = f["array"][()]

    reference_result_file = resource_filename(
        __name__, "data/ivector_results.hdf5"
    )
    with HDF5File(reference_result_file, "r") as f:
        reference_result = f["array"][()]

    # Serial test
    np.random.seed(0)
    fit_data = to_numpy(fit_data)
    projected_data = ubm.transform(fit_data)
    m = IVectorMachine(ubm=ubm, dim_t=2, max_iterations=2)
    m.fit(projected_data)
    result = m.transform(ubm.transform(test_data))
    np.testing.assert_almost_equal(result, reference_result, decimal=5)

    # Parallel test
    with _dask_distributed_context():
        for transform in [to_numpy, to_dask_bag]:
            np.random.seed(0)
            fit_data = transform(fit_data)
            projected_data = ubm.transform(fit_data)
            projected_data = transform(projected_data)
            m = IVectorMachine(ubm=ubm, dim_t=2, max_iterations=2)
            m.fit(projected_data)
            result = m.transform(ubm.transform(test_data))
            np.testing.assert_almost_equal(
                np.array(result), reference_result, decimal=5
            )
