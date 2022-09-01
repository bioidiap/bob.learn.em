#!/usr/bin/env python
# @author: Yannick Dayer <yannick.dayer@idiap.ch>
# @date: Fri 06 May 2022 14:18:25 UTC+02

import copy
import logging

from typing import Any, Dict, List, Optional, Tuple

import dask
import dask.array as da
import numpy as np

from sklearn.base import BaseEstimator

from bob.learn.em import GMMMachine, GMMStats

logger = logging.getLogger("__name__")


class IVectorStats:
    """Stores I-Vector statistics. Can be used to accumulate multiple statistics.

    **Attributes:**
        nij_sigma_wij2: numpy.ndarray of shape (n_gaussians,dim_t,dim_t)
        fnorm_sigma_wij: numpy.ndarray of shape (n_gaussians,n_features,dim_t)
        snormij: numpy.ndarray of shape (n_gaussians,n_features)
        nij: numpy.ndarray of shape (n_gaussians,)
    """

    def __init__(self, dim_c, dim_d, dim_t):
        self.dim_c = dim_c
        self.dim_d = dim_d
        self.dim_t = dim_t

        # Accumulator storage variables

        # nij sigma wij2: shape = (c,t,t)
        self.nij_sigma_wij2 = np.zeros(
            shape=(self.dim_c, self.dim_t, self.dim_t), dtype=float
        )
        # fnorm sigma wij: shape = (c,d,t)
        self.fnorm_sigma_wij = np.zeros(
            shape=(self.dim_c, self.dim_d, self.dim_t), dtype=float
        )
        # Snormij (used only when updating sigma)
        self.snormij = np.zeros(
            shape=(
                self.dim_c,
                self.dim_d,
            ),
            dtype=float,
        )
        # Nij (used only when updating sigma)
        self.nij = np.zeros(shape=(self.dim_c,), dtype=float)

    @property
    def shape(self) -> Tuple[int, int, int]:
        return (self.dim_c, self.dim_d, self.dim_t)

    def __add__(self, other):
        if self.shape != other.shape:
            raise ValueError("Cannot add stats of different shapes")
        result = IVectorStats(self.dim_c, self.dim_d, self.dim_t)
        result.nij_sigma_wij2 = self.nij_sigma_wij2 + other.nij_sigma_wij2
        result.fnorm_sigma_wij = self.fnorm_sigma_wij + other.fnorm_sigma_wij
        result.snormij = self.snormij + other.snormij
        result.nij = self.nij + other.nij
        return result

    def __iadd__(self, other):
        if self.shape != other.shape:
            raise ValueError("Cannot add stats of different shapes")
        self.nij_sigma_wij2 += other.nij_sigma_wij2
        self.fnorm_sigma_wij += other.fnorm_sigma_wij
        self.snormij += other.snormij
        self.nij += other.nij
        return self


def compute_tct_sigmac_inv(T: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """Computes T_{c}^{T}.sigma_{c}^{-1}"""
    # TT_sigma_inv (c,t,d) = T.T (c,t,d) / sigma (c,1,d)
    Tct_sigmacInv = T.transpose(0, 2, 1) / sigma[:, None, :]

    # Tt_sigma_inv (c,t,d)
    return Tct_sigmacInv


def compute_tct_sigmac_inv_tc(T: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """Computes T_{c}^{T}.sigma_{c}^{-1}.T_{c}"""
    tct_sigmac_inv = compute_tct_sigmac_inv(T, sigma)

    # (c,t,t) = (c,t,d) @ (c,d,t)
    Tct_sigmacInv_Tc = tct_sigmac_inv @ T

    # Output: shape (c,t,t)
    return Tct_sigmacInv_Tc


def compute_id_tt_sigma_inv_t(
    stats: GMMStats, T: np.ndarray, sigma: np.ndarray
) -> np.ndarray:
    dim_t = T.shape[-1]
    tct_sigmac_inv_tc = compute_tct_sigmac_inv_tc(T, sigma)

    output = np.eye(dim_t, dim_t) + np.einsum(
        "c,ctu->tu", stats.n, tct_sigmac_inv_tc
    )

    # Output: (t,t)
    return output


def compute_tt_sigma_inv_fnorm(
    ubm_means: np.ndarray, stats: GMMStats, T: np.ndarray, sigma: np.ndarray
) -> np.ndarray:
    """Computes \f$(Id + \\sum_{c=1}^{C} N_{i,j,c} T^{T} \\Sigma_{c}^{-1} T)\f$

    Returns an array of shape (t,)
    """

    tct_sigmac_inv = compute_tct_sigmac_inv(T, sigma)  # (c,t,d)
    fnorm = stats.sum_px - stats.n[:, None] * ubm_means  # (c,d)

    # (t,) += (t,d) @ (d) [repeated c times]
    output = np.einsum("ctd,cd->t", tct_sigmac_inv, fnorm)

    # Output: shape (t,)
    return output


def e_step(machine: "IVectorMachine", data: List[GMMStats]) -> IVectorStats:
    """Computes the expectation step of the e-m algorithm."""
    stats = IVectorStats(machine.dim_c, machine.dim_d, machine.dim_t)

    for sample in data:
        Nij = sample.n
        Fij = sample.sum_px
        Sij = sample.sum_pxx

        # Estimate latent variables
        TtSigmaInv_Fnorm = compute_tt_sigma_inv_fnorm(
            machine.ubm.means, sample, machine.T, machine.sigma
        )  # self.compute_TtSigmaInvFnorm(data[n]) # shape: (t,)
        I_TtSigmaInvNT = compute_id_tt_sigma_inv_t(
            sample, machine.T, machine.sigma
        )  # self.compute_Id_TtSigmaInvT(data[n]), # shape: (t,t)

        # Latent variables
        I_TtSigmaInvNT_inv = np.linalg.inv(I_TtSigmaInvNT)  # shape: (t,t)
        sigma_w_ij = np.dot(I_TtSigmaInvNT_inv, TtSigmaInv_Fnorm)  # shape: (t,)
        sigma_w_ij2 = I_TtSigmaInvNT_inv + np.outer(
            sigma_w_ij, sigma_w_ij
        )  # shape: (t,t)

        # Compute normalized statistics
        Fnorm = Fij - Nij[:, None] * machine.ubm.means
        Snorm = (
            Sij
            - (2 * Fij * machine.ubm.means)
            + (Nij[:, None] * machine.ubm.means * machine.ubm.means)
        )

        # Do the accumulation for each component
        stats.snormij = stats.snormij + Snorm  # shape: (c, d)

        # (c,t,t) += (c,) * (t,t)
        stats.nij_sigma_wij2 = stats.nij_sigma_wij2 + (
            Nij[:, None, None] * sigma_w_ij2[None, :, :]
        )  # (c,t,t)
        stats.nij = stats.nij + Nij
        stats.fnorm_sigma_wij = stats.fnorm_sigma_wij + np.matmul(
            Fnorm[:, :, None], sigma_w_ij[None, :]
        )  # (c,d,t)

    return stats


def m_step(machine: "IVectorMachine", stats: IVectorStats) -> "IVectorMachine":
    """Updates the Machine with the maximization step of the e-m algorithm."""
    for c in range(machine.dim_c):  # TODO Vectorize
        # T update
        A = stats.nij_sigma_wij2[c].transpose()
        B = stats.fnorm_sigma_wij[c].transpose()
        if not A.any():  # if all A == 0
            X = np.zeros(shape=(machine.dim_t, machine.dim_d), dtype=np.float64)
        else:
            X = np.linalg.solve(A, B)
        machine.T[c, :] = X.transpose()
        # Sigma update
        if machine.update_sigma:
            Fnorm_sigma_w_ij_Tt = np.diag(np.dot(stats.fnorm_sigma_wij[c], X))
            machine.sigma[c] = (
                stats.snormij[c] - Fnorm_sigma_w_ij_Tt
            ) / stats.nij[c]
            machine.sigma[c][
                machine.sigma[c] < machine.variance_floor
            ] = machine.variance_floor
    return machine


class IVectorMachine(BaseEstimator):
    """Trains and projects data using I-Vector.

    Dimensions:
        - dim_c: number of Gaussians
        - dim_d: number of features
        - dim_t: dimension of the i-vector

    **Attributes**

    T (c,d,t):
        The total variability matrix :math:`T`
    sigma (c,d):
        The diagonal covariance matrix :math:`Sigma`

    """

    def __init__(
        self,
        ubm: GMMMachine,
        dim_t: int = 2,
        convergence_threshold: Optional[float] = None,
        max_iterations: int = 25,
        update_sigma: bool = True,
        variance_floor: float = 1e-10,
        **kwargs,
    ) -> None:
        """Initializes the IVectorMachine object.

        **Parameters**

        ubm
            The Universal Background Model.
        dim_t
            The dimension of the i-vector.
        """

        super().__init__(**kwargs)
        self.ubm = ubm
        self.dim_t = dim_t
        self.convergence_threshold = convergence_threshold
        self.max_iterations = max_iterations
        self.update_sigma = update_sigma
        self.dim_c = None
        self.dim_d = None
        self.variance_floor = variance_floor

        self.T = None
        self.sigma = None

        if self.convergence_threshold:
            logger.info(
                "The convergence threshold is ignored by IVectorMachine."
            )

    def fit(self, X: np.ndarray, y=None) -> "IVectorMachine":
        """Trains the IVectorMachine.

        Repeats the e-m steps until the convergence criterion is met or
        ``max_iterations`` is reached.
        """

        # if not isinstance(X[0], GMMStats):
        # logger.info("Received non-GMM data. Will train it on the UBM.")
        # if self.ubm._means is None:  # Train a GMMMachine if not set
        #     logger.info("UBM not trained. Training it inside IVectorMachine.")
        #     self.ubm.fit(X)
        # __import__("ipdb").set_trace()
        # X = self.ubm.transform(X)  # Transform to GMMStats
        chunky = False
        if isinstance(X, da.Array):
            chunky = True

        self.dim_c = self.ubm.n_gaussians
        self.dim_d = self.ubm.means.shape[-1]

        self.T = np.random.normal(
            loc=0.0,
            scale=1.0,
            size=(self.dim_c, self.dim_d, self.dim_t),
        )
        self.sigma = copy.deepcopy(self.ubm.variances)

        for step in range(self.max_iterations):
            if chunky:
                stats = [
                    dask.delayed(e_step)(
                        machine=self,
                        data=xx,
                    )
                    for xx in X
                ]
                new_machine = dask.compute(dask.delayed(m_step)(stats, self))[0]
                for attr in ["T", "sigma"]:
                    setattr(self, attr, getattr(new_machine, attr))
            else:
                stats = [
                    e_step(
                        machine=self,
                        data=X,
                    )
                ]
                _ = m_step(stats, self)
            logger.debug(
                f"IVector step {step+1:{len(str(self.max_iterations))}d}/{self.max_iterations}."
            )
        logger.info(f"Reached {step+1} steps.")
        return self

    def project(self, stats: GMMStats) -> np.ndarray:
        """Projects the GMMStats on the IVectorMachine.

        This takes data already projected onto the UBM.

        **Returns:**

        The IVector of the input stats.

        """

        return np.linalg.solve(
            compute_id_tt_sigma_inv_t(stats, self.T, self.sigma),
            compute_tt_sigma_inv_fnorm(
                self.ubm.means, stats, self.T, self.sigma
            ),
        )

    def transform(self, X: List[GMMStats]) -> List[np.ndarray]:
        """Transforms the data using the trained IVectorMachine.

        This takes MFCC data, will project them onto the ubm, and compute the IVector
        statistics.

        **Parameters:**

        data
            The data (MFCC features) to transform.
            Arrays of shape (n_samples, n_features).

        **Returns:**

        The IVector for each sample. Arrays of shape (dim_t,)
        """

        return [self.project(x) for x in X]

    def _more_tags(self) -> Dict[str, Any]:
        return {
            "requires_fit": True,
            "bob_fit_supports_dask_array": True,
        }
