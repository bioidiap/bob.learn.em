#!/usr/bin/env python
# @author: Yannick Dayer <yannick.dayer@idiap.ch>
# @date: Fri 06 May 2022 14:18:25 UTC+02

import logging

from typing import List, Tuple

import numpy as np

from sklearn.base import BaseEstimator

from bob.learn.em import GMMMachine, GMMStats

logger = logging.getLogger("__name__")


class IVectorStats:
    def __init__(self, dim_c, dim_d, dim_t):
        self.dim_c = dim_c
        self.dim_d = dim_d
        self.dim_t = dim_t

        # Accumulator storage variables

        # nij sigma wij2: shape = (c,t,t)
        self.acc_nij_sigma_wij2 = np.zeros(
            shape=(self.dim_c, self.dim_t, self.dim_t), dtype=float
        )
        # fnorm sigma wij: shape = (c,d,t)
        self.acc_fnorm_sigma_wij = np.zeros(
            shape=(self.dim_c, self.dim_d, self.dim_t), dtype=float
        )
        # Snormij (used only when updating sigma)
        self.acc_snormij = np.zeros(
            shape=(
                self.dim_c,
                self.dim_d,
            ),
            dtype=float,
        )
        # Nij (used only when updating sigma)
        self.acc_nij = np.zeros(shape=(self.dim_c,), dtype=float)

    @property
    def shape(self) -> Tuple[int, int, int]:
        return (self.dim_c, self.dim_d, self.dim_t)

    def __add__(self, other):
        if self.shape != other.shape:
            raise ValueError("Cannot add stats of different shapes")
        result = IVectorStats(self.dim_c, self.dim_d, self.dim_t)
        result.acc_nij_sigma_wij2 = (
            self.acc_nij_sigma_wij2 + other.acc_nij_sigma_wij2
        )
        result.acc_fnorm_sigma_wij = (
            self.acc_fnorm_sigma_wij + other.acc_fnorm_sigma_wij
        )
        result.acc_snormij = self.acc_snormij + other.acc_snormij
        result.acc_nij = self.acc_nij + other.acc_nij
        return result

    def __iadd__(self, other):
        if self.shape != other.shape:
            raise ValueError("Cannot add stats of different shapes")
        self.acc_nij_sigma_wij2 += other.acc_nij_sigma_wij2
        self.acc_fnorm_sigma_wij += other.acc_fnorm_sigma_wij
        self.acc_snormij += other.acc_snormij
        self.acc_nij += other.acc_nij
        return self


def compute_tct_sigmac_inv(T: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """Computes T_{c}^{T}.sigma_{c}^{-1}"""
    # C++ to port to python:
    #
    # for (int c=0; c<C; ++c)
    # {
    # blitz::Array<double,2> Tct_sigmacInv = m_cache_Tct_sigmacInv(c, rall, rall);
    # blitz::Array<double,2> Tc = m_T(blitz::Range(c*D,(c+1)*D-1), rall);
    # blitz::Array<double,2> Tct = Tc.transpose(1,0);
    # blitz::Array<double,1> sigma_c = m_sigma(blitz::Range(c*D,(c+1)*D-1));
    # Tct_sigmacInv = Tct(i,j) / sigma_c(j);
    # }

    # Python version:
    # Tct_sigmacInv = np.zeros(shape=(c,d,t))
    # for c in range(dim_c):
    #     Tc = T[c]
    #     Tct = Tc.transpose(1, 0)
    #     sigma_c = sigma[c]
    #     Tct_sigmacInv[c] = T[c].t / sigma[c]

    # Vectorized version:

    # T.T (c,t,d) / sigma (c,1,d)
    Tct_sigmacInv = T.transpose(0, 2, 1) / sigma[:, None, :]

    # Tt_sigma_inv (c,t,d)
    return Tct_sigmacInv


def compute_tct_sigmac_inv_tc(T: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """Computes T_{c}^{T}.sigma_{c}^{-1}.T_{c}"""

    # for (int c=0; c<C; ++c)
    # {
    # blitz::Array<double,2> Tc = m_T(blitz::Range(c*D,(c+1)*D-1), rall);
    # blitz::Array<double,2> Tct_sigmacInv = m_cache_Tct_sigmacInv(c, rall, rall);
    # blitz::Array<double,2> Tct_sigmacInv_Tc = m_cache_Tct_sigmacInv_Tc(c, rall, rall);
    # bob::math::prod(Tct_sigmacInv, Tc, Tct_sigmacInv_Tc);
    # }

    Tct_sigmacInv_Tc = np.zeros(shape=(T.shape[0], T.shape[-1], T.shape[-1]))

    # Python version:
    tct_sigmac_inv = compute_tct_sigmac_inv(T, sigma)
    for c in range(T.shape[0]):  # TODO Vectorize
        # (c,t,t) = (c,t,d) @ (c,d,t)
        Tct_sigmacInv_Tc[c] = tct_sigmac_inv[c] @ T[c]

    # Output: shape (c,t,t)
    return Tct_sigmacInv_Tc


def compute_id_tt_sigma_inv_t(
    stats: GMMStats, T: np.ndarray, sigma: np.ndarray
) -> np.ndarray:
    # void bob::learn::em::IVectorMachine::computeIdTtSigmaInvT(
    #   const bob::learn::em::GMMStats& gs, blitz::Array<double,2>& output) const
    # {
    #   blitz::Range rall = blitz::Range::all();
    #   bob::math::eye(output);
    #   for (int c=0; c<(int)getNGaussians(); ++c)
    #     output += gs.n(c) * m_cache_Tct_sigmacInv_Tc(c, rall, rall);
    # }
    dim_t = T.shape[-1]
    output = np.eye(dim_t, dim_t)
    tct_sigmac_inv_tc = compute_tct_sigmac_inv_tc(T, sigma)
    for c in range(stats.n.shape[0]):  # TODO Vectorize
        # (t,t) += scalar * (c,t,t)
        output += stats.n[c] * tct_sigmac_inv_tc[c]

    # Output: (t,t)
    return output


def compute_tt_sigma_inv_fnorm(
    ubm_means: np.ndarray, stats: GMMStats, T: np.ndarray, sigma: np.ndarray
) -> np.ndarray:
    """Computes \f$(Id + \\sum_{c=1}^{C} N_{i,j,c} T^{T} \\Sigma_{c}^{-1} T)\f$

    Returns an array of shape (t,)
    """

    # void bob::learn::em::IVectorMachine::computeTtSigmaInvFnorm(
    #   const bob::learn::em::GMMStats& gs, blitz::Array<double,1>& output) const
    # {
    #   // Computes \f$T^{T} \Sigma^{-1} \sum_{c=1}^{C} (F_c - N_c ubmmean_{c})\f$
    #   blitz::Range rall = blitz::Range::all();
    #   output = 0;
    #   for (int c=0; c<(int)getNGaussians(); ++c)
    #   {
    #     m_tmp_d = gs.sumPx(c,rall) - gs.n(c) * m_ubm->getGaussian(c)->getMean();
    #     blitz::Array<double,2> Tct_sigmacInv = m_cache_Tct_sigmacInv(c, rall, rall);
    #     bob::math::prod(Tct_sigmacInv, m_tmp_d, m_tmp_t2);

    #     output += m_tmp_t2;
    #   }
    # }

    output = np.zeros(shape=T.shape[-1])
    tct_sigmac_inv = compute_tct_sigmac_inv(T, sigma)  # (c,t,d)
    fnorm = stats.sum_px - stats.n[:, None] * ubm_means  # (c,d)
    for c in range(T.shape[0]):  # TODO Vectorize
        # (t,) += (c,t,d) @ (c,d)
        output += tct_sigmac_inv[c] @ fnorm[c]

    # Output: shape (t,)
    return output


def forward(
    ubm_means: np.ndarray, stats: GMMStats, T: np.ndarray, sigma: np.ndarray
):
    # void bob::learn::em::IVectorMachine::forward_(const bob::learn::em::GMMStats& gs,
    #   blitz::Array<double,1>& ivector) const
    # {
    #   // Computes \f$(Id + \sum_{c=1}^{C} N_{i,j,c} T^{T} \Sigma_{c}^{-1} T)\f$
    #   computeIdTtSigmaInvT(gs, m_tmp_tt);

    #   // Computes \f$T^{T} \Sigma^{-1} \sum_{c=1}^{C} (F_c - N_c ubmmean_{c})\f$
    #   computeTtSigmaInvFnorm(gs, m_tmp_t1);

    #   // Solves m_tmp_tt.ivector = m_tmp_t1
    #   bob::math::linsolve(m_tmp_tt, m_tmp_t1, ivector);
    # }
    return np.linalg.solve(
        compute_id_tt_sigma_inv_t(stats, T, sigma),
        compute_tt_sigma_inv_fnorm(ubm_means, stats, T, sigma),
    )


class IVectorMachine(BaseEstimator):
    """Trains and projects data using I-Vector.

    Dimensions:
        - dim_c: number of Gaussians
        - dim_d: number of features
        - dim_t: dimension of the i-vector
    Attributes:
        T (c,d,t): The total variability matrix \f$T\f$
        sigma (c,d): The diagonal covariance matrix \f$\\Sigma\f$
    """

    def __init__(
        self,
        ubm: GMMMachine,
        dim_t: int = 2,
        convergence_threshold: float = 1e-5,
        max_iterations: int = 25,
        update_sigma: bool = True,
        **kwargs,
    ) -> None:
        """Initializes the IVectorMachine object.

        Parameters
        ----------
        ubm:
            The Universal Background Model.
        dim_t:
            The dimension of the i-vector.
        """

        super().__init__(**kwargs)
        self.ubm = ubm
        self.dim_t = dim_t
        self.convergence_threshold = convergence_threshold
        self.max_iterations = max_iterations
        self.update_sigma = update_sigma
        # TODO: add params
        # self.compute_likelihood = compute_likelihood
        # self.variance_floor = variance_floor
        self.dim_c = self.ubm.n_gaussians
        self.dim_d = self.ubm.means.shape[-1]

        self.T = np.zeros(shape=(self.dim_c, self.dim_d, self.dim_t))
        self.sigma = np.zeros(shape=(self.dim_c, self.dim_d))

    def e_step(self, data: List[GMMStats]) -> IVectorStats:
        """Computes the expectation step of the e-m algorithm."""
        n_samples = len(data)

        stats = IVectorStats(self.dim_c, self.dim_d, self.dim_t)
        ubm_means = self.ubm.means

        for n in range(n_samples):
            Nij = data[n].n
            Fij = data[n].sum_px
            Sij = data[n].sum_pxx

            # Estimate latent variables
            TtSigmaInv_Fnorm = compute_tt_sigma_inv_fnorm(
                ubm_means, data[n], self.T, self.sigma
            )  # self.compute_TtSigmaInvFnorm(data[n]) # shape: (t,)
            I_TtSigmaInvNT = compute_id_tt_sigma_inv_t(
                data[n], self.T, self.sigma
            )  # self.compute_Id_TtSigmaInvT(data[n]), # shape: (t,t)

            Fnorm = np.zeros(
                shape=(
                    self.dim_c,
                    self.dim_d,
                ),
                dtype=float,
            )
            Snorm = np.zeros(
                shape=(
                    self.dim_c,
                    self.dim_d,
                ),
                dtype=float,
            )

            # Latent variables
            I_TtSigmaInvNT_inv = np.linalg.inv(I_TtSigmaInvNT)  # shape: (t,t)
            sigma_w_ij = np.dot(
                I_TtSigmaInvNT_inv, TtSigmaInv_Fnorm
            )  # shape: (t,)
            sigma_w_ij2 = I_TtSigmaInvNT_inv + np.outer(
                sigma_w_ij, sigma_w_ij
            )  # shape: (t,t)

            # Compute normalized statistics
            # for c in range(self.dim_c):  # TODO Vectorize

            #     Fc = Fij[c, :]
            #     Sc = Sij[c, :]
            #     mc = self.ubm.means[c]

            #     Fc_mc = Fc * mc
            #     Nc_mc_mcT = Nij[c] * mc * mc

            #     Fnorm[c] = Fc - Nij[c] * mc
            #     Snorm[c] = Sc - (2 * Fc_mc) + Nc_mc_mcT

            Fnorm = Fij - Nij[:, None] * self.ubm.means
            Snorm = (
                Sij
                - (2 * Fij * self.ubm.means)
                + (Nij[:, None] * self.ubm.means * self.ubm.means)
            )

            # Do the accumulation for each component
            stats.acc_snormij += Snorm  # (dim_c, dim_d)

            for c in range(self.dim_c):
                stats.acc_nij_sigma_wij2 += (
                    Nij[c] * sigma_w_ij2
                )  # (dim_t, dim_t)
            # stats.acc_nij_sigma_wij2 += Nij[:, None] * sigma_w_ij2  # (c, t, t) # TODO Not working
            stats.acc_nij += Nij
            # for c in range(self.dim_c):  # TODO Vectorize
            #     stats.acc_fnorm_sigma_wij[c] += np.outer(
            #         Fnorm[c], sigma_w_ij  # (c,d) x (t,)
            #     )  # (dim_d, dim_t)
            stats.acc_fnorm_sigma_wij += np.matmul(
                Fnorm[:, :, None], sigma_w_ij[None, :]
            )  # (c,d,t)

        return stats

    def m_step(self, stats: IVectorStats) -> None:
        """Updates the Machine with the maximization step of the e-m algorithm."""
        A = stats.acc_nij_sigma_wij2

        self.T = np.zeros(
            shape=(self.dim_c, self.dim_d, self.dim_t),
            dtype=np.float64,
        )
        if self.update_sigma:
            self.sigma = np.zeros(
                shape=stats.acc_snormij.shape, dtype=np.float64
            )
        for c in range(self.dim_c):  # TODO Vectorize
            # T update
            A = stats.acc_nij_sigma_wij2[c].transpose()
            B = stats.acc_fnorm_sigma_wij[c].transpose()
            if not A.any():  # if all A == 0
                X = np.zeros(shape=(self.dim_t, self.dim_d), dtype=np.float64)
            else:
                X = np.linalg.solve(A, B)
            self.T[c, :] = X.transpose()
            # Sigma update
            if self.update_sigma:
                # t_old_c = t_old[c, :].transpose()
                # warning: Use of the new T estimate! (toggle the two next line if you don't want that)
                Fnorm_sigma_w_ij_Tt = np.diag(
                    np.dot(stats.acc_fnorm_sigma_wij[c], X)
                )
                # Fnorm_Ewij_Tt = np.diag(np.dot(stats.fnorm_sigma_wij[c], t_old_c))
                self.sigma[c] = (
                    stats.acc_snormij[c] - Fnorm_sigma_w_ij_Tt
                ) / stats.acc_nij[c]

    def fit(self, data: np.ndarray) -> "IVectorMachine":
        """Trains the IVectorMachine.

        Repeats the e-m steps until the convergence criterion is met or
        ``max_iterations`` is reached.
        """

        t_old = 0
        for step in range(self.max_iterations):
            logger.debug(f"IVector step {step+1}.")
            # E-step
            stats = self.e_step(data)
            # M-step
            self.m_step(stats)
            # Convergence
            if step > 0:
                if np.linalg.norm(stats.t - t_old) < self.convergence_threshold:
                    logger.info(f"Converged after {step+1} steps.")
                    logger.debug(
                        f"t_diff: {stats.t - t_old} (Convergence threshold: {self.convergence_threshold})"
                    )
                    break
            t_old = stats.t
        else:
            logger.info(f"Did not converge after {step+1} steps.")
        return self

    def project(self, stats: GMMStats) -> IVectorStats:
        return forward(self.ubm.means, stats, self.T, self.sigma)
