#!/usr/bin/env python
# @author: Yannick Dayer <yannick.dayer@idiap.ch>
# @date: Fri 06 May 2022 14:18:25 UTC+02

from typing import List

import numpy as np

from sklearn.base import BaseEstimator

from bob.learn.em import GMMMachine, GMMStats


class IVectorStats:
    """This class is a container for the statistics of an IVectorMachine."""

    def __init__(self, dim_c, dim_d, dim_t) -> None:
        self.dim_c = dim_c
        self.dim_d = dim_d
        self.dim_t = dim_t
        # nij sigma wij2: shape = (c,t,t)
        self.nij_sigma_wij2 = np.zeros(
            shape=(self.dim_c, self.dim_t, self.dim_t), dtype=float
        )
        # fnorm sigma wij: shape = (c,d,t)
        self.fnorm_sigma_wij = np.zeros(
            shape=(self.dim_c, self.dim_d, self.dim_t), dtype=float
        )
        self.snorm = np.zeros(
            shape=(
                self.dim_c,
                self.dim_d,
            ),
            dtype=float,
        )
        self.n = np.zeros(shape=(self.dim_c,), dtype=float)


def compute_tct_sigmac_inv(T, sigma):
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
    # python version:
    # for c in range(dim_c):
    #     Tc = T[c]
    #     Tct = Tc.transpose(1, 0)
    #     sigma_c = sigma[c]
    #     Tct_sigmacInv = T[c].t / sigma[c]

    # Vectorized version:

    Tct_sigmacInv = T.transpose(0, 2, 1) / sigma[None]
    return Tct_sigmacInv


def compute_tct_sigmac_inv_tc(self, stats: GMMStats):
    """Computes T_{c}^{T}.sigma_{c}^{-1}.T_{c}"""

    # for (int c=0; c<C; ++c)
    # {
    # blitz::Array<double,2> Tc = m_T(blitz::Range(c*D,(c+1)*D-1), rall);
    # blitz::Array<double,2> Tct_sigmacInv = m_cache_Tct_sigmacInv(c, rall, rall);
    # blitz::Array<double,2> Tct_sigmacInv_Tc = m_cache_Tct_sigmacInv_Tc(c, rall, rall);
    # bob::math::prod(Tct_sigmacInv, Tc, Tct_sigmacInv_Tc);
    # }
    # python version:

    pass


def compute_tt_sigma_inv_fnorm(self, stats: GMMStats) -> np.ndarray:
    """Computes \f$(Id + \\sum_{c=1}^{C} N_{i,j,c} T^{T} \\Sigma_{c}^{-1} T)\f$

    Returns an array of shape (?,?) TODO dims!
    """
    # void bob::learn::em::IVectorMachine::computeIdTtSigmaInvT(
    #   const bob::learn::em::GMMStats& gs, blitz::Array<double,2>& output) const
    # {
    #   blitz::Range rall = blitz::Range::all();
    #   bob::math::eye(output);
    #   for (int c=0; c<(int)getNGaussians(); ++c)
    #     output += gs.n(c) * m_cache_Tct_sigmacInv_Tc(c, rall, rall);
    # }
    # output = np.eye((self.dim_t, self.dim_t))  # TODO dims!
    # for c in range(self.dim_c):
    #     output += stats.n[c] * Tct_sigmacInv_Tc[c]
    pass
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


class IVectorMachine(BaseEstimator):
    """Trains and projects data using I-Vector.

    Dimensions:
        - dim_c: number of Gaussians
        - dim_d: numer of features
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
        **kwargs
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
        # self.sigma_update = sigma_update
        # self.variance_floor = variance_floor
        self.dim_d = self.ubm.n_gaussians
        self.dim_c = self.ubm.means.shape[-1]

    def initialize(self, features: np.ndarray) -> None:
        """Initializes the I-Vector parameters at fit time."""
        # TODO implement
        # self.
        # self.dim_d =
        # self.dim_c =
        pass

    def e_step(self, data: List[GMMStats]) -> IVectorStats:
        """Computes the expectation step of the e-m algorithm."""
        n_samples = len(data)

        stats = IVectorStats(self.dim_c, self.dim_d, self.dim_t)

        for n in range(n_samples):
            Nij = data[n].n
            Fij = data[n].sum_px
            Sij = data[n].sum_pxx

            # Estimate latent variables
            TtSigmaInv_Fnorm = self.compute_TtSigmaInvFnorm(data[n])
            I_TtSigmaInvNT = self.compute_Id_TtSigmaInvT(data[n])

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
            I_TtSigmaInvNT_inv = np.linalg.inv(I_TtSigmaInvNT)
            E_w_ij = np.dot(I_TtSigmaInvNT_inv, TtSigmaInv_Fnorm)
            E_w_ij2 = I_TtSigmaInvNT_inv + np.outer(E_w_ij, E_w_ij)

            # Compute normalized statistics
            for c in range(self.dim_c):

                Fc = Fij[c, :]
                Sc = Sij[c, :]
                mc = self.means[c]

                Fc_mc = Fc * mc
                Nc_mc_mcT = Nij[c] * mc * mc

                Fnorm[c] = Fc - Nij[c] * mc
                Snorm[c] = Sc - (2 * Fc_mc) + Nc_mc_mcT

            # Do the accumulation for each component
            stats.snorm += Snorm  # (dim_c, dim_d)

            for c in range(self.dim_c):
                stats.nij_sigma_wij2[c] += Nij[c] * E_w_ij2  # (dim_t, dim_t)
                stats.fnorm_sigma_wij[c] += np.outer(
                    Fnorm[c], E_w_ij
                )  # (dim_d, dim_t)
                stats.n[c] += Nij[c]

        # TODO convert
        pass

    def m_step(self, stats: IVectorStats) -> None:
        """Updates the Machine with the maximization step of the e-m algorithm."""
        A = stats.nij_sigma_wij2

        T = np.zeros(
            shape=(self.dim_c, self.dim_d, self.dim_t),
            dtype=np.float64,
        )
        t_old = stats.t
        if self.update_sigma:
            sigma = np.zeros(shape=stats.snorm.shape, dtype=np.float64)
        for c in range(self.dim_c):
            # T update
            A = stats.nij_sigma_wij2[c].transpose()
            B = stats.fnorm_sigma_wij[c].transpose()
            if not A.any():  # if all A == 0
                X = np.zeros(shape=(self.dim_t, self.dim_d), dtype=np.float64)
            else:
                X = np.linalg.solve(A, B)
            T[c, :] = X.transpose()
            # Sigma update
            if self.update_sigma:
                _ = t_old  # TODO: remove
                # t_old_c = t_old[c, :].transpose()
                # warning: Use of the new T estimate! (toggle the two next line if you don't want that)
                Fnorm_Ewij_Tt = np.diag(np.dot(stats.fnorm_sigma_wij[c], X))
                # Fnorm_Ewij_Tt = np.diag(np.dot(stats.fnorm_sigma_wij[c], t_old_c))
                sigma[c] = (stats.snorm[c] - Fnorm_Ewij_Tt) / stats.n[c]
        pass

    def fit(self, data: np.ndarray) -> "IVectorMachine":
        """Trains the IVectorMachine."""
        # TODO implement
        return self
