#!/usr/bin/env python
# @author: Yannick Dayer <yannick.dayer@idiap.ch>
# @date: Fri 06 May 2022 14:18:25 UTC+02

import numpy as np

from sklearn.base import BaseEstimator

from bob.learn.em import GMMMachine, GMMStats


class IVectorStats:
    """This class is a container for the statistics of an IVectorMachine."""

    def __init__(self) -> None:
        self.nij_sigma_wij2 = {}
        self.fnorm_sigma_wij = {}
        self.snorm = np.zeros(shape=(self.dim_c * self.dim_d,), dtype=float)
        self.n = np.zeros(shape=(self.dim_c,), dtype=float)


class IVectorMachine(BaseEstimator):
    """Trains and projects data using I-Vector."""

    def __init__(
        self,
        ubm: GMMMachine,
        dim_t: int = 2,
        convergence_threshold: float = 1e-5,
        max_iterations: int = 25,
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
        # TODO: add params
        # self.compute_likelihood = compute_likelihood
        # self.sigma_update = sigma_update
        # self.variance_floor = variance_floor

    def initialize(self, features: np.ndarray) -> None:
        """Initializes the I-Vector parameters at fit time."""
        # TODO implement
        # self.
        pass

    def e_step(self, data: np.ndarray) -> GMMStats:
        """Computes the expectation step of the e-m algorithm."""
        # n_samples = len(data)
        # self.m_acc_Nij_Sigma_wij2  = {}
        # self.m_acc_Fnorm_Sigma_wij = {}
        # self.m_acc_Snorm = numpy.zeros(shape=(self.m_dim_c*self.m_dim_d,), dtype=numpy.float64)
        # self.m_N = numpy.zeros(shape=(self.m_dim_c,), dtype=numpy.float64)

        # for c in range(self.m_dim_c):
        #   self.m_acc_Nij_Sigma_wij2[c]  = numpy.zeros(shape=(self.m_dim_t,self.m_dim_t), dtype=numpy.float64)
        #   self.m_acc_Fnorm_Sigma_wij[c] = numpy.zeros(shape=(self.m_dim_d,self.m_dim_t), dtype=numpy.float64)

        # for n in range(n_samples):
        #   Nij = data[n].n
        #   Fij = data[n].sum_px
        #   Sij = data[n].sum_pxx

        #   # Estimate latent variables
        #   TtSigmaInv_Fnorm = machine.__compute_TtSigmaInvFnorm__(data[n])
        #   I_TtSigmaInvNT = machine.__compute_Id_TtSigmaInvT__(data[n])

        #   Fnorm = numpy.zeros(shape=(self.m_dim_c*self.m_dim_d,), dtype=numpy.float64)
        #   Snorm = numpy.zeros(shape=(self.m_dim_c*self.m_dim_d,), dtype=numpy.float64)

        #   # Compute normalized statistics
        #   for c in range(self.m_dim_c):
        #     start            = c*self.m_dim_d
        #     end              = (c+1)*self.m_dim_d

        #     Fc               = Fij[c,:]
        #     Sc               = Sij[c,:]
        #     mc               = self.m_meansupervector[start:end]

        #     Fc_mc            = Fc * mc
        #     Nc_mc_mcT        = Nij[c] * mc * mc

        #     Fnorm[start:end] = Fc - Nij[c] * mc
        #     Snorm[start:end] = Sc - (2 * Fc_mc) + Nc_mc_mcT

        #   # Latent variables
        #   I_TtSigmaInvNT_inv = numpy.linalg.inv(I_TtSigmaInvNT)
        #   E_w_ij             = numpy.dot(I_TtSigmaInvNT_inv, TtSigmaInv_Fnorm)
        #   E_w_ij2            = I_TtSigmaInvNT_inv + numpy.outer(E_w_ij, E_w_ij)

        #   # Do the accumulation for each component
        #   self.m_acc_Snorm   = self.m_acc_Snorm + Snorm    # (dim_c*dim_d)
        #   for c in range(self.m_dim_c):
        #     start            = c*self.m_dim_d
        #     end              = (c+1)*self.m_dim_d
        #     current_Fnorm    = Fnorm[start:end]            # (dim_d)
        #     self.m_acc_Nij_Sigma_wij2[c]  = self.m_acc_Nij_Sigma_wij2[c] + Nij[c] * E_w_ij2                    # (dim_t, dim_t)
        #     self.m_acc_Fnorm_Sigma_wij[c] = self.m_acc_Fnorm_Sigma_wij[c] + numpy.outer(current_Fnorm, E_w_ij) # (dim_d, dim_t)
        #     self.m_N[c]                   = self.m_N[c] + Nij[c]

        # TODO convert
        pass

    def m_step(self, stats: GMMStats) -> None:
        """Updates the Machine with the maximization step of the e-m algorithm."""
        # A = self.m_acc_Nij_Sigma_wij2

        # T = numpy.zeros(shape=(self.m_dim_c*self.m_dim_d,self.m_dim_t), dtype=numpy.float64)
        # Told = machine.t
        # if self.m_sigma_update:
        #   sigma = numpy.zeros(shape=self.m_acc_Snorm.shape, dtype=numpy.float64)
        # for c in range(self.m_dim_c):
        #   start = c*self.m_dim_d;
        #   end   = (c+1)*self.m_dim_d;
        #   # T update
        #   A     = self.m_acc_Nij_Sigma_wij2[c].transpose()
        #   B     = self.m_acc_Fnorm_Sigma_wij[c].transpose()
        #   if numpy.array_equal(A, numpy.zeros(A.shape)):
        #     X = numpy.zeros(shape=(self.m_dim_t,self.m_dim_d), dtype=numpy.float64)
        #   else:
        #     X = numpy.linalg.solve(A, B)
        #   T[start:end,:] = X.transpose()
        #   # Sigma update
        #   if self.m_sigma_update:
        #     Told_c           = Told[start:end,:].transpose()
        #     # warning: Use of the new T estimate! (revert second next line if you don't want that)
        #     Fnorm_Ewij_Tt    = numpy.diag(numpy.dot(self.m_acc_Fnorm_Sigma_wij[c], X))
        #     #Fnorm_Ewij_Tt = numpy.diag(numpy.dot(self.m_acc_Fnorm_Sigma_wij[c], Told_c))
        #     sigma[start:end] = (self.m_acc_Snorm[start:end] - Fnorm_Ewij_Tt) / self.m_N[c]
        # TODO convert
        pass

    def fit(self, data: np.ndarray) -> "IVectorMachine":
        """Trains the IVectorMachine."""
        # TODO implement
        return self
