#!/usr/bin/env python
# @author: Yannick Dayer <yannick.dayer@idiap.ch>
# @date: Fri 30 Jul 2021 10:06:47 UTC+02

import logging
from abc import ABC
from abc import abstractmethod

import dask.array as da
import numpy as np
from sklearn.base import BaseEstimator

from bob.learn.em.cluster import KMeansMachine

logger = logging.getLogger(__name__)


class Gaussian:
    mean = None
    variance = None
    variance_threshold = None

class GMMMachine(BaseEstimator):
    """Stores a GMM mixtures parameters (Gaussians means and variances)

    Parameters
    ----------
    n_gaussians: int
        The number of gaussians to be represented by the machine.
    """

    def __init__(self, n_gaussians: int, convergence_threshold=1e-5):
        self.n_gaussians = n_gaussians
        self.log_weights = da.log(da.full((n_gaussians,), fill_value=1 / n_gaussians))
        self.convergence_threshold = convergence_threshold
        # Precomputed constants:
        self.N_LOG_2PI = self.n_dims * np.log(2 * np.pi)

    def log_likelihood(self, x: da.Array):
        """Returns the current log likelihood for a set of data in this Machine.

        Parameters
        ----------
        x: 2D array
            Data to compute the log likelihood on.
        """
        # Possibility to pre-compute g_norm (would need update on variance change)
        g_norm = self.N_LOG_2PI + np.sum(np.log(self.gaussians_variance))

        # Compute the likelihood for each data point on each gaussian
        z = da.sum(
            da.power(x[None, :, :] - self.gaussians_mean[:, None, :], 2)
            / self.gaussians_variance[:, None, :],
            axis=2,
        )
        l = -0.5 * (g_norm + z)
        weighted_l = self.log_weights[:, None] + l

        def logaddexp_reduce(a, axis, keepdims):
            return np.logaddexp.reduce(a, axis=axis, keepdims=keepdims, initial=-np.inf)

        # Sum along gaussians axis (using logAddExp to prevent underflow)
        ll_reduced = da.reduction(
            x=weighted_l,
            chunk=logaddexp_reduce,
            aggregate=logaddexp_reduce,
            axis=0,
            dtype=np.float,
            keepdims=False,
        )

        # Mean along data axis
        return da.mean(ll_reduced)

    def acc_statistics(self, x, stats):
        raise NotImplementedError

    def fit(self, X, y=None, trainer=None, **kwargs):
        if trainer is None:
            logger.info("Creating the default GMM trainer.")
            # TODO

        for step in range(self.max_steps):
            logger.info(f"Iteration = {step:3d}/{self.max_steps}")
            average_output_previous = average_output
            trainer.e_step(self, X)
            trainer.m_step(self, X)

            average_output = trainer.compute_likelihood(self.machine)
            logger.info(f"Likelihood = {average_output}")

            if step > 0:
                convergence_value = abs(
                    (average_output_previous - average_output) / average_output_previous
                )
                logger.info(f"convergence value = {convergence_value.compute()}")

                # Terminates if converged (and likelihood computation is set)
                if (
                    self.convergence_threshold is not None
                    and convergence_value <= self.convergence_threshold
                ):
                    return self
        return self

    def transform(self, X, **kwargs):
        raise NotImplementedError # what to return?
        return X


class BaseGMMTrainer(ABC):
    """Base class for the different GMM Trainer implementations.

    Parameters
    ----------
    init_method: str or KMeansMachine
    """

    def __init__(self, init_method="k-means"):
        self.stat_log_likelihood = 0.0  # The accumulated log likelihood of all samples
        self.stat_T = 0  # The accumulated number of samples
        self.stat_n = None  # For each Gaussian, the accumulated sum of responsibilities, i.e. the sum of P(gaussian_i|x)
        self.stat_sumPx = None  # For each Gaussian, the accumulated sum of responsibility times the sample
        self.stat_sumPxx = None  # For each Gaussian, the accumulated sum of responsibility times the sample squared

    @abstractmethod
    def initialize(self, machine: GMMMachine, data: da.Array):
        pass

    def e_step(self, machine: GMMMachine, data: da.Array):
        # The e-step is the same for each GMM Trainer
        raise NotImplementedError

    @abstractmethod
    def m_step(self, machine: GMMMachine, data: da.Array):
        # The m-step must be implemented by the specialized GMM Trainers
        pass

    def compute_likelihood(self, machine: GMMMachine):
        return self.stat_log_likelihood / self.stat_T


class MLGMMTrainer(BaseGMMTrainer):
    """Maximum Likelihood trainer for GMM"""

    def initialize(self, machine: GMMMachine, data: da.Array):
        self.stat_log_likelihood = 0.0
        self.stat_T = 0
        self.stat_n = da.zeros((machine.n_gaussians,))
        self.stat_sumPx = da.zeros((machine.n_gaussians, machine.n_dims))
        self.stat_sumPxx = da.zeros((machine.n_gaussians, machine.n_dims))
        raise NotImplementedError

    def m_step(self, machine: GMMMachine, data: da.Array):
        #   // - Update weights if requested
        #   //   Equation 9.26 of Bishop, "Pattern recognition and machine learning", 2006
        #   if (m_gmm_base_trainer.getUpdateWeights()) {
        #     blitz::Array<double,1>& weights = gmm.updateWeights();
        #     weights = m_gmm_base_trainer.getGMMStats()->n / static_cast<double>(m_gmm_base_trainer.getGMMStats()->T); //cast req. for linux/32-bits & osx
        #     // Recompute the log weights in the cache of the GMMMachine
        #     gmm.recomputeLogWeights();
        #   }

        #   // Generate a thresholded version of m_ss.n
        #   for(size_t i=0; i<n_gaussians; ++i)
        #     m_cache_ss_n_thresholded(i) = std::max(m_gmm_base_trainer.getGMMStats()->n(i), m_gmm_base_trainer.getMeanVarUpdateResponsibilitiesThreshold());

        #   // Update GMM parameters using the sufficient statistics (m_ss)
        #   // - Update means if requested
        #   //   Equation 9.24 of Bishop, "Pattern recognition and machine learning", 2006
        #   if (m_gmm_base_trainer.getUpdateMeans()) {
        #     for(size_t i=0; i<n_gaussians; ++i) {
        #       blitz::Array<double,1>& means = gmm.getGaussian(i)->updateMean();
        #       means = m_gmm_base_trainer.getGMMStats()->sumPx(i, blitz::Range::all()) / m_cache_ss_n_thresholded(i);
        #     }
        #   }

        #   // - Update variance if requested
        #   //   See Equation 9.25 of Bishop, "Pattern recognition and machine learning", 2006
        #   //   ...but we use the "computational formula for the variance", i.e.
        #   //   var = 1/n * sum (P(x-mean)(x-mean))
        #   //       = 1/n * sum (Pxx) - mean^2
        #   if (m_gmm_base_trainer.getUpdateVariances()) {
        #     for(size_t i=0; i<n_gaussians; ++i) {
        #       const blitz::Array<double,1>& means = gmm.getGaussian(i)->getMean();
        #       blitz::Array<double,1>& variances = gmm.getGaussian(i)->updateVariance();
        #       variances = m_gmm_base_trainer.getGMMStats()->sumPxx(i, blitz::Range::all()) / m_cache_ss_n_thresholded(i) - blitz::pow2(means);
        #       gmm.getGaussian(i)->applyVarianceThresholds();
        #     }
        #   }
        raise NotImplementedError


class MAPGMMTrainer(BaseGMMTrainer):
    """Maximum A Posteriori trainer for GMM"""

    def initialize(self, machine: GMMMachine, data: da.Array):
        self.stat_log_likelihood = 0.0
        self.stat_T = 0
        self.stat_n = da.zeros((machine.n_gaussians,))
        self.stat_sumPx = da.zeros((machine.n_gaussians, machine.n_dims))
        self.stat_sumPxx = da.zeros((machine.n_gaussians, machine.n_dims))
        raise NotImplementedError

    def m_step(self, machine: GMMMachine, data: da.Array):
        raise NotImplementedError
