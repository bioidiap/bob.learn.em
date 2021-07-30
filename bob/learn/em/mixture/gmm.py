#!/usr/bin/env python
# @author: Yannick Dayer <yannick.dayer@idiap.ch>
# @date: Fri 30 Jul 2021 10:06:47 UTC+02

import logging
from abc import ABC
from abc import abstractmethod

import math

import dask.array as da
from sklearn.base import BaseEstimator

logger = logging.getLogger(__name__)


class GMMMachine:
    """Stores a GMM mixtures parameters (Gaussians means and variances)

    Parameters
    ----------
    n_gaussians: int
        The number of gaussians to be represented by the machine.
    n_dims: int
        The dimensionality of the data.
    """

    def __init__(self, n_gaussians: int, n_dims: int):
        self.n_gaussians = n_gaussians
        self.n_dims = n_dims
        self.gaussians_mean = da.zeros((n_gaussians, n_dims))
        self.gaussians_variance = da.zeros((n_gaussians, n_dims))
        self.gaussians_variance_threshold = da.zeros((n_gaussians, n_dims))
        self.logweights = da.log(da.full((n_gaussians,), fill_value=1 / n_gaussians))
        # Precomputed constants:
        self.N_LOG_2PI = self.n_dims * math.log(2*math.pi)

    def log_likelihood(self, x: da.Array):

#   C++: x is one sample, then output is averaged when x is 2D

#   // Initialize variables
#   double log_likelihood = bob::math::Log::LogZero;

#   // Accumulate the weighted log likelihoods from each Gaussian
#   for(size_t i=0; i<m_n_gaussians; ++i) {
#     double l = m_cache_log_weights(i) + m_gaussians[i]->logLikelihood_(x);
#     log_weighted_gaussian_likelihoods(i) = l;
#     log_likelihood = bob::math::Log::logAdd(log_likelihood, l);
#   }

#   // Return log(p(x|GMMMachine))
#   return log_likelihood;
        log_likelihood = -math.inf

        # for each mean: compute the loglikelihood of x, add the log_weight
# logLikelihood_(x):
#   double z = blitz::sum(blitz::pow2(x - m_mean) / m_variance);
#   // Log Likelihood
#   return (-0.5 * (m_g_norm + z));
# m_variance: The diagonal of the covariance matrix (assumed to be diagonal)
# Pre-computed on variance change: m_g_norm = m_n_log2pi + blitz::sum(blitz::log(m_variance));
# Pre-computed at start: m_n_log2pi = m_n_inputs * bob::math::Log::Log2Pi; --> n_dims * ln(2*PI)
        raise NotImplementedError

    def acc_statistics(self, x, stats):
        raise NotImplementedError


class BaseGMMTrainer(ABC):
    """Base class for the different GMM Trainer implementations."""

    def __init__(self):
        self.stat_log_likelihood = 0.0  # The accumulated log likelihood of all samples
        self.stat_T = 0  # The accumulated number of samples
        self.stat_n = None  # For each Gaussian, the accumulated sum of responsibilities, i.e. the sum of P(gaussian_i|x)
        self.stat_sumPx = None  # For each Gaussian, the accumulated sum of responsibility times the sample
        self.stat_sumPxx = None  # For each Gaussian, the accumulated sum of responsibility times the sample squared

    @abstractmethod
    def initialize(self, machine: GMMMachine, data: da.Array):
        pass

    def e_step(self, machine: GMMMachine, data: da.Array):
        pass

    @abstractmethod
    def m_step(self, machine: GMMMachine, data: da.Array):
        pass

    def compute_likelihood(self, machine: GMMMachine):
        raise self.stat_log_likelihood / self.stat_T


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


class GMM(BaseEstimator):
    def __init__(
        self,
        n_gaussians: int,
        n_dims: int,
        max_steps: int = 200,
        convergence_threshold=1e-5,
        ml_trainer: bool = True,
    ):
        self.n_gaussians = n_gaussians
        self.n_dims = n_dims
        self.max_steps = max_steps
        self.convergence_threshold = convergence_threshold
        self.machine = None
        self.trainer = MLGMMTrainer() if ml_trainer else MAPGMMTrainer()

    def fit(self, X, y=None, **kwargs):
        if self.machine is None:
            logger.info("Creating the GMM machine.")
            self.machine = GMMMachine(n_gaussians=self.n_gaussians, n_dims=self.n_dims)
            logger.debug("Initializing Trainer.")
            self.trainer.initialize(
                machine=self.machine,
                data=X,
            )

        self.trainer.e_step(self.machine, X)
        for step in range(self.max_steps):
            logger.info(f"Iteration = {step:3d}/{self.max_steps}")
            average_output_previous = average_output
            self.trainer.m_step(self.machine, X)
            self.trainer.e_step(self.machine, X)

            average_output = self.trainer.compute_likelihood(self.machine)
            logger.info(f"Likelihood = {average_output}")

            convergence_value = abs(
                (average_output_previous - average_output) / average_output_previous
            )
            logger.info(f"convergence value = {convergence_value}")

            # Terminates if converged (and likelihood computation is set)
            if (
                self.convergence_threshold is not None
                and convergence_value <= self.convergence_threshold
            ):
                break
        return self

    def transform(self, X, **kwargs):
        pass
