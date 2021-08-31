#!/usr/bin/env python
# @author: Yannick Dayer <yannick.dayer@idiap.ch>
# @date: Fri 30 Jul 2021 10:06:47 UTC+02

import logging
from abc import ABC
from abc import abstractmethod
from typing import Union


import dask.array as da
import numpy as np
from sklearn.base import BaseEstimator

from bob.learn.em.cluster import KMeansMachine
from bob.learn.em.cluster import KMeansTrainer

logger = logging.getLogger(__name__)


class Gaussian(np.ndarray):
    """Represents a multi-dimensional Gaussian.

    This is basically three 1D-arrays: for the mean, the (diagonal) variance, and the
    variance threshold values.

    Each array can be accessed with the `[]` operator (`my_gaussian["variance"]`).

    When using multiple Gaussian in an array, the attributes can be accessed in the
    same way:
    >>> multi_gaussian = np.array([Gaussian([0,0]),Gaussian([1,1])])
    >>> print(multi_gaussian["mean"])
    ... [[0. 0.]
    ...  [1. 1.]]

    Parameters
    ----------
    mean: array of shape (n_features,)
        Center of the Gaussian distribution.
    variance: array of shape (n_features,)
        Diagonal variance matrix of the Gaussian distribution. Defaults to 1.
    variance_threshold: array of shape (n_features,)
        Threshold values. Defaults to 1e-5.
    """

    def __new__(cls, mean, variance=None, variance_threshold=None):
        n_features = len(mean)
        if variance is None:
            variance = np.ones_like(mean, dtype=float)
        if variance_threshold is None:
            variance_threshold = np.full_like(mean, fill_value=1e-5, dtype=float)
        rec = np.ndarray(
            shape=(n_features,),
            dtype=[("mean", float), ("variance", float), ("variance_threshold", float)],
        )
        rec["mean"] = mean
        rec["variance"] = variance
        rec["variance_threshold"] = variance_threshold
        return rec.view(cls)

    def log_likelihood(self, x):
        """Returns the log-likelihood for x on this gaussian.

        Parameters
        ----------
        x: array of shape (n_features,) or (n_samples, n_features)
            The point or points to compute the log-likelihood of.

        Returns
        -------
        float or array of shape (n_samples,)
            The log likelihood of each points in x.
        """
        # Precomputable constants if n_dims is known:
        N_LOG_2PI = x.shape[-1] * np.log(2 * np.pi)
        # Possibility to pre-compute g_norm (would need update on variance change)
        g_norm = N_LOG_2PI + np.sum(np.log(self["variance"]))

        # Compute the likelihood for each data point this Gaussian
        z = da.sum(da.power(x - self["mean"], 2) / self["variance"], axis=-1)
        return -0.5 * (g_norm + z)


class GMMMachine(BaseEstimator):
    """Stores a GMM parameters.

    Each mixture is a Gaussian represented by a mean and a diagonal variance matrix.

    A Trainer is needed to fit the data. The trainer initializes the machine's
    Gaussians, and iteratively trains with an e-m algorithm.
    If no Trainer is given, a default is used, initializing the Gaussians with k-means
    and training on the data with the maximum likelihood algorithm.

    Parameters
    ----------
    n_gaussians
        The number of gaussians to be represented by the machine.
    convergence_threshold
        The threshold value of likelihood difference between e-m steps used for
        stopping the training iterations.
    random_state
        Specifies a RandomState or a seed for reproducibility.
    weights
        The weight of each Gaussian.

    """

    def __init__(
        self,
        n_gaussians: int,
        convergence_threshold=1e-5,
        random_state: Union[int, da.random.RandomState] = 0,
        weights: Union[da.Array, None] = None,
    ):
        self.n_gaussians = n_gaussians
        self.convergence_threshold = convergence_threshold
        self.random_state = random_state
        if weights is None:
            weights = da.full(shape=(n_gaussians,), fill_value=(1.0 / n_gaussians))
        self.weights = weights

    @property
    def weights(self):
        return self.__weights

    @weights.setter
    def weights(self, w: da.Array):
        self.__weights = w
        self.__log_weights = da.log(self.__weights)

    @property
    def log_weights(self):
        return self.__log_weights

    def log_weighted_likelihood(self, x: da.Array):
        """Returns the weighted log likelihood for each Gaussian for a set of data.

        Parameters
        ----------
        x: array of shape (n_samples, n_features)
            Data to compute the log likelihood on.

        Returns
        -------
        array of shape (n_gaussians, n_samples)
            The weighted log likelihood of each sample of each Gaussian.
        """
        # Precomputable constants if n_dims is known:
        N_LOG_2PI = x.shape[-1] * np.log(2 * np.pi)
        # Possibility to pre-compute g_norm (would need update on variance change)
        g_norm = N_LOG_2PI + np.sum(np.log(self.gaussians_["variance"]))

        # Compute the likelihood for each data point on this Gaussian
        z = da.sum(
            da.power(x[None, ...] - self.gaussians_["mean"][:, None, :], 2)
            / self.gaussians_["variance"][:, None, :],
            axis=-1,
        )
        l = -0.5 * (g_norm + z)
        log_weighted_likelihood = self.log_weights[:, None] + l
        return log_weighted_likelihood

    def log_likelihood(self, x: da.Array):
        """Returns the current log likelihood for a set of data in this Machine.

        Parameters
        ----------
        x: array of shape (n_samples, n_features)
            Data to compute the log likelihood on.

        Returns
        -------
        array of shape (n_samples)
            The log likelihood of each sample.
        """
        log_weighted_likelihood = self.log_weighted_likelihood(x)

        def logaddexp_reduce(a, axis, keepdims):
            return np.logaddexp.reduce(a, axis=axis, keepdims=keepdims, initial=-np.inf)

        # Sum along gaussians axis (using logAddExp to prevent underflow)
        ll_reduced = da.reduction(
            x=log_weighted_likelihood,
            chunk=logaddexp_reduce,
            aggregate=logaddexp_reduce,
            axis=0,
            dtype=np.float,
            keepdims=False,
        )

        return ll_reduced

    def fit(self, X, y=None, trainer=None, max_steps: int = 200, **kwargs):
        if trainer is None:
            logger.info("Creating a default GMM trainer (ML GMM).")
            trainer = MLGMMTrainer(
                init_method="k-means", random_state=self.random_state
            )

        trainer.initialize(self, X)
        average_output = 0
        for step in range(max_steps):
            logger.info(f"Iteration = {step:3d}/{max_steps}")
            average_output_previous = average_output
            trainer.e_step(self, X)
            trainer.m_step(self, X)

            average_output = trainer.compute_likelihood(self)
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
        raise NotImplementedError  # what to return? The closest gaussian ID? each and all likelihoods?
        return X


class Statistics:
    def __init__(self, n_gaussians: int, n_features: int) -> None:
        self.n_gaussians = n_gaussians
        self.n_features = n_features
        # The accumulated log likelihood of all samples
        self.log_likelihood = 0.0
        # The accumulated number of samples
        self.T = 0
        # For each Gaussian, the accumulated sum of responsibilities, i.e. the sum of P(gaussian_i|x)
        self.n = np.zeros(shape=(n_gaussians,), dtype=float)
        # For each Gaussian, the accumulated sum of responsibility times the sample
        self.sumPx = np.zeros(shape=(n_gaussians, n_features), dtype=float)
        # For each Gaussian, the accumulated sum of responsibility times the sample squared
        self.sumPxx = np.zeros(shape=(n_gaussians, n_features), dtype=float)

    def reset(self):
        self.log_likelihood = 0.0
        self.T = 0
        self.n[:] = 0.0
        self.sumPx[:] = 0.0
        self.sumPxx[:] = 0.0

    def __add__(self, other):
        if self.n_gaussians != other.n_gaussians or self.n_features != other.n_features:
            raise ValueError("Statistics could not be added together (shape mismatch)")
        new_stats = Statistics(self.n_gaussians, self.n_features)
        new_stats.log_likelihood = self.log_likelihood + other.log_likelihood
        new_stats.T = self.T + other.T
        new_stats.n = self.n + other.n
        new_stats.sumPx = self.sumPx + other.sumPx
        new_stats.sumPxx = self.sumPxx + other.sumPxx
        return new_stats

    def __iadd__(self, other):
        if self.n_gaussians != other.n_gaussians or self.n_features != other.n_features:
            raise ValueError("Statistics could not be added together (shape mismatch)")
        self.log_likelihood += other.log_likelihood
        self.T += other.T
        self.n += other.n
        self.sumPx += other.sumPx
        self.sumPxx += other.sumPxx
        return self


class BaseGMMTrainer(ABC):
    """Base class for the different GMM Trainer implementations.

    Parameters
    ----------
    init_method
        How to initialize the GMM machine.
    random_state
        Sets a seed or RandomState for reproducibility
    update_means
        Update means on each iteration (m-step).
    update_variances
        Update variances on each iteration (m-step).
    update_weights
        Update weights on each iteration (m-step).
    mean_var_update_responsibilities_threshold
        Threshold over the responsibilities of the Gaussians Equations 9.24, 9.25 of
        Bishop, `Pattern recognition and machine learning`, 2006 require a division by
        the responsibilities, which might be equal to zero because of numerical issue.
        This threshold is used to avoid such divisions.
    """

    def __init__(
        self,
        init_method: Union[str, da.Array, KMeansTrainer] = "k-means",
        random_state: Union[int, da.random.RandomState] = 0,
        update_means: bool = True,
        update_variances: bool = False,
        update_weights: bool = False,
        mean_var_update_responsibilities_threshold: float = np.finfo(float).eps # TODO
    ):
        self.init_method = init_method
        self.random_state = random_state
        self.last_step_stats = None
        self.update_means = update_means
        self.update_variances = update_variances
        self.update_weights = update_weights

    @abstractmethod
    def initialize(self, machine: GMMMachine, data: da.Array):
        pass

    def e_step(self, machine: GMMMachine, data: da.Array):
        # The e-step is the same for each GMM Trainer
        self.last_step_stats = Statistics(
            n_gaussians=machine.n_gaussians, n_features=data.shape[-1]
        )
        if len(data.shape) == 1:
            data = data.reshape(shape=(1, -1))

        # Log weighted Gaussian likelihoods [array of shape (n_gaussians,n_samples)]
        log_weighted_likelihoods = machine.log_weighted_likelihood(data)
        # Log likelihood [array of shape (n_samples,)]
        log_likelihood = machine.log_likelihood(data)
        # Responsibility P [array of shape (n_gaussians, n_samples)]
        responsibility = da.exp(log_weighted_likelihoods - log_likelihood[None, :])

        # Accumulate

        # Total likelihood [float]
        self.last_step_stats.log_likelihood = da.sum(log_likelihood)
        # Count of samples [int]
        self.last_step_stats.T = data.shape[0]
        # Responsibilities [array of shape (n_gaussians,)]
        self.last_step_stats.n = da.sum(responsibility, axis=-1)
        # p * x [array of shape (n_gaussians, n_samples, n_features)]
        px = da.multiply(responsibility[:, :, None], data[None, :, :])
        # First order stats [array of shape (n_gaussians, n_features)]
        self.last_step_stats.sumPx = da.sum(px, axis=1)
        # Second order stats [array of shape (n_gaussians, n_features)]
        pxx = da.multiply(px[:, :, :], data[None, :, :])
        self.last_step_stats.sumPxx = da.sum(pxx, axis=1)

    @abstractmethod
    def m_step(self, machine: GMMMachine, data: da.Array):
        # The m-step must be implemented by the specialized GMM Trainers
        pass

    def compute_likelihood(self, machine: GMMMachine):
        return self.last_step_stats.log_likelihood / self.last_step_stats.T


class MLGMMTrainer(BaseGMMTrainer):
    """Maximum Likelihood trainer for GMM"""

    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)

    def initialize(
        self,
        machine: GMMMachine,
        data: da.Array,
        threshold=1.0e-5,
        k_means_trainer=None,
    ):
        """Sets the initial values of the machine, and sets up the Trainer.

        The default initialization of the Maximum Likelihood GMM parameters uses
        k-means on the data to find centroids and variances.
        """
        n_features = data.shape[-1]
        self.last_step_stats.log_likelihood = 0.0
        self.last_step_stats.T = 0
        self.last_step_stats.n = da.zeros((machine.n_gaussians,))
        self.last_step_stats.sumPx = da.zeros((machine.n_gaussians, n_features))
        self.last_step_stats.sumPxx = da.zeros((machine.n_gaussians, n_features))
        if self.init_method == "k-means":
            if k_means_trainer is None:
                k_means_trainer = KMeansTrainer()
            self.init_method = k_means_trainer
        if isinstance(self.init_method, KMeansTrainer):
            logger.info("Initializing GMM with k-means.")
            kmeans_machine = KMeansMachine(machine.n_gaussians).fit(
                data, trainer=self.init_method
            )
            (
                variances,
                weights,
            ) = kmeans_machine.get_variances_and_weights_for_each_cluster(data)
            # Set the GMM machine gaussians with the results of k-means
            machine.gaussians_ = da.array(
                [
                    Gaussian(mean=m, variance=var, variance_threshold=threshold)
                    for m, var in zip(kmeans_machine.centroids_, variances)
                ]
            )
            machine.weights = weights
        else:
            machine.gaussians_ = self.init_method

    def m_step(self, machine: GMMMachine, data: da.Array):
        """Updates a gmm machine parameter according to the e-step statistics."""
        if self.last_step_stats is None:
            raise RuntimeError("e_step must be called before m_step.")

        # Update weights if requested
        # (Equation 9.26 of Bishop, "Pattern recognition and machine learning", 2006)
        if self.update_weights:
            machine.weights = self.last_step_stats.n / self.last_step_stats.T

        # thresholded_n = da.where(self.last_step_stats.n<self.mean_var_update_responsibilities_threshold, self.mean_var_update_responsibilities_threshold, self.last_step_stats.n)
        # Threshold the low n to prevent divide by zero
        self.last_step_stats.n[self.last_step_stats.n<self.mean_var_update_responsibilities_threshold] = self.mean_var_update_responsibilities_threshold

        # Update GMM parameters using the sufficient statistics (m_ss)

        # Update means if requested
        # (Equation 9.24 of Bishop, "Pattern recognition and machine learning", 2006)
        if self.update_means:
            # Using n with the applied threshold
            machine.gaussians_["mean"] = self.last_step_stats.sumPx / self.last_step_stats.n[:,None]

        # Update variance if requested
        # (Equation 9.25 of Bishop, "Pattern recognition and machine learning", 2006)
        # ...but we use the "computational formula for the variance", i.e.
        #  var = 1/n * sum (P(x-mean)(x-mean))
        #      = 1/n * sum (Pxx) - mean^2
        if self.update_variances:
            machine.gaussians_["variance"] = self.last_step_stats.sumPxx / self.last_step_stats.n[:,None] - da.pow(machine.gaussians_["mean"], 2)
            # TODO apply variance thresholds after changes to variance!!


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
