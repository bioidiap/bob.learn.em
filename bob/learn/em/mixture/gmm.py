#!/usr/bin/env python
# @author: Yannick Dayer <yannick.dayer@idiap.ch>
# @date: Fri 30 Jul 2021 10:06:47 UTC+02

import logging
from abc import ABC
from abc import abstractmethod
from typing import Union, Any


import dask.array as da
import numpy as np
from sklearn.base import BaseEstimator

from bob.learn.em.cluster import KMeansMachine
from bob.learn.em.cluster import KMeansTrainer

logger = logging.getLogger(__name__)


EPSILON = np.finfo(float).eps


class Gaussian(np.ndarray):
    """Represents a multi-dimensional Gaussian.

    This is basically three 1D-arrays: for the mean, the (diagonal) variance, and the
    variance threshold values.

    Each array can be accessed with the `[]` operator (`my_gaussian["variance"]`).

    Variance thresholds are automatically applied when setting the variance (and when
    setting the threshold values).

    Usage:
    >>> my_gaussian = Gaussian(mean=np.array([0,1,2]))
    >>> print(my_gaussian["mean"])
    ... [0. 1. 2.]
    >>> print(my_gaussian["variance"])
    ... [1. 1. 1.]

    A numpy array of multiple Gaussian objects can be accessed in the same way:
    >>> my_gaussians = np.array(Gaussian(np.array([0,0])), Gaussian(np.array([1,1])))
    >>> print(my_gaussians["mean"])
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
            variance_threshold = np.full_like(mean, fill_value=EPSILON, dtype=float)
        rec = np.ndarray(
            shape=(n_features,),
            dtype=[("mean", float), ("variance", float), ("variance_threshold", float)],
        )
        rec["mean"] = mean
        rec["variance_threshold"] = variance_threshold
        rec["variance"] = np.maximum(variance_threshold, variance)
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
        N_LOG_2PI = x.shape[-1] * np.log(2 * np.pi)
        g_norm = N_LOG_2PI + np.sum(np.log(self["variance"]))

        # Compute the likelihood for each data point this Gaussian
        z = da.sum(da.power(x - self["mean"], 2) / self["variance"], axis=-1)
        return -0.5 * (g_norm + z)

    def __setitem__(self, key, value) -> None:
        """Set values of items (operator `[]`) of this numpy array.

        Applies the threshold on the variance when setting `variance` or
        `variance_threshold`.
        """
        if key == "variance":
            value = np.maximum(self["variance_threshold"], value)
        elif key == "variance_threshold":
            super().__setitem__("variance", np.maximum(value, self["variance"]))
        return super().__setitem__(key, value)

    def __eq__(self, other):
        return np.array_equal(self, other)

    def __ne__(self, other):
        return not np.array_equal(self, other)

    def is_similar_to(self, other, rtol=1e-5, atol=1e-8):
        return (
            np.allclose(self["mean"], other["mean"], rtol=rtol, atol=atol)
            and np.allclose(self["variance"], other["variance"], rtol=rtol, atol=atol)
            and np.allclose(
                self["variance_threshold"],
                other["variance_threshold"],
                rtol=rtol,
                atol=atol,
            )
        )


class MultiGaussian(Gaussian):
    """Container for multiple Gaussian objects.

    Ensures that the attributes and method are accessed correctly when using multiple
    Gaussian in one array.

    Usage (creates two Gaussians centered in (0,0) and (1,1), with variances of 1):
    >>> gaussians = MultiGaussian(means=np.array([[0,0],[1,1]]))
    >>> print(gaussians["mean"])
    ... [[0. 0.]
    ...  [1. 1.]]
    """

    def __new__(cls, means, variances=None, variance_thresholds=None):
        if means.ndim < 2:
            raise ValueError(f"means should be 2D but has ndim={means.ndim}.")
        n_features = means.shape[-1]
        n_gaussians = means.shape[0]
        if variances is None:
            variances = np.ones_like(means, dtype=float)
        if variance_thresholds is None:
            variance_thresholds = np.full_like(means, fill_value=EPSILON, dtype=float)
        rec = np.ndarray(
            shape=(n_gaussians, n_features),
            dtype=[("mean", float), ("variance", float), ("variance_threshold", float)],
        )
        rec["mean"] = means
        rec["variance_threshold"] = variance_thresholds
        rec["variance"] = np.maximum(variance_thresholds, variances)
        return rec.view(cls)


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

    Attributes
    ----------
    means, variances, variance_thresholds:
        Gaussians parameters.
    gaussians_:
        All Gaussians parameters.
    weights:
        Gaussians weights.

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
            weights = np.full(shape=(n_gaussians,), fill_value=(1 / n_gaussians))
        self.weights = weights

    @property
    def weights(self):
        return self.__weights

    @weights.setter
    def weights(self, w: np.ndarray):
        self.__weights = w
        self.__log_weights = np.log(self.__weights)

    @property
    def means(self):
        return self.gaussians_["mean"]

    @means.setter
    def means(self, m: np.ndarray):
        if hasattr(self, "gaussians_"):
            self.gaussians_["mean"] = m
        else:
            self.gaussians_ = MultiGaussian(means=m)

    @property
    def variances(self):
        return self.gaussians_["variance"]

    @variances.setter
    def variances(self, v: np.ndarray):
        if hasattr(self, "gaussians_"):
            self.gaussians_["variance"] = v
        else:
            self.gaussians_ = MultiGaussian(means=np.zeros_like(v), variances=v)

    @property
    def variance_thresholds(self):
        return self.gaussians_["variance_threshold"]

    @variance_thresholds.setter
    def variance_thresholds(self, t: np.ndarray):
        if hasattr(self, "gaussians_"):
            self.gaussians_["variance_threshold"] = t
        else:
            self.gaussians_ = MultiGaussian(
                means=np.zeros_like(t), variance_thresholds=t
            )

    @property
    def log_weights(self):
        return self.__log_weights

    @property
    def shape(self):
        return self.gaussians_.shape

    def __eq__(self, other):
        return np.array_equal(self.gaussians_, other.gaussians_)

    def is_similar_to(self, other, rtol=1e-5, atol=1e-8):
        return self.gaussians_.is_similar_to(
            other.gaussians_, rtol=rtol, atol=atol
        ) and np.allclose(self.weights, other.weights, rtol=rtol, atol=atol)

    def copy(self):
        copy_machine = GMMMachine(
            self.n_gaussians,
            convergence_threshold=self.convergence_threshold,
            random_state=self.random_state,
            weights=self.weights,
        )
        if hasattr(self, "gaussians_"):
            copy_machine.gaussians_ = self.gaussians_.copy()
        return copy_machine

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
        # [array of shape (n_gaussians,)]
        g_norms = N_LOG_2PI + np.sum(np.log(self.gaussians_["variance"]), axis=-1)

        # Compute the likelihood for each data point on this Gaussian
        z = da.sum(
            da.power(x[None, :, :] - self.gaussians_["mean"][:, None, :], 2)
            / self.gaussians_["variance"][:, None, :],
            axis=-1,
        )
        # Unweighted log likelihoods [array of shape (n_gaussians, n_samples)]
        l = -0.5 * (g_norms[:, None] + z)
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
        if x.ndim == 1:
            x = x.reshape((1, -1))

        # All likelihoods [array of shape (n_gaussians, n_samples)]
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
        logger.info("Training GMM...")
        for step in range(max_steps):
            logger.info(f"Iteration = {step:3d}/{max_steps}")
            average_output_previous = average_output
            trainer.e_step(self, X)
            trainer.m_step(self, X)

            average_output = trainer.compute_likelihood(self)

            if step > 0:
                convergence_value = abs(
                    (average_output_previous - average_output) / average_output_previous
                )

                # Terminates if converged (and likelihood computation is set)
                if (
                    self.convergence_threshold is not None
                    and convergence_value <= self.convergence_threshold
                ):
                    logger.info("Reached convergence threshold. Training stopped.")
                    return self
        logger.info("Reached maximum step. Training stopped without convergence.")
        return self

    def fit_partial(self, X, y=None, trainer=None, **kwargs):
        if trainer is None:
            raise ValueError("Please provide a GMM trainer for fit_partial.")

        trainer.e_step(self, X)
        trainer.m_step(self, X)
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
        self.t = 0
        # For each Gaussian, the accumulated sum of responsibilities, i.e. the sum of P(gaussian_i|x)
        self.n = np.zeros(shape=(n_gaussians,), dtype=float)
        # For each Gaussian, the accumulated sum of responsibility times the sample
        self.sumPx = np.zeros(shape=(n_gaussians, n_features), dtype=float)
        # For each Gaussian, the accumulated sum of responsibility times the sample squared
        self.sumPxx = np.zeros(shape=(n_gaussians, n_features), dtype=float)

    def from_file(self, file):
        print(dir(file))
        print(file.keys())
        self.log_likelihood = file[
            "log_liklihood"
        ]  # TODO? Fix this typo (requires files edit)
        self.t = file["T"]
        self.n = file["n"]
        self.sumPx = file["sumPx"]
        self.sumPxx = file["sumPxx"]

    def reset(self):
        self.log_likelihood = 0.0
        self.t = 0
        self.n = np.zeros_like(self.n)
        self.sumPx = np.zeros_like(self.sumPx)
        self.sumPxx = np.zeros_like(self.sumPxx)

    def __add__(self, other):
        if self.n_gaussians != other.n_gaussians or self.n_features != other.n_features:
            raise ValueError("Statistics could not be added together (shape mismatch)")
        new_stats = Statistics(self.n_gaussians, self.n_features)
        new_stats.log_likelihood = self.log_likelihood + other.log_likelihood
        new_stats.t = self.t + other.t
        new_stats.n = self.n + other.n
        new_stats.sumPx = self.sumPx + other.sumPx
        new_stats.sumPxx = self.sumPxx + other.sumPxx
        return new_stats

    def __iadd__(self, other):
        if self.n_gaussians != other.n_gaussians or self.n_features != other.n_features:
            raise ValueError("Statistics could not be added together (shape mismatch)")
        self.log_likelihood += other.log_likelihood
        self.t += other.t
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
        mean_var_update_responsibilities_threshold: float = np.finfo(float).eps,
    ):
        self.init_method = init_method
        self.random_state = random_state
        self.last_step_stats = None
        self.update_means = update_means
        self.update_variances = update_variances
        self.update_weights = update_weights
        self.mean_var_update_responsibilities_threshold = (
            mean_var_update_responsibilities_threshold
        )

    @abstractmethod
    def initialize(self, machine: GMMMachine, data: da.Array):
        pass

    def e_step(self, machine: GMMMachine, data: da.Array):
        # The e-step is the same for each GMM Trainer
        logger.debug(f"GMM Trainer e-step")

        if self.last_step_stats is None:
            self.last_step_stats = Statistics(machine.n_gaussians, data.shape[-1])
        else:
            self.last_step_stats.reset()

        if data.ndim == 1:
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
        self.last_step_stats.t = data.shape[0]
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
        """Returns the likelihood computed at the last e-step."""
        return self.last_step_stats.log_likelihood / self.last_step_stats.t


class MLGMMTrainer(BaseGMMTrainer):
    """Maximum Likelihood trainer for GMM"""

    def __init__(
        self,
        init_method: Union[str, da.Array, KMeansTrainer] = "k-means",
        random_state: Union[int, da.random.RandomState] = 0,
        update_means: bool = True,
        update_variances: bool = False,
        update_weights: bool = False,
        mean_var_update_responsibilities_threshold: float = np.finfo(float).eps,
        **kwargs,
    ):
        super().__init__(
            init_method,
            random_state,
            update_means,
            update_variances,
            update_weights,
            mean_var_update_responsibilities_threshold,
            **kwargs,
        )

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
        self.last_step_stats = Statistics(machine.n_gaussians, n_features)
        if type(self.init_method) is str and self.init_method == "k-means":
            if k_means_trainer is None:
                k_means_trainer = KMeansTrainer()
            logger.info("Initializing GMM with k-means.")
            kmeans_machine = KMeansMachine(machine.n_gaussians).fit(
                data, trainer=k_means_trainer
            )
            (
                variances,
                weights,
            ) = kmeans_machine.get_variances_and_weights_for_each_cluster(data)
            # Set the GMM machine gaussians with the results of k-means
            machine.gaussians_ = MultiGaussian(
                means=kmeans_machine.centroids_,
                variances=variances,
            )
            machine.weights = weights
        else:
            machine.gaussians_ = self.init_method  # TODO other name for this param...

    def m_step(self, machine: GMMMachine, data: da.Array):
        """Updates a gmm machine parameter according to the e-step statistics."""
        logger.debug(f"ML GMM Trainer m-step")
        if self.last_step_stats is None:
            raise RuntimeError("Initialize and e_step must be called before m_step.")

        # Update weights if requested
        # (Equation 9.26 of Bishop, "Pattern recognition and machine learning", 2006)
        if self.update_weights:
            logger.debug("Update weights.")
            machine.weights = self.last_step_stats.n / self.last_step_stats.t

        # Threshold the low n to prevent divide by zero
        thresholded_n = da.where(
            self.last_step_stats.n < self.mean_var_update_responsibilities_threshold,
            self.mean_var_update_responsibilities_threshold,
            self.last_step_stats.n,
        )
        # self.last_step_stats.n[self.last_step_stats.n<self.mean_var_update_responsibilities_threshold] = self.mean_var_update_responsibilities_threshold

        # Update GMM parameters using the sufficient statistics (m_ss):

        # Update means if requested
        # (Equation 9.24 of Bishop, "Pattern recognition and machine learning", 2006)
        if self.update_means:
            logger.debug("Update means.")
            # Using n with the applied threshold
            machine.gaussians_["mean"] = (
                self.last_step_stats.sumPx / thresholded_n[:, None]
            )

        # Update variances if requested
        # (Equation 9.25 of Bishop, "Pattern recognition and machine learning", 2006)
        # ...but we use the "computational formula for the variance", i.e.
        #  var = 1/n * sum (P(x-mean)(x-mean))
        #      = 1/n * sum (Pxx) - mean^2
        if self.update_variances:
            logger.debug("Update variances.")
            machine.gaussians_[
                "variance"
            ] = self.last_step_stats.sumPxx / thresholded_n[:, None] - da.power(
                machine.gaussians_["mean"], 2
            )


class MAPGMMTrainer(BaseGMMTrainer):
    """Maximum A Posteriori trainer for GMM.

    This class implements the maximum a posteriori (:ref:`MAP <map>`) M-step of the
    expectation-maximization algorithm for a GMM Machine.
    The prior parameters are encoded in the form of a GMM (e.g. a universal background
    model). The EM algorithm thus performs GMM adaptation.


    Parameters
    ----------
    relevance_factor:
        If set, the Reynolds adaptation will be applied with this factor.
    """

    def __init__(
        self,
        prior_gmm: GMMMachine,
        init_method: Union[str, da.Array, KMeansTrainer] = "k-means",
        random_state: Union[int, da.random.RandomState] = 0,
        update_means: bool = True,
        update_variances: bool = False,
        update_weights: bool = False,
        mean_var_update_responsibilities_threshold: float = np.finfo(float).eps,
        relevance_factor: Union[float, None] = 4,
        alpha: float = 0.5,
        **kwargs,
    ):
        super().__init__(
            init_method,
            random_state,
            update_means,
            update_variances,
            update_weights,
            mean_var_update_responsibilities_threshold,
            **kwargs,
        )
        self.reynolds_adaptation = not relevance_factor is None
        self.relevance_factor = relevance_factor
        self.alpha = alpha
        self.prior_gmm = prior_gmm

    def initialize(self, machine: GMMMachine, data: da.Array):
        if machine.n_gaussians != self.prior_gmm.n_gaussians:
            raise ValueError(
                f"Prior GMM machine (n_gaussians={self.prior_gmm.n_gaussians}) not"
                f"compatible with current machine (n_gaussians={machine.n_gaussians})."
            )
        self.last_step_stats = Statistics(machine.n_gaussians, data.shape[-1])
        machine.weights = self.prior_gmm.weights.copy()
        machine.gaussians_ = self.prior_gmm.gaussians_.copy()

    def m_step(self, machine: GMMMachine, data: da.Array):
        # Calculate the "data-dependent adaptation coefficient", alpha_i
        # [array of shape (n_gaussians, )]
        if self.reynolds_adaptation:
            alpha = self.last_step_stats.n / (
                self.last_step_stats.n + self.relevance_factor
            )
        else:
            if not hasattr(self.alpha, "ndim"):
                self.alpha = np.full((machine.n_gaussians,), self.alpha)
            alpha = self.alpha

        # - Update weights if requested
        #   Equation 11 of Reynolds et al., "Speaker Verification Using Adapted
        #   Gaussian Mixture Models", Digital Signal Processing, 2000
        if self.update_weights:
            # Calculate the maximum likelihood weights [array of shape (n_gaussians,)]
            ml_weights = self.last_step_stats.n / self.last_step_stats.t

            # Calculate the new weights
            machine.weights = alpha * ml_weights + (1 - alpha) * self.prior_gmm.weights

            # Apply the scale factor, gamma, to ensure the new weights sum to unity
            gamma = machine.weights.sum()
            machine.weights /= gamma

        # Update GMM parameters
        # - Update means if requested
        #   Equation 12 of Reynolds et al., "Speaker Verification Using Adapted
        #   Gaussian Mixture Models", Digital Signal Processing, 2000
        if self.update_means:
            new_means = (
                da.multiply(alpha[:, None], (self.last_step_stats.sumPx / self.last_step_stats.n[:, None]))
                + da.multiply((1 - alpha[:, None]), self.prior_gmm.means)
            )
            machine.means = da.where(
                self.last_step_stats.n[:, None]
                < self.mean_var_update_responsibilities_threshold,
                self.prior_gmm.means,
                new_means,
            )

        # - Update variance if requested
        #   Equation 13 of Reynolds et al., "Speaker Verification Using Adapted
        #   Gaussian Mixture Models", Digital Signal Processing, 2000
        if self.update_variances:
            # Calculate new variances (equation 13)
            prior_norm_variances = (
                self.prior_gmm.variances + self.prior_gmm.means
            ) - da.power(machine.means, 2)
            new_variances = (
                alpha[:, None] * self.last_step_stats.sumPxx / self.last_step_stats.n[:, None]
                + (1 - alpha[:, None]) * (self.prior_gmm.variances + self.prior_gmm.means)
                - da.power(machine.means, 2)
            )
            machine.variances = da.where(
                self.last_step_stats.n[:, None]
                < self.mean_var_update_responsibilities_threshold,
                prior_norm_variances,
                new_variances,
            )
