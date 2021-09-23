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


class Gaussians(np.ndarray):
    """Represents a set of multi-dimensional Gaussian.

    One Gaussian is represented by three 1D-arrays:
        - its mean,
        - its (diagonal) variance,
        - the variance threshold values.

    Each array can be accessed with the `[]` operator (`my_gaussians["variances"]`).

    Variances thresholds are automatically applied when setting the variances (and when
    setting the threshold values).

    Usage:
    >>> my_gaussians = Gaussians(means=np.array([[0,0,0],[1,1,1]]))
    >>> print(my_gaussians["means"])
    ... [[0. 0. 0.]
         [1. 1. 1.]]
    >>> print(my_gaussians["variances"])
    ... [[1. 1. 1.]
         [1. 1. 1.]]
    """

    def __new__(cls, means, variances=None, variance_thresholds=None):
        """
        Creates the backend 'structured numpy array' with initial values.

        Parameters
        ----------
        means: array of shape (n_gaussians, n_features)
            Center of the Gaussian distribution.
        variances: array of shape (n_gaussians, n_features)
            Diagonal variance matrix of the Gaussian distribution. Defaults to 1.
        variance_thresholds: array of shape (n_gaussians, n_features)
            Threshold values applied to the variances. Defaults to epsilon.
        """
        means = np.array(means)
        if means.ndim < 2:
            means = means.reshape((1, -1))
        n_features = means.shape[-1]
        n_gaussians = means.shape[0]
        if variances is None:
            variances = np.ones_like(means, dtype=float)
        if variance_thresholds is None:
            variance_thresholds = np.full_like(means, fill_value=EPSILON, dtype=float)
        rec = np.ndarray(
            shape=(n_gaussians, n_features),
            dtype=[
                ("means", float),
                ("variances", float),
                ("variance_thresholds", float),
            ],
        )
        rec["means"] = means
        rec["variance_thresholds"] = variance_thresholds
        rec["variances"] = np.maximum(variance_thresholds, variances)
        return rec.view(cls)

    def log_likelihood(self, x):
        """Returns the log-likelihood for x on each gaussian.

        Parameters
        ----------
        x: array of shape (n_features,) or (n_samples, n_features)
            The point (or points) to compute the log-likelihood of.

        Returns
        -------
        array of shape (n_gaussians, n_samples)
            The log likelihood of each points in x for each Gaussian.
        """
        N_LOG_2PI = x.shape[-1] * np.log(2 * np.pi)
        g_norm = N_LOG_2PI + np.sum(np.log(self["variances"]), axis=-1)

        # Compute the likelihood for each data point on each Gaussian
        z = da.sum(
            da.power(x[None, ..., :] - self["means"][..., None, :], 2)
            / self["variances"][..., None, :],
            axis=-1,
        )
        return -0.5 * (g_norm + z)

    def __setitem__(self, key, value) -> None:
        """Set values of items (operator `[]`) of this numpy array.

        Applies the threshold on the variances when setting `variances` or
        `variance_thresholds`.
        """
        if key == "variances":
            value = np.maximum(self["variance_thresholds"], value)
        elif key == "variance_thresholds":
            super().__setitem__("variances", np.maximum(value, self["variances"]))
        return super().__setitem__(key, value)

    def __eq__(self, other):
        return np.array_equal(self, other)

    def __ne__(self, other):
        return not np.array_equal(self, other)

    def is_similar_to(self, other, rtol=1e-5, atol=1e-8):
        return (
            np.allclose(self["means"], other["means"], rtol=rtol, atol=atol)
            and np.allclose(self["variances"], other["variances"], rtol=rtol, atol=atol)
            and np.allclose(
                self["variance_thresholds"],
                other["variance_thresholds"],
                rtol=rtol,
                atol=atol,
            )
        )


class GMMStats:
    """Stores accumulated statistics of a GMM."""

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
        self.sum_px = np.zeros(shape=(n_gaussians, n_features), dtype=float)
        # For each Gaussian, the accumulated sum of responsibility times the sample squared
        self.sum_pxx = np.zeros(shape=(n_gaussians, n_features), dtype=float)

    @classmethod
    def from_hdf5(cls, hdf5):
        """Creates a new GMMStats object from an `HDF5File` object."""
        try:
            version_major, version_minor = hdf5.get("meta_file_version").split(".")
        except RuntimeError:
            version_major, version_minor = 0, 0
        if int(version_major) >= 1:
            if hdf5["meta_writer_class"] != str(cls):
                logger.warning(f"{hdf5['meta_writer_class']} is not a GMMStats.")
            self = cls(n_gaussians=hdf5["n_gaussians"], n_features=hdf5["n_inputs"])
            self.log_likelihood = hdf5["log_likelihood"]
            self.t = hdf5["T"]
            self.n = hdf5["n"]
            self.sum_px = hdf5["sumPx"]
            self.sum_pxx = hdf5["sumPxx"]
        else:  # Legacy file version
            logger.info("Loading a legacy HDF5 stats file.")
            self = cls(
                n_gaussians=int(hdf5["n_gaussians"]), n_features=int(hdf5["n_inputs"])
            )
            self.log_likelihood = float(hdf5["log_liklihood"])
            self.t = int(hdf5["T"])
            self.n = hdf5["n"].reshape((self.n_gaussians,))
            self.sum_px = hdf5["sumPx"].reshape(self.shape)
            self.sum_pxx = hdf5["sumPxx"].reshape(self.shape)
        return self

    def save(self, hdf5):
        """Saves the current statistsics in an `HDF5File` object."""
        hdf5["meta_file_version"] = "1.0"
        hdf5["meta_writer_class"] = str(self.__class__)
        hdf5["n_gaussians"] = self.n_gaussians
        hdf5["n_inputs"] = self.n_features
        hdf5["log_likelihood"] = self.log_likelihood
        hdf5["T"] = self.t
        hdf5["n"] = self.n
        hdf5["sumPx"] = self.sum_px
        hdf5["sumPxx"] = self.sum_pxx

    def load(self, hdf5):
        """Overwrites the current statistics with those in an `HDF5File` object."""
        new_self = self.from_hdf5(hdf5)
        if new_self.shape != self.shape:
            logger.warning("Loaded GMMStats from hdf5 with a different shape.")
            self.resize(*new_self.shape)
        self.init(
            new_self.log_likelihood,
            new_self.t,
            new_self.n,
            new_self.sum_px,
            new_self.sum_pxx,
        )

    def init(self, log_likelihood=0.0, t=0, n=None, sum_px=None, sum_pxx=None):
        """Resets the statistics values to zero or a defined value."""
        self.log_likelihood = log_likelihood
        self.t = t
        self.n = np.zeros(shape=(self.n_gaussians,), dtype=float) if n is None else n
        self.sum_px = (
            np.zeros(shape=(self.n_gaussians, self.n_features), dtype=float)
            if sum_px is None
            else sum_px
        )
        self.sum_pxx = (
            np.zeros(shape=(self.n_gaussians, self.n_features), dtype=float)
            if sum_pxx is None
            else sum_pxx
        )

    def __add__(self, other):
        if self.n_gaussians != other.n_gaussians or self.n_features != other.n_features:
            raise ValueError("Statistics could not be added together (shape mismatch)")
        new_stats = GMMStats(self.n_gaussians, self.n_features)
        new_stats.log_likelihood = self.log_likelihood + other.log_likelihood
        new_stats.t = self.t + other.t
        new_stats.n = self.n + other.n
        new_stats.sum_px = self.sum_px + other.sum_px
        new_stats.sum_pxx = self.sum_pxx + other.sum_pxx
        return new_stats

    def __iadd__(self, other):
        if self.n_gaussians != other.n_gaussians or self.n_features != other.n_features:
            raise ValueError("Statistics could not be added together (shape mismatch)")
        self.log_likelihood += other.log_likelihood
        self.t += other.t
        self.n += other.n
        self.sum_px += other.sum_px
        self.sum_pxx += other.sum_pxx
        return self

    def __eq__(self, other):
        return (
            self.log_likelihood == other.log_likelihood
            and self.t == other.t
            and np.array_equal(self.n, other.n)
            and np.array_equal(self.sum_px, other.sum_px)
            and np.array_equal(self.sum_pxx, other.sum_pxx)
        )

    def is_similar_to(self, other, rtol=1e-5, atol=1e-8):
        return (
            np.isclose(self.log_likelihood, other.log_likelihood, rtol=rtol, atol=atol)
            and np.isclose(self.t, other.t, rtol=rtol, atol=atol)
            and np.allclose(self.n, other.n, rtol=rtol, atol=atol)
            and np.allclose(self.sum_px, other.sum_px, rtol=rtol, atol=atol)
            and np.allclose(self.sum_pxx, other.sum_pxx, rtol=rtol, atol=atol)
        )

    def resize(self, n_gaussians, n_features):
        self.n_gaussians = n_gaussians
        self.n_features = n_features
        self.init()

    @property
    def shape(self):
        return (self.n_gaussians, self.n_features)


class GMMMachine(BaseEstimator):
    """Stores a GMM parameters.

    Each mixture is a Gaussian represented by a mean and a diagonal variance matrix.

    TODO doc

    Usage:
    >>> machine = GMMMachine(n_gaussians=2, trainer="ml")
    >>> machine = machine.fit(data)
    >>> print(machine.means)

    >>> map_machine = GMMMachine(n_gaussians=2, trainer="map", ubm=machine)
    >>> map_machine = map_machine.fit(post_data)
    >>> print(map_machine.means)

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
        trainer: str = "ml",
        ubm=None,
        convergence_threshold: float = 1e-5,
        random_state: Union[int, da.random.RandomState] = 0,
        initial_gaussians: Union[Gaussians, None] = None,
        initial_weights: Union[np.ndarray, None] = None,
        k_means_trainer: Union[KMeansTrainer, None] = None,
    ):
        """
        Parameters
        ----------
        n_gaussians:
            The number of gaussians to be represented by the machine.
        trainer:
            `"ml"` for the maximum likelihood estimator method;
            `"map"` for the maximum a posteriori method. (MAP Requires `ubm`)
        ubm: GMMMachine
            Universal Background Model. GMMMachine Required for the MAP method.
        convergence_threshold:
            The threshold value of likelihood difference between e-m steps used for
            stopping the training iterations.
        random_state:
            Specifies a RandomState or a seed for reproducibility.
        initial_gaussians:
            Optional set of values to skip the k-means initialization.
        initial_weights: array of shape (n_gaussians,) or None
            The weight of each Gaussian. (defaults to `1/n_gaussians`)
        k_means_trainer:
            Optional trainer for the k-means method, replacing the default one.
        """
        self.n_gaussians = n_gaussians
        self.trainer = trainer
        self.m_step_func = map_gmm_m_step if self.trainer == "map" else ml_gmm_m_step
        if self.trainer == "map" and ubm is None:
            raise ValueError("A ubm is required for MAP GMM.")
        self.ubm = ubm
        self.convergence_threshold = convergence_threshold
        self.random_state = random_state
        self.initial_gaussians = initial_gaussians
        if initial_weights is None:
            weights = np.full(shape=(n_gaussians,), fill_value=(1 / n_gaussians))
        self.weights = weights
        self.kmeans_trainer = k_means_trainer

    @property
    def weights(self):
        return self._weights

    @weights.setter
    def weights(self, w: np.ndarray):
        self._weights = w
        self._log_weights = np.log(self._weights)

    @property
    def means(self):
        return self.gaussians_["means"]

    @means.setter
    def means(self, m: np.ndarray):
        if hasattr(self, "gaussians_"):
            self.gaussians_["means"] = m
        else:
            self.gaussians_ = Gaussians(means=m)

    @property
    def variances(self):
        return self.gaussians_["variances"]

    @variances.setter
    def variances(self, v: np.ndarray):
        if hasattr(self, "gaussians_"):
            self.gaussians_["variances"] = v
        else:
            self.gaussians_ = Gaussians(means=np.zeros_like(v), variances=v)

    @property
    def variance_thresholds(self):
        return self.gaussians_["variance_thresholds"]

    @variance_thresholds.setter
    def variance_thresholds(self, t: np.ndarray):
        if hasattr(self, "gaussians_"):
            self.gaussians_["variance_thresholds"] = t
        else:
            self.gaussians_ = Gaussians(means=np.zeros_like(t), variance_thresholds=t)

    @property
    def log_weights(self):
        return self._log_weights

    @property
    def shape(self):
        return self.gaussians_.shape

    def __eq__(self, other):
        return np.array_equal(self.gaussians_, other.gaussians_)

    def is_similar_to(self, other, rtol=1e-5, atol=1e-8):
        return self.gaussians_.is_similar_to(
            other.gaussians_, rtol=rtol, atol=atol
        ) and np.allclose(self.weights, other.weights, rtol=rtol, atol=atol)

    def initialize_gaussians(self, data: Union[da.Array, None] = None):
        if self.trainer == "map":
            self.weights = self.ubm.weights.copy()
            self.gaussians_ = self.ubm.gaussians_.copy()
        else:
            if self.initial_gaussians is None:
                if data is None:
                    raise ValueError("Data is required when training with k-means.")
                logger.info("Initializing GMM with k-means.")
                kmeans_trainer = self.kmeans_trainer or KMeansTrainer()
                kmeans_machine = KMeansMachine(self.n_gaussians).fit(
                    data, trainer=kmeans_trainer
                )

                (
                    variances,
                    weights,
                ) = kmeans_machine.get_variances_and_weights_for_each_cluster(data)

                # Set the GMM machine gaussians with the results of k-means
                self.gaussians_ = Gaussians(
                    means=kmeans_machine.centroids_,
                    variances=variances,
                )
                self.weights = weights.copy()
            else:
                logger.info("Initializing GMM with user-provided values.")
                self.gaussians_ = self.initial_gaussians.copy()
                self.weights = np.full((self.n_gaussians,), 1 / self.n_gaussians)

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
        N_LOG_2PI = x.shape[-1] * np.log(2 * np.pi)
        # g_norm for each gaussian [array of shape (n_gaussians,)]
        g_norms = N_LOG_2PI + np.sum(np.log(self.gaussians_["variances"]), axis=-1)

        # Compute the likelihood for each data point on this Gaussian
        z = da.sum(
            da.power(x[None, :, :] - self.gaussians_["means"][:, None, :], 2)
            / self.gaussians_["variances"][:, None, :],
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

    def acc_statistics(self, data: da.Array):  # TODO? accumulate successive calls?
        """Accumulates the statistics of GMMStats for a set of data."""
        statistics = GMMStats(self.n_gaussians, data.shape[-1])

        if data.ndim == 1:
            data = data.reshape(shape=(1, -1))

        # Log weighted Gaussian likelihoods [array of shape (n_gaussians,n_samples)]
        log_weighted_likelihoods = self.log_weighted_likelihood(data)
        # Log likelihood [array of shape (n_samples,)]
        log_likelihood = self.log_likelihood(data)
        # Responsibility P [array of shape (n_gaussians, n_samples)]
        responsibility = da.exp(log_weighted_likelihoods - log_likelihood[None, :])

        # Accumulate

        # Total likelihood [float]
        statistics.log_likelihood = da.sum(log_likelihood)
        # Count of samples [int]
        statistics.t = data.shape[0]
        # Responsibilities [array of shape (n_gaussians,)]
        statistics.n = da.sum(responsibility, axis=-1)
        # p * x [array of shape (n_gaussians, n_samples, n_features)]
        px = da.multiply(responsibility[:, :, None], data[None, :, :])
        # First order stats [array of shape (n_gaussians, n_features)]
        statistics.sum_px = da.sum(px, axis=1)
        # Second order stats [array of shape (n_gaussians, n_features)]
        pxx = da.multiply(px[:, :, :], data[None, :, :])
        statistics.sum_pxx = da.sum(pxx, axis=1)

        return statistics

    def linear_scoring(self, models, test_stats, test_channel_offsets = 0, frame_length_normalization: bool = False):
        """Returns a score for each model against `self`, representing the UBM.

        Parameters
        ----------
        models: list of GMMMachine objects
        test_stats: list of GMMStats objects
        test_channel_offsets: array of shape (n_test_stats, n_gaussians)

        """
        # All models.means [array of shape (n_models, n_gaussians, n_features)]
        means = np.array([model.means for model in models]) # TODO elegant way to put into arrays
        # All stats.sum_px [array of shape (n_test_stats, n_gaussians, n_features)]
        sum_px = np.array([stat.sum_px for stat in test_stats])
        # All stats.n [array of shape (n_test_stats, n_gaussians)]
        n = np.array([stat.n for stat in test_stats])
        # All stats.t [array of shape (n_test_stats,)]
        t = np.array([stat.t for stat in test_stats])
        # Offsets [array of shape (n_test_stats, `n_gaussians * n_features`? TODO)]
        test_channel_offsets = np.array(test_channel_offsets)

        # TODO sizes broadcast, to accept one or multiple GMM, and one or multiple stats
        # TODO? special case for MAP? (score self against ubm?)

        # Compute A [array of shape (n_models, n_gaussians * n_features)]
        a = (means - self.means) / self.variances
        # Compute B [array of shape (n_gaussians * n_features, n_test_stats)]
        # TODO: dims check
        # TODO: channel offset dims test when not 0
        b = sum_px[:,:,:] - (n[:,:,None] * (self.means[None,:,:] + test_channel_offsets))
        b = da.transpose(b, axes=(1,2,0))
        # Apply normalization if needed.
        if frame_length_normalization:
            b = da.where(abs(t)<=EPSILON, 0, b[:,:]/t[None,:])
        # Compute LLR  TODO: dims check
        print(f"n_gaussians = {means.shape[1]}")
        print(f"n_features = {means.shape[2]}")
        print(f"n_models = {means.shape[0]}")
        print(f"n_test_stats = {sum_px.shape[0]}")
        print(f"A: {a.shape}")
        print(f"B: {b.shape}")
        return da.tensordot(a, b, 2)

    def linear_scoring_means_input(self, means, test_stats, test_channel_offsets = 0, frame_length_normalization: bool = False):
        """Returns a score for each model against `self`, representing the UBM."""
        # TODO sizes broadcast
        # TODO special case for MAP? (score self against ubm?)
        # Compute A  TODO: dims check
        a = (means - self.means) / self.variances
        # Compute B  TODO: dims check
        b = test_stats.sum_px - (test_stats.n * (self.means + test_channel_offsets))
        # Apply normalization if needed.
        if frame_length_normalization:
            if test_stats.t == 0:
                b = 0
            else:
                b /= test_stats.t
        # Compute LLR  TODO: dims check
        return da.dot(a,b)

    def e_step(self, data: da.Array):
        return self.acc_statistics(data)

    def m_step(
        self,
        stats: GMMStats,
        update_means=True,
        update_variances=False,
        update_weights=False,
        **kwargs,
    ):
        self.m_step_func(
            self,
            stats,
            update_means=update_means,
            update_variances=update_variances,
            update_weights=update_weights,
            **kwargs,
        )

    def fit(self, X, y=None, max_steps: int = 200, **kwargs):
        if not hasattr(self, "gaussians_"):
            self.initialize_gaussians(X)

        average_output = 0
        logger.info("Training GMM...")
        for step in range(max_steps):
            logger.info(f"Iteration = {step:3d}/{max_steps}")
            average_output_previous = average_output
            self.last_statistics = self.e_step(X)
            self.m_step_func(
                machine=self,
                statistics=self.last_statistics,
            )

            average_output = (
                self.last_statistics.log_likelihood / self.last_statistics.t
            )  # Note: Uses the stats from before m_step, leading to an additional m_step

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

    def fit_partial(self, X, y=None, **kwargs):
        if not hasattr(self, "gaussians_"):
            self.initialize_gaussians(X)

        self.last_statistics = self.e_step(X)
        self.m_step_func(
            machine=self,
            statistics=self.last_statistics,
        )
        return self

    def transform(self, X, **kwargs):
        return self.e_step(X)


def ml_gmm_m_step(
    machine: GMMMachine,
    statistics: GMMStats,
    update_means=True,
    update_variances=False,
    update_weights=False,
    mean_var_update_threshold=EPSILON,
):
    """Updates a gmm machine parameter according to the e-step statistics."""
    logger.debug("ML GMM Trainer m-step")

    # Update weights if requested
    # (Equation 9.26 of Bishop, "Pattern recognition and machine learning", 2006)
    if update_weights:
        logger.debug("Update weights.")
        machine.weights = statistics.n / statistics.t

    # Threshold the low n to prevent divide by zero
    thresholded_n = da.where(
        statistics.n < mean_var_update_threshold,
        mean_var_update_threshold,
        statistics.n,
    )
    # self.last_step_stats.n[self.last_step_stats.n<self.mean_var_update_responsibilities_threshold] = self.mean_var_update_responsibilities_threshold

    # Update GMM parameters using the sufficient statistics (m_ss):

    # Update means if requested
    # (Equation 9.24 of Bishop, "Pattern recognition and machine learning", 2006)
    if update_means:
        logger.debug("Update means.")
        # Using n with the applied threshold
        machine.means = statistics.sum_px / thresholded_n[:, None]

    # Update variances if requested
    # (Equation 9.25 of Bishop, "Pattern recognition and machine learning", 2006)
    # ...but we use the "computational formula for the variance", i.e.
    #  var = 1/n * sum (P(x-mean)(x-mean))
    #      = 1/n * sum (Pxx) - mean^2
    if update_variances:
        logger.debug("Update variances.")
        machine.variances = statistics.sum_pxx / thresholded_n[:, None] - da.power(
            machine.means, 2
        )


def map_gmm_m_step(
    machine: GMMMachine,
    statistics: GMMStats,
    update_means=True,
    update_variances=False,
    update_weights=False,
    reynolds_adaptation=True,
    relevance_factor=4,
    alpha=0.5,
    mean_var_update_threshold=EPSILON,
):
    """Updates a GMMMachine parameters using statistics."""
    # Calculate the "data-dependent adaptation coefficient", alpha_i
    # [array of shape (n_gaussians, )]
    if reynolds_adaptation:
        alpha = statistics.n / (statistics.n + relevance_factor)
    else:
        if not hasattr(alpha, "ndim"):
            alpha = np.full((machine.n_gaussians,), alpha)

    # - Update weights if requested
    #   Equation 11 of Reynolds et al., "Speaker Verification Using Adapted
    #   Gaussian Mixture Models", Digital Signal Processing, 2000
    if update_weights:
        # Calculate the maximum likelihood weights [array of shape (n_gaussians,)]
        ml_weights = statistics.n / statistics.t

        # Calculate the new weights
        machine.weights = alpha * ml_weights + (1 - alpha) * machine.ubm.weights

        # Apply the scale factor, gamma, to ensure the new weights sum to unity
        gamma = machine.weights.sum()
        machine.weights /= gamma

    # Update GMM parameters
    # - Update means if requested
    #   Equation 12 of Reynolds et al., "Speaker Verification Using Adapted
    #   Gaussian Mixture Models", Digital Signal Processing, 2000
    if update_means:
        new_means = (
            da.multiply(
                alpha[:, None],
                (statistics.sum_px / statistics.n[:, None]),
            )
            + da.multiply((1 - alpha[:, None]), machine.ubm.means)
        )
        machine.means = da.where(
            statistics.n[:, None] < mean_var_update_threshold,
            machine.ubm.means,
            new_means,
        )

    # - Update variance if requested
    #   Equation 13 of Reynolds et al., "Speaker Verification Using Adapted
    #   Gaussian Mixture Models", Digital Signal Processing, 2000
    if update_variances:
        # Calculate new variances (equation 13)
        prior_norm_variances = (
            machine.ubm.variances + machine.ubm.means
        ) - da.power(machine.means, 2)
        new_variances = (
            alpha[:, None] * statistics.sum_pxx / statistics.n[:, None]
            + (1 - alpha[:, None])
            * (machine.ubm.variances + machine.ubm.means)
            - da.power(machine.means, 2)
        )
        machine.variances = da.where(
            statistics.n[:, None] < mean_var_update_threshold,
            prior_norm_variances,
            new_variances,
        )
