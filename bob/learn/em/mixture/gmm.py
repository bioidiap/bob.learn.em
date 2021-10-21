#!/usr/bin/env python
# @author: Yannick Dayer <yannick.dayer@idiap.ch>
# @date: Fri 30 Jul 2021 10:06:47 UTC+02

"""This module provides classes and functions for the training and usage of GMM.

"""

import logging
from typing import Union

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
    [[0. 0. 0.]
     [1. 1. 1.]]
    >>> print(my_gaussians["variances"])
    [[1. 1. 1.]
     [1. 1. 1.]]


    Methods
    -------
    log_likelihood(X)
        Returns the log likelihood of each element of X on each Gaussian.
    is_similar_to(other, rtol=1e-5, atol=1e-8)
        Returns True if other is equal to self (with tolerances).
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

    def log_likelihood(self, data):
        """Returns the log-likelihood for x on each gaussian.

        Parameters
        ----------
        data: array of shape (n_features,) or (n_samples, n_features)
            The point (or points) to compute the log-likelihood of.

        Returns
        -------
        array of shape (n_gaussians, n_samples)
            The log likelihood of each points in x for each Gaussian.
        """

        # TODO precompute those
        n_log_2pi = data.shape[-1] * np.log(2 * np.pi)
        g_norm = n_log_2pi + np.sum(np.log(self["variances"]), axis=-1)

        # Compute the likelihood for each data point on each Gaussian
        z = da.sum(
            da.power(data[None, ..., :] - self["means"][..., None, :], 2)
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
        """Returns True if `other` has the same values (within a tolerance)."""
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
    """Stores accumulated statistics of a GMM.

    Attributes
    ----------
    log_likelihood: float
        The sum of log_likelihood of each sample on a GMM.
    t: int
        The number of considered samples.
    n: array of shape (n_gaussians,)
        Sum of responsibility.
    sum_px: array of shape (n_gaussians, n_features)
        First order statistic
    sum_pxx: array of shape (n_gaussians, n_features)
        Second order statistic
    """

    def __init__(self, n_gaussians: int, n_features: int) -> None:
        self.n_gaussians = n_gaussians
        self.n_features = n_features
        self.log_likelihood = 0
        self.t = 0
        self.n = da.zeros(shape=(self.n_gaussians,), dtype=float)
        self.sum_px = da.zeros(shape=(self.n_gaussians, self.n_features), dtype=float)
        self.sum_pxx = da.zeros(shape=(self.n_gaussians, self.n_features), dtype=float)

    def init_fields(self, log_likelihood=0.0, t=0, n=None, sum_px=None, sum_pxx=None):
        """Initializes the statistics values to a defined value, or zero by default."""
        # The accumulated log likelihood of all samples
        self.log_likelihood = log_likelihood
        # The accumulated number of samples
        self.t = t
        # For each Gaussian, the accumulated sum of responsibilities, i.e. the sum of
        # P(gaussian_i|x)
        self.n = da.zeros(shape=(self.n_gaussians,), dtype=float) if n is None else n
        # For each Gaussian, the accumulated sum of responsibility times the sample
        self.sum_px = (
            da.zeros(shape=(self.n_gaussians, self.n_features), dtype=float)
            if sum_px is None
            else sum_px
        )
        # For each Gaussian, the accumulated sum of responsibility times the sample
        # squared
        self.sum_pxx = (
            da.zeros(shape=(self.n_gaussians, self.n_features), dtype=float)
            if sum_pxx is None
            else sum_pxx
        )

    def reset(self):
        """Sets all statistics to zero."""
        self.init_fields()

    @classmethod
    def from_hdf5(cls, hdf5):
        """Creates a new GMMStats object from an `HDF5File` object."""
        try:
            version_major, version_minor = hdf5.get("meta_file_version").split(".")
            logger.debug(
                f"Reading a GMMStats HDF5 file of version {version_major}.{version_minor}"
            )
        except RuntimeError:
            version_major, version_minor = 0, 0
        if int(version_major) >= 1:
            if hdf5["meta_writer_class"] != str(cls):
                logger.warning(f"{hdf5['meta_writer_class']} is not {cls}.")
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
        self.init_fields(
            new_self.log_likelihood,
            new_self.t,
            new_self.n,
            new_self.sum_px,
            new_self.sum_pxx,
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
        """Returns True if `other` has the same values (within a tolerance)."""
        return (
            np.isclose(self.log_likelihood, other.log_likelihood, rtol=rtol, atol=atol)
            and np.isclose(self.t, other.t, rtol=rtol, atol=atol)
            and np.allclose(self.n, other.n, rtol=rtol, atol=atol)
            and np.allclose(self.sum_px, other.sum_px, rtol=rtol, atol=atol)
            and np.allclose(self.sum_pxx, other.sum_pxx, rtol=rtol, atol=atol)
        )

    def resize(self, n_gaussians, n_features):
        """Reinitializes the machine with new dimensions."""
        self.n_gaussians = n_gaussians
        self.n_features = n_features
        self.init_fields()

    @property
    def shape(self):
        """The number of gaussians and their dimensionality."""
        return (self.n_gaussians, self.n_features)


class GMMMachine(BaseEstimator):
    """Transformer that stores a Gaussian Mixture Model (GMM) parameters.

    This class implements the statistical model for multivariate diagonal mixture
    Gaussian distribution (GMM), as well as ways to train a model on data.

    A GMM is defined as
    :math:`\\sum_{c=0}^{C} \\omega_c \\mathcal{N}(x | \\mu_c, \\sigma_c)`, where
    :math:`C` is the number of Gaussian components :math:`\\mu_c`, :math:`\\sigma_c`
    and :math:`\\omega_c` are respectively the the mean, variance and the weight of
    each gaussian component :math:`c`.
    See Section 2.3.9 of Bishop, \"Pattern recognition and machine learning\", 2006

    Two types of training are available MLE and MAP, chosen with `trainer`.

    Maximum Likelihood Estimation (:ref:`MLE <mle>`, ML)
    ---------------------------------------
    The mixtures are initialized (with k-means by default).
    The means, variances, and weights of the mixtures are then trained on the data to
    increase the likelihood value. (:ref:`MLE <mle>`)

    Maximum a Posteriori (:ref:`MAP <map>`)
    --------------------------
    The MAP machine takes another GMM machine as prior, called Universal Background
    Model (UBM).
    The means, variances, and weights of the MAP mixtures are then trained on the data
    as adaptation of the UBM.

    Both training method use a Expectation-Maximization (e-m) algorithm to iteratively
    train the GMM.

    Note
    ----
    When setting manually any of the means, variances or variance thresholds, the
    k-means initialization will be skipped in `fit`.

    Usage
    -----
    Maximum likelihood:
    >>> data = np.array([[0,0,0],[1,1,1]])
    >>> ml_machine = GMMMachine(n_gaussians=2, trainer="ml")
    >>> ml_machine = ml_machine.fit(data)
    >>> print(ml_machine.means)
    [[1. 1. 1.]
     [0. 0. 0.]]

    Maximum a Posteriori:
    >>> post_data = np.array([[0.5, 0.5, 0],[1.5, 1, 1.5]])
    >>> map_machine = GMMMachine(n_gaussians=2, trainer="map", ubm=ml_machine)
    >>> map_machine = map_machine.fit(post_data)
    >>> print(map_machine.means)
    [[1.1 1.  1.1]
     [0.1 0.1 0. ]]

    Partial fitting:
    >>> machine = GMMMachine(n_gaussians=2, trainer="ml")
    >>> for step in range(5):
    ...     machine = machine.fit_partial(data)
    >>> print(machine.means)
    [[1. 1. 1.]
     [0. 0. 0.]]

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
        ubm: "Union[GMMMachine, None]" = None,
        convergence_threshold: float = 1e-5,
        max_fitting_steps: Union[int, None] = 200,
        random_state: Union[int, da.random.RandomState] = 0,
        initial_gaussians: Union[Gaussians, None] = None,
        weights: "Union[np.ndarray[('n_gaussians',), float], None]" = None,
        k_means_trainer: Union[KMeansTrainer, None] = None,
        update_means: bool = True,
        update_variances: bool = False,
        update_weights: bool = False,
        mean_var_update_threshold: float = EPSILON,
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
        max_fitting_steps:
            The number of e-m iterations to fit the GMM. Stop the training even when
            the convergence threshold isn't met.
        random_state:
            Specifies a RandomState or a seed for reproducibility.
        initial_gaussians:
            Optional set of values to skip the k-means initialization.
        weights: array of shape (n_gaussians,) or None
            The weight of each Gaussian. (defaults to `1/n_gaussians`)
        k_means_trainer:
            Optional trainer for the k-means method, replacing the default one.
        update_means:
            Update the Gaussians means at every m step.
        update_variances:
            Update the Gaussians variances at every m step.
        update_weights:
            Update the GMM weights at every m step.
        """
        self.n_gaussians = n_gaussians
        self.trainer = trainer if trainer in ["ml", "map"] else "ml"
        self.m_step_func = map_gmm_m_step if self.trainer == "map" else ml_gmm_m_step
        if self.trainer == "map" and ubm is None:
            raise ValueError("A ubm is required for MAP GMM.")
        self.ubm = ubm
        if max_fitting_steps is None and convergence_threshold is None:
            raise ValueError(
                "Either or both convergence_threshold and max_fitting_steps must be set"
            )
        self.convergence_threshold = convergence_threshold
        self.max_fitting_steps = max_fitting_steps
        self.random_state = random_state
        self.initial_gaussians = initial_gaussians
        if weights is None:
            weights = np.full(
                shape=(self.n_gaussians,), fill_value=(1 / self.n_gaussians)
            )
        self.weights = weights
        self.k_means_trainer = k_means_trainer
        self.update_means = update_means
        self.update_variances = update_variances
        self.update_weights = update_weights
        self.mean_var_update_threshold = mean_var_update_threshold

    @property
    def weights(self):
        """The weights of each Gaussian mixture."""
        return self._weights

    @weights.setter
    def weights(self, weights: "np.ndarray[('n_gaussians',), float]"):
        self._weights = weights
        self._log_weights = np.log(self._weights)

    @property
    def means(self):
        """The means of each Gaussian."""
        return self.gaussians_["means"]

    @means.setter
    def means(self, means: "np.ndarray[('n_gaussians', 'n_features'), float]"):
        if hasattr(self, "gaussians_"):
            self.gaussians_["means"] = means
        else:
            self.gaussians_ = Gaussians(means=means)

    @property
    def variances(self):
        """The (diagonal) variances of the gaussians."""
        return self.gaussians_["variances"]

    @variances.setter
    def variances(self, variances: "np.ndarray[('n_gaussians', 'n_features'), float]"):
        if hasattr(self, "gaussians_"):
            self.gaussians_["variances"] = variances
        else:
            self.gaussians_ = Gaussians(
                means=np.zeros_like(variances), variances=variances
            )

    @property
    def variance_thresholds(self):
        """Threshold below which variances are clamped to prevent precision losses."""
        return self.gaussians_["variance_thresholds"]

    @variance_thresholds.setter
    def variance_thresholds(
        self, threshold: "np.ndarray[('n_gaussians', 'n_features'), float]"
    ):
        if hasattr(self, "gaussians_"):
            self.gaussians_["variance_thresholds"] = threshold
        else:
            self.gaussians_ = Gaussians(
                means=np.zeros_like(threshold), variance_thresholds=threshold
            )

    @property
    def log_weights(self):
        """Retrieve the logarithm of the weights."""
        return self._log_weights

    @property
    def shape(self):
        """Shape of the gaussians in the GMM machine."""
        return self.gaussians_.shape

    @classmethod
    def from_hdf5(cls, hdf5):
        """Creates a new GMMMachine object from an `HDF5File` object."""
        try:
            version_major, version_minor = hdf5.get("meta_file_version").split(".")
            logger.debug(
                f"Reading a GMMStats HDF5 file of version {version_major}.{version_minor}"
            )
        except RuntimeError:
            version_major, version_minor = 0, 0
        if int(version_major) >= 1:
            if hdf5["meta_writer_class"] != str(cls):
                logger.warning(f"{hdf5['meta_writer_class']} is not {cls}.")
            n_gaussians = hdf5["n_gaussians"]
            gaussians = None  # TODO load the gaussians here
            ubm = None  # TODO load the UBM here
            self = cls(
                n_gaussians=n_gaussians,
                trainer=hdf5["trainer"],
                ubm=ubm,
                convergence_threshold=1e-5,
                max_fitting_steps=hdf5["max_fitting_steps"],
                random_state=0,
                initial_gaussians=gaussians,
                weights=hdf5["weights"],
                k_means_trainer=None,
                update_means=hdf5["update_means"],
                update_variances=hdf5["update_variances"],
                update_weights=hdf5["update_weights"],
            )
        else:  # Legacy file version
            logger.info("Loading a legacy HDF5 stats file.")
            n_gaussians = hdf5["m_n_gaussians"]
            g_means = []
            g_variances = []
            g_variance_thresholds = []
            for i in range(n_gaussians):
                g_means.append = hdf5[f"m_gaussians{i}"]["m_mean"]
                g_variances.append = hdf5[f"m_gaussians{i}"]["m_variance"]
                g_variance_thresholds.append = hdf5[f"m_gaussians{i}"][
                    "m_variance_thresholds"
                ]
            gaussians = Gaussians(
                means=g_means,
                variances=g_variances,
                variance_thresholds=g_variance_thresholds,
            )
            ubm = None  # TODO handle that. UBM was not part of the machine before...
            self = cls(
                n_gaussians=n_gaussians,
                ubm=ubm,
                initial_gaussians=gaussians,
                weights=hdf5["m_weights"],
            )
        return self

    def save(self, hdf5):
        """Saves the current statistsics in an `HDF5File` object."""
        hdf5["meta_file_version"] = "1.0"
        hdf5["meta_writer_class"] = str(self.__class__)
        hdf5["n_gaussians"] = self.n_gaussians
        hdf5["trainer"] = self.trainer
        hdf5["convergence_threshold"] = self.convergence_threshold
        hdf5["max_fitting_steps"] = self.max_fitting_steps
        hdf5["random_state"] = self.random_state
        hdf5["weights"] = self.weights
        hdf5["update_means"] = self.update_means
        hdf5["update_variances"] = self.update_variances
        hdf5["update_weights"] = self.update_weights
        hdf5["ubm"] = self.ubm  # TODO
        hdf5["gaussians_"] = self.gaussians_  # TODO

    def load(self, hdf5):
        """Overwrites the current statistics with those in an `HDF5File` object."""
        new_self = self.from_hdf5(hdf5)
        if new_self.shape != self.shape:
            logger.warning("Loaded GMMMachine from hdf5 with a different shape.")
            # self.resize(*new_self.shape)
        # TODO

    def __eq__(self, other):
        return np.array_equal(self.gaussians_, other.gaussians_)

    def is_similar_to(self, other, rtol=1e-5, atol=1e-8):
        """Returns True if `other` has the same gaussians (within a tolerance)."""
        return self.gaussians_.is_similar_to(
            other.gaussians_, rtol=rtol, atol=atol
        ) and np.allclose(self.weights, other.weights, rtol=rtol, atol=atol)

    def initialize_gaussians(
        self, data: "Union[np.ndarray[('n_samples', 'n_features'), float], None]" = None
    ):
        """Populates `gaussians_` with either k-means or the UBM values."""
        if self.trainer == "map":
            self.weights = self.ubm.weights.copy()
            self.gaussians_ = self.ubm.gaussians_.copy()
        else:
            if self.initial_gaussians is None:
                if data is None:
                    raise ValueError("Data is required when training with k-means.")
                logger.info("Initializing GMM with k-means.")
                kmeans_trainer = self.k_means_trainer or KMeansTrainer()
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

    def log_weighted_likelihood(
        self, data: "np.ndarray[('n_samples', 'n_features'), float]"
    ):
        """Returns the weighted log likelihood for each Gaussian for a set of data.

        Parameters
        ----------
        data
            Data to compute the log likelihood on.

        Returns
        -------
        array of shape (n_gaussians, n_samples)
            The weighted log likelihood of each sample of each Gaussian.
        """
        # TODO precompute those
        n_log_2pi = data.shape[-1] * np.log(2 * np.pi)
        # g_norm for each gaussian [array of shape (n_gaussians,)]
        g_norms = n_log_2pi + np.sum(np.log(self.gaussians_["variances"]), axis=-1)

        # Compute the likelihood for each data point on this Gaussian
        z = da.sum(
            da.power(data[None, :, :] - self.gaussians_["means"][:, None, :], 2)
            / self.gaussians_["variances"][:, None, :],
            axis=-1,
        )
        # Unweighted log likelihoods [array of shape (n_gaussians, n_samples)]
        l = -0.5 * (g_norms[:, None] + z)
        log_weighted_likelihood = self.log_weights[:, None] + l
        return log_weighted_likelihood

    def log_likelihood(self, data: "np.ndarray[('n_samples', 'n_features'), float]"):
        """Returns the current log likelihood for a set of data in this Machine.

        Parameters
        ----------
        data
            Data to compute the log likelihood on.

        Returns
        -------
        array of shape (n_samples)
            The log likelihood of each sample.
        """
        if data.ndim == 1:
            data = data.reshape((1, -1))

        # All likelihoods [array of shape (n_gaussians, n_samples)]
        log_weighted_likelihood = self.log_weighted_likelihood(data)

        def logaddexp_reduce(array, axis, keepdims):
            return np.logaddexp.reduce(
                array, axis=axis, keepdims=keepdims, initial=-np.inf
            )

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

    def acc_statistics(
        self,
        data: "np.ndarray[('n_samples', 'n_features'), float]",
        statistics: Union[GMMStats, None] = None,
    ):
        """Accumulates the statistics of GMMStats for a set of data.

        Parameters
        ----------
        data:
            The data to extract the statistics on the GMM.
        statistics:
            A GMMStats object that will accumulate the previous and current stats.
            Values are modified in-place AND returned. (or only returned if
            `statistics` is None)
        """
        # Ensure data is a series of samples (2D array)
        if data.ndim == 1:
            data = data.reshape(shape=(1, -1))

        # Allow the absence of previous statistics
        if statistics is None:
            statistics = GMMStats(self.n_gaussians, data.shape[-1])

        # Log weighted Gaussian likelihoods [array of shape (n_gaussians,n_samples)]
        log_weighted_likelihoods = self.log_weighted_likelihood(data)
        # Log likelihood [array of shape (n_samples,)]
        log_likelihood = self.log_likelihood(data)
        # Responsibility P [array of shape (n_gaussians, n_samples)]
        responsibility = da.exp(log_weighted_likelihoods - log_likelihood[None, :])

        # Accumulate

        # Total likelihood [float]
        statistics.log_likelihood += da.sum(log_likelihood)
        # Count of samples [int]
        statistics.t += data.shape[0]
        # Responsibilities [array of shape (n_gaussians,)]
        statistics.n += da.sum(responsibility, axis=-1)
        # p * x [array of shape (n_gaussians, n_samples, n_features)]
        px = da.multiply(responsibility[:, :, None], data[None, :, :])
        # First order stats [array of shape (n_gaussians, n_features)]
        statistics.sum_px += da.sum(px, axis=1)
        # Second order stats [array of shape (n_gaussians, n_features)]
        pxx = da.multiply(px[:, :, :], data[None, :, :])
        statistics.sum_pxx += da.sum(pxx, axis=1)

        return statistics

    def e_step(self, data: "np.ndarray[('n_samples', 'n_features'), float]"):
        """Expectation step of the e-m algorithm."""
        return self.acc_statistics(data)

    def m_step(
        self,
        stats: GMMStats,
        **kwargs,
    ):
        """Maximization step of the e-m algorithm."""
        self.m_step_func(
            self,
            statistics=stats,
            update_means=self.update_means,
            update_variances=self.update_variances,
            update_weights=self.update_weights,
            mean_var_update_threshold=self.mean_var_update_threshold,
            **kwargs,
        )

    def fit(self, X, y=None, **kwargs):
        """Trains the GMM on data until convergence or maximum step is reached."""
        if not hasattr(self, "gaussians_"):
            self.initialize_gaussians(X)

        average_output = 0
        logger.info("Training GMM...")
        step = 0
        while self.max_fitting_steps is None or step < self.max_fitting_steps:
            step += 1
            logger.info(
                f"Iteration {step:3d}"
                + (
                    f"/{self.max_fitting_steps:3d}"
                    if self.max_fitting_steps is not None
                    else ""
                )
            )

            average_output_previous = average_output
            stats = self.e_step(X)
            self.m_step(
                stats=stats,
            )

            # Note: Uses the stats from before m_step, leading to an additional m_step
            # (which is not bad because it will always converge)
            average_output = stats.log_likelihood / stats.t

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
        """Applies one iteration of GMM training."""
        if not hasattr(self, "gaussians_"):
            self.initialize_gaussians(X)

        stats = self.e_step(X)
        self.m_step(stats=stats)
        return self

    def transform(self, X, **kwargs):
        """Returns the statistics for `X`."""
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
    thresholded_n = da.clip(statistics.n, mean_var_update_threshold, None)

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
    """Updates a GMMMachine parameters using statistics adapted from a UBM."""
    if machine.ubm is None:
        raise ValueError("A machine used for MAP must have a UBM.")
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
        prior_norm_variances = (machine.ubm.variances + machine.ubm.means) - da.power(
            machine.means, 2
        )
        new_variances = (
            alpha[:, None] * statistics.sum_pxx / statistics.n[:, None]
            + (1 - alpha[:, None]) * (machine.ubm.variances + machine.ubm.means)
            - da.power(machine.means, 2)
        )
        machine.variances = da.where(
            statistics.n[:, None] < mean_var_update_threshold,
            prior_norm_variances,
            new_variances,
        )
