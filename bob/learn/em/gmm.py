#!/usr/bin/env python
# @author: Yannick Dayer <yannick.dayer@idiap.ch>
# @date: Fri 30 Jul 2021 10:06:47 UTC+02

"""This module provides classes and functions for the training and usage of GMM."""

import copy
import functools
import logging
import operator

from typing import Union

import dask
import dask.array as da
import numpy as np

from h5py import File as HDF5File
from sklearn.base import BaseEstimator

from .kmeans import KMeansMachine
from .utils import array_to_delayed_list, check_and_persist_dask_input

logger = logging.getLogger(__name__)

EPSILON = np.finfo(float).eps


def logaddexp_reduce(array, axis=0, keepdims=False):
    return np.logaddexp.reduce(
        array, axis=axis, keepdims=keepdims, initial=-np.inf
    )


def log_weighted_likelihood(data, machine):
    """Returns the weighted log likelihood for each Gaussian for a set of data.

    Parameters
    ----------
    data
        Data to compute the log likelihood on.
    machine
        The GMM machine.

    Returns
    -------
    array of shape (n_gaussians, n_samples)
        The weighted log likelihood of each sample of each Gaussian.
    """
    # Compute the likelihood for each data point on each Gaussian
    n_gaussians = len(machine.means)
    z = []
    for i in range(n_gaussians):
        temp = np.sum(
            (data - machine.means[i]) ** 2 / machine.variances[i], axis=-1
        )
        z.append(temp)
    z = np.vstack(z)
    ll = -0.5 * (machine.g_norms[:, None] + z)
    log_weighted_likelihoods = machine.log_weights[:, None] + ll
    return log_weighted_likelihoods


def reduce_loglikelihood(log_weighted_likelihoods):
    if isinstance(log_weighted_likelihoods, np.ndarray):
        log_likelihood = logaddexp_reduce(log_weighted_likelihoods)
    else:
        # Sum along gaussians axis (using logAddExp to prevent underflow)
        log_likelihood = da.reduction(
            x=log_weighted_likelihoods,
            chunk=logaddexp_reduce,
            aggregate=logaddexp_reduce,
            axis=0,
            dtype=float,
            keepdims=False,
        )
    return log_likelihood


def log_likelihood(data, machine):
    """Returns the current log likelihood for a set of data in this Machine.

    Parameters
    ----------
    data
        Data to compute the log likelihood on.
    machine
        The GMM machine.

    Returns
    -------
    array of shape (n_samples)
        The log likelihood of each sample.
    """
    data = np.atleast_2d(data)

    # All likelihoods [array of shape (n_gaussians, n_samples)]
    log_weighted_likelihoods = log_weighted_likelihood(
        data=data,
        machine=machine,
    )
    # Likelihoods of each sample on this machine. [array of shape (n_samples,)]
    ll_reduced = reduce_loglikelihood(log_weighted_likelihoods)
    return ll_reduced


def e_step(data, machine):
    """Expectation step of the e-m algorithm."""
    # Ensure data is a series of samples (2D array)
    data = np.atleast_2d(data)

    n_gaussians = len(machine.weights)

    # Allow the absence of previous statistics
    statistics = GMMStats(n_gaussians, data.shape[-1], like=data)

    # Log weighted Gaussian likelihoods [array of shape (n_gaussians,n_samples)]
    log_weighted_likelihoods = log_weighted_likelihood(
        data=data, machine=machine
    )

    # Log likelihood [array of shape (n_samples,)]
    log_likelihood = reduce_loglikelihood(log_weighted_likelihoods)

    # Responsibility P [array of shape (n_gaussians, n_samples)]
    responsibility = np.exp(log_weighted_likelihoods - log_likelihood[None, :])

    # Accumulate

    # Total likelihood [float]
    statistics.log_likelihood = log_likelihood.sum()
    # Count of samples [int]
    statistics.t = data.shape[0]
    # Responsibilities [array of shape (n_gaussians,)]
    statistics.n = responsibility.sum(axis=-1)
    sum_px, sum_pxx = [], []
    for i in range(n_gaussians):
        # p * x [array of shape (n_gaussians, n_samples, n_features)]
        px = responsibility[i, :, None] * data
        # First order stats [array of shape (n_gaussians, n_features)]
        # statistics.sum_px[i] = np.sum(px, axis=0)
        sum_px.append(np.sum(px, axis=0))
        # Second order stats [array of shape (n_gaussians, n_features)]
        # statistics.sum_pxx[i] = np.sum(px * data, axis=0)
        sum_pxx.append(np.sum(px * data, axis=0))

    statistics.sum_px = np.vstack(sum_px)
    statistics.sum_pxx = np.vstack(sum_pxx)

    return statistics


def m_step(
    statistics,
    machine,
):
    """Maximization step of the e-m algorithm."""
    m_step_func = map_gmm_m_step if machine.trainer == "map" else ml_gmm_m_step
    statistics = functools.reduce(operator.iadd, statistics)
    m_step_func(
        machine=machine,
        statistics=statistics,
        update_means=machine.update_means,
        update_variances=machine.update_variances,
        update_weights=machine.update_weights,
        mean_var_update_threshold=machine.mean_var_update_threshold,
        reynolds_adaptation=machine.map_relevance_factor is not None,
        alpha=machine.map_alpha,
        relevance_factor=machine.map_relevance_factor,
    )
    average_output = float(statistics.log_likelihood / statistics.t)
    return machine, average_output


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

    def __init__(
        self, n_gaussians: int, n_features: int, like=None, **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.n_gaussians = n_gaussians
        self.n_features = n_features
        self.log_likelihood = 0
        self.t = 0
        # create dask arrays if required
        kw = dict(like=like) if like is not None else {}
        self.n = np.zeros(shape=(self.n_gaussians,), dtype=float, **kw)
        self.sum_px = np.zeros(
            shape=(self.n_gaussians, self.n_features), dtype=float, **kw
        )
        self.sum_pxx = np.zeros(
            shape=(self.n_gaussians, self.n_features), dtype=float, **kw
        )

    def init_fields(
        self, log_likelihood=0.0, t=0, n=None, sum_px=None, sum_pxx=None
    ):
        """Initializes the statistics values to a defined value, or zero by default."""
        # The accumulated log likelihood of all samples
        self.log_likelihood = log_likelihood
        # The accumulated number of samples
        self.t = t
        # For each Gaussian, the accumulated sum of responsibilities, i.e. the sum of
        # P(gaussian_i|x)
        self.n = (
            np.zeros(shape=(self.n_gaussians,), dtype=float) if n is None else n
        )
        # For each Gaussian, the accumulated sum of responsibility times the sample
        self.sum_px = (
            np.zeros(shape=(self.n_gaussians, self.n_features), dtype=float)
            if sum_px is None
            else sum_px
        )
        # For each Gaussian, the accumulated sum of responsibility times the sample
        # squared
        self.sum_pxx = (
            np.zeros(shape=(self.n_gaussians, self.n_features), dtype=float)
            if sum_pxx is None
            else sum_pxx
        )

    def reset(self):
        """Sets all statistics to zero."""
        self.init_fields()

    @classmethod
    def from_hdf5(cls, hdf5):
        """Creates a new GMMStats object from an `HDF5File` object."""
        if isinstance(hdf5, str):
            hdf5 = HDF5File(hdf5, "r")
        try:
            version_major, version_minor = hdf5.attrs["file_version"].split(".")
            logger.debug(
                f"Reading a GMMStats HDF5 file of version {version_major}.{version_minor}"
            )
        except (KeyError, RuntimeError):
            version_major, version_minor = 0, 0
        if int(version_major) >= 1:
            if hdf5.attrs["writer_class"] != str(cls):
                logger.warning(f"{hdf5.attrs['writer_class']} is not {cls}.")
            self = cls(
                n_gaussians=hdf5["n_gaussians"][()],
                n_features=hdf5["n_features"][()],
            )
            self.log_likelihood = hdf5["log_likelihood"][()]
            self.t = hdf5["T"][()]
            self.n = hdf5["n"][...]
            self.sum_px = hdf5["sumPx"][...]
            self.sum_pxx = hdf5["sumPxx"][...]
        else:  # Legacy file version
            logger.info("Loading a legacy HDF5 stats file.")
            self = cls(
                n_gaussians=int(hdf5["n_gaussians"][()]),
                n_features=int(hdf5["n_inputs"][()]),
            )
            self.log_likelihood = hdf5["log_liklihood"][()]
            self.t = int(hdf5["T"][()])
            self.n = np.reshape(hdf5["n"], (self.n_gaussians,))
            self.sum_px = np.reshape(hdf5["sumPx"], (self.shape))
            self.sum_pxx = np.reshape(hdf5["sumPxx"], (self.shape))
        return self

    def save(self, hdf5):
        """Saves the current statistsics in an `HDF5File` object."""
        if isinstance(hdf5, str):
            hdf5 = HDF5File(hdf5, "w")
        hdf5.attrs["file_version"] = "1.0"
        hdf5.attrs["writer_class"] = str(self.__class__)
        hdf5["n_gaussians"] = self.n_gaussians
        hdf5["n_features"] = self.n_features
        hdf5["log_likelihood"] = float(self.log_likelihood)
        hdf5["T"] = int(self.t)
        hdf5["n"] = np.array(self.n)
        hdf5["sumPx"] = np.array(self.sum_px)
        hdf5["sumPxx"] = np.array(self.sum_pxx)

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
        if (
            self.n_gaussians != other.n_gaussians
            or self.n_features != other.n_features
        ):
            raise ValueError(
                "Statistics could not be added together (shape mismatch)"
            )
        new_stats = GMMStats(self.n_gaussians, self.n_features)
        new_stats.log_likelihood = self.log_likelihood + other.log_likelihood
        new_stats.t = self.t + other.t
        new_stats.n = self.n + other.n
        new_stats.sum_px = self.sum_px + other.sum_px
        new_stats.sum_pxx = self.sum_pxx + other.sum_pxx
        return new_stats

    def __iadd__(self, other):
        if (
            self.n_gaussians != other.n_gaussians
            or self.n_features != other.n_features
        ):
            raise ValueError(
                "Statistics could not be added together (shape mismatch)"
            )
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
            np.isclose(
                self.log_likelihood, other.log_likelihood, rtol=rtol, atol=atol
            )
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

    @property
    def nbytes(self):
        """The number of bytes used by the statistics n, sum_px, sum_pxx."""
        return self.n.nbytes + self.sum_px.nbytes + self.sum_pxx.nbytes


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

    * Maximum Likelihood Estimation (:ref:`MLE <mle>`, ML)

    The mixtures are initialized (with k-means by default). The means,
    variances, and weights of the mixtures are then trained on the data to
    increase the likelihood value. (:ref:`MLE <mle>`)

    * Maximum a Posteriori (:ref:`MAP <map>`)

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

    Attributes
    ----------
    means, variances, variance_thresholds
        Gaussians parameters.
    """

    def __init__(
        self,
        n_gaussians: int,
        trainer: str = "ml",
        ubm: "Union[GMMMachine, None]" = None,
        convergence_threshold: float = 1e-5,
        max_fitting_steps: Union[int, None] = 200,
        random_state: Union[int, np.random.RandomState] = 0,
        weights: "Union[np.ndarray[('n_gaussians',), float], None]" = None,  # noqa: F821
        k_means_trainer: Union[KMeansMachine, None] = None,
        update_means: bool = True,
        update_variances: bool = False,
        update_weights: bool = False,
        mean_var_update_threshold: float = EPSILON,
        map_alpha: float = 0.5,
        map_relevance_factor: Union[None, float] = 4,
        **kwargs,
    ):
        """
        Parameters
        ----------
        n_gaussians
            The number of gaussians to be represented by the machine.
        trainer
            `"ml"` for the maximum likelihood estimator method;
            `"map"` for the maximum a posteriori method. (MAP Requires `ubm`)
        ubm: GMMMachine
            Universal Background Model. GMMMachine Required for the MAP method.
        convergence_threshold
            The threshold value of likelihood difference between e-m steps used for
            stopping the training iterations.
        max_fitting_steps
            The number of e-m iterations to fit the GMM. Stop the training even when
            the convergence threshold isn't met.
        random_state
            Specifies a RandomState or a seed for reproducibility.
        weights
            The weight of each Gaussian. (defaults to `1/n_gaussians`)
        k_means_trainer
            Optional trainer for the k-means method, replacing the default one.
        update_means
            Update the Gaussians means at every m step.
        update_variances
            Update the Gaussians variances at every m step.
        update_weights
            Update the GMM weights at every m step.
        mean_var_update_threshold
            Threshold value used when updating the means and variances.
        map_alpha
            Ratio for MAP adaptation. Used when `trainer == "map"` and
            `relevance_factor is None`)
        map_relevance_factor
            Factor for the computation of alpha with Reynolds adaptation. (Used when
            `trainer == "map"`)
        """

        super().__init__(**kwargs)

        self.n_gaussians = n_gaussians
        self.trainer = trainer if trainer in ["ml", "map"] else "ml"
        self.m_step_func = (
            map_gmm_m_step if self.trainer == "map" else ml_gmm_m_step
        )
        if self.trainer == "map" and ubm is None:
            raise ValueError("A UBM is required for MAP GMM.")
        if ubm is not None and ubm.n_gaussians != self.n_gaussians:
            raise ValueError(
                "The UBM machine is not compatible with this machine."
            )
        self.ubm = ubm
        if max_fitting_steps is None and convergence_threshold is None:
            raise ValueError(
                "Either or both convergence_threshold and max_fitting_steps must be set"
            )
        self.convergence_threshold = convergence_threshold
        self.max_fitting_steps = max_fitting_steps
        self.random_state = random_state
        self.k_means_trainer = k_means_trainer
        self.update_means = update_means
        self.update_variances = update_variances
        self.update_weights = update_weights
        self.mean_var_update_threshold = mean_var_update_threshold
        self._means = None
        self._variances = None
        self._variance_thresholds = mean_var_update_threshold
        self._g_norms = None

        if self.ubm is not None:
            self.means = copy.deepcopy(self.ubm.means)
            self.variances = copy.deepcopy(self.ubm.variances)
            self.variance_thresholds = copy.deepcopy(
                self.ubm.variance_thresholds
            )
            self.weights = copy.deepcopy(self.ubm.weights)
        else:
            self.weights = np.full(
                (self.n_gaussians,),
                fill_value=(1 / self.n_gaussians),
                dtype=float,
            )
        if weights is not None:
            self.weights = weights
        self.map_alpha = map_alpha
        self.map_relevance_factor = map_relevance_factor

    @property
    def weights(self):
        """The weights of each Gaussian mixture."""
        return self._weights

    @weights.setter
    def weights(
        self, weights: "np.ndarray[('n_gaussians',), float]"  # noqa: F821
    ):  # noqa: F821
        self._weights = weights
        self._log_weights = np.log(self._weights)

    @property
    def means(self):
        """The means of each Gaussian."""
        if self._means is None:
            raise ValueError("GMMMachine means were never set.")
        return self._means

    @means.setter
    def means(
        self,
        means: "np.ndarray[('n_gaussians', 'n_features'), float]",  # noqa: F821
    ):  # noqa: F821
        self._means = means

    @property
    def variances(self):
        """The (diagonal) variances of the gaussians."""
        if self._variances is None:
            raise ValueError("GMMMachine variances were never set.")
        return self._variances

    @variances.setter
    def variances(
        self,
        variances: "np.ndarray[('n_gaussians', 'n_features'), float]",  # noqa: F821
    ):
        self._variances = np.maximum(self.variance_thresholds, variances)
        # Recompute g_norm for each gaussian [array of shape (n_gaussians,)]
        n_log_2pi = self._variances.shape[-1] * np.log(2 * np.pi)
        self._g_norms = n_log_2pi + np.log(self._variances).sum(axis=-1)

    @property
    def variance_thresholds(self):
        """Threshold below which variances are clamped to prevent precision losses."""
        if self._variance_thresholds is None:
            return EPSILON
        return self._variance_thresholds

    @variance_thresholds.setter
    def variance_thresholds(
        self,
        threshold: "Union[float, np.ndarray[('n_gaussians', 'n_features'), float]]",  # noqa: F821
    ):
        self._variance_thresholds = threshold
        if self._variances is not None:
            self.variances = np.maximum(threshold, self._variances)

    @property
    def g_norms(self):
        """Precomputed g_norms (depends on variances and feature shape)."""
        if self._g_norms is None:
            # Recompute g_norm for each gaussian [array of shape (n_gaussians,)]
            n_log_2pi = self.variances.shape[-1] * np.log(2 * np.pi)
            self._g_norms = n_log_2pi + np.log(self._variances).sum(axis=-1)
        return self._g_norms

    @property
    def log_weights(self):
        """Retrieve the logarithm of the weights."""
        return self._log_weights

    @property
    def shape(self):
        """Shape of the gaussians in the GMM machine."""
        return self.means.shape

    @classmethod
    def from_hdf5(cls, hdf5, ubm=None):
        """Creates a new GMMMachine object from an `HDF5File` object."""
        if isinstance(hdf5, str):
            hdf5 = HDF5File(hdf5, "r")
        try:
            version_major, version_minor = hdf5.attrs["file_version"].split(".")
            logger.debug(
                f"Reading a GMMMachine HDF5 file of version {version_major}.{version_minor}"
            )
        except (KeyError, RuntimeError):
            version_major, version_minor = 0, 0
        if int(version_major) >= 1:
            if hdf5.attrs["writer_class"] != str(cls):
                logger.warning(f"{hdf5.attrs['writer_class']} is not {cls}.")
            if hdf5["trainer"] == "map" and ubm is None:
                raise ValueError(
                    "The UBM is needed when loading a MAP machine."
                )
            self = cls(
                n_gaussians=hdf5["n_gaussians"][()],
                trainer=hdf5["trainer"][()],
                ubm=ubm,
                convergence_threshold=1e-5,
                max_fitting_steps=hdf5["max_fitting_steps"][()],
                weights=hdf5["weights"][...],
                k_means_trainer=None,
                update_means=hdf5["update_means"][()],
                update_variances=hdf5["update_variances"][()],
                update_weights=hdf5["update_weights"][()],
            )
            gaussians_group = hdf5["gaussians"]
            self.means = gaussians_group["means"][...]
            self.variances = gaussians_group["variances"][...]
            self.variance_thresholds = gaussians_group["variance_thresholds"][
                ...
            ]
        else:  # Legacy file version
            logger.info("Loading a legacy HDF5 machine file.")
            n_gaussians = hdf5["m_n_gaussians"][()]
            g_means = []
            g_variances = []
            g_variance_thresholds = []
            for i in range(n_gaussians):
                gaussian_group = hdf5[f"m_gaussians{i}"]
                g_means.append(gaussian_group["m_mean"][...])
                g_variances.append(gaussian_group["m_variance"][...])
                g_variance_thresholds.append(
                    gaussian_group["m_variance_thresholds"][...]
                )
            weights = np.reshape(hdf5["m_weights"], (n_gaussians,))
            self = cls(n_gaussians=n_gaussians, ubm=ubm, weights=weights)
            self.means = np.array(g_means).reshape(n_gaussians, -1)
            self.variances = np.array(g_variances).reshape(n_gaussians, -1)
            self.variance_thresholds = np.array(g_variance_thresholds).reshape(
                n_gaussians, -1
            )
        return self

    def load(self, hdf5):
        """Overwrites the current state with those in an `HDF5File` object."""
        new_self = self.from_hdf5(hdf5)
        self.__dict__.update(new_self.__dict__)

    def save(self, hdf5):
        """Saves the current statistics in an `HDF5File` object."""
        if isinstance(hdf5, str):
            hdf5 = HDF5File(hdf5, "w")
        hdf5.attrs["file_version"] = "1.0"
        hdf5.attrs["writer_class"] = str(self.__class__)
        hdf5["n_gaussians"] = self.n_gaussians
        hdf5["trainer"] = self.trainer
        hdf5["convergence_threshold"] = self.convergence_threshold
        hdf5["max_fitting_steps"] = self.max_fitting_steps
        hdf5["weights"] = self.weights
        hdf5["update_means"] = self.update_means
        hdf5["update_variances"] = self.update_variances
        hdf5["update_weights"] = self.update_weights
        gaussians_group = hdf5.create_group("gaussians")
        gaussians_group["means"] = self.means
        gaussians_group["variances"] = self.variances
        gaussians_group["variance_thresholds"] = self.variance_thresholds

    def __eq__(self, other):
        if self._means is None:
            return False

        return (
            np.allclose(self.means, other.means)
            and np.allclose(self.variances, other.variances)
            and np.allclose(self.variance_thresholds, other.variance_thresholds)
            and np.allclose(self.weights, other.weights)
        )

    def is_similar_to(self, other, rtol=1e-5, atol=1e-8):
        """Returns True if `other` has the same gaussians (within a tolerance)."""
        return (
            np.allclose(self.means, other.means, rtol=rtol, atol=atol)
            and np.allclose(
                self.variances, other.variances, rtol=rtol, atol=atol
            )
            and np.allclose(
                self.variance_thresholds,
                other.variance_thresholds,
                rtol=rtol,
                atol=atol,
            )
            and np.allclose(self.weights, other.weights, rtol=rtol, atol=atol)
        )

    def initialize_gaussians(
        self,
        data: "Union[np.ndarray[('n_samples', 'n_features'), float], None]" = None,  # noqa: F821
    ):
        """Populates gaussians parameters with either k-means or the UBM values."""
        if self.trainer == "map":
            self.means = copy.deepcopy(self.ubm.means)
            self.variances = copy.deepcopy(self.ubm.variances)
            self.variance_thresholds = copy.deepcopy(
                self.ubm.variance_thresholds
            )
            self.weights = copy.deepcopy(self.ubm.weights)
        else:
            logger.debug("GMM means was never set. Initializing with k-means.")
            if data is None:
                raise ValueError("Data is required when training with k-means.")
            logger.info("Initializing GMM with k-means.")
            kmeans_machine = self.k_means_trainer or KMeansMachine(
                self.n_gaussians,
                random_state=self.random_state,
            )
            kmeans_machine = kmeans_machine.fit(data)

            # Set the GMM machine's gaussians with the results of k-means
            self.means = copy.deepcopy(kmeans_machine.centroids_)
            logger.debug(
                "Estimating the variance and weights of each gaussian from kmeans."
            )
            (
                self.variances,
                self.weights,
            ) = kmeans_machine.get_variances_and_weights_for_each_cluster(data)
            logger.debug("Done.")

    def log_weighted_likelihood(
        self,
        data: "np.ndarray[('n_samples', 'n_features'), float]",  # noqa: F821
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
        return log_weighted_likelihood(
            data=data,
            machine=self,
        )

    def log_likelihood(
        self,
        data: "np.ndarray[('n_samples', 'n_features'), float]",  # noqa: F821
    ):
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
        return log_likelihood(
            data=data,
            machine=self,
        )

    def fit(self, X, y=None):
        """Trains the GMM on data until convergence or maximum step is reached."""

        input_is_dask, X = check_and_persist_dask_input(X)

        if self._means is None:
            self.initialize_gaussians(X)
        else:
            logger.debug("GMM means already set. Initialization was not run!")

        if self._variances is None:
            logger.warning(
                "Variances were not defined before fit. Using variance=1"
            )
            self.variances = np.ones_like(self.means)

        X = array_to_delayed_list(X, input_is_dask)

        average_output = 0
        logger.info("Training GMM...")
        step = 0
        while self.max_fitting_steps is None or step < self.max_fitting_steps:
            step += 1
            logger.info(
                f"Iteration {step:3d}"
                + (
                    f"/{self.max_fitting_steps:3d}"
                    if self.max_fitting_steps
                    else ""
                )
            )

            average_output_previous = average_output

            # compute the e-m steps
            if input_is_dask:
                stats = [
                    dask.delayed(e_step)(
                        data=xx,
                        machine=self,
                    )
                    for xx in X
                ]
                new_machine, average_output = dask.compute(
                    dask.delayed(m_step)(stats, self)
                )[0]
                for attr in ["weights", "means", "variances"]:
                    setattr(self, attr, getattr(new_machine, attr))
            else:
                stats = [
                    e_step(
                        data=X,
                        machine=self,
                    )
                ]
                _, average_output = m_step(stats, self)

            logger.debug(f"log likelihood = {average_output}")
            if step > 1:
                convergence_value = abs(
                    (average_output_previous - average_output)
                    / average_output_previous
                )
                logger.debug(
                    f"convergence val = {convergence_value} and threshold = {self.convergence_threshold}"
                )

                # Terminates if converged (and likelihood computation is set)
                if (
                    self.convergence_threshold is not None
                    and convergence_value <= self.convergence_threshold
                ):
                    logger.info(
                        "Reached convergence threshold. Training stopped."
                    )
                    break

        else:
            logger.info(
                "Reached maximum step. Training stopped without convergence."
            )
        return self

    def acc_stats(self, X):
        """Returns the statistics for `X`."""
        # we need this because sometimes the transform function gets overridden
        return e_step(data=X, machine=self)

    def transform(self, X):
        """Returns the statistics for `X`."""
        return self.stats_per_sample(X)

    def stats_per_sample(self, X):
        return [e_step(data=xx, machine=self) for xx in X]


def ml_gmm_m_step(
    machine: GMMMachine,
    statistics: GMMStats,
    update_means=True,
    update_variances=False,
    update_weights=False,
    mean_var_update_threshold=EPSILON,
    **kwargs,
):
    """Updates a gmm machine parameter according to the e-step statistics."""
    logger.debug("ML GMM Trainer m-step")

    # Threshold the low n to prevent divide by zero
    thresholded_n = np.clip(statistics.n, mean_var_update_threshold, None)

    # Update weights if requested
    # (Equation 9.26 of Bishop, "Pattern recognition and machine learning", 2006)
    if update_weights:
        logger.debug("Update weights.")
        machine.weights = thresholded_n / statistics.t

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
        machine.variances = statistics.sum_pxx / thresholded_n[
            :, None
        ] - np.power(machine.means, 2)


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
        # Apply threshold to prevent divide by zero below
        n_threshold = np.where(
            statistics.n < mean_var_update_threshold,
            mean_var_update_threshold,
            statistics.n,
        )
        # n_threshold = np.full(statistics.n.shape, fill_value=mean_var_update_threshold)
        # n_threshold[statistics.n > mean_var_update_threshold] = statistics.n[
        #     statistics.n > mean_var_update_threshold
        # ]
        new_means = np.multiply(
            alpha[:, None],
            (statistics.sum_px / n_threshold[:, None]),
        ) + np.multiply((1 - alpha[:, None]), machine.ubm.means)
        machine.means = np.where(
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
        ) - np.power(machine.means, 2)
        new_variances = (
            alpha[:, None] * statistics.sum_pxx / statistics.n[:, None]
            + (1 - alpha[:, None]) * (machine.ubm.variances + machine.ubm.means)
            - np.power(machine.means, 2)
        )
        machine.variances = np.where(
            statistics.n[:, None] < mean_var_update_threshold,
            prior_norm_variances,
            new_variances,
        )
