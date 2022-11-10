#!/usr/bin/env python
# @author: Yannick Dayer <yannick.dayer@idiap.ch>
# @date: Tue 27 Jul 2021 11:04:10 UTC+02

import logging

from typing import Union

import dask
import dask.array as da
import dask.delayed
import numpy as np
import scipy.spatial.distance

from dask_ml.cluster.k_means import k_init
from sklearn.base import BaseEstimator

from .utils import array_to_delayed_list, check_and_persist_dask_input

logger = logging.getLogger(__name__)


def get_centroids_distance(x: np.ndarray, means: np.ndarray) -> np.ndarray:
    """Returns the distance values between x and each cluster's centroid.

    The returned values are squared Euclidean distances.

    Parameters
    ----------
    x: ndarray of shape (n_samples, n_features)
        A series of data points.
    means: ndarray of shape (n_clusters, n_features)
        The centroids.

    Returns
    -------
    distances: ndarray of shape (n_clusters, n_samples)
        For each cluster, the squared Euclidian distance (or distances) to x.
    """
    x = np.atleast_2d(x)
    if isinstance(x, da.Array):
        distances = []
        for i in range(means.shape[0]):
            distances.append(np.sum((means[i] - x) ** 2, axis=-1))
        return da.vstack(distances)
    else:
        return scipy.spatial.distance.cdist(means, x, metric="sqeuclidean")


def get_closest_centroid_index(centroids_dist: np.ndarray) -> np.ndarray:
    """Returns the index of the closest cluster mean to x.

    Parameters
    ----------
    centroids_dist: ndarray of shape (n_clusters, n_samples)
        The squared Euclidian distance (or distances) to each cluster mean.

    Returns
    -------
    closest_centroid_indices: ndarray of shape (n_samples,)
        The index of the closest cluster mean to x.
    """
    return np.argmin(centroids_dist, axis=0)


def e_step(data, means):
    """Computes the zero-th and first order statistics and average min distance
    for each data point.

    Parameters
    ----------
    data : array-like, shape (n_samples, n_features)
        The data.

    means : array-like, shape (n_clusters, n_features)
        The cluster centers.


    Returns
    -------
    zeroeth_order_statistics : array-like, shape (n_samples,)
        The zero-th order statistics.
    first_order_statistics : array-like, shape (n_samples, n_clusters)
        The first order statistics.
    avg_min_dist : float
    """
    n_clusters = len(means)
    distances = get_centroids_distance(data, means)
    closest_k_indices = get_closest_centroid_index(distances)
    zeroeth_order_statistics = np.bincount(
        closest_k_indices, minlength=n_clusters
    )
    # Compute first_order_statistics in a memory efficient way
    first_order_statistics = np.zeros((n_clusters, data.shape[1]))
    for i in range(n_clusters):
        first_order_statistics[i] = np.sum(data[closest_k_indices == i], axis=0)
    min_distance = np.min(distances, axis=0)
    average_min_distance = min_distance.mean()
    return (
        zeroeth_order_statistics,
        first_order_statistics,
        average_min_distance,
    )


def m_step(stats, n_samples):
    """Computes the cluster centers and average minimum distance.

    Parameters
    ----------
    stats : list
        A list which contains the results of the :any:`e_step` function applied
        on each chunk of data.
    n_samples : int
        The total number of samples.

    Returns
    -------
    means : array-like, shape (n_clusters, n_features)
        The cluster centers.
    avg_min_dist : float
        The average minimum distance.
    """
    (
        zeroeth_order_statistics,
        first_order_statistics,
        average_min_distance,
    ) = (0, 0, 0)
    for zeroeth_, first_, average_ in stats:
        zeroeth_order_statistics += zeroeth_
        first_order_statistics += first_
        average_min_distance += average_
    average_min_distance /= n_samples

    means = first_order_statistics / zeroeth_order_statistics[:, None]
    return means, average_min_distance


def accumulate_indices_means_vars(data, means):
    """Accumulates statistics needed to compute weights and variances of the clusters."""
    n_clusters, n_features = len(means), data.shape[1]
    dist = get_centroids_distance(data, means)
    closest_centroid_indices = get_closest_centroid_index(dist)
    # the means_sum and variances_sum must be initialized with zero here since
    # they get accumulated in the next function
    means_sum = np.zeros((n_clusters, n_features), like=data)
    variances_sum = np.zeros((n_clusters, n_features), like=data)
    for i in range(n_clusters):
        means_sum[i] = np.sum(data[closest_centroid_indices == i], axis=0)
    for i in range(n_clusters):
        variances_sum[i] = np.sum(
            data[closest_centroid_indices == i] ** 2, axis=0
        )
    return closest_centroid_indices, means_sum, variances_sum


def reduce_indices_means_vars(stats):
    """Computes weights and variances of the clusters given the statistics."""
    closest_centroid_indices = [s[0] for s in stats]
    means_sum = [s[1] for s in stats]
    variances_sum = [s[2] for s in stats]

    closest_centroid_indices = np.concatenate(closest_centroid_indices, axis=0)
    means_sum = np.sum(means_sum, axis=0)
    variances_sum = np.sum(variances_sum, axis=0)

    n_clusters = len(means_sum)
    weights_count = np.bincount(closest_centroid_indices, minlength=n_clusters)
    weights = weights_count / weights_count.sum()
    means = means_sum / weights_count[:, None]
    variances = (variances_sum / weights_count[:, None]) - (means**2)

    return variances, weights


class KMeansMachine(BaseEstimator):
    """Stores the k-means clusters parameters (centroid of each cluster).

    Allows the clustering of data with the ``fit`` method.

    The training works in two phases:
        - An initialization (setting the initial values of the centroids)
        - An e-m loop reducing the total distance between the data points and their
          closest centroid.

    The initialization can use an iterative process to find the best set of
    coordinates, use random starting points, or take specified coordinates. The
    ``init_method`` parameter specifies which of these behavior is considered.

    Attributes
    ----------
    centroids_: ndarray of shape (n_clusters, n_features)
        The current clusters centroids. Available after fitting.
    """

    def __init__(
        self,
        n_clusters: int,
        init_method: Union[str, np.ndarray] = "k-means||",
        convergence_threshold: float = 1e-5,
        max_iter: int = 20,
        random_state: Union[int, np.random.RandomState] = 0,
        init_max_iter: Union[int, None] = 5,
        oversampling_factor: float = 2,
        **kwargs,
    ) -> None:
        """
        Parameters
        ----------
        n_clusters: int
            The number of represented clusters.
        init_method:
            One of: "random", "k-means++", or "k-means||", or an array with the wanted
            starting values of the centroids.
        max_iter:
            The maximum number of iterations for the e-m part.
        random_state:
            A seed or RandomState used for the initialization part.
        init_max_iter:
            The maximum number of iterations for the initialization part.
        """

        super().__init__(**kwargs)

        if n_clusters < 1:
            raise ValueError("The Number of cluster should be greater thant 0.")
        self.n_clusters = n_clusters
        self.init_method = init_method
        self.convergence_threshold = convergence_threshold
        self.max_iter = max_iter
        self.random_state = random_state
        self.init_max_iter = init_max_iter
        self.oversampling_factor = oversampling_factor
        self.average_min_distance = np.inf
        self.zeroeth_order_statistics = None
        self.first_order_statistics = None
        self.centroids_ = None

    @property
    def means(self) -> np.ndarray:
        """An alias for `centroids_`."""
        return self.centroids_

    @means.setter
    def means(self, value: np.ndarray):
        self.centroids_ = value

    def __eq__(self, obj) -> bool:
        return self.is_similar_to(obj, r_epsilon=0, a_epsilon=0)

    def is_similar_to(self, obj, r_epsilon=1e-05, a_epsilon=1e-08) -> bool:
        if self.centroids_ is not None and obj.centroids_ is not None:
            return np.allclose(
                self.centroids_, obj.centroids_, rtol=r_epsilon, atol=a_epsilon
            )
        else:
            logger.warning(
                "KMeansMachine `centroids_` was not set. You should call 'fit' first."
            )
            return False

    def get_variances_and_weights_for_each_cluster(self, data: np.ndarray):
        """Returns the clusters variance and weight for data clustered by the machine.

        For each cluster, finds the subset of the samples that is closest to that
        centroid, and calculates:
        1) the variance of that subset (the cluster variance)
        2) the proportion of samples represented by that subset (the cluster weight)

        Parameters
        ----------
        data:
            The data to compute the variance of.

        Returns
        -------
        Tuple of arrays:
            variances: ndarray of shape (n_clusters, n_features)
                For each cluster, the variance in each dimension of the data.
            weights: ndarray of shape (n_clusters, )
                Weight (proportion of quantity of data point) of each cluster.
        """
        input_is_dask, data = check_and_persist_dask_input(data)
        data = array_to_delayed_list(data, input_is_dask)

        if input_is_dask:
            stats = [
                dask.delayed(accumulate_indices_means_vars)(
                    xx, means=self.centroids_
                )
                for xx in data
            ]
            variances, weights = dask.compute(
                dask.delayed(reduce_indices_means_vars)(stats)
            )[0]
        else:
            # Accumulate
            stats = accumulate_indices_means_vars(data, self.centroids_)
            # Reduce
            variances, weights = reduce_indices_means_vars([stats])

        return variances, weights

    def initialize(self, data: np.ndarray):
        """Assigns the means to an initial value using a specified method or randomly."""
        logger.debug("k-means initialization")
        logger.debug(f"Initializing k-means means with '{self.init_method}'.")
        # k_init requires da.Array as input.
        logger.debug("Transform k-means data to dask array")
        data = da.array(data)
        data.rechunk(1, data.shape[-1])  # Prevents issue with large arrays.
        logger.debug("Get k-means centroids")
        self.centroids_ = k_init(
            X=data,
            n_clusters=self.n_clusters,
            init=self.init_method,
            random_state=self.random_state,
            max_iter=self.init_max_iter,
            oversampling_factor=self.oversampling_factor,
        )
        logger.debug("End of k-means initialization")

    def fit(self, X, y=None):
        """Fits this machine on data samples."""

        input_is_dask, X = check_and_persist_dask_input(X)

        logger.debug("Initializing trainer.")
        self.initialize(data=X)

        logger.debug("Get the number of samples")
        n_samples = len(X)

        logger.debug("Transform X array to delayed list")
        X = array_to_delayed_list(X, input_is_dask)

        logger.info("Training k-means.")
        distance = np.inf
        step = 0
        while self.max_iter is None or step < self.max_iter:
            step += 1
            logger.info(
                f"Iteration {step:3d}"
                + (f"/{self.max_iter:3d}" if self.max_iter else "")
            )
            distance_previous = distance

            # compute the e-m steps
            if input_is_dask:
                stats = [
                    dask.delayed(e_step)(xx, means=self.centroids_) for xx in X
                ]
                self.centroids_, self.average_min_distance = dask.compute(
                    dask.delayed(m_step)(stats, n_samples)
                )[0]
            else:
                stats = [e_step(X, means=self.centroids_)]
                self.centroids_, self.average_min_distance = m_step(
                    stats, n_samples
                )

            distance = self.average_min_distance

            logger.debug(
                f"Average minimal squared Euclidean distance = {distance}"
            )

            if step > 1:
                convergence_value = abs(
                    (distance_previous - distance) / distance_previous
                )
                logger.debug(
                    f"Convergence value = {convergence_value} and threshold is {self.convergence_threshold}"
                )

                # Terminates if converged (and threshold is set)
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

    def transform(self, X):
        """Returns all the distances between the data and each cluster's mean.

        Parameters
        ----------
        X: ndarray of shape (n_samples, n_features)
            Series of data points.

        Returns
        -------
        distances: ndarray of shape (n_clusters, n_samples)
            For each mean, for each point, the squared Euclidian distance between them.
        """
        return get_centroids_distance(X, self.centroids_)

    def predict(self, X):
        """Returns the labels of the closest cluster centroid to the data.

        Parameters
        ----------
        X: ndarray of shape (n_samples, n_features)
            Series of data points.

        Returns
        -------
        indices: ndarray of shape (n_samples)
            The indices of the closest cluster for each data point.
        """
        return get_closest_centroid_index(
            get_centroids_distance(X, self.centroids_)
        )
