#!/usr/bin/env python
# @author: Yannick Dayer <yannick.dayer@idiap.ch>
# @date: Tue 27 Jul 2021 11:04:10 UTC+02

import logging
from typing import Union
from typing import Tuple

import numpy as np
import dask.array as da
from dask_ml.cluster.k_means import k_init
from sklearn.base import BaseEstimator

logger = logging.getLogger(__name__)


class KMeansMachine(BaseEstimator):
    """Stores the k-means clusters parameters (centroid of each cluster).

    Allows the clustering of data with the ``fit`` method.

    Parameters
    ----------
    n_clusters: int
        The number of represented clusters.

    Attributes
    ----------
    centroids_: ndarray of shape (n_clusters, n_features)
        The current clusters centroids. Available after fitting.

    Example
    -------
    >>> data = dask.array.array([[0,-1,0],[-1,1,1],[3,2,1],[2,2,1],[1,0,2]])
    >>> machine = KMeansMachine(2).fit(data)
    >>> machine.centroids_.compute()
    ... array([[0. , 0. , 1. ],
    ...        [2.5, 2. , 1. ]])
    """

    def __init__(
        self,
        n_clusters: int,
        convergence_threshold: float = 1e-5,
        random_state: Union[int, da.random.RandomState] = 0,
    ) -> None:
        if n_clusters < 1:
            raise ValueError("The Number of cluster should be greater thant 0.")
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.convergence_threshold = convergence_threshold

    def get_centroids_distance(self, x: da.Array) -> da.Array:
        """Returns the distance values between x and each cluster's centroid.

        The returned values are squared Euclidean distances.

        Parameters
        ----------
        x: ndarray of shape (n_features,) or (n_samples, n_features)
            One data point, or a series of data points.

        Returns
        -------
        distances: ndarray of shape (n_clusters,) or (n_clusters, n_samples)
            For each cluster, the squared Euclidian distance (or distances) to x.
        """
        return da.sum((self.centroids_[:, None] - x[None, :]) ** 2, axis=-1)

    def get_closest_centroid(self, x: da.Array) -> Tuple[int, float]:
        """Returns the closest mean's index and squared Euclidian distance to x."""
        dists = self.get_centroids_distance(x)
        min_id = da.argmin(dists, axis=0)
        min_dist = dists[min_id]
        return min_id, min_dist

    def get_closest_centroid_index(self, x: da.Array) -> da.Array:
        """Returns the index of the closest cluster mean to x."""
        return da.argmin(self.get_centroids_distance(x), axis=0)

    def get_min_distance(self, x: da.Array) -> da.Array:
        """Returns the smallest distance between that point and the clusters centroids.

        For each point in x, the minimum distance to each cluster's mean is returned.

        The returned values are squared Euclidean distances.
        """
        return da.min(self.get_centroids_distance(x), axis=0)

    def __eq__(self, obj) -> bool:
        if hasattr(self, "centroids_") and hasattr(obj, "centroids_"):
            return da.allclose(self.centroids_, obj.centroids_, rtol=0, atol=0)
        else:
            raise ValueError("centroids_ was not set. You should call 'fit' first.")

    def is_similar_to(self, obj, r_epsilon=1e-05, a_epsilon=1e-08) -> bool:
        if hasattr(self, "centroids_") and hasattr(obj, "centroids_"):
            return da.allclose(
                self.centroids_, obj.centroids_, rtol=r_epsilon, atol=a_epsilon
            )
        else:
            raise ValueError("centroids_ was not set. You should call 'fit' first.")

    def get_variances_and_weights_for_each_cluster(self, data: da.Array):
        """Returns the clusters variance and weight for data clustered by the machine.

        For each cluster, finds the subset of the samples that is closest to that
        centroid, and calculates:
        1) the variance of that subset (the cluster variance)
        2) the proportion of samples represented by that subset (the cluster weight)

        Parameters
        ----------
        data: dask.array
            The data to compute the variance of.

        Returns
        -------
        Tuple of arrays:
            variances: ndarray of shape (n_clusters, n_features)
                For each cluster, the variance in each dimension of the data.
            weights: ndarray of shape (n_clusters, )
                Weight (proportion of quantity of data point) of each cluster.
        """
        n_cluster = self.n_clusters
        closest_centroid_indices = self.get_closest_centroid_index(data)
        weights_count = da.bincount(closest_centroid_indices, minlength=n_cluster)
        weights = weights_count / weights_count.sum()

        # Accumulate
        means_sum = da.sum(
            da.eye(n_cluster)[closest_centroid_indices][:, :, None] * data[:, None],
            axis=0,
        )
        variances_sum = da.sum(
            da.eye(n_cluster)[closest_centroid_indices][:, :, None]
            * (data[:, None] ** 2),
            axis=0,
        )

        # Reduce
        means = means_sum / weights_count[:, None]
        variances = (variances_sum / weights_count[:, None]) - (means ** 2)

        return variances, weights

    def fit(self, X, y=None, trainer=None):
        """Fits this machine with a k-means trainer.

        The default trainer (when None is given) uses k-means|| for init, then uses e-m
        until it converges or the limit number of iterations is reached.
        """
        if trainer is None:
            logger.info("Using default k-means trainer.")
            trainer = KMeansTrainer(init_method="k-means||", random_state=self.random_state)

        logger.debug(f"Initializing trainer.")
        trainer.initialize(
            machine=self,
            data=X,
        )

        logger.info("Training k-means.")
        distance = np.inf
        for step in range(trainer.max_iter):
            logger.info(f"Iteration {step:3d}/{trainer.max_iter}")
            distance_previous = distance
            trainer.e_step(machine=self, data=X)
            trainer.m_step(machine=self, data=X)

            distance = trainer.compute_likelihood(self)

            # logger.info(f"Average squared Euclidean distance = {distance.compute()}")

            if step > 0:
                convergence_value = abs(
                    (distance_previous - distance) / distance_previous
                )
                # logger.info(f"Convergence value = {convergence_value.compute()}")

                # Terminates if converged (and threshold is set)
                if (
                    self.convergence_threshold is not None
                    and convergence_value <= self.convergence_threshold
                ):
                    logger.info("Stopping Training: Convergence threshold met.")
                    return self
        logger.info("Stopping Training: Iterations limit reached.")
        return self

    def partial_fit(self, X, y=None, trainer=None):
        if trainer is None:
            logger.info("Using default k-means trainer.")
            trainer = KMeansTrainer(init_method="k-means||")
        if not hasattr(self, "means_"):
            logger.debug(f"First call of 'partial_fit'. Initializing trainer.")
            trainer.initialize(
                machine=self,
                data=X,
            )
        for step in range(trainer.max_iter):
            logger.info(f"Iteration = {step:3d}/{trainer.max_iter}")
            distance_previous = distance
            trainer.e_step(machine=self, data=X)
            trainer.m_step(machine=self, data=X)

            distance = trainer.compute_likelihood(self)

            logger.info(f"Average squared Euclidean distance = {distance}")

            convergence_value = abs((distance_previous - distance) / distance_previous)
            logger.info(f"Convergence value = {convergence_value}")

            # Terminates if converged (and threshold is set)
            if (
                self.convergence_threshold is not None
                and convergence_value <= self.convergence_threshold
            ):
                logger.info("Stopping Training: Convergence threshold met.")
                return self
        logger.info("Stopping Training: Iterations limit reached.")
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
        return self.get_centroids_distance(X)

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
        return self.get_closest_centroid_index(X)


class KMeansTrainer:
    """E-M Trainer that applies k-means on a KMeansMachine.

    This trainer works in two phases:
        - An initialization (setting the initial values of the centroids)
        - An e-m loop reducing the total distance between the data points and their
          closest centroid.

    The initialization can use an iterative process to find the best set of
    coordinates, use random starting points, or take specified coordinates. The
    ``init_method`` parameter specifies which of these behavior is considered.

    Parameters
    ----------
    init_method:
        One of: "random", "k-means++", or "k-means||", or an array with the wanted
        starting values of the centroids.
    init_max_iter:
        The maximum number of iterations for the initialization part.
    random_state:
        A seed or RandomState used for the initialization part.
    max_iter:
        The maximum number of iterations for the e-m part.
    """

    def __init__(
        self,
        init_method: Union[str, da.Array] = "k-means||",
        init_max_iter: Union[int, None] = None,
        random_state: Union[int, da.random.RandomState] = 0,
        max_iter: int = 20,
    ):
        self.init_method = init_method
        self.average_min_distance = None
        self.zeroeth_order_statistics = None
        self.first_order_statistics = None
        self.max_iter = max_iter
        self.init_max_iter = init_max_iter
        self.random_state = random_state

    def initialize(
        self,
        machine: KMeansMachine,
        data: da.Array,
    ):
        """Assigns the means to an initial value using a specified method or randomly."""
        logger.debug(f"Initializing k-means means with '{self.init_method}'.")
        data = da.array(data)
        machine.centroids_ = k_init(
            X=data,
            n_clusters=machine.n_clusters,
            init=self.init_method,
            random_state=self.random_state,
            max_iter=self.init_max_iter,
        )

    def e_step(self, machine: KMeansMachine, data: da.Array):
        data = da.array(data)
        closest_centroid_indices = machine.get_closest_centroid_index(data)
        # Number of data points in each cluster
        self.zeroeth_order_statistics = da.bincount(
            closest_centroid_indices, minlength=machine.n_clusters
        )
        # Sum of data points coordinates in each cluster
        self.first_order_statistics = da.sum(
            da.eye(machine.n_clusters)[closest_centroid_indices][:, :, None]
            * data[:, None],
            axis=0,
        )
        self.average_min_distance = machine.get_min_distance(data).mean()

    def m_step(self, machine: KMeansMachine, data: da.Array):
        machine.centroids_ = (
            self.first_order_statistics / self.zeroeth_order_statistics[:, None]
        ).persist()

    def compute_likelihood(self, machine: KMeansMachine):
        if self.average_min_distance is None:
            logger.error("compute_likelihood should be called after e_step.")
            return 0
        return self.average_min_distance

    def copy(self):
        new_trainer = KMeansTrainer()
        new_trainer.average_min_distance = self.average_min_distance
        new_trainer.zeroeth_order_statistics = self.zeroeth_order_statistics
        new_trainer.first_order_statistics = self.first_order_statistics
        return new_trainer

    def reset_accumulators(self, machine: KMeansMachine):
        self.average_min_distance = 0
        self.zeroeth_order_statistics = da.zeros((machine.n_clusters,), dtype="float64")
        self.first_order_statistics = da.zeros(
            (machine.n_clusters, machine.n_dims), dtype="float64"
        )
