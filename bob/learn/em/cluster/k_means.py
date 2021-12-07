#!/usr/bin/env python
# @author: Yannick Dayer <yannick.dayer@idiap.ch>
# @date: Tue 27 Jul 2021 11:04:10 UTC+02

import logging
from typing import Tuple
from typing import Union

import dask.array as da
import numpy as np
from dask_ml.cluster.k_means import k_init
from sklearn.base import BaseEstimator

logger = logging.getLogger(__name__)


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
        init_method: Union[str, np.ndarray] = "k-means||",
        convergence_threshold: float = 1e-5,
        max_iter: int = 20,
        random_state: Union[int, np.random.RandomState] = 0,
        init_max_iter: Union[int, None] = None,
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

        if n_clusters < 1:
            raise ValueError("The Number of cluster should be greater thant 0.")
        self.n_clusters = n_clusters
        self.init_method = init_method
        self.convergence_threshold = convergence_threshold
        self.max_iter = max_iter
        self.random_state = random_state
        self.init_max_iter = init_max_iter
        self.average_min_distance = np.inf
        self.zeroeth_order_statistics = None
        self.first_order_statistics = None

    def get_centroids_distance(self, x: np.ndarray) -> np.ndarray:
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
        return np.sum((self.centroids_[:, None] - x[None, :]) ** 2, axis=-1)

    def get_closest_centroid(self, x: np.ndarray) -> Tuple[int, float]:
        """Returns the closest mean's index and squared Euclidian distance to x."""
        dists = self.get_centroids_distance(x)
        min_id = np.argmin(dists, axis=0)
        min_dist = dists[min_id]
        return min_id, min_dist

    def get_closest_centroid_index(self, x: np.ndarray) -> np.ndarray:
        """Returns the index of the closest cluster mean to x."""
        return np.argmin(self.get_centroids_distance(x), axis=0)

    def get_min_distance(self, x: np.ndarray) -> np.ndarray:
        """Returns the smallest distance between that point and the clusters centroids.

        For each point in x, the minimum distance to each cluster's mean is returned.

        The returned values are squared Euclidean distances.
        """
        return np.min(self.get_centroids_distance(x), axis=0)

    def __eq__(self, obj) -> bool:
        if hasattr(self, "centroids_") and hasattr(obj, "centroids_"):
            return np.allclose(self.centroids_, obj.centroids_, rtol=0, atol=0)
        else:
            raise ValueError("centroids_ was not set. You should call 'fit' first.")

    def is_similar_to(self, obj, r_epsilon=1e-05, a_epsilon=1e-08) -> bool:
        if hasattr(self, "centroids_") and hasattr(obj, "centroids_"):
            return np.allclose(
                self.centroids_, obj.centroids_, rtol=r_epsilon, atol=a_epsilon
            )
        else:
            raise ValueError("centroids_ was not set. You should call 'fit' first.")

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
        n_cluster = self.n_clusters
        closest_centroid_indices = self.get_closest_centroid_index(data)
        weights_count = np.bincount(closest_centroid_indices, minlength=n_cluster)
        weights = weights_count / weights_count.sum()

        # FIX for `too many indices for array` error if using `np.eye(n_cluster)` alone:
        dask_compatible_eye = np.eye(n_cluster) * np.array(1, like=data)

        # Accumulate
        means_sum = np.sum(
            dask_compatible_eye[closest_centroid_indices][:, :, None] * data[:, None],
            axis=0,
        )
        variances_sum = np.sum(
            dask_compatible_eye[closest_centroid_indices][:, :, None]
            * (data[:, None] ** 2),
            axis=0,
        )

        # Reduce
        means = means_sum / weights_count[:, None]
        variances = (variances_sum / weights_count[:, None]) - (means ** 2)

        return variances, weights

    def initialize(self, data: np.ndarray):
        """Assigns the means to an initial value using a specified method or randomly."""
        logger.debug(f"Initializing k-means means with '{self.init_method}'.")
        # k_init requires da.Array as input.
        data = da.array(data)
        self.centroids_ = k_init(
            X=data,
            n_clusters=self.n_clusters,
            init=self.init_method,
            random_state=self.random_state,
            max_iter=self.init_max_iter,
        )

    def e_step(self, data: np.ndarray):
        closest_k_indices = self.get_closest_centroid_index(data)
        # Number of data points in each cluster
        self.zeroeth_order_statistics = np.bincount(
            closest_k_indices, minlength=self.n_clusters
        )
        # Sum of data points coordinates in each cluster
        self.first_order_statistics = np.sum(
            np.eye(self.n_clusters)[closest_k_indices][:, :, None] * data[:, None],
            axis=0,
        )
        self.average_min_distance = self.get_min_distance(data).mean()

    def m_step(self, data: np.ndarray):
        self.centroids_ = (
            self.first_order_statistics / self.zeroeth_order_statistics[:, None]
        )

    def fit(self, X, y=None):
        """Fits this machine on data samples."""
        logger.debug(f"Initializing trainer.")
        self.initialize(data=X)

        logger.info("Training k-means.")
        distance = np.inf
        step = 0
        while self.max_iter is None or step < self.max_iter:
            step += 1
            logger.info(
                f"Iteration {step:3d}" + (f"/{self.max_iter}" if self.max_iter else "")
            )
            distance_previous = distance
            self.e_step(data=X)
            self.m_step(data=X)

            # If we're running in dask, persist the centroids so we don't recompute them
            # from the start of the graph at every step.
            for attr in ("centroids_",):
                arr = getattr(self, attr)
                if isinstance(arr, da.Array):
                    setattr(self, attr, arr.persist())

            distance = float(self.average_min_distance)

            logger.info(f"Average minimal squared Euclidean distance = {distance}")

            if step > 1:
                convergence_value = abs(
                    (distance_previous - distance) / distance_previous
                )
                logger.info(f"Convergence value = {convergence_value}")

                # Terminates if converged (and threshold is set)
                if (
                    self.convergence_threshold is not None
                    and convergence_value <= self.convergence_threshold
                ):
                    logger.info("Reached convergence threshold. Training stopped.")
                    break
        else:
            logger.info("Reached maximum step. Training stopped without convergence.")
        self.compute()
        return self

    def partial_fit(self, X, y=None):
        """Applies one e-m step of k-means on the data."""
        if not hasattr(self, "centroids_"):
            logger.debug(f"First call to 'partial_fit'. Initializing...")
            self.initialize(data=X)

        self.e_step(data=X)
        self.m_step(data=X)

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

    def compute(self, *args, **kwargs):
        """Computes delayed arrays if needed."""
        for name in ("centroids_",):
            setattr(self, name, np.asarray(getattr(self, name)))
