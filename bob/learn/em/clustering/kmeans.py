#!/usr/bin/env python
# @author: Yannick Dayer <yannick.dayer@idiap.ch>
# @date: Tue 27 Jul 2021 11:04:10 UTC+02

import dask.array as da
import numpy as np
from dask_ml.cluster.k_means import k_init

from sklearn.base import BaseEstimator

import logging

logger = logging.getLogger(__name__)


class KMeansMachine:
    """Stores a k-means cluster set parameters (mean of each cluster)

    Parameters
    ----------
    n_means: int
        The number of clusters
    n_dims: int
        The number of dimensions in the data
    """

    def __init__(self, n_means, n_dims):
        if n_means < 1:
            raise ValueError("The Number of cluster should be greater thant 1.")
        if n_dims < 1:
            raise ValueError("The Number of dimensions should be greater than 1.")
        self.n_means = n_means
        self.n_dims = n_dims
        self.means = da.zeros((self.n_means, self.n_dims), dtype="float64")
        self.shape = self.means.shape

    def get_means_distance(self, x):
        """Returns the distance values between that point and each mean.

        The returned values are squared Euclidean distances.
        """
        return da.sum((self.means[:, None] - x[None, :]) ** 2, axis=-1)

    def get_distance_from_mean(self, x, i):
        """Returns the distance between one mean and that point.

        The returned value is a squared Euclidean distance.
        """
        return self.get_means_distance(x)[i]

    def get_closest_mean(self, x):
        """Returns the closest mean's index to that point."""
        dists = self.get_means_distance(x)
        min_id = da.argmin(dists, axis=0)
        min_dist = dists[min_id]
        return min_id, min_dist

    def get_closest_mean_index(self, x):
        """Returns the closest mean's index to that point."""
        return da.argmin(self.get_means_distance(x), axis=0)

    def get_min_distance(self, x):
        """Returns the smallest distance value between that point and each mean.

        The returned value is a squared Euclidean distance.
        """
        return da.min(self.get_means_distance(x), axis=0)

    def set_means(self, new_means):
        if not hasattr(new_means, "shape"):
            raise TypeError(
                f"new_means '{new_means}' of type {type(new_means)} is not valid."
            )
        if new_means.shape != (self.n_means, self.n_dims):
            raise ValueError(
                f"new_means of shape {new_means.shape} should be of shape "
                f"{(self.n_means, self.n_dims)}"
            )
        self.means = new_means

    def get_mean(self, mean_index):
        return self.means[mean_index]

    def set_mean(self, mean_index, mean):
        self.means[mean_index,:] = mean

    def copy(self):
        new_machine = KMeansMachine(n_means=self.n_means, n_dims=self.n_dims)
        new_machine.set_means(self.means)
        return new_machine

    def __eq__(self, obj):
        return da.allclose(self.means, obj.means, rtol=0, atol=0)

    def is_similar_to(self, obj, r_epsilon=1e-05, a_epsilon=1e-08):
        return da.allclose(self.means, obj.means, rtol=r_epsilon, atol=a_epsilon)

    def get_variances_and_weights_for_each_cluster(self, data):
        """Returns the clusters variance and weight for data clustered by the machine.

        For each mean, finds the subset of the samples that is closest to that mean,
        and calculates:
        1) the variance of that subset (the cluster variance)
        2) the proportion of samples represented by that subset (the cluster weight)

        Parameters
        ----------
        data: dask.array
            The data to compute the variance of.

        Returns
        -------
        2-tuple of arrays:
            variances: 2D array
                For each cluster, the variance in each dimension of the data.
            weights: 1D array
                Weight (proportion of quantity of data point) of each cluster.
        """
        n_cluster = self.n_means
        closest_means_indices = self.get_closest_mean_index(data)
        weights_count = da.bincount(closest_means_indices, minlength=n_cluster)
        weights = weights_count / weights_count.sum()

        # Accumulate
        means_sum = da.array(
            [data[closest_means_indices == i,:].sum(axis=0) for i in range(n_cluster)]
        )
        variances_sum = da.array(
            [
                (data[closest_means_indices == i,:] ** 2).sum(axis=0)
                for i in range(n_cluster)
            ]
        )

        # Reduce
        means = means_sum / weights_count[:, None]
        variances = (variances_sum / weights_count[:, None]) - (means ** 2)

        return variances, weights


class KMeansTrainer:
    def __init__(self):
        self.average_min_distance = None
        self.zeroeth_order_statistics = None
        self.first_order_statistics = None

    def initialize(self, machine, data, random_state=0, means_init="k-means||", max_iter=None):
        """Assigns the means to an initial value."""
        machine.set_means(
            k_init(
                X=data,
                n_clusters=machine.n_means,
                init=means_init,
                random_state=random_state,
                max_iter=max_iter,
            )
        )

    def e_step(self, machine: KMeansMachine, data: da.Array):
        n_cluster = machine.n_means
        closest_mean_indices = machine.get_closest_mean_index(data)
        # number of data point in each cluster
        self.zeroeth_order_statistics = da.bincount(closest_mean_indices, minlength=n_cluster)
        # sum of data points coordinates in each cluster
        self.first_order_statistics = da.array(
            [data[closest_mean_indices == i].sum(axis=0) for i in range(n_cluster)]
        )
        self.average_min_distance = machine.get_min_distance(data).mean()

    def m_step(self, machine: KMeansMachine, data: da.Array):
        machine.set_means(self.first_order_statistics / self.zeroeth_order_statistics[:, None])

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
        self.zeroeth_order_statistics = da.zeros((machine.n_means,))
        self.first_order_statistics = da.zeros((machine.n_means, machine.n_dims))

class KMeans(BaseEstimator):
    """Transformer clustering data using k-means.

    Parameters
    ----------
    n_means: int
        Number of means to fit (number of clusters).
    n_dims: int
        Dimension of the data.
    means_init: str or numpy.ndarray
        One of `["k-means||", "k-means++", "random"]`, or an array of means of shape
        `(n_means, n_dims)`.
    max_iter: int
        K-means e-m maximum iterations, stopping criterion if `threshold` is not
        reached.
    convergence_threshold: float or None
        K-means stopping criterion.
    init_max_iter: int
        maximum iterations of `k-means||` or `k-means++` initialization methods.
    random_state: int or dask.array.random.RandomState or numpy.random.RandomState
        The seed for the random generator.
    dask_client: dask.distributed.client
        TODO
    """
    def __init__(self, n_means, n_dims, means_init="k-means||", max_iter=20, convergence_threshold=1e-5, init_max_iter=20, random_state=0, dask_client=None):
        self.n_means = n_means
        self.n_dims = n_dims
        self.means_init = means_init
        self.max_iter = max_iter
        self.convergence_threshold = convergence_threshold
        self.init_max_iter = init_max_iter
        self.random_state = random_state
        self.dask_client = dask_client
        self.machine = None
        self.trainer = KMeansTrainer()

    def fit(self, data):
        if self.machine is None:
            logger.info("Creating k-Means machine.")
            self.machine = KMeansMachine(n_means=self.n_means, n_dims=self.n_dims)
            logger.debug("Initializing means.")
            self.trainer.initialize(machine=self.machine, data=data, means_init=self.means_init, random_state=self.random_state, max_iter=self.init_max_iter)

        logger.info("Training k-Means...")
        self.trainer.e_step(self.machine, data)
        average_output = self.trainer.compute_likelihood(self.machine)
        for i in range(self.max_iter):
            logger.info(f"Iteration = {i:3d}/{self.max_iter}")
            average_output_previous = average_output
            self.trainer.m_step(self.machine, data)
            self.trainer.e_step(self.machine, data)

            average_output = self.trainer.compute_likelihood(self.machine)

            logger.info(f"Average squared Euclidean distance = {average_output}")

            convergence_value = abs((average_output_previous - average_output)/average_output_previous)
            logger.info(f"convergence value = {convergence_value}")

            #Terminates if converged (and likelihood computation is set)
            if self.convergence_threshold is not None and convergence_value <= self.convergence_threshold:
                break
        return self

    def transform(self, data):
        return self.machine.get_means_distance(data)
