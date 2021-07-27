#!/usr/bin/env python
# @author: Yannick Dayer <yannick.dayer@idiap.ch>
# @date: Tue 27 Jul 2021 11:04:10 UTC+02

import dask.array as da
import numpy as np
from dask_ml.cluster.k_means import k_init

from sklearn.base import BaseEstimator


class KMeansMachine:
    """Stores a k-means cluster set parameters (mean of each cluster)

    Parameters
    ----------
    n_means: int
        The number of clusters
    n_dims: int
        The number of dimensions in the data
    means_init: None | str | array
        The means initialization method. Passing None will init the means with random
        values, a str will specify the use of `dask_ml.cluster.k_means.k_init`. Passing
        an array will set the means to those values.
    """

    def __init__(self, n_means, n_dims, means_init=None):
        if n_means < 1:
            raise ValueError("The Number of cluster should be greater thant 1.")
        if n_dims < 1:
            raise ValueError("The Number of dimensions should be greater than 1.")
        self.n_means = n_means
        self.n_dims = n_dims

    def get_means_distance(self, x):
        """Returns the distance value between that point and each mean.

        The returned values are squared Euclidean distances.
        """
        return da.sum((self._means[:, None] - x[None, :]) ** 2, axis=-1)

    def get_distance_from_mean(self, x, i):
        """Returns the distance between one mean and that point.

        The returned value is a squared Euclidean distance.
        """
        return self.get_means_distance(x)[i]

    def get_closest_mean(self, x):
        """Returns the closest mean's index to that point."""
        min_id = da.argmin(self.get_means_distance(x), axis=0)
        return min_id

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
        self._means = new_means



class KMeansTrainer:
    def __init__(self, means_init):
        if means_init is None or isinstance(means_init, str):
            self._means = None
            self._init = means_init
        elif isinstance(means_init, (da.core.Array, np.ndarray)):
            self.set_means(means_init)
        else:
            raise ValueError(f"means_init '{means_init}' is not recognized.")

    def init_means(self, machine, data, random_state, max_iter=None):
        """Assigns the means to an initial value."""
        machine.set_means(
            k_init(
                X=data,
                n_clusters=self.n_means,
                init=self._init,
                random_state=random_state,
                max_iter=max_iter,
            )
        )

    def e_step(self, machine, data):
        pass

    def m_step(self, machine, data):
        pass


class KMeans(BaseEstimator):
    def __init__(self, n_means, n_dims, means_init, dask_client, max_iter=200, threshold=1e-5, init_max_iter=200):
        self.n_means = n_means
        self.n_dims = n_dims
        self.means_init = means_init
        self.max_iter = max_iter
        self.threshold = threshold
        self.init_max_iter = init_max_iter
        self.machine = None
        # TODO random

    def fit(self, data, seed=0):
        self.machine.init_means(data, random_state=seed, max_iter=self.init_max_iter)
        self.machine = KMeansMachine(n_means=self.n_means, n_dims=self.n_dims, means_init=self.means_init)
        trainer = KMeansTrainer() # TODO
        trainer.init_means()
        return self

    def transform(self, data):
        return self.machine.get_means_distance(data)

    def get_variances_and_weights_of_clusters(self, data):
        """Returns the clusters variance and weight for data clustered by the machine.

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
        n_cluster = self.machine.n_means
        closest_means_indices = self.machine.get_closest_mean(data)  # TOCHECK
        weights_count = da.bincount(closest_means_indices, minlength=n_cluster)
        weights = weights_count / weights_count.sum()

        # Accumulate
        means_sum = da.array(
            [data[closest_means_indices == i].sum(axis=0) for i in range(n_cluster)]
        )
        variances_sum = da.array(
            [
                (data[closest_means_indices == i] ** 2).sum(axis=0)
                for i in range(n_cluster)
            ]
        )

        # Reduce
        means = means_sum / weights_count[:, None]
        variances = (variances_sum / weights_count[:, None]) - (means ** 2)

        return variances, weights
