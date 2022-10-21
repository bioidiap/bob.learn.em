#!/usr/bin/env python
# @author: Tiago de Freitas Pereira

import dask

# Dask doesn't have an implementation for `pinv`
from scipy.linalg import pinv
from sklearn.base import BaseEstimator, TransformerMixin


class WCCN(TransformerMixin, BaseEstimator):
    """
    Trains a linear machine to perform Within-Class Covariance Normalization (WCCN)
    WCCN finds the projection matrix W that allows us to linearly project the data matrix X to another (sub) space such that:

    .. math::
       (1/N) S_{w} = W W^T

    where :math:`W` is an upper triangular matrix computed using Cholesky Decomposition:

    .. math::
       W = cholesky([(1/K) S_{w} ]^{-1})


    where:
        - :math:`K`  the number of classes
        - :math:`S_w` the within-class scatter; it also has dimensions ``(X.shape[0], X.shape[0])`` and is defined as :math:`S_w = \\sum_{k=1}^K \\sum_{n \\in C_k} (x_n-m_k)(x_n-m_k)^T`, with :math:`C_k` being a set representing all samples for class k.
        - :math:`m_k`  the class *k* empirical mean, defined as :math:`m_k = \\frac{1}{N_k}\\sum_{n \\in C_k} x_n`


    References:
        - 1. Within-class covariance normalization for SVM-based speaker recognition, Andrew O. Hatch, Sachin Kajarekar, and Andreas Stolcke, In INTERSPEECH, 2006.
        - 2. http://en.wikipedia.org/wiki/Cholesky_decomposition"

    """

    def __init__(self, pinv=False, **kwargs):
        super().__init__(**kwargs)
        self.pinv = pinv

    def fit(self, X, y):

        # CHECKING THE TYPES
        if isinstance(X, dask.array.Array):
            import dask.array as numerical_module

            from dask.array.linalg import cholesky, inv
        else:
            import numpy as numerical_module

            from scipy.linalg import cholesky, inv

            X = numerical_module.array(X)

        possible_labels = set(y)
        y_ = numerical_module.array(y)

        n_classes = len(possible_labels)

        # 1. compute the means for each label
        mu_l = numerical_module.array(
            [
                numerical_module.mean(
                    X[numerical_module.where(y_ == label)[0]], axis=0
                )
                for label in possible_labels
            ]
        )

        # 2. Compute Sw
        Sw = numerical_module.zeros((X.shape[1], X.shape[1]), dtype=float)

        for label in possible_labels:
            indexes = numerical_module.where(y_ == label)[0]
            X_l_mu_l = X[indexes] - mu_l[label]

            Sw += X_l_mu_l.T @ X_l_mu_l

        # 3. Compute inv
        scaled_Sw = (1 / n_classes) * Sw
        inv_scaled_Sw = pinv(scaled_Sw) if self.pinv else inv(scaled_Sw)

        # 3. Computes the Cholesky decomposition
        self.weights = cholesky(
            inv_scaled_Sw, lower=True
        )  # Setting lower true to have the same implementation as in the previous code
        self.input_subtract = 0
        self.input_divide = 1.0

        return self

    def transform(self, X):

        return [
            ((x - self.input_subtract) / self.input_divide) @ self.weights
            for x in X
        ]
