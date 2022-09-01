#!/usr/bin/env python
# @author: Tiago de Freitas Pereira

import dask

from scipy.linalg import pinv
from sklearn.base import BaseEstimator, TransformerMixin


class Whitening(TransformerMixin, BaseEstimator):
    """
    Trains an Estimator perform Cholesky whitening.

    The whitening transformation is a decorrelation method that converts the covariance matrix of a set of samples into the identity matrix :math:`I`.
    This effectively linearly transforms random variables such that the resulting variables are uncorrelated and have the same variances as the original random variables.

    This transformation is invertible.
    The method is called the whitening transform because it transforms the input matrix :math:`X` closer towards white noise (let's call it :math:`\\tilde{X}`):

    .. math::
        Cov(\\tilde{X}) = I

    with:
      .. math::   \\tilde{X} = X W

    where :math:`W` is the projection matrix that allows us to linearly project the data matrix :math:`X` to another (sub) space such that:

    .. math::
        Cov(X) = W W^T


    :math:`W` is computed using Cholesky decomposition:

    .. math::
        W = cholesky([Cov(X)]^{-1})


    References:
        - 1. https://rtmath.net/help/html/e9c12dc0-e813-4ca9-aaa3-82340f1c5d24.htm
        - 2. http://en.wikipedia.org/wiki/Cholesky_decomposition

    """

    def __init__(self, pinv: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.pinv = pinv

    def fit(self, X, y=None):
        # CHECKING THE TYPES
        if isinstance(X, dask.array.Array):
            import dask.array as numerical_module

            from dask.array.linalg import cholesky, inv

        else:
            import numpy as numerical_module

            from scipy.linalg import cholesky, inv

        # 1. Computes the mean vector and the covariance matrix of the training set
        mu = numerical_module.mean(X, axis=0)
        cov = numerical_module.cov(numerical_module.transpose(X))

        # 2. Computes the inverse of the covariance matrix
        inv_cov = pinv(cov) if self.pinv else inv(cov)

        # 3. Computes the Cholesky decomposition of the inverse covariance matrix
        self.weights = cholesky(
            inv_cov, lower=True
        )  # Setting lower true to have the same implementation as in the previous code
        self.input_subtract = mu
        self.input_divide = 1.0

        return self

    def transform(self, X):
        return ((X - self.input_subtract) / self.input_divide) @ self.weights
