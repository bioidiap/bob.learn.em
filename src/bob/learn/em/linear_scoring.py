#!/usr/bin/env python
# @author: Yannick Dayer <yannick.dayer@idiap.ch>
# @date: Thu 21 Oct 2021 14:14:07 UTC+02

"""Implements the linear scoring function."""

import logging

from typing import Union

import numpy as np

from .gmm import GMMMachine, GMMStats

logger = logging.getLogger(__name__)

EPSILON = np.finfo(float).eps


def linear_scoring(
    models_means: "Union[list[GMMMachine], np.ndarray[('n_models', 'n_gaussians', 'n_features'), float]]",  # noqa: F821
    ubm: GMMMachine,
    test_stats: Union["list[GMMStats]", GMMStats],
    test_channel_offsets: "np.ndarray[('n_test_stats', 'n_gaussians'), float]" = 0,  # noqa: F821
    frame_length_normalization: bool = False,
) -> "np.ndarray[('n_models', 'n_test_stats'), float]":  # noqa: F821
    """Estimation of the LLR between a target model and the UBM for a test instance.

    The Linear scoring is an approximation to the log-likelihood ratio (LLR) that was
    shown to be as accurate and up to two orders of magnitude more efficient to
    compute. [Glembek2009]

    Parameters
    ----------
    models_means
        The model(s) to score against. If a list of `GMMMachine` is given, the means
        of each model are considered.
    ubm:
        The Universal Background Model. Accepts a `GMMMachine` object. If the
        `GMMMachine` uses MAP, it's `ubm` attribute is used.
    test_stats:
        The instances to score.
    test_channel_offsets
        Offset values added to the test instances.

    Returns
    -------
    Array of shape (n_models, n_probes)
        The scores of each probe against each model.
    """
    if isinstance(models_means[0], GMMMachine):
        models_means = np.array([model.means for model in models_means])
    if not hasattr(models_means, "ndim"):
        models_means = np.array(models_means)
    if models_means.ndim < 2:
        raise ValueError(
            "models_means must be of shape `(n_models, n_gaussians, n_features)`."
        )
    if models_means.ndim == 2:
        models_means = models_means[None, :, :]

    if ubm.trainer == "map":
        ubm = ubm.ubm

    if isinstance(test_stats, GMMStats):
        test_stats = [test_stats]

    # All stats.sum_px [array of shape (n_test_stats, n_gaussians, n_features)]
    sum_px = np.array([stat.sum_px for stat in test_stats])
    # All stats.n [array of shape (n_test_stats, n_gaussians)]
    n = np.array([stat.n for stat in test_stats])
    # All stats.t [array of shape (n_test_stats,)]
    t = np.array([stat.t for stat in test_stats])
    # Offsets [array of shape (n_test_stats, `n_gaussians * n_features`)]
    test_channel_offsets = np.array(test_channel_offsets)

    # Compute A [array of shape (n_models, n_gaussians * n_features)]
    a = (models_means - ubm.means) / ubm.variances
    # Compute B [array of shape (n_gaussians * n_features, n_test_stats)]
    b = sum_px[:, :, :] - (
        n[:, :, None] * (ubm.means[None, :, :] + test_channel_offsets)
    )
    b = np.transpose(b, axes=(1, 2, 0))
    # Apply normalization if needed.
    if frame_length_normalization:
        b = np.where(abs(t) <= EPSILON, 0, b[:, :] / t[None, :])
    return np.tensordot(a, b, 2)
