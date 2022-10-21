#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Francois Moulin <Francois.Moulin@idiap.ch>
# Wed Jul 13 16:00:04 2011 +0200
#
# Copyright (C) 2011-2014 Idiap Research Institute, Martigny, Switzerland

"""Tests on the LinearScoring function
"""

import numpy as np

from bob.learn.em import GMMMachine, GMMStats, linear_scoring


def test_LinearScoring():

    ubm = GMMMachine(n_gaussians=2)
    ubm.weights = np.array([0.5, 0.5], "float64")
    ubm.means = np.array([[3, 70], [4, 72]], "float64")
    ubm.variances = np.array([[1, 10], [2, 5]], "float64")
    ubm.variance_thresholds = np.array([[0, 0], [0, 0]], "float64")

    model1 = GMMMachine(n_gaussians=2)
    model1.weights = np.array([0.5, 0.5], "float64")
    model1.means = np.array([[1, 2], [3, 4]], "float64")
    model1.variances = np.array([[9, 10], [11, 12]], "float64")
    model1.variance_thresholds = np.array([[0, 0], [0, 0]], "float64")

    model2 = GMMMachine(n_gaussians=2)
    model2.weights = np.array([0.5, 0.5], "float64")
    model2.means = np.array([[5, 6], [7, 8]], "float64")
    model2.variances = np.array([[13, 14], [15, 16]], "float64")
    model2.variance_thresholds = np.array([[0, 0], [0, 0]], "float64")

    stats1 = GMMStats(2, 2)
    stats1.sum_px = np.array([[1, 2], [3, 4]], "float64")
    stats1.n = np.array([1, 2], "float64")
    stats1.t = 1 + 2

    stats2 = GMMStats(2, 2)
    stats2.sum_px = np.array([[5, 6], [7, 8]], "float64")
    stats2.n = np.array([3, 4], "float64")
    stats2.t = 3 + 4

    stats3 = GMMStats(2, 2)
    stats3.sum_px = np.array([[5, 6], [7, 3]], "float64")
    stats3.n = np.array([3, 4], "float64")
    stats3.t = 3 + 4

    test_channeloffset = [
        np.array([[9, 8], [7, 6]], "float64"),
        np.array([[5, 4], [3, 2]], "float64"),
        np.array([[1, 0], [1, 2]], "float64"),
    ]

    # Reference scores (from Idiap internal matlab implementation)
    ref_scores_00 = np.array(
        [[2372.9, 5207.7, 5275.7], [2215.7, 4868.1, 4932.1]], "float64"
    )
    ref_scores_01 = np.array(
        [
            [790.9666666666667, 743.9571428571428, 753.6714285714285],
            [738.5666666666667, 695.4428571428572, 704.5857142857144],
        ],
        "float64",
    )
    ref_scores_10 = np.array(
        [[2615.5, 5434.1, 5392.5], [2381.5, 4999.3, 5022.5]], "float64"
    )
    ref_scores_11 = np.array(
        [
            [871.8333333333332, 776.3000000000001, 770.3571428571427],
            [793.8333333333333, 714.1857142857143, 717.5000000000000],
        ],
        "float64",
    )

    # 1/ Use GMMMachines
    # 1/a/ Without test_channelOffset, without frame-length normalisation
    scores = linear_scoring([model1, model2], ubm, [stats1, stats2, stats3])
    np.testing.assert_almost_equal(scores, ref_scores_00, decimal=7)

    # 1/b/ Without test_channelOffset, with frame-length normalisation
    scores = linear_scoring(
        [model1, model2],
        ubm,
        [stats1, stats2, stats3],
        frame_length_normalization=True,
    )
    np.testing.assert_almost_equal(scores, ref_scores_01, decimal=7)
    scores = linear_scoring(
        [model1, model2], ubm, [stats1, stats2, stats3], 0, True
    )
    np.testing.assert_almost_equal(scores, ref_scores_01, decimal=7)

    # 1/c/ With test_channelOffset, without frame-length normalisation
    scores = linear_scoring(
        [model1, model2], ubm, [stats1, stats2, stats3], test_channeloffset
    )
    np.testing.assert_almost_equal(scores, ref_scores_10, decimal=7)

    # 1/d/ With test_channelOffset, with frame-length normalisation
    scores = linear_scoring(
        [model1, model2],
        ubm,
        [stats1, stats2, stats3],
        test_channeloffset,
        frame_length_normalization=True,
    )
    np.testing.assert_almost_equal(scores, ref_scores_11, decimal=7)

    # 2/ Use means instead of models
    # 2/a/ Without test_channelOffset, without frame-length normalisation
    scores = linear_scoring(
        [model1.means, model2.means], ubm, [stats1, stats2, stats3]
    )
    assert (abs(scores - ref_scores_00) < 1e-7).all()

    # 2/b/ Without test_channelOffset, with frame-length normalisation
    scores = linear_scoring(
        [model1.means, model2.means],
        ubm,
        [stats1, stats2, stats3],
        frame_length_normalization=True,
    )
    assert (abs(scores - ref_scores_01) < 1e-7).all()

    # 2/c/ With test_channelOffset, without frame-length normalisation
    scores = linear_scoring(
        [model1.means, model2.means],
        ubm,
        [stats1, stats2, stats3],
        test_channeloffset,
    )
    assert (abs(scores - ref_scores_10) < 1e-7).all()

    # 2/d/ With test_channelOffset, with frame-length normalisation
    scores = linear_scoring(
        [model1.means, model2.means],
        ubm,
        [stats1, stats2, stats3],
        test_channeloffset,
        frame_length_normalization=True,
    )
    assert (abs(scores - ref_scores_11) < 1e-7).all()

    # 3/ Using single model/sample
    # 3/a/ without frame-length normalisation
    score = linear_scoring(model1.means, ubm, stats1, test_channeloffset[0])
    np.testing.assert_almost_equal(score, ref_scores_10[0, 0], decimal=7)
    score = linear_scoring(model1.means, ubm, stats2, test_channeloffset[1])
    np.testing.assert_almost_equal(score, ref_scores_10[0, 1], decimal=7)
    score = linear_scoring(model1.means, ubm, stats3, test_channeloffset[2])
    np.testing.assert_almost_equal(score, ref_scores_10[0, 2], decimal=7)
    score = linear_scoring(model2.means, ubm, stats1, test_channeloffset[0])
    np.testing.assert_almost_equal(score, ref_scores_10[1, 0], decimal=7)
    score = linear_scoring(model2.means, ubm, stats2, test_channeloffset[1])
    np.testing.assert_almost_equal(score, ref_scores_10[1, 1], decimal=7)
    score = linear_scoring(model2.means, ubm, stats3, test_channeloffset[2])
    np.testing.assert_almost_equal(score, ref_scores_10[1, 2], decimal=7)

    # 3/b/ with frame-length normalisation
    score = linear_scoring(
        model1.means, ubm, stats1, test_channeloffset[0], True
    )
    np.testing.assert_almost_equal(score, ref_scores_11[0, 0], decimal=7)
    score = linear_scoring(
        model1.means, ubm, stats2, test_channeloffset[1], True
    )
    np.testing.assert_almost_equal(score, ref_scores_11[0, 1], decimal=7)
    score = linear_scoring(
        model1.means, ubm, stats3, test_channeloffset[2], True
    )
    np.testing.assert_almost_equal(score, ref_scores_11[0, 2], decimal=7)
    score = linear_scoring(
        model2.means, ubm, stats1, test_channeloffset[0], True
    )
    np.testing.assert_almost_equal(score, ref_scores_11[1, 0], decimal=7)
    score = linear_scoring(
        model2.means, ubm, stats2, test_channeloffset[1], True
    )
    np.testing.assert_almost_equal(score, ref_scores_11[1, 1], decimal=7)
    score = linear_scoring(
        model2.means, ubm, stats3, test_channeloffset[2], True
    )
    np.testing.assert_almost_equal(score, ref_scores_11[1, 2], decimal=7)
