#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.anjos@idiap.ch>
# Tiago de Freitas Pereira <tiago.pereira@idiap.ch>
# Tue May 31 16:55:10 2011 +0200
#
# Copyright (C) 2011-2014 Idiap Research Institute, Martigny, Switzerland

"""Tests on the machine infrastructure.
"""

import numpy as np

from bob.learn.em import WCCN, Whitening


def run_whitening(with_dask):

    # CHECKING THE TYPES
    if with_dask:
        import dask.array as numerical_module
    else:
        import numpy as numerical_module

    # Tests our Whitening extractor.
    data = numerical_module.array(
        [
            [1.2622, -1.6443, 0.1889],
            [0.4286, -0.8922, 1.3020],
            [-0.6613, 0.0430, 0.6377],
            [-0.8718, -0.4788, 0.3988],
            [-0.0098, -0.3121, -0.1807],
            [0.4301, 0.4886, -0.1456],
        ]
    )
    sample = numerical_module.array([1, 2, 3.0])

    # Expected results (from matlab)
    mean_ref = numerical_module.array(
        [0.096324163333333, -0.465965438333333, 0.366839091666667]
    )
    whit_ref = numerical_module.array(
        [
            [1.608410253685985, 0, 0],
            [1.079813355720326, 1.411083365535711, 0],
            [0.693459921529905, 0.571417184139332, 1.800117179839927],
        ]
    )
    sample_whitened_ref = numerical_module.array(
        [5.942255453628436, 4.984316201643742, 4.739998188373740]
    )

    # Runs whitening (first method)

    t = Whitening()
    t.fit(data)

    s = t.transform(sample)

    # Makes sure results are good
    eps = 1e-4
    assert np.allclose(t.input_subtract, mean_ref, eps, eps)
    assert np.allclose(t.weights, whit_ref, eps, eps)
    assert np.allclose(s, sample_whitened_ref, eps, eps)

    # Runs whitening (second method)
    m2 = t.fit(data)
    s2 = t.transform(sample)

    # Makes sure results are good
    eps = 1e-4
    assert np.allclose(m2.input_subtract, mean_ref, eps, eps)
    assert np.allclose(m2.weights, whit_ref, eps, eps)
    assert np.allclose(s2, sample_whitened_ref, eps, eps)


def run_wccn(with_dask):

    # CHECKING THE TYPES
    if with_dask:
        import dask.array as numerical_module
    else:
        import numpy as numerical_module

    # Tests our Whitening extractor.
    X = numerical_module.array(
        [
            [1.2622, -1.6443, 0.1889],
            [0.4286, -0.8922, 1.3020],
            [-0.6613, 0.0430, 0.6377],
            [-0.8718, -0.4788, 0.3988],
            [-0.0098, -0.3121, -0.1807],
            [0.4301, 0.4886, -0.1456],
        ]
    )
    y = [0, 0, 1, 1, 2, 2]

    sample = numerical_module.array([1, 2, 3.0])

    # Expected results
    mean_ref = numerical_module.array([0.0, 0.0, 0.0])
    weight_ref = numerical_module.array(
        [
            [15.8455444, 0.0, 0.0],
            [-10.7946764, 2.87942129, 0.0],
            [18.76762201, -2.19719292, 2.1505817],
        ]
    )
    sample_wccn_ref = numerical_module.array(
        [50.55905765, -0.83273618, 6.45174511]
    )

    # Runs WCCN (first method)
    t = WCCN()
    t.fit(X, y=y)
    s = t.transform(sample)

    # Makes sure results are good
    eps = 1e-4
    assert np.allclose(t.input_subtract, mean_ref, eps, eps)
    assert np.allclose(t.weights, weight_ref, eps, eps)
    assert np.allclose(s, sample_wccn_ref, eps, eps)

    # Runs WCCN (second method)
    t.fit(X, y)
    s2 = t.transform(sample)

    # Makes sure results are good
    eps = 1e-4
    assert np.allclose(t.input_subtract, mean_ref, eps, eps)
    assert np.allclose(t.weights, weight_ref, eps, eps)
    assert np.allclose(s2, sample_wccn_ref, eps, eps)


def test_wccn_numpy():
    run_wccn(with_dask=False)


def test_wccn_dask():
    run_wccn(with_dask=True)


def test_whitening_numpy():
    run_whitening(with_dask=False)


def test_whitening_dask():
    run_whitening(with_dask=True)
