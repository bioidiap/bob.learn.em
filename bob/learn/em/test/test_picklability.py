#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Tiago de Freitas Pereira <tiago.pereira@idiap.ch>

import pickle

import numpy

from bob.learn.em import KMeansMachine


def test_kmeans_machine():
    # Test a KMeansMachine

    means = numpy.array([[3, 70, 0], [4, 72, 0]], "float64")

    # Initializes a KMeansMachine
    kmeans_machine = KMeansMachine(2, 3)
    kmeans_machine.means = means

    kmeans_machine_after_pickle = pickle.loads(pickle.dumps(kmeans_machine))
    assert numpy.allclose(
        kmeans_machine_after_pickle.means, kmeans_machine.means, 10e-3
    )
