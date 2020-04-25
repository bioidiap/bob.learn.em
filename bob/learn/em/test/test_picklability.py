#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Tiago de Freitas Pereira <tiago.pereira@idiap.ch>

from bob.pipelines.utils import assert_picklable
from bob.learn.em import GMMMachine, ISVBase
import numpy
import pickle


def test_gmm_machine():
    gmm_machine = GMMMachine(3,3) 
    gmm_machine.means = numpy.arange(9).reshape(3,3).astype("float")
    gmm_machine_after_pickle = pickle.loads(pickle.dumps(gmm_machine))
    
    assert numpy.allclose(gmm_machine_after_pickle.means, gmm_machine_after_pickle.means, 10e-3)
    assert numpy.allclose(gmm_machine_after_pickle.variances, gmm_machine_after_pickle.variances, 10e-3)
    assert numpy.allclose(gmm_machine_after_pickle.weights, gmm_machine_after_pickle.weights, 10e-3)


def test_isv():
    ubm = GMMMachine(3,3) 
    ubm.means = numpy.arange(9).reshape(3,3).astype("float")
    isv_base = ISVBase(ubm, 2)
    isv_base.u = numpy.arange(18).reshape(9,2).astype("float")
    isv_base.d = numpy.arange(9).astype("float")

    isv_base_after_pickle = pickle.loads(pickle.dumps(isv_base))

    assert numpy.allclose(isv_base.u, isv_base_after_pickle.u, 10e-3)
    assert numpy.allclose(isv_base.d, isv_base_after_pickle.d, 10e-3)