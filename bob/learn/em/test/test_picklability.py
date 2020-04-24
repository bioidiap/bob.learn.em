#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Tiago de Freitas Pereira <tiago.pereira@idiap.ch>import numpy

from bob.pipelines.utils import assert_picklable
from bob.learn.em import GMMMachine
from .test_em import equals
import numpy
import pickle


def test_gmm_machine():
    gmm_machine = GMMMachine(3,3) 
    gmm_machine.means = numpy.arange(9).reshape(3,3).astype("float")
    gmm_machine_after_pickle = pickle.loads(pickle.dumps(gmm_machine))
    
    assert equals(gmm_machine_after_pickle.means, gmm_machine_after_pickle.means, 10e-3)
    assert equals(gmm_machine_after_pickle.variances, gmm_machine_after_pickle.variances, 10e-3)
    assert equals(gmm_machine_after_pickle.weights, gmm_machine_after_pickle.weights, 10e-3)
