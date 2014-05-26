#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
#
# Copyright (C) 2011-2014 Idiap Research Institute, Martigny, Switzerland

"""Tests the WienerTrainer
"""

import numpy
import bob.sp

from . import WienerMachine, WienerTrainer

def train_wiener_ps(training_set):

  # Python implementation
  n_samples = training_set.shape[0]
  height = training_set.shape[1]
  width = training_set.shape[2]
  training_fftabs = numpy.zeros((n_samples, height, width), dtype=numpy.float64)

  for n in range(n_samples):
    sample = (training_set[n,:,:]).astype(numpy.complex128)
    training_fftabs[n,:,:] = numpy.absolute(bob.sp.fft(sample))

  mean = numpy.mean(training_fftabs, axis=0)

  for n in range(n_samples):
    training_fftabs[n,:,:] -= mean

  training_fftabs = training_fftabs * training_fftabs
  var_ps = numpy.mean(training_fftabs, axis=0)

  return var_ps


def test_initialization():

  # Constructors and comparison operators
  t1 = WienerTrainer()
  t2 = WienerTrainer()
  t3 = WienerTrainer(t2)
  t4 = t3
  assert t1 == t2
  assert (t1 != t2) is False
  assert t1.is_similar_to(t2)
  assert t1 == t3
  assert (t1 != t3) is False
  assert t1.is_similar_to(t3)
  assert t1 == t4
  assert (t1 != t4) is False
  assert t1.is_similar_to(t4)


def test_train():

  n_samples = 20
  height = 5
  width = 6
  training_set = 0.2 + numpy.fabs(numpy.random.randn(n_samples, height, width))

  # Python implementation
  var_ps = train_wiener_ps(training_set)
  # Bob C++ implementation (variant 1) + comparison against python one
  t = WienerTrainer()
  m1 = t.train(training_set)
  assert numpy.allclose(var_ps, m1.ps)
  # Bob C++ implementation (variant 2) + comparison against python one
  m2 = WienerMachine(height, width, 0.)
  t.train(m2, training_set)
  assert numpy.allclose(var_ps, m2.ps)

