#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Manuel Guenther <Manuel.Guenther@idiap.ch>
# Thu Jun 14 14:45:06 CEST 2012
#
# Copyright (C) 2011-2014 Idiap Research Institute, Martigny, Switzerland

"""Test BIC trainer and machine
"""

import numpy
import nose.tools
from . import BICMachine, BICTrainer

eps = 1e-5

def equals(x, y, epsilon):
  return (abs(x - y) < epsilon).all()

def training_data():
  data = numpy.array([
    (10., 4., 6., 8., 2.),
    (8., 2., 4., 6., 0.),
    (12., 6., 8., 10., 4.),
    (11., 3., 7., 7., 3.),
    (9., 5., 5., 9., 1.)], dtype='float64')

  return data, -1. * data

def eval_data(which):
  eval_data = numpy.ndarray((5,), dtype=numpy.float64)
  if which == 0:
    eval_data.fill(0.)
  elif which == 1:
    eval_data.fill(10.)

  return eval_data

def test_IEC():
  # Tests the IEC training of the BICTrainer
  intra_data, extra_data = training_data()

  # train BIC machine
  machine = BICMachine()
  trainer = BICTrainer()

  # train machine with intrapersonal data only
  trainer.train(machine, intra_data, intra_data)
  # => every result should be zero
  assert abs(machine(eval_data(0))) < eps
  assert abs(machine(eval_data(1))) < eps

  # re-train the machine with intra- and extrapersonal data
  trainer.train(machine, intra_data, extra_data)
  # now, only the input vector 0 should give log-likelihood 0
  assert abs(machine(eval_data(0))) < eps
  # while a positive vector should give a positive result
  assert machine(eval_data(1)) > 0.

@nose.tools.raises(RuntimeError)
def test_raises():

  # Tests the BIC training of the BICTrainer
  intra_data, extra_data = training_data()

  # train BIC machine
  trainer = BICTrainer(2,2)

  # The data are chosen such that the third eigenvalue is zero.
  # Hence, calculating rho (i.e., using the Distance From Feature Space) is impossible
  machine = BICMachine(True)
  trainer.train(machine, intra_data, intra_data)

def test_BIC():
  # Tests the BIC training of the BICTrainer
  intra_data, extra_data = training_data()

  # train BIC machine
  trainer = BICTrainer(2,2)

  # So, now without rho...
  machine = BICMachine(False)

  # First, train the machine with intrapersonal data only
  trainer.train(machine, intra_data, intra_data)

  # => every result should be zero
  assert abs(machine(eval_data(0))) < eps
  assert abs(machine(eval_data(1))) < eps

  # re-train the machine with intra- and extrapersonal data
  trainer.train(machine, intra_data, extra_data)
  # now, only the input vector 0 should give log-likelihood 0
  assert abs(machine(eval_data(0))) < eps
  # while a positive vector should give a positive result
  assert machine(eval_data(1)) > 0.
