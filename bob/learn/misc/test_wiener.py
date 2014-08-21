#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
#
# Copyright (C) 2011-2014 Idiap Research Institute, Martigny, Switzerland

"""Tests the WienerMachine
"""

import os
import numpy
import tempfile
import nose.tools

import bob.sp
import bob.io.base

from . import WienerMachine

def test_initialization():

  # Getters/Setters
  m = WienerMachine(5,4,0.5)
  nose.tools.eq_(m.height, 5)
  nose.tools.eq_(m.width, 4)
  nose.tools.eq_(m.shape, (5,4))
  m.height = 8
  m.width = 7
  nose.tools.eq_(m.height, 8)
  nose.tools.eq_(m.width, 7)
  nose.tools.eq_(m.shape, (8,7))
  m.shape = (5,6)
  nose.tools.eq_(m.height, 5)
  nose.tools.eq_(m.width, 6)
  nose.tools.eq_(m.shape, (5,6))
  ps1 = 0.2 + numpy.fabs(numpy.random.randn(5,6))
  ps2 = 0.2 + numpy.fabs(numpy.random.randn(5,6))
  m.ps = ps1
  assert numpy.allclose(m.ps, ps1)
  m.ps = ps2
  assert numpy.allclose(m.ps, ps2)
  pn1 = 0.5
  m.pn = pn1
  assert abs(m.pn - pn1) < 1e-5
  var_thd = 1e-5
  m.variance_threshold = var_thd
  assert abs(m.variance_threshold - var_thd) < 1e-5

  # Comparison operators
  m2 = WienerMachine(m)
  assert m == m2
  assert (m != m2 ) is False
  m3 = WienerMachine(ps2, pn1)
  m3.variance_threshold = var_thd
  assert m == m3
  assert (m != m3 ) is False

  # Computation of the Wiener filter W
  w_py = 1 / (1. + m.pn / m.ps)
  assert numpy.allclose(m.w, w_py)

def test_load_save():

  m = WienerMachine(5,4,0.5)

  # Save and read from file
  filename = str(tempfile.mkstemp(".hdf5")[1])
  m.save(bob.io.base.HDF5File(filename, 'w'))
  m_loaded = WienerMachine(bob.io.base.HDF5File(filename))
  assert m == m_loaded
  assert (m != m_loaded ) is False
  assert m.is_similar_to(m_loaded)
  # Make them different
  m_loaded.variance_threshold = 0.001
  assert (m == m_loaded ) is False
  assert m != m_loaded

  # Clean-up
  os.unlink(filename)

def test_forward():

  ps = 0.2 + numpy.fabs(numpy.random.randn(5,6))
  pn = 0.5
  m = WienerMachine(ps,pn)

  # Python way
  sample = numpy.random.randn(5,6)
  sample_fft = bob.sp.fft(sample.astype(numpy.complex128))
  w = m.w
  sample_fft_filtered = sample_fft * m.w
  sample_filtered_py = numpy.absolute(bob.sp.ifft(sample_fft_filtered))

  # Bob c++ way
  sample_filtered0 = m.forward(sample)
  sample_filtered1 = m(sample)
  sample_filtered2 = numpy.zeros((5,6),numpy.float64)
  m.forward_(sample, sample_filtered2)
  sample_filtered3 = numpy.zeros((5,6),numpy.float64)
  m.forward(sample, sample_filtered3)
  sample_filtered4 = numpy.zeros((5,6),numpy.float64)
  m(sample, sample_filtered4)
  assert numpy.allclose(sample_filtered0, sample_filtered_py)
  assert numpy.allclose(sample_filtered1, sample_filtered_py)
  assert numpy.allclose(sample_filtered2, sample_filtered_py)
  assert numpy.allclose(sample_filtered3, sample_filtered_py)
  assert numpy.allclose(sample_filtered4, sample_filtered_py)
