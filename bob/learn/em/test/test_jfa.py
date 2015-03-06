#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
# Wed Feb 15 23:24:35 2012 +0200
#
# Copyright (C) 2011-2014 Idiap Research Institute, Martigny, Switzerland

"""Tests on the JFA-based machines
"""

import os
import numpy
import numpy.linalg
import tempfile

import bob.io.base

from bob.learn.em import GMMMachine, GMMStats, JFABase, ISVBase, ISVMachine, JFAMachine

def estimate_x(dim_c, dim_d, mean, sigma, U, N, F):
  # Compute helper values
  UtSigmaInv = {}
  UtSigmaInvU = {}
  dim_ru = U.shape[1]
  for c in range(dim_c):
    start                       = c*dim_d
    end                         = (c+1)*dim_d
    Uc                          = U[start:end,:]
    UtSigmaInv[c]  = Uc.transpose() / sigma[start:end]
    UtSigmaInvU[c] = numpy.dot(UtSigmaInv[c], Uc);

  # I + (U^{T} \Sigma^-1 N U)
  I_UtSigmaInvNU = numpy.eye(dim_ru, dtype=numpy.float64)
  for c in range(dim_c):
    I_UtSigmaInvNU = I_UtSigmaInvNU + UtSigmaInvU[c] * N[c]

  # U^{T} \Sigma^-1 F
  UtSigmaInv_Fnorm = numpy.zeros((dim_ru,), numpy.float64)
  for c in range(dim_c):
    start             = c*dim_d
    end               = (c+1)*dim_d
    Fnorm             = F[c,:] - N[c] * mean[start:end]
    UtSigmaInv_Fnorm  = UtSigmaInv_Fnorm + numpy.dot(UtSigmaInv[c], Fnorm)

  return numpy.linalg.solve(I_UtSigmaInvNU, UtSigmaInv_Fnorm)

def estimate_ux(dim_c, dim_d, mean, sigma, U, N, F):
  return numpy.dot(U, estimate_x(dim_c, dim_d, mean, sigma, U, N, F))


def test_JFABase():

  # Creates a UBM
  weights = numpy.array([0.4, 0.6], 'float64')
  means = numpy.array([[1, 6, 2], [4, 3, 2]], 'float64')
  variances = numpy.array([[1, 2, 1], [2, 1, 2]], 'float64')
  ubm = GMMMachine(2,3)
  ubm.weights = weights
  ubm.means = means
  ubm.variances = variances

  # Creates a JFABase
  U = numpy.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]], 'float64')
  V = numpy.array([[6, 5], [4, 3], [2, 1], [1, 2], [3, 4], [5, 6]], 'float64')
  d = numpy.array([0, 1, 0, 1, 0, 1], 'float64')
  m = JFABase(ubm, ru=1, rv=1)

  _,_,ru,rv = m.shape
  assert ru == 1
  assert rv == 1

  # Checks for correctness
  m.resize(2,2)
  m.u = U
  m.v = V
  m.d = d
  n_gaussians,dim,ru,rv = m.shape
  supervector_length    = m.supervector_length

  assert (m.u == U).all()
  assert (m.v == V).all()
  assert (m.d == d).all()
  assert n_gaussians        == 2
  assert dim                == 3
  assert supervector_length == 6
  assert ru                 == 2
  assert rv                 == 2

  # Saves and loads
  filename = str(tempfile.mkstemp(".hdf5")[1])
  m.save(bob.io.base.HDF5File(filename, 'w'))
  m_loaded = JFABase(bob.io.base.HDF5File(filename))
  m_loaded.ubm = ubm
  assert m == m_loaded
  assert (m != m_loaded) is False
  assert m.is_similar_to(m_loaded)

  # Copy constructor
  mc = JFABase(m)
  assert m == mc

  # Variant
  #mv = JFABase()
  # Checks for correctness
  #mv.ubm = ubm
  #mv.resize(2,2)
  #mv.u = U
  #mv.v = V
  #mv.d = d
  #assert (m.u == U).all()
  #assert (m.v == V).all()
  #assert (m.d == d).all()
  #assert m.dim_c == 2
  #assert m.dim_d == 3
  #assert m.dim_cd == 6
  #assert m.dim_ru == 2
  #assert m.dim_rv == 2

  # Clean-up
  os.unlink(filename)

def test_ISVBase():

  # Creates a UBM
  weights = numpy.array([0.4, 0.6], 'float64')
  means = numpy.array([[1, 6, 2], [4, 3, 2]], 'float64')
  variances = numpy.array([[1, 2, 1], [2, 1, 2]], 'float64')
  ubm           = GMMMachine(2,3)
  ubm.weights   = weights
  ubm.means     = means
  ubm.variances = variances

  # Creates a ISVBase
  U = numpy.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]], 'float64')
  d = numpy.array([0, 1, 0, 1, 0, 1], 'float64')
  m = ISVBase(ubm, ru=1)
  _,_,ru = m.shape
  assert ru == 1

  # Checks for correctness
  m.resize(2)
  m.u = U
  m.d = d
  n_gaussians,dim,ru = m.shape
  supervector_length = m.supervector_length
  assert (m.u == U).all()
  assert (m.d == d).all()
  assert n_gaussians        == 2
  assert dim                == 3
  assert supervector_length == 6
  assert ru                 == 2

  # Saves and loads
  filename = str(tempfile.mkstemp(".hdf5")[1])
  m.save(bob.io.base.HDF5File(filename, 'w'))
  m_loaded = ISVBase(bob.io.base.HDF5File(filename))
  m_loaded.ubm = ubm
  assert m == m_loaded
  assert (m != m_loaded) is False
  assert m.is_similar_to(m_loaded)

  # Copy constructor
  mc = ISVBase(m)
  assert m == mc

  # Variant
  #mv = ISVBase()
  # Checks for correctness
  #mv.ubm = ubm
  #mv.resize(2)
  #mv.u = U
  #mv.d = d
  #assert (m.u == U).all()
  #assert (m.d == d).all()
  #ssert m.dim_c == 2
  #assert m.dim_d == 3
  #assert m.dim_cd == 6
  #assert m.dim_ru == 2

  # Clean-up
  os.unlink(filename)

def test_JFAMachine():

  # Creates a UBM
  weights   = numpy.array([0.4, 0.6], 'float64')
  means     = numpy.array([[1, 6, 2], [4, 3, 2]], 'float64')
  variances = numpy.array([[1, 2, 1], [2, 1, 2]], 'float64')
  ubm           = GMMMachine(2,3)
  ubm.weights   = weights
  ubm.means     = means
  ubm.variances = variances

  # Creates a JFABase
  U = numpy.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]], 'float64')
  V = numpy.array([[6, 5], [4, 3], [2, 1], [1, 2], [3, 4], [5, 6]], 'float64')
  d = numpy.array([0, 1, 0, 1, 0, 1], 'float64')
  base = JFABase(ubm,2,2)
  base.u = U
  base.v = V
  base.d = d

  # Creates a JFAMachine
  y = numpy.array([1,2], 'float64')
  z = numpy.array([3,4,1,2,0,1], 'float64')
  m = JFAMachine(base)
  m.y = y
  m.z = z
  n_gaussians,dim,ru,rv = m.shape
  supervector_length    = m.supervector_length

  assert n_gaussians        == 2
  assert dim                == 3
  assert supervector_length == 6
  assert ru                 == 2
  assert rv                 == 2
  assert (m.y == y).all()
  assert (m.z == z).all()

  # Saves and loads
  filename = str(tempfile.mkstemp(".hdf5")[1])
  m.save(bob.io.base.HDF5File(filename, 'w'))
  m_loaded = JFAMachine(bob.io.base.HDF5File(filename))
  m_loaded.jfa_base = base
  assert m == m_loaded
  assert (m != m_loaded) is False
  assert m.is_similar_to(m_loaded)

  # Copy constructor
  mc = JFAMachine(m)
  assert m == mc

  # Variant
  #mv = JFAMachine()
  # Checks for correctness
  #mv.jfa_base = base
  #m.y = y
  #m.z = z
  #assert m.dim_c == 2
  #assert m.dim_d == 3
  #assert m.dim_cd == 6
  #assert m.dim_ru == 2
  #assert m.dim_rv == 2
  #assert (m.y == y).all()
  #assert (m.z == z).all()

  # Defines GMMStats
  gs = GMMStats(2,3)
  log_likelihood = -3.
  T = 1
  n = numpy.array([0.4, 0.6], 'float64')
  sumpx = numpy.array([[1., 2., 3.], [4., 5., 6.]], 'float64')
  sumpxx = numpy.array([[10., 20., 30.], [40., 50., 60.]], 'float64')
  gs.log_likelihood = log_likelihood
  gs.t = T
  gs.n = n
  gs.sum_px = sumpx
  gs.sum_pxx = sumpxx

  # Forward GMMStats and check estimated value of the x speaker factor
  eps = 1e-10
  x_ref = numpy.array([0.291042849767692, 0.310273618998444], 'float64')
  score_ref = -2.111577181208289
  score = m.log_likelihood(gs)
  assert numpy.allclose(m.x, x_ref, eps)
  assert abs(score_ref-score) < eps

  # x and Ux
  x = numpy.ndarray((2,), numpy.float64)
  m.estimate_x(gs, x)
  n_gaussians, dim,_,_ = m.shape
  x_py = estimate_x(n_gaussians, dim, ubm.mean_supervector, ubm.variance_supervector, U, n, sumpx)
  assert numpy.allclose(x, x_py, eps)

  ux = numpy.ndarray((6,), numpy.float64)
  m.estimate_ux(gs, ux)
  n_gaussians, dim,_,_ = m.shape
  ux_py = estimate_ux(n_gaussians, dim, ubm.mean_supervector, ubm.variance_supervector, U, n, sumpx)
  assert numpy.allclose(ux, ux_py, eps)
  assert numpy.allclose(m.x, x, eps)

  score = m.forward_ux(gs, ux)

  assert abs(score_ref-score) < eps

  # Clean-up
  os.unlink(filename)

def test_ISVMachine():

  # Creates a UBM
  weights = numpy.array([0.4, 0.6], 'float64')
  means = numpy.array([[1, 6, 2], [4, 3, 2]], 'float64')
  variances = numpy.array([[1, 2, 1], [2, 1, 2]], 'float64')
  ubm = GMMMachine(2,3)
  ubm.weights = weights
  ubm.means = means
  ubm.variances = variances

  # Creates a ISVBaseMachine
  U = numpy.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]], 'float64')
  #V = numpy.array([[0], [0], [0], [0], [0], [0]], 'float64')
  d = numpy.array([0, 1, 0, 1, 0, 1], 'float64')
  base = ISVBase(ubm,2)
  base.u = U
  #base.v = V
  base.d = d

  # Creates a JFAMachine
  z = numpy.array([3,4,1,2,0,1], 'float64')
  m = ISVMachine(base)
  m.z = z

  n_gaussians,dim,ru    = m.shape
  supervector_length    = m.supervector_length
  assert n_gaussians          == 2
  assert dim                  == 3
  assert supervector_length   == 6
  assert ru                   == 2
  assert (m.z == z).all()

  # Saves and loads
  filename = str(tempfile.mkstemp(".hdf5")[1])
  m.save(bob.io.base.HDF5File(filename, 'w'))
  m_loaded = ISVMachine(bob.io.base.HDF5File(filename))
  m_loaded.isv_base = base
  assert m == m_loaded
  assert (m != m_loaded) is False
  assert m.is_similar_to(m_loaded)

  # Copy constructor
  mc = ISVMachine(m)
  assert m == mc

  # Variant
  mv = ISVMachine(base)
  # Checks for correctness
  #mv.isv_base = base
  m.z = z

  n_gaussians,dim,ru    = m.shape
  supervector_length    = m.supervector_length
  assert n_gaussians        == 2
  assert dim                == 3
  assert supervector_length == 6
  assert ru                 == 2
  assert (m.z == z).all()

  # Defines GMMStats
  gs = GMMStats(2,3)
  log_likelihood = -3.
  T = 1
  n = numpy.array([0.4, 0.6], 'float64')
  sumpx = numpy.array([[1., 2., 3.], [4., 5., 6.]], 'float64')
  sumpxx = numpy.array([[10., 20., 30.], [40., 50., 60.]], 'float64')
  gs.log_likelihood = log_likelihood
  gs.t = T
  gs.n = n
  gs.sum_px = sumpx
  gs.sum_pxx = sumpxx

  # Forward GMMStats and check estimated value of the x speaker factor
  eps = 1e-10
  x_ref = numpy.array([0.291042849767692, 0.310273618998444], 'float64')
  score_ref = -3.280498193082100

  score = m(gs)
  assert numpy.allclose(m.x, x_ref, eps)
  assert abs(score_ref-score) < eps

  # Check using alternate forward() method
  supervector_length = m.supervector_length
  Ux = numpy.ndarray(shape=(supervector_length,), dtype=numpy.float64)
  m.estimate_ux(gs, Ux)
  score = m.forward_ux(gs, Ux)
  assert abs(score_ref-score) < eps

  # x and Ux
  x = numpy.ndarray((2,), numpy.float64)
  m.estimate_x(gs, x)
  n_gaussians,dim,_    = m.shape
  x_py = estimate_x(n_gaussians, dim, ubm.mean_supervector, ubm.variance_supervector, U, n, sumpx)
  assert numpy.allclose(x, x_py, eps)

  ux = numpy.ndarray((6,), numpy.float64)
  m.estimate_ux(gs, ux)
  n_gaussians,dim,_    = m.shape
  ux_py = estimate_ux(n_gaussians, dim, ubm.mean_supervector, ubm.variance_supervector, U, n, sumpx)
  assert numpy.allclose(ux, ux_py, eps)
  assert numpy.allclose(m.x, x, eps)

  score = m.forward_ux(gs, ux)
  assert abs(score_ref-score) < eps

  # Clean-up
  os.unlink(filename)
