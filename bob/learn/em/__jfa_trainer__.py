#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Tiago de Freitas Pereira <tiago.pereira@idiap.ch>
# Sun Fev 01 21:10:10 2015 +0200
#
# Copyright (C) 2011-2015 Idiap Research Institute, Martigny, Switzerland

from ._library import _JFATrainer
import numpy

# define the class
class JFATrainer (_JFATrainer):

  def __init__(self, max_iterations=10):
    """
    :py:class:`bob.learn.em.JFATrainer` constructor

    Keyword Parameters:
      max_iterations
        Number of maximum iterations
    """

    _JFATrainer.__init__(self)
    self._max_iterations         = max_iterations


  def train_loop(self, jfa_base, data):
    """
    Train the :py:class:`bob.learn.em.JFABase` using data

    Keyword Parameters:
      jfa_base
        The `:py:class:bob.learn.em.JFABase` class
      data
        The data to be trained
    """
    #V Subspace
    for i in range(self._max_iterations):
      self.e_step1(jfa_base, data)
      self.m_step1(jfa_base, data)
    self.finalize1(jfa_base, data)

    #U subspace
    for i in range(self._max_iterations):
      self.e_step2(jfa_base, data)
      self.m_step2(jfa_base, data)
    self.finalize2(jfa_base, data)

    # d subspace
    for i in range(self._max_iterations):
      self.e_step3(jfa_base, data)
      self.m_step3(jfa_base, data)
    self.finalize3(jfa_base, data)


  def train(self, jfa_base, data):
    """
    Train the :py:class:`bob.learn.em.JFABase` using data

    Keyword Parameters:
      jfa_base
        The `:py:class:bob.learn.em.JFABase` class
      data
        The data to be trained
    """
    self.initialize(jfa_base, data)
    self.train_loop(jfa_base, data)


# copy the documentation from the base class
__doc__ = _JFATrainer.__doc__
