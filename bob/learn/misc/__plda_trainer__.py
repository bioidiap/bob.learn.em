#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Tiago de Freitas Pereira <tiago.pereira@idiap.ch>
# Mon Fev 02 21:40:10 2015 +0200
#
# Copyright (C) 2011-2015 Idiap Research Institute, Martigny, Switzerland

from ._library import _PLDATrainer
import numpy

# define the class
class PLDATrainer (_PLDATrainer):

  def __init__(self, max_iterations=10, use_sum_second_order=False):
    """
    :py:class:`bob.learn.misc.PLDATrainer` constructor

    Keyword Parameters:
      max_iterations
        Number of maximum iterations
    """
    _PLDATrainer.__init__(self, use_sum_second_order)
    self._max_iterations         = max_iterations


  def train(self, plda_base, data):
    """
    Train the :py:class:`bob.learn.misc.PLDABase` using data

    Keyword Parameters:
      jfa_base
        The `:py:class:bob.learn.misc.PLDABase` class
      data
        The data to be trained
    """

    #Initialization
    self.initialize(plda_base, data);
      
    for i in range(self._max_iterations):
      #eStep
      self.e_step(plda_base, data);
      #mStep
      self.m_step(plda_base, data);
    self.finalize(plda_base, data);



# copy the documentation from the base class
__doc__ = _PLDATrainer.__doc__
