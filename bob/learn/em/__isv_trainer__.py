#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Tiago de Freitas Pereira <tiago.pereira@idiap.ch>
# Mon Fev 02 21:40:10 2015 +0200
#
# Copyright (C) 2011-2015 Idiap Research Institute, Martigny, Switzerland

from ._library import _ISVTrainer
import numpy

# define the class
class ISVTrainer (_ISVTrainer):

  def __init__(self, max_iterations=10, relevance_factor=4.):
    """
    :py:class:`bob.learn.em.ISVTrainer` constructor

    Keyword Parameters:
      max_iterations
        Number of maximum iterations
    """
    _ISVTrainer.__init__(self, relevance_factor)
    self._max_iterations         = max_iterations


  def train(self, isv_base, data):
    """
    Train the :py:class:`bob.learn.em.ISVBase` using data

    Keyword Parameters:
      jfa_base
        The `:py:class:bob.learn.em.ISVBase` class
      data
        The data to be trained
    """

    #Initialization
    self.initialize(isv_base, data);
      
    for i in range(self._max_iterations):
      #eStep
      self.eStep(isv_base, data);
      #mStep
      self.mStep(isv_base);



# copy the documentation from the base class
__doc__ = _ISVTrainer.__doc__
