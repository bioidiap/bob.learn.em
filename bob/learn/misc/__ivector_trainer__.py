#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Tiago de Freitas Pereira <tiago.pereira@idiap.ch>
# Tue Fev 03 13:20:10 2015 +0200
#
# Copyright (C) 2011-2015 Idiap Research Institute, Martigny, Switzerland

from ._library import _IVectorTrainer
import numpy

# define the class
class IVectorTrainer (_IVectorTrainer):

  def __init__(self, max_iterations=10, update_sigma=False):
    """
    :py:class:`bob.learn.misc.IVectorTrainer` constructor

    Keyword Parameters:
      max_iterations
        Number of maximum iterations
      update_sigma
        
    """
    _IVectorTrainer.__init__(self, update_sigma)
    self._max_iterations         = max_iterations


  def train(self, ivector_machine, data):
    """
    Train the :py:class:`bob.learn.misc.IVectorMachine` using data

    Keyword Parameters:
      ivector_machine
        The `:py:class:bob.learn.misc.IVectorMachine` class
      data
        The data to be trained
    """

    #Initialization
    self.initialize(ivector_machine, data);
      
    for i in range(self._max_iterations):
      #eStep
      self.eStep(ivector_machine, data);
      #mStep
      self.mStep(ivector_machine);



# copy the documentation from the base class
__doc__ = _IVectorTrainer.__doc__
