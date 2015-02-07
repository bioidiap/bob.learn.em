#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Tiago de Freitas Pereira <tiago.pereira@idiap.ch>
# Wed Fev 04 13:35:10 2015 +0200
#
# Copyright (C) 2011-2015 Idiap Research Institute, Martigny, Switzerland

from ._library import _EMPCATrainer
import numpy

# define the class
class EMPCATrainer (_EMPCATrainer):

  def __init__(self, convergence_threshold=0.001, max_iterations=10, compute_likelihood=True):
    """
    :py:class:`bob.learn.em.EMPCATrainer` constructor

    Keyword Parameters:
      convergence_threshold
        Convergence threshold
      max_iterations
        Number of maximum iterations
      compute_likelihood
        
    """

    _EMPCATrainer.__init__(self,convergence_threshold)
    self._max_iterations        = max_iterations
    self._compute_likelihood    = compute_likelihood


  def train(self, linear_machine, data):
    """
    Train the :py:class:bob.learn.em.LinearMachine using data

    Keyword Parameters:
      linear_machine
        The :py:class:bob.learn.em.LinearMachine class
      data
        The data to be trained
    """

    #Initialization
    self.initialize(linear_machine, data);
      
    #Do the Expectation-Maximization algorithm
    average_output_previous = 0
    average_output = -numpy.inf;

    #eStep
    self.eStep(linear_machine, data);

    if(self._compute_likelihood):
      average_output = self.compute_likelihood(linear_machine);
    
    for i in range(self._max_iterations):

      #saves average output from last iteration
      average_output_previous = average_output;

      #mStep
      self.mStep(linear_machine);

      #eStep
      self.eStep(linear_machine, data);

      #Computes log likelihood if required
      if(self._compute_likelihood):
        average_output = self.compute_likelihood(linear_machine);

        #Terminates if converged (and likelihood computation is set)
        if abs((average_output_previous - average_output)/average_output_previous) <= self._convergence_threshold:
          break


# copy the documentation from the base class
__doc__ = _EMPCATrainer.__doc__
