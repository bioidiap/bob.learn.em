#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Tiago de Freitas Pereira <tiago.pereira@idiap.ch>
# Mon Jan 22 18:29:10 2015
#
# Copyright (C) 2011-2015 Idiap Research Institute, Martigny, Switzerland

from ._library import _ML_GMMTrainer
import numpy

# define the class
class ML_GMMTrainer(_ML_GMMTrainer):

  def __init__(self, update_means=True, update_variances=False, update_weights=False, convergence_threshold=0.001, max_iterations=10, converge_by_likelihood=True):
    """
    :py:class:bob.learn.em.ML_GMMTrainer constructor

    Keyword Parameters:
      update_means

      update_variances

      update_weights
 
      convergence_threshold
        Convergence threshold
      max_iterations
        Number of maximum iterations
      converge_by_likelihood
        Tells whether we compute log_likelihood as a convergence criteria, or not 
        
    """

    _ML_GMMTrainer.__init__(self, update_means=update_means, update_variances=update_variances, update_weights=update_weights)
    self.convergence_threshold  = convergence_threshold
    self.max_iterations         = max_iterations
    self.converge_by_likelihood = converge_by_likelihood


  def train(self, gmm_machine, data):
    """
    Train the :py:class:bob.learn.em.GMMMachine using data

    Keyword Parameters:
      gmm_machine
        The :py:class:bob.learn.em.GMMMachine class
      data
        The data to be trained
    """

    #Initialization
    self.initialize(gmm_machine);

    #Do the Expectation-Maximization algorithm
    average_output_previous = 0
    average_output = -numpy.inf;


    #eStep
    self.eStep(gmm_machine, data);

    if(self.converge_by_likelihood):
      average_output = self.compute_likelihood(gmm_machine);    

    for i in range(self.max_iterations):
      #saves average output from last iteration
      average_output_previous = average_output;

      #mStep
      self.mStep(gmm_machine);

      #eStep
      self.eStep(gmm_machine, data);

      #Computes log likelihood if required
      if(self.converge_by_likelihood):
        average_output = self.compute_likelihood(gmm_machine);

        #Terminates if converged (and likelihood computation is set)
        if abs((average_output_previous - average_output)/average_output_previous) <= self.convergence_threshold:
          break


# copy the documentation from the base class
__doc__ = _ML_GMMTrainer.__doc__
