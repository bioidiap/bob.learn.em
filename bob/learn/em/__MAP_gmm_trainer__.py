#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Tiago de Freitas Pereira <tiago.pereira@idiap.ch>
# Mon Jan 23 18:31:10 2015
#
# Copyright (C) 2011-2015 Idiap Research Institute, Martigny, Switzerland

from ._library import _MAP_GMMTrainer
import numpy

# define the class
class MAP_GMMTrainer(_MAP_GMMTrainer):

  def __init__(self, prior_gmm, update_means=True, update_variances=False, update_weights=False, convergence_threshold=0.001, max_iterations=10, converge_by_likelihood=True, **kwargs):
    """
    :py:class:`bob.learn.em.MAP_GMMTrainer` constructor

    Keyword Parameters:
      update_means

      update_variances

      update_weights

      prior_gmm
        A :py:class:`bob.learn.em.GMMMachine` to be adapted
      convergence_threshold
        Convergence threshold
      max_iterations
        Number of maximum iterations
      converge_by_likelihood
        Tells whether we compute log_likelihood as a convergence criteria, or not 
      alpha
        Set directly the alpha parameter (Eq (14) from [Reynolds2000]_), ignoring zeroth order statistics as a weighting factor.
      relevance_factor
        If set the :py:class:`bob.learn.em.MAP_GMMTrainer.reynolds_adaptation` parameters, will apply the Reynolds Adaptation procedure. See Eq (14) from [Reynolds2000]_  
    """

    if kwargs.get('alpha')!=None:
      alpha = kwargs.get('alpha')
      _MAP_GMMTrainer.__init__(self, prior_gmm,alpha=alpha, update_means=update_means, update_variances=update_variances,update_weights=update_weights)
    else:
      relevance_factor = kwargs.get('relevance_factor')
      _MAP_GMMTrainer.__init__(self, prior_gmm, relevance_factor=relevance_factor, update_means=update_means, update_variances=update_variances,update_weights=update_weights)
    
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
__doc__ = _MAP_GMMTrainer.__doc__
