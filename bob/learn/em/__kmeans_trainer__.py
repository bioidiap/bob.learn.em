#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Tiago de Freitas Pereira <tiago.pereira@idiap.ch>
# Mon Jan 19 11:35:10 2015 +0200
#
# Copyright (C) 2011-2015 Idiap Research Institute, Martigny, Switzerland

from ._library import _KMeansTrainer
import numpy

# define the class
class KMeansTrainer (_KMeansTrainer):

  def __init__(self, initialization_method="RANDOM", convergence_threshold=0.001, max_iterations=10, converge_by_average_min_distance=True):
    """
    :py:class:`bob.learn.em.KMeansTrainer` constructor

    Keyword Parameters:
      initialization_method
        The initialization method to generate the initial means
      convergence_threshold
        Convergence threshold
      max_iterations
        Number of maximum iterations
      converge_by_average_min_distance
        Tells whether we compute the average min (square Euclidean) distance, as a convergence criteria, or not 
        
    """

    _KMeansTrainer.__init__(self, initialization_method="RANDOM", )
    self._convergence_threshold = convergence_threshold
    self._max_iterations         = max_iterations
    self._converge_by_average_min_distance = converge_by_average_min_distance


  def train(self, kmeans_machine, data):
    """
    Train the :py:class:bob.learn.em.KMeansMachine using data

    Keyword Parameters:
      kmeans_machine
        The :py:class:bob.learn.em.KMeansMachine class
      data
        The data to be trained
    """

    #Initialization
    self.initialize(kmeans_machine, data);
      
    #Do the Expectation-Maximization algorithm
    average_output_previous = 0
    average_output = -numpy.inf;

    #eStep
    self.eStep(kmeans_machine, data);

    if(self._converge_by_average_min_distance):
      average_output = self.compute_likelihood(kmeans_machine);
    
    for i in range(self._max_iterations):

      #saves average output from last iteration
      average_output_previous = average_output;

      #mStep
      self.mStep(kmeans_machine);

      #eStep
      self.eStep(kmeans_machine, data);

      #Computes log likelihood if required
      if(self._converge_by_average_min_distance):
        average_output = self.compute_likelihood(kmeans_machine);

        #Terminates if converged (and likelihood computation is set)
        if abs((average_output_previous - average_output)/average_output_previous) <= self._convergence_threshold:
          break


# copy the documentation from the base class
__doc__ = _KMeansTrainer.__doc__
