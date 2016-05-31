#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Tiago de Freitas Pereira <tiago.pereira@idiap.ch>
# Fri Feb 13 13:18:10 2015 +0200
#
# Copyright (C) 2011-2015 Idiap Research Institute, Martigny, Switzerland
import numpy
import bob.learn.em
import logging
logger = logging.getLogger('bob.learn.em')

def train(trainer, machine, data, max_iterations = 50, convergence_threshold=None, initialize=True, rng=None):

  """
  Trains a machine given a trainer and the proper data

  **Parameters**:
    trainer : one of :py:class:`KMeansTrainer`, :py:class:`MAP_GMMTrainer`, :py:class:`ML_GMMTrainer`, :py:class:`ISVTrainer`, :py:class:`IVectorTrainer`, :py:class:`PLDATrainer`, :py:class:`EMPCATrainer`
      A trainer mechanism
    machine : one of :py:class:`KMeansMachine`, :py:class:`GMMMachine`, :py:class:`ISVBase`, :py:class:`IVectorMachine`, :py:class:`PLDAMachine`, :py:class:`bob.learn.linear.Machine`
      A container machine
    data : array_like <float, 2D>
      The data to be trained
    max_iterations : int
      The maximum number of iterations to train a machine
    convergence_threshold : float
      The convergence threshold to train a machine. If None, the training procedure will stop with the iterations criteria
    initialize : bool
      If True, runs the initialization procedure
    rng :  :py:class:`bob.core.random.mt19937`
      The Mersenne Twister mt19937 random generator used for the initialization of subspaces/arrays before the EM loop
  """
  #Initialization
  if initialize:
    if rng is not None:
      trainer.initialize(machine, data, rng)
    else:
      trainer.initialize(machine, data)

  trainer.e_step(machine, data)
  average_output          = 0
  average_output_previous = 0

  if hasattr(trainer,"compute_likelihood"):
    average_output          = trainer.compute_likelihood(machine)

  for i in range(max_iterations):
    logger.info("Iteration = %d/%d", i, max_iterations)
    average_output_previous = average_output
    trainer.m_step(machine, data)
    trainer.e_step(machine, data)
    
    if hasattr(trainer,"compute_likelihood"):
      average_output = trainer.compute_likelihood(machine)
      
      if type(machine) is bob.learn.em.KMeansMachine:
        logger.info("average euclidean distance = %f", average_output)
      else:
        logger.info("log likelihood = %f", average_output)
      
      convergence_value = abs((average_output_previous - average_output)/average_output_previous)
      logger.info("convergence value = %f",convergence_value)
    
      #Terminates if converged (and likelihood computation is set)
      if convergence_threshold!=None and convergence_value <= convergence_threshold:
        break
  if hasattr(trainer,"finalize"):
    trainer.finalize(machine, data)


def train_jfa(trainer, jfa_base, data, max_iterations=10, initialize=True, rng=None):
  """
  Trains a :py:class:`bob.learn.em.JFABase` given a :py:class:`bob.learn.em.JFATrainer` and the proper data

  **Parameters**:
    trainer : :py:class:`bob.learn.em.JFATrainer`
      A JFA trainer mechanism
    jfa_base : :py:class:`bob.learn.em.JFABase`
      A container machine
    data : [[:py:class:`bob.learn.em.GMMStats`]]
      The data to be trained
    max_iterations : int
      The maximum number of iterations to train a machine
    initialize : bool
      If True, runs the initialization procedure
    rng :  :py:class:`bob.core.random.mt19937`
      The Mersenne Twister mt19937 random generator used for the initialization of subspaces/arrays before the EM loops
  """

  if initialize:
    if rng is not None:
      trainer.initialize(jfa_base, data, rng)
    else:
      trainer.initialize(jfa_base, data)

  #V Subspace
  logger.info("V subspace estimation...")
  for i in range(max_iterations):
    logger.info("Iteration = %d/%d", i, max_iterations)
    trainer.e_step_v(jfa_base, data)
    trainer.m_step_v(jfa_base, data)
  trainer.finalize_v(jfa_base, data)

  #U subspace
  logger.info("U subspace estimation...")  
  for i in range(max_iterations):
    logger.info("Iteration = %d/%d", i, max_iterations)
    trainer.e_step_u(jfa_base, data)
    trainer.m_step_u(jfa_base, data)
  trainer.finalize_u(jfa_base, data)

  # D subspace
  logger.info("D subspace estimation...")  
  for i in range(max_iterations):
    logger.info("Iteration = %d/%d", i, max_iterations)
    trainer.e_step_d(jfa_base, data)
    trainer.m_step_d(jfa_base, data)
  trainer.finalize_d(jfa_base, data)
