#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Tiago de Freitas Pereira <tiago.pereira@idiap.ch>
# Fri Feb 13 13:18:10 2015 +0200
#
# Copyright (C) 2011-2015 Idiap Research Institute, Martigny, Switzerland
import numpy
from ._library import *
import logging
from multiprocessing.pool import ThreadPool

logger = logging.getLogger(__name__)


def _set_average(trainer, trainers, machine, data, trainer_type):
  """_set_average(trainer, data) -> None

  This function computes the average of the given data and sets it to the given machine.

  This function works for different types of trainers, and can be used to parallelize the training.
  For some trainers, the data is returned instead of set in the trainer.

  **Parameters:**

  trainer : one of :py:class:`KMeansTrainer`, :py:class:`MAP_GMMTrainer`, :py:class:`ML_GMMTrainer`, :py:class:`ISVTrainer`, :py:class:`IVectorTrainer`, :py:class:`PLDATrainer`, :py:class:`EMPCATrainer`
    The trainer to set the data to.

  trainers : [ trainer ]
    The list of trainer objects that were used in the parallel training process.
    All trainers must be of the same class as the ``trainer``.

  data : [ object ]
    The list of data objects that should be set to the trainer.
    Usually this list is generated by parallelizing the e-step of the ``trainer``.
  """

  if trainer_type == "KMeansTrainer":
    # K-Means statistics
    trainer.reset_accumulators(machine)
    for t in trainers:
      trainer.zeroeth_order_statistics = trainer.zeroeth_order_statistics + t.zeroeth_order_statistics
      trainer.first_order_statistics = trainer.first_order_statistics + t.first_order_statistics
      trainer.average_min_distance = trainer.average_min_distance + t.average_min_distance

    trainer.average_min_distance /= data.shape[0]

  elif trainer_type in ("ML_GMMTrainer", "MAP_GMMTrainer"):
    # GMM statistics
    trainer.gmm_statistics = trainers[0].gmm_statistics
    for t in trainers[1:]:
      trainer.gmm_statistics += t.gmm_statistics

  elif trainer_type == "IVectorTrainer":
    # GMM statistics
    trainer.reset_accumulators(machine)
    trainer.acc_fnormij_wij = trainers[0].acc_fnormij_wij
    trainer.acc_nij_wij2 = trainers[0].acc_nij_wij2
    trainer.acc_nij = trainers[0].acc_nij
    trainer.acc_snormij = trainers[0].acc_snormij
    
    for t in trainers[1:]:
      trainer.acc_fnormij_wij = trainer.acc_fnormij_wij + t.acc_fnormij_wij
      trainer.acc_nij_wij2    = trainer.acc_nij_wij2 + t.acc_nij_wij2
      trainer.acc_nij         = trainer.acc_nij + t.acc_nij
      trainer.acc_snormij     = trainer.acc_snormij + t.acc_snormij


  else:
    raise NotImplementedError("Implement Me!")


def _parallel_e_step(args):
  """This function applies the e_step of the given trainer (first argument) on the given data (second argument).
  It is called by each parallel process.
  """
  trainer, machine, data = args
  trainer.e_step(machine, data)


def train(trainer, machine, data, max_iterations=50, convergence_threshold=None, initialize=True, rng=None, check_inputs=True, pool=None, trainer_type=None):

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
    check_inputs:
      Shallow checks in the inputs. Check for inf and NaN
    pool : ``int`` or ``multiprocessing.ThreadPool`` or ``None``
      If given, the provided process pool will be used to parallelize the M-step of the
      EM algorithm. You should provide a ThreadPool not a multi process Pool. If pool is
      an integer, it will be used to create a ThreadPool with that many processes.
    trainer_type : ``str`` or ``None``
      This is used for the parallel e_step method to see how several processes' data can
      be merged into one trainer before the m_step. By default
      ``trainer.__class__.__name__`` is used. This is useful if you have custom trainers
      and want to use this function.
  """
  if check_inputs and isinstance(data, numpy.ndarray):
    sum_data = numpy.sum(data)

    if numpy.isinf(sum_data):
        raise ValueError("Please, check your inputs; numpy.inf detected in `data` ")

    if numpy.isnan(sum_data):
        raise ValueError("Please, check your inputs; numpy.nan detected in `data` ")

  if isinstance(pool, int):
    pool = ThreadPool(pool)

  if trainer_type is None:
    trainer_type = trainer.__class__.__name__

  def _e_step(trainer, machine, data):

    # performs the e-step, possibly in parallel
    if pool is None:
      # use only one core
      trainer.e_step(machine, data)
    else:

      # use the given process pool
      n_processes = pool._processes

      # Mapping references of the data
      split_data = []
      offset = 0
      
      step = int(len(data) // n_processes)
   
      for p in range(n_processes):
        if p == n_processes - 1:
          # take all the data in the last chunk
          split_data.append(data[offset:])
        else:
          split_data.append(data[offset: offset + step])

        offset += step

      # create trainers for each process
      trainers = [trainer.__class__(trainer) for p in range(n_processes)]
      # no need to copy the machines
      machines = [machine.__class__(machine) for p in range(n_processes)]
      # call the parallel processes
      pool.map(_parallel_e_step, zip(trainers, machines, split_data))
      # update the trainer with the data of the other trainers
      _set_average(trainer, trainers, machine, data, trainer_type)

  # Initialization
  if initialize:
    if rng is not None:
      trainer.initialize(machine, data, rng)
    else:
      trainer.initialize(machine, data)

  _e_step(trainer, machine, data)
  average_output          = 0
  average_output_previous = 0

  if hasattr(trainer,"compute_likelihood"):
    average_output          = trainer.compute_likelihood(machine)

  for i in range(max_iterations):
    logger.info("Iteration = %d/%d", i, max_iterations)
    average_output_previous = average_output
    trainer.m_step(machine, data)
    _e_step(trainer, machine,data)

    if hasattr(trainer,"compute_likelihood"):
      average_output = trainer.compute_likelihood(machine)

      if isinstance(machine, KMeansMachine):
        logger.info("average euclidean distance = %f", average_output)
      else:
        logger.info("log likelihood = %f", average_output)

      convergence_value = abs((average_output_previous - average_output)/average_output_previous)
      logger.info("convergence value = %f",convergence_value)

      #Terminates if converged (and likelihood computation is set)
      if convergence_threshold is not None and convergence_value <= convergence_threshold:
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

  # V Subspace
  logger.info("V subspace estimation...")
  for i in range(max_iterations):
    logger.info("Iteration = %d/%d", i, max_iterations)
    trainer.e_step_v(jfa_base, data)
    trainer.m_step_v(jfa_base, data)
  trainer.finalize_v(jfa_base, data)

  # U subspace
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
