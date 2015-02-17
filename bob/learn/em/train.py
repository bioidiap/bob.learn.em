#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Tiago de Freitas Pereira <tiago.pereira@idiap.ch>
# Fri Feb 13 13:18:10 2015 +0200
#
# Copyright (C) 2011-2015 Idiap Research Institute, Martigny, Switzerland
import numpy
import bob.learn.em

def train(trainer, machine, data, max_iterations = 50, convergence_threshold=None, initialize=True):

  #Initialization
  if initialize:
    trainer.initialize(machine, data)

  trainer.eStep(machine, data)  
  average_output          = 0
  average_output_previous = 0

  if convergence_threshold!=None and hasattr(trainer,"compute_likelihood"):
    average_output          = trainer.compute_likelihood(machine)
  
  for i in range(max_iterations):
    average_output_previous = average_output
    trainer.mStep(machine, data)
    trainer.eStep(machine, data)

    if convergence_threshold!=None and hasattr(trainer,"compute_likelihood"):    
      average_output = trainer.compute_likelihood(machine)

    #Terminates if converged (and likelihood computation is set)
    if convergence_threshold!=None and abs((average_output_previous - average_output)/average_output_previous) <= convergence_threshold:
      break

  if hasattr(trainer,"finalize"):
    trainer.finalize(machine, data)


def train_jfa(trainer, jfa_base, data, max_iterations=10, initialize=True):

  if initialize:
    trainer.initialize(jfa_base, data)
    
  #V Subspace
  for i in range(max_iterations):
    trainer.e_step1(jfa_base, data)
    trainer.m_step1(jfa_base, data)
  trainer.finalize1(jfa_base, data)

  #U subspace
  for i in range(max_iterations):
    trainer.e_step2(jfa_base, data)
    trainer.m_step2(jfa_base, data)
  trainer.finalize2(jfa_base, data)

  # d subspace
  for i in range(max_iterations):
    trainer.e_step3(jfa_base, data)
    trainer.m_step3(jfa_base, data)
  trainer.finalize3(jfa_base, data)

