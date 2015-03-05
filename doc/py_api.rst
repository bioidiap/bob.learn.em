.. vim: set fileencoding=utf-8 :
.. Andre Anjos <andre.dos.anjos@gmail.com>
.. Sat 16 Nov 20:52:58 2013

============
 Python API
============

This section includes information for using the pure Python API of
``bob.learn.em``.

Classes
-------


Trainers
........

.. autosummary::
  
  bob.learn.em.KMeansTrainer
  bob.learn.em.ML_GMMTrainer
  bob.learn.em.MAP_GMMTrainer
  bob.learn.em.ISVTrainer
  bob.learn.em.JFATrainer  
  bob.learn.em.IVectorTrainer
  bob.learn.em.PLDATrainer
  bob.learn.em.EMPCATrainer
  
Machines
........

.. autosummary::  
  
  bob.learn.em.KMeansMachine
  bob.learn.em.Gaussian
  bob.learn.em.GMMStats
  bob.learn.em.GMMMachine
  bob.learn.em.ISVBase
  bob.learn.em.ISVMachine
  bob.learn.em.JFABase
  bob.learn.em.JFAMachine
  bob.learn.em.IVectorMachine
  bob.learn.em.PLDABase
  bob.learn.em.PLDAMachine
  
Functions
---------
.. autosummary::

  bob.learn.em.linear_scoring
  bob.learn.em.tnorm
  bob.learn.em.train
  bob.learn.em.train_jfa
  bob.learn.em.znorm
  bob.learn.em.ztnorm
  bob.learn.em.ztnorm_same_value

Detailed Information
--------------------

.. automodule:: bob.learn.em

