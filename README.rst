.. vim: set fileencoding=utf-8 :
.. Mon 15 Aug 2016 09:48:28 CEST

.. image:: http://img.shields.io/badge/docs-stable-yellow.png
   :target: http://pythonhosted.org/bob.learn.em/index.html
.. image:: http://img.shields.io/badge/docs-latest-orange.png
   :target: https://www.idiap.ch/software/bob/docs/latest/bob/bob.learn.em/master/index.html
.. image:: https://gitlab.idiap.ch/bob/bob.learn.em/badges/v2.0.11/build.svg
   :target: https://gitlab.idiap.ch/bob/bob.learn.em/commits/v2.0.11
.. image:: https://img.shields.io/badge/gitlab-project-0000c0.svg
   :target: https://gitlab.idiap.ch/bob/bob.learn.em
.. image:: http://img.shields.io/pypi/v/bob.learn.em.png
   :target: https://pypi.python.org/pypi/bob.learn.em
.. image:: http://img.shields.io/pypi/dm/bob.learn.em.png
   :target: https://pypi.python.org/pypi/bob.learn.em


=================================================
 Expectation Maximization Machine Learning Tools
=================================================

This package is part of the signal-processing and machine learning toolbox
Bob_. It contains routines for learning probabilistic models via Expectation
Maximization (EM).

The EM algorithm is an iterative method that estimates parameters for
statistical models, where the model depends on unobserved latent variables. The
EM iteration alternates between performing an expectation (E) step, which
creates a function for the expectation of the log-likelihood evaluated using
the current estimate for the parameters, and a maximization (M) step, which
computes parameters maximizing the expected log-likelihood found on the E step.
These parameter-estimates are then used to determine the distribution of the
latent variables in the next E step.

The package includes the machine definition per se and a selection of different trainers for specialized purposes:

 - Maximum Likelihood (ML)
 - Maximum a Posteriori (MAP)
 - K-Means
 - Inter Session Variability Modelling (ISV)
 - Joint Factor Analysis (JFA)
 - Total Variability Modeling (iVectors)
 - Probabilistic Linear Discriminant Analysis (PLDA)
 - EM Principal Component Analysis (EM-PCA)


Installation
------------

Follow our `installation`_ instructions. Then, using the Python interpreter
provided by the distribution, bootstrap and buildout this package::

  $ python bootstrap-buildout.py
  $ ./bin/buildout


Contact
-------

For questions or reporting issues to this software package, contact our
development `mailing list`_.


.. Place your references here:
.. _bob: https://www.idiap.ch/software/bob
.. _installation: https://gitlab.idiap.ch/bob/bob/wikis/Installation
.. _mailing list: https://groups.google.com/forum/?fromgroups#!forum/bob-devel
