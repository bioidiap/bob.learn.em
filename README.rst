.. vim: set fileencoding=utf-8 :
.. Andre Anjos <andre.anjos@idiap.ch>
.. Thu 22 May 2014 15:39:03 CEST

.. image:: http://img.shields.io/badge/docs-stable-yellow.png
   :target: http://pythonhosted.org/bob.learn.em/index.html
.. image:: http://img.shields.io/badge/docs-latest-orange.png
   :target: https://www.idiap.ch/software/bob/docs/latest/bioidiap/bob.learn.em/master/index.html
.. image:: https://travis-ci.org/bioidiap/bob.learn.em.svg?branch=v2.0.7
   :target: https://travis-ci.org/bioidiap/bob.learn.em?branch=v2.0.7
.. image:: https://coveralls.io/repos/bioidiap/bob.learn.em/badge.png?branch=v2.0.7
   :target: https://coveralls.io/r/bioidiap/bob.learn.em?branch=v2.0.7
.. image:: https://img.shields.io/badge/github-master-0000c0.png
   :target: https://github.com/bioidiap/bob.learn.em/tree/master
.. image:: http://img.shields.io/pypi/v/bob.learn.em.png
   :target: https://pypi.python.org/pypi/bob.learn.em
.. image:: http://img.shields.io/pypi/dm/bob.learn.em.png
   :target: https://pypi.python.org/pypi/bob.learn.em

==================================================
  Expectation Maximization Machine Learning Tools
==================================================

The EM algorithm is an iterative method that estimates parameters for statistical models, where the model depends on unobserved latent variables. The EM iteration alternates between performing an expectation (E) step, which creates a function for the expectation of the log-likelihood evaluated using the current estimate for the parameters, and a maximization (M) step, which computes parameters maximizing the expected log-likelihood found on the E step. These parameter-estimates are then used to determine the distribution of the latent variables in the next E step.

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
To install this package -- alone or together with other `Packages of Bob <https://github.com/idiap/bob/wiki/Packages>`_ -- please read the `Installation Instructions <https://github.com/idiap/bob/wiki/Installation>`_.
For Bob_ to be able to work properly, some dependent packages are required to be installed.
Please make sure that you have read the `Dependencies <https://github.com/idiap/bob/wiki/Dependencies>`_ for your operating system.

Documentation
-------------
For further documentation on this package, please read the `Stable Version <http://pythonhosted.org/bob.learn.em/index.html>`_ or the `Latest Version <https://www.idiap.ch/software/bob/docs/latest/bioidiap/bob.learn.em/master/index.html>`_ of the documentation.
For a list of tutorials on this or the other packages ob Bob_, or information on submitting issues, asking questions and starting discussions, please visit its website.

.. _bob: https://www.idiap.ch/software/bob
