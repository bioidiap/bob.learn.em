.. vim: set fileencoding=utf-8 :
.. Tiago de Freitas Pereira <tiago.pereira@idiap.ch>
.. Tue 17 Feb 2015 13:50:06 CET
..
.. Copyright (C) 2011-2014 Idiap Research Institute, Martigny, Switzerland

.. _bob.learn.em:

================================================
 Expectation Maximization Machine Learning Tools
================================================

The EM algorithm is an iterative method that estimates parameters for statistical models, where the model depends on unobserved latent variables. The EM iteration alternates between performing an expectation (E) step, which creates a function for the expectation of the log-likelihood evaluated using the current estimate for the parameters, and a maximization (M) step, which computes parameters maximizing the expected log-likelihood found on the E step. These parameter-estimates are then used to determine the distribution of the latent variables in the next E step [WikiEM]_. 

The package includes the machine definition per se and a selection of different trainers for specialized purposes:
 - Maximum Likelihood (ML)
 - Maximum a Posteriori (MAP)
 - K-Means
 - Inter Session Variability Modelling (ISV)
 - Joint Factor Analysis (JFA)
 - Total Variability Modeling (iVectors)
 - Probabilistic Linear Discriminant Analysis (PLDA)
 - EM Principal Component Analysis (EM-PCA)


Documentation
-------------

.. toctree::
   :maxdepth: 2

   guide
   py_api
   
References
-----------

.. [Reynolds2000] *Reynolds, Douglas A., Thomas F. Quatieri, and Robert B. Dunn*. **Speaker Verification Using Adapted Gaussian Mixture Models**, Digital signal processing 10.1 (2000): 19-41.
..   [Vogt2008]   *R. Vogt, S. Sridharan*. **'Explicit Modelling of Session Variability for Speaker Verification'**, Computer Speech & Language, 2008, vol. 22, no. 1, pp. 17-38
..   [McCool2013] *C. McCool, R. Wallace, M. McLaren, L. El Shafey, S. Marcel*. **'Session Variability Modelling for Face Authentication'**, IET Biometrics, 2013
..   [Dehak2010] *N. Dehak, P. Kenny, R. Dehak, P. Dumouchel, P. Ouellet*, **'Front End Factor Analysis for Speaker Verification'**, IEEE Transactions on Audio, Speech and Language Processing, 2010, vol. 19, issue 4, pp. 788-798
..   [ElShafey2014] *Laurent El Shafey, Chris McCool, Roy Wallace, Sebastien Marcel*. **'A Scalable Formulation of Probabilistic Linear Discriminant Analysis: Applied to Face Recognition'**, TPAMI'2014
..   [PrinceElder2007] *Prince and Elder*. **'Probabilistic Linear Discriminant Analysis for Inference About Identity'**, ICCV'2007
..   [LiFu2012] *Li, Fu, Mohammed, Elder and Prince*. **'Probabilistic Models for Inference about Identity'**,  TPAMI'2012

..   [Bishop1999] Tipping, Michael E., and Christopher M. Bishop. "Probabilistic principal component analysis." Journal of the Royal Statistical Society: Series B (Statistical Methodology) 61.3 (1999): 611-622.
..   [Roweis1998] Roweis, Sam. "EM algorithms for PCA and SPCA." Advances in neural information processing systems (1998): 626-632.

..   [WikiEM] `Expectation Maximization <http://en.wikipedia.org/wiki/Expectation%E2%80%93maximization_algorithm>`_



Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. include:: links.rst
