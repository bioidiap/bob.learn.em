#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.anjos@idiap.ch>
# Mon 16 Apr 08:18:08 2012 CEST

from setuptools import setup, find_packages, dist
dist.Distribution(dict(setup_requires=['xbob.blitz', 'xbob.io.base']))
from xbob.blitz.extension import Extension
import xbob.io.base

import os
include_dirs = [xbob.io.base.get_include()]

packages = ['bob-machine >= 2.0.0a2', 'bob-trainer >= 2.0.0a2', 'boost']
version = '2.0.0a0'

setup(

    name='xbob.learn.misc',
    version=version,
    description='Bindings for miscelaneous machines and trainers',
    url='http://github.com/bioidiap/xbob.learn.misc',
    license='BSD',
    author='Andre Anjos',
    author_email='andre.anjos@idiap.ch',

    long_description=open('README.rst').read(),

    packages=find_packages(),
    include_package_data=True,

    install_requires=[
      'setuptools',
      'xbob.blitz',
      'xbob.core',
      'xbob.io.base',
      'xbob.sp',
      ],

    namespace_packages=[
      "xbob",
      "xbob.learn",
      ],

    ext_modules = [
      Extension("xbob.learn.misc.version",
        [
          "xbob/learn/misc/version.cpp",
          ],
        packages = packages,
        include_dirs = include_dirs,
        version = version,
        ),
      Extension("xbob.learn.misc._library",
        [
          "xbob/learn/misc/bic.cpp",
          "xbob/learn/misc/bic_trainer.cpp",
          "xbob/learn/misc/empca_trainer.cpp",
          "xbob/learn/misc/gabor.cpp",
          "xbob/learn/misc/gaussian.cpp",
          "xbob/learn/misc/gmm.cpp",
          "xbob/learn/misc/gmm_trainer.cpp",
          "xbob/learn/misc/ivector.cpp",
          "xbob/learn/misc/ivector_trainer.cpp",
          "xbob/learn/misc/jfa.cpp",
          "xbob/learn/misc/jfa_trainer.cpp",
          "xbob/learn/misc/kmeans.cpp",
          "xbob/learn/misc/kmeans_trainer.cpp",
          "xbob/learn/misc/machine.cpp",
          "xbob/learn/misc/linearscoring.cpp",
          "xbob/learn/misc/plda.cpp",
          "xbob/learn/misc/plda_trainer.cpp",
          "xbob/learn/misc/wiener.cpp",
          "xbob/learn/misc/wiener_trainer.cpp",
          "xbob/learn/misc/ztnorm.cpp",

          # external requirements as boost::python bindings
          "xbob/learn/misc/GaborWaveletTransform.cpp",
          "xbob/learn/misc/blitz_numpy.cpp",
          "xbob/learn/misc/ndarray.cpp",
          "xbob/learn/misc/ndarray_numpy.cpp",
          "xbob/learn/misc/tinyvector.cpp",
          "xbob/learn/misc/hdf5.cpp",
          "xbob/learn/misc/random.cpp",

          "xbob/learn/misc/main.cpp",
        ],
        packages = packages,
        boost_modules = ['python'],
        include_dirs = include_dirs,
        version = version,
        ),
      ],

    classifiers = [
      'Development Status :: 3 - Alpha',
      'Intended Audience :: Developers',
      'License :: OSI Approved :: BSD License',
      'Natural Language :: English',
      'Programming Language :: Python',
      'Programming Language :: Python :: 3',
      'Topic :: Software Development :: Libraries :: Python Modules',
      ],

    )
