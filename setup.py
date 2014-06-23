#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.anjos@idiap.ch>
# Mon 16 Apr 08:18:08 2012 CEST

from setuptools import setup, find_packages, dist
dist.Distribution(dict(setup_requires=['bob.blitz', 'bob.io.base']))
from bob.blitz.extension import Extension
import bob.io.base

import os
include_dirs = [bob.io.base.get_include()]

packages = ['bob-machine >= 2.0.0a2', 'bob-trainer >= 2.0.0a2', 'boost']
version = '2.0.0a0'

setup(

    name='bob.learn.misc',
    version=version,
    description='Bindings for miscelaneous machines and trainers',
    url='http://github.com/bioidiap/bob.learn.misc',
    license='BSD',
    author='Andre Anjos',
    author_email='andre.anjos@idiap.ch',

    long_description=open('README.rst').read(),

    packages=find_packages(),
    include_package_data=True,

    install_requires=[
      'setuptools',
      'bob.blitz',
      'bob.core',
      'bob.io.base',
      'bob.sp',
      ],

    namespace_packages=[
      "bob",
      "bob.learn",
      ],

    ext_modules = [
      Extension("bob.learn.misc.version",
        [
          "bob/learn/misc/version.cpp",
          ],
        packages = packages,
        include_dirs = include_dirs,
        version = version,
        ),
      Extension("bob.learn.misc._library",
        [
          "bob/learn/misc/bic.cpp",
          "bob/learn/misc/bic_trainer.cpp",
          "bob/learn/misc/empca_trainer.cpp",
          "bob/learn/misc/gaussian.cpp",
          "bob/learn/misc/gmm.cpp",
          "bob/learn/misc/gmm_trainer.cpp",
          "bob/learn/misc/ivector.cpp",
          "bob/learn/misc/ivector_trainer.cpp",
          "bob/learn/misc/jfa.cpp",
          "bob/learn/misc/jfa_trainer.cpp",
          "bob/learn/misc/kmeans.cpp",
          "bob/learn/misc/kmeans_trainer.cpp",
          "bob/learn/misc/machine.cpp",
          "bob/learn/misc/linearscoring.cpp",
          "bob/learn/misc/plda.cpp",
          "bob/learn/misc/plda_trainer.cpp",
          "bob/learn/misc/wiener.cpp",
          "bob/learn/misc/wiener_trainer.cpp",
          "bob/learn/misc/ztnorm.cpp",

          # external requirements as boost::python bindings
          "bob/learn/misc/blitz_numpy.cpp",
          "bob/learn/misc/ndarray.cpp",
          "bob/learn/misc/ndarray_numpy.cpp",
          "bob/learn/misc/tinyvector.cpp",
          "bob/learn/misc/hdf5.cpp",
          "bob/learn/misc/random.cpp",

          "bob/learn/misc/main.cpp",
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
